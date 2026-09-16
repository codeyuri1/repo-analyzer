import re

from repo_agent_chat.chunking import CodeChunk
from repo_agent_chat.embeddings import OllamaEmbeddings
from repo_agent_chat.vector_store import InMemoryVectorStore, SearchResult

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ_][\w]*")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
STOP_WORDS = {
    "como",
    "desde",
    "para",
    "pelos",
    "mais",
    "uma",
    "dos",
    "das",
    "liste",
    "módulos",
    "modulos",
    "projeto",
    "explique",
    "onde",
    "ocorre",
}
QUERY_ALIASES = {
    "arquivo": {"file", "source"},
    "arquivos": {"file", "files", "source"},
    "leitura": {"read"},
    "ler": {"read"},
    "repositório": {"repository", "repo"},
    "repositorio": {"repository", "repo"},
    "segura": {"safe"},
    "seguro": {"safe"},
}


class RepositoryRetriever:
    """Recupera chunks semanticamente relacionados a uma pergunta."""

    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    @classmethod
    def from_chunks(
        cls,
        chunks: list[CodeChunk],
        embeddings: OllamaEmbeddings,
    ) -> "RepositoryRetriever":
        """Indexa chunks e constrói um retriever pronto para consultas."""

        embedded_chunks = embeddings.embed_chunks(chunks)
        vector_store = InMemoryVectorStore(embedded_chunks)
        return cls(embeddings, vector_store)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Transforma uma consulta em vetor e recupera os melhores chunks."""

        if not query.strip():
            raise ValueError("A consulta não pode estar vazia.")

        query_vectors = self._embeddings.embed_texts([query])
        if len(query_vectors) != 1:
            raise RuntimeError(
                "O embedding da consulta não foi retornado corretamente."
            )

        candidate_count = min(max(top_k * 3, top_k), 30)
        semantic_results = self._vector_store.search(query_vectors[0], candidate_count)
        lexical_results = lexical_candidates(
            query,
            self._vector_store.chunks(),
            candidate_count,
        )
        merged: dict[tuple[str, int], SearchResult] = {}
        for result in (*semantic_results, *lexical_results):
            key = (result.chunk.path, result.chunk.start_line)
            current = merged.get(key)
            if current is None or result.score > current.score:
                merged[key] = result
        return rerank_hybrid(query, list(merged.values()))[:top_k]


def lexical_candidates(
    query: str,
    chunks: tuple[CodeChunk, ...],
    limit: int,
) -> list[SearchResult]:
    """Recupera candidatos por termos exatos, complementando embeddings."""

    query_tokens = expand_query_tokens(query)

    def lexical_score(chunk: CodeChunk) -> int:
        path = chunk.path.casefold()
        content_tokens = tokenize_code(chunk.content)
        return 3 * sum(token in path for token in query_tokens) + len(
            query_tokens & content_tokens
        )

    ranked = sorted(chunks, key=lexical_score, reverse=True)
    return [
        SearchResult(chunk=chunk, score=0.0)
        for chunk in ranked[:limit]
        if lexical_score(chunk) > 0
    ]


def expand_query_tokens(query: str) -> set[str]:
    """Acrescenta equivalentes técnicos comuns sem alterar a pergunta da LLM."""

    base_tokens = {
        token.casefold()
        for token in WORD_PATTERN.findall(query)
        if len(token) >= 3 and token.casefold() not in STOP_WORDS
    }
    expanded = set(base_tokens)
    for token in base_tokens:
        expanded.update(QUERY_ALIASES.get(token, set()))
    return expanded


def tokenize_code(text: str) -> set[str]:
    """Tokeniza prosa e identificadores snake_case/camelCase para busca lexical."""

    tokens: set[str] = set()
    for raw_token in WORD_PATTERN.findall(text):
        tokens.add(raw_token.casefold())
        for snake_part in raw_token.split("_"):
            for camel_part in CAMEL_BOUNDARY.split(snake_part):
                if len(camel_part) >= 2:
                    tokens.add(camel_part.casefold())
    return tokens


def rerank_exact_symbols(
    query: str,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Prioriza definições exatas sem alterar o score vetorial original."""

    symbols = re.findall(
        r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b",
        query,
    )
    if not symbols:
        return results

    matching_results = [
        result
        for result in results
        if any(
            re.search(rf"\b{re.escape(symbol)}\b", result.chunk.content)
            for symbol in symbols
        )
    ]
    candidates = matching_results or results

    def lexical_priority(result: SearchResult) -> tuple[int, int, float]:
        content = result.chunk.content
        definition = any(
            re.search(rf"\b(?:class|def)\s+{re.escape(symbol)}\b", content)
            for symbol in symbols
        )
        mentions = sum(
            len(re.findall(rf"\b{re.escape(symbol)}\b", content)) for symbol in symbols
        )
        return int(definition), mentions, result.score

    return sorted(candidates, key=lexical_priority, reverse=True)


def rerank_hybrid(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Combina prioridade lexical, código de produção e similaridade vetorial."""

    exact_ranked = rerank_exact_symbols(query, results)
    if len(exact_ranked) != len(results):
        return exact_ranked

    query_tokens = expand_query_tokens(query)

    def priority(result: SearchResult) -> tuple[int, int, int, int, float]:
        path = result.chunk.path.casefold()
        content_tokens = tokenize_code(result.chunk.content)
        path_overlap = sum(token in path for token in query_tokens)
        content_overlap = len(query_tokens & content_tokens)
        definition_overlap = max(
            (
                len(query_tokens & tokenize_code(identifier))
                for identifier in re.findall(
                    r"\b(?:class|def)\s+([A-Za-z_]\w*)",
                    result.chunk.content,
                )
            ),
            default=0,
        )
        production_code = int(not path.startswith("tests/") and "/test_" not in path)
        return (
            production_code,
            definition_overlap,
            path_overlap,
            content_overlap,
            result.score,
        )

    return sorted(exact_ranked, key=priority, reverse=True)
