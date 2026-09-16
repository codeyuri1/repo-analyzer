from unittest.mock import Mock

import pytest

from repo_agent_chat.retrieval.chunking import CodeChunk
from repo_agent_chat.retrieval.embeddings import EmbeddedChunk
from repo_agent_chat.retrieval.retriever import (
    RepositoryRetriever,
    lexical_candidates,
    rerank_exact_symbols,
    rerank_hybrid,
    tokenize_code,
)
from repo_agent_chat.retrieval.vector_store import InMemoryVectorStore, SearchResult


def test_search_converte_pergunta_e_consulta_indice() -> None:
    embeddings = Mock()
    embeddings.embed_texts.return_value = [(1.0, 0.0)]
    vector_store = Mock()
    expected = [
        SearchResult(
            chunk=CodeChunk("auth.py", 1, 2, "def login(): pass"),
            score=0.95,
        )
    ]
    vector_store.search.return_value = expected
    vector_store.chunks.return_value = ()
    retriever = RepositoryRetriever(embeddings, vector_store)

    results = retriever.search("Como funciona o login?", top_k=3)

    assert results == expected
    embeddings.embed_texts.assert_called_once_with(["Como funciona o login?"])
    vector_store.search.assert_called_once_with((1.0, 0.0), 9)
    vector_store.chunks.assert_called_once_with()


def test_from_chunks_indexa_e_permite_busca() -> None:
    auth_chunk = CodeChunk("auth.py", 1, 1, "login")
    css_chunk = CodeChunk("styles.css", 1, 1, "color")
    embeddings = Mock()
    embeddings.embed_chunks.return_value = [
        EmbeddedChunk(auth_chunk, (1.0, 0.0)),
        EmbeddedChunk(css_chunk, (0.0, 1.0)),
    ]
    embeddings.embed_texts.return_value = [(1.0, 0.0)]

    retriever = RepositoryRetriever.from_chunks(
        [auth_chunk, css_chunk],
        embeddings,
    )
    results = retriever.search("autenticação", top_k=1)

    embeddings.embed_chunks.assert_called_once_with([auth_chunk, css_chunk])
    assert results[0].chunk == auth_chunk
    assert results[0].score == pytest.approx(1.0)


def test_search_rejeita_consulta_vazia() -> None:
    retriever = RepositoryRetriever(Mock(), InMemoryVectorStore([]))

    with pytest.raises(ValueError, match="não pode estar vazia"):
        retriever.search("   ")


def test_search_detecta_resposta_de_embedding_invalida() -> None:
    embeddings = Mock()
    embeddings.embed_texts.return_value = []
    retriever = RepositoryRetriever(embeddings, InMemoryVectorStore([]))

    with pytest.raises(RuntimeError, match="não foi retornado corretamente"):
        retriever.search("login")


def test_rerank_prioriza_definicao_exata_sobre_uso_do_simbolo() -> None:
    usage = SearchResult(
        CodeChunk("tests/test_rag.py", 1, 5, "assistant = RagAssistant(...)"),
        0.99,
    )
    definition = SearchResult(
        CodeChunk("rag.py", 20, 40, "class RagAssistant:\n    pass"),
        0.70,
    )

    results = rerank_exact_symbols(
        "Como funciona RagAssistant?",
        [usage, definition],
    )

    assert results == [definition, usage]


def test_rerank_preserva_ordem_sem_identificador_camel_case() -> None:
    first = SearchResult(CodeChunk("a.py", 1, 1, "primeiro"), 0.9)
    second = SearchResult(CodeChunk("b.py", 1, 1, "segundo"), 0.8)

    assert rerank_exact_symbols("como funciona a busca?", [first, second]) == [
        first,
        second,
    ]


def test_rerank_camel_case_remove_resultado_sem_o_simbolo() -> None:
    definition = SearchResult(
        CodeChunk("rag.py", 1, 2, "class RagAssistant: pass"), 0.8
    )
    unrelated = SearchResult(CodeChunk("agent.py", 1, 2, "class ToolAgent: pass"), 0.99)

    results = rerank_hybrid("Como funciona RagAssistant?", [unrelated, definition])

    assert results == [definition]


def test_rerank_hibrido_prioriza_arquivo_fonte_relacionado() -> None:
    test_result = SearchResult(
        CodeChunk("tests/test_embeddings.py", 1, 2, "embeddings chunks"), 0.99
    )
    source_result = SearchResult(
        CodeChunk("src/app/embeddings.py", 1, 2, "def embed_chunks(): pass"),
        0.70,
    )

    results = rerank_hybrid(
        "fluxo embeddings chunks",
        [test_result, source_result],
    )

    assert results[0] == source_result


def test_lexical_candidates_recupera_chunk_ausente_da_busca_vetorial() -> None:
    unrelated = CodeChunk("security.py", 1, 2, "analyze vulnerabilities")
    relevant = CodeChunk(
        "repository.py",
        10,
        20,
        "def read_source_file(): leitura segura dentro da raiz",
    )

    results = lexical_candidates(
        "onde ocorre a leitura segura de arquivos",
        (unrelated, relevant),
        limit=2,
    )

    assert results[0].chunk == relevant


def test_tokenize_code_separa_snake_case_e_camel_case() -> None:
    tokens = tokenize_code("read_source_file resolvedRoot")

    assert {"read", "source", "file", "resolved", "root"} <= tokens


def test_rerank_hibrido_entende_identificadores_de_codigo() -> None:
    security = SearchResult(
        CodeChunk("security.py", 1, 2, "def analyze_source_files(): pass"), 0.95
    )
    repository = SearchResult(
        CodeChunk(
            "repository.py",
            1,
            4,
            "def read_source_file():\n    resolved_file.read_text(encoding='utf-8')",
        ),
        0.70,
    )

    results = rerank_hybrid(
        "onde ocorre a leitura segura de arquivos",
        [security, repository],
    )

    assert results[0] == repository


def test_rerank_prioriza_definicao_sobre_texto_que_repete_a_pergunta() -> None:
    prompt = SearchResult(
        CodeChunk(
            "agent.py",
            1,
            3,
            'prompt = "explique onde ocorre a leitura segura de arquivos"',
        ),
        0.99,
    )
    implementation = SearchResult(
        CodeChunk("repository.py", 1, 3, "def read_source_file(): pass"),
        0.60,
    )

    results = rerank_hybrid(
        "Liste os módulos do projeto e explique onde ocorre a leitura segura de arquivos",
        [prompt, implementation],
    )

    assert results[0] == implementation
