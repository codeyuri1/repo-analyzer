from collections.abc import Iterator

from repo_agent_chat.llm import OllamaChat
from repo_agent_chat.retriever import RepositoryRetriever
from repo_agent_chat.vector_store import SearchResult


def format_retrieved_context(results: list[SearchResult]) -> str:
    """Formata resultados da busca como contexto legível e rastreável."""

    sections = [
        (
            f"Fonte: {result.chunk.path}:{result.chunk.start_line}-"
            f"{result.chunk.end_line}\n{result.chunk.content}"
        )
        for result in results
    ]
    return "\n\n---\n\n".join(sections)


class RagAssistant:
    """Orquestra recuperação semântica e geração da resposta."""

    def __init__(
        self,
        chat: OllamaChat,
        retriever: RepositoryRetriever,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k deve ser maior que zero.")

        self._chat = chat
        self._retriever = retriever
        self._top_k = top_k

    def stream(self, question: str) -> Iterator[str]:
        """Recupera evidências e transmite uma resposta fundamentada."""

        results = self._retriever.search(question, self._top_k)
        context = format_retrieved_context(results)
        yield from self._chat.stream(question, context=context)
