from dataclasses import dataclass
from pathlib import Path

from repo_agent_chat.repository import load_repository
from repo_agent_chat.retrieval.chunking import CodeChunk, chunk_repository
from repo_agent_chat.retrieval.embeddings import OllamaEmbeddings
from repo_agent_chat.retrieval.retriever import RepositoryRetriever


class EmptyRepositoryError(Exception):
    """Nenhum conteúdo adequado para indexação foi encontrado."""


@dataclass(frozen=True, slots=True)
class RepositoryIndex:
    """Resultado da indexação acompanhado de métricas básicas."""

    retriever: RepositoryRetriever
    file_count: int
    chunk_count: int
    truncated: bool = False


def index_repository(
    root: Path,
    embeddings: OllamaEmbeddings,
    chunk_size: int = 80,
    overlap: int = 15,
    excluded_paths: frozenset[str] = frozenset(),
    max_chunks: int | None = None,
) -> RepositoryIndex:
    """Carrega, divide e indexa o conteúdo de um repositório local."""

    source_files = [
        source_file
        for source_file in load_repository(root)
        if source_file.path not in excluded_paths
    ]
    if not source_files:
        raise EmptyRepositoryError("Nenhum arquivo suportado foi encontrado.")

    chunks = chunk_repository(source_files, chunk_size, overlap)
    if not chunks:
        raise EmptyRepositoryError(
            "Nenhum conteúdo foi encontrado nos arquivos suportados."
        )

    truncated = max_chunks is not None and len(chunks) > max_chunks
    if truncated:
        chunks = sorted(chunks, key=_chunk_priority)[:max_chunks]

    retriever = RepositoryRetriever.from_chunks(chunks, embeddings)
    return RepositoryIndex(
        retriever=retriever,
        file_count=len(source_files),
        chunk_count=len(chunks),
        truncated=truncated,
    )


def _chunk_priority(chunk: CodeChunk) -> tuple[int, int, int, str]:
    """Prioriza documentação, configuração e código mais próximo da raiz."""

    path = chunk.path
    parts = path.split("/")
    name = parts[-1].casefold()
    community_penalty = 1 if "community_contributions" in parts else 0
    important_name = 0 if name.startswith(("readme", "pyproject", "dockerfile")) else 1
    return community_penalty, important_name, len(parts), path
