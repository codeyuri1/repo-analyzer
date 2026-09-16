from dataclasses import dataclass
from pathlib import Path

from repo_agent_chat.chunking import chunk_repository
from repo_agent_chat.embeddings import OllamaEmbeddings
from repo_agent_chat.repository import load_repository
from repo_agent_chat.retriever import RepositoryRetriever


class EmptyRepositoryError(Exception):
    """Nenhum conteúdo adequado para indexação foi encontrado."""


@dataclass(frozen=True, slots=True)
class RepositoryIndex:
    """Resultado da indexação acompanhado de métricas básicas."""

    retriever: RepositoryRetriever
    file_count: int
    chunk_count: int


def index_repository(
    root: Path,
    embeddings: OllamaEmbeddings,
    chunk_size: int = 80,
    overlap: int = 15,
    excluded_paths: frozenset[str] = frozenset(),
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

    retriever = RepositoryRetriever.from_chunks(chunks, embeddings)
    return RepositoryIndex(
        retriever=retriever,
        file_count=len(source_files),
        chunk_count=len(chunks),
    )
