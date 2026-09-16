from dataclasses import dataclass
from math import sqrt

from repo_agent_chat.chunking import CodeChunk
from repo_agent_chat.embeddings import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Chunk recuperado acompanhado de sua similaridade com a consulta."""

    chunk: CodeChunk
    score: float


def cosine_similarity(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    """Calcula a similaridade de cosseno entre vetores de mesma dimensão."""

    if len(first) != len(second):
        raise ValueError("Os vetores devem possuir a mesma dimensão.")

    if not first:
        raise ValueError("Os vetores não podem estar vazios.")

    dot_product = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = sqrt(sum(value * value for value in first))
    second_norm = sqrt(sum(value * value for value in second))

    if first_norm == 0 or second_norm == 0:
        raise ValueError("Não é possível comparar um vetor nulo.")

    return dot_product / (first_norm * second_norm)


class InMemoryVectorStore:
    """Índice vetorial simples mantido somente na memória do processo."""

    def __init__(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        self._embedded_chunks = list(embedded_chunks)
        self._dimension = self._validate_dimensions()

    def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retorna os chunks mais similares ao vetor da consulta."""

        if top_k <= 0:
            raise ValueError("top_k deve ser maior que zero.")

        if self._dimension is None:
            return []

        if len(query_vector) != self._dimension:
            raise ValueError("O vetor da consulta possui dimensão incompatível.")

        results = [
            SearchResult(
                chunk=item.chunk,
                score=cosine_similarity(query_vector, item.vector),
            )
            for item in self._embedded_chunks
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def chunks(self) -> tuple[CodeChunk, ...]:
        """Expõe somente os chunks para geração de candidatos lexicais."""

        return tuple(item.chunk for item in self._embedded_chunks)

    def _validate_dimensions(self) -> int | None:
        """Confirma que todos os vetores do índice são compatíveis."""

        if not self._embedded_chunks:
            return None

        dimension = len(self._embedded_chunks[0].vector)
        if dimension == 0:
            raise ValueError("Os embeddings não podem estar vazios.")

        if any(len(item.vector) != dimension for item in self._embedded_chunks):
            raise ValueError("Todos os embeddings devem possuir a mesma dimensão.")

        return dimension
