from dataclasses import dataclass

from openai import OpenAI

from repo_agent_chat.chunking import CodeChunk
from repo_agent_chat.config import Settings


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Chunk de código associado ao vetor que representa seu significado."""

    chunk: CodeChunk
    vector: tuple[float, ...]


class OllamaEmbeddings:
    """Gera embeddings em lotes usando a API compatível do Ollama."""

    def __init__(
        self,
        settings: Settings,
        client: OpenAI | None = None,
    ) -> None:
        self._model = settings.ollama_embedding_model
        self._batch_size = settings.embedding_batch_size
        self._client = client or OpenAI(
            base_url=str(settings.ollama_base_url),
            api_key="ollama",
            timeout=120.0,
        )

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Converte textos em vetores, preservando a ordem de entrada."""

        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        ordered_data = sorted(response.data, key=lambda item: item.index)

        if len(ordered_data) != len(texts):
            raise RuntimeError("A quantidade de embeddings retornada é inválida.")

        return [tuple(item.embedding) for item in ordered_data]

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
        """Gera embeddings dos chunks em lotes de tamanho configurável."""

        embedded_chunks: list[EmbeddedChunk] = []

        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            texts = [self._text_for_embedding(chunk) for chunk in batch]
            vectors = self.embed_texts(texts)
            embedded_chunks.extend(
                EmbeddedChunk(chunk=chunk, vector=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )

        return embedded_chunks

    @staticmethod
    def _text_for_embedding(chunk: CodeChunk) -> str:
        """Inclui metadados úteis na representação semântica do chunk."""

        return (
            f"Arquivo: {chunk.path}\n"
            f"Linhas: {chunk.start_line}-{chunk.end_line}\n"
            f"{chunk.content}"
        )
