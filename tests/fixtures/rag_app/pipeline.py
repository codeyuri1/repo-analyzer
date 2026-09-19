def load_documents() -> list[str]:
    return ["known document"]


def chunk_documents(documents: list[str]) -> list[str]:
    return documents


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return [[0.1] for _ in chunks]


def retrieve(query: str) -> list[str]:
    return [query]


def rerank(candidates: list[str]) -> list[str]:
    return candidates


def build_context(results: list[str]) -> str:
    return "\n".join(results)
