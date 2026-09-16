import pytest

from repo_agent_chat.chunking import CodeChunk
from repo_agent_chat.embeddings import EmbeddedChunk
from repo_agent_chat.vector_store import InMemoryVectorStore, cosine_similarity


def embedded(path: str, vector: tuple[float, ...]) -> EmbeddedChunk:
    chunk = CodeChunk(path=path, start_line=1, end_line=1, content=path)
    return EmbeddedChunk(chunk=chunk, vector=vector)


def test_cosine_similarity_identifica_direcoes() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_search_ordena_por_similaridade_e_limita_resultados() -> None:
    store = InMemoryVectorStore(
        [
            embedded("login.py", (1.0, 0.0)),
            embedded("users.py", (0.8, 0.2)),
            embedded("styles.css", (0.0, 1.0)),
        ]
    )

    results = store.search((1.0, 0.0), top_k=2)

    assert [result.chunk.path for result in results] == ["login.py", "users.py"]
    assert results[0].score == pytest.approx(1.0)


def test_search_em_indice_vazio_retorna_lista_vazia() -> None:
    store = InMemoryVectorStore([])

    assert store.search((1.0,)) == []


def test_chunks_expoe_colecao_somente_para_leitura() -> None:
    store = InMemoryVectorStore([embedded("main.py", (1.0,))])

    chunks = store.chunks()

    assert isinstance(chunks, tuple)
    assert chunks[0].path == "main.py"


def test_search_rejeita_dimensao_incompativel() -> None:
    store = InMemoryVectorStore([embedded("main.py", (1.0, 0.0))])

    with pytest.raises(ValueError, match="dimensão incompatível"):
        store.search((1.0, 0.0, 0.0))


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejeita_top_k_invalido(top_k: int) -> None:
    store = InMemoryVectorStore([embedded("main.py", (1.0,))])

    with pytest.raises(ValueError, match="top_k"):
        store.search((1.0,), top_k=top_k)
