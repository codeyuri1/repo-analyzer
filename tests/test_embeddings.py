from types import SimpleNamespace
from unittest.mock import Mock

from repo_agent_chat.chunking import CodeChunk
from repo_agent_chat.config import Settings
from repo_agent_chat.embeddings import OllamaEmbeddings


def test_embed_texts_preserva_ordem_dos_vetores() -> None:
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
    )
    embeddings = OllamaEmbeddings(Settings(_env_file=None), client=client)

    vectors = embeddings.embed_texts(["primeiro", "segundo"])

    assert vectors == [(1.0, 0.0), (0.0, 1.0)]
    client.embeddings.create.assert_called_once_with(
        model="qwen3-embedding:0.6b",
        input=["primeiro", "segundo"],
    )


def test_embed_chunks_processa_em_lotes_e_inclui_metadados() -> None:
    client = Mock()
    client.embeddings.create.side_effect = [
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])]),
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[2.0])]),
    ]
    settings = Settings(_env_file=None, embedding_batch_size=1)
    embeddings = OllamaEmbeddings(settings, client=client)
    chunks = [
        CodeChunk("a.py", 1, 1, "def a(): pass"),
        CodeChunk("b.py", 10, 10, "def b(): pass"),
    ]

    result = embeddings.embed_chunks(chunks)

    assert [item.chunk for item in result] == chunks
    assert [item.vector for item in result] == [(1.0,), (2.0,)]
    assert client.embeddings.create.call_count == 2
    assert client.embeddings.create.call_args_list[0].kwargs["input"] == [
        "Arquivo: a.py\nLinhas: 1-1\ndef a(): pass"
    ]


def test_embed_chunks_vazio_nao_chama_api() -> None:
    client = Mock()
    embeddings = OllamaEmbeddings(Settings(_env_file=None), client=client)

    assert embeddings.embed_chunks([]) == []
    client.embeddings.create.assert_not_called()
