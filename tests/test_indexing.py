from pathlib import Path
from unittest.mock import Mock

import pytest

from repo_agent_chat.embeddings import EmbeddedChunk
from repo_agent_chat.indexing import EmptyRepositoryError, index_repository


def test_index_repository_orquestra_pipeline(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("linha 1\nlinha 2\nlinha 3", encoding="utf-8")
    (tmp_path / "README.md").write_text("documentação", encoding="utf-8")
    embeddings = Mock()
    embeddings.embed_chunks.side_effect = lambda chunks: [
        EmbeddedChunk(chunk=chunk, vector=(float(index + 1), 1.0))
        for index, chunk in enumerate(chunks)
    ]

    index = index_repository(
        tmp_path,
        embeddings,
        chunk_size=2,
        overlap=1,
    )

    assert index.file_count == 2
    assert index.chunk_count == 3
    assert embeddings.embed_chunks.call_count == 1


def test_index_repository_rejeita_repositorio_sem_arquivos_suportados(
    tmp_path: Path,
) -> None:
    (tmp_path / "image.png").write_bytes(b"PNG")

    with pytest.raises(EmptyRepositoryError, match="Nenhum arquivo suportado"):
        index_repository(tmp_path, Mock())


def test_index_repository_rejeita_arquivos_sem_conteudo(tmp_path: Path) -> None:
    (tmp_path / "empty.py").write_text("", encoding="utf-8")

    with pytest.raises(EmptyRepositoryError, match="Nenhum conteúdo"):
        index_repository(tmp_path, Mock())


def test_index_repository_exclui_caminhos_do_corpus(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('app')", encoding="utf-8")
    (tmp_path / "evals.py").write_text("pergunta secreta", encoding="utf-8")
    embeddings = Mock()
    embeddings.embed_chunks.side_effect = lambda chunks: [
        EmbeddedChunk(chunk=chunk, vector=(1.0,)) for chunk in chunks
    ]

    index = index_repository(
        tmp_path,
        embeddings,
        excluded_paths=frozenset({"evals.py"}),
    )

    indexed_chunks = embeddings.embed_chunks.call_args.args[0]
    assert index.file_count == 1
    assert {chunk.path for chunk in indexed_chunks} == {"app.py"}
