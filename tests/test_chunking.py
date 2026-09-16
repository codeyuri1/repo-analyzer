import pytest

from repo_agent_chat.repository import SourceFile
from repo_agent_chat.retrieval.chunking import (
    CodeChunk,
    chunk_repository,
    chunk_source_file,
)


def make_source(line_count: int) -> SourceFile:
    content = "\n".join(f"linha {number}" for number in range(1, line_count + 1))
    return SourceFile(path="src/example.py", content=content)


def test_chunk_source_file_mantem_arquivo_pequeno_inteiro() -> None:
    source = make_source(3)

    chunks = chunk_source_file(source, chunk_size=5, overlap=1)

    assert chunks == [
        CodeChunk(
            path="src/example.py",
            start_line=1,
            end_line=3,
            content="linha 1\nlinha 2\nlinha 3",
        )
    ]


def test_chunk_source_file_cria_sobreposicao() -> None:
    source = make_source(7)

    chunks = chunk_source_file(source, chunk_size=4, overlap=1)

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 4), (4, 7)]
    assert chunks[0].content.splitlines()[-1] == "linha 4"
    assert chunks[1].content.splitlines()[0] == "linha 4"


def test_chunk_source_file_vazio_nao_cria_chunks() -> None:
    source = SourceFile(path="empty.py", content="")

    assert chunk_source_file(source) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_chunk_source_file_rejeita_parametros_invalidos(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_source_file(make_source(3), chunk_size=chunk_size, overlap=overlap)


def test_chunk_repository_reune_chunks_de_todos_os_arquivos() -> None:
    files = [
        SourceFile(path="a.py", content="a1\na2"),
        SourceFile(path="b.py", content="b1\nb2"),
    ]

    chunks = chunk_repository(files, chunk_size=10, overlap=0)

    assert [chunk.path for chunk in chunks] == ["a.py", "b.py"]
