from dataclasses import dataclass

from repo_agent_chat.repository import SourceFile


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """Trecho de código acompanhado de sua localização no arquivo original."""

    path: str
    start_line: int
    end_line: int
    content: str


def chunk_source_file(
    source_file: SourceFile,
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[CodeChunk]:
    """Divide um arquivo em chunks de linhas parcialmente sobrepostos."""

    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")

    if overlap < 0:
        raise ValueError("overlap não pode ser negativo.")

    if overlap >= chunk_size:
        raise ValueError("overlap deve ser menor que chunk_size.")

    lines = source_file.content.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    step = chunk_size - overlap
    start = 0

    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunks.append(
            CodeChunk(
                path=source_file.path,
                start_line=start + 1,
                end_line=end,
                content="\n".join(lines[start:end]),
            )
        )

        if end == len(lines):
            break

        start += step

    return chunks


def chunk_repository(
    source_files: list[SourceFile],
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[CodeChunk]:
    """Divide todos os arquivos carregados em uma única coleção de chunks."""

    return [
        chunk
        for source_file in source_files
        for chunk in chunk_source_file(source_file, chunk_size, overlap)
    ]
