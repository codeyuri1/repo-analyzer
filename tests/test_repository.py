from pathlib import Path

import pytest

from repo_agent_chat.repository import (
    MAX_FILE_SIZE_BYTES,
    FileTooLargeError,
    SourceFile,
    UnsafePathError,
    discover_source_files,
    load_repository,
    read_source_file,
)


def test_read_source_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("print('Olá')", encoding="utf-8")

    source = read_source_file(tmp_path, file_path)

    assert source.path == "example.py"
    assert source.content == "print('Olá')"


def test_discover_source_files_filtra_arquivos(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text("print('ok')", encoding="utf-8")
    (src / "styles.css").write_text("body {}", encoding="utf-8")
    (src / "image.png").write_bytes(b"imagem")
    (tmp_path / "README").write_text("Projeto de exemplo", encoding="utf-8")

    git_directory = tmp_path / ".git"
    git_directory.mkdir()
    (git_directory / "config.py").write_text("ignorar", encoding="utf-8")

    pytest_cache = tmp_path / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "README.md").write_text("ignorar", encoding="utf-8")

    artifact_directory = tmp_path / ".repo-agent-chat" / "artifacts"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "diagram.md").write_text("flowchart LR", encoding="utf-8")

    files = discover_source_files(tmp_path)

    relative_paths = [path.relative_to(tmp_path).as_posix() for path in files]

    assert relative_paths == [
        "README",
        "src/main.py",
        "src/styles.css",
    ]


def test_discover_source_files_rejeita_raiz_invalida(
    tmp_path: Path,
) -> None:
    nonexistent = tmp_path / "nao-existe"

    with pytest.raises(ValueError, match="diretório"):
        discover_source_files(nonexistent)


def test_read_source_file_rejeita_arquivo_fora_da_raiz(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("segredo = True", encoding="utf-8")

    with pytest.raises(UnsafePathError, match="fora da raiz"):
        read_source_file(root, outside_file)


def test_read_source_file_rejeita_arquivo_grande(tmp_path: Path) -> None:
    file_path = tmp_path / "large.py"
    file_path.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

    with pytest.raises(FileTooLargeError):
        read_source_file(tmp_path, file_path)


def test_load_repository_ignora_arquivos_invalidos(tmp_path: Path) -> None:
    (tmp_path / "valid.py").write_text("print('válido')", encoding="utf-8")
    (tmp_path / "invalid.py").write_bytes(b"\xff\xfe")
    (tmp_path / "large.py").write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

    files = load_repository(tmp_path)

    assert files == [SourceFile(path="valid.py", content="print('válido')")]
