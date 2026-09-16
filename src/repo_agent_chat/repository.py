from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".html",
        ".css",
        ".md",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
    }
)
SUPPORTED_FILENAMES = frozenset(
    {
        "dockerfile",
        "license",
        "makefile",
        "readme",
    }
)

IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".repo-agent-chat",
        ".venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
    }
)
MAX_FILE_SIZE_BYTES = 500_000


class RepositoryFileError(Exception):
    """Erro causado por um arquivo inadequado para indexação."""


class UnsafePathError(RepositoryFileError):
    """O arquivo está fora da raiz permitida."""


class FileTooLargeError(RepositoryFileError):
    """O arquivo excede o tamanho permitido."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Representa um arquivo de código-fonte em um repositório."""

    path: str
    content: str


def read_source_file(root: Path, file_path: Path) -> SourceFile:
    """Lê um arquivo validado e retorna caminho relativo e conteúdo."""

    resolved_root = root.resolve()
    resolved_file = file_path.resolve()

    if not resolved_file.is_relative_to(resolved_root):
        raise UnsafePathError("O arquivo está fora da raiz do repositório.")

    if not resolved_file.is_file():
        raise RepositoryFileError("O caminho não aponta para um arquivo.")

    if resolved_file.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError("O arquivo excede o tamanho máximo permitido.")

    relative_path = resolved_file.relative_to(resolved_root).as_posix()
    content = resolved_file.read_text(encoding="utf-8")
    return SourceFile(path=relative_path, content=content)


def discover_source_files(root: Path) -> list[Path]:
    """Descobre arquivos de código-fonte suportados dentro do repositório."""

    if not root.is_dir():
        raise ValueError("A raiz do repositório deve ser um diretório.")

    source_files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(root)

        if any(part in IGNORED_DIRECTORIES for part in relative_path.parts):
            continue

        if (
            path.suffix.lower() not in SUPPORTED_EXTENSIONS
            and path.name.casefold() not in SUPPORTED_FILENAMES
        ):
            continue

        source_files.append(path)

    return sorted(source_files)


def load_repository(root: Path) -> list[SourceFile]:
    """Carrega arquivos válidos e ignora entradas inadequadas para indexação."""

    source_files: list[SourceFile] = []

    for path in discover_source_files(root):
        try:
            source_files.append(read_source_file(root, path))
        except (RepositoryFileError, UnicodeDecodeError, OSError):
            continue

    return source_files
