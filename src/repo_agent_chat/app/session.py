import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Self
from urllib.parse import urlparse

from repo_agent_chat.repository import IGNORED_DIRECTORIES

MAX_REPOSITORY_FILES = 20_000
MAX_REPOSITORY_BYTES = 200 * 1024 * 1024
GITHUB_PATH = re.compile(r"^/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$")


class RepositorySourceError(ValueError):
    """A fonte do repositório é inválida ou excede os limites da sessão."""


@dataclass(slots=True)
class RepositorySession:
    """Controla recursos temporários sem assumir posse do repositório local."""

    source: str | Path
    repository_root: Path = field(init=False)
    workspace: Path = field(init=False)
    artifact_directory: Path = field(init=False)
    _temporary_directory: TemporaryDirectory[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._temporary_directory = TemporaryDirectory(prefix="repo-agent-chat-")
        self.workspace = Path(self._temporary_directory.name)
        self.artifact_directory = self.workspace / "artifacts"
        self.artifact_directory.mkdir()
        try:
            self.repository_root = self._prepare_repository(self.source)
            self._validate_repository_limits()
        except Exception:
            self.cleanup()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove artefatos e o workspace; nunca remove o repositório local."""

        self._temporary_directory.cleanup()

    def _prepare_repository(self, source: str | Path) -> Path:
        raw_source = str(source)
        if raw_source.startswith(("http://", "https://")):
            clone_url = validate_github_url(raw_source)
            destination = self.workspace / "repository"
            environment = os.environ.copy()
            environment.update(
                {
                    "GIT_LFS_SKIP_SMUDGE": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                }
            )
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--single-branch",
                        "--filter=blob:limit=2m",
                        clone_url,
                        str(destination),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=environment,
                )
            except (FileNotFoundError, subprocess.SubprocessError) as error:
                raise RepositorySourceError(
                    "Não foi possível clonar o repositório público do GitHub."
                ) from error
            return destination.resolve()

        local_root = Path(source).expanduser().resolve()
        if not local_root.is_dir():
            raise RepositorySourceError("O caminho local deve ser um diretório existente.")
        return local_root

    def _validate_repository_limits(self) -> None:
        file_count = 0
        total_bytes = 0
        for path in self.repository_root.rglob("*"):
            relative = path.relative_to(self.repository_root)
            if (
                any(part in IGNORED_DIRECTORIES for part in relative.parts)
                or path.is_symlink()
                or not path.is_file()
            ):
                continue
            file_count += 1
            total_bytes += path.stat().st_size
            if file_count > MAX_REPOSITORY_FILES:
                raise RepositorySourceError(
                    f"O repositório excede o limite de {MAX_REPOSITORY_FILES} arquivos."
                )
            if total_bytes > MAX_REPOSITORY_BYTES:
                raise RepositorySourceError(
                    "O repositório excede o limite de 200 MB para a sessão."
                )


def validate_github_url(url: str) -> str:
    """Aceita somente URLs HTTPS públicas no formato github.com/dono/repositório."""

    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RepositorySourceError(
            "Use uma URL HTTPS pública no formato https://github.com/dono/repositorio."
        )
    match = GITHUB_PATH.fullmatch(parsed.path)
    if not match:
        raise RepositorySourceError(
            "A URL deve apontar diretamente para a raiz de um repositório do GitHub."
        )
    owner = match.group("owner")
    repository = match.group("repo")
    return f"https://github.com/{owner}/{repository}.git"
