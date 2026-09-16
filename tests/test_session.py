from pathlib import Path
from unittest.mock import patch

import pytest

from repo_agent_chat.app.session import (
    RepositorySession,
    RepositorySourceError,
    validate_github_url,
)


def test_sessao_remove_workspace_sem_apagar_repositorio(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "main.py"
    source.write_text("print('ok')", encoding="utf-8")

    with RepositorySession(repository) as session:
        workspace = session.workspace
        artifact = session.artifact_directory / "diagram.mmd"
        artifact.write_text("flowchart LR", encoding="utf-8")
        assert workspace.exists()
        assert artifact.exists()

    assert not workspace.exists()
    assert source.exists()


def test_sessao_nao_contabiliza_diretorios_ignorados(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    virtual_environment = repository / ".venv"
    virtual_environment.mkdir()
    (virtual_environment / "large.bin").write_bytes(b"x" * 1_000_000)
    monkeypatch.setattr("repo_agent_chat.app.session.MAX_REPOSITORY_BYTES", 10)

    with RepositorySession(repository) as session:
        assert session.repository_root == repository.resolve()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/openai/openai-python", "https://github.com/openai/openai-python.git"),
        ("https://github.com/openai/openai-python.git", "https://github.com/openai/openai-python.git"),
    ],
)
def test_valida_url_publica_do_github(url: str, expected: str) -> None:
    assert validate_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/openai/openai-python",
        "https://gitlab.com/openai/openai-python",
        "https://token@github.com/openai/openai-python",
        "https://github.com/openai/openai-python/tree/main",
        "https://github.com/openai/openai-python?tab=readme",
    ],
)
def test_rejeita_url_git_insegura_ou_fora_do_escopo(url: str) -> None:
    with pytest.raises(RepositorySourceError):
        validate_github_url(url)


def test_clona_github_no_workspace_e_remove_ao_sair(tmp_path: Path) -> None:
    def fake_clone(command, **kwargs) -> None:
        destination = Path(command[-1])
        destination.mkdir()
        (destination / "main.py").write_text("print('ok')", encoding="utf-8")

    with (
        patch("repo_agent_chat.app.session.subprocess.run", side_effect=fake_clone) as run,
        RepositorySession("https://github.com/example/demo") as session,
    ):
        workspace = session.workspace
        assert session.repository_root == workspace / "repository"
        assert (session.repository_root / "main.py").exists()

    assert not workspace.exists()
    command = run.call_args.args[0]
    assert command[:6] == ["git", "clone", "--depth", "1", "--single-branch", "--filter=blob:limit=2m"]
    assert command[6] == "https://github.com/example/demo.git"
