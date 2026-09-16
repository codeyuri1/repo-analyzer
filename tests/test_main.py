from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from repo_agent_chat.app.main import main, run_chat


def test_run_chat_responde_em_streaming(capsys) -> None:
    chat = Mock()
    chat.stream.return_value = iter(
        [
            "RAG combina ",
            "busca e geração.",
        ]
    )

    with patch(
        "builtins.input",
        side_effect=["O que é RAG?", "sair"],
    ):
        run_chat(chat)

    output = capsys.readouterr().out

    chat.stream.assert_called_once_with("O que é RAG?")
    assert "Assistente: RAG combina busca e geração." in output
    assert "Chat encerrado." in output


def test_run_chat_orienta_quando_entrada_estiver_vazia(capsys) -> None:
    chat = Mock()

    with patch("builtins.input", side_effect=["   ", "sair"]):
        run_chat(chat)

    output = capsys.readouterr().out

    assert "Digite uma pergunta ou 'sair' para encerrar." in output
    chat.stream.assert_not_called()


def test_main_indexa_repositorio_e_inicia_chat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assistant = Mock()
    repository_index = SimpleNamespace(file_count=2, chunk_count=4)
    monkeypatch.setattr("sys.argv", ["repo-agent-chat", str(tmp_path)])

    with (
        patch(
            "repo_agent_chat.app.main.build_tool_agent",
            return_value=(assistant, repository_index),
        ) as build,
        patch("repo_agent_chat.app.main.run_chat") as chat_runner,
    ):
        exit_code = main()

    assert exit_code == 0
    build.assert_called_once()
    assert build.call_args.args[0] == tmp_path.resolve()
    chat_runner.assert_called_once_with(assistant)


def test_main_retorna_erro_quando_indexacao_falha(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr("sys.argv", ["repo-agent-chat", str(tmp_path)])

    with patch(
        "repo_agent_chat.app.main.build_tool_agent",
        side_effect=ValueError("raiz inválida"),
    ):
        exit_code = main()

    assert exit_code == 1
    assert "raiz inválida" in capsys.readouterr().out
