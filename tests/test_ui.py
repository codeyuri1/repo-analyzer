from types import SimpleNamespace
from unittest.mock import Mock, call

import gradio as gr
import pytest

from repo_agent_chat.app.ui import (
    QUESTION_CATALOG,
    STANDARD_QUESTION_LABELS,
    STANDARD_QUESTIONS,
    GradioChatAdapter,
    WebRepositoryController,
    _format_tool_event,
    create_app,
    create_repository_app,
    resolve_question,
    save_mermaid_artifact,
)
from repo_agent_chat.tracing import ToolTraceEvent


def test_gradio_adapter_carrega_historico_antes_da_pergunta() -> None:
    agent = Mock()

    def ask(message: str, on_tool_event, on_token):
        on_tool_event(
            ToolTraceEvent(
                phase="completed",
                tool_name="semantic_search",
                arguments="{}",
                success=True,
                duration_ms=12.0,
            )
        )
        on_token("Resposta ")
        on_token("fundamentada.")
        return "Resposta fundamentada."

    agent.ask.side_effect = ask
    adapter = GradioChatAdapter(agent)
    history = [
        {"role": "user", "content": "Pergunta anterior"},
        {"role": "assistant", "content": "Resposta anterior"},
    ]

    updates = list(adapter("Nova pergunta", history))

    assert "Buscando trechos relevantes" in updates[0]
    assert "Gerando resposta" in updates[1]
    assert updates[-1] == "Resposta fundamentada."
    assert agent.method_calls[0] == call.replace_history(history)
    assert agent.method_calls[1].args == ("Nova pergunta",)
    assert callable(agent.method_calls[1].kwargs["on_tool_event"])
    assert callable(agent.method_calls[1].kwargs["on_token"])


def test_create_app_constroi_chat_sem_iniciar_servidor(tmp_path) -> None:
    repository_index = SimpleNamespace(file_count=12, chunk_count=30)
    artifacts = tmp_path / "session" / "artifacts"

    app = create_app(Mock(), tmp_path, repository_index, artifacts)

    assert isinstance(app, gr.Blocks)


def test_salva_diagrama_mermaid_em_diretorio_ignorado(tmp_path) -> None:
    artifact_directory = tmp_path / "artifacts"
    artifact = save_mermaid_artifact(artifact_directory, "flowchart LR\n    A --> B")

    assert artifact.parent == artifact_directory
    assert artifact.suffix == ".mmd"
    assert artifact.read_text(encoding="utf-8") == "flowchart LR\n    A --> B\n"


def test_controller_carrega_repositorio_local_e_limpa_sessao(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "main.py").write_text("print('ok')", encoding="utf-8")
    agent_factory = Mock(
        return_value=(Mock(), SimpleNamespace(file_count=1, chunk_count=2))
    )
    controller = WebRepositoryController(agent_factory)

    status = controller.load(str(repository))
    runtime = controller._runtime

    assert status.startswith("✅")
    assert runtime is not None
    workspace = runtime.session.workspace
    assert workspace.exists()
    agent_factory.assert_called_once_with(repository.resolve())

    controller.close()

    assert not workspace.exists()


def test_controller_publica_etapas_de_carregamento(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "main.py").write_text("print('ok')", encoding="utf-8")
    controller = WebRepositoryController(
        Mock(return_value=(Mock(), SimpleNamespace(file_count=1, chunk_count=2)))
    )

    updates = list(controller.load_updates(str(repository)))

    assert updates[0].startswith("⏳ Validando")
    assert "Gerando embeddings" in updates[1]
    assert updates[-1].startswith("✅")
    controller.close()


def test_controller_preserva_sessao_quando_nova_carga_falha(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "main.py").write_text("print('ok')", encoding="utf-8")
    factory = Mock(
        return_value=(Mock(), SimpleNamespace(file_count=1, chunk_count=2))
    )
    controller = WebRepositoryController(factory)
    controller.load(str(repository))
    original_runtime = controller._runtime
    factory.side_effect = RuntimeError("embedding indisponível")

    updates = list(controller.load_updates(str(repository)))

    assert updates[-1].startswith("❌")
    assert controller._runtime is original_runtime
    controller.close()


def test_controller_exige_repositorio_antes_do_chat() -> None:
    controller = WebRepositoryController(Mock())

    updates = list(controller.respond("Olá", []))

    assert updates[0][0] == "Carregue um repositório antes de iniciar o chat."
    assert updates[0][1]["__type__"] == "update"


def test_create_repository_app_constroi_tela_inicial(tmp_path) -> None:
    app, controller = create_repository_app(Mock(), str(tmp_path))

    assert isinstance(app, gr.Blocks)
    config = str(app.get_config_file())
    assert "Sobre" in config
    assert "Perguntas disponíveis" in config
    assert all(label in config for label in STANDARD_QUESTION_LABELS)
    controller.close()


def test_menu_de_perguntas_padrao_tem_rotulos_unicos() -> None:
    assert len(STANDARD_QUESTIONS) == len(STANDARD_QUESTION_LABELS) == 8
    assert len(set(STANDARD_QUESTIONS)) == len(STANDARD_QUESTIONS)
    assert len(set(STANDARD_QUESTION_LABELS)) == len(STANDARD_QUESTION_LABELS)


def test_catalogo_publico_tem_ids_unicos_e_prompts_resolvidos() -> None:
    ids = [question.id for question in QUESTION_CATALOG]

    assert len(ids) == len(set(ids))
    assert resolve_question("architecture").prompt == "Qual é a arquitetura deste projeto?"


def test_catalogo_rejeita_id_arbitrario() -> None:
    with pytest.raises(ValueError, match="inválida"):
        resolve_question("qualquer-coisa")


def test_progresso_da_tool_usa_descricao_amigavel() -> None:
    event = ToolTraceEvent(
        phase="completed",
        tool_name="semantic_search",
        arguments="{}",
        success=True,
        duration_ms=42,
    )

    progress = _format_tool_event(event)

    assert "Buscando trechos relevantes" in progress
    assert "semantic_search" not in progress
