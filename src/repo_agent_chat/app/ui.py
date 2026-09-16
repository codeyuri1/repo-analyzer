from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from uuid import uuid4

import gradio as gr

from repo_agent_chat.agent import ToolAgent
from repo_agent_chat.app.session import RepositorySession
from repo_agent_chat.retrieval.indexing import RepositoryIndex
from repo_agent_chat.tracing import ToolTraceEvent

type WorkerMessage = ToolTraceEvent | tuple[str, str]
type AgentFactory = Callable[[Path], tuple[ToolAgent, RepositoryIndex]]

APP_CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.hero { padding: 0.5rem 0 0.25rem; }
.hero h1 { margin-bottom: 0.25rem; }
.repository-status { min-height: 2.5rem; }
"""


@dataclass(slots=True)
class WebRepositoryRuntime:
    """Recursos pertencentes ao repositório atualmente carregado na página."""

    session: RepositorySession
    adapter: "GradioChatAdapter"
    index: RepositoryIndex


class WebRepositoryController:
    """Troca sessões de repositório e serializa acesso ao runtime ativo."""

    def __init__(self, agent_factory: AgentFactory) -> None:
        self._agent_factory = agent_factory
        self._runtime: WebRepositoryRuntime | None = None
        self._lock = Lock()

    def load(self, source: str) -> str:
        """Executa todas as etapas e retorna o último estado para integrações síncronas."""

        final_status = "❌ Não foi possível carregar o repositório."
        for final_status in self.load_updates(source):
            pass
        return final_status

    @property
    def has_runtime(self) -> bool:
        with self._lock:
            return self._runtime is not None

    def load_updates(self, source: str) -> Iterator[str]:
        """Expõe progresso e preserva a sessão anterior se a nova carga falhar."""

        if not source.strip():
            yield "❌ Informe um caminho local ou uma URL pública do GitHub."
            return

        new_session: RepositorySession | None = None
        committed = False
        try:
            yield "⏳ Validando e preparando a fonte do repositório..."
            new_session = RepositorySession(source.strip())
            source_kind = (
                "Clone temporário preparado"
                if source.strip().startswith("https://")
                else "Diretório local validado"
            )
            yield f"⏳ {source_kind}. Gerando embeddings e construindo o índice..."
            agent, repository_index = self._agent_factory(
                new_session.repository_root
            )
            new_runtime = WebRepositoryRuntime(
                session=new_session,
                adapter=GradioChatAdapter(agent),
                index=repository_index,
            )
            with self._lock:
                previous = self._runtime
                self._runtime = new_runtime
                committed = True
            if previous is not None:
                previous.session.cleanup()

            yield (
                f"✅ `{new_session.repository_root.name}` carregado · "
                f"{repository_index.file_count} arquivos · "
                f"{repository_index.chunk_count} chunks"
            )
        except Exception as error:  # noqa: BLE001 - fronteira da UI
            yield f"❌ Não foi possível carregar o repositório: {error}"
            return
        finally:
            if new_session is not None and not committed:
                new_session.cleanup()

    def respond(
        self,
        message: str,
        history: list[dict[str, object]],
    ) -> Iterator[tuple[str, object]]:
        """Delega ao agente da sessão ativa ou orienta a carregar uma fonte."""

        with self._lock:
            runtime = self._runtime
            if runtime is None:
                yield "Carregue um repositório antes de iniciar o chat.", gr.skip()
                return
            yield from runtime.adapter.respond_with_artifact(
                message,
                history,
                runtime.session.artifact_directory,
            )

    def close(self) -> None:
        """Descarta o clone, artefatos e referências ao índice em memória."""

        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            runtime.session.cleanup()


class GradioChatAdapter:
    """Adapta o histórico do Gradio ao agente sem compartilhar conversas."""

    def __init__(self, agent: ToolAgent) -> None:
        self._agent = agent
        self._lock = Lock()

    def __call__(
        self,
        message: str,
        history: list[dict[str, object]],
    ) -> Iterator[str]:
        """Mantém o adaptador utilizável diretamente em testes e integrações."""

        yield from self.respond(message, history)

    def respond(
        self,
        message: str,
        history: list[dict[str, object]],
    ) -> Iterator[str]:
        """Expõe um generator function detectável pelo Gradio 6."""

        events: Queue[WorkerMessage] = Queue()

        def run_agent() -> None:
            try:
                with self._lock:
                    self._agent.replace_history(history)
                    answer = self._agent.ask(
                        message,
                        on_tool_event=events.put,
                        on_token=lambda token: events.put(("token", token)),
                    )
                events.put(("answer", answer))
            except Exception as error:  # noqa: BLE001 - fronteira da thread/UI
                events.put(("error", str(error)))

        Thread(target=run_agent, daemon=True).start()
        progress: list[str] = []
        draft_parts: list[str] = []
        while True:
            event = events.get()
            if isinstance(event, ToolTraceEvent):
                progress.append(_format_tool_event(event))
                yield "### Investigando o repositório\n\n" + "\n".join(progress)
                continue
            kind, content = event
            if kind == "token":
                draft_parts.append(content)
                yield "### Gerando resposta\n\n" + "".join(draft_parts)
                continue
            if kind == "error":
                yield f"Não foi possível concluir a resposta: {content}"
            else:
                yield content
            return

    def respond_with_artifact(
        self,
        message: str,
        history: list[dict[str, object]],
        artifact_directory: Path,
    ) -> Iterator[tuple[str, object]]:
        """Transmite o chat e publica o Mermaid como arquivo para download."""

        artifact_path: Path | None = None
        for update in self.respond(message, history):
            diagram = self._agent.last_mermaid_diagram
            if artifact_path is None and diagram:
                artifact_path = save_mermaid_artifact(artifact_directory, diagram)
            artifact_update = (
                gr.update(value=str(artifact_path), visible=True, interactive=True)
                if artifact_path
                else gr.skip()
            )
            yield update, artifact_update


def _format_tool_event(event: ToolTraceEvent) -> str:
    if event.phase == "started":
        return f"- ⏳ `{event.tool_name}`"
    status = "✓" if event.success else "✗"
    duration = event.duration_ms or 0.0
    return f"- {status} `{event.tool_name}` ({duration:.0f} ms)"


def save_mermaid_artifact(artifact_directory: Path, diagram: str) -> Path:
    """Persiste um diagrama versionado no workspace temporário da sessão."""

    artifact_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    suffix = uuid4().hex[:8]
    artifact_path = artifact_directory / f"dependency-diagram-{timestamp}-{suffix}.mmd"
    artifact_path.write_text(f"{diagram.rstrip()}\n", encoding="utf-8")
    return artifact_path


def create_app(
    agent: ToolAgent,
    repository_root: Path,
    repository_index: RepositoryIndex,
    artifact_directory: Path,
) -> gr.Blocks:
    """Cria a interface web sem acoplar Gradio ao núcleo do agente."""

    description = (
        f"Repositório: `{repository_root.name}` · "
        f"{repository_index.file_count} arquivos · "
        f"{repository_index.chunk_count} chunks"
    )
    adapter = GradioChatAdapter(agent)

    def respond(
        message: str,
        history: list[dict[str, object]],
    ) -> Iterator[tuple[str, object]]:
        yield from adapter.respond_with_artifact(message, history, artifact_directory)

    with gr.Blocks(title="Repo Agent Chat") as app:
        gr.Markdown(f"# Repo Agent Chat\n\n{description}")
        download = gr.DownloadButton(
            "Baixar último diagrama Mermaid (.mmd)",
            value=None,
            visible=False,
        )
        chatbot = gr.Chatbot(
            render_markdown=True,
            allow_file_downloads=True,
            height=520,
            placeholder="Faça uma pergunta sobre o repositório carregado.",
        )
        gr.ChatInterface(
            fn=respond,
            chatbot=chatbot,
            additional_outputs=[download],
            examples=[
                "Explique o projeto.",
                "Explique o fluxo principal da aplicação.",
                "Onde ocorre a validação de entrada?",
                "Analise possíveis vulnerabilidades no código-fonte.",
                "Gere um diagrama Mermaid das dependências entre os módulos.",
            ],
        )
    return app


def create_repository_app(
    agent_factory: AgentFactory,
    initial_source: str,
) -> tuple[gr.Blocks, WebRepositoryController]:
    """Cria a tela que seleciona, indexa e troca repositórios por sessão."""

    controller = WebRepositoryController(agent_factory)

    def load_repository(
        source: str,
    ) -> Iterator[tuple[object, object, object, object, object]]:
        for update in controller.load_updates(source):
            if update.startswith("✅"):
                yield (
                    update,
                    [],
                    gr.update(value=None, visible=False),
                    gr.update(visible=True),
                    gr.update(value="Carregar repositório", interactive=True),
                )
            elif update.startswith("❌"):
                yield (
                    update,
                    gr.skip(),
                    gr.skip(),
                    gr.update(visible=controller.has_runtime),
                    gr.update(value="Tentar novamente", interactive=True),
                )
            else:
                yield (
                    update,
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.update(value="Indexando...", interactive=False),
                )

    with gr.Blocks(title="Repo Agent Chat") as app:
        gr.Markdown(
            "# 🔎 Repo Agent Chat\n\n"
            "Analise arquitetura, fluxos e possíveis vulnerabilidades usando "
            "evidências do próprio código.",
            elem_classes="hero",
        )
        with gr.Accordion("Como usar", open=True):
            gr.Markdown(
                "1. Informe um diretório local ou uma URL pública do GitHub.\n"
                "2. Clique em **Carregar repositório** e aguarde a indexação.\n"
                "3. Use uma pergunta sugerida ou escreva a sua.\n\n"
                "O índice, clones e diagramas existem somente durante esta sessão."
            )
        with gr.Row():
            source = gr.Textbox(
                value=initial_source,
                label="Repositório",
                placeholder="/caminho/local ou https://github.com/dono/repositorio",
                scale=5,
            )
            load_button = gr.Button("Carregar repositório", variant="primary", scale=1)
        status = gr.Markdown(
            "ℹ️ Nenhum repositório carregado.",
            elem_classes="repository-status",
        )
        with gr.Column(visible=False) as chat_area:
            download = gr.DownloadButton(
                "Baixar último diagrama Mermaid (.mmd)",
                value=None,
                visible=False,
            )
            chatbot = gr.Chatbot(
                render_markdown=True,
                allow_file_downloads=True,
                height=520,
                placeholder="Escolha uma sugestão ou pergunte sobre o código.",
            )
            gr.ChatInterface(
                fn=controller.respond,
                chatbot=chatbot,
                additional_outputs=[download],
                submit_btn="Enviar",
                stop_btn="Parar",
                examples=[
                    "Explique o projeto.",
                    "Explique o fluxo principal da aplicação.",
                    "Verifique a segurança do código em src/.",
                    "Gere um diagrama Mermaid das dependências entre os módulos.",
                ],
            )
        load_outputs = [status, chatbot, download, chat_area, load_button]
        load_button.click(
            fn=load_repository,
            inputs=[source],
            outputs=load_outputs,
        )
        source.submit(
            fn=load_repository,
            inputs=[source],
            outputs=load_outputs,
        )
    return app, controller
