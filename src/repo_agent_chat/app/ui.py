from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from uuid import uuid4

import gradio as gr

from repo_agent_chat.agent import ToolAgent
from repo_agent_chat.app.questions import (
    QUESTION_CATALOG,
    resolve_question,
)
from repo_agent_chat.app.session import RepositorySession
from repo_agent_chat.retrieval.indexing import RepositoryIndex
from repo_agent_chat.tracing import ToolTraceEvent

type WorkerMessage = ToolTraceEvent | tuple[str, str]
type AgentFactory = Callable[[Path], tuple[ToolAgent, RepositoryIndex]]

STANDARD_QUESTIONS = [question.prompt for question in QUESTION_CATALOG]
STANDARD_QUESTION_LABELS = [question.label for question in QUESTION_CATALOG]

TOOL_LABELS = {
    "list_files": "Mapeando arquivos",
    "read_file": "Lendo evidências no código",
    "semantic_search": "Buscando trechos relevantes",
    "analyze_vulnerabilities": "Executando análise estática",
    "generate_mermaid_diagram": "Gerando diagrama de módulos",
}

APP_CSS = r"""
:root {
  --rac-ink: #202b36;
  --rac-muted: #435261;
  --rac-line: #cbd5dd;
  --rac-panel: rgba(255, 255, 255, 0.94);
  --rac-primary: #315b7d;
  --rac-primary-dark: #244762;
  --rac-accent: #54758e;
  --rac-soft: #edf2f5;
}

body {
  background:
    radial-gradient(circle at 8% 0%, rgba(49, 91, 125, 0.08), transparent 30rem),
    #f4f6f7;
}

body, button, input, textarea, .gradio-container {
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif !important;
  font-variant-ligatures: contextual;
  text-rendering: optimizeLegibility;
}

h1, h2, h3 {
  font-weight: 720 !important;
  letter-spacing: -0.025em;
}

.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 22px 28px 44px !important;
  color: var(--rac-ink);
}

.hero {
  position: relative;
  overflow: hidden;
  margin-bottom: 18px;
  padding: 36px 40px !important;
  border: 1px solid rgba(255, 255, 255, 0.16) !important;
  border-radius: 24px !important;
  background: linear-gradient(125deg, #334654 0%, #465d6e 100%);
  box-shadow: 0 16px 40px rgba(51, 65, 85, 0.14);
}

.hero::after {
  content: "";
  position: absolute;
  width: 260px;
  height: 260px;
  top: -150px;
  right: -70px;
  border: 46px solid rgba(255, 255, 255, 0.07);
  border-radius: 999px;
}

.hero h1 {
  margin: 0 0 8px !important;
  color: #fff !important;
  font-size: clamp(2rem, 5vw, 3.25rem) !important;
  letter-spacing: -0.045em;
  line-height: 1.02 !important;
}

.hero-eyebrow {
  display: block;
  margin-bottom: 12px;
  color: #d6e0e5;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.hero p {
  max-width: 720px;
  margin: 0 !important;
  color: #e8eef1 !important;
  font-size: 1.05rem;
  line-height: 1.65;
}

.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 20px;
}

.hero-badge {
  padding: 6px 10px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.10);
  color: #f8fafc;
  font-size: 0.78rem;
  font-weight: 650;
  letter-spacing: 0.02em;
}

.portfolio-section {
  margin-top: 18px;
  margin-bottom: 4px;
  border: 1px solid var(--rac-line) !important;
  border-radius: 20px !important;
  background: var(--rac-panel) !important;
  box-shadow: 0 10px 30px rgba(51, 65, 85, 0.06);
}

.portfolio-section > .label-wrap {
  padding: 4px 8px !important;
  font-weight: 750 !important;
}

.portfolio-tabs { gap: 16px !important; }

.portfolio-tabs .tab-nav {
  width: fit-content;
  margin: 0 auto 18px !important;
  padding: 4px !important;
  border: 1px solid var(--rac-line) !important;
  border-radius: 12px !important;
  background: rgba(255, 255, 255, 0.72) !important;
}

.portfolio-tabs button[role="tab"] {
  min-height: 40px !important;
  padding: 8px 18px !important;
  border: 0 !important;
  border-radius: 9px !important;
  color: var(--rac-muted) !important;
  font-size: 0.88rem !important;
  font-weight: 650 !important;
}

.portfolio-tabs button[role="tab"]:hover {
  background: #eef2f4 !important;
  color: var(--rac-ink) !important;
}

.portfolio-tabs button[role="tab"][aria-selected="true"] {
  background: #315b7d !important;
  color: #fff !important;
  box-shadow: 0 2px 8px rgba(36, 71, 98, 0.20) !important;
}

.about-intro {
  max-width: 820px;
  margin: 6px auto 20px !important;
  text-align: center;
}

.about-intro h2 {
  margin-bottom: 8px !important;
  color: var(--rac-ink) !important;
  font-size: clamp(1.45rem, 3vw, 2rem) !important;
}

.about-intro p {
  color: var(--rac-muted) !important;
  font-size: 0.98rem;
  line-height: 1.7;
}

.portfolio-intro {
  padding: 4px 8px 8px !important;
}

.portfolio-intro h2 {
  margin-bottom: 6px !important;
  color: var(--rac-ink) !important;
  font-size: 1.35rem !important;
}

.portfolio-intro p {
  max-width: 900px;
  color: var(--rac-muted) !important;
  line-height: 1.65;
}

.portfolio-card {
  box-sizing: border-box;
  min-width: 0 !important;
  max-width: 100%;
  min-height: 168px;
  padding: 18px 20px !important;
  border: 1px solid var(--rac-line) !important;
  border-radius: 16px !important;
  background: #fff !important;
  overflow-wrap: anywhere;
  word-break: normal;
  overflow-x: clip;
}

.portfolio-card > *,
.portfolio-card .prose,
.portfolio-card p,
.portfolio-card li,
.portfolio-card code {
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
  white-space: normal;
}

.about-cards,
.about-details {
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px !important;
  min-width: 0 !important;
  max-width: 100%;
  align-items: stretch !important;
}

.about-cards > *,
.about-details > * {
  box-sizing: border-box;
  min-width: 0 !important;
  max-width: 100%;
  width: 100%;
}

.portfolio-card h3 {
  margin: 0 0 8px !important;
  color: var(--rac-ink) !important;
  font-size: 1rem !important;
}

.portfolio-card p, .portfolio-card li {
  color: var(--rac-muted) !important;
  font-size: 0.88rem;
  line-height: 1.55;
}

.portfolio-card ul { margin: 6px 0 0 !important; padding-left: 18px !important; }

.stack-line {
  margin: 10px 8px 6px !important;
  padding: 13px 16px !important;
  border: 1px solid #c7d5df !important;
  border-left: 4px solid var(--rac-accent) !important;
  border-radius: 8px !important;
  background: #eaf0f4 !important;
  color: #344757 !important;
  font-size: 0.84rem;
  line-height: 1.6;
}

.setup-card, .workspace-card {
  border: 1px solid var(--rac-line) !important;
  border-radius: 20px !important;
  background: var(--rac-panel) !important;
  box-shadow: 0 12px 34px rgba(51, 65, 85, 0.07);
}

.setup-card {
  padding: 24px !important;
}
.workspace-card { padding: 10px 14px 16px !important; }

.section-heading h2 {
  margin: 0 0 4px !important;
  color: var(--rac-ink) !important;
  font-size: 1.05rem !important;
  letter-spacing: -0.015em;
}

.section-heading p {
  margin: 0 0 14px !important;
  color: var(--rac-muted) !important;
  font-size: 0.9rem;
}

.repository-input textarea,
.repository-input input {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
  border-radius: 11px !important;
}

.repository-input textarea:focus,
.repository-input input:focus {
  border-color: var(--rac-primary) !important;
  box-shadow: 0 0 0 3px rgba(49, 91, 125, 0.16) !important;
}

.load-button {
  min-height: 54px !important;
  border: 0 !important;
  border-radius: 12px !important;
  background: var(--rac-primary) !important;
  box-shadow: 0 5px 14px rgba(36, 71, 98, 0.22) !important;
  font-weight: 700 !important;
  transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease !important;
}

.load-button:hover {
  transform: translateY(-1px);
  background: var(--rac-primary-dark) !important;
  box-shadow: 0 8px 18px rgba(36, 71, 98, 0.28) !important;
}

.load-button:focus-visible, .diagram-download:focus-visible {
  outline: 3px solid rgba(49, 91, 125, 0.28) !important;
  outline-offset: 2px;
}

.repository-status {
  min-height: 46px;
  margin-top: 10px;
  padding: 12px 14px !important;
  border: 1px solid #cbd5dd !important;
  border-radius: 12px !important;
  background: var(--rac-soft) !important;
}

.repository-status p { margin: 0 !important; color: var(--rac-ink) !important; }

.usage-guide {
  height: 100%;
  padding: 20px 22px !important;
  border: 1px solid var(--rac-line) !important;
  border-radius: 20px !important;
  background: rgba(255, 255, 255, 0.72) !important;
}

.usage-guide h3 {
  margin-top: 0 !important;
  color: var(--rac-ink) !important;
  font-size: 1rem !important;
}

.usage-guide ol { margin-bottom: 12px !important; padding-left: 22px !important; }
.usage-guide li { margin: 7px 0; color: var(--rac-muted); }
.privacy-note { color: var(--rac-muted); font-size: 0.82rem; line-height: 1.45; }

.question-menu {
  margin: 14px 0 !important;
  padding: 14px 16px !important;
  border: 1px solid var(--rac-line) !important;
  border-radius: 12px !important;
  background: var(--rac-soft) !important;
}

.question-menu strong { color: var(--rac-ink); }
.question-menu li { margin: 5px 0; color: var(--rac-muted); font-size: 0.84rem; }

.workspace-title {
  padding: 12px 8px 2px !important;
}

.workspace-title h2 { margin: 0 !important; font-size: 1.1rem !important; }
.workspace-title p { margin: 4px 0 0 !important; color: var(--rac-muted) !important; }

.diagram-download {
  max-width: 280px;
  margin: 6px 6px 10px auto !important;
  border-color: #b8c8e0 !important;
}

.agent-chat {
  overflow: hidden;
  border: 1px solid var(--rac-line) !important;
  border-radius: 16px !important;
  background: #fbfcfe !important;
}

.chat-input textarea {
  min-height: 58px !important;
  padding: 12px 14px !important;
  border-radius: 12px !important;
  line-height: 1.5 !important;
}

.chat-input textarea:focus {
  border-color: var(--rac-primary) !important;
  box-shadow: 0 0 0 3px rgba(49, 91, 125, 0.16) !important;
}

.workspace-card button.primary {
  background: var(--rac-primary) !important;
  border-color: var(--rac-primary) !important;
}

.workspace-card button.primary:hover {
  background: var(--rac-primary-dark) !important;
}

.footer-note {
  margin-top: 16px;
  text-align: center;
  color: var(--rac-muted);
  font-size: 0.78rem;
}

@media (max-width: 760px) {
  .gradio-container { padding: 12px 12px 28px !important; }
  .hero { padding: 26px 22px !important; border-radius: 18px !important; }
  .hero h1 { font-size: 2.15rem !important; }
  .setup-card, .usage-guide { border-radius: 16px !important; }
  .load-button { width: 100% !important; }
  .diagram-download { max-width: none; margin-left: 6px !important; }
  .portfolio-card { min-height: auto; }
  .about-cards, .about-details { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 520px) {
  .about-cards, .about-details { grid-template-columns: minmax(0, 1fr); }
}

@media (prefers-color-scheme: dark) {
  :root {
    --rac-ink: #f1f5f8;
    --rac-muted: #c4d0da;
    --rac-line: #465564;
    --rac-panel: rgba(27, 37, 48, 0.97);
    --rac-soft: #253441;
  }
  body { background: #111820; }
  .repository-status { background: #253441 !important; border-color: #465564 !important; }
  .usage-guide { background: rgba(27, 37, 48, 0.92) !important; }
  .agent-chat { background: #18232d !important; }
  .portfolio-card { background: #1b2530 !important; }
  .portfolio-tabs .tab-nav { background: #202c37 !important; }
  .portfolio-tabs button[role="tab"] { color: #c4d0da !important; }
  .portfolio-tabs button[role="tab"]:hover { background: #344554 !important; }
  .portfolio-tabs button[role="tab"][aria-selected="true"] {
    background: #315b7d !important;
    color: #fff !important;
  }
  .setup-card.bg-white\/90,
  .portfolio-section.bg-white\/90 {
    background: var(--rac-panel) !important;
    border-color: var(--rac-line) !important;
  }
  .usage-guide.bg-slate-50\/90 {
    background: #1b2530 !important;
    border-color: var(--rac-line) !important;
  }
  .stack-line { background: #253441 !important; color: #d7e1e8 !important; }
}
"""

TAILWIND_CSS_PATH = Path(__file__).with_name("static") / "tailwind.css"
if TAILWIND_CSS_PATH.exists():
    APP_CSS += TAILWIND_CSS_PATH.read_text(encoding="utf-8")


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
                adapter=GradioChatAdapter(agent, new_session.repository_root),
                index=repository_index,
            )
            with self._lock:
                previous = self._runtime
                self._runtime = new_runtime
                committed = True
            if previous is not None:
                previous.session.cleanup()

            coverage = (
                " · índice parcial otimizado"
                if getattr(repository_index, "truncated", False)
                else ""
            )
            yield (
                f"✅ `{new_session.repository_root.name}` carregado · "
                f"{repository_index.file_count} arquivos · "
                f"{repository_index.chunk_count} chunks{coverage}"
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

    def respond_question(
        self,
        question_id: str,
        history: list[dict[str, object]] | None,
    ) -> Iterator[tuple[object, ...]]:
        """Executa somente uma ação do catálogo público da interface."""

        question = resolve_question(question_id)
        current_history = list(history or [])
        user_message = {"role": "user", "content": question.label}
        base_history = current_history + [user_message]

        with self._lock:
            runtime = self._runtime
            if runtime is None:
                yield (
                    current_history,
                    gr.skip(),
                    "Carregue um repositório antes de iniciar uma análise.",
                    *_question_button_updates(disabled=False),
                )
                return

        yield (
            base_history + [{"role": "assistant", "content": f"### {question.label}\n\nAnalisando..."}],
            gr.skip(),
            f"⏳ Analisando: **{question.label}**",
            *_question_button_updates(disabled=True),
        )
        try:
            for answer, artifact in runtime.adapter.respond_with_artifact(
                question.prompt,
                current_history,
                runtime.session.artifact_directory,
                question_id=question.id,
            ):
                yield (
                    base_history + [{"role": "assistant", "content": answer}],
                    artifact,
                    f"⏳ Analisando: **{question.label}**",
                    *_question_button_updates(disabled=True),
                )
            yield (
                base_history + [{"role": "assistant", "content": answer}],
                artifact,
                f"✓ Análise concluída: **{question.label}**",
                *_question_button_updates(disabled=False),
            )
        except Exception as error:  # noqa: BLE001 - fronteira da UI
            error_message = f"Não foi possível concluir a análise: {error}"
            yield (
                base_history + [{"role": "assistant", "content": error_message}],
                gr.skip(),
                error_message,
                *_question_button_updates(disabled=False),
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

    def __init__(self, agent: ToolAgent, repository_root: Path | None = None) -> None:
        self._agent = agent
        self._repository_root = repository_root
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
        question_id: str | None = None,
    ) -> Iterator[str]:
        """Expõe um generator function detectável pelo Gradio 6."""

        events: Queue[WorkerMessage] = Queue()

        def run_agent() -> None:
            try:
                with self._lock:
                    self._agent.replace_history(history)
                    kwargs = {
                        "on_tool_event": events.put,
                        "on_token": lambda token: events.put(("token", token)),
                    }
                    if question_id is not None:
                        kwargs["question_id"] = question_id
                    answer = self._agent.ask(message, **kwargs)
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
                yield _with_analysis_trace(
                    content, self._agent, question_id, self._repository_root
                )
            return

    def respond_with_artifact(
        self,
        message: str,
        history: list[dict[str, object]],
        artifact_directory: Path,
        question_id: str | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Transmite o chat e publica o Mermaid como arquivo para download."""

        artifact_path: Path | None = None
        for update in self.respond(message, history, question_id=question_id):
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
    label = TOOL_LABELS.get(event.tool_name, event.tool_name)
    if event.phase == "started":
        return f"- ⏳ {label}"
    status = "✓" if event.success else "✗"
    duration = event.duration_ms or 0.0
    return f"- {status} {label} ({duration:.0f} ms)"


def _with_analysis_trace(
    answer: str,
    agent: ToolAgent,
    question_id: str | None,
    repository_root: Path | None,
) -> str:
    """Expõe auditoria operacional compacta, nunca raciocínio interno."""

    if question_id is None or repository_root is None:
        return answer
    trace = agent.last_tool_trace
    if not trace:
        return answer
    events = "\n".join(
        f"- {'✓' if item.status == 'ok' else '✗'} {TOOL_LABELS.get(item.tool_name, item.tool_name)}"
        for item in trace
    )
    from repo_agent_chat.evaluation.investigations import evaluate_investigation

    result = evaluate_investigation(
        resolve_question(question_id),
        answer,
        trace,
        repository_root,
        agent.last_evidence_paths,
        agent.last_evidence_ranges,
        agent.last_evidence_excerpts,
    )
    checks = "\n".join(
        f"- {'✓' if check.passed else '✗'} {check.name}: {check.reason}"
        for check in result.checks
    )
    sources = len(agent.last_evidence_paths)
    citations = sum(check.name == "citations_valid" and check.passed for check in result.checks)
    return (
        f"{answer}\n\n<details><summary>Analysis trace</summary>\n\n{events}\n\n"
        f"Sources: {sources} · Citations verified: {'yes' if citations else 'no'}\n\n"
        f"Evaluation: {'✓ passed' if result.passed else '✗ failed'}\n{checks}\n</details>"
    )


def _question_button_updates(*, disabled: bool) -> tuple[object, ...]:
    return tuple(gr.update(interactive=not disabled) for _ in QUESTION_CATALOG)


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
        gr.Markdown(
            "# Repo Agent Chat\n\n"
            "Explore o código com respostas fundamentadas em evidências.\n\n"
            '<div class="hero-badges">'
            '<span class="hero-badge">RAG local</span>'
            '<span class="hero-badge">Citações verificáveis</span>'
            '<span class="hero-badge">Análise segura</span>'
            "</div>",
            elem_classes="hero",
        )
        gr.Markdown(f"## Repositório ativo\n\n{description}", elem_classes="section-heading")
        download = gr.DownloadButton(
            "↓ Baixar diagrama Mermaid",
            value=None,
            visible=False,
            elem_classes="diagram-download",
        )
        chatbot = gr.Chatbot(
            render_markdown=True,
            allow_file_downloads=True,
            height=520,
            placeholder="Faça uma pergunta sobre o repositório carregado.",
            elem_classes="agent-chat",
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

    def question_handler(question_id: str):
        def handle(history: list[dict[str, object]] | None):
            yield from controller.respond_question(question_id, history or [])

        return handle

    def load_repository(
        source: str,
    ) -> Iterator[tuple[object, ...]]:
        for update in controller.load_updates(source):
            if update.startswith("✅"):
                yield (
                    update,
                    [],
                    gr.update(value=None, visible=False),
                    gr.update(visible=True),
                    gr.update(value="Carregar outro repositório", interactive=True),
                    gr.update(value="Selecione uma investigação para começar."),
                )
            elif update.startswith("❌"):
                yield (
                    update,
                    gr.skip(),
                    gr.skip(),
                    gr.update(visible=controller.has_runtime),
                    gr.update(value="Tentar novamente", interactive=True),
                    gr.skip(),
                )
            else:
                yield (
                    update,
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.update(value="Indexando...", interactive=False),
                    gr.skip(),
                )

    with gr.Blocks(title="Repo Agent Chat") as app:
        gr.Markdown(
            '<div class="brand"><span class="brand-mark">&gt;_</span>'
            '<span>Repo Agent Chat</span><span class="nav-hint">Ambiente de engenharia de software com IA</span>'
            '<span class="nav-hint" style="margin-left:auto">'
            '<a href="https://github.com/codeyuri1/repo-analyzer" target="_blank">GitHub</a></span></div>',
            elem_classes="app-header",
        )
        gr.Markdown(
            '<span class="hero-eyebrow">AI SOFTWARE ENGINEERING</span>\n\n'
            "# Entenda qualquer codebase. Pergunte. Rastreie as evidências.\n\n"
            "Analise a arquitetura, rastreie fluxos de execução e investigue riscos com respostas "
            "fundamentadas diretamente no código-fonte.\n\n"
            '<div class="hero-badges">'
            '<span class="hero-badge">RAG</span><span class="hero-badge">Busca híbrida</span>'
            '<span class="hero-badge">Reranking</span><span class="hero-badge">Tools</span>'
            '<span class="hero-badge">IA local</span><span class="hero-badge">Citações de fontes</span>'
            "</div>",
            elem_classes="hero",
        )
        with gr.Tabs(elem_id="primary-navigation", elem_classes="portfolio-tabs"):
            with gr.Tab("Analisar", id="analisar"):
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(
                        scale=7,
                        min_width=360,
                        elem_classes=[
                            "setup-card",
                            "bg-white/90",
                            "border-slate-200",
                        ],
                        elem_id="repository-analyzer",
                    ):
                        gr.Markdown(
                            "## Analisar um repositório\n\n"
                            "Conecte um repositório local ou uma URL pública do GitHub.",
                            elem_classes="section-heading",
                        )
                        source = gr.Textbox(
                            value=initial_source,
                            label="Repositório",
                            placeholder="/workspace/meu-projeto ou https://github.com/usuario/repositorio",
                            elem_id="repository-source",
                            elem_classes="repository-input",
                        )
                        load_button = gr.Button(
                            "Analisar repositório",
                            variant="primary",
                            elem_id="analyze-repository-button",
                            elem_classes=[
                                "load-button",
                                "bg-brand-600",
                                "hover:bg-brand-700",
                            ],
                        )
                        status = gr.Markdown(
                            "ℹ️ Pronto para analisar um repositório.",
                            elem_id="repository-status",
                            elem_classes="repository-status",
                        )
                    with gr.Column(
                        scale=4,
                        min_width=280,
                        elem_classes=[
                            "usage-guide",
                            "bg-slate-50/90",
                            "border-slate-200",
                        ],
                    ):
                        gr.Markdown(
                            "### Como funciona\n\n"
                            "**01** Conecte o repositório  \n**02** Indexe o código  \n"
                            "**03** Faça perguntas  \n**04** Verifique as fontes\n\n"
                            '<div class="privacy-note">🔒 Acesso somente leitura. O índice e '
                            "os artefatos são temporários nesta sessão.</div>"
                        )
                with gr.Column(
                    visible=False,
                        elem_id="repository-workspace",
                        elem_classes="workspace-card",
                ) as chat_area:
                    gr.Markdown(
                        "## Workspace do repositório\n\n"
                        "Faça perguntas, acompanhe as tools e valide as fontes.",
                        elem_classes="workspace-title",
                    )
                    download = gr.DownloadButton(
                            "↓ Baixar diagrama Mermaid",
                        value=None,
                        visible=False,
                        elem_classes="diagram-download",
                    )
                    chatbot = gr.Chatbot(
                        label=None,
                        show_label=False,
                        render_markdown=True,
                        allow_file_downloads=True,
                        height=560,
                        placeholder=(
                            "Escolha uma pergunta abaixo para iniciar a análise do repositório."
                        ),
                        elem_classes="agent-chat",
                        layout="bubble",
                        buttons=["copy", "copy_all"],
                        feedback_options=("Útil", "Precisa melhorar"),
                    )
                    gr.Markdown(
                        "### O que você quer investigar?",
                        elem_classes="question-menu-title",
                    )
                    question_buttons: list[gr.Button] = []
                    with gr.Row(elem_classes="question-actions"):
                        for question in QUESTION_CATALOG[:4]:
                            question_buttons.append(
                                gr.Button(
                                    f"◇ {question.label}\n{question.description}",
                                    elem_id=f"question-{question.id}",
                                    elem_classes="question-action",
                                )
                            )
                    with gr.Row(elem_classes="question-actions"):
                        for question in QUESTION_CATALOG[4:]:
                            question_buttons.append(
                                gr.Button(
                                    f"◇ {question.label}\n{question.description}",
                                    elem_id=f"question-{question.id}",
                                    elem_classes="question-action",
                                )
                            )
                    analysis_status = gr.Markdown(
                        "Selecione uma investigação para começar.",
                        elem_id="analysis-status",
                        elem_classes="analysis-status",
                    )
                    for question, button in zip(QUESTION_CATALOG, question_buttons):
                        button.click(
                            fn=question_handler(question.id),
                            inputs=[chatbot],
                            outputs=[chatbot, download, analysis_status, *question_buttons],
                        )
                    gr.Markdown(
                        '<div class="question-reminder">'
                        "<strong>Perguntas disponíveis</strong>"
                        '<div class="question-reminder-list">'
                        + "".join(
                            f'<span class="question-reminder-chip">{label}</span>'
                            for label in STANDARD_QUESTION_LABELS
                        )
                        + "</div></div>",
                        elem_id="available-questions-after-chat",
                        elem_classes="question-reminder",
                    )
            with gr.Tab("Arquitetura", id="arquitetura"), gr.Column(
                elem_id="architecture-content", elem_classes="architecture-panel"
            ):
                    gr.Markdown(
                        "## Arquitetura do Repo Agent Chat\n\n"
                        "O fluxo combina análise determinística com raciocínio de LLM. "
                        "A recuperação usa candidatos vetoriais e lexicais, seguidos de reranking "
                        "determinístico; as tools confirmam evidências antes da resposta final."
                    )
                    gr.Markdown(
                        '<div class="pipeline">'
                        '<span class="pipeline-step">Repository</span><span class="pipeline-arrow">→</span>'
                        '<span class="pipeline-step">Read &amp; chunk</span><span class="pipeline-arrow">→</span>'
                        '<span class="pipeline-step">Embeddings</span><span class="pipeline-arrow">→</span>'
                        '<span class="pipeline-step">Vector + lexical search</span><span class="pipeline-arrow">→</span>'
                        '<span class="pipeline-step">Deterministic reranking</span><span class="pipeline-arrow">→</span>'
                        '<span class="pipeline-step">Context</span><span class="pipeline-arrow">→</span>'
                        '<span class="pipeline-step">LLM + tools</span><span class="pipeline-arrow">→</span>'
                        '<span class="pipeline-step">Cited answer</span></div>'
                    )
                    gr.Markdown(
                        "**Tools disponíveis** · busca semântica · leitura de arquivos · listagem de arquivos · "
                        "análise estática heurística · geração de diagramas Mermaid\n\n"
                        "O projeto mantém acesso somente leitura ao repositório, providers intercambiáveis "
                        "e retorna referências `arquivo:linha` quando há evidência."
                    )
            with gr.Tab("Sobre", id="sobre"):
                gr.Markdown(
                    "## Engineering an AI agent for code understanding\n\n"
                    "O **Repo Agent Chat** combina análise determinística de software com "
                    "raciocínio de LLM e retrieval para transformar um repositório desconhecido "
                    "em uma investigação verificável. A interface expõe o caminho da pergunta "
                    "até a evidência, em vez de esconder o processo em um chatbot genérico.",
                    elem_classes="about-editorial",
                )
                gr.Markdown(
                    "### Pipeline\n\n"
                    '<div class="pipeline">'
                    '<span class="pipeline-step">Repositório</span><span class="pipeline-arrow">→</span>'
                    '<span class="pipeline-step">Leitura</span><span class="pipeline-arrow">→</span>'
                    '<span class="pipeline-step">Chunking</span><span class="pipeline-arrow">→</span>'
                    '<span class="pipeline-step">Embeddings</span><span class="pipeline-arrow">→</span>'
                    '<span class="pipeline-step">Busca híbrida</span><span class="pipeline-arrow">→</span>'
                    '<span class="pipeline-step">Reranking</span><span class="pipeline-arrow">→</span>'
                    '<span class="pipeline-step">Resposta citada</span></div>\n\n'
                    "### Destaques de engenharia\n\n"
                    "Inferência local com Ollama ou providers compatíveis · healthcheck no Docker Compose · "
                    "testes automatizados e lint · acesso somente leitura · citações verificáveis.",
                    elem_classes="about-editorial",
                )
                gr.Markdown(
                    "**Stack:** Python · Gradio · Tailwind CSS · Ollama/OpenAI · RAG · Pytest · Ruff · Docker",
                    elem_classes="stack-line",
                )
        gr.Markdown(
            "Repo Agent Chat · análise baseada em evidências sob seu controle.",
            elem_classes="footer-note",
        )
        load_outputs = [status, chatbot, download, chat_area, load_button, analysis_status]
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
