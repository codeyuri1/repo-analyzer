import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from openai import OpenAIError

from repo_agent_chat.agent import ToolAgent
from repo_agent_chat.app.config import Settings, get_settings
from repo_agent_chat.app.session import RepositorySession, RepositorySourceError
from repo_agent_chat.evaluation.cases import (
    EVAL_EXCLUDED_PATHS,
    print_evaluation_report,
    run_evaluation_suite,
)
from repo_agent_chat.retrieval.embeddings import OllamaEmbeddings
from repo_agent_chat.retrieval.indexing import (
    EmptyRepositoryError,
    RepositoryIndex,
    index_repository,
)
from repo_agent_chat.tools import RepositoryTools
from repo_agent_chat.tracing import print_tool_trace


class StreamingChat(Protocol):
    """Contrato mínimo exigido pela interface de terminal."""

    def stream(self, question: str) -> Iterator[str]: ...


def run_chat(chat: StreamingChat) -> None:
    """Executa o loop interativo do chat no terminal."""

    print("Repo Agent Chat")
    print("Digite 'sair' para encerrar.\n")

    while True:
        question = input("Você: ").strip()

        if question.lower() in {"sair", "exit", "quit"}:
            print("Chat encerrado.")
            break

        if not question:
            print("Digite uma pergunta ou 'sair' para encerrar.")
            continue

        try:
            answer_started = False
            for content in chat.stream(question):
                if not answer_started:
                    print("\nAssistente: ", end="", flush=True)
                    answer_started = True
                print(content, end="", flush=True)
        except (OpenAIError, RuntimeError, ValueError) as error:
            print(f"\nErro ao se comunicar com o modelo: {error}")
            continue

        print("\n")


def build_tool_agent(
    root: Path,
    settings: Settings,
    excluded_paths: frozenset[str] = frozenset(),
) -> tuple[ToolAgent, RepositoryIndex]:
    """Monta as dependências e indexa o repositório escolhido."""

    embeddings = OllamaEmbeddings(settings)
    repository_index = index_repository(
        root,
        embeddings,
        excluded_paths=excluded_paths,
        max_chunks=settings.max_index_chunks,
    )
    tools = RepositoryTools(root, repository_index.retriever)
    agent = ToolAgent(settings, tools, on_tool_event=print_tool_trace)
    return agent, repository_index


def parse_args() -> argparse.Namespace:
    """Lê os argumentos fornecidos na linha de comando."""

    parser = argparse.ArgumentParser(
        description="Converse com um repositório local ou público do GitHub usando RAG.",
    )
    parser.add_argument(
        "repository",
        nargs="?",
        default=str(Path.cwd()),
        help="Caminho local ou URL HTTPS do GitHub (padrão: diretório atual).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Executa a suíte de avaliações em vez do chat interativo.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Inicia a interface web Gradio em vez do terminal.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host da interface web.")
    parser.add_argument("--port", type=int, default=7860, help="Porta da interface web.")
    return parser.parse_args()


def main() -> int:
    """Indexa o repositório e inicia a aplicação."""

    args = parse_args()
    settings = get_settings()

    print(f"Provedor de IA: {settings.ai_provider}")
    print(f"Modelo de chat: {settings.chat_model}")
    print(f"Modelo de embeddings: {settings.embedding_model}")

    if args.web and not args.eval:
        from repo_agent_chat.app.ui import APP_CSS, create_repository_app

        app, controller = create_repository_app(
            lambda repository_root: build_tool_agent(repository_root, settings),
            args.repository,
        )
        try:
            app.launch(server_name=args.host, server_port=args.port, css=APP_CSS)
        except (KeyboardInterrupt, EOFError):
            print("\nInterface encerrada.")
        finally:
            controller.close()
        return 0

    try:
        session_context = RepositorySession(args.repository)
    except (OSError, RepositorySourceError) as error:
        print(f"Não foi possível preparar o repositório: {error}")
        return 1

    with session_context as session:
        print(f"Indexando repositório: {session.repository_root}")
        try:
            excluded_paths = EVAL_EXCLUDED_PATHS if args.eval else frozenset()
            assistant, repository_index = build_tool_agent(
                session.repository_root,
                settings,
                excluded_paths,
            )
        except (EmptyRepositoryError, OpenAIError, OSError, ValueError) as error:
            print(f"Não foi possível indexar o repositório: {error}")
            return 1

        print(
            f"Indexação concluída: {repository_index.file_count} arquivos, "
            f"{repository_index.chunk_count} chunks.\n"
        )

        try:
            if args.eval:
                results = run_evaluation_suite(assistant, session.repository_root)
                print_evaluation_report(results)
                return 0 if all(result.passed for result in results) else 1
            run_chat(assistant)
        except (KeyboardInterrupt, EOFError):
            print("\nChat encerrado.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
