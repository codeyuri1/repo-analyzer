import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ToolTraceEvent:
    """Evento observável emitido durante a execução de uma tool."""

    phase: Literal["started", "completed"]
    tool_name: str
    arguments: str
    success: bool | None = None
    duration_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ToolTraceRecord:
    """Registro de auditoria de uma execução, sem conteúdo ou segredos da tool."""

    question_id: str | None
    tool_name: str
    arguments: dict[str, Any]
    status: str
    duration_ms: float
    result_metadata: dict[str, Any]


def print_tool_trace(event: ToolTraceEvent) -> None:
    """Apresenta um evento de tool de maneira compacta no terminal."""

    if event.phase == "started":
        print(f"\n[tool →] {event.tool_name} {event.arguments}")
        return

    status = "ok" if event.success else "erro"
    duration = event.duration_ms or 0.0
    print(f"[tool ←] {event.tool_name}: {status} ({duration:.1f} ms)")


def tool_result_succeeded(result: object) -> bool:
    """Extrai o status padronizado de uma resposta JSON de tool."""

    if not isinstance(result, str):
        return False

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("ok") is True
