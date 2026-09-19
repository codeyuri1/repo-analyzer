from dataclasses import dataclass, field

from repo_agent_chat.tracing import ToolTraceRecord


@dataclass(slots=True)
class AgentTurnState:
    """Estado transitório e observável de uma única pergunta ao agente."""

    tool_calls: list[str] = field(default_factory=list)
    evidence_paths: set[str] = field(default_factory=set)
    evidence_ranges: set[str] = field(default_factory=set)
    evidence_excerpts: list[str] = field(default_factory=list)
    semantic_candidates: list[tuple[str, int, int]] = field(default_factory=list)
    listed_paths: set[str] = field(default_factory=set)
    workflow_evidence_paths: set[str] = field(default_factory=set)
    workflow_documents: dict[str, str] = field(default_factory=dict)
    mermaid_diagram: str | None = None
    tool_trace: list[ToolTraceRecord] = field(default_factory=list)
