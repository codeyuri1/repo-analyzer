"""Evals em camadas para as ações declaradas no catálogo."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from repo_agent_chat.app.questions import QuestionDefinition
from repo_agent_chat.evaluation.cases import extract_citations
from repo_agent_chat.evaluation.citations import CitationEvaluator
from repo_agent_chat.evaluation.faithfulness import (
    extract_referenced_symbols,
    symbol_exists,
)
from repo_agent_chat.tracing import ToolTraceRecord


@dataclass(frozen=True, slots=True)
class EvalCheck:
    name: str
    passed: bool
    level: int
    reason: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class InvestigationEvalResult:
    question_id: str
    passed: bool
    checks: tuple[EvalCheck, ...]
    aggregate_score: float


def evaluate_investigation(definition: QuestionDefinition, answer: str, trace: list[ToolTraceRecord], repository_root: Path, evidence_paths: set[str], evidence_ranges: set[str], evidence_excerpts: list[str]) -> InvestigationEvalResult:
    """Executa Level 1/2 sem LLM; grupos esperados são apenas diagnósticos."""
    citations = extract_citations(answer)
    citation_result = CitationEvaluator().evaluate(citations, repository_root, evidence_ranges, evidence_paths)
    allowed = set(definition.allowed_tools)
    checks = [
        EvalCheck("answer_non_empty", bool(answer.strip()), 1, "resposta presente" if answer.strip() else "resposta vazia"),
        EvalCheck("allowed_tools", all(item.tool_name in allowed for item in trace), 1, "todas as tools respeitam a política" if all(item.tool_name in allowed for item in trace) else "há tool fora da política"),
        EvalCheck("tools_succeeded", all(item.status == "ok" for item in trace), 1, "execuções concluídas" if all(item.status == "ok" for item in trace) else "há execução de tool com erro"),
        EvalCheck("sources_present", bool(evidence_paths) if definition.evidence_policy.require_sources else True, 1, "fontes recuperadas" if evidence_paths else "nenhuma fonte recuperada"),
        EvalCheck("citations_present", bool(citations) if definition.evidence_policy.require_citations else True, 1, "citações presentes" if citations else "nenhuma citação encontrada"),
        EvalCheck("citations_valid", citation_result.passed if citations else not definition.evidence_policy.require_citations, 1, "citações verificadas" if citation_result.passed else "citação fora do repositório, sem linha ou sem evidência recuperada"),
    ]
    symbols = extract_referenced_symbols(answer)
    evidence = "\n".join(evidence_excerpts)
    checks.append(EvalCheck("symbols_exist", all(symbol_exists(symbol, evidence) for symbol in symbols), 2, "símbolos confirmados" if all(symbol_exists(symbol, evidence) for symbol in symbols) else "símbolo citado não aparece nas evidências"))
    if definition.evidence_policy.require_mermaid:
        valid = _valid_mermaid(answer)
        checks.append(EvalCheck("mermaid_valid", valid, 1, "Mermaid estruturalmente válido" if valid else "bloco Mermaid flowchart ausente ou inválido"))
    for group, tools in definition.expected_tool_groups:
        observed = any(item.tool_name in tools and item.status == "ok" for item in trace)
        checks.append(EvalCheck(f"expected:{group}", observed, 1, "capacidade esperada observada" if observed else "capacidade esperada não foi necessária/observada", required=False))
    required = [item for item in checks if item.required]
    return InvestigationEvalResult(definition.id, all(item.passed for item in required), tuple(checks), sum(item.passed for item in checks) / len(checks))


def _valid_mermaid(answer: str) -> bool:
    match = re.search(r"```mermaid\s+(flowchart\s+(?:LR|RL|TD|TB|BT)\b[\s\S]*?)```", answer, re.IGNORECASE)
    return bool(match and "-->" in match.group(1))


class JudgeClient(Protocol):
    def judge(self, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class JudgeResult:
    grounded: bool
    relevant: bool
    unsupported_claims: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    reason: str


def run_llm_judge(client: JudgeClient, definition: QuestionDefinition, answer: str, evidence: list[str], sources: set[str], trace: list[ToolTraceRecord]) -> JudgeResult:
    """Level 3 opcional; provider/modelo é fornecido externamente ao CI básico."""
    prompt = json.dumps({"question": definition.prompt, "criteria": definition.semantic_evals, "answer": answer, "evidence": evidence, "sources": sorted(sources), "tool_trace": [{"tool_name": item.tool_name, "status": item.status, "result_metadata": item.result_metadata} for item in trace], "return_schema": {"grounded": "boolean", "relevant": "boolean", "unsupported_claims": "string[]", "missing_evidence": "string[]", "reason": "string"}}, ensure_ascii=False)
    payload = json.loads(client.judge(prompt))
    return JudgeResult(bool(payload.get("grounded")), bool(payload.get("relevant")), tuple(str(item) for item in payload.get("unsupported_claims", [])), tuple(str(item) for item in payload.get("missing_evidence", [])), str(payload.get("reason", "")))
