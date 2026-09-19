from pathlib import Path
from unittest.mock import Mock

from repo_agent_chat.agent import ToolAgent
from repo_agent_chat.agent.models import RequestedTool
from repo_agent_chat.app.config import Settings
from repo_agent_chat.app.questions import QUESTION_BY_ID
from repo_agent_chat.evaluation.citations import CitationEvaluator
from repo_agent_chat.evaluation.investigations import evaluate_investigation
from repo_agent_chat.tracing import ToolTraceRecord


def test_catalogo_define_politica_para_todas_as_oito_acoes() -> None:
    assert len(QUESTION_BY_ID) == 8
    assert "analyze_vulnerabilities" not in QUESTION_BY_ID["architecture"].allowed_tools
    assert "analyze_vulnerabilities" in QUESTION_BY_ID["security"].allowed_tools
    assert "generate_mermaid_diagram" in QUESTION_BY_ID["diagram"].allowed_tools


def test_agent_filtra_e_rejeita_tool_fora_da_politica() -> None:
    tools = Mock()
    agent = ToolAgent(Settings(_env_file=None), tools, client=Mock())
    agent._question_definition = QUESTION_BY_ID["architecture"]

    exposed = {item["function"]["name"] for item in agent._available_tool_definitions()}
    result = agent._execute_tool(
        RequestedTool("test", "analyze_vulnerabilities", "{}"), {}
    )

    assert "analyze_vulnerabilities" not in exposed
    assert "não permitida" in result
    tools.execute.assert_not_called()
    assert agent.last_tool_trace[-1].status == "error"


def test_citation_evaluator_exige_faixa_recuperada(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("one\ntwo\n", encoding="utf-8")
    citation = type("Citation", (), {"path": "app.py", "start_line": 2, "end_line": 2})()
    result = CitationEvaluator().evaluate([citation], tmp_path, {"app.py:1-2"}, {"app.py"})

    assert result.passed
    outside = type("Citation", (), {"path": "../secret.py", "start_line": 1, "end_line": 1})()
    assert not CitationEvaluator().evaluate([outside], tmp_path, set(), set()).passed


def test_eval_mostra_tool_esperada_como_diagnostico_nao_regra(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def component():\n    pass\n", encoding="utf-8")
    trace = [ToolTraceRecord("architecture", "read_file", {"path": "app.py"}, "ok", 1.0, {"path": "app.py"})]
    result = evaluate_investigation(QUESTION_BY_ID["architecture"], "Componente `component` em `app.py:1`.", trace, tmp_path, {"app.py"}, {"app.py:1-2"}, ["1: def component():"])

    assert result.passed
    expected = next(check for check in result.checks if check.name == "expected:retrieval")
    assert expected.passed and expected.required is False
