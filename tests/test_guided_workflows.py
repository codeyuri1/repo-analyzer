import json

from repo_agent_chat.guided_workflows import (
    format_mermaid_diagram,
    format_security_analysis,
)


def test_formata_analise_de_seguranca_com_limitacoes() -> None:
    result = json.dumps(
        {
            "ok": True,
            "result": {
                "findings": [
                    {
                        "rule": "weak-hash",
                        "severity": "medium",
                        "path": "src/auth.py",
                        "line": 10,
                        "message": "Hash fraco.",
                    }
                ]
            },
        }
    )

    answer = format_security_analysis(result)

    assert "`src/auth.py:10`" in answer
    assert "análise estática" in answer
    assert "falso positivo" in answer
    assert "revisão manual" in answer
    assert "não garante" in answer


def test_formata_mermaid_sem_alterar_diagrama() -> None:
    diagram = "flowchart LR\n    n0 --> n1"
    result = json.dumps(
        {"ok": True, "result": {"diagram": diagram, "nodes": 2, "edges": 1}}
    )

    answer = format_mermaid_diagram(result)

    assert f"```mermaid\n{diagram}\n```" in answer
