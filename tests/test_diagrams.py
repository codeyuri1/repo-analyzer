from repo_agent_chat.repository import SourceFile
from repo_agent_chat.tools.diagrams import generate_dependency_diagram


def test_generate_dependency_diagram_liga_imports_python() -> None:
    files = [
        SourceFile("src/app.py", "from src.service import PaymentService"),
        SourceFile("src/service.py", "class PaymentService: pass"),
    ]

    result = generate_dependency_diagram(files)

    assert result["diagram"].startswith("flowchart LR")
    assert 'n0["src/app.py"]' in result["diagram"]
    assert "n0 --> n1" in result["diagram"]
    assert result["edges"] == 1


def test_generate_dependency_diagram_limita_nos() -> None:
    files = [SourceFile(f"file_{index}.py", "") for index in range(3)]

    result = generate_dependency_diagram(files, max_nodes=2)

    assert result["nodes"] == 2
    assert result["truncated"] is True
