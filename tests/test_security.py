from repo_agent_chat.repository import SourceFile
from repo_agent_chat.security import analyze_source_files


def test_analyze_source_files_detecta_sem_expor_segredo() -> None:
    files = [
        SourceFile(
            path="config.py",
            content='API_KEY = "valor-super-secreto"\nresult = eval(user_input)',
        )
    ]

    findings = analyze_source_files(files)

    assert [finding["rule"] for finding in findings] == [
        "hardcoded-secret",
        "dynamic-code-execution",
    ]
    assert findings[0]["path"] == "config.py"
    assert findings[0]["line"] == 1
    assert "valor-super-secreto" not in str(findings)


def test_analyze_source_files_respeita_limite() -> None:
    files = [SourceFile("unsafe.py", "eval(a)\neval(b)\neval(c)")]

    findings = analyze_source_files(files, max_findings=2)

    assert len(findings) == 2
