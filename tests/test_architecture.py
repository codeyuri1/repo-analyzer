import ast
from pathlib import Path


def test_agent_nao_depende_da_interface() -> None:
    agent_root = Path("src/repo_agent_chat/agent")
    forbidden = {"repo_agent_chat.app.ui", "repo_agent_chat.app.session"}

    for path in agent_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert imported.isdisjoint(forbidden), path
