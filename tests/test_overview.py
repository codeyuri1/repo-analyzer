from repo_agent_chat.overview import build_project_overview


def test_constroi_visao_geral_fundamentada() -> None:
    paths = {"README.md", "pyproject.toml", "src/demo/main.py", "src/demo/config.py"}
    documents = {
        "README.md": (
            "1: # Demo\n2: \n3: Assistente para analisar código.\n"
            "4: ```bash\n5: uv run demo\n6: ```"
        ),
        "pyproject.toml": (
            "1: [project]\n2: dependencies = [\n3:   \"gradio>=6\",\n4: ]\n"
            "5: [project.scripts]\n6: demo = \"demo.main:main\""
        ),
        "src/demo/main.py": (
            "1: def main():\n2:     settings = get_settings()\n"
            "3:     agent = build_tool_agent(root, settings)\n4:     run_chat(agent)"
        ),
    }

    answer = build_project_overview(paths, documents)

    assert "Assistente para analisar código" in answer
    assert "Python" in answer
    assert "gradio" in answer
    assert "`demo` aponta para `demo.main:main`" in answer
    assert "`src/demo/main.py:2`" in answer
    assert "`uv run demo`" in answer
    assert "Java" not in answer
