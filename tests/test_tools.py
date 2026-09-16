import json
from pathlib import Path
from unittest.mock import Mock

from repo_agent_chat.chunking import CodeChunk
from repo_agent_chat.tools import TOOL_DEFINITIONS, RepositoryTools
from repo_agent_chat.vector_store import SearchResult


def decode(result: str):
    return json.loads(result)


def test_semantic_search_serializa_resultados(tmp_path: Path) -> None:
    retriever = Mock()
    retriever.search.return_value = [
        SearchResult(
            CodeChunk("auth.py", 10, 12, "def login(): pass"),
            0.987654321,
        )
    ]
    tools = RepositoryTools(tmp_path, retriever)

    result = decode(
        tools.execute(
            "semantic_search",
            '{"query":"autenticação","top_k":3}',
        )
    )

    assert result["ok"] is True
    assert result["result"][0]["path"] == "auth.py"
    assert result["result"][0]["score"] == 0.987654
    retriever.search.assert_called_once_with("autenticação", 3)


def test_read_file_limita_intervalo_e_bloqueia_saida_da_raiz(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("linha 1\nlinha 2\nlinha 3", encoding="utf-8")
    tools = RepositoryTools(tmp_path, Mock())

    result = decode(
        tools.execute(
            "read_file",
            '{"path":"main.py","start_line":2,"end_line":3}',
        )
    )
    unsafe = decode(
        tools.execute(
            "read_file",
            '{"path":"../outside.py","start_line":1,"end_line":1}',
        )
    )

    assert result["result"]["content"] == "2: linha 2\n3: linha 3"
    assert result["result"]["total_lines"] == 3
    assert result["result"]["has_more"] is False
    assert result["result"]["next_start_line"] is None
    assert unsafe["ok"] is False
    assert "fora da raiz" in unsafe["error"]


def test_read_file_avisa_quando_existe_continuacao(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("linha 1\nlinha 2\nlinha 3", encoding="utf-8")
    tools = RepositoryTools(tmp_path, Mock())

    result = decode(
        tools.execute(
            "read_file",
            '{"path":"main.py","start_line":1,"end_line":2}',
        )
    )

    assert result["result"]["end_line"] == 2
    assert result["result"]["total_lines"] == 3
    assert result["result"]["has_more"] is True
    assert result["result"]["next_start_line"] == 3


def test_list_files_aplica_prefixo_e_ignora_arquivos_nao_suportados(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (src / "image.png").write_bytes(b"PNG")
    tools = RepositoryTools(tmp_path, Mock())

    result = decode(tools.execute("list_files", '{"prefix":"src/"}'))

    assert result == {"ok": True, "result": ["src/main.py"]}


def test_list_files_aceita_argumentos_vazios(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    tools = RepositoryTools(tmp_path, Mock())

    result = decode(tools.execute("list_files", "{}"))

    assert result == {"ok": True, "result": ["main.py"]}


def test_execute_devolve_erro_para_json_ou_tool_invalidos(tmp_path: Path) -> None:
    tools = RepositoryTools(tmp_path, Mock())

    invalid_json = decode(tools.execute("list_files", "não é json"))
    unknown_tool = decode(tools.execute("delete_repository", "{}"))

    assert invalid_json["ok"] is False
    assert unknown_tool == {
        "ok": False,
        "error": "Tool desconhecida: delete_repository",
    }


def test_tools_expoem_seguranca_e_mermaid() -> None:
    names = {definition["function"]["name"] for definition in TOOL_DEFINITIONS}

    assert "analyze_vulnerabilities" in names
    assert "generate_mermaid_diagram" in names


def test_execute_tools_de_seguranca_e_mermaid(tmp_path: Path) -> None:
    (tmp_path / "unsafe.py").write_text("eval(user_input)", encoding="utf-8")
    tools = RepositoryTools(tmp_path, Mock())

    security = decode(tools.execute("analyze_vulnerabilities", "{}"))
    diagram = decode(tools.execute("generate_mermaid_diagram", "{}"))

    assert security["result"]["findings"][0]["rule"] == "dynamic-code-execution"
    assert diagram["result"]["diagram"].startswith("flowchart LR")
