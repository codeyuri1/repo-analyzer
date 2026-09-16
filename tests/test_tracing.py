from repo_agent_chat.tracing import (
    ToolTraceEvent,
    print_tool_trace,
    tool_result_succeeded,
)


def test_tool_result_succeeded_le_status_json() -> None:
    assert tool_result_succeeded('{"ok":true,"result":[]}') is True
    assert tool_result_succeeded('{"ok":false,"error":"falhou"}') is False
    assert tool_result_succeeded("inválido") is False


def test_print_tool_trace_formata_inicio_e_conclusao(capsys) -> None:
    print_tool_trace(ToolTraceEvent("started", "list_files", "{}"))
    print_tool_trace(
        ToolTraceEvent(
            "completed",
            "list_files",
            "{}",
            success=True,
            duration_ms=12.34,
        )
    )

    output = capsys.readouterr().out

    assert "[tool →] list_files {}" in output
    assert "[tool ←] list_files: ok (12.3 ms)" in output
