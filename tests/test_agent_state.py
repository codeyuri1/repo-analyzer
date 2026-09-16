from repo_agent_chat.agent.evidence import AgentTurnState


def test_estado_de_rodada_inicia_isolado() -> None:
    first = AgentTurnState()
    second = AgentTurnState()

    first.tool_calls.append("read_file")
    first.evidence_paths.add("main.py")

    assert second.tool_calls == []
    assert second.evidence_paths == set()
