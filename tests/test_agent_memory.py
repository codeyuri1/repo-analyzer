from repo_agent_chat.agent.memory import ConversationMemory


def test_memory_normaliza_e_limita_mensagens() -> None:
    memory = ConversationMemory(max_messages=2)

    memory.replace(
        [
            {"role": "system", "content": "ignorada"},
            {"role": "user", "content": "primeira"},
            {"role": "assistant", "content": "segunda"},
            {"role": "user", "content": "terceira"},
        ]
    )

    assert memory.context() == [
        {"role": "assistant", "content": "segunda"},
        {"role": "user", "content": "terceira"},
    ]


def test_memory_salva_turno_e_pode_ser_limpa() -> None:
    memory = ConversationMemory(max_messages=2)

    memory.save_turn({"role": "user", "content": "pergunta"}, "resposta")
    assert len(memory.context()) == 2

    memory.clear()
    assert memory.context() == []
