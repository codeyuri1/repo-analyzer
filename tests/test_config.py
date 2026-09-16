from repo_agent_chat.config import Settings


def test_settings_possui_valores_padrao() -> None:
    settings = Settings(_env_file=None)

    assert str(settings.ollama_base_url) == "http://localhost:11434/v1"
    assert settings.ollama_model == "qwen2.5-coder:7b"
    assert settings.ollama_embedding_model == "qwen3-embedding:0.6b"
    assert settings.embedding_batch_size == 32
    assert settings.retrieval_top_k == 5
    assert settings.max_tool_rounds == 6
    assert settings.max_history_turns == 6


def test_settings_aceita_variaveis_de_ambiente(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "modelo-de-teste")

    settings = Settings(_env_file=None)

    assert settings.ollama_model == "modelo-de-teste"
