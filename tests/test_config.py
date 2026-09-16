import pytest
from pydantic import ValidationError

from repo_agent_chat.app.config import Settings


def test_settings_possui_valores_padrao() -> None:
    settings = Settings(_env_file=None)

    assert str(settings.ollama_base_url) == "http://localhost:11434/v1"
    assert settings.ollama_model == "qwen2.5-coder:7b"
    assert settings.ollama_embedding_model == "qwen3-embedding:0.6b"
    assert settings.embedding_batch_size == 32
    assert settings.retrieval_top_k == 5
    assert settings.max_tool_rounds == 6
    assert settings.max_history_turns == 6
    assert settings.ai_provider == "ollama"
    assert settings.chat_model == "qwen2.5-coder:7b"
    assert settings.embedding_model == "qwen3-embedding:0.6b"
    assert settings.api_key == "ollama"


def test_settings_aceita_variaveis_de_ambiente(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "modelo-de-teste")

    settings = Settings(_env_file=None)

    assert settings.ollama_model == "modelo-de-teste"


def test_settings_configura_openai() -> None:
    settings = Settings(
        _env_file=None,
        ai_provider="openai",
        openai_api_key="segredo-de-teste",
    )

    assert settings.chat_model == "gpt-4.1-mini"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.api_base_url == "https://api.openai.com/v1"
    assert settings.api_key == "segredo-de-teste"


def test_settings_exige_chave_para_openai() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, ai_provider="openai")
