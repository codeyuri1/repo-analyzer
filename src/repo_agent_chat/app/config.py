from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação carregadas do ambiente."""

    ai_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: HttpUrl = HttpUrl("http://localhost:11434/v1")
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    openai_base_url: HttpUrl = HttpUrl("https://api.openai.com/v1")
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    max_index_chunks: int = Field(default=1500, ge=100, le=10_000)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    max_tool_rounds: int = Field(default=6, ge=1, le=20)
    max_history_turns: int = Field(default=6, ge=1, le=50)
    ollama_timeout_seconds: float = Field(default=300.0, ge=10.0, le=900.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "Settings":
        if self.ai_provider == "openai" and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY é obrigatória quando AI_PROVIDER=openai.")
        return self

    @property
    def api_base_url(self) -> str:
        url = self.openai_base_url if self.ai_provider == "openai" else self.ollama_base_url
        return str(url)

    @property
    def api_key(self) -> str:
        if self.ai_provider == "openai":
            assert self.openai_api_key is not None
            return self.openai_api_key.get_secret_value()
        return "ollama"

    @property
    def chat_model(self) -> str:
        return self.openai_model if self.ai_provider == "openai" else self.ollama_model

    @property
    def embedding_model(self) -> str:
        if self.ai_provider == "openai":
            return self.openai_embedding_model
        return self.ollama_embedding_model


@lru_cache
def get_settings() -> Settings:
    """Retorna uma única instância validada das configurações."""
    return Settings()
