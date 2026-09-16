from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação carregadas do ambiente."""

    ollama_base_url: HttpUrl = HttpUrl("http://localhost:11434/v1")
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    max_tool_rounds: int = Field(default=6, ge=1, le=20)
    max_history_turns: int = Field(default=6, ge=1, le=50)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Retorna uma única instância validada das configurações."""
    return Settings()
