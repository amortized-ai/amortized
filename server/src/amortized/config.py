"""Application configuration via environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = Path("./data/amortized.db")
    data_dir: Path = Path("./data")
    recipes_dir: Path | None = None
    schemas_dir: Path | None = None
    datasets_dir: Path | None = None

    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (reads AMORTIZED_OPENAI_API_KEY or OPENAI_API_KEY)",
    )
    openai_model: str = "gpt-5-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    model_config = {
        "env_prefix": "AMORTIZED_",
        "extra": "ignore",
    }


def _load_settings() -> Settings:
    """Load settings, falling back OPENAI_API_KEY if AMORTIZED_OPENAI_API_KEY is unset."""
    import os

    s = Settings()
    if not s.openai_api_key:
        s.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    return s


settings = _load_settings()
