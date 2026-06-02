"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = Path("./data/amortized.db")
    data_dir: Path = Path("./data")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    model_config = {"env_prefix": "AMORTIZED_"}


settings = Settings()
