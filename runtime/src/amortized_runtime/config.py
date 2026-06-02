"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = Path("./data/amortized.db")
    data_dir: Path = Path("./data")

    claude_command: str = "claude"
    claude_model: str = "sonnet"
    claude_max_turns: int = 10

    model_config = {"env_prefix": "AMORTIZED_"}


settings = Settings()
