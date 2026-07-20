"""Application configuration via environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = Path("./data/amortized.db")
    data_dir: Path = Path("./data")
    recipes_dir: Path | None = None

    api_key: str = Field(default="", description="API key for auth (empty = no auth)")
    cors_origins: str = Field(default="*", description="Comma-separated allowed CORS origins")

    forward_env: list[str] = Field(
        default_factory=list, description="Env var names to forward to job containers"
    )

    compute_backend: str = Field("local", description="Compute backend: local, ssh, kubernetes")
    compute_namespace: str = Field("amortized-jobs", description="K8s namespace for jobs")
    image_registry: str = Field("ghcr.io/amortized-ai", description="Container image registry")
    image_pull_policy: str = Field("Always", description="K8s image pull policy for job containers")
    mlflow_tracking_uri: str = Field("", description="MLflow tracking URI (empty = disabled)")
    storage_bucket: str = Field("", description="S3 bucket name for artifact storage")

    external_url: str = Field("", description="Externally reachable server URL")
    gateway_url: str = Field("", description="MLflow AI Gateway URL for LLM routing")

    default_backend: str = Field(
        "", description="Default compute backend (falls back to compute_backend if empty)"
    )

    @property
    def resolved_default_backend(self) -> str:
        return self.default_backend or self.compute_backend or "local"

    model_config = {
        "env_prefix": "AMORTIZED_",
        "extra": "ignore",
    }


settings = Settings()
