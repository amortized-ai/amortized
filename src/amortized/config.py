"""Application configuration via environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = Field(
        "postgresql://amortized:amortized@localhost:5432/amortized",
        description="PostgreSQL connection string",
    )
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
    mlflow_tracking_token_file: str = Field(
        "",
        description="Path to a bearer-token file for MLflow auth (e.g. a K8s service-account "
        "token). Empty = no bearer auth (self-hosted MLflow).",
    )
    mlflow_workspace: str = Field(
        "",
        description="X-MLFLOW-WORKSPACE header for RHOAI MLflow workspaces. Empty with a token "
        "file set auto-reads the pod's K8s namespace.",
    )
    mlflow_ca_bundle: str = Field(
        "", description="Path to a CA bundle used to verify the MLflow server TLS certificate."
    )
    mlflow_tracking_insecure_tls: bool = Field(
        False, description="Skip TLS verification for the MLflow server (not recommended)."
    )

    agent_upstream_url: str = Field(
        "http://opencode:4096", description="OpenCode upstream URL for agent session proxy"
    )
    agent_upstream_client_cert: str = Field(
        "",
        description="Path to a client cert (PEM) for mTLS to the agent upstream, e.g. the "
        "OpenShell gateway. Empty = no client cert (plain opencode Service).",
    )
    agent_upstream_client_key: str = Field(
        "", description="Path to the client key (PEM) for agent_upstream_client_cert."
    )
    agent_upstream_ca_bundle: str = Field(
        "", description="Path to a CA bundle to verify the agent upstream's TLS certificate."
    )
    agent_upstream_insecure_tls: bool = Field(
        False, description="Skip TLS verification for the agent upstream (not recommended)."
    )
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
