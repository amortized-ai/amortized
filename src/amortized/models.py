"""Pydantic models for the Amortized v1 API."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobType(StrEnum):
    training = "training"
    sdg = "sdg"
    upload = "upload"
    eval = "eval"


class JobStatus(StrEnum):
    queued = "queued"
    provisioning = "provisioning"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: list[dict[str, Any]] = Field(default_factory=list)


class ComputeSpec(BaseModel):
    backend: str = Field("local", description="Compute backend name")
    gpus: int = Field(0, ge=0, description="Number of GPUs requested")
    gpu_type: str | None = Field(None, description="GPU type (e.g. 'A100', 'H100')")


class TrainingJobConfig(BaseModel):
    model_config = {"extra": "allow"}

    algorithm: str = Field(
        ..., description="Training algorithm (sft, lora_sft, osft, dpo, grpo, lora_grpo, kto, gkd)"
    )
    model_name_or_path: str = Field(..., description="HuggingFace model ID or local path")
    data_path: str | None = Field(
        None, description="Path to training data (resolved from parent if chaining)"
    )
    output_dir: str | None = Field(None, description="Output directory")
    learning_rate: float | None = Field(None, description="Learning rate")
    num_train_epochs: int | None = Field(None, ge=1, description="Number of training epochs")
    per_device_train_batch_size: int | None = Field(None, ge=1, description="Batch size per GPU")
    max_length: int | None = Field(None, ge=1, description="Maximum sequence length")
    bf16: bool | None = Field(None, description="Use bfloat16 mixed precision")
    gradient_checkpointing: bool | None = Field(None, description="Enable gradient checkpointing")
    gradient_accumulation_steps: int | None = Field(None, ge=1)
    load_in_4bit: bool | None = Field(None, description="Use QLoRA 4-bit quantization")
    use_peft: bool | None = Field(None, description="Enable LoRA via PEFT")
    lora_r: int | None = Field(None, ge=1, description="LoRA rank")
    lora_alpha: int | None = Field(None, ge=1, description="LoRA alpha")
    lora_dropout: float | None = Field(None, description="LoRA dropout rate")
    unfreeze_rank_ratio: float | None = Field(
        None, description="OSFT: fraction of weights trainable (default 0.2)"
    )
    topic: str = Field(
        "",
        description="1-5 word model topic for tracking (e.g. 'support ticket classification')",
    )

    @model_validator(mode="after")
    def check_osft_requires_urr(self) -> "TrainingJobConfig":
        if self.algorithm == "osft" and self.unfreeze_rank_ratio is None:
            msg = "unfreeze_rank_ratio is required for OSFT (e.g. 0.2)"
            raise ValueError(msg)
        return self


class TrainingJobRequest(TrainingJobConfig):
    parent_job_id: str = Field("", description="Parent SDG job ID for chaining (SDG -> Training)")


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType
    status: JobStatus = JobStatus.queued
    config: dict[str, Any] = Field(default_factory=dict)
    recipe: str = ""
    parent_job_id: str = ""
    user_id: str = ""
    k8s_job_name: str = ""
    k8s_namespace: str = ""
    mlflow_run_id: str = ""
    mlflow_experiment: str = ""
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None


class ValidatedJobConfig(BaseModel):
    valid: bool = True
    job_type: JobType
    config: dict[str, Any] = Field(default_factory=dict)
    parent_job_id: str = ""
    recipe: str = ""
    warnings: list[str] = Field(default_factory=list)


class RecipeSummary(BaseModel):
    name: str
    description: str = ""
    type: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    gpu: dict[str, Any] = Field(default_factory=dict)
    db: str = "ok"


class GatewayModel(BaseModel):
    name: str = Field(..., description="Endpoint name (use as 'model' in job config)")
    provider: str = Field("", description="Model provider (e.g. openai, anthropic)")
    model_name: str = Field("", description="Underlying model (e.g. gpt-4.1-mini)")


class ModelsResponse(BaseModel):
    models: list[GatewayModel] = Field(default_factory=list)
    gateway_url: str = Field("", description="Gateway URL these models are served from")


class OutputFormat(StrEnum):
    md = "md"
    text = "text"
    json = "json"
    html = "html"


class ChunkerType(StrEnum):
    sentence = "sentence"
    token = "token"
    recursive = "recursive"


class ConvertOptions(BaseModel):
    chunker_type: ChunkerType = Field(ChunkerType.sentence, description="Chunker type")
    chunk_size: int = Field(2048, ge=64, le=8192, description="Max tokens per chunk")
    chunk_overlap: int = Field(200, ge=0, description="Token overlap between chunks")


class ConvertUrlRequest(BaseModel):
    url: str = Field(..., description="URL of document to convert")
    options: ConvertOptions = Field(default_factory=ConvertOptions)


class _DocumentBase(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    mlflow_run_id: str | None = Field(None, description="MLflow run ID for artifact tracking")
    filename: str = Field("", description="Original filename")
    format: OutputFormat = Field(OutputFormat.md, description="Output format used")


class DocumentResult(_DocumentBase):
    content: str = Field("", description="Parsed document content")
    chunk_count: int = Field(0, ge=0, description="Number of chunks created")
    processing_time: float = Field(0.0, ge=0, description="Processing time in seconds")
    status: str = Field("success", description="Conversion status")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues")


class DocumentUploadAccepted(BaseModel):
    job_id: str = Field(..., description="Job ID to poll for status")
    filename: str = Field("", description="Original filename")
    status: str = Field("processing", description="Processing status")


class DocumentChunk(BaseModel):
    chunk_index: int = Field(..., description="Chunk position in document")
    text: str = Field("", description="Chunk content")
    num_tokens: int | None = Field(None, description="Token count")
    headings: list[str] = Field(default_factory=list, description="Section headings")
    page_numbers: list[int] = Field(default_factory=list, description="Source pages")


class DocumentChunks(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field("", description="Original filename")
    chunks: list[DocumentChunk] = Field(default_factory=list)


class DocumentSummary(_DocumentBase):
    created_at: str | None = Field(None, description="When the document was processed")
    content_available: bool = Field(True, description="Whether the parsed content artifact exists")


class ConfigResponse(BaseModel):
    version: str = "1.0.0"
    default_compute_backend: str = ""
    compute_namespace: str = ""
    mlflow_tracking_uri: str = ""
    mlflow_gateway_uri: str = ""
    image_registry: str = ""
    available_backends: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SDG job models — uses Data Designer's own Pydantic config types
# ---------------------------------------------------------------------------

from data_designer.config.column_types import ColumnConfigT as SDGColumn  # noqa: E402
from data_designer.config.mcp import ToolConfig as DDToolConfig  # noqa: E402
from data_designer.config.models import ModelConfig as DDModelConfig  # noqa: E402
from data_designer.config.processor_types import ProcessorConfigT as SDGProcessor  # noqa: E402
from data_designer.config.sampler_constraints import ColumnConstraintInputT  # noqa: E402


class SDGJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[SDGColumn] = Field(
        ...,
        description=(
            "Generation pipeline columns. Evaluated in order; "
            "later columns reference earlier ones via {{ column_name }}"
        ),
    )
    model_configs: list[DDModelConfig] = Field(
        default_factory=list,
        description=(
            "LLM model configurations. Required when using llm-text/code/judge/structured columns"
        ),
    )
    processors: list[SDGProcessor] = Field(
        default_factory=list,
        description=(
            "Output processors. Use schema_transform for SFT format, "
            "drop_columns to remove intermediate columns"
        ),
    )
    seed_config: dict[str, Any] | None = Field(
        None,
        description=(
            "Seed data configuration (auto-configured when document_ids is provided). "
            "source.seed_type: local, hf, directory, file_contents, agent_rollout"
        ),
    )
    constraints: list[ColumnConstraintInputT] = Field(
        default_factory=list,
        description="Column value constraints (inequality checks)",
    )
    tool_configs: list[DDToolConfig] = Field(
        default_factory=list,
        description="MCP tool configurations for tool-use columns",
    )

    num_records: int = Field(100, ge=1, description="Number of samples to generate")
    document_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Document IDs (from the Documents page) to use as seed data. "
            "Chunks are fetched from MLflow."
        ),
    )
    topic: str = Field(
        "",
        description=(
            "1-5 word dataset topic for MLflow tracking (e.g. 'OpenShift troubleshooting')"
        ),
    )
    parent_job_id: str = Field(
        "",
        description="Parent job ID for lineage chaining (SDG -> Training)",
    )
    mode: Literal["create", "preview"] = Field(
        "create",
        description=(
            "'create' for full generation, 'preview' to generate "
            "~10 samples and verify config before committing"
        ),
    )

    @model_validator(mode="after")
    def check_model_aliases(self) -> "SDGJobRequest":
        aliases_needed: list[str] = []
        for i, col in enumerate(self.columns):
            alias = getattr(col, "model_alias", None)
            if alias is not None:
                if not alias:
                    msg = f"columns[{i}].model_alias: must not be empty"
                    raise ValueError(msg)
                aliases_needed.append(alias)

        if not aliases_needed:
            return self

        if not self.model_configs:
            msg = (
                "model_configs is required when columns use "
                "LLM generation (llm-text, llm-code, etc.)"
            )
            raise ValueError(msg)

        defined = {mc.alias for mc in self.model_configs}
        missing = [a for a in aliases_needed if a not in defined]
        if missing:
            msg = (
                f"model_alias {missing} not found in model_configs "
                f"(available: {sorted(defined) or 'none'})"
            )
            raise ValueError(msg)

        return self
