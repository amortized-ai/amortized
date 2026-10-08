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
    serve = "serve"


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
    data_run_id: str = Field(
        "",
        description=(
            "MLflow run holding the training dataset (generated_data artifact)"
            " — e.g. an uploaded dataset or a split. Alternative to"
            " parent_job_id; the worker downloads it and sets data_path"
        ),
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
    # Embedding classifier (embedding_sft) fields. Declared explicitly (not just
    # via extra="allow") so the validate_training_job MCP tool exposes them as
    # parameters — otherwise the agent can't pass e.g. label_column and the worker
    # falls back to a 'label' column that a {text, category} dataset doesn't have.
    text_column: str | None = Field(
        None, description="embedding_sft: text column name in the dataset (default 'text')"
    )
    label_column: str | None = Field(
        None,
        description=(
            "embedding_sft: label/category column name (default 'label'). Set to"
            " 'category' for the embedding-classifier SDG datasets."
        ),
    )
    loss_type: Literal["batch_all_triplet", "batch_hard_triplet", "mnrl"] | None = (
        Field(
            None,
            description=(
                "embedding_sft contrastive loss: batch_all_triplet, "
                "batch_hard_triplet, or mnrl"
            ),
        )
    )
    batch_sampler: Literal["group_by_label", "no_duplicates", "default"] | None = Field(
        None,
        description=(
            "embedding_sft batch sampler: group_by_label, no_duplicates, or default"
        ),
    )
    warmup_ratio: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="embedding_sft: warmup fraction of total steps (0.0-1.0)",
    )
    seed: int | None = Field(None, ge=0, description="embedding_sft: random seed")
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
    retry_of: str = ""


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
    provider: str = Field(
        "",
        description=(
            "Provider a job uses to reach the model — 'gateway' for gateway-served"
            " models, else the direct provider (e.g. openai)"
        ),
    )
    model_name: str = Field("", description="Underlying model (e.g. openai/gpt-4.1-mini)")


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


# ---------------------------------------------------------------------------
# Eval job models — compare two model endpoints on an eval dataset
# ---------------------------------------------------------------------------


class EvalEndpoint(BaseModel):
    base_url: str = Field(
        ...,
        min_length=1,
        description="OpenAI-compatible API base URL (e.g. http://vllm:8000/v1)",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model name sent as 'model' in chat completion requests",
    )
    api_key: str = Field(
        "",
        description=(
            "Optional bearer token. Never echoed in API responses;"
            " scrubbed from the DB once the job is dispatched."
        ),
    )


class EvalRubricCriterion(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        description="Short criterion name, e.g. 'factual_accuracy' (used as the results key)",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="One sentence telling the judge what to check for this criterion",
    )


class EvalJobConfig(BaseModel):
    model_config = {"extra": "allow"}

    rubric: list[EvalRubricCriterion] = Field(
        default_factory=list,
        description=(
            "Custom judge criteria, designed with the user. The LLM judge scores"
            " the model's response against the reference answer on each criterion"
            " (absolute 0-1), averaged over the dataset."
        ),
    )
    endpoint: EvalEndpoint | None = Field(
        None,
        description=(
            "External OpenAI-compatible endpoint serving the model to evaluate"
            " (e.g. a gateway model). When omitted, set training_job_id or"
            " model_name_or_path and the eval job serves the model itself."
        ),
    )
    training_job_id: str = Field(
        "",
        description=(
            "Training job whose tuned model to evaluate. The eval job serves"
            " the model itself (vLLM inside the job) and stops when scores"
            " are computed. Takes precedence over model_name_or_path."
        ),
    )
    model_name_or_path: str = Field(
        "",
        description=(
            "HF hub model id (e.g. Qwen/Qwen3.5-4B) or local path to evaluate."
            " The eval job serves the model itself."
        ),
    )
    served_model_name: str = Field(
        "",
        description=(
            "Name passed as 'model' in requests. Defaults to the training"
            " job's model_display_name (tuned models) or model_name_or_path."
        ),
    )
    judge: EvalEndpoint | None = Field(
        None,
        description=(
            "Optional LLM-as-judge endpoint for scoring free-form outputs"
            " against the reference answer"
        ),
    )
    max_samples: int = Field(
        0,
        ge=0,
        le=10000,
        description=(
            "Max eval samples to run; 0 (default) evaluates ALL records"
            " in the dataset — set lower only to bound eval time/cost"
        ),
    )
    judge_max_samples: int = Field(
        0,
        ge=0,
        le=10000,
        description=(
            "Max samples for LLM-judge scoring; 0 (default) judges ALL samples"
            " — set lower only to cap judge cost/latency"
        ),
    )
    temperature: float = Field(0.0, description="Sampling temperature for evaluated endpoints")
    eval_data_run_id: str = Field(
        "",
        description=(
            "MLflow run ID holding the eval dataset (alternative to parent_job_id). "
            "Use with datasets uploaded via /api/v1/datasets."
        ),
    )
    topic: str = Field("", description="1-5 word eval topic for tracking")

    # --- Classification / embedding eval (eval_mode="classification") ---
    eval_mode: Literal["generative", "classification"] = Field(
        "generative",
        description=(
            "Evaluation mode. 'generative' (default) serves the model and scores"
            " free-form outputs with an LLM judge against a rubric. 'classification'"
            " evaluates an embedding classifier/router: the held-out labeled dataset"
            " is split into per-class anchors and query examples, and the tuned"
            " embedding model routes each query to its nearest class (accuracy /"
            " macro-F1 / confusion). No judge or rubric is used."
        ),
    )
    class_labels: list[str] | None = Field(
        None,
        description=(
            "Optional human-readable class names indexed by integer label"
            " (classification mode). When omitted, labels are shown as their"
            " integer values."
        ),
    )
    text_column: str = Field(
        "text",
        description="Name of the text column in the eval dataset (classification mode)",
    )
    label_column: str = Field(
        "label",
        description=(
            "Name of the label column in the eval dataset (classification mode)"
        ),
    )
    anchors_per_class: int = Field(
        16,
        ge=1,
        le=512,
        description=(
            "How many labeled examples per class to hold out as routing anchors"
            " (classification mode); the rest of the dataset becomes queries."
        ),
    )
    top_k: int = Field(
        3,
        ge=1,
        le=64,
        description="Per-class score = mean of the top-k anchor similarities (classification mode)",
    )
    tau: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence threshold (classification mode): a query whose best class"
            " score is below tau routes to a fallback/abstain bucket. 0 (default)"
            " never abstains."
        ),
    )

    @model_validator(mode="after")
    def check_class_labels_unique(self) -> "EvalJobConfig":
        """Duplicate display names collapse per-class F1 / confusion keys (two
        integer classes mapping to the same name would silently merge in the
        metrics), so reject them at the boundary."""
        if self.class_labels is not None:
            seen: set[str] = set()
            dupes: set[str] = set()
            for n in self.class_labels:
                if n in seen:
                    dupes.add(n)
                seen.add(n)
            if dupes:
                raise ValueError(
                    "class_labels must be unique; duplicate name(s) would merge"
                    f" per-class metrics: {sorted(dupes)}"
                )
        return self

    @model_validator(mode="after")
    def check_model_source(self) -> "EvalJobConfig":
        """Fail at the boundary (like TrainingJobConfig) — a config with no
        model to evaluate would otherwise validate, create a job, and only
        fail at dispatch."""
        # Legacy configs evaluated two endpoints via endpoint_base/endpoint_tuned
        # (extra fields) — still a valid model source.
        legacy = self.model_extra or {}
        has_legacy_endpoint = any(
            isinstance(legacy.get(k), dict) and legacy.get(k)
            for k in ("endpoint_base", "endpoint_tuned")
        )
        if not (
            self.endpoint
            or has_legacy_endpoint
            or str(self.training_job_id).strip()
            or str(self.model_name_or_path).strip()
        ):
            msg = (
                "eval jobs require a model source: endpoint (external"
                " OpenAI-compatible endpoint), training_job_id (tuned model"
                " to serve in-job), or model_name_or_path (HF id / local"
                " path to serve in-job)"
            )
            raise ValueError(msg)
        return self


class EvalJobRequest(EvalJobConfig):
    parent_job_id: str = Field("", description="Parent SDG/upload job ID holding the eval dataset")


# ---------------------------------------------------------------------------
# Dataset splits — materialized subsets of an existing dataset run
# ---------------------------------------------------------------------------


class DatasetSplitRequest(BaseModel):
    """Split a dataset run into a portion (+ optionally its complement).

    Both outputs are materialized as new dataset runs (full copies of the
    selected rows), so they are immutable, self-contained, and consumable
    by anything that takes a dataset run id (eval_data_run_id, training
    data_run_id).
    """

    count: int = Field(
        0, ge=0, description="Portion size in records (mutually exclusive with fraction)"
    )
    fraction: float = Field(
        0.0, ge=0.0, le=1.0, description="Portion size as a fraction of the dataset"
    )
    strategy: Literal["random", "head", "tail"] = Field(
        "random", description="How rows are selected for the portion"
    )
    seed: int = Field(42, ge=0, description="Random seed (strategy=random) for determinism")
    name: str = Field("", description="Label for the portion dataset (default: derived)")
    create_complement: bool = Field(
        True,
        description=(
            "Also materialize the remaining rows as a second dataset (the train/eval two-way split)"
        ),
    )
    complement_name: str = Field(
        "", description="Label for the complement dataset (default: derived)"
    )

class DatasetMergeRequest(BaseModel):
    """Merge several datasets into one dataset.

    All source datasets are concatenated in order into a single new
    dataset run, usable anywhere a dataset run ID is accepted
    (training data_run_id, eval eval_data_run_id, etc.).
    """

    run_ids: list[str] = Field(
        ...,
        min_length=2,
        description="MLflow run IDs of datasets to merge (at least 2)",
    )
    name: str = Field(
        "",
        description="Name for the merged dataset (default: derived from source names)",
    )
