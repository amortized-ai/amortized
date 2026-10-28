"""Pydantic models for the Amortized v1 API."""

import json
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class _ConfigJsonMixin(BaseModel):
    """Single-string escape hatch for weak models whose tool-call encoder corrupts large
    multi-argument calls.

    Observed with GLM-5.3-Flash: its native ``<arg_key>/<arg_value>`` tool-call format is
    mis-parsed for configs ≳5KB — the nested arguments get merged into the first field, so
    ``validate_sdg_job`` never parses and the agent spins (session ses_ea63dced…, 27 failed
    attempts). A SINGLE string argument round-trips intact (the ``write`` tool proves this),
    so letting the agent pass the ENTIRE config as one ``config_json`` string sidesteps the
    bug for any model. When ``config_json`` is set it FULLY REPLACES the flat arguments.
    """

    config_json: str | None = Field(
        default=None,
        description=(
            "The entire job config as ONE JSON-string argument. PREFER THIS for any "
            "non-trivial config: put every field inside this single string instead of "
            "passing columns/model_configs/etc. as separate arguments. When set, all other "
            "arguments are ignored. Avoids a tool-call corruption some models hit on large "
            "multi-field calls."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_config_json(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw = data.get("config_json")
            if raw not in (None, ""):
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError) as e:
                    msg = f"config_json must be a valid JSON object string: {e}"
                    raise ValueError(msg) from e
                if not isinstance(parsed, dict):
                    msg = "config_json must decode to a JSON object (the config fields)"
                    raise ValueError(msg)
                parsed.pop("config_json", None)
                # config_json fully replaces the flat args; anything also passed alongside it
                # is dropped. Keep that behavior (the escape hatch must absorb a weak model's
                # smeared extra keys, not reject them), but make the drop auditable by naming
                # exactly which keys were ignored and the values discarded.
                ignored = {
                    k: v for k, v in data.items() if k != "config_json" and v not in (None, "")
                }
                if ignored:
                    logger.warning(
                        "config_json is set; ignoring %d flat argument(s) in favor of the "
                        "config_json contents: %s",
                        len(ignored),
                        ", ".join(f"{k}={v!r}" for k, v in sorted(ignored.items())),
                    )
                return parsed
        return data


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
    model_name_or_path: str = Field(
        "",
        description="HuggingFace model ID or local path.",
    )
    training_job_id: str = Field(
        "",
        description=(
            "Training job ID whose finetuned model to resume from."
            " The model artifacts are downloaded from MLflow before"
            " training starts. Mutually exclusive with model_name_or_path"
            " — set one or the other, not both."
        ),
    )
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
    nproc_per_node: int = Field(
        1,
        ge=1,
        description=(
            "GPUs for this single training job (single-node multi-GPU). Training Hub"
            " launches torchrun across this many GPUs, and the pod requests that many"
            " nvidia.com/gpu — so it needs ONE node with >= nproc_per_node GPUs. Default 1."
            " Must be <= the per-user GPU budget. IGNORED for embedding_sft (single-device,"
            " no torchrun) — use train_on_gpu instead."
        ),
    )
    train_on_gpu: bool = Field(
        False,
        description=(
            "embedding_sft ONLY: train the embedding classifier on a GPU (1) instead of the"
            " CPU default (0). embedding_sft trains in-process on a single device, so this is a"
            " boolean, not a GPU count. CPU is fine for small/quick runs; a GPU is ~100x faster"
            " (benchmark: all-MiniLM ~24 samples/s CPU vs ~2800/s on an H100), which matters for"
            " many epochs or large datasets. Ignored for all other algorithms (they use"
            " nproc_per_node)."
        ),
    )
    topic: str = Field(
        "",
        description="1-5 word model topic for tracking (e.g. 'support ticket classification')",
    )

    @model_validator(mode="after")
    def check_model_source(self) -> "TrainingJobConfig":
        if not self.model_name_or_path and not self.training_job_id:
            msg = (
                "either model_name_or_path (HuggingFace model ID) or"
                " training_job_id (resume from a finetuned model) is required"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_osft_requires_urr(self) -> "TrainingJobConfig":
        if self.algorithm == "osft" and self.unfreeze_rank_ratio is None:
            msg = "unfreeze_rank_ratio is required for OSFT (e.g. 0.2)"
            raise ValueError(msg)
        return self


class TrainingJobRequest(TrainingJobConfig, _ConfigJsonMixin):
    parent_job_id: str = Field("", description="Parent SDG job ID for chaining (SDG -> Training)")

    # Override the base (where algorithm is required) so a config_json-only call passes
    # MCP input validation. The real requirement is a SEMANTIC one enforced after
    # config_json expansion, in _validate_training_data — which catches a missing key,
    # an empty value, AND a config_json string that forgot algorithm (none of which a
    # request-layer key-guard could cover), with a clean 422 instead of a dispatch failure.
    algorithm: str = Field(
        "",
        description=(
            "Training algorithm (sft, lora_sft, osft, dpo, grpo, lora_grpo, kto, gkd). "
            "Required unless the whole config is passed via config_json."
        ),
    )


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


class PromptView(BaseModel):
    """One reviewable system prompt in an SDG recipe, resolved server-side so the
    confirmation card can render EVERY prompt the recipe carries (input-generator,
    assessor, and any future ones) rather than only the assessor. A recipe commonly
    has more than one — e.g. a ticket-generation prompt plus the assessor prompt."""

    # "assessor" = the system prompt the teacher follows that ships in the SFT training
    # data; "input" = an input-generator column's own system prompt.
    role: str
    # Human-readable card heading (e.g. "Assessor system prompt", "Ticket prompt").
    label: str
    # The recipe column the prompt lives on, so an approved preview can be bound back
    # to the exact config slot.
    column: str
    text: str


class ValidatedJobConfig(BaseModel):
    valid: bool = True
    job_type: JobType
    config: dict[str, Any] = Field(default_factory=dict)
    parent_job_id: str = ""
    recipe: str = ""
    warnings: list[str] = Field(default_factory=list)
    # Backend-resolved display facts the confirmation card shows verbatim, so the
    # user always sees them even when the model omits them from its prose. Kept out
    # of `config` so they are never re-submitted to the create endpoint.
    data_record_count: int | None = None
    # The assessor/system prompt the teacher follows (and that ships in the SFT
    # training data), resolved server-side from the SDG config so the card renders
    # the real prompt instead of the client guessing it from config internals.
    # None when the config has no confidently-identifiable assessor prompt. Retained
    # for back-compat; it mirrors the "assessor"-role entry in `prompts`.
    assessor_prompt: str | None = None
    # Every reviewable system prompt in the recipe (input generators + assessor + any
    # future ones), in generation order, so the card can render all of them instead of
    # just the assessor. Empty when none are confidently identifiable.
    prompts: list[PromptView] = Field(default_factory=list)
    # Tunable hyperparameters the config left unset, so the trainer's own default applies —
    # rendered as "trainer default" on the confirmation card. Kept out of `config` (never
    # re-submitted to the create endpoint). Empty for non-training validate paths.
    trainer_default_fields: list[str] = Field(default_factory=list)


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


class SDGJobRequest(_ConfigJsonMixin):
    model_config = ConfigDict(extra="forbid")

    columns: list[SDGColumn] = Field(
        default_factory=list,
        description=(
            "Generation pipeline columns. Evaluated in order; "
            "later columns reference earlier ones via {{ column_name }}. "
            "Required (non-empty) unless the whole config is passed via config_json."
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

    # columns is schema-optional (so a config_json-only call passes MCP input validation)
    # and an EMPTY list is deliberately allowed (DD permits it — see test_empty_columns_list).
    # A wholly missing key defaults to the same empty list, so there is nothing left to
    # enforce at the request layer: no key-guard, unlike training's algorithm.

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
    nproc_per_node: int = Field(
        1,
        ge=1,
        description=(
            "GPUs for an in-job-served model (training_job_id / model_name_or_path): vLLM"
            " serves with --tensor-parallel-size set to this, and the pod requests that many"
            " nvidia.com/gpu — so it needs ONE node with >= nproc_per_node GPUs. Default 1."
            " Ignored for external-endpoint and classification evals. Must be <= the per-user"
            " GPU budget."
        ),
    )
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


class EvalJobRequest(EvalJobConfig, _ConfigJsonMixin):
    parent_job_id: str = Field("", description="Parent SDG/upload job ID holding the eval dataset")
    acknowledge_warnings: str = Field(
        "",
        description=(
            "Acknowledgment TOKEN for validation warnings. Leave empty on the first"
            " validate: any warning (leakage, rubric drift, recipe mismatch) returns a 422"
            " whose message contains a token. Relay the warning(s) to the user; only if"
            " they choose to proceed, re-validate with this set to that exact token. The"
            " token is tied to the specific warnings, so changing the config yields a new"
            " warning set and a new token - forcing a fresh decision, never suppressing it."
        ),
    )


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
        description="MLflow run IDs of datasets to merge (at least 2, no duplicates)",
    )

    @model_validator(mode="after")
    def _no_duplicate_run_ids(self) -> Self:
        if len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("run_ids must not contain duplicates")
        return self

    name: str = Field(
        "",
        description="Name for the merged dataset (default: derived from source names)",
    )
