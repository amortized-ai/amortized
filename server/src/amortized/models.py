"""Pydantic models for the Amortized runtime API."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobType(StrEnum):
    """Type of job."""

    training = "training"
    sdg = "sdg"
    inference = "inference"
    eval = "eval"


class JobStatus(StrEnum):
    """Status of a job."""

    validating = "validating"
    queued = "queued"
    provisioning = "provisioning"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

    # Backward-compat aliases
    pending = "queued"
    completed = "succeeded"


# --- Error models ---


class ErrorResponse(BaseModel):
    """Structured error envelope returned by all error responses."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: list[dict[str, Any]] = Field(
        default_factory=list, description="Additional error details"
    )


# --- Request models ---


class TrainingJobConfig(BaseModel):
    """Configuration for a training job."""

    algorithm: str = Field(..., description="Training algorithm (lora_sft, full_sft, dpo, grpo)")
    model_path: str = Field(..., description="HuggingFace model path or local path")
    data_path: str = Field(..., description="Path to training data (JSONL)")
    ckpt_output_dir: str | None = Field(None, description="Directory for checkpoints and outputs")
    learning_rate: float | None = Field(None, description="Learning rate (default: 2e-4)")
    num_epochs: int | None = Field(None, ge=1, description="Number of training epochs")
    lora_r: int | None = Field(None, ge=1, description="LoRA rank")
    lora_alpha: int | None = Field(None, ge=1, description="LoRA alpha")
    load_in_4bit: bool | None = Field(None, description="Enable QLoRA 4-bit quantization")
    micro_batch_size: int | None = Field(None, ge=1, description="Micro batch size")
    batch_size: int | None = Field(None, ge=1, description="Training batch size")
    max_seq_len: int | None = Field(None, ge=1, description="Maximum sequence length")
    gradient_checkpointing: bool | None = Field(None, description="Enable gradient checkpointing")
    model_max_length: int | None = Field(None, ge=1, description="Maximum model sequence length")


class SynthJobConfig(BaseModel):
    """Configuration for a synthetic data generation job."""

    model: str = Field(..., description="Teacher model (LiteLLM format)")
    api_base: str | None = Field(None, description="Model API base URL")
    api_key: str | None = Field(None, description="Model API key")
    num_samples: int = Field(100, ge=1, description="Number of samples to generate")
    max_concurrency: int = Field(16, ge=1, description="Max concurrent LLM requests")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1, description="Max tokens per LLM response")
    top_p: float | None = Field(None, ge=0, le=1, description="Nucleus sampling parameter")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    num_retries: int | None = Field(None, ge=0, description="Number of LLM retries on failure")
    input_data: list[dict[str, Any]] | None = Field(
        None, description="Input dataset sources (JSONL/CSV/HuggingFace paths)"
    )
    input_documents: list[dict[str, Any]] | None = Field(
        None, description="Input document sources (PDF/DOCX/TXT files)"
    )
    strategy_params: dict[str, Any] | None = Field(
        None, description="Raw asynth GeneralSynthesisParams (advanced)"
    )


class ComputeSpec(BaseModel):
    """Specifies which compute backend to use for a job."""

    backend: str = Field("local", description="Compute backend name (e.g. 'local', 'ssh')")
    gpus: int = Field(0, ge=0, description="Number of GPUs requested")
    gpu_type: str | None = Field(None, description="GPU type (e.g. 'A100', 'H100')")


class JobRequest(BaseModel):
    """Universal job submission request."""

    type: str = Field(..., description="Job type (e.g. 'training', 'sdg')")
    config: dict[str, Any] = Field(..., description="Job-type-specific configuration")
    compute: ComputeSpec = Field(default_factory=ComputeSpec, description="Compute backend spec")
    metadata: dict[str, Any] = Field(default_factory=dict, description="User-defined metadata")
    dry_run: bool = Field(True, description="Validate and preview without creating the job")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Artifact references this job depends on (e.g. ['artifact:job-abc/model'])",
    )


# --- Response models ---


class Job(BaseModel):
    """Job record returned by the API."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType
    status: JobStatus = JobStatus.queued
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    output_dir: str | None = None


class ArtifactRequest(BaseModel):
    """Request to register an artifact."""

    name: str = Field(..., description="Human-readable artifact name")
    artifact_type: str = Field(..., description="Artifact type (e.g. adapter_weights, dataset)")
    location: str = Field(..., description="File path or URI where the artifact lives")
    metadata: dict[str, Any] = Field(default_factory=dict, description="User-defined metadata")
    producer_job: str | None = Field(None, description="Job ID that produced this artifact")


class Artifact(BaseModel):
    """An output artifact from a job."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str | None = None
    artifact_type: str
    path: str = ""
    size: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    name: str = ""
    location: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    producer_job: str | None = None


class MemoryEstimateRequest(BaseModel):
    """Request for GPU memory estimation."""

    model_path: str = Field(..., description="HuggingFace model path")
    lora_r: int = Field(16, ge=1, description="LoRA rank")
    batch_size: int = Field(2, ge=1, description="Batch size")
    max_seq_len: int = Field(2048, ge=1, description="Maximum sequence length")
    load_in_4bit: bool = Field(False, description="Use QLoRA 4-bit quantization")


class MemoryEstimateResponse(BaseModel):
    """Response with estimated GPU VRAM requirements."""

    model_path: str
    lora_r: int
    batch_size: int
    max_seq_len: int
    estimated_vram_gb: float
    load_in_4bit: bool


class PipelineInfo(BaseModel):
    """Information about an available synthesis pipeline."""

    name: str
    description: str
    supports_multi_turn: bool
    config_schema: dict[str, Any] = Field(default_factory=dict)


class SynthCapabilities(BaseModel):
    """asynth synthesis engine capabilities."""

    available: bool = Field(..., description="Whether asynth is installed")
    version: str = Field("", description="asynth version")
    strategies: list[str] = Field(
        default_factory=list, description="Available synthesis strategies"
    )
    attribute_types: list[str] = Field(
        default_factory=list, description="Supported attribute types"
    )
    data_sources: list[str] = Field(default_factory=list, description="Supported data source types")
    environment_types: list[str] = Field(
        default_factory=list, description="Supported environment types"
    )
    judge_templates: list[str] = Field(
        default_factory=list, description="Available judge templates"
    )


class TrainingMetric(BaseModel):
    """A single training metrics data point."""

    model_config = {"extra": "ignore"}

    step: int
    loss: float | None = None
    epoch: float | None = None
    learning_rate: float | None = None
    max_steps: int | None = None
    grad_norm: float | None = None


# --- Typed response models ---


class ConfigValidateRequest(BaseModel):
    """Lightweight config validation request (no compute spec)."""

    type: str = Field(..., description="Job type (e.g. 'training', 'sdg')")
    config: dict[str, Any] = Field(..., description="Job-type-specific configuration")


class ConfigValidateResponse(BaseModel):
    """Response from config validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResumeRequest(BaseModel):
    """Request to resume a failed job."""

    checkpoint_id: str | None = Field(
        None, description="Artifact ID of the checkpoint to resume from"
    )


class UploadUrlRequest(BaseModel):
    """Request for a pre-signed upload URL."""

    name: str = Field(..., description="Object name / key")
    content_type: str = Field("application/octet-stream", description="MIME type of the upload")


class UploadUrlResponse(BaseModel):
    """Response with a pre-signed upload URL."""

    url: str
    method: str = "PUT"
    headers: dict[str, str] = Field(default_factory=dict)


class DryRunResponse(BaseModel):
    """Response from a dry-run / validation request."""

    dry_run: bool = True
    valid: bool
    errors: list[str] = Field(default_factory=list)
    type: str
    compute: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    gpu: dict[str, Any] = Field(default_factory=dict)


class JobTypeInfo(BaseModel):
    """Information about a registered job type."""

    type: str
    description: str


class ComputeBackendInfo(BaseModel):
    """Summary of a registered compute backend."""

    name: str
    capabilities: list[str] = Field(default_factory=list)


class ComputeStatusResponse(BaseModel):
    """Status of a compute backend."""

    name: str
    capabilities: list[str] = Field(default_factory=list)
    healthy: bool


class RecipeSummary(BaseModel):
    """Summary of an available recipe."""

    name: str
    description: str = ""
    type: str = ""


class ArtifactPreview(BaseModel):
    """Preview of an artifact's contents."""

    type: str
    format: str
    filename: str
    size: int | None = None
    lines: list[str] | None = None
    total_size: int | None = None


class EventResponse(BaseModel):
    """A single event record."""

    id: str
    job_id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


# --- Judge models ---


class JudgeRequest(BaseModel):
    """Request to judge data quality."""

    template: str = Field(..., description="Judge template name (e.g. generic/safety)")
    data: list[dict[str, Any]] = Field(..., description="Data rows to judge")
    model: str = Field(..., description="LLM model for judging (LiteLLM format)")
    api_base: str | None = Field(None, description="Model API base URL")
    api_key: str | None = Field(None, description="Model API key")


class JudgeResult(BaseModel):
    """Result from judging data."""

    results: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# --- Evaluator models ---


class EvaluatorCreate(BaseModel):
    """Request to create an evaluator."""

    name: str = Field(..., description="Evaluator name")
    description: str = Field("", description="Description")
    type: str = Field("llm", description="Evaluator type: llm or rule_based")
    prompt: str = Field(..., description="Evaluation prompt template")
    judgment_type: str = Field("bool", description="Judgment output type: bool, int, float, enum")
    response_format: str = Field("json", description="Response format: json, xml, raw")
    variables: list[str] = Field(default_factory=list, description="Template variable names")
    model: str | None = Field(None, description="LiteLLM model string for judging")
    inference_params: dict[str, Any] = Field(
        default_factory=dict, description="LLM inference parameters"
    )
    rule_config: dict[str, Any] | None = Field(None, description="Rule config for rule_based type")


class Evaluator(EvaluatorCreate):
    """Evaluator record returned by the API."""

    id: str
    created_at: str
    updated_at: str


class EvaluationCreate(BaseModel):
    """Request to run an evaluation."""

    evaluator_id: str = Field(..., description="Evaluator ID to run")
    dataset: str = Field(..., description="Artifact ID or file path to evaluate")
    model_override: str | None = Field(None, description="Override evaluator's model")
    inference_params_override: dict[str, Any] | None = Field(
        None, description="Override inference params"
    )


class Evaluation(BaseModel):
    """Evaluation run record."""

    id: str
    evaluator_id: str
    dataset_artifact_id: str | None = None
    job_id: str | None = None
    status: str = "pending"
    results: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


# --- Agent / Chat models ---


class MessageRole(StrEnum):
    """Role of a chat message."""

    user = "user"
    assistant = "assistant"


class SuggestedAction(BaseModel):
    """An action the agent suggests the user take."""

    type: str = Field(..., description="Action type (e.g. create_training_job)")
    config: dict[str, Any] = Field(default_factory=dict)
    label: str = Field("", description="Human-readable label for the action")


class AgentResponse(BaseModel):
    """Response from the agent."""

    message: str
    suggested_action: SuggestedAction | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request to send a message to the agent."""

    message: str = Field(..., min_length=1, description="User message")
    conversation_id: str | None = Field(
        None, description="Existing conversation ID (creates new if omitted)"
    )


class ChatResponse(BaseModel):
    """Response from the agent chat endpoint."""

    conversation_id: str
    message: str
    suggested_action: SuggestedAction | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A chat message."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: MessageRole
    content: AgentResponse | str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Conversation(BaseModel):
    """A chat conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ConversationDetail(BaseModel):
    """A conversation with its messages."""

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[Message] = Field(default_factory=list)
