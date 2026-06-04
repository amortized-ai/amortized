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


# --- Request models ---


class TrainingJobConfig(BaseModel):
    """Configuration for a LoRA SFT training job."""

    model_path: str = Field(..., description="HuggingFace model path or local path")
    data_path: str = Field(..., description="Path to training data (JSONL)")
    ckpt_output_dir: str = Field(..., description="Directory for checkpoints and outputs")
    learning_rate: float | None = Field(None, description="Learning rate (default: 2e-4)")
    num_epochs: int | None = Field(None, ge=1, description="Number of training epochs")
    lora_r: int | None = Field(None, ge=1, description="LoRA rank")
    lora_alpha: int | None = Field(None, ge=1, description="LoRA alpha")
    load_in_4bit: bool | None = Field(None, description="Enable QLoRA 4-bit quantization")
    micro_batch_size: int | None = Field(None, ge=1, description="Micro batch size")
    max_seq_len: int | None = Field(None, ge=1, description="Maximum sequence length")


class SDGJobConfig(BaseModel):
    """Configuration for a synthetic data generation job."""

    flow_id: str = Field(..., description="SDG flow identifier")
    dataset_path: str = Field(..., description="Path to input dataset")
    model: str = Field(..., description="Teacher model name (e.g. openai/gpt-4o)")
    api_base: str | None = Field(None, description="Teacher model API base URL")
    api_key: str | None = Field(None, description="Teacher model API key")
    runtime_params: dict[str, Any] | None = Field(
        None, description="Per-block runtime parameter overrides"
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


class FlowInfo(BaseModel):
    """Information about an available SDG flow."""

    id: str
    name: str
    description: str
    category: str
    required_columns: list[str] = Field(default_factory=list)
    dataset_description: str = ""


class TrainingMetric(BaseModel):
    """A single training metrics data point."""

    model_config = {"extra": "ignore"}

    step: int
    loss: float | None = None
    epoch: float | None = None
    learning_rate: float | None = None
    max_steps: int | None = None
    grad_norm: float | None = None


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
