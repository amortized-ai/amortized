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


class JobStatus(StrEnum):
    """Status of a job."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


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


# --- Response models ---


class Job(BaseModel):
    """Job record returned by the API."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType
    status: JobStatus = JobStatus.pending
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    output_dir: str | None = None


class Artifact(BaseModel):
    """An output artifact from a job."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    artifact_type: str
    path: str
    size: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


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


class TrainingMetric(BaseModel):
    """A single training metrics data point."""

    step: int
    loss: float
    epoch: float | None = None
    learning_rate: float | None = None
    max_steps: int | None = None
