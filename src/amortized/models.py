"""Pydantic models for the Amortized v1 API."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobType(StrEnum):
    training = "training"
    sdg = "sdg"
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
    data_path: str = Field(..., description="Path to training data")
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
    report_to: str | None = Field(None, description="Logging backend (none, mlflow)")


class SynthJobConfig(BaseModel):
    model: str = Field(..., description="Teacher model (LiteLLM format)")
    api_base: str | None = Field(None, description="Model API base URL")
    api_key: str | None = Field(None, description="Model API key")
    num_samples: int = Field(100, ge=1, description="Number of samples to generate")
    max_concurrency: int = Field(16, ge=1, description="Max concurrent LLM requests")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, ge=0, le=1)
    seed: int | None = Field(None, description="Random seed")
    num_retries: int | None = Field(None, ge=0)
    input_data: list[dict[str, Any]] | None = Field(None, description="Input dataset sources")
    input_documents: list[dict[str, Any]] | None = Field(None, description="Input document sources")
    strategy_params: dict[str, Any] | None = Field(None, description="Raw asynth params")


class EvalJobConfig(BaseModel):
    model_config = {"extra": "allow"}

    dataset: str = Field(..., description="Dataset path or artifact URI to evaluate")
    judge: dict[str, Any] = Field(default_factory=dict, description="Judge configuration")


class JobRequest(BaseModel):
    type: JobType = Field(..., description="Job type: training, sdg, eval")
    config: dict[str, Any] = Field(..., description="Job-type-specific configuration")
    recipe: str = Field("", description="Recipe name if used")
    parent_job_id: str = Field("", description="Parent job ID for lineage")
    compute: ComputeSpec = Field(default_factory=ComputeSpec)
    dry_run: bool = Field(False, description="Validate without creating the job")


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


class RecipeSummary(BaseModel):
    name: str
    description: str = ""
    type: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    gpu: dict[str, Any] = Field(default_factory=dict)


class DryRunResponse(BaseModel):
    dry_run: bool = True
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    version: str = "1.0.0"
    default_compute_backend: str = ""
    compute_namespace: str = ""
    mlflow_tracking_uri: str = ""
    mlflow_gateway_uri: str = ""
    image_registry: str = ""
    available_backends: list[str] = Field(default_factory=list)
