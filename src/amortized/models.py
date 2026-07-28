"""Pydantic models for the Amortized v1 API."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobType(StrEnum):
    training = "training"
    sdg = "sdg"


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
    report_to: str | None = Field(None, description="Logging backend (none, mlflow)")


class JobRequest(BaseModel):
    type: JobType = Field(..., description="Job type: training, sdg")
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


class GpuDeviceUtilization(BaseModel):
    index: int
    name: str
    gpu_utilization_pct: float
    memory_utilization_pct: float
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    temperature_celsius: int | None


class GpuUtilizationResponse(BaseModel):
    available: bool
    devices: list[GpuDeviceUtilization] = Field(default_factory=list)
    timestamp: str


class DryRunResponse(BaseModel):
    dry_run: bool = True
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    type: str
    config: dict[str, Any] = Field(default_factory=dict)


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


class ConvertOptions(BaseModel):
    output_format: OutputFormat = Field(OutputFormat.md, description="Output format")
    do_ocr: bool = Field(True, description="Enable OCR for scanned documents")
    ocr_engine: str = Field("easyocr", description="OCR engine: easyocr, tesseract")
    table_mode: str = Field("fast", description="Table detection mode: fast, accurate")


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
    processing_time: float = Field(0.0, ge=0, description="Processing time in seconds")
    status: str = Field("success", description="Conversion status")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues")


class DocumentSummary(_DocumentBase):
    created_at: str | None = Field(None, description="When the document was processed")


class ConfigResponse(BaseModel):
    version: str = "1.0.0"
    default_compute_backend: str = ""
    compute_namespace: str = ""
    mlflow_tracking_uri: str = ""
    mlflow_gateway_uri: str = ""
    docling_enabled: bool = False
    image_registry: str = ""
    available_backends: list[str] = Field(default_factory=list)
