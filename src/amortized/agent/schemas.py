"""Pydantic tool input models and typed ToolDef registry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolDef:
    """Wraps a tool name, description, and Pydantic input model."""

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model

    def to_openai_schema(self) -> dict[str, Any]:
        if self.input_model is not None:
            schema = self.input_model.model_json_schema()
            schema.pop("title", None)
            schema.pop("$defs", None)
        else:
            schema = {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# ---------------------------------------------------------------------------
# Input models for each tool
# ---------------------------------------------------------------------------


class ListSdgFlowsInput(BaseModel):
    pass


class SubmitSdgJobInput(BaseModel):
    model: str = Field(..., description="Teacher model name (e.g. openai/gpt-4o)")
    num_samples: int | None = Field(
        None, description="Number of samples to generate (default: 100)"
    )
    api_base: str | None = Field(None, description="Teacher model API base URL")
    api_key: str | None = Field(None, description="Teacher model API key")
    temperature: float | None = Field(None, description="Sampling temperature (default: 0.7)")
    max_tokens: int | None = Field(None, description="Max tokens per LLM response")
    top_p: float | None = Field(None, description="Nucleus sampling parameter (0-1)")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    input_data: list[dict[str, Any]] | None = Field(
        None, description="Input dataset source configs (JSONL/CSV/HuggingFace)"
    )
    strategy_params: dict[str, Any] | None = Field(
        None, description="Raw asynth GeneralSynthesisParams (advanced)"
    )


class SubmitTrainingJobInput(BaseModel):
    model_name_or_path: str = Field(..., description="HuggingFace model ID or local path")
    data_path: str = Field(..., description="Path to training data (JSONL)")
    output_dir: str = Field(..., description="Output directory for checkpoints")
    learning_rate: float | None = Field(None, description="Learning rate (default: 2e-4)")
    num_train_epochs: int | None = Field(None, description="Number of training epochs (default: 3)")
    lora_r: int | None = Field(None, description="LoRA rank (default: 16)")
    lora_alpha: int | None = Field(None, description="LoRA alpha scaling factor (default: 32)")
    load_in_4bit: bool | None = Field(None, description="Enable QLoRA 4-bit quantization")
    per_device_train_batch_size: int | None = Field(
        None, description="Batch size per GPU (default: 2)"
    )
    max_length: int | None = Field(None, description="Maximum sequence length (default: 2048)")


class CheckJobStatusInput(BaseModel):
    job_id: str = Field(..., description="The job ID to check")


class GetJobMetricsInput(BaseModel):
    job_id: str = Field(..., description="The job ID to get metrics for")


class ListJobsInput(BaseModel):
    status: str | None = Field(
        None,
        description="Filter by job status",
    )
    type: str | None = Field(
        None,
        description="Filter by job type",
    )


class EstimateVramInput(BaseModel):
    model_name_or_path: str = Field(..., description="HuggingFace model ID")
    lora_r: int | None = Field(None, description="LoRA rank (default: 16)")
    batch_size: int | None = Field(None, description="Batch size (default: 2)")
    max_length: int | None = Field(None, description="Max sequence length (default: 2048)")
    load_in_4bit: bool | None = Field(None, description="Use QLoRA 4-bit quantization")


class ReadArtifactPreviewInput(BaseModel):
    job_id: str = Field(..., description="Job ID")
    artifact_id: str | None = Field(
        None, description="Artifact ID (optional — if omitted, previews the main output file)"
    )
    lines: int | None = Field(None, description="Number of lines to preview (default 5, max 50)")


class CreateDatasetInput(BaseModel):
    filename: str = Field(
        ...,
        description=(
            "Filename for the dataset (e.g. pokemon_seed.jsonl). "
            "Will be saved in the datasets/ directory."
        ),
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Array of JSON objects, each representing one row of "
            "the dataset. Must include the columns required by "
            "the target SDG flow."
        ),
    )


class PreviewDatasetInput(BaseModel):
    path: str = Field(..., description="Path to the dataset file")
    rows: int | None = Field(None, description="Number of rows to preview (default 3, max 10)")


class ConvertDatasetInput(BaseModel):
    source_path: str = Field(..., description="Path to the SDG output dataset")
    output_filename: str = Field(..., description="Filename for the converted dataset")


class JudgeDataInput(BaseModel):
    template: str = Field(
        ...,
        description=(
            "Judge template (e.g. generic/safety, "
            "code/correctness, doc_qa/groundedness)"
        ),
    )
    job_id: str = Field(..., description="Job ID whose output data to judge")
    model: str = Field(..., description="LLM model for judging")
    sample_size: int | None = Field(None, description="Number of rows to judge (default: 10)")


class ListJudgeTemplatesInput(BaseModel):
    pass


class ListApiKeysInput(BaseModel):
    pass


class AddApiKeyInput(BaseModel):
    provider: str = Field(..., description="Provider name (e.g. openai, anthropic, google)")
    key: str = Field(..., description="The API key value")


class ProposeActionInput(BaseModel):
    action_type: str = Field(..., description="The action to propose")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration for the action")
    label: str = Field(..., description="Human-readable button label (e.g. 'Start Training')")


class PresentOptionsInput(BaseModel):
    prompt: str = Field(..., description="Question or prompt for the user")
    options: list[dict[str, str]] = Field(
        ...,
        description="List of options, each with 'label' and optional 'description' and 'value'",
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: list[ToolDef] = [
    ToolDef(
        "list_sdg_flows",
        "List available SDG (synthetic data generation) flows.",
        ListSdgFlowsInput,
    ),
    ToolDef(
        "submit_sdg_job",
        (
            "Submit a synthetic data generation job using asynth. "
            "Use propose_action instead if you want the user to confirm first."
        ),
        SubmitSdgJobInput,
    ),
    ToolDef(
        "submit_training_job",
        (
            "Submit a LoRA SFT training job via TRL. "
            "Use propose_action instead if you want the user to confirm first."
        ),
        SubmitTrainingJobInput,
    ),
    ToolDef(
        "check_job_status",
        "Check the status and details of a specific job.",
        CheckJobStatusInput,
    ),
    ToolDef(
        "get_job_metrics",
        "Get training metrics (loss, learning rate, epoch) for a job.",
        GetJobMetricsInput,
    ),
    ToolDef(
        "list_jobs",
        "List all jobs, optionally filtered by status or type.",
        ListJobsInput,
    ),
    ToolDef(
        "estimate_vram",
        "Estimate GPU VRAM requirements for a training configuration.",
        EstimateVramInput,
    ),
    ToolDef(
        "read_artifact_preview",
        (
            "Preview the contents of a job artifact (first few lines of "
            "generated data, metrics, etc). Use this to assess data quality, "
            "check training metrics, or show the user a sample of generated data."
        ),
        ReadArtifactPreviewInput,
    ),
    ToolDef(
        "create_dataset",
        (
            "Create a JSONL dataset file on disk. Use this to prepare seed "
            "data for SDG flows. Each row should be a JSON object with the "
            "columns required by the target SDG flow. Returns the file path "
            "that can be used as dataset_path in submit_sdg_job."
        ),
        CreateDatasetInput,
    ),
    ToolDef(
        "preview_dataset",
        (
            "Preview the first few rows of a dataset file. Use this to "
            "verify seed data before submitting an SDG job."
        ),
        PreviewDatasetInput,
    ),
    ToolDef(
        "convert_dataset",
        (
            "Convert an SDG output dataset to messages format for training. "
            "Auto-detects the input format (question/answer, input/output, "
            "prompt/response) and converts to the messages format required "
            "by TRL."
        ),
        ConvertDatasetInput,
    ),
    ToolDef(
        "judge_data",
        (
            "Judge the quality of generated data using asynth's built-in judges. "
            "Use after SDG to assess quality before training."
        ),
        JudgeDataInput,
    ),
    ToolDef(
        "list_judge_templates",
        "List available judge templates for assessing data quality.",
        ListJudgeTemplatesInput,
    ),
    ToolDef(
        "list_api_keys",
        (
            "List configured LLM API keys. Shows provider names and "
            "redacted preview (last 4 chars) — never the full key. "
            "Check this before proposing SDG jobs to verify the needed "
            "provider key is configured."
        ),
        ListApiKeysInput,
    ),
    ToolDef(
        "add_api_key",
        (
            "Store an LLM provider API key on the server. The key is "
            "encrypted at rest and injected automatically into SDG and "
            "eval jobs. Supports any LiteLLM provider."
        ),
        AddApiKeyInput,
    ),
    ToolDef(
        "propose_action",
        (
            "Propose an action for the user to confirm before executing. "
            "Use this instead of directly calling submit_training_job or "
            "submit_sdg_job so the user can review and approve the configuration."
        ),
        ProposeActionInput,
    ),
    ToolDef(
        "present_options",
        (
            "Present 2-4 options for the user to choose from. Use this when "
            "the user needs to make a decision between a small set of choices "
            "(e.g. which model to use, which approach to take)."
        ),
        PresentOptionsInput,
    ),
]

TOOLS: list[dict[str, Any]] = [t.to_openai_schema() for t in TOOL_REGISTRY]
