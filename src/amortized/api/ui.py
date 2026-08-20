"""UI interaction endpoints — structured tool calls for the chat frontend."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/ui", tags=["ui"])


class OptionItem(BaseModel):
    title: str = Field(..., description="Short label (1-3 words)")
    description: str = Field("", description="Brief explanation of this option")
    value: str = Field(
        ...,
        description=(
            "Natural language sentence sent as the user's message when clicked "
            "(e.g. 'No, just classify by category' not 'no_urgency')"
        ),
    )


class PresentOptionsRequest(BaseModel):
    step: str = Field(
        ..., description="Workflow step identifier (e.g. 'sdg-domain', 'training-method')"
    )
    question: str = Field("", description="The question being asked")
    options: list[OptionItem] = Field(
        ...,
        description="Options to present as clickable cards. Maximum 4 options, prefer 3.",
        min_length=1,
        max_length=4,
    )


class PresentOptionsResponse(BaseModel):
    step: str
    question: str
    options: list[OptionItem]
    rendered: bool = Field(True, description="Indicates the frontend rendered these as cards")


@router.post(
    "/present_options",
    response_model=PresentOptionsResponse,
    operation_id="present_options",
    summary=(
        "Render clickable option cards in the chat UI. EVERY message that asks a question "
        "or offers choices MUST use this tool — do NOT write numbered lists. Call once per "
        "message, then STOP and wait for the user to respond. Do NOT call this tool after "
        "submitting a job — the UI renders a job monitor card automatically."
    ),
)
async def present_options(body: PresentOptionsRequest) -> PresentOptionsResponse:
    return PresentOptionsResponse(
        step=body.step,
        question=body.question,
        options=body.options,
        rendered=True,
    )


class ModelPricingItem(BaseModel):
    model_id: str = Field(..., description="Model ID (e.g. 'openai/gpt-4o-mini')")
    name: str = Field(..., description="Display name")
    prompt_cost_per_1m: float = Field(..., description="Input cost per 1M tokens")
    completion_cost_per_1m: float = Field(..., description="Output cost per 1M tokens")
    context_length: int = Field(0, description="Context window size")


class ShowModelPricingRequest(BaseModel):
    models: list[ModelPricingItem] = Field(
        ..., description="Models with pricing to display", min_length=1
    )


class ShowModelPricingResponse(BaseModel):
    models: list[ModelPricingItem]
    rendered: bool = Field(True)


@router.post(
    "/show_model_pricing",
    response_model=ShowModelPricingResponse,
    operation_id="show_model_pricing",
    summary=(
        "Display a model pricing comparison card in the chat UI. "
        "Use when presenting teacher model options so the user can compare costs."
    ),
)
async def show_model_pricing(body: ShowModelPricingRequest) -> ShowModelPricingResponse:
    return ShowModelPricingResponse(models=body.models, rendered=True)


class VRAMEstimateItem(BaseModel):
    model_size: str = Field(..., description="Model size (e.g. '8B')")
    method: str = Field(..., description="Training method (e.g. 'lora', 'qlora', 'osft')")
    vram_per_gpu_gb: float = Field(..., description="Expected VRAM per GPU in GB")
    vram_range: str = Field("", description="Low-high range (e.g. '17.6-22.2 GB')")


class ShowVRAMEstimateRequest(BaseModel):
    estimates: list[VRAMEstimateItem] = Field(
        ..., description="VRAM estimates to display", min_length=1
    )


class ShowVRAMEstimateResponse(BaseModel):
    estimates: list[VRAMEstimateItem]
    rendered: bool = Field(True)


@router.post(
    "/show_vram_estimate",
    response_model=ShowVRAMEstimateResponse,
    operation_id="show_vram_estimate",
    summary=(
        "Display a VRAM estimate comparison card in the chat UI. "
        "Use before presenting model or training method options so the user "
        "can see GPU memory requirements."
    ),
)
async def show_vram_estimate(body: ShowVRAMEstimateRequest) -> ShowVRAMEstimateResponse:
    return ShowVRAMEstimateResponse(estimates=body.estimates, rendered=True)


class SignalPhaseRequest(BaseModel):
    phase: str = Field(
        ...,
        description=(
            "Current workflow phase: 'sdg' for data generation workflows, "
            "'training' for model training workflows"
        ),
    )
    step: str = Field(
        "",
        description=(
            "Current step within the phase. Signal transitions in order: "
            "understand_task (identified what the user wants to build), "
            "load_skill (read the relevant skill guide), "
            "gather_requirements (asking user for parameters — signal on first ask), "
            "estimate_cost (checking models, pricing, or VRAM estimates), "
            "confirm (presenting the job validation/confirmation card), "
            "execute (job has been submitted), "
            "review (checking job results or presenting next steps). "
            "Signal the step you are currently at. Do not repeat a step already signaled."
        ),
    )


class SignalPhaseResponse(BaseModel):
    phase: str
    step: str


@router.post(
    "/signal_phase",
    response_model=SignalPhaseResponse,
    operation_id="signal_phase",
    summary=(
        "Update the UI progress bar. You MUST call this once on every response "
        "during an SDG or training workflow. Set phase to 'sdg' or 'training' "
        "based on the current workflow. Call once per response at the current "
        "step — do not batch or skip."
    ),
)
async def signal_phase(body: SignalPhaseRequest) -> SignalPhaseResponse:
    return SignalPhaseResponse(phase=body.phase, step=body.step)


class DelegateRequest(BaseModel):
    target: str = Field(..., description="Workflow agent to delegate to: 'sdg' or 'training'")
    context: str = Field(
        ...,
        description=(
            "Full context for the workflow agent. Include: "
            "(1) conversation history summary — what has been done so far "
            "(completed jobs with IDs, models used, dataset sizes, outcomes), "
            "(2) current user intent — what the user wants to do now, "
            "(3) relevant artifact IDs (job IDs, dataset IDs, document IDs). "
            "The workflow agent starts with no memory of prior conversation, "
            "so this context is all it has."
        ),
    )


class DelegateResponse(BaseModel):
    status: str = Field("ok")
    target: str


@router.post(
    "/delegate_to_subagent",
    response_model=DelegateResponse,
    operation_id="delegate_to_subagent",
    summary="Delegate the conversation to a specialized workflow agent for SDG or training",
)
async def delegate_to_subagent(body: DelegateRequest) -> DelegateResponse:
    from amortized.api.agent import queue_delegation

    queue_delegation(body.target, body.context)
    return DelegateResponse(target=body.target)


class SubagentCompletionRequest(BaseModel):
    summary: str = Field(
        ...,
        description=(
            "Summary of completed work including: job ID, job type (sdg/training), "
            "key parameters (model, method, sample count, parent job ID if chained)"
        ),
    )


class SubagentCompletionResponse(BaseModel):
    status: str = Field("ok")


@router.post(
    "/signal_subagent_completion",
    response_model=SubagentCompletionResponse,
    operation_id="signal_subagent_completion",
    summary="Signal that the workflow agent has completed its task and hand control back to the orchestrator",
)
async def signal_subagent_completion(
    body: SubagentCompletionRequest,
) -> SubagentCompletionResponse:
    from amortized.api.agent import queue_completion

    queue_completion(body.summary)
    return SubagentCompletionResponse()
