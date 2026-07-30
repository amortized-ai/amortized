"""UI interaction endpoints — structured tool calls for the chat frontend."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/ui", tags=["ui"])


class OptionItem(BaseModel):
    title: str = Field(..., description="Short option label")
    description: str = Field("", description="Brief explanation of this option")
    value: str = Field(..., description="Value sent back when user selects this option")


class PresentOptionsRequest(BaseModel):
    step: str = Field(
        ..., description="Workflow step identifier (e.g. 'sdg-domain', 'training-method')"
    )
    question: str = Field("", description="The question being asked")
    options: list[OptionItem] = Field(
        ..., description="Options to present to the user", min_length=1
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
    summary="Present structured options to the user as clickable cards in the chat UI",
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
    summary="Display a model pricing comparison card in the chat UI",
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
    summary="Display a VRAM estimate comparison card in the chat UI",
)
async def show_vram_estimate(body: ShowVRAMEstimateRequest) -> ShowVRAMEstimateResponse:
    return ShowVRAMEstimateResponse(estimates=body.estimates, rendered=True)


class SignalPhaseRequest(BaseModel):
    phase: str = Field(..., description="Current workflow phase: 'sdg', 'training', or 'eval'")
    step: str = Field(
        "", description="Current step within the phase (e.g. 'gather_requirements', 'confirm')"
    )


class SignalPhaseResponse(BaseModel):
    phase: str
    step: str


@router.post(
    "/signal_phase",
    response_model=SignalPhaseResponse,
    operation_id="signal_phase",
    summary="Signal the current workflow phase and step to the chat UI",
)
async def signal_phase(body: SignalPhaseRequest) -> SignalPhaseResponse:
    return SignalPhaseResponse(phase=body.phase, step=body.step)


class SignalProgressRequest(BaseModel):
    phase: str = Field(
        ..., description="Current workflow phase: 'sdg', 'training', or 'eval'"
    )
    step_id: str = Field(
        ..., description="Stable snake_case step identifier, e.g. 'check_models'"
    )
    label: str = Field(
        ..., description="Human-readable step label, e.g. 'Checking available models'"
    )
    status: str = Field(
        "active",
        description="Step status: 'active' or 'completed'",
    )


class SignalProgressResponse(BaseModel):
    phase: str
    step_id: str
    label: str
    status: str


@router.post(
    "/signal_progress",
    response_model=SignalProgressResponse,
    operation_id="signal_progress",
    summary="Signal the current workflow step to the chat UI for dynamic progress display",
)
async def signal_progress(body: SignalProgressRequest) -> SignalProgressResponse:
    return SignalProgressResponse(
        phase=body.phase, step_id=body.step_id, label=body.label, status=body.status
    )
