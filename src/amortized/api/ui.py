"""UI interaction endpoints — structured tool calls for the chat frontend."""

from __future__ import annotations

from typing import Literal

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
    phase: str = Field(..., description="Current workflow phase: 'sdg', 'training', or 'eval'")
    step_id: str = Field(..., description="Stable identifier for dedup (e.g. 'check_models')")
    label: str = Field(
        ..., description="Human-readable step label (e.g. 'Checking available models')"
    )
    status: Literal["active", "completed"] = Field("active", description="Step status")


class SignalProgressResponse(BaseModel):
    phase: str
    step_id: str
    label: str
    status: str


@router.post(
    "/signal_progress",
    response_model=SignalProgressResponse,
    operation_id="signal_progress",
    summary="Report a progress step to the chat UI progress bar",
)
async def signal_progress(body: SignalProgressRequest) -> SignalProgressResponse:
    return SignalProgressResponse(
        phase=body.phase,
        step_id=body.step_id,
        label=body.label,
        status=body.status,
    )
