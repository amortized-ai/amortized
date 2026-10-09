"""UI interaction endpoints — structured tool calls for the chat frontend."""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/api/v1/ui", tags=["ui"])

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# A leading number with an optional magnitude suffix: '128k', '1.5M', '131072'.
_INT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([kKmMgG])?")
_SUFFIX_MULTIPLIER = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}


def _coerce_number(value: Any) -> Any:
    """Pull a leading number out of a string like '$0.037', '0.17/1M tokens'.

    The pricing card is a display-only convenience; an agent that passes a cost as
    a formatted string (observed with GLM) should still render the card rather than
    hard-fail validation. Only a leading number (optionally after a currency symbol)
    is coerced — we don't dig a number out of arbitrary prose, so non-numeric input
    passes through untouched and pydantic reports a normal error for it.
    """
    if isinstance(value, str):
        stripped = value.replace(",", "").strip().lstrip("$€£ ")
        match = _NUM_RE.match(stripped)
        if match:
            return match.group(0)
    return value


def _coerce_context_length(value: Any) -> Any:
    """Coerce a context-window string to an int, tolerating magnitude suffixes and
    whole-number floats: '128k' -> 128000, '1.5M' -> 1500000, '131072.0' -> 131072.

    context_length is an INTEGER field, so the float-tolerant _coerce_number is wrong
    for it — it would truncate '128k' to 128 and leave '131072.0' as a string that
    then 422s on int coercion (the very hard-fail the coercion exists to avoid). Like
    _coerce_number this is a display-only best-effort: unparseable input passes through
    untouched for pydantic to report normally.
    """
    if not isinstance(value, str):
        return value
    stripped = value.replace(",", "").strip()
    match = _INT_RE.match(stripped)
    if not match:
        return value
    number, suffix = match.group(1), match.group(2)
    return int(float(number) * _SUFFIX_MULTIPLIER.get((suffix or "").lower(), 1))


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


_PRESENT_OPTIONS_HALT = (
    "You have presented this question to the user. STOP NOW — this ENDS your turn."
    " Do NOT call any more tools (no further present_options, show_prompt,"
    " validate_*, create_*, or any read), and write no more text. Wait for the"
    " user's selection; act on it in your NEXT turn."
)


class PresentOptionsResponse(BaseModel):
    step: str
    question: str
    options: list[OptionItem]
    rendered: bool = Field(True, description="Indicates the frontend rendered these as cards")
    # Authoritative server directive the model reads in the tool result. The system
    # prompt already says "ask one question, then wait", but weaker drivers batch
    # several present_options + a validate_* into one turn and never wait; a tool
    # result outranks the prompt, so repeat the stop here. As a backstop the proxy's
    # _enforce_single_interaction keeps this turn's passive content but strips any
    # additional interactions (other present_options / validate_*) from it (agent.py).
    agent_instruction: str = Field(default=_PRESENT_OPTIONS_HALT, exclude=False)


@router.post(
    "/present_options",
    response_model=PresentOptionsResponse,
    operation_id="present_options",
    summary=(
        "Render clickable option cards in the chat UI. EVERY message that asks a question "
        "or offers choices MUST use this tool — do NOT write numbered lists. Call once per "
        "message, then STOP and wait for the user to respond. Do NOT call this tool after "
        "submitting a job — the UI renders a job monitor card automatically. Do NOT call it "
        "after validate_* either — the confirmation card already has its own Confirm/Cancel, "
        "so a 'confirm?' question under it is redundant and will be dropped."
    ),
)
async def present_options(body: PresentOptionsRequest) -> PresentOptionsResponse:
    return PresentOptionsResponse(
        step=body.step,
        question=body.question,
        options=_dedup_options(body.options),
        rendered=True,
    )


def _dedup_options(options: list[OptionItem]) -> list[OptionItem]:
    """Drop duplicate option cards within a single call.

    A model under protocol stress sometimes lists the same choice twice (same
    click-text), which renders as redundant cards the user cannot tell apart. Dedup
    by the click `value` — the choice's identity — preserving first-seen order. We
    deliberately do NOT dedup by `title`: titles are 1-3 word labels, so two
    genuinely different choices can share one (e.g. two base models both labelled
    "8B" with different `value`s), and dropping the second would make it
    unselectable. Title is used only as the identity when an option has no `value`.
    (Cross-turn re-asking of an identical option set is a separate, client-side
    concern — these cards are rendered from the session message history, not this
    response alone.)
    """
    seen: set[str] = set()
    deduped: list[OptionItem] = []
    for opt in options:
        key = opt.value.strip().lower() or f"title:{opt.title.strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opt)
    return deduped


class ShowPromptRequest(BaseModel):
    title: str = Field(
        "System prompt",
        description="Short card heading (e.g. 'Assessor system prompt')",
    )
    prompt: str = Field(
        ...,
        description="The FULL prompt text to display verbatim for the user to review",
        min_length=1,
    )
    purpose: str = Field(
        "",
        description="One-line note on what the prompt is used for (optional)",
    )


_SHOW_PROMPT_REVIEW = (
    "The prompt is now shown for review. Do NOT submit or confirm a job in this"
    " same turn (no validate_*/create_*) — the user must review this prompt first,"
    " because it ships in the training data. Ask them to approve it or request"
    " edits (present_options), then STOP and wait; build/confirm the job in your"
    " NEXT turn, after they approve."
)


class ShowPromptResponse(BaseModel):
    title: str
    prompt: str
    purpose: str
    rendered: bool = Field(True)
    # Server directive the model reads in the tool result. A weaker driver shows
    # the assessor prompt and stacks a validate_* confirm card under it in the same
    # turn, so the user confirms the job before reviewing the prompt. A tool result
    # outranks the system prompt; the proxy also drops the stacked confirm card as
    # a backstop (see _enforce_single_interaction in agent.py).
    agent_instruction: str = Field(default=_SHOW_PROMPT_REVIEW, exclude=False)


@router.post(
    "/show_prompt",
    response_model=ShowPromptResponse,
    operation_id="show_prompt",
    summary=(
        "Render a prompt in a review card in the chat UI. Call this with the "
        "FULL prompt text WHENEVER you ask the user to review or approve a "
        "prompt (e.g. a generated system/assessor prompt) — the prompt text is "
        "otherwise never shown to the user. Never say 'here is the prompt' or "
        "'the prompt above' without calling this tool in the same response."
    ),
)
async def show_prompt(body: ShowPromptRequest) -> ShowPromptResponse:
    return ShowPromptResponse(
        title=body.title,
        prompt=body.prompt,
        purpose=body.purpose,
        rendered=True,
        agent_instruction=_SHOW_PROMPT_REVIEW,
    )


class ModelPricingItem(BaseModel):
    model_id: str = Field(..., description="Model ID (e.g. 'openai/gpt-4o-mini')")
    name: str = Field(..., description="Display name")
    prompt_cost_per_1m: float = Field(..., description="Input cost per 1M tokens")
    completion_cost_per_1m: float = Field(..., description="Output cost per 1M tokens")
    context_length: int = Field(0, description="Context window size")

    @field_validator("prompt_cost_per_1m", "completion_cost_per_1m", mode="before")
    @classmethod
    def _accept_numeric_strings(cls, value: Any) -> Any:
        return _coerce_number(value)

    @field_validator("context_length", mode="before")
    @classmethod
    def _accept_context_length_strings(cls, value: Any) -> Any:
        return _coerce_context_length(value)


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
            "'training' for model training workflows, 'eval' for model "
            "evaluation workflows"
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
        "during an SDG, training, or eval workflow. Set phase to 'sdg', "
        "'training', or 'eval' based on the current workflow. Call once per "
        "response at the current step — do not batch or skip."
    ),
)
async def signal_phase(body: SignalPhaseRequest) -> SignalPhaseResponse:
    return SignalPhaseResponse(phase=body.phase, step=body.step)


class DelegateRequest(BaseModel):
    target: Literal["sdg", "training", "eval"] = Field(
        ..., description="Workflow agent to delegate to: 'sdg', 'training', or 'eval'"
    )
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
    resume: bool = Field(
        False,
        description=(
            "If true, resume the previous workflow agent session for this "
            "target instead of creating a new one. The resumed agent keeps "
            "its full conversation history and can pick up where it left off. "
            "Set to true when the user wants to adjust, retry, or iterate on "
            "a job that was just completed or failed (e.g. 'resubmit with "
            "more samples', 'try a lower learning rate', 'use a different "
            "model'). Set to false (default) when the user wants a "
            "fundamentally new job (new task, new dataset, different skill)."
        ),
    )


class DelegateResponse(BaseModel):
    status: str = Field("ok")
    target: str


@router.post(
    "/delegate_to_subagent",
    response_model=DelegateResponse,
    operation_id="delegate_to_subagent",
    summary="Delegate the conversation to a specialized workflow agent (SDG, training, or eval)",
)
async def delegate_to_subagent(body: DelegateRequest) -> DelegateResponse:
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
    summary=(
        "Signal that the workflow agent has completed its task "
        "and hand control back to the orchestrator"
    ),
)
async def signal_subagent_completion(
    body: SubagentCompletionRequest,
) -> SubagentCompletionResponse:
    return SubagentCompletionResponse()
