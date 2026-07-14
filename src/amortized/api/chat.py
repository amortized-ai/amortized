"""Chat agent endpoint — LLM with tool-use over the Amortized API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import litellm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from amortized.config import settings
from amortized.core.compute import get_backend
from amortized.core.jobs import (
    InvalidJobStateError,
    JobNotFoundError,
    deserialize_handle,
)
from amortized.core.jobs import cancel_job as core_cancel_job
from amortized.core.jobs import create_job as core_create_job
from amortized.core.jobs import get_job as core_get_job
from amortized.core.jobs import list_jobs as core_list_jobs
from amortized.core.recipes import (
    RecipeNotFoundError,
    apply_overrides,
    flatten_recipe_to_config,
    list_recipes,
    load_recipe,
)
from amortized.core.redact import redact_config
from amortized.db import get_db
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType

logger = logging.getLogger("amortized.api.chat")

router = APIRouter(prefix="/agent", tags=["agent"])

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = 3600


@dataclass
class Session:
    id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now(UTC).timestamp())
    last_active_at: float = field(default_factory=lambda: datetime.now(UTC).timestamp())
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_sessions: dict[str, Session] = {}


def _prune_sessions() -> None:
    now = datetime.now(UTC).timestamp()
    expired = [sid for sid, s in _sessions.items() if now - s.last_active_at > SESSION_TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]


# ---------------------------------------------------------------------------
# Request / response models (matches frontend OpenCodeResponse contract)
# ---------------------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    id: str


class MessagePart(BaseModel):
    type: str
    text: str | None = None


class SendMessageRequest(BaseModel):
    agent: str = "morty"
    parts: list[MessagePart]


class TokenInfo(BaseModel):
    input: int = 0
    output: int = 0
    reasoning: int = 0


class ResponseInfo(BaseModel):
    providerID: str = ""
    modelID: str = ""
    cost: float = 0.0
    tokens: TokenInfo = TokenInfo()
    finish: str = "stop"
    id: str = ""
    sessionID: str = ""


class ResponsePart(BaseModel):
    type: str
    text: str | None = None
    tool: str | None = None
    callID: str | None = None
    state: str | None = None
    input: dict[str, Any] | None = None
    output: Any = None


class ChatResponse(BaseModel):
    info: ResponseInfo
    parts: list[ResponsePart]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Morty, a helpful AI assistant for the Amortized platform.

Amortized is a control plane for building task-specific AI models through three stages:
1. **SDG (Synthetic Data Generation)** — Generate labeled training data using a teacher LLM
2. **Training** — Fine-tune a smaller model on that data (SFT, LoRA, DPO, GRPO)
3. **Eval** — Evaluate the fine-tuned model against the teacher baseline

## How to interact with users

**Ask ONE question at a time.** Never present multiple questions in a
single message. Wait for the user's answer before moving to the next
question.

**NEVER ask open-ended questions.** Every question you ask MUST include
a numbered list of options for the user to click. The frontend renders
numbered lists as clickable buttons. If you ask a question without
options, the user has no buttons to click and the experience is broken.

**Always gather requirements before acting.** When a user describes
what they want to build, walk through these steps ONE AT A TIME, each
with clickable options:

1. **What domain/type?** — Present common domains
2. **What sub-categories?** — Based on their domain, suggest specific categories to classify into
3. **What output labels?** — Ask if they also need urgency levels, sentiment, priority, etc.
4. **How many samples?** — Offer 2-3 numeric choices
5. **Which teacher model?** — Present the available models
6. **Confirm plan** — Show a summary TABLE and ask yes/no to submit
7. **Execute** — Submit the job

**Example flow for "build a support ticket classifier":**

Step 1 — Ask domain:

Great! What type of support tickets will this handle?

1) Software/technical support — Bug reports, feature requests, troubleshooting
2) Billing & payments — Invoices, refunds, subscription issues
3) Customer service — Account access, onboarding, general inquiries
4) E-commerce — Orders, shipping, returns, product questions

Step 2 — After they pick billing, ask sub-categories:

What specific billing categories should the classifier use?

1) Invoice & payment issues — Failed payments, missing invoices, overcharges
2) Refunds & disputes — Refund requests, chargebacks, billing errors
3) Subscription management — Plan changes, cancellations, renewals
4) All of the above — Cover all billing sub-categories

Step 3 — Ask about output labels:

Should the classifier also assign urgency levels to each ticket?

1) Yes, 3 levels — Low, Medium, High
2) Yes, 4 levels — Low, Medium, High, Critical
3) No, just categories — Only classify by topic

## Formatting rules for options

**CRITICAL: EVERY question MUST end with a numbered list.** Format exactly like this:

1) Option name — Brief description
2) Option name — Brief description
3) Option name — Brief description

**Rules:**
- Use `N)` format (e.g., `1)`, `2)`, `3)`)
- Each option on its own line
- Keep each option under 120 characters
- Include ALL relevant options — never skip any
- For numeric inputs (like "how many samples"), suggest 2-3 common values
- The user can always type a custom answer, so don't worry about covering every case

**Example for numeric choices:**

How many training samples should we generate?

1) 100 samples — Quick test run
2) 500 samples — Good for most use cases
3) 1000 samples — Higher quality, takes longer

## Confirmation and submission

When summarizing the plan before submission, use a markdown TABLE
(not bullet points or bold labels). Example:

Here's the plan:

| Setting | Value |
|---------|-------|
| Domain | Billing & payments |
| Categories | Invoices, Refunds, Subscriptions |
| Urgency levels | Low, Medium, High, Critical |
| Samples | 500 |
| Teacher model | Claude Haiku |

Ready to submit?

1) Yes, submit the job — Start generating the training data
2) No, change something — Adjust the configuration

## After job submission

When a job is successfully submitted:

1. Show a brief summary of what's running (type, teacher model, sample count, labels)
2. Mention the Job ID clearly on its own line: "Job ID: <uuid>"
3. Do NOT include numbered next-step options — the UI automatically
   adds navigation buttons after job submission

## When the user asks for more details about a job

When the user asks to "see more details" or "show details" for a job:
- The job ID will be in the user's message or in the conversation
  history — NEVER ask the user for the ID
- Call `get_job` with the job ID to get the latest status
- Show a detailed markdown TABLE with ALL configuration: splits,
  percentages, labels, model, sample count, status, duration, artifacts
- Do NOT include numbered next-step options — the UI automatically
  adds navigation buttons

## Model options for SDG

When the user needs to choose a teacher model, present these options with their exact IDs:

1) Claude Haiku — Fast and cheap, good for straightforward tasks
   (ID: anthropic/claude-haiku-4-5-20251001)
2) Claude Sonnet — Higher quality, better for nuanced data
   (ID: anthropic/claude-sonnet-4-20250514)
3) GPT-4o — Most capable, best for complex reasoning (ID: openai/gpt-4o)

When calling submit_recipe_job, always pass the selected model ID in the `model` parameter.

## Using tools

- Use `list_recipes` and `get_recipe` to find recipes that match what the user wants
- Use `list_jobs` and `get_job` to check on running or completed work
- Use `submit_recipe_job` only AFTER gathering requirements and
  confirming the plan. NEVER call it more than once per conversation.
  If the user asks about a submitted job, use `get_job` instead —
  do NOT resubmit
- When calling submit_recipe_job, ALWAYS include a
  `task_description` that describes the classification task in
  detail. This is what drives the actual content generation. Without
  it, the system only generates labels with no training text.
  Example: "Classify billing support tickets into categories
  (invoices, refunds, subscriptions) and assign urgency levels
  (Low, Medium, High, Critical)"
- Use `get_job_logs` to help debug failed jobs
- Use `get_config` to check available backends and capabilities

## Formatting

- Use markdown for clarity
- Use tables when presenting lists of jobs or recipes
- Keep messages concise — one concept per message
- Use bold for key terms and options
- Do NOT use emoji in option lists
"""

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_jobs",
            "description": "List all jobs, optionally filtered by status or type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "running", "succeeded", "failed", "cancelled"],
                        "description": "Filter by job status",
                    },
                    "job_type": {
                        "type": "string",
                        "enum": ["sdg", "training", "eval"],
                        "description": "Filter by job type",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job",
            "description": "Get details for a specific job by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job ID"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_job",
            "description": "Cancel a running or pending job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job ID to cancel"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_logs",
            "description": "Fetch log output for a job. Returns the last N lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job ID"},
                    "tail": {
                        "type": "integer",
                        "description": "Number of lines to return (default 100)",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_artifacts",
            "description": "Get the MLflow artifact URI for a completed job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job ID"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_recipes",
            "description": "List all available recipes (SDG, training, eval templates).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipe",
            "description": "Load a specific recipe by name to see its configuration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Recipe name (e.g. 'sdg/basic')"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_recipe_job",
            "description": (
                "Submit a job from a recipe. Optionally provide overrides for recipe config values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe": {
                        "type": "string",
                        "description": "Recipe name (e.g. 'sdg/basic')",
                    },
                    "model": {
                        "type": "string",
                        "enum": [
                            "anthropic/claude-haiku-4-5-20251001",
                            "anthropic/claude-sonnet-4-20250514",
                            "openai/gpt-4o",
                            "openai/gpt-4o-mini",
                        ],
                        "description": "LiteLLM model ID for the teacher/generator model",
                    },
                    "task_description": {
                        "type": "string",
                        "description": (
                            "Description of the task for synthetic"
                            " data generation. This drives the"
                            " content of generated samples."
                            " Example: 'Classify billing support"
                            " tickets by urgency level (Low,"
                            " Medium, High, Critical) and category"
                            " (invoices, refunds, subscriptions)'"
                        ),
                    },
                    "overrides": {
                        "type": "object",
                        "description": "Dot-notation overrides for recipe config values",
                    },
                },
                "required": ["recipe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_config",
            "description": "Get the platform configuration (available backends, MLflow URI, etc.).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    try:
        return await _dispatch_tool(name, arguments)
    except Exception as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return json.dumps({"error": str(exc)})


@asynccontextmanager
async def _get_repo():
    db_gen = get_db()
    db = await db_gen.__anext__()
    try:
        yield Repository(db)
    finally:
        with contextlib.suppress(StopAsyncIteration):
            await db_gen.__anext__()


async def _dispatch_tool(name: str, args: dict[str, Any]) -> str:
    async with _get_repo() as repo:
        return await _dispatch_with_repo(name, args, repo)


async def _dispatch_with_repo(name: str, args: dict[str, Any], repo: Repository) -> str:
    if name == "list_jobs":
        status = JobStatus(args["status"]) if args.get("status") else None
        job_type = JobType(args["job_type"]) if args.get("job_type") else None
        rows = await core_list_jobs(repo, status=status, job_type=job_type)
        for r in rows:
            r["config"] = redact_config(r.get("config", {}))
        return json.dumps(rows, default=str)

    if name == "get_job":
        row = await core_get_job(repo, args["job_id"])
        if row is None:
            return json.dumps({"error": f"Job {args['job_id']} not found"})
        row["config"] = redact_config(row.get("config", {}))
        return json.dumps(row, default=str)

    if name == "cancel_job":
        try:
            row = await core_cancel_job(repo, args["job_id"])
            row["config"] = redact_config(row.get("config", {}))
            return json.dumps(row, default=str)
        except (JobNotFoundError, InvalidJobStateError) as exc:
            return json.dumps({"error": str(exc)})

    if name == "get_job_logs":
        row = await core_get_job(repo, args["job_id"])
        if row is None:
            return json.dumps({"error": f"Job {args['job_id']} not found"})
        handle = deserialize_handle(row.get("backend_handle"))
        if handle is None:
            return json.dumps(
                {
                    "job_id": args["job_id"],
                    "logs": [],
                    "message": "Job has not started yet",
                }
            )
        try:
            backend = get_backend(handle.backend_name)
        except KeyError:
            return json.dumps({"error": f"Backend {handle.backend_name!r} not available"})
        tail = args.get("tail", 100)
        lines: list[str] = []
        async for line in backend.logs(handle):
            lines.append(line)
            if len(lines) > tail:
                lines = lines[-tail:]
        return json.dumps({"job_id": args["job_id"], "logs": lines})

    if name == "get_job_artifacts":
        row = await core_get_job(repo, args["job_id"])
        if row is None:
            return json.dumps({"error": f"Job {args['job_id']} not found"})
        mlflow_run_id = row.get("mlflow_run_id", "")
        if not mlflow_run_id:
            return json.dumps(
                {
                    "job_id": args["job_id"],
                    "artifact_uri": "",
                    "message": "No MLflow run ID",
                }
            )
        from amortized.worker import _resolve_mlflow_artifact_uri

        artifact_uri = await _resolve_mlflow_artifact_uri(
            mlflow_run_id,
        )
        return json.dumps(
            {
                "job_id": args["job_id"],
                "mlflow_run_id": mlflow_run_id,
                "artifact_uri": artifact_uri,
            }
        )

    if name == "list_recipes":
        return json.dumps(list_recipes(), default=str)

    if name == "get_recipe":
        try:
            return json.dumps(load_recipe(args["name"]), default=str)
        except RecipeNotFoundError as exc:
            return json.dumps({"error": str(exc)})

    if name == "submit_recipe_job":
        try:
            recipe = load_recipe(args["recipe"])
        except RecipeNotFoundError as exc:
            return json.dumps({"error": str(exc)})
        recipe = apply_overrides(recipe, args.get("overrides", {}))
        recipe_type = recipe.get("type")
        if not recipe_type:
            return json.dumps({"error": "Recipe is missing 'type' field"})
        try:
            job_type = JobType(recipe_type)
        except ValueError:
            return json.dumps({"error": f"Unknown job type: {recipe_type}"})
        config = flatten_recipe_to_config(recipe)
        if args.get("model"):
            config["model"] = args["model"]
        if args.get("task_description"):
            config["task_description"] = args["task_description"]
        # Normalize sampled_attributes: asynth requires name+description on each value
        sp = config.get("strategy_params")
        if isinstance(sp, dict):
            for attr in sp.get("sampled_attributes", []):
                if isinstance(attr, dict):
                    attr.setdefault("name", attr.get("id", ""))
                    attr.setdefault("description", attr.get("id", ""))
                    for val in attr.get("possible_values", []):
                        if isinstance(val, dict):
                            val.setdefault("name", val.get("id", ""))
                            val.setdefault("description", val.get("id", ""))
        row = await core_create_job(repo, job_type=job_type, config=config, recipe=args["recipe"])
        row["config"] = redact_config(row.get("config", {}))
        return json.dumps(row, default=str)

    if name == "get_config":
        from amortized.core.compute import get_all_backends

        return json.dumps(
            {
                "default_compute_backend": settings.resolved_default_backend,
                "available_backends": list(get_all_backends().keys()),
                "mlflow_tracking_uri": settings.mlflow_tracking_uri,
                "image_registry": settings.image_registry,
            }
        )

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Chat turn (agentic loop)
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 10


async def _run_chat_turn(session: Session, user_text: str) -> ChatResponse:
    if len(session.messages) > 50:
        session.messages = session.messages[-50:]
    session.messages.append({"role": "user", "content": user_text})

    parts: list[ResponsePart] = []
    total_input = 0
    total_output = 0
    model_id = settings.chat_model
    response_id = ""
    finish = "stop"

    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *session.messages,
    ]

    for _ in range(MAX_ITERATIONS):
        response = await litellm.acompletion(
            model=settings.chat_model,
            messages=full_messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=settings.chat_max_tokens,
            temperature=settings.chat_temperature,
        )

        choice = response.choices[0]
        message = choice.message
        finish = choice.finish_reason or "stop"

        if response.usage:
            total_input += response.usage.prompt_tokens or 0
            total_output += response.usage.completion_tokens or 0
        response_id = response.id or ""

        # Build the assistant message dict for history
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_msg["content"] = message.content
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        session.messages.append(assistant_msg)
        full_messages.append(assistant_msg)

        if not message.tool_calls:
            if message.content:
                parts.append(ResponsePart(type="text", text=message.content))
            break

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

            result = await _execute_tool(fn_name, fn_args)

            parts.append(
                ResponsePart(
                    type="tool",
                    tool=fn_name,
                    callID=tc.id,
                    state="completed",
                    input=fn_args,
                    output=result,
                )
            )

            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            session.messages.append(tool_msg)
            full_messages.append(tool_msg)

    return ChatResponse(
        info=ResponseInfo(
            providerID="litellm",
            modelID=model_id,
            tokens=TokenInfo(input=total_input, output=total_output),
            finish=finish,
            id=response_id,
            sessionID=session.id,
        ),
        parts=parts,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/session", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    _prune_sessions()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = Session(id=session_id)
    logger.info("Created chat session %s", session_id)
    return CreateSessionResponse(id=session_id)


class GenerateTitleRequest(BaseModel):
    message: str


class GenerateTitleResponse(BaseModel):
    title: str


@router.post("/title", response_model=GenerateTitleResponse)
async def generate_title(request: GenerateTitleRequest) -> GenerateTitleResponse:
    """Generate a short conversational title from the first user message."""
    try:
        resp = await litellm.acompletion(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (3-6 words) for"
                        " a conversation that starts with the user"
                        " message below. The title should capture"
                        " the user's intent. Do not use quotes or"
                        " punctuation. Reply with ONLY the title,"
                        " nothing else."
                    ),
                },
                {"role": "user", "content": request.message},
            ],
            max_tokens=30,
            temperature=0.3,
        )
        title = resp.choices[0].message.content.strip().strip("\"'")
        return GenerateTitleResponse(title=title[:60])
    except Exception as exc:
        logger.warning("Title generation failed: %s", exc)
        fallback = request.message[:40] + ("..." if len(request.message) > 40 else "")
        return GenerateTitleResponse(title=fallback)


@router.post("/session/{session_id}/message", response_model=ChatResponse)
async def send_message(session_id: str, request: SendMessageRequest) -> ChatResponse:
    _prune_sessions()
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    user_text = " ".join(p.text for p in request.parts if p.type == "text" and p.text)
    if not user_text.strip():
        raise HTTPException(status_code=400, detail="No text content in message")

    async with session.lock:
        session.last_active_at = datetime.now(UTC).timestamp()
        try:
            return await _run_chat_turn(session, user_text.strip())
        except Exception as exc:
            logger.exception("Chat error in session %s", session_id)
            raise HTTPException(
                status_code=500,
                detail="Internal error processing message",
            ) from exc
