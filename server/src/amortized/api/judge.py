"""Judge API — quality assessment via asynth judges."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from amortized.core.judge_templates import (
    list_judge_templates as _list_templates,
)
from amortized.core.judge_templates import (
    load_judge_template,
)
from amortized.models import JudgeRequest, JudgeResult

logger = logging.getLogger("amortized.api.judge")

router = APIRouter(prefix="/api/v1/judge", tags=["judge"])


@router.post("", response_model=JudgeResult)
async def judge_data(request: JudgeRequest) -> JudgeResult:
    """Judge data quality using an asynth judge template."""
    try:
        from asynth import JudgeConfig, LiteLLMInferenceConfig, create_judge
    except ImportError as err:
        raise HTTPException(
            status_code=501,
            detail="asynth is not installed — judge functionality unavailable",
        ) from err

    try:
        template_data = load_judge_template(request.template)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        config = JudgeConfig(**template_data)
        inference_config = LiteLLMInferenceConfig(
            model=request.model,
            api_base=request.api_base,
            api_key=request.api_key,
        )
        j = create_judge(config, inference_config=inference_config)
        results = j.judge(request.data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    serialized: list[dict[str, Any]] = []
    for r in results:
        if hasattr(r, "model_dump"):
            serialized.append(r.model_dump())
        elif isinstance(r, dict):
            serialized.append(r)
        else:
            serialized.append({"raw": str(r)})

    passed = sum(1 for r in serialized if r.get("passed", False))
    return JudgeResult(
        results=serialized,
        summary={
            "total": len(serialized),
            "passed": passed,
            "failed": len(serialized) - passed,
            "pass_rate": passed / len(serialized) if serialized else 0,
        },
    )


@router.get("/templates")
async def list_judge_templates() -> list[dict[str, str]]:
    """List available judge templates."""
    templates = _list_templates()
    return [{"name": t} for t in templates]
