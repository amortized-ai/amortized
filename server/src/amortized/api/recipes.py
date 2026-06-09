"""Recipe browsing and job submission from recipes."""

import logging
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from amortized.core.jobs import InvalidJobStateError
from amortized.core.jobs import create_job as core_create_job
from amortized.core.jobs import validate_job as core_validate_job
from amortized.core.recipes import RecipeNotFoundError, apply_overrides, list_recipes, load_recipe
from amortized.core.redact import redact_config
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import DryRunResponse, Job, JobType, RecipeSummary

logger = logging.getLogger("amortized.api.recipes")

router = APIRouter(prefix="/api/v1/recipes", tags=["recipes"])


@router.get("", response_model=list[RecipeSummary])
async def get_recipes() -> list[dict[str, Any]]:
    return list_recipes()


@router.get("/{name:path}")
async def get_recipe(name: str) -> dict[str, Any]:
    try:
        return load_recipe(name)
    except RecipeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class RecipeJobRequest(BaseModel):
    recipe: str = Field(..., description="Recipe name (e.g. 'models/qwen-1.5b-lora')")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Dot-notation overrides")
    dry_run: bool = Field(True, description="Validate without creating the job")


recipe_jobs_router = APIRouter(tags=["recipes"])


@recipe_jobs_router.post("/api/v1/jobs/recipe", status_code=201, response_model=None)
async def submit_recipe_job(
    request: RecipeJobRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job | JSONResponse:
    try:
        recipe = load_recipe(request.recipe)
    except RecipeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    recipe = apply_overrides(recipe, request.overrides)

    recipe_type = recipe.get("type")
    if not recipe_type:
        raise HTTPException(status_code=422, detail="Recipe is missing 'type' field")

    try:
        job_type = JobType(recipe_type)
    except ValueError:
        raise HTTPException(  # noqa: B904
            status_code=422, detail=f"Unknown job type in recipe: {recipe_type}"
        )

    config: dict[str, Any] = recipe.get("config", {})

    if request.dry_run:
        result = await core_validate_job(job_type=recipe_type, config=config)
        dry_resp = DryRunResponse(
            valid=result["valid"],
            errors=result["errors"],
            warnings=result.get("warnings", []),
            type=recipe_type,
            config=redact_config(config),
        )
        return JSONResponse(content=dry_resp.model_dump(), status_code=200)

    output_dir = config.get("ckpt_output_dir")
    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=job_type,
            config=config,
            output_dir=output_dir,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row["config"] = redact_config(row["config"])
    return Job(**row)
