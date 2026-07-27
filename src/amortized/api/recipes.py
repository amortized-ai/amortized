"""Recipe browsing and job submission from recipes."""

import logging
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from amortized.core.jobs import InvalidJobStateError
from amortized.core.jobs import create_job as core_create_job
from amortized.core.recipes import (
    ProtectedRecipeError,
    RecipeNotFoundError,
    apply_overrides,
    flatten_recipe_to_config,
    get_recipes_dir,
    list_recipes,
    load_recipe,
)
from amortized.core.recipes import (
    delete_recipe as core_delete_recipe,
)
from amortized.core.redact import redact_config
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import DryRunResponse, Job, JobType, RecipeSummary

logger = logging.getLogger("amortized.api.recipes")

router = APIRouter(prefix="/api/v1/recipes", tags=["recipes"])


@router.get("", response_model=list[RecipeSummary], operation_id="list_recipes")
async def get_recipes() -> list[dict[str, Any]]:
    return list_recipes()


@router.get("/{name:path}", operation_id="get_recipe")
async def get_recipe(name: str) -> dict[str, Any]:
    try:
        return load_recipe(name)
    except RecipeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class SaveRecipeRequest(BaseModel):
    type: str = Field(..., description="Recipe type: training, sdg, eval")
    description: str = Field("", description="Recipe description")
    config: dict[str, Any] = Field(default_factory=dict, description="Recipe configuration")


@router.put("/{name:path}", operation_id="save_recipe")
async def save_recipe(name: str, body: SaveRecipeRequest) -> dict[str, Any]:
    import yaml as _yaml

    if ".." in name.split("/"):
        raise HTTPException(status_code=400, detail="Invalid recipe name")

    base_dir = get_recipes_dir()
    path = base_dir / f"{name}.yaml"

    resolved = path.resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid recipe name")

    if not path.is_file() and not name.startswith(("templates/", "examples/")):
        name = f"templates/custom/{name}"
        path = base_dir / f"{name}.yaml"
        if not path.resolve().is_relative_to(base_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid recipe name")

    path.parent.mkdir(parents=True, exist_ok=True)

    recipe_data: dict[str, Any] = {
        "type": body.type,
        "description": body.description,
        "config": body.config,
    }
    path.write_text(_yaml.dump(recipe_data, default_flow_style=False, sort_keys=False))
    logger.info("Saved recipe %s to %s", name, path)
    return {"name": name, **recipe_data}



@router.delete("/{name:path}", status_code=204, operation_id="delete_recipe")
async def delete_recipe(name: str) -> None:
    try:
        core_delete_recipe(name)
    except RecipeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProtectedRecipeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class RecipeJobRequest(BaseModel):
    recipe: str = Field(..., description="Recipe name (e.g. 'models/qwen-1.5b-lora')")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Dot-notation overrides")
    parent_job_id: str = Field("", description="Parent job ID for chaining (SDG→Training→Eval)")
    dry_run: bool = Field(False, description="Validate without creating the job")


recipe_jobs_router = APIRouter(tags=["recipes"])


@recipe_jobs_router.post(
    "/api/v1/jobs/recipe",
    status_code=201,
    response_model=None,
    operation_id="submit_recipe_job",
)
async def submit_recipe_job(
    request: RecipeJobRequest,
    http_request: Request,
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

    config = flatten_recipe_to_config(recipe)

    if request.dry_run:
        from amortized.api.jobs import _validate_config

        errors = _validate_config(job_type, config)
        dry_resp = DryRunResponse(
            valid=not errors,
            errors=errors,
            warnings=[],
            type=recipe_type,
            config=redact_config(config),
        )
        return JSONResponse(content=dry_resp.model_dump(), status_code=200)

    from amortized.api.jobs import _strip_secrets

    clean_config, secrets = _strip_secrets(config)
    user_id = http_request.headers.get("X-Forwarded-User", "")

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=job_type,
            config=clean_config,
            recipe=request.recipe,
            parent_job_id=request.parent_job_id,
            user_id=user_id,
            secrets=secrets,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row["config"] = redact_config(row["config"])
    return Job(**row)
