"""Job type registry and universal job submission endpoints."""

import logging
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from amortized.core.job_types import (
    UnknownJobTypeError,
    get_schema,
    list_job_types,
    validate_config,
    validate_semantic,
    warn_semantic,
)
from amortized.core.jobs import InvalidJobStateError
from amortized.core.jobs import create_job as core_create_job
from amortized.core.jobs import validate_job as core_validate_job
from amortized.core.redact import redact_config
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import (
    ConfigValidateRequest,
    ConfigValidateResponse,
    DryRunResponse,
    Job,
    JobRequest,
    JobType,
    JobTypeInfo,
)

logger = logging.getLogger("amortized.api.job_types")

router = APIRouter(tags=["jobs"])


@router.post("/api/v1/jobs", status_code=201, response_model=Job | DryRunResponse)
async def create_job_universal(
    request: JobRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job | DryRunResponse:
    try:
        schema_errors = validate_config(request.type, request.config)
    except UnknownJobTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.dry_run:
        semantic_errors = await validate_semantic(request.type, request.config)
        semantic_warnings = await warn_semantic(request.type, request.config)
        all_errors = schema_errors + semantic_errors
        return DryRunResponse(
            valid=not all_errors,
            errors=all_errors,
            warnings=semantic_warnings,
            type=request.type,
            compute=request.compute.model_dump(),
            config=request.config,
        )

    if schema_errors:
        raise HTTPException(status_code=422, detail=schema_errors)

    job_type = JobType(request.type)
    output_dir = request.config.get("ckpt_output_dir")

    merged_metadata = {**request.metadata}
    if request.compute.backend != "local":
        merged_metadata["backend"] = request.compute.backend
    if request.compute.gpus > 0:
        merged_metadata["gpus"] = request.compute.gpus
    if request.compute.gpu_type:
        merged_metadata["gpu_type"] = request.compute.gpu_type

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=job_type,
            config=request.config,
            output_dir=output_dir,
            metadata=merged_metadata,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row["config"] = redact_config(row["config"])
    return Job(**row)


@router.post("/api/v1/jobs/validate", response_model=DryRunResponse)
async def validate_job(request: JobRequest) -> DryRunResponse:
    """Validate a job configuration without creating it."""
    try:
        result = await core_validate_job(
            job_type=request.type,
            config=request.config,
        )
    except UnknownJobTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DryRunResponse(
        valid=result["valid"],
        errors=result["errors"],
        warnings=result.get("warnings", []),
        type=request.type,
        compute=request.compute.model_dump(),
        config=request.config,
    )


@router.post("/api/v1/config/validate", response_model=ConfigValidateResponse)
async def validate_config_endpoint(request: ConfigValidateRequest) -> ConfigValidateResponse:
    """Lightweight config validation — just type + config, no compute spec."""
    try:
        result = await core_validate_job(
            job_type=request.type,
            config=request.config,
        )
    except UnknownJobTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConfigValidateResponse(
        valid=result["valid"],
        errors=result["errors"],
        warnings=result.get("warnings", []),
    )


job_types_router = APIRouter(prefix="/api/v1/job-types", tags=["job-types"])


@job_types_router.get("", response_model=list[JobTypeInfo])
async def get_job_types() -> list[dict[str, str]]:
    return list_job_types()


@job_types_router.get("/{job_type}/schema")
async def get_job_type_schema(job_type: str) -> dict[str, Any]:
    try:
        return get_schema(job_type)
    except UnknownJobTypeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@job_types_router.get("/{job_type}/examples")
async def get_job_type_examples(job_type: str) -> list[dict[str, Any]]:
    """Return working config examples for a job type."""
    if job_type == "sdg":
        return [
            {
                "name": "Simple QA generation",
                "description": "Generate question-answer pairs using a teacher model",
                "config": {
                    "model": "openai/gpt-4o-mini",
                    "num_samples": 50,
                    "strategy_params": {
                        "generated_attributes": [
                            {
                                "id": "question",
                                "instruction_messages": [
                                    {
                                        "role": "user",
                                        "content": "Generate a diverse, interesting "
                                        "question about science.",
                                    }
                                ],
                            },
                            {
                                "id": "answer",
                                "instruction_messages": [
                                    {
                                        "role": "user",
                                        "content": "Answer this question thoroughly: {question}",
                                    }
                                ],
                            },
                        ],
                        "passthrough_attributes": ["question", "answer"],
                    },
                },
            },
            {
                "name": "Customer support with categories",
                "description": "Generate support tickets with sampled urgency and category",
                "config": {
                    "model": "openai/gpt-4o-mini",
                    "num_samples": 100,
                    "strategy_params": {
                        "sampled_attributes": [
                            {
                                "id": "urgency",
                                "name": "Urgency",
                                "description": "How urgent the support request is",
                                "possible_values": [
                                    {
                                        "id": "high",
                                        "name": "High",
                                        "description": "Needs immediate attention",
                                    },
                                    {
                                        "id": "medium",
                                        "name": "Medium",
                                        "description": "Should be handled today",
                                    },
                                    {
                                        "id": "low",
                                        "name": "Low",
                                        "description": "Can wait",
                                    },
                                ],
                            }
                        ],
                        "generated_attributes": [
                            {
                                "id": "ticket",
                                "instruction_messages": [
                                    {
                                        "role": "user",
                                        "content": "Write a {urgency} urgency "
                                        "customer support ticket. "
                                        "The urgency level means: "
                                        "{urgency.description}",
                                    }
                                ],
                            }
                        ],
                        "passthrough_attributes": ["urgency", "ticket"],
                    },
                },
            },
        ]
    elif job_type == "training":
        return [
            {
                "name": "Basic LoRA fine-tune",
                "config": {
                    "algorithm": "lora_sft",
                    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "data_path": "./data.jsonl",
                },
            }
        ]
    return []
