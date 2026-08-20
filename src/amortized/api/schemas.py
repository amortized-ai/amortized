"""Serve JSON schemas for job config models so the frontend can render
schema-aware forms (descriptions, enums, required fields)."""

from fastapi import APIRouter

from amortized.models import SDGJobRequest, TrainingJobConfig

router = APIRouter(prefix="/api/v1/schemas", tags=["schemas"])


@router.get("")
async def get_schemas() -> dict:
    return {
        "sdg": SDGJobRequest.model_json_schema(),
        "training": TrainingJobConfig.model_json_schema(),
    }
