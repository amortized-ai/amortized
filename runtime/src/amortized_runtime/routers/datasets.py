"""Dataset creation and preview endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger("amortized_runtime.datasets")

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

DATASETS_DIR = Path("datasets")


class CreateDatasetRequest(BaseModel):
    filename: str
    rows: list[dict[str, Any]]


class CreateDatasetResponse(BaseModel):
    path: str
    rows_written: int
    columns: list[str]


class PreviewDatasetResponse(BaseModel):
    path: str
    rows: list[dict[str, Any]]
    columns: list[str]
    total_rows_previewed: int


@router.post("", response_model=CreateDatasetResponse)
async def create_dataset(req: CreateDatasetRequest) -> CreateDatasetResponse:
    """Create a JSONL dataset file on disk."""
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")

    # Sanitize filename
    filename = Path(req.filename).name
    if not filename.endswith(".jsonl"):
        filename += ".jsonl"

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATASETS_DIR / filename

    with open(file_path, "w") as f:
        for row in req.rows:
            f.write(json.dumps(row) + "\n")

    columns = list(req.rows[0].keys()) if req.rows else []
    logger.info("Created dataset %s with %d rows", file_path, len(req.rows))

    return CreateDatasetResponse(
        path=str(file_path),
        rows_written=len(req.rows),
        columns=columns,
    )


@router.get("/{path:path}/preview", response_model=PreviewDatasetResponse)
async def preview_dataset(
    path: str,
    rows: int = Query(default=3, ge=1, le=10),
) -> PreviewDatasetResponse:
    """Preview the first few rows of a dataset file."""
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {path}")

    parsed_rows: list[dict[str, Any]] = []
    try:
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= rows:
                    break
                line = line.strip()
                if line:
                    parsed_rows.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in dataset: {exc}"
        ) from exc

    columns = list(parsed_rows[0].keys()) if parsed_rows else []

    return PreviewDatasetResponse(
        path=path,
        rows=parsed_rows,
        columns=columns,
        total_rows_previewed=len(parsed_rows),
    )
