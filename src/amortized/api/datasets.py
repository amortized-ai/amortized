"""Dataset creation, preview, and conversion endpoints.

Deprecated: use /api/v1/artifacts with artifact_type='dataset' instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import Response

import amortized.config as _config_mod

logger = logging.getLogger("amortized.datasets")

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

_DEPRECATION_HEADER = "Use /api/v1/artifacts with artifact_type='dataset' instead"


def _add_deprecation_headers(response: Response) -> None:
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "2027-01-01"
    response.headers["Link"] = '</api/v1/artifacts?type=dataset>; rel="successor-version"'


def _get_datasets_dir() -> Path:
    configured = _config_mod.settings.datasets_dir
    if configured is not None:
        return configured
    return _config_mod.settings.data_dir / "datasets"


# Column patterns that can be auto-detected and converted to messages format
_COLUMN_PATTERNS: list[tuple[str, str]] = [
    ("question", "answer"),
    ("question", "response"),
    ("input", "output"),
    ("prompt", "response"),
    ("prompt", "completion"),
]


class CreateDatasetRequest(BaseModel):
    filename: str
    rows: list[dict[str, Any]]


class CreateDatasetResponse(BaseModel):
    path: str
    rows_written: int
    columns: list[str]


class ConvertDatasetRequest(BaseModel):
    source_path: str
    output_filename: str
    input_format: str | None = None


class ConvertDatasetResponse(BaseModel):
    path: str
    rows_converted: int
    sample_row: dict[str, Any]


class PreviewDatasetResponse(BaseModel):
    path: str
    rows: list[dict[str, Any]]
    columns: list[str]
    total_rows_previewed: int


@router.post("", response_model=CreateDatasetResponse)
async def create_dataset(
    req: CreateDatasetRequest,
    response: Response,
) -> CreateDatasetResponse:
    """Create a JSONL dataset file on disk. Deprecated: use POST /api/v1/artifacts."""
    _add_deprecation_headers(response)
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")

    # Sanitize filename
    filename = Path(req.filename).name
    if not filename.endswith(".jsonl"):
        filename += ".jsonl"

    datasets_dir = _get_datasets_dir()
    datasets_dir.mkdir(parents=True, exist_ok=True)
    file_path = datasets_dir / filename

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
    response: Response,
    rows: int = Query(default=3, ge=1, le=10),
) -> PreviewDatasetResponse:
    """Preview the first few rows of a dataset file. Deprecated: use GET /api/v1/artifacts."""
    _add_deprecation_headers(response)
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
        raise HTTPException(status_code=400, detail=f"Invalid JSON in dataset: {exc}") from exc

    columns = list(parsed_rows[0].keys()) if parsed_rows else []

    return PreviewDatasetResponse(
        path=path,
        rows=parsed_rows,
        columns=columns,
        total_rows_previewed=len(parsed_rows),
    )


def _detect_format(row: dict[str, Any], hint: str | None) -> tuple[str, str] | None:
    """Detect the user/assistant column pair in a row.

    Returns ``(user_col, assistant_col)`` or ``None`` if undetectable.
    """
    if hint:
        for user_col, asst_col in _COLUMN_PATTERNS:
            if hint == f"{user_col}/{asst_col}" and user_col in row and asst_col in row:
                return user_col, asst_col
    # Auto-detect
    for user_col, asst_col in _COLUMN_PATTERNS:
        if user_col in row and asst_col in row:
            return user_col, asst_col
    return None


def _row_to_messages(row: dict[str, Any], user_col: str, asst_col: str) -> dict[str, Any]:
    """Convert a single row to messages format."""
    return {
        "messages": [
            {"role": "user", "content": str(row[user_col])},
            {"role": "assistant", "content": str(row[asst_col])},
        ]
    }


@router.post("/convert", response_model=ConvertDatasetResponse)
async def convert_dataset(
    req: ConvertDatasetRequest,
    response: Response,
) -> ConvertDatasetResponse:
    """Convert an SDG output dataset to messages format for training. Deprecated."""
    _add_deprecation_headers(response)
    source = Path(req.source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Source dataset not found: {req.source_path}")

    # Read all rows
    rows: list[dict[str, Any]] = []
    try:
        with open(source) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in source: {exc}") from exc

    if not rows:
        raise HTTPException(status_code=400, detail="Source dataset is empty")

    first_row = rows[0]

    # Check if already in messages format
    if "messages" in first_row:
        converted = rows
    else:
        fmt = _detect_format(first_row, req.input_format)
        if fmt is None:
            known = [f"{u}/{a}" for u, a in _COLUMN_PATTERNS]
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot detect input format. Columns found: {list(first_row.keys())}. "
                    f"Expected one of: {known}, or a 'messages' column."
                ),
            )
        user_col, asst_col = fmt
        converted = [_row_to_messages(r, user_col, asst_col) for r in rows]

    # Write output
    output_filename = Path(req.output_filename).name
    if not output_filename.endswith(".jsonl"):
        output_filename += ".jsonl"

    datasets_dir = _get_datasets_dir()
    datasets_dir.mkdir(parents=True, exist_ok=True)
    output_path = datasets_dir / output_filename

    with open(output_path, "w") as f:
        for row in converted:
            f.write(json.dumps(row) + "\n")

    logger.info("Converted %d rows from %s to %s", len(converted), source, output_path)

    return ConvertDatasetResponse(
        path=str(output_path),
        rows_converted=len(converted),
        sample_row=converted[0],
    )
