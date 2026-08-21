"""Tests for GPU allocation endpoint."""

from unittest.mock import AsyncMock, patch

import pytest

from amortized.api.gpu import (
    GpuAllocationResponse,
    _estimate_memory_gib,
    _extract_gpu_info_from_job,
    get_gpu_allocation,
)


def test_extract_gpu_from_compute_section() -> None:
    job: dict = {"config": {"compute": {"gpus": 2, "gpu_type": "A100"}}}
    gpus, gpu_type = _extract_gpu_info_from_job(job)
    assert gpus == 2
    assert gpu_type == "A100"


def test_extract_gpu_from_top_level_config() -> None:
    job: dict = {"config": {"gpus": 4, "gpu_type": "H100"}}
    gpus, gpu_type = _extract_gpu_info_from_job(job)
    assert gpus == 4
    assert gpu_type == "H100"


def test_extract_gpu_no_gpus_configured() -> None:
    job: dict = {"config": {"algorithm": "sft"}}
    gpus, gpu_type = _extract_gpu_info_from_job(job)
    assert gpus == 0
    assert gpu_type is None


def test_extract_gpu_empty_config() -> None:
    job: dict = {"config": {}}
    gpus, gpu_type = _extract_gpu_info_from_job(job)
    assert gpus == 0
    assert gpu_type is None


def test_extract_gpu_non_dict_config() -> None:
    job: dict = {"config": "not a dict"}
    gpus, gpu_type = _extract_gpu_info_from_job(job)
    assert gpus == 0
    assert gpu_type is None


def test_estimate_memory_known_gpu_types() -> None:
    assert _estimate_memory_gib("A100", 2) == 160.0
    assert _estimate_memory_gib("t4", 1) == 16.0
    assert _estimate_memory_gib("H100", 1) == 80.0
    assert _estimate_memory_gib("L4", 3) == 72.0


def test_estimate_memory_unknown_type_uses_default() -> None:
    assert _estimate_memory_gib("unknown-gpu", 1) == 16.0


def test_estimate_memory_none_type_uses_default() -> None:
    assert _estimate_memory_gib(None, 2) == 32.0


@pytest.mark.asyncio
async def test_gpu_allocation_no_gpu_backend_no_jobs() -> None:
    """Only local backend, no GPU jobs → available=False."""
    mock_conn = AsyncMock()
    with (
        patch("amortized.api.gpu.get_all_backends", return_value={"local": object()}),
        patch(
            "amortized.api.gpu._get_running_gpu_jobs",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await get_gpu_allocation(conn=mock_conn)
    assert isinstance(result, GpuAllocationResponse)
    assert result.available is False
    assert result.jobs == []


@pytest.mark.asyncio
async def test_gpu_allocation_gpu_backend_no_active_jobs() -> None:
    """K8s backend registered, no active jobs → available=True, zero allocation."""
    mock_conn = AsyncMock()
    with (
        patch(
            "amortized.api.gpu.get_all_backends",
            return_value={"local": object(), "kubernetes": object()},
        ),
        patch(
            "amortized.api.gpu._get_running_gpu_jobs",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await get_gpu_allocation(conn=mock_conn)
    assert result.available is True
    assert result.allocated_gpus == 0
    assert result.total_memory_requested_gib == 0.0
    assert result.jobs == []


@pytest.mark.asyncio
async def test_gpu_allocation_with_active_gpu_jobs() -> None:
    """Active GPU jobs → returns allocation with memory calculated."""
    mock_conn = AsyncMock()
    fake_jobs = [
        {
            "id": "job-1",
            "status": "running",
            "config": {"compute": {"gpus": 2, "gpu_type": "A100"}, "topic": "sentiment"},
        },
        {
            "id": "job-2",
            "status": "queued",
            "config": {"gpus": 1, "gpu_type": "T4", "topic": "classification"},
        },
    ]
    with (
        patch(
            "amortized.api.gpu.get_all_backends",
            return_value={"local": object()},
        ),
        patch(
            "amortized.api.gpu._get_running_gpu_jobs",
            new_callable=AsyncMock,
            return_value=fake_jobs,
        ),
    ):
        result = await get_gpu_allocation(conn=mock_conn)
    assert result.available is True
    assert result.allocated_gpus == 3
    assert result.total_memory_requested_gib == 176.0  # 2xA100(80) + 1xT4(16)
    assert len(result.jobs) == 2
    assert result.jobs[0].gpus_requested == 2
    assert result.jobs[0].memory_requested_gib == 160.0
    assert result.jobs[1].gpus_requested == 1
    assert result.jobs[1].memory_requested_gib == 16.0


@pytest.mark.asyncio
async def test_gpu_allocation_jobs_without_gpu_config_excluded() -> None:
    """Jobs with gpus=0 or no GPU config are excluded from results."""
    mock_conn = AsyncMock()
    fake_jobs = [
        {
            "id": "job-no-gpu",
            "status": "running",
            "config": {"algorithm": "sft", "topic": "test"},
        },
        {
            "id": "job-with-gpu",
            "status": "running",
            "config": {"compute": {"gpus": 1, "gpu_type": "L4"}, "topic": "classify"},
        },
    ]
    with (
        patch("amortized.api.gpu.get_all_backends", return_value={"local": object()}),
        patch(
            "amortized.api.gpu._get_running_gpu_jobs",
            new_callable=AsyncMock,
            return_value=fake_jobs,
        ),
    ):
        result = await get_gpu_allocation(conn=mock_conn)
    assert result.available is True
    assert result.allocated_gpus == 1
    assert len(result.jobs) == 1
    assert result.jobs[0].job_id == "job-with-gpu"


@pytest.mark.asyncio
async def test_gpu_allocation_db_error_returns_unavailable() -> None:
    """Database error → returns available=False with error reason."""
    mock_conn = AsyncMock()
    with (
        patch("amortized.api.gpu.get_all_backends", return_value={"local": object()}),
        patch(
            "amortized.api.gpu._get_running_gpu_jobs",
            new_callable=AsyncMock,
            side_effect=Exception("connection refused"),
        ),
    ):
        result = await get_gpu_allocation(conn=mock_conn)
    assert result.available is False
    assert result.reason == "Cannot reach compute cluster"
