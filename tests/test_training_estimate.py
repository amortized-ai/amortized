"""VRAM estimation: api/costs.py builds the request, core/training_estimate.py runs
training-hub's estimator (jobs/assets/estimate_vram.py) as a job in the training image."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from amortized import config as config_mod
from amortized.api import costs
from amortized.api.costs import TrainingEstimateRequest, estimate_training_resources
from amortized.backends import BackendHandle, BackendStatus, Capability, JobSpec
from amortized.core import compute, training_estimate
from amortized.core.training_estimate import TrainingEstimateError, run_training_estimate

GIB = 1024**3

RESULT = {
    "training_hub_version": "0.11.2",
    "num_params": 8_392_695_024,
    "single_gpu": [int(68.2 * GIB), int(75.0 * GIB), int(88.7 * GIB)],
    "per_gpu": {"1": int(75.0 * GIB), "2": int(50 * GIB), "4": int(40 * GIB), "8": int(35 * GIB)},
}


class TestEndpoint:
    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []

        async def _fake(request: dict[str, Any]) -> dict[str, Any]:
            seen.append(request)
            return RESULT

        monkeypatch.setattr(costs, "run_training_estimate", _fake)
        return seen

    async def _estimate(self, **kwargs: Any) -> costs.TrainingEstimateResponse:
        return await estimate_training_resources(TrainingEstimateRequest(**kwargs))

    async def test_response_from_training_hub_result(self, calls: list[dict[str, Any]]) -> None:
        resp = await self._estimate(model_name_or_path="Qwen/Qwen3.5-9B", method="osft")
        assert resp.model_size == "8.4B"
        assert resp.arch_source == "training-hub 0.11.2"
        assert resp.vram_gb == {"low": 68.2, "expected": 75.0, "high": 88.7}
        assert resp.per_gpu_vram_gb_by_gpu_count == {"1": 75.0, "2": 50.0, "4": 40.0, "8": 35.0}

    async def test_reports_gpu_budget(
        self, calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config_mod.settings, "gpu_budget", 3)
        resp = await self._estimate(model_name_or_path="Qwen/Qwen3-8B")
        assert resp.gpu_budget == 3

    async def test_lora_uses_per_step_batch(self, calls: list[dict[str, Any]]) -> None:
        # Job dceac892: effective_batch_size 32 with gradient_accumulation_steps 8 runs 4/step.
        await self._estimate(
            model_name_or_path="Qwen/Qwen3.5-9B",
            method="lora_sft",
            effective_batch_size=32,
            gradient_accumulation_steps=8,
            max_seq_len=8192,
            lora_r=16,
        )
        assert calls[0] == {
            "model": "Qwen/Qwen3.5-9B",
            "method": "lora",
            "batch_size": 4,
            "max_seq_len": 8192,
            "max_tokens_per_gpu": 4096,
            "lora_r": 16,
            "unfreeze_rank_ratio": 0.2,
            "gpu_counts": [1, 2, 4, 8],
        }

    async def test_sft_osft_keep_full_batch_and_defaults(self, calls: list[dict[str, Any]]) -> None:
        await self._estimate(
            model_name_or_path=" Qwen/Qwen3-8B ", method="OSFT", gradient_accumulation_steps=4
        )
        req = calls[0]
        assert req["model"] == "Qwen/Qwen3-8B"
        assert req["method"] == "osft"
        assert req["batch_size"] == 8
        assert (req["max_seq_len"], req["max_tokens_per_gpu"]) == (2048, 4096)

    @pytest.mark.parametrize(
        ("method", "expected"),
        [("sft", "sft"), ("osft", "osft"), ("lora", "lora"), ("qlora_sft", "qlora")],
    )
    async def test_method_aliases(
        self, calls: list[dict[str, Any]], method: str, expected: str
    ) -> None:
        await self._estimate(model_name_or_path="Qwen/Qwen3-8B", method=method)
        assert calls[0]["method"] == expected

    async def test_unknown_method_does_not_start_a_job(self, calls: list[dict[str, Any]]) -> None:
        with pytest.raises(HTTPException) as exc:
            await self._estimate(model_name_or_path="Qwen/Qwen3-8B", method="dpo")
        assert exc.value.status_code == 400
        assert calls == []

    async def test_job_failure_is_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fail(request: dict[str, Any]) -> dict[str, Any]:
            raise TrainingEstimateError("gated repo")

        monkeypatch.setattr(costs, "run_training_estimate", _fail)
        with pytest.raises(HTTPException) as exc:
            await self._estimate(model_name_or_path="org/private")
        assert exc.value.status_code == 502
        assert "gated repo" in exc.value.detail

    def test_requires_a_model(self) -> None:
        with pytest.raises(ValidationError):
            TrainingEstimateRequest(model_name_or_path="")


class FakeBackend:
    name = "fake"

    def __init__(self, logs: list[str], exit_code: int = 0, polls_running: int = 1) -> None:
        self._logs = logs
        self._exit_code = exit_code
        self._polls_running = polls_running
        self.spec: JobSpec | None = None
        self.cancelled = False

    def capabilities(self) -> set[Capability]:
        return set()

    async def submit(self, spec: JobSpec) -> BackendHandle:
        self.spec = spec
        return BackendHandle(backend_name=self.name, job_id=spec.job_id, scheduler_id="x")

    async def status(self, handle: BackendHandle) -> BackendStatus:
        if self._polls_running > 0:
            self._polls_running -= 1
            return BackendStatus(running=True)
        return BackendStatus(running=False, exit_code=self._exit_code)

    async def cancel(self, handle: BackendHandle) -> None:
        self.cancelled = True

    async def logs(self, handle: BackendHandle) -> AsyncIterator[str]:
        for line in self._logs:
            yield line

    def reconstruct_handle(self, job_id: str) -> BackendHandle | None:
        return None


class TestRunTrainingEstimate:
    @pytest.fixture(autouse=True)
    def _settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config_mod.settings, "compute_backend", "fake")
        monkeypatch.setattr(config_mod.settings, "default_backend", "")
        monkeypatch.setattr(config_mod.settings, "training_image_tag", "estimator-dev")
        monkeypatch.setattr(config_mod.settings, "forward_env", ["HF_TOKEN"])
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        monkeypatch.setattr(training_estimate, "_POLL_S", 0)

    def _use(self, backend: FakeBackend) -> FakeBackend:
        compute.register_backend(backend)
        return backend

    @pytest.fixture(autouse=True)
    def _cleanup(self) -> Iterator[None]:
        yield
        compute.unregister_backend("fake")

    async def test_runs_script_in_training_image(self) -> None:
        backend = self._use(FakeBackend(["loading...", json.dumps(RESULT)]))
        request = {"model": "Qwen/Qwen3-8B", "method": "sft"}
        assert await run_training_estimate(request) == RESULT

        spec = backend.spec
        assert spec is not None
        assert spec.image == "ghcr.io/amortized-ai/training:estimator-dev"
        assert spec.command == ["python3", "/amortized/estimate_vram.py", json.dumps(request)]
        assert "training_hub.profiling" in spec.config_files["estimate_vram.py"]
        assert spec.resources.gpus == 0
        assert spec.env == {"HF_TOKEN": "hf_test"}
        assert backend.cancelled

    async def test_nonzero_exit_reports_log_tail(self) -> None:
        backend = self._use(FakeBackend(["OSError: gated repo", ""], exit_code=1))
        with pytest.raises(TrainingEstimateError, match="gated repo"):
            await run_training_estimate({})
        assert backend.cancelled

    async def test_missing_result_line(self) -> None:
        self._use(FakeBackend(["no json here"]))
        with pytest.raises(TrainingEstimateError, match="no json here"):
            await run_training_estimate({})

    async def test_skips_non_result_brace_lines_after_result(self) -> None:
        self._use(
            FakeBackend(
                [json.dumps(RESULT), "--- stderr ---", "{'not': 'json'}", json.dumps({"x": 1})]
            )
        )
        assert await run_training_estimate({}) == RESULT

    async def test_incomplete_result_is_an_estimate_error(self) -> None:
        partial = {k: v for k, v in RESULT.items() if k != "per_gpu"}
        self._use(FakeBackend([json.dumps(partial)]))
        with pytest.raises(TrainingEstimateError, match="estimate job failed"):
            await run_training_estimate({})

    async def test_timeout_cleans_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(training_estimate, "_TIMEOUT_S", -1)
        backend = self._use(FakeBackend([], polls_running=10))
        with pytest.raises(TrainingEstimateError, match="timed out"):
            await run_training_estimate({})
        assert backend.cancelled
