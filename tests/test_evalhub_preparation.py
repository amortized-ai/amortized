"""Eval Hub preparation and results: classifier settings, staged-input cleanup, and the
comparison key of task evaluations."""

import copy
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import amortized.config as config_mod
from amortized.integrations.evalhub import preparation, results

TASK = {
    "endpoint": {"base_url": "http://model/v1", "model": "test"},
    "eval_data_run_id": "a" * 32,
    "evaluation": {"kind": "task"},
    "rubric": [{"name": "match", "description": "Matches the reference"}],
    "judge": {"base_url": "http://judge/v1", "model": "judge"},
}
# Hub's provider record as the SDK dumps it into inputs.catalog.
PROVIDER = {
    "resource": {
        "id": "amortized",
        "created_at": "2026-11-01T00:00:00Z",
        "updated_at": "2026-11-01T00:00:00Z",
    },
    "name": "Amortized task evaluation v1",
    "description": "Versioned deterministic, rubric and anchor/query task scoring",
    "runtime": {
        "k8s": {
            "image": "ghcr.io/amortized-ai/evalhub-adapter@sha256:one",
            "cpu_limit": "4000m",
            "memory_limit": "8Gi",
        }
    },
    "benchmarks": [
        {"id": "task", "name": "User-approved task recipe", "category": "task", "num_few_shot": 0}
    ],
}


# --------------------------------------------------------------------------- #
# Classifier settings
# --------------------------------------------------------------------------- #


async def test_base_classifier_scores_at_the_evals_max_length():
    repo = MagicMock(get_job=AsyncMock())
    config = {"eval_mode": "classification", "max_length": 128}
    assert await preparation._classifier_parameters(config, repo) == {"max_length": 128}
    repo.get_job.assert_not_awaited()


async def test_tuned_classifier_prefers_the_evals_max_length_over_its_training():
    repo = MagicMock(get_job=AsyncMock(return_value={"config": {"max_length": 256}}))
    tuned = {"eval_mode": "classification", "training_job_id": "train-1"}
    assert await preparation._classifier_parameters(tuned, repo) == {
        "max_length": 256,
        "trust_remote_code": False,
    }
    assert await preparation._classifier_parameters({**tuned, "max_length": 128}, repo) == {
        "max_length": 128,
        "trust_remote_code": False,
    }


# --------------------------------------------------------------------------- #
# Staged inputs are deleted at cleanup
# --------------------------------------------------------------------------- #


class FakeS3:
    def __init__(self, keys: set[str]) -> None:
        self.keys = set(keys)

    def get_paginator(self, _: str) -> "FakeS3":
        return self

    def paginate(self, **request: str) -> Any:
        matching = sorted(k for k in self.keys if k.startswith(request["Prefix"]))
        for start in range(0, len(matching), 2):
            yield {"Contents": [{"Key": k} for k in matching[start : start + 2]]}

    def delete_objects(self, **request: Any) -> dict[str, Any]:
        self.keys -= {item["Key"] for item in request["Delete"]["Objects"]}
        return {}


def staging(monkeypatch, s3: Any) -> None:
    settings = config_mod.Settings(evalhub_staging_bucket="snapshots")
    monkeypatch.setattr(config_mod, "settings", settings)
    monkeypatch.setattr(preparation, "_s3", lambda: s3)


async def test_cleanup_deletes_only_the_jobs_staged_inputs(monkeypatch):
    staged = "amortized/evaluations/job-1/inputs/abc/"
    other = "amortized/evaluations/job-10/inputs/def/data/test.jsonl"
    s3 = FakeS3(
        {
            staged + "data/test.jsonl",
            staged + "model/model.safetensors",
            staged + "manifest.json",
            other,
        }
    )
    staging(monkeypatch, s3)
    await preparation.cleanup({"job_id": "job-1"})
    assert s3.keys == {other}
    await preparation.cleanup({"job_id": "job-1"})
    assert s3.keys == {other}


async def test_cleanup_finds_staged_inputs_after_the_hub_settings_are_gone(monkeypatch):
    staged = "amortized/evaluations/job-1/inputs/abc/"
    s3 = FakeS3({staged + "data/test.jsonl", staged + "manifest.json"})
    staging(monkeypatch, s3)
    monkeypatch.setattr(config_mod, "settings", config_mod.Settings())
    inputs = {"data_ref": {"s3": {"bucket": "snapshots", "key": staged}}}
    await preparation.cleanup({"job_id": "job-1", "inputs": inputs})
    assert s3.keys == set()


async def test_cleanup_logs_but_does_not_wait_on_storage(monkeypatch, caplog):
    s3 = MagicMock()
    s3.get_paginator.side_effect = OSError("storage unavailable")
    staging(monkeypatch, s3)
    with caplog.at_level(logging.WARNING):
        await preparation.cleanup({"job_id": "job-1"})
    assert "s3://snapshots/amortized/evaluations/job-1/" in caplog.text


# --------------------------------------------------------------------------- #
# Task comparisons key on the adapter build, not the whole provider record
# --------------------------------------------------------------------------- #


async def test_task_comparisons_ignore_provider_timestamps_and_resources(monkeypatch):
    execution = {
        "snapshot": {
            "status": {"state": "completed"},
            "results": {"benchmarks": [{"id": "task", "metrics": {"match": 0.8}}]},
        },
        "request": {"benchmarks": [{"id": "task", "provider_id": "amortized"}]},
        "inputs": {"dataset_identity": "sha256:dataset", "catalog": PROVIDER},
    }
    executions = {"a": execution, "b": copy.deepcopy(execution)}
    store = MagicMock(get=AsyncMock(side_effect=lambda job_id: executions[job_id]))
    monkeypatch.setattr(results, "EvalHubRepository", lambda _: store)
    jobs = [
        {"id": job_id, "config": TASK, "status": "succeeded", "created_at": "now"}
        for job_id in "ab"
    ]
    provider = executions["b"]["inputs"]["catalog"]
    provider["resource"].update(updated_at="2026-11-18T09:30:00Z")
    provider["runtime"]["k8s"].update(cpu_limit="8000m", memory_limit="16Gi")
    provider["description"] = "Reworded"
    [group] = await results.comparison_groups(MagicMock(), jobs)
    assert len(group["evals"]) == 2
    provider["runtime"]["k8s"]["image"] = "ghcr.io/amortized-ai/evalhub-adapter@sha256:two"
    assert len(await results.comparison_groups(MagicMock(), jobs)) == 2
