"""Eval Hub adapter for Amortized's task evaluations: rubric judge or embedding classification."""

import asyncio
import importlib.util
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from evalhub.adapter import DefaultCallbacks, FrameworkAdapter
from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials
from evalhub.adapter.mlflow import MlflowFileArtifact
from evalhub.adapter.models import JobCallbacks, JobResults, JobSpec, JobStatusUpdate
from evalhub.adapter.models.job import MessageInfo
from evalhub.models import EvaluationResult, JobPhase, JobStatus

from amortized.evaluation.generation import run

TASK_DATA_PATH = "/test_data/data"


class AmortizedTaskAdapter(FrameworkAdapter):
    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        started = time.monotonic()
        output = Path(tempfile.mkdtemp(prefix="amortized-results-"))
        self.output_dir = output
        parameters = dict(config.parameters)
        parameters.update(eval_data_path=TASK_DATA_PATH, output_dir=str(output))
        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
        )
        credentials = resolve_model_credentials()
        if parameters.get("eval_mode") == "classification":
            if credentials.hf_token:
                os.environ["HF_TOKEN"] = credentials.hf_token
            asset = Path(__file__).parents[1] / "jobs/assets/classify_eval.py"
            spec = importlib.util.spec_from_file_location("amortized_embedding_scorer", asset)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parameters["model_path"] = config.model.name
            summary = module.run(parameters)["results"]
            summary["metric_metadata"] = {
                name: {
                    "unit": "fraction",
                    "direction": "higher",
                    "range": [0, 1],
                    "num_samples": summary["num_queries"],
                    "scorer_id": "embedding_anchor_query",
                    "scorer_version": "1",
                }
                for name in ("accuracy", "macro_f1")
            }
        else:
            os.environ["EVAL_MODEL_API_KEY"] = credentials.api_key or "token:"
            parameters["endpoints"] = {
                "model": {
                    "base_url": config.model.url,
                    "model": config.model.name,
                    "api_key_env": "EVAL_MODEL_API_KEY",
                }
            }
            if parameters.get("judge_model"):
                judge_url = read_model_auth_key("judge_url")
                judge_key = read_model_auth_key("judge_api-key")
                if not judge_url or not judge_key:
                    raise ValueError(
                        "Judge requires a projected judge_url and judge_api-key reference"
                    )
                os.environ["EVAL_JUDGE_API_KEY"] = judge_key
                parameters["endpoints"]["judge"] = {
                    "base_url": judge_url,
                    "model": parameters["judge_model"],
                    "api_key_env": "EVAL_JUDGE_API_KEY",
                }
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION)
            )
            summary = asyncio.run(run(parameters))["results"]
            if summary["model"]["error_rate"] > 0.5:
                raise RuntimeError("Model errored on more than half of evaluation samples")
        # Re-emit classification metadata too, keeping detailed output in the same run.
        (output / "metrics.json").write_text(json.dumps({"results": summary}))
        self.mlflow_artifacts = [
            MlflowFileArtifact(f"eval_results/{name}", output / name)
            for name in ("metrics.json", "results.jsonl")
        ]
        count = int(summary.get("model", {}).get("num_samples", summary.get("num_queries", 0)))
        metadata = summary.get("metric_metadata") or {}
        results = [
            EvaluationResult(
                metric_name=name,
                metric_value=value,
                num_samples=summary.get("scores_n", {}).get(name, count),
                metadata=metadata.get(name, {}),
            )
            for name, value in (summary.get("scores") or {}).items()
            if value is not None
        ]
        return JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name,
            results=results,
            num_examples_evaluated=count,
            duration_seconds=time.monotonic() - started,
            evaluation_metadata={
                "dataset_identity": parameters.get("dataset_identity"),
                "recipe": config.parameters,
                "artifacts": {
                    "amortized_metrics_path": "eval_results/metrics.json",
                    "amortized_samples_path": "eval_results/results.jsonl",
                },
            },
            additional_info={"amortized_results": summary},
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    adapter = AmortizedTaskAdapter()
    callbacks = DefaultCallbacks.from_adapter(adapter)
    try:
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS)
        )
        results.mlflow_run_id = callbacks.mlflow.save(
            results, adapter.job_spec, artifacts=adapter.mlflow_artifacts
        )
        if not results.mlflow_run_id:
            raise RuntimeError("Task evaluation requires persisted sample artifacts")
        callbacks.report_results(results)
    except (Exception, SystemExit):
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.FAILED,
                error_message=MessageInfo(
                    message="Task evaluation failed; inspect adapter logs",
                    message_code="amortized_task_failed",
                ),
            )
        )
        raise


if __name__ == "__main__":
    main()
