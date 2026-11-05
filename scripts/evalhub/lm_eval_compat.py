"""Release helper: record what each LM Evaluation Harness catalog entry needs.

Runs INSIDE the pinned TrustyAI lm-eval adapter image, so every answer comes from the
lm-eval task configs that will run the benchmark, never from a benchmark's name or
category. See `make evalhub-compat`. Writes JSON to stdout. Anything it cannot
establish is recorded as "unknown" with a reason.

The adapter (main.py) always uses lm-eval's `local-completions` backend: every request
goes to /v1/completions, and it loads the model's Hugging Face tokenizer.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hf_identity import dataset

SCHEMA = "amortized.evalhub.compat/v1"
# Names the parameter contract Amortized implements for this adapter
# (amortized.evaluation.spec.LMEvalParameters). Bump it when the adapter's accepted
# parameters change, so a stale Amortized treats entries as unknown.
SETTINGS_PROFILE = "lm-eval-completions-v1"
# lm-eval output types: multiple_choice and loglikelihood* score the answer's log
# probability (echo + logprobs on /v1/completions); generate_until reads generated text.
LIKELIHOOD = {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}
GENERATIVE = {"generate_until"}


def _subtasks(config: dict[str, Any]) -> list[str]:
    tasks = config.get("task") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    return [t if isinstance(t, str) else str(t.get("task") or t.get("group")) for t in tasks]


def describe(benchmark_id: str, manager: Any) -> dict[str, Any]:
    index = manager.task_index.get(benchmark_id)
    if index is None:
        return {"tasks": [benchmark_id], "status": "unknown", "reason": "lm-eval has no such task"}
    kind = index["type"]
    if kind in {"group", "tag"}:
        tasks = list(index.get("task") or []) if kind == "tag" else []
        if kind == "group":
            tasks = _subtasks(manager._get_config(benchmark_id))
        return {"tasks": tasks, "suite": True, "status": "known", "reason": f"lm-eval {kind}"}
    record: dict[str, Any] = {"tasks": [benchmark_id], "suite": False}
    if kind != "task":
        return {**record, "status": "unknown", "reason": f"lm-eval {kind} (Python class task)"}
    config = manager._get_config(benchmark_id)
    output_type = config.get("output_type") or "generate_until"
    if output_type in LIKELIHOOD:
        scoring = "likelihood"
    elif output_type in GENERATIVE:
        scoring = "generative"
    else:
        return {**record, "status": "unknown", "reason": f"unrecognized output_type {output_type}"}
    if not config.get("dataset_path"):
        return {**record, "status": "unknown", "reason": "the task config names no dataset"}
    # lm-eval scores the test split when the task has one, else validation.
    split = config.get("test_split") or config.get("validation_split")
    if not split:
        return {**record, "status": "unknown", "reason": "the task config names no test split"}
    kwargs = config.get("dataset_kwargs") or {}
    return {
        **record,
        "status": "known",
        "scoring": scoring,
        "answers_per_question": int(config.get("repeats") or 1),
        "settings_profile": SETTINGS_PROFILE,
        "datasets": [
            dataset(
                benchmark_id,
                config["dataset_path"],
                config.get("dataset_name"),
                [str(split)],
                kwargs.get("revision") if isinstance(kwargs, dict) else None,
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Eval Hub provider catalog YAML")
    parser.add_argument("--catalog-repo", default="eval-hub/eval-hub")
    parser.add_argument("--catalog-commit", required=True)
    parser.add_argument("--adapter-image", required=True, help="image@sha256:... being described")
    args = parser.parse_args()
    if "@sha256:" not in args.adapter_image:
        parser.error("--adapter-image must be pinned by digest")

    from importlib.metadata import version

    import yaml
    from lm_eval.tasks import TaskManager

    manager = TaskManager()  # the adapter's own task index (main.py)
    catalog = yaml.safe_load(Path(args.catalog).read_text())
    benchmarks = {}
    for entry in catalog["benchmarks"]:
        try:
            benchmarks[entry["id"]] = describe(entry["id"], manager)
        except Exception as exc:  # record, never guess; the adapter would hit the same error
            # First sentence-like line: skips banners such as "!!!WARNING!!!" and "####".
            lines = [ln.strip() for ln in str(exc).splitlines() if any(c.islower() for c in ln)]
            reason = f"loading it fails: {type(exc).__name__}: {lines[0] if lines else ''}"
            benchmarks[entry["id"]] = {"status": "unknown", "reason": reason[:300]}
    json.dump(
        {
            "schema": SCHEMA,
            "catalog": {"repo": args.catalog_repo, "commit": args.catalog_commit},
            "adapter_image": args.adapter_image,
            "adapter_digest": args.adapter_image.split("@", 1)[1],
            # The adapter-wide contract, also for entries whose own record is unknown.
            "settings_profile": SETTINGS_PROFILE,
            "lm_eval_version": version("lm_eval"),
            "benchmarks": benchmarks,
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
