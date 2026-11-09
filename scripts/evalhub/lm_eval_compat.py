"""Release helper: record what each LM Evaluation Harness catalog entry needs.

Runs INSIDE the pinned TrustyAI lm-eval adapter image, so every answer comes from the
lm-eval task configs that will run the benchmark, never from a benchmark's name or
category. See `make evalhub-describe`. Writes JSON to stdout. Anything it cannot
establish is recorded as "unknown" with a reason. Metric names are not predicted here:
`make evalhub-conform` records what the adapter reports.

The adapter (main.py) always uses lm-eval's `local-completions` backend: every request
goes to /v1/completions, and it loads the model's Hugging Face tokenizer. Benchmark IDs are
mapped to lm-eval tasks and task configs are loaded through main.py's own functions, so the
records follow the adapter's aliases and its code-execution allow list.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hf_identity import dataset

SCHEMA = "amortized.evalhub.described/v1"
# Names the parameter contract Amortized implements for this adapter
# (amortized.evaluation.spec.LMEvalParameters). Bump it when the adapter's accepted
# parameters change, so a stale Amortized treats entries as unknown.
SETTINGS_PROFILE = "lm-eval-completions-v1"
# lm-eval output types: multiple_choice and loglikelihood* score the answer's log
# probability (echo + logprobs on /v1/completions); generate_until reads generated text.
LIKELIHOOD = {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}
GENERATIVE = {"generate_until"}
# What Amortized sends (preparation.py adds the tokenizer); main.py rejects a benchmark
# that needs more, e.g. LongBench without max_length.
SENT_PARAMETERS = {"batch_size": 1, "num_few_shot": 0, "tokenizer": "org/model"}


def _subtasks(config: dict[str, Any]) -> list[str]:
    tasks = config.get("task") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    return [t if isinstance(t, str) else str(t.get("task") or t.get("group")) for t in tasks]


def directions(config: dict[str, Any]) -> dict[str, bool]:
    """higher_is_better per metric the task config computes, under the config's own names."""
    from lm_eval.api.registry import DEFAULT_METRIC_REGISTRY, is_higher_better

    items = (
        config.get("metric_list")
        or DEFAULT_METRIC_REGISTRY[config.get("output_type") or "generate_until"]
    )
    found: dict[str, bool] = {}
    for item in items:
        metric = item.get("metric") if isinstance(item, dict) else item
        name = str(getattr(metric, "__name__", None) or metric)
        higher = item.get("higher_is_better") if isinstance(item, dict) else None
        if higher is None:
            try:
                higher = is_higher_better(name)
            except Exception:
                higher = None
        if higher is not None:
            found[name] = bool(higher)
    return found


def max_few_shot(config: dict[str, Any]) -> int | None:
    """How many worked examples the task config holds, when it lists them itself.

    BBH's chain-of-thought tasks list 3, so asking for more fails in lm-eval ("number of
    fewshot samples requested exceeds the 3 that are available"). None when the examples
    come from a dataset split.
    """
    samples = (config.get("fewshot_config") or {}).get("samples")
    if callable(samples):
        try:
            samples = samples()
        except Exception:
            return None
    return len(samples) if isinstance(samples, list) else None


def load_adapter(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("lmes_adapter", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(benchmark_id: str, manager: Any, adapter: Any) -> dict[str, Any]:
    task = adapter._resolve_lmeval_task(benchmark_id, manager)  # raises for tags, as in a job
    adapter.build_lmeval_config(
        SimpleNamespace(
            benchmark_id=benchmark_id,
            model=SimpleNamespace(name="org/model", url="http://model/v1"),
            parameters=dict(SENT_PARAMETERS),
        )
    )
    index = manager.task_index.get(task)
    if index is None:
        return {"tasks": [task], "status": "unknown", "reason": "lm-eval has no such task"}
    kind = index["type"]
    if kind == "group":
        tasks = _subtasks(manager._get_config(task))
        return {"tasks": tasks, "suite": True, "status": "known", "reason": "lm-eval group"}
    record: dict[str, Any] = {"tasks": [task], "suite": False}
    if kind != "task":
        return {**record, "status": "unknown", "reason": f"lm-eval {kind} (Python class task)"}
    with adapter._code_eval_environment(benchmark_id):
        config = manager._get_config(task)
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
    examples = max_few_shot(config)
    return {
        **record,
        "status": "known",
        "scoring": scoring,
        "answers_per_question": int(config.get("repeats") or 1),
        **({"max_few_shot": examples} if examples is not None else {}),
        "settings_profile": SETTINGS_PROFILE,
        "datasets": [
            dataset(
                task,
                config["dataset_path"],
                config.get("dataset_name"),
                [str(split)],
                kwargs.get("revision") if isinstance(kwargs, dict) else None,
            )
        ],
        "directions": directions(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Eval Hub provider catalog YAML")
    parser.add_argument("--catalog-repo", default="eval-hub/eval-hub")
    parser.add_argument("--catalog-commit", required=True)
    parser.add_argument("--adapter-image", required=True, help="image@sha256:... being described")
    parser.add_argument("--adapter-main", default="/opt/app-root/src/main.py")
    args = parser.parse_args()
    if "@sha256:" not in args.adapter_image:
        parser.error("--adapter-image must be pinned by digest")

    from importlib.metadata import version

    import yaml

    out, sys.stdout = sys.stdout, sys.stderr  # main.py logs to stdout; keep it for the JSON
    adapter = load_adapter(args.adapter_main)
    manager = adapter.TaskManager()  # the adapter's own task index (main.py)
    catalog = yaml.safe_load(Path(args.catalog).read_text())
    benchmarks = {}
    for entry in catalog["benchmarks"]:
        try:
            benchmarks[entry["id"]] = describe(entry["id"], manager, adapter)
        except Exception as exc:  # record, never guess; the adapter would hit the same error
            # First sentence-like line: skips banners such as "!!!WARNING!!!" and "####".
            lines = [ln.strip() for ln in str(exc).splitlines() if any(c.islower() for c in ln)]
            reason = f"the adapter fails on it: {type(exc).__name__}: {lines[0] if lines else ''}"
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
        out,
        indent=2,
        sort_keys=True,
    )
    out.write("\n")


if __name__ == "__main__":
    main()
