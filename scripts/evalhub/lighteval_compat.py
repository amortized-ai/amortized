"""Release helper: record what each LightEval catalog entry needs, read from the pinned adapter.

Runs INSIDE the pinned adapter image, so every answer comes from the code that will
run the benchmark, never from a benchmark's name or category. See `make evalhub-compat`.
Writes JSON to stdout. Anything it cannot establish is recorded as "unknown" with a reason.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hf_identity import dataset

SCHEMA = "amortized.evalhub.compat/v1"
# Names the parameter contract Amortized implements for this adapter
# (amortized.evaluation.spec.LightEvalParameters). Bump it when the adapter's
# accepted parameters change, so a stale Amortized treats entries as unknown.
SETTINGS_PROFILE = "lighteval-endpoint-v1"


def describe(benchmark_id: str, adapter: Any, configs: dict[str, Any], methods: Any) -> dict:
    cls = adapter.LightEvalAdapter
    suite = benchmark_id in cls.SUPPORTED_TASKS
    # The adapter's own resolution: suite expansion, then custom-task aliases.
    tasks = cls._parse_benchmark_tasks(cls, benchmark_id, {})
    record: dict[str, Any] = {"tasks": tasks, "suite": suite}
    missing = [t for t in tasks if configs.get(t.split("|", 1)[0]) is None]
    if missing:
        reason = f"LightEval's task registry has no task {', '.join(missing)}"
        return {**record, "status": "unknown", "reason": reason}
    used = {m.category for t in tasks for m in configs[t.split("|", 1)[0]].metrics}
    if used == {methods.GENERATIVE}:
        scoring = "generative"
    elif methods.LOGPROBS in used and used <= {methods.GENERATIVE, methods.LOGPROBS}:
        # Same rule as the adapter's _loglikelihood_tasks(): any LOGPROBS metric
        # needs /v1/completions with echo and logprobs.
        scoring = "likelihood"
    else:
        names = ", ".join(sorted(getattr(m, "name", str(m)) for m in used))
        return {**record, "status": "unknown", "reason": f"unrecognized sampling methods: {names}"}
    datasets = []
    for task in tasks:
        config = configs[task.split("|", 1)[0]]
        datasets.append(
            dataset(
                task,
                getattr(config, "hf_repo", None),
                getattr(config, "hf_subset", None),
                [str(s) for s in getattr(config, "evaluation_splits", None) or ()],
                getattr(config, "hf_revision", None),
            )
        )
    return {
        **record,
        "status": "known",
        "scoring": scoring,
        "answers_per_question": cls._max_samples_per_prompt(tasks),
        "settings_profile": SETTINGS_PROFILE,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Eval Hub provider catalog YAML")
    parser.add_argument("--catalog-repo", default="eval-hub/eval-hub")
    parser.add_argument("--catalog-commit", required=True)
    parser.add_argument("--adapter-image", required=True, help="image@sha256:... being described")
    parser.add_argument("--adapter-dir", default="/app", help="directory holding the adapter's main.py")
    args = parser.parse_args()
    if "@sha256:" not in args.adapter_image:
        parser.error("--adapter-image must be pinned by digest")

    sys.path.insert(0, args.adapter_dir)
    import main as adapter  # the pinned adapter's own module
    import yaml
    from lighteval.tasks.requests import SamplingMethod

    configs = adapter._task_configs()
    catalog = yaml.safe_load(Path(args.catalog).read_text())
    benchmarks = {}
    for entry in catalog["benchmarks"]:
        try:
            benchmarks[entry["id"]] = describe(entry["id"], adapter, configs, SamplingMethod)
        except Exception as exc:  # record, never guess
            benchmarks[entry["id"]] = {"status": "unknown", "reason": f"helper error: {exc}"}
    try:
        from importlib.metadata import version

        lighteval_version = version("lighteval")
    except Exception:
        lighteval_version = "unknown"
    json.dump(
        {
            "schema": SCHEMA,
            "catalog": {"repo": args.catalog_repo, "commit": args.catalog_commit},
            "adapter_image": args.adapter_image,
            "adapter_digest": args.adapter_image.split("@", 1)[1],
            # The adapter-wide contract, also for entries whose own record is unknown.
            "settings_profile": SETTINGS_PROFILE,
            "lighteval_version": lighteval_version,
            "benchmarks": benchmarks,
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
