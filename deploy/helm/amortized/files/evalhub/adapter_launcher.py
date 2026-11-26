"""Run a pinned Eval Hub adapter's main.py with overall_score set to the catalog's metric.

Provider entrypoint: python -c "$AMORTIZED_ADAPTER_LAUNCHER" /path/to/main.py

The TrustyAI lm-eval and LightEval adapters report the first metric they collect as
overall_score, and that value reaches their log, MLflow and the additional_info they send
to Hub. Hub, the Eval Hub SDK and Amortized score a benchmark by the metric named in the
job spec's primary_score, matched by exact name. This loads main.py, makes overall_score
that metric's value (none when the adapter did not report it), then runs main.py's main().
"""

import importlib.util
import json
import os
import sys

from evalhub.adapter import FrameworkAdapter

path = os.path.abspath(sys.argv[1])
with open(os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")) as spec_file:
    metric = (json.load(spec_file).get("primary_score") or {}).get("metric") or ""
sys.argv = [path]
sys.path[0] = os.path.dirname(path)  # as when Python runs main.py itself
spec = importlib.util.spec_from_file_location("adapter_main", path)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


def primary(results):
    value = next((r.metric_value for r in results if r.metric_name == metric), None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def scored(run):
    def run_benchmark_job(self, config, callbacks):
        results = run(self, config, callbacks)
        results.overall_score = primary(results.results)
        # lm-eval keeps its additional_info in _run_info; LightEval puts it on the results.
        for info in (getattr(self, "_run_info", None), results.additional_info):
            if not isinstance(info, dict):
                continue
            if "primary_metric" in info:
                info["primary_metric"] = metric or None
            key = "alt_prompting" if info.get("alt_prompting_description") else "zero_shot"
            if key in info:
                info[key] = results.overall_score
        return results

    return run_benchmark_job


for value in list(vars(adapter).values()):
    if (
        isinstance(value, type)
        and issubclass(value, FrameworkAdapter)
        and value.__module__ == spec.name
    ):
        value.run_benchmark_job = scored(value.run_benchmark_job)
sys.exit(adapter.main())
