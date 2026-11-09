"""Run every benchmark a served model can take through Eval Hub, as a user's job runs it.

    python -m amortized.integrations.evalhub.conformance --provider amortized-lm-eval \\
        --described described.json --evidence evidence.json      (make evalhub-conform)

Runs in the server pod, with Amortized's own code: one model server for the whole run
(preparation.prepare_serving), requests built like a user's (catalog.resolve_recipe,
preparation.benchmark_selection, preparation.hub_request), Hub jobs of --chunk benchmarks.
It records what each adapter reported, so `make evalhub-contract` can pick each score from
real names. An entry first runs with upstream's metric as its primary_score; when the
contract's rule picks another metric (or none), it runs again with that, so the evidence
shows the whole score chain (adapter, Hub, MLflow) for the score users will see.

Resumes: entries already recorded for this adapter digest and model are skipped unless
--rerun. Each record keeps the question count it ran with: an entry whose scoring needs all
its questions (tinyTruthfulQA) runs again with --only <id> --rerun --questions 0. The
evidence file is rewritten after every Hub job.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient
from amortized.integrations.evalhub import catalog, client, contract, lifecycle, preparation
from amortized.jobs.base import JobBuildError

MODEL = "Qwen/Qwen3-0.6B"
EXPERIMENT = "amortized/conformance"
_TERMINAL = {"completed", "failed", "partially_failed", "cancelled"}
_REPORTED = {"completed", "failed", "cancelled"}


def _log(message: str) -> None:
    print(f"{datetime.now(UTC):%H:%M:%S} {message}", file=sys.stderr, flush=True)


class _Store:
    """The slice of EvalHubRepository that serving and credential preparation write to."""

    def __init__(self, execution: dict[str, Any]) -> None:
        self.execution = execution

    async def update(self, job_id: str, **fields: Any) -> None:
        self.execution.update(fields)


def _score(primary: Any) -> dict[str, Any] | None:
    if not isinstance(primary, dict) or not primary.get("metric"):
        return None
    return {
        "metric": str(primary["metric"]),
        "lower_is_better": bool(primary.get("lower_is_better")),
    }


def _error(status: dict[str, Any]) -> str:
    for key in ("error_message", "error", "message"):
        value = status.get(key)
        if isinstance(value, dict):
            value = value.get("message")
        if value:
            return str(value)
    return ""


async def _mlflow(run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    try:
        run = await MLflowClient(config_mod.settings.mlflow_tracking_uri).get_run(run_id)
    except Exception as exc:
        return {"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
    metrics = {m["key"]: m["value"] for m in (run.get("data") or {}).get("metrics") or []}
    return {
        "run_id": run_id,
        "status": (run.get("info") or {}).get("status"),
        "overall_score": metrics.get("overall_score"),
    }


class Runner:
    def __init__(self, args: argparse.Namespace, described: dict[str, Any]) -> None:
        self.args = args
        self.described = described
        settings = config_mod.settings
        self.tenant = settings.evalhub_tenant or settings.compute_namespace
        short = args.provider.removeprefix("amortized-")
        self.job: dict[str, Any] = {
            "id": f"conformance-{short}",
            "user_id": "",
            "config": {"model_name_or_path": args.model},
        }
        self.serving: dict[str, Any] = {}
        self.open: dict[str, Any] | None = None  # the Hub job in flight, cancelled on exit

    async def serve(self) -> tuple[dict[str, Any], str]:
        store = _Store(self.serving)
        deadline = time.monotonic() + self.args.serve_timeout
        while True:
            try:
                endpoint = await preparation.prepare_serving(self.job, store, self.serving)  # type: ignore[arg-type]
                break
            except (httpx.HTTPError, ConnectionError) as exc:
                if time.monotonic() > deadline:
                    raise JobBuildError(f"the model server did not start: {exc}") from exc
                _log(f"waiting for the model server: {exc}")
                await asyncio.sleep(15)
        secret = await preparation._credential_secret(self.job, store, endpoint, None)  # type: ignore[arg-type]
        return endpoint, secret

    async def cleanup(self) -> None:
        if self.open is not None:
            with suppress(Exception):
                await client.cancel(self.open)
        for _ in range(40):
            try:
                await preparation.cleanup(self.serving)
                return
            except ConnectionError:
                await asyncio.sleep(5)
        _log("the model server is still stopping; check for leftover resources")

    async def run_chunk(
        self,
        items: list[tuple[catalog.Entry, dict[str, Any], dict[str, Any] | None, int]],
        endpoint: dict[str, Any],
        secret: str,
    ) -> list[dict[str, Any]]:
        correlation = f"amortized-conformance-{uuid.uuid4().hex[:12]}"
        execution: dict[str, Any] = {
            "hub_url": config_mod.settings.evalhub_url,
            "tenant": self.tenant,
            "correlation": correlation,
        }
        selected = []
        for _entry, recipe, sent, _attempt in items:
            benchmark = preparation.benchmark_selection(recipe, {})
            benchmark.pop("primary_score", None)
            if sent:
                benchmark["primary_score"] = sent
            if preparation.needs_tokenizer(recipe):
                benchmark["parameters"]["tokenizer"] = self.args.model
            selected.append(benchmark)
        payload = preparation.hub_request(
            correlation, correlation, endpoint, secret, selected, EXPERIMENT
        )
        started = time.monotonic()
        snapshot = await client.submit(payload, execution)
        execution["remote_id"] = snapshot["resource"]["id"]
        self.open = execution
        stopped: dict[int, str] = {}
        timed_out = False
        while (snapshot.get("status") or {}).get("state") not in _TERMINAL:
            await asyncio.sleep(self.args.poll)
            snapshot = await client.observe(execution)
            status = snapshot.get("status") or {}
            if status.get("state") in _TERMINAL:
                break
            reported = {
                b.get("benchmark_index")
                for b in status.get("benchmarks") or []
                if b.get("status") in _REPORTED
            }
            try:
                stopped = await lifecycle.stopped_adapters(execution, snapshot)
            except Exception as exc:
                _log(f"cannot watch the adapter Jobs: {exc}")
            if -1 in stopped or all(i in reported or i in stopped for i in range(len(items))):
                break
            if time.monotonic() - started > self.args.timeout:
                timed_out = True
                break
        if (snapshot.get("status") or {}).get("state") not in _TERMINAL:
            with suppress(Exception):
                await client.cancel(execution)
        self.open = None
        seconds = round(time.monotonic() - started)
        statuses = {
            b.get("benchmark_index"): b
            for b in (snapshot.get("status") or {}).get("benchmarks") or []
        }
        results = {
            b.get("benchmark_index"): b
            for b in (snapshot.get("results") or {}).get("benchmarks") or []
        }
        records = []
        for index, (_entry, _recipe, sent, attempt) in enumerate(items):
            status = statuses.get(index) or {}
            result = results.get(index) or {}
            state = str(status.get("status") or "")
            error = _error(status) if state != "completed" else ""
            if state not in _REPORTED:
                state = "adapter_stopped" if index in stopped or -1 in stopped else "unreported"
                if timed_out:
                    state = "timed_out"
                error = stopped.get(index) or stopped.get(-1) or f"no result after {seconds} s"
            # Hub's error can be a wrapper ("LightEval CLI failed with exit code 1") whose
            # cause is only in the adapter's log, e.g. a gated dataset.
            tail = ""
            if state != "completed":
                with suppress(Exception):
                    tail = await lifecycle.adapter_log_tail(execution, snapshot, index)
            records.append(
                {
                    "status": state,
                    "error": error[:500],
                    "log": tail,
                    "hub_job": execution["remote_id"],
                    "date": datetime.now(UTC).isoformat(timespec="seconds"),
                    "seconds": seconds,
                    "attempt": attempt,
                    "questions": self.args.questions or None,
                    "sent_score": sent,
                    "metrics": result.get("metrics") or {},
                    "additional_info": result.get("additional_info") or {},
                    "mlflow": await _mlflow(str(result.get("mlflow_run_id") or "")),
                }
            )
        return records

    async def run(self, evidence_path: Path) -> None:
        args, described = self.args, self.described
        evidence = _load_evidence(evidence_path, args, described)
        async with client.client() as sdk:
            entries = await catalog.load(sdk, [args.provider], {args.provider: described})
        if entries and entries[0].provider.problem:
            raise SystemExit(entries[0].provider.problem)
        capability = catalog.Capability.from_model({"model_name_or_path": args.model})
        only = set(filter(None, (args.only or "").split(",")))
        todo = [
            e
            for e in entries
            if contract.runnable(e)
            and (not only or e.id in only)
            and (args.rerun or e.id not in evidence["benchmarks"])
        ]
        if args.limit:
            todo = todo[: args.limit]
        _log(
            f"{args.provider}: {len(todo)} entries to run, {len(evidence['benchmarks'])} on record"
        )
        if not todo:
            return
        prefixed = (described.get("settings_profile") or "") in contract.PREFIXED
        queue = []
        for entry in todo:
            definition = {"provider_id": args.provider, "benchmark_id": entry.id, "parameters": {}}
            limit = {"max_samples": args.questions} if args.questions else {}
            recipe, errors = catalog.resolve_recipe(entries, capability, definition, limit)
            if recipe is None:
                evidence["benchmarks"][entry.id] = {
                    "status": "not_submitted",
                    "error": "; ".join(errors),
                }
                continue
            queue.append((entry, recipe, _score(entry.raw.get("primary_score")), 1))
        _save(evidence_path, evidence)
        endpoint, secret = await self.serve()
        _log(f"model server ready: {endpoint['base_url']} ({endpoint['model']})")
        done = 0
        try:
            while queue:
                chunk, queue = queue[: args.chunk], queue[args.chunk :]
                _log(f"Hub job: {', '.join(e.id for e, *_ in chunk)}")
                records = await self.run_chunk(chunk, endpoint, secret)
                for (entry, recipe, sent, attempt), record in zip(chunk, records, strict=True):
                    if record["status"] == "completed" and record["metrics"] and attempt == 1:
                        known = contract.directions(
                            record["metrics"],
                            (entry.compat or {}).get("directions") or {},
                            prefixed,
                        )
                        chosen = contract.choose_score(
                            entry.raw.get("primary_score"), record["metrics"], prefixed, known
                        )
                        want = _score(chosen)
                        if (want or {}).get("metric") != (sent or {}).get("metric"):
                            # Check the chain with the score users will see.
                            queue.append((entry, recipe, want, 2))
                            _log(f"  {entry.id}: score is {(want or {}).get('metric')}; runs again")
                            continue
                    evidence["benchmarks"][entry.id] = record
                    done += 1
                    _log(
                        f"  [{done}] {entry.id}: {record['status']}, {len(record['metrics'])}"
                        f" metrics, {record['seconds']} s {record['error'][:160]}"
                    )
                _save(evidence_path, evidence)
        finally:
            await self.cleanup()


def _load_evidence(
    path: Path, args: argparse.Namespace, described: dict[str, Any]
) -> dict[str, Any]:
    fresh = {
        "schema": contract.EVIDENCE,
        "provider_id": args.provider,
        "adapter_image": described["adapter_image"],
        "adapter_digest": described["adapter_digest"],
        "catalog": described.get("catalog"),
        "model": args.model,
        "questions": args.questions,
        "benchmarks": {},
    }
    if not path.exists() or not path.read_text().strip():
        return fresh
    found = json.loads(path.read_text())
    # Each run records its own question count, so --rerun --questions changes it per entry.
    same = all(found.get(k) == fresh[k] for k in ("schema", "adapter_digest", "model"))
    if not same:
        _log("the evidence on record is for another image or model; starting over")
        return fresh
    found["benchmarks"] = {
        k: v
        for k, v in (found.get("benchmarks") or {}).items()
        if v.get("status") != "not_submitted"
    }
    return dict(found)


def _save(path: Path, evidence: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", required=True)
    parser.add_argument("--described", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--model", default=MODEL, help="Hugging Face id of the model to serve")
    parser.add_argument(
        "--questions", type=int, default=10, help="questions per benchmark (0 = all of them)"
    )
    parser.add_argument("--chunk", type=int, default=8, help="benchmarks per Hub job")
    parser.add_argument("--timeout", type=int, default=2400, help="seconds per Hub job")
    parser.add_argument("--serve-timeout", type=int, default=1800)
    parser.add_argument("--poll", type=int, default=15)
    parser.add_argument("--only", default="", help="comma-separated benchmark ids")
    parser.add_argument("--limit", type=int, default=0, help="run at most this many entries")
    parser.add_argument("--rerun", action="store_true", help="run entries already on record")
    args = parser.parse_args()
    described = json.loads(args.described.read_text())
    if described.get("schema") != catalog.DESCRIBED:
        raise SystemExit(f"{args.described}: expected schema {catalog.DESCRIBED}")
    from amortized.main import _load_backends

    _load_backends()  # the server registers its compute backends at startup; so must we
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with open(args.evidence.with_suffix(".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another conformance run holds {lock.name}") from None
        asyncio.run(Runner(args, described).run(args.evidence))


if __name__ == "__main__":
    main()
