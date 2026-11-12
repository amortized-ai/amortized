"""The benchmark contract: compat/<provider>.json, built from what each adapter really did.

    python -m amortized.integrations.evalhub.contract        (make evalhub-contract)

Inputs per provider, under deploy/helm/amortized/files/evalhub/:
- described/<provider>.json: what each entry needs, read from the task configs in the
  pinned adapter image without a GPU (make evalhub-describe);
- conformance/<provider>.json: what each entry reported when it ran through Eval Hub on a
  served model, the way a user's job runs (make evalhub-conform);
- the vendored catalog: upstream's name for each entry's score.

Outputs: compat/<provider>.json, the only file the server reads (catalog.score); REPORT.md,
one row per entry for review; and the Studio test fixture, each entry's results as the API
returns them. Building fails, naming the entry, when the surfaces of a run disagree on its
score (the adapter's metrics, additional_info, MLflow's overall_score, the results API).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from amortized.integrations.evalhub import catalog
from amortized.integrations.evalhub.results import present

EVIDENCE = "amortized.evalhub.conformance/v1"
FILES = Path("deploy/helm/amortized/files/evalhub")
FIXTURE = Path("studio/src/test/fixtures/evalhub-results.json")
# Each registered provider's vendored catalog.
CATALOGS = {
    "amortized-lm-eval": "lm_evaluation_harness.yaml",
    "amortized-lighteval": "lighteval.yaml",
}
# Settings profiles whose adapter reports "<task>.<metric>" plus an "all.<metric>" mean.
PREFIXED = {"lighteval-endpoint-v1"}
# The same metric under another framework's name: upstream's catalog name → the adapter's.
ALIASES = {
    "accuracy": "acc",
    "exact_match": "em",
    "matthews_correlation": "mcc",
    "mc1": "truthfulqa_mc1",
    "mc2": "truthfulqa_mc2",
}
# Entries in two providers that ask the same questions are offered as one benchmark only if
# they also score them the same way. Metric names differ across frameworks (lm-eval's
# truthfulqa_mc1 reports "acc", LightEval's truthfulqa:mc "truthfulqa_mc1"), so each pair is
# confirmed here once, from both task definitions. REPORT.md lists the pairs not confirmed.
SAME_SCORING: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # Both: MC1 accuracy, the log-likelihood of each mc1_targets choice, best one correct.
    (("amortized-lm-eval", "truthfulqa_mc1"), ("amortized-lighteval", "truthfulqa:mc")),
]
# Stands for any model Amortized serves from a Hugging Face id: every feature a benchmark
# can need, so "runnable" means an entry some served model can take.
SERVED = catalog.Capability.from_model({"model_name_or_path": "org/model"})


class ContractError(ValueError):
    """The evidence does not support a contract; the message names each entry."""


def runnable(entry: catalog.Entry) -> bool:
    """Whether an entry can be offered at all, so conformance must run it."""
    return catalog.compatibility(entry, SERVED)[0] == "compatible"


def _prefix(names: list[str]) -> str:
    tasks = {n.split(".", 1)[0] for n in names if "." in n} - {"all"}
    return tasks.pop() if len(tasks) == 1 else "all"


def directions(
    metrics: Iterable[str], described: dict[str, bool], prefixed: bool
) -> dict[str, str]:
    """Each reported metric's direction, where the task config states it under that name."""
    found = {}
    for name in metrics:
        base = name.split(".", 1)[1] if prefixed and "." in name else name
        if base in described:
            found[name] = "higher" if described[base] else "lower"
    return found


def choose_score(
    primary: Any,
    metrics: Iterable[str],
    prefixed: bool,
    known: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The metric that scores an entry, picked from the names its adapter reported.

    In order: upstream's metric as is; with the adapter's prefix ("<task>." or "all."); under
    a known alias; the only metric reported. Otherwise no score, and the reason says why.
    """
    primary = primary if isinstance(primary, dict) else {}
    wanted = str(primary.get("metric") or "")
    names = [n for n in metrics if not n.endswith("_stderr")]
    prefix = _prefix(names) if prefixed else ""
    scoped = [n for n in names if n.startswith(f"{prefix}.")] if prefix else names

    def chosen(name: str, source: str) -> dict[str, Any]:
        direction = (known or {}).get(name)
        lower = direction == "lower" if direction else bool(primary.get("lower_is_better"))
        return {"metric": name, "lower_is_better": lower, "source": source}

    for name in dict.fromkeys(n for n in (wanted, ALIASES.get(wanted, "")) if n):
        for candidate in (name, f"{prefix}.{name}" if prefix else ""):
            if candidate and candidate in names:
                if candidate == wanted:
                    return chosen(candidate, "catalog")
                return chosen(candidate, f"catalog's {wanted}, as the adapter names it")
    if len(scoped) == 1:
        return chosen(scoped[0], "only metric reported")
    listed = ", ".join(sorted(scoped)) or "nothing"
    return {
        "metric": None,
        "reason": f"the catalog's metric is {wanted or 'unset'}; the adapter reports {listed}",
    }


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def chain_problems(
    evidence: dict[str, Any], chosen: dict[str, Any], view: dict[str, Any] | None
) -> list[str]:
    """Where the surfaces of a conformance run disagree with the contract's score."""
    sent = (evidence.get("sent_score") or {}).get("metric")
    want = chosen.get("metric")
    if sent != want:
        return [
            f"it ran with score {sent or 'none'}, but the score rule picks {want or 'none'};"
            " run `make evalhub-conform` again"
        ]
    value = evidence["metrics"].get(want) if want else None
    problems = []
    mlflow = evidence.get("mlflow") or {}
    if not mlflow.get("run_id") or mlflow.get("error"):
        problems.append(f"no readable MLflow run ({mlflow.get('error') or 'none recorded'})")
    elif not _same(mlflow.get("overall_score"), value):
        problems.append(f"MLflow overall_score is {mlflow.get('overall_score')}, not {value}")
    info = evidence.get("additional_info") or {}
    if "primary_metric" in info and info["primary_metric"] != want:
        problems.append(f"additional_info primary_metric is {info['primary_metric']}, not {want}")
    for key in ("zero_shot", "alt_prompting"):
        if key in info and not _same(info[key], value):
            problems.append(f"additional_info {key} is {info[key]}, not {value}")
    if view is None:
        problems.append("the results API shows nothing")
    elif view.get("primary_metric") != want or not _same(view.get("primary_score"), value):
        problems.append(
            f"the results API shows {view.get('primary_metric')} {view.get('primary_score')},"
            f" not {want} {value}"
        )
    return problems


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ContractError(f"{path} is not a JSON object")
    return data


def _provider(provider_id: str, raw: dict[str, Any], described: dict[str, Any]) -> catalog.Provider:
    return catalog.Provider(
        id=provider_id,
        name=str(raw.get("title") or raw.get("name") or provider_id),
        image=str(described["adapter_image"]),
        catalog_source=dict(described.get("catalog") or {}),
        problem="",
        profile=str(described.get("settings_profile") or ""),
    )


def _execution(benchmark_id: str, provider_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """The Hub snapshot of one entry's run, as a user's job stores it."""
    return {
        "snapshot": {
            "status": {
                "state": "completed",
                "benchmarks": [{"id": benchmark_id, "status": "completed", "benchmark_index": 0}],
            },
            "results": {
                "benchmarks": [
                    {
                        "id": benchmark_id,
                        "provider_id": provider_id,
                        "benchmark_index": 0,
                        "metrics": evidence.get("metrics") or {},
                        "additional_info": evidence.get("additional_info") or {},
                        "mlflow_run_id": (evidence.get("mlflow") or {}).get("run_id"),
                    }
                ]
            },
        },
        "remote_id": evidence.get("hub_job"),
        "observed_at": evidence.get("date") or "",
    }


def _failure(run: dict[str, Any]) -> str:
    """Why a run failed: Hub's error, then the adapter's last error line if it adds to it."""
    error = (run.get("error") or "").strip()
    first = error.splitlines()[0] if error else f"Hub status {run.get('status')}, no metrics"
    log = (run.get("log") or "").strip()
    return f"{first}; adapter log: {log}" if log and log not in error else first


def _questions(run: dict[str, Any], found: dict[str, Any]) -> int | None:
    """How many questions a run asked (None = all); older records carry only the file's."""
    return run["questions"] if "questions" in run else found.get("questions")


def _job_and_card(
    entries: list[catalog.Entry], entry: catalog.Entry, questions: int | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from amortized.api.jobs import _benchmark_view

    definition = {"provider_id": entry.provider.id, "benchmark_id": entry.id, "parameters": {}}
    limit = {"max_samples": questions} if questions else {}
    recipe, _ = catalog.resolve_recipe(entries, SERVED, definition, limit)
    if recipe is None:
        return None, None
    evaluation = {
        **definition,
        "kind": "benchmark",
        "recipe": recipe,
        "recipe_fingerprint": catalog.fingerprint(recipe),
    }
    view = _benchmark_view(recipe, {"evaluation": evaluation})
    return {"config": {"evaluation": evaluation}}, view.model_dump() if view else None


def build(files: Path = FILES) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    """(compat file per provider, REPORT.md, Studio fixture) from the files on disk."""
    problems: list[str] = []
    compat: dict[str, dict[str, Any]] = {}
    raws: dict[str, dict[str, Any]] = {}
    described: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for provider_id, catalog_file in CATALOGS.items():
        described[provider_id] = _read(files / f"described/{provider_id}.json")
        if described[provider_id].get("schema") != catalog.DESCRIBED:
            raise ContractError(f"described/{provider_id}.json: expected {catalog.DESCRIBED}")
        raws[provider_id] = yaml.safe_load((files / catalog_file).read_text())
        path = files / f"conformance/{provider_id}.json"
        evidence[provider_id] = _read(path) if path.exists() else {"benchmarks": {}}
        found = evidence[provider_id]
        if path.exists() and (
            found.get("schema") != EVIDENCE
            or found.get("adapter_digest") != described[provider_id]["adapter_digest"]
        ):
            raise ContractError(
                f"conformance/{provider_id}.json is not evidence for the described adapter"
                f" image {described[provider_id]['adapter_digest']}; run `make evalhub-conform`"
            )
    providers = {p: _provider(p, raws[p], described[p]) for p in CATALOGS}
    before = [
        catalog.Entry(providers[p], raw, described[p]["benchmarks"].get(raw["id"]))
        for p in CATALOGS
        for raw in raws[p]["benchmarks"]
    ]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    verdicts: dict[tuple[str, str], str] = {}
    for entry in before:
        key = (entry.provider.id, entry.id)
        record = {k: v for k, v in (entry.compat or {}).items() if k != "directions"}
        records[key] = record
        if not runnable(entry):
            status, reason = catalog.compatibility(entry, SERVED)
            verdicts[key] = f"no: {reason}" if status != "compatible" else "no"
            continue
        found = evidence[entry.provider.id]
        run = (found.get("benchmarks") or {}).get(entry.id)
        if run is None:
            record.update(status="unknown", reason="no conformance run on this adapter image yet")
            verdicts[key] = "no: not run yet"
            continue
        if run.get("status") != "completed" or not run.get("metrics"):
            error = _failure(run)
            record.update(status="unknown", reason=f"its conformance run failed: {error}"[:400])
            verdicts[key] = f"no: run failed: {error}"[:200]
            continue
        prefixed = entry.provider.profile in PREFIXED
        known = directions(run["metrics"], (entry.compat or {}).get("directions") or {}, prefixed)
        chosen = choose_score(entry.raw.get("primary_score"), run["metrics"], prefixed, known)
        total = (run.get("additional_info") or {}).get("num_samples_original")
        if isinstance(total, int):
            # lm-eval reports how many questions the full task holds. LightEval's results
            # carry no total, so its count comes from describe (question_count).
            record["question_count"] = total
        record.update(
            metrics=sorted(run["metrics"]),
            directions=known,
            score=chosen,
            verified={
                "adapter_digest": found["adapter_digest"],
                "date": run.get("date"),
                "model": found.get("model"),
                "questions": _questions(run, found),
                "hub_job": run.get("hub_job"),
            },
        )
        verdicts[key] = "yes"
    confirmed = {frozenset(pair) for pair in SAME_SCORING}
    # Which entries ask the same questions is a fact of their task configs, not of their runs.
    groups = [frozenset((e.provider.id, e.id) for e in g) for g in catalog.same_questions(before)]
    for pair in confirmed - set(groups):
        problems.append(f"SAME_SCORING lists {sorted(pair)}; they no longer ask the same questions")
    for keys in groups:
        # A copy is offered as one benchmark, so every member must have run and have a score.
        scored = all(
            verdicts[key] == "yes" and (records[key].get("score") or {}).get("metric")
            for key in keys
        )
        if keys in confirmed and scored:
            for key in keys:
                records[key]["same_scoring"] = [
                    {"provider_id": p, "id": i} for p, i in sorted(keys - {key})
                ]
    entries = [
        catalog.Entry(providers[p], raw, records[(p, raw["id"])])
        for p in CATALOGS
        for raw in raws[p]["benchmarks"]
    ]
    fixture_entries = []
    for entry in entries:
        key = (entry.provider.id, entry.id)
        if verdicts[key] != "yes":
            continue
        found = evidence[entry.provider.id]
        run = found["benchmarks"][entry.id]
        job, card = _job_and_card(entries, entry, _questions(run, found))
        view = present(_execution(entry.id, entry.provider.id, run), job) if job else None
        for problem in chain_problems(run, records[key]["score"], view):
            problems.append(f"{entry.provider.id} {entry.id}: {problem}")
        fixture_entries.append(
            {"provider_id": entry.provider.id, "id": entry.id, "card": card, "results": view}
        )
    if problems:
        raise ContractError("\n".join(problems))
    for provider_id in CATALOGS:
        top = {k: v for k, v in described[provider_id].items() if k != "benchmarks"}
        found = evidence[provider_id]
        compat[provider_id] = {
            **top,
            "schema": catalog.SCHEMA,
            "conformance": {"model": found.get("model"), "questions": found.get("questions")},
            "benchmarks": {
                raw["id"]: records[(provider_id, raw["id"])]
                for raw in raws[provider_id]["benchmarks"]
            },
        }
    fixture = {
        "adapter_digests": {p: described[p]["adapter_digest"] for p in CATALOGS},
        "entries": fixture_entries,
    }
    return compat, report(entries, verdicts, groups, confirmed, described, evidence), fixture


def _suites(entries: list[catalog.Entry]) -> dict[tuple[str, str], list[str]]:
    """Per entry, the suites in its provider whose tasks include all of its own."""
    found: dict[tuple[str, str], list[str]] = {}
    for suite in entries:
        if not suite.suite or not suite.compat:
            continue
        tasks = set(suite.compat.get("tasks") or ())
        for entry in entries:
            own = set((entry.compat or {}).get("tasks") or ())
            if entry is not suite and entry.provider is suite.provider and own and own <= tasks:
                found.setdefault((entry.provider.id, entry.id), []).append(suite.id)
    return found


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def report(
    entries: list[catalog.Entry],
    verdicts: dict[tuple[str, str], str],
    groups: list[frozenset[tuple[str, str]]],
    confirmed: set[frozenset[tuple[str, str]]],
    described: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
) -> str:
    suites = _suites(entries)
    by_key = {(e.provider.id, e.id): e for e in entries}
    copies: dict[tuple[str, str], str] = {}
    for keys in groups:
        if all((by_key[key].compat or {}).get("same_scoring") for key in keys):
            how = "copy"
        elif keys in confirmed:
            how = "would be a copy, but not every member ran with a score"
        else:
            how = "scored differently, listed separately"
        for key in keys:
            others = ", ".join(f"{p} {i}" for p, i in sorted(keys - {key}))
            copies[key] = f"{others} ({how})"
    lines = [
        "# Benchmark contract",
        "",
        "Generated by `make evalhub-contract` from `described/` (task configs read in each",
        "pinned adapter image) and `conformance/` (each entry run through Eval Hub). Do not edit.",
        "",
        "| Provider | Adapter image | Catalog commit | Model | Questions per entry |",
        "|---|---|---|---|---|",
    ]
    for provider_id, record in described.items():
        found = evidence[provider_id]
        lines.append(
            f"| {provider_id} | `{record['adapter_image']}` |"
            f" {(record.get('catalog') or {}).get('commit', '')} |"
            f" {found.get('model') or ''} | {found.get('questions') or ''} |"
        )
    for provider_id in described:
        mine = [e for e in entries if e.provider.id == provider_id]
        mine_keys = [(e.provider.id, e.id) for e in mine]
        default = evidence[provider_id].get("questions")
        runs = sum(verdicts[k] == "yes" for k in mine_keys)
        failed = sum(verdicts[k].startswith("no: run failed") for k in mine_keys)
        lines += [
            "",
            f"## {provider_id}: {len(mine)} entries, {runs} verified, {failed} failed their run",
            "",
            "| Entry | Runs | Questions (full) | Scoring | Score | Why this score"
            " | Same questions as | In suites |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for entry in mine:
            key = (entry.provider.id, entry.id)
            compat = entry.compat or {}
            record = compat.get("score") or {}
            if record.get("metric"):
                direction = "lower" if record.get("lower_is_better") else "higher"
                score, why = f"{record['metric']} ({direction} is better)", record.get("source")
            elif record:
                score, why = "none", record.get("reason")
            else:
                score, why = "", ""
            ran = verdicts[key]
            asked = (compat.get("verified") or {}).get("questions", default)
            if ran == "yes" and asked != default:
                ran = f"yes ({asked or 'all'} questions)"
            lines.append(
                f"| {_cell(entry.id)} | {_cell(ran)} | {compat.get('question_count') or ''} |"
                f" {compat.get('scoring') or ''} |"
                f" {_cell(score)} | {_cell(why or '')} | {_cell(copies.get(key, ''))} |"
                f" {_cell(', '.join(suites.get(key, [])))} |"
            )
    return "\n".join(lines) + "\n"


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--files", type=Path, default=FILES)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args()
    try:
        compat, text, fixture = build(args.files)
    except ContractError as exc:
        sys.exit(f"The contract cannot be built:\n{exc}")
    for provider_id, data in compat.items():
        (args.files / f"compat/{provider_id}.json").write_text(_dump(data))
    (args.files / "REPORT.md").write_text(text)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(_dump(fixture))
    for provider_id, data in compat.items():
        records = data["benchmarks"].values()
        verified = sum(1 for r in records if r.get("verified"))
        print(f"{provider_id}: {verified} of {len(records)} entries verified")


if __name__ == "__main__":
    main()
