"""Benchmark contract: the score rule over reported names, the merge of described records
with conformance evidence, the cross-surface checks, copies, REPORT.md, the Studio fixture,
the conformance runner's records, and the CI gate over the files in the chart."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import pytest

from amortized.integrations.evalhub import catalog, conformance, contract
from amortized.integrations.evalhub.results import present

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "deploy/helm/amortized/files/evalhub"
LM, LE = "amortized-lm-eval", "amortized-lighteval"


# --------------------------------------------------------------------------- #
# The score rule: picked from the names the adapter reported
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("primary", "metrics", "prefixed", "expected"),
    [
        # lm-eval arc_easy: the catalog's name, reported as is.
        ("acc_norm", ["acc", "acc_norm", "acc_stderr", "acc_norm_stderr"], False,
         ("acc_norm", "catalog")),
        # LightEval prefixes the task.
        ("extractive_match", ["gsm8k.extractive_match", "all.extractive_match"], True,
         ("gsm8k.extractive_match", "catalog's extractive_match, as the adapter names it")),
        # matthews_correlation under LightEval's name for it.
        ("matthews_correlation", ["glue:cola.acc", "glue:cola.mcc", "all.acc", "all.mcc"], True,
         ("glue:cola.mcc", "catalog's matthews_correlation, as the adapter names it")),
        # LightEval arc:easy: the catalog names acc_norm; acc is all it reports.
        ("acc_norm", ["arc:easy.acc", "all.acc"], True, ("arc:easy.acc", "only metric reported")),
        # A suite: the "all." mean.
        ("acc", ["a.acc", "b.acc", "all.acc"], True,
         ("all.acc", "catalog's acc, as the adapter names it")),
    ],
)  # fmt: skip
def test_score_is_the_catalogs_metric_under_the_name_the_adapter_reported(
    primary, metrics, prefixed, expected
):
    chosen = contract.choose_score({"metric": primary}, metrics, prefixed)
    assert (chosen["metric"], chosen["source"]) == expected
    assert chosen["lower_is_better"] is False


def test_no_score_rather_than_a_guess():
    metrics = ["word_perplexity", "byte_perplexity", "bits_per_byte"]
    assert contract.choose_score({"metric": "acc"}, metrics, False) == {
        "metric": None,
        "reason": "the catalog's metric is acc; the adapter reports bits_per_byte,"
        " byte_perplexity, word_perplexity",
    }


def test_direction_comes_from_the_task_config_when_it_names_the_metric():
    known = contract.directions(["t.loss", "all.loss"], {"loss": False}, prefixed=True)
    assert known == {"t.loss": "lower", "all.loss": "lower"}
    chosen = contract.choose_score({"metric": "acc"}, ["t.loss", "all.loss"], True, known)
    assert (chosen["metric"], chosen["lower_is_better"]) == ("t.loss", True)


# --------------------------------------------------------------------------- #
# Building the contract from evidence
# --------------------------------------------------------------------------- #


def _run(metrics, sent, info=None, **mlflow):
    value = metrics.get(sent) if sent else None
    return {
        "status": "completed",
        "error": "",
        "hub_job": "hub-1",
        "date": "2026-11-09T00:00:00+00:00",
        "seconds": 60,
        "attempt": 1,
        "sent_score": {"metric": sent, "lower_is_better": False} if sent else None,
        "metrics": metrics,
        "additional_info": info or {},
        "mlflow": {"run_id": "r1", "status": "FINISHED", "overall_score": value} | mlflow,
    }


def _evidence(provider_id, runs):
    described = json.loads((FILES / f"described/{provider_id}.json").read_text())
    return {
        "schema": contract.EVIDENCE,
        "provider_id": provider_id,
        "adapter_image": described["adapter_image"],
        "adapter_digest": described["adapter_digest"],
        "model": "Qwen/Qwen3-0.6B",
        "questions": 10,
        "benchmarks": runs,
    }


RUNS = {
    LM: {
        "arc_easy": _run(
            {"acc": 0.6, "acc_norm": 0.4, "acc_stderr": 0.11, "acc_norm_stderr": 0.11},
            "acc_norm",
            {"primary_metric": "acc_norm", "zero_shot": 0.4},
        ),
        "truthfulqa_mc1": _run({"acc": 0.3, "acc_stderr": 0.1}, "acc"),
        "pile_arxiv": _run(
            {"word_perplexity": 30.0, "byte_perplexity": 1.9, "bits_per_byte": 0.9},
            None,
            {"primary_metric": None, "zero_shot": None},
        ),
    },
    LE: {
        "arc:easy": _run({"arc:easy.acc": 0.5, "all.acc": 0.5}, "arc:easy.acc"),
        "truthfulqa:mc": _run(
            {
                "truthfulqa:mc.truthfulqa_mc1": 0.3,
                "truthfulqa:mc.truthfulqa_mc2": 0.45,
                "all.truthfulqa_mc1": 0.3,
                "all.truthfulqa_mc2": 0.45,
            },
            "truthfulqa:mc.truthfulqa_mc1",
        ),
        "glue:cola": {
            "status": "adapter_stopped",
            "error": "The benchmark adapter failed (BackoffLimitExceeded) without reporting",
            "metrics": {},
        },
    },
}


@pytest.fixture
def files(tmp_path):
    """The chart's catalogs and described records, with synthetic conformance evidence."""
    for name in ("lighteval.yaml", "lm_evaluation_harness.yaml"):
        shutil.copy(FILES / name, tmp_path / name)
    shutil.copytree(FILES / "described", tmp_path / "described")
    (tmp_path / "conformance").mkdir()
    for provider_id, runs in RUNS.items():
        write(tmp_path, provider_id, copy.deepcopy(runs))
    return tmp_path


def write(path, provider_id, runs):
    evidence = _evidence(provider_id, runs)
    (path / f"conformance/{provider_id}.json").write_text(json.dumps(evidence))


def test_a_verified_entry_records_what_it_reported_and_its_score(files):
    compat, _, _ = contract.build(files)
    arc = compat[LM]["benchmarks"]["arc_easy"]
    assert compat[LM]["schema"] == catalog.SCHEMA
    assert arc["status"] == "known"
    assert arc["metrics"] == ["acc", "acc_norm", "acc_norm_stderr", "acc_stderr"]
    assert arc["score"] == {"metric": "acc_norm", "lower_is_better": False, "source": "catalog"}
    assert arc["directions"] == {"acc": "higher", "acc_norm": "higher"}
    assert arc["verified"]["adapter_digest"] == compat[LM]["adapter_digest"]
    assert "directions" not in compat[LM]["benchmarks"]["hellaswag_ar"]  # not run: unverified
    lighteval = compat[LE]["benchmarks"]["arc:easy"]
    assert lighteval["score"]["metric"] == "arc:easy.acc"
    assert lighteval["score"]["source"] == "only metric reported"


def test_an_entry_without_a_score_says_why_and_keeps_its_directions(files):
    compat, _, fixture = contract.build(files)
    pile = compat[LM]["benchmarks"]["pile_arxiv"]
    assert pile["score"]["metric"] is None
    assert pile["score"]["reason"].startswith("the catalog's metric is")
    shown = next(e for e in fixture["entries"] if e["id"] == "pile_arxiv")
    assert shown["results"]["primary_metric"] is None
    assert shown["results"]["no_score_reason"] == pile["score"]["reason"]
    assert shown["results"]["metric_metadata"]["word_perplexity"]["direction"] == "lower"
    assert shown["card"]["metric"] == "" and shown["card"]["no_score"] == pile["score"]["reason"]


def test_entries_not_verified_are_unknown_so_morty_does_not_offer_them(files):
    compat, _, _ = contract.build(files)
    failed = compat[LE]["benchmarks"]["glue:cola"]
    assert failed["status"] == "unknown"
    assert failed["reason"].startswith("its conformance run failed: The benchmark adapter failed")
    assert "score" not in failed
    missing = compat[LE]["benchmarks"]["gsm8k"]
    assert missing["status"] == "unknown"
    assert missing["reason"] == "no conformance run on this adapter image yet"
    assert "score" not in missing and "directions" not in missing


def test_copies_must_also_score_the_questions_the_same_way(files):
    compat, report, _ = contract.build(files)
    assert compat[LM]["benchmarks"]["truthfulqa_mc1"]["same_scoring"] == [
        {"provider_id": LE, "id": "truthfulqa:mc"}
    ]
    assert "same_scoring" not in compat[LM]["benchmarks"]["arc_easy"]
    provider = {p: catalog.Provider(p, p, compat[p]["adapter_image"], {}, "") for p in compat}
    entries = [
        catalog.Entry(provider[p], {"id": i, "evaluation_targets": ["model"]}, r)
        for p in (LM, LE)
        for i, r in compat[p]["benchmarks"].items()
    ]
    groups = catalog.overlaps(entries).values()
    pairs = {frozenset((e.provider.id, e.id) for e in group) for group in groups}
    assert pairs == {frozenset({(LM, "truthfulqa_mc1"), (LE, "truthfulqa:mc")})}
    assert (
        "| arc_easy | yes | likelihood | acc_norm (higher is better) | catalog |"
        " amortized-lighteval arc:easy (scored differently, listed separately) |"
    ) in report


def test_a_failed_run_says_why_from_the_adapter_log_when_hub_only_wraps_it(files):
    # As LightEval gpqa:diamond: Hub's error wraps the CLI, the cause is in the adapter's log.
    runs = copy.deepcopy(RUNS[LE])
    runs["glue:cola"] = {
        "status": "failed",
        "error": "Unexpected error during LightEval evaluation: RuntimeError: LightEval CLI"
        " failed with exit code 1\nStdout: \nStderr: INFO: litellm response cache disabled",
        "log": "DatasetNotFoundError: Dataset 'nyu-mll/glue' is a gated dataset on the Hub.",
        "metrics": {},
    }
    write(files, LE, runs)
    compat, _, _ = contract.build(files)
    assert compat[LE]["benchmarks"]["glue:cola"]["reason"] == (
        "its conformance run failed: Unexpected error during LightEval evaluation:"
        " RuntimeError: LightEval CLI failed with exit code 1; adapter log:"
        " DatasetNotFoundError: Dataset 'nyu-mll/glue' is a gated dataset on the Hub."
    )


def test_a_copy_whose_other_member_failed_its_run_is_not_offered_as_one(files):
    # As in the pilot: lm-eval's truthfulqa_mc1 cannot load `truthful_qa` on its image.
    runs = copy.deepcopy(RUNS[LM])
    runs["truthfulqa_mc1"] = {
        "status": "failed",
        "error": "Evaluation failed: Invalid HF URI 'hf://datasets/truthful_qa@741b8276'",
        "metrics": {},
    }
    write(files, LM, runs)
    compat, report, _ = contract.build(files)
    assert "same_scoring" not in compat[LM]["benchmarks"]["truthfulqa_mc1"]
    assert "same_scoring" not in compat[LE]["benchmarks"]["truthfulqa:mc"]
    assert "would be a copy, but not every member ran with a score" in report


def test_an_entry_rerun_with_all_its_questions_keeps_its_own_count(files):
    # As tinyTruthfulQA: its scoring needs every question, so it ran again with --questions 0.
    runs = copy.deepcopy(RUNS[LM])
    runs["truthfulqa_mc1"]["questions"] = None
    write(files, LM, runs)
    compat, report, fixture = contract.build(files)
    assert compat[LM]["benchmarks"]["truthfulqa_mc1"]["verified"]["questions"] is None
    assert compat[LM]["benchmarks"]["arc_easy"]["verified"]["questions"] == 10
    cards = {e["id"]: e["card"] for e in fixture["entries"]}
    assert cards["truthfulqa_mc1"]["max_samples"] is None
    assert cards["arc_easy"]["max_samples"] == 10
    assert "| truthfulqa_mc1 | yes (all questions) |" in report
    assert "| arc_easy | yes |" in report


def test_report_has_one_row_per_entry_and_names_its_suites(files):
    _, report, _ = contract.build(files)
    rows = [line for line in report.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 3 + 2 + 167 + 28  # provider table (header + 2) + entry tables
    arc = next(r for r in rows if r.startswith("| arc:easy |"))
    assert "scientific_reasoning" in arc


def test_the_fixture_is_each_verified_entrys_real_api_payload(files):
    _, _, fixture = contract.build(files)
    assert {(e["provider_id"], e["id"]) for e in fixture["entries"]} == {
        (LM, "arc_easy"),
        (LM, "truthfulqa_mc1"),
        (LM, "pile_arxiv"),
        (LE, "arc:easy"),
        (LE, "truthfulqa:mc"),
    }
    arc = next(e for e in fixture["entries"] if e["id"] == "arc_easy")
    assert arc["card"]["metric"] == "acc_norm"
    results = arc["results"]
    assert (results["primary_metric"], results["primary_score"]) == ("acc_norm", 0.4)
    assert list(results["scores"]) == ["acc_norm", "acc"]  # the score first, no stderr rows
    assert results["stderr"] == {"acc": 0.11, "acc_norm": 0.11}
    assert results["primary_lower_is_better"] is False


def test_surfaces_that_disagree_on_the_score_stop_the_build(files):
    runs = copy.deepcopy(RUNS[LM])
    runs["arc_easy"]["mlflow"]["overall_score"] = 0.6  # the adapter's first metric, as before
    runs["arc_easy"]["additional_info"]["zero_shot"] = 0.6
    write(files, LM, runs)
    with pytest.raises(contract.ContractError) as failure:
        contract.build(files)
    assert "amortized-lm-eval arc_easy: MLflow overall_score is 0.6, not 0.4" in str(failure.value)
    assert "additional_info zero_shot is 0.6, not 0.4" in str(failure.value)


def test_a_run_with_another_score_than_the_rule_picks_must_run_again(files):
    runs = copy.deepcopy(RUNS[LE])
    runs["arc:easy"]["sent_score"] = {"metric": "acc_norm", "lower_is_better": False}
    write(files, LE, runs)
    with pytest.raises(contract.ContractError, match=r"arc:easy: it ran with score acc_norm"):
        contract.build(files)


def test_evidence_for_another_image_is_refused(files):
    evidence = json.loads((files / f"conformance/{LM}.json").read_text())
    evidence["adapter_digest"] = "sha256:other"
    (files / f"conformance/{LM}.json").write_text(json.dumps(evidence))
    with pytest.raises(contract.ContractError, match="not evidence for the described adapter"):
        contract.build(files)


# --------------------------------------------------------------------------- #
# Results: a benchmark job's view comes from its approved recipe
# --------------------------------------------------------------------------- #


def _job(score, directions=None):
    compat = {"score": score, "directions": directions or {}}
    return {"config": {"evaluation": {"kind": "benchmark", "recipe": {"compatibility": compat}}}}


def _execution(metrics):
    return {
        "snapshot": {
            "status": {"state": "completed"},
            "results": {"benchmarks": [{"id": "b", "metrics": metrics}]},
        }
    }


def test_results_show_the_contracts_score_with_its_direction_and_stderr():
    job = _job({"metric": "acc_norm", "lower_is_better": False}, {"acc": "higher"})
    view = present(_execution({"acc": 0.6, "acc_norm": 0.4, "acc_norm_stderr": 0.1}), job)
    assert view["kind"] == "benchmark"
    assert (view["primary_metric"], view["primary_score"]) == ("acc_norm", 0.4)
    assert view["primary_lower_is_better"] is False
    assert view["scores"] == {"acc_norm": 0.4, "acc": 0.6}
    assert view["stderr"] == {"acc_norm": 0.1}
    assert view["metric_metadata"]["acc_norm"]["direction"] == "higher"
    assert view["no_score_reason"] is None


def test_results_say_why_there_is_no_score():
    view = present(_execution({"ppl": 3.0}), _job({"metric": None, "reason": "only ppl"}))
    assert (view["primary_metric"], view["no_score_reason"]) == (None, "only ppl")
    missing = present(_execution({"acc": 0.5}), _job({"metric": "acc_norm"}))
    assert missing["primary_metric"] is None
    assert missing["no_score_reason"] == "the adapter did not report acc_norm; it reported acc"


# --------------------------------------------------------------------------- #
# The conformance runner's records
# --------------------------------------------------------------------------- #


async def test_conformance_records_each_benchmark_of_a_hub_job(monkeypatch):
    snapshots = [
        {"resource": {"id": "hub-9"}, "status": {"state": "pending"}},
        {
            "resource": {"id": "hub-9"},
            "status": {
                "state": "running",
                "benchmarks": [{"benchmark_index": 0, "status": "completed"}],
            },
            "results": {
                "benchmarks": [
                    {
                        "benchmark_index": 0,
                        "metrics": {"acc": 0.5},
                        "additional_info": {"zero_shot": 0.5},
                        "mlflow_run_id": "run-0",
                    }
                ]
            },
        },
    ]
    submitted = []

    async def submit(payload, execution):
        submitted.append(payload)
        return snapshots[0]

    async def observe(execution):
        return snapshots[1]

    async def stopped(execution, snapshot):
        return {1: "The benchmark adapter failed (BackoffLimitExceeded) without reporting"}

    async def mlflow(run_id):
        return {"run_id": run_id, "overall_score": 0.5} if run_id else {}

    monkeypatch.setattr(conformance.client, "submit", submit)
    monkeypatch.setattr(conformance.client, "observe", observe)
    monkeypatch.setattr(conformance.lifecycle, "stopped_adapters", stopped)

    async def tail(execution, snapshot, index):
        return f"OSError: adapter {index} cannot read its dataset"

    monkeypatch.setattr(conformance.lifecycle, "adapter_log_tail", tail)
    monkeypatch.setattr(conformance, "_mlflow", mlflow)
    args = argparse.Namespace(provider=LM, model="org/model", poll=0, timeout=60, questions=10)
    runner = conformance.Runner(args, {})
    recipe = {
        "benchmark_id": "a",
        "provider_id": LM,
        "parameters": {"num_examples": 10},
        "compatibility": {"settings_profile": "lm-eval-completions-v1"},
    }
    sent = {"metric": "acc", "lower_is_better": False}
    items = [(None, recipe, sent, 1), (None, {**recipe, "benchmark_id": "b"}, None, 2)]
    endpoint = {"base_url": "http://m/v1", "model": "m"}
    records = await runner.run_chunk(items, endpoint, "secret")
    benchmarks = submitted[0]["benchmarks"]
    assert [b["id"] for b in benchmarks] == ["a", "b"]
    assert benchmarks[0]["primary_score"] == sent and "primary_score" not in benchmarks[1]
    assert benchmarks[0]["parameters"]["tokenizer"] == "org/model"
    assert submitted[0]["experiment"]["name"] == conformance.EXPERIMENT
    assert records[0]["status"] == "completed"
    assert records[0]["metrics"] == {"acc": 0.5}
    assert records[0]["mlflow"] == {"run_id": "run-0", "overall_score": 0.5}
    assert records[0]["questions"] == 10
    assert records[1]["status"] == "adapter_stopped"
    assert records[1]["error"].startswith("The benchmark adapter failed")
    assert records[1]["attempt"] == 2 and records[1]["sent_score"] is None
    assert records[1]["log"] == "OSError: adapter 1 cannot read its dataset"
    assert records[0]["log"] == ""  # completed: no log read


# --------------------------------------------------------------------------- #
# CI gate: the contract in the chart is the one its evidence builds
# --------------------------------------------------------------------------- #


def test_contract_in_the_chart_is_what_its_evidence_builds():
    compat, report, fixture = contract.build(FILES)
    for provider_id, data in compat.items():
        on_disk = json.loads((FILES / f"compat/{provider_id}.json").read_text())
        assert on_disk == data, f"compat/{provider_id}.json is stale: run make evalhub-contract"
    assert (FILES / "REPORT.md").read_text() == report, "REPORT.md is stale"
    on_disk = json.loads((ROOT / contract.FIXTURE).read_text())
    assert on_disk == fixture, "Studio's fixture is stale: run make evalhub-contract"


@pytest.mark.parametrize("provider_id", [LM, LE])
def test_every_runnable_entry_ran_on_the_pinned_image_or_says_why_not(provider_id):
    data = json.loads((FILES / f"compat/{provider_id}.json").read_text())
    for benchmark_id, record in data["benchmarks"].items():
        if record.get("suite"):
            continue  # never offered: catalog.compatibility says suites are not supported yet
        if record["status"] != "known":
            assert record.get("reason"), benchmark_id
            continue
        assert record["verified"]["adapter_digest"] == data["adapter_digest"], benchmark_id
        score = record["score"]
        if score["metric"] is None:
            assert score["reason"], benchmark_id
        else:
            assert score["metric"] in record["metrics"], benchmark_id
