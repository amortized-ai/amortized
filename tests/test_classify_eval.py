"""Regression tests for the classification-eval asset (jobs/assets/classify_eval.py).

Loads the asset with stubbed `training_hub` + `sentence_transformers` (the heavy
deps it uses at runtime in the training image) and runs it end to end. Guards the
label-type-agnostic behavior: the SDG datasets store string category labels
(e.g. "telemetry"), so the router must NOT force int(label) — a regression that
previously slipped back in during an unrelated revert and broke eval.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ASSET = Path(__file__).resolve().parent.parent / "src/amortized/jobs/assets/classify_eval.py"


def _load_classify_eval(rows, monkeypatch):
    """Load the asset module with training_hub/sentence_transformers stubbed.

    The fake encoder returns a one-hot per class (perfect separation), so a
    correct router scores 100% — the test asserts the pipeline runs and handles
    the label type, not model quality.
    """
    labels_in_order = sorted({r["category"] for r in rows}, key=lambda x: str(x))
    _idx_by_str = {str(lbl): i for i, lbl in enumerate(labels_in_order)}

    class _FakeDS:
        def __init__(self, rs):
            self._rows = rs
            self.column_names = list(rs[0].keys())

        def __iter__(self):
            return iter(self._rows)

    th = types.ModuleType("training_hub")
    thu = types.ModuleType("training_hub.utils")
    thu.load_training_dataset = lambda _path: _FakeDS(rows)
    th.utils = thu
    monkeypatch.setitem(sys.modules, "training_hub", th)
    monkeypatch.setitem(sys.modules, "training_hub.utils", thu)

    st = types.ModuleType("sentence_transformers")

    class _FakeST:
        def __init__(self, _path):
            pass

        def encode(self, texts, normalize_embeddings=True):
            vecs = []
            for t in texts:
                cls = t.rsplit("_", 1)[0]
                v = np.zeros(len(labels_in_order))
                v[_idx_by_str[cls]] = 1.0
                vecs.append(v)
            return np.array(vecs)

    st.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    spec = importlib.util.spec_from_file_location("classify_eval_under_test", ASSET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, cfg, tmp_path, monkeypatch):
    cfg = {**cfg, "output_dir": str(tmp_path)}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["classify_eval", "--config", str(cfg_path)])
    mod.main()
    return json.loads((tmp_path / "metrics.json").read_text())["results"]


def test_string_category_labels(tmp_path, monkeypatch):
    cats = ["telemetry", "propulsion", "payload_operations", "other"]
    rows = [{"text": f"{c}_ex{i}", "category": c} for c in cats for i in range(20)]
    mod = _load_classify_eval(rows, monkeypatch)
    res = _run(
        mod,
        {"model_path": "x", "eval_data_path": "y", "text_column": "text",
         "label_column": "category", "class_labels": cats, "anchors_per_class": 8, "tau": 0.0},
        tmp_path, monkeypatch,
    )
    assert res["accuracy"] == 1.0
    assert set(res["per_class_f1"]) == set(cats)   # keyed by the string names
    assert set(res["confusion"]) == set(cats)


def test_abstentions_count_as_misses(tmp_path, monkeypatch):
    # tau above the max achievable score forces every query to abstain. An
    # abstention must NOT be credited to its (correct) best_label: accuracy and
    # macro-F1 go to 0, every query lands in the FALLBACK confusion column.
    cats = ["telemetry", "propulsion", "other"]
    rows = [{"text": f"{c}_ex{i}", "category": c} for c in cats for i in range(20)]
    mod = _load_classify_eval(rows, monkeypatch)
    res = _run(
        mod,
        {"model_path": "x", "eval_data_path": "y", "text_column": "text",
         "label_column": "category", "class_labels": cats,
         "anchors_per_class": 8, "tau": 1.1},
        tmp_path, monkeypatch,
    )
    assert res["accuracy"] == 0.0
    assert res["macro_f1"] == 0.0
    assert res["abstained"] == res["num_queries"]
    # every true class routed only to the FALLBACK bucket
    for cat in cats:
        row = res["confusion"][cat]
        assert row[mod.FALLBACK] > 0
        assert sum(v for k, v in row.items() if k != mod.FALLBACK) == 0


def test_integer_labels_still_work(tmp_path, monkeypatch):
    rows = [{"text": f"{c}_ex{i}", "category": c} for c in [0, 1, 2] for i in range(20)]
    mod = _load_classify_eval(rows, monkeypatch)
    res = _run(
        mod,
        {"model_path": "x", "eval_data_path": "y", "text_column": "text",
         "label_column": "category", "class_labels": ["a", "b", "c"],
         "anchors_per_class": 8, "tau": 0.0},
        tmp_path, monkeypatch,
    )
    assert res["accuracy"] == 1.0
    # integer labels map through class_labels by position
    assert set(res["per_class_f1"]) == {"a", "b", "c"}
