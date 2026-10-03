"""Classification / router eval for an embedding model.

Runs inside the training image (which ships sentence-transformers). Given a
tuned sentence-transformers model and a held-out labeled dataset, it:

  1. groups the dataset by integer label,
  2. holds out ``anchors_per_class`` examples per class as routing *anchors* and
     uses the rest as *query* examples (stratified, seeded — anchors and queries
     are disjoint, so there is no anchor/query leakage),
  3. routes each query to the class whose anchors it is most similar to
     (per-class score = mean of the top-k cosine similarities; a query whose best
     score is below ``tau`` abstains to a FALLBACK bucket),
  4. writes accuracy / macro-F1 / per-class F1 / confusion to
     ``metrics.json`` and per-query rows to ``results.jsonl``.

**What the number means.** The anchors are drawn from the eval set itself, so the
accuracy/F1 is a *separability* proxy — "how well do the tuned embeddings route
unseen examples given ~N labeled anchors per class" — not a deployed-classifier
number. There is no persisted serving anchor set yet, and anchors + queries come
from the same SDG run, so treat it as an upper-ish bound on in-distribution
routing quality rather than real-world accuracy. ``metrics.json`` records this
caveat under ``results.metric_meaning``.

The metrics.json shape mirrors the generative eval runner
(``results.scores`` + ``results.model.num_samples``) so the existing
``on_success`` tagging and the eval-results UI work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

# Internal confusion-matrix key for abstentions. An object() sentinel can never
# collide with a dataset label (string OR int), so a dataset that happens to use
# the literal "__fallback__" as a class name is still scored correctly (its real
# column stays separate from the abstain bucket). FALLBACK is only the display
# name used in the emitted metrics/results.
_ABSTAIN = object()
FALLBACK = "__fallback__"

# Flag a class as low-signal when it ends up with fewer than this many queries —
# its per-class F1 is then a noisy single-draw estimate.
MIN_QUERIES_PER_CLASS = 5


def _load_rows(path: str, text_column: str, label_column: str) -> list[dict]:
    """Load a labeled dataset (jsonl/json/parquet/csv file, dir, or HF id)."""
    from training_hub.utils import load_training_dataset

    ds = load_training_dataset(path)
    cols = ds.column_names
    if text_column not in cols:
        raise ValueError(
            f"text column {text_column!r} not found in dataset columns {cols}"
        )
    if label_column not in cols:
        raise ValueError(
            f"label column {label_column!r} not found in dataset columns {cols}"
        )
    rows = []
    for r in ds:
        label = r[label_column]
        if label is None or r[text_column] is None:
            continue
        # Keep the label as-is — int index OR category string (e.g. "telemetry").
        # The router groups by label value, so both work; do NOT force int() (the
        # SDG datasets store string category names).
        rows.append({"text": str(r[text_column]), "label": label})
    if not rows:
        raise ValueError("eval dataset has no usable (text, label) rows")
    return rows


def _macro_f1(confusion: dict, labels: list) -> tuple[float, dict]:
    per_class: dict = {}
    for i in labels:
        tp = confusion[i][i]
        fp = sum(confusion[o][i] for o in labels if o != i)
        # fn sums over ALL predicted columns (incl. the FALLBACK/abstain column),
        # so abstentions count as false negatives for the true class.
        fn = sum(c for o, c in confusion[i].items() if o != i)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[i] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return macro, per_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    with open(ap.parse_args().config) as _cfg_f:
        cfg = json.load(_cfg_f)

    text_col = cfg.get("text_column", "text")
    label_col = cfg.get("label_column", "label")
    anchors_per_class = int(cfg.get("anchors_per_class", 16))
    top_k = int(cfg.get("top_k", 3))
    tau = float(cfg.get("tau", 0.0))
    seed = int(cfg.get("seed", 42))
    class_labels = cfg.get("class_labels") or None
    output_dir = cfg.get("output_dir", "/amortized/work/results")
    os.makedirs(output_dir, exist_ok=True)

    def name(label) -> str:
        # Map an integer index through class_labels when provided; a string
        # category label (the common case) is already its own display name.
        if class_labels and isinstance(label, int) and 0 <= label < len(class_labels):
            return str(class_labels[label])
        return str(label)

    rng = random.Random(seed)
    rows = _load_rows(cfg["eval_data_path"], text_col, label_col)

    by_label: dict[int, list[str]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r["text"])
    labels = sorted(by_label)

    # Stratified anchors/queries split (disjoint per class). Cap anchors at half
    # the class so query count scales with the eval size — a fixed
    # anchors_per_class would otherwise leave small classes with only a query or
    # two, making accuracy/macro-F1 a noisy single draw.
    anchors: dict[int, list[str]] = {}
    queries: list[dict] = []
    skipped_classes: list[int] = []
    low_query_classes: list[int] = []
    for label in labels:
        texts = list(by_label[label])
        rng.shuffle(texts)
        if len(texts) <= 1:
            anchors[label] = texts  # single example: anchor only, no query
            skipped_classes.append(label)
            continue
        cap = max(1, len(texts) // 2)  # never use more than half the class as anchors
        n_anchor = min(anchors_per_class, cap)
        anchors[label] = texts[:n_anchor]
        class_queries = texts[n_anchor:]
        for t in class_queries:
            queries.append({"text": t, "label": label})
        if len(class_queries) < MIN_QUERIES_PER_CLASS:
            low_query_classes.append(label)

    if not queries:
        raise ValueError(
            "no query examples left after holding out anchors — reduce"
            " anchors_per_class or provide more examples per class"
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["model_path"])
    anchor_emb = {
        label: model.encode(anchors[label], normalize_embeddings=True)
        for label in labels
    }

    query_texts = [q["text"] for q in queries]
    query_emb = model.encode(query_texts, normalize_embeddings=True)

    # Confusion columns include a FALLBACK (abstain) bucket so abstentions are
    # counted as misses (false negatives for the true class), not as best_label.
    confusion: dict = {i: {j: 0 for j in labels} for i in labels}
    for i in labels:
        confusion[i][_ABSTAIN] = 0
    correct = 0
    abstained = 0
    results = []
    for q, emb in zip(queries, query_emb, strict=True):
        scores: dict = {}
        for label in labels:
            sims = anchor_emb[label] @ emb
            k = min(top_k, len(sims))
            scores[label] = float(np.sort(sims)[-k:].mean())
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        # tau=0 disables abstention (per the EvalJobConfig contract). Guard the
        # threshold so a query with all-negative cosine similarities doesn't
        # abstain when abstention is meant to be off.
        abstain = tau > 0.0 and best_score < tau
        if abstain:
            abstained += 1
        # An abstention predicts the abstain bucket (never the true class) — count
        # it as a miss and record it in the FALLBACK column.
        pred_key = _ABSTAIN if abstain else best_label
        confusion[q["label"]][pred_key] += 1
        is_correct = (not abstain) and (best_label == q["label"])
        correct += int(is_correct)
        results.append({
            "text": q["text"],
            "actual": q["label"],
            "actual_name": name(q["label"]),
            "predicted": FALLBACK if abstain else best_label,
            "predicted_name": FALLBACK if abstain else name(best_label),
            "score": round(best_score, 4),
            "correct": is_correct,
        })

    total = len(queries)
    accuracy = correct / total if total else 0.0
    # Score macro/per-class F1 only over classes that actually have queries. A
    # single-example (skipped) class has no queries, so including it would add a
    # spurious 0.0 to the macro average and understate a perfect model.
    scored_labels = sorted({q["label"] for q in queries})
    macro_f1, per_class_f1 = _macro_f1(confusion, scored_labels)

    metrics = {
        "eval_mode": "classification",
        "results": {
            # Mirror the generative runner so on_success + UI reuse the shape.
            "scores": {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4)},
            "model": {"num_samples": total},
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": {name(i): round(per_class_f1[i], 4) for i in scored_labels},
            "confusion": {
                name(i): {
                    **{name(j): confusion[i][j] for j in labels},
                    # Only surface the abstain column when it's non-trivial, so the
                    # common tau=0 output stays a plain NxN matrix.
                    **({FALLBACK: confusion[i][_ABSTAIN]} if abstained else {}),
                }
                for i in labels
            },
            "num_classes": len(labels),
            "num_scored_classes": len(scored_labels),
            "num_anchors_per_class": {name(i): len(anchors[i]) for i in labels},
            "num_queries": total,
            "abstained": abstained,
            "tau": tau,
            "top_k": top_k,
            "skipped_single_example_classes": [name(i) for i in skipped_classes],
            "low_query_classes": [name(i) for i in low_query_classes],
            "metric_meaning": (
                "Separability proxy: anchors are drawn from the eval set, so this "
                "measures how well the tuned embeddings route unseen in-distribution "
                "examples given ~N labeled anchors/class — not a deployed-classifier "
                "accuracy (no persisted serving anchor set yet)."
            ),
        },
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(
        f"classification eval: accuracy={accuracy:.2%} macro_f1={macro_f1:.4f} "
        f"over {total} queries in {len(scored_labels)} scored classes "
        f"({abstained} abstained at tau={tau})"
    )
    if low_query_classes:
        print(
            f"WARNING: {len(low_query_classes)} class(es) have < "
            f"{MIN_QUERIES_PER_CLASS} queries — per-class F1 for "
            f"{[name(i) for i in low_query_classes]} is a noisy estimate; "
            "consider a larger eval set."
        )
    if skipped_classes:
        print(
            f"NOTE: {len(skipped_classes)} single-example class(es) had no queries "
            f"and were excluded from macro-F1: {[name(i) for i in skipped_classes]}"
        )


if __name__ == "__main__":
    main()
