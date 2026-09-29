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

FALLBACK = "__fallback__"


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
        rows.append({"text": str(r[text_column]), "label": int(label)})
    if not rows:
        raise ValueError("eval dataset has no usable (text, label) rows")
    return rows


def _macro_f1(confusion: dict[int, dict[int, int]], labels: list[int]) -> tuple[float, dict[int, float]]:
    per_class: dict[int, float] = {}
    for i in labels:
        tp = confusion[i][i]
        fp = sum(confusion[o][i] for o in labels if o != i)
        fn = sum(confusion[i][o] for o in labels if o != i)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[i] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return macro, per_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))

    text_col = cfg.get("text_column", "text")
    label_col = cfg.get("label_column", "label")
    anchors_per_class = int(cfg.get("anchors_per_class", 16))
    top_k = int(cfg.get("top_k", 3))
    tau = float(cfg.get("tau", 0.0))
    seed = int(cfg.get("seed", 42))
    class_labels = cfg.get("class_labels") or None
    output_dir = cfg.get("output_dir", "/amortized/work/results")
    os.makedirs(output_dir, exist_ok=True)

    def name(label: int) -> str:
        if class_labels and 0 <= label < len(class_labels):
            return str(class_labels[label])
        return str(label)

    rng = random.Random(seed)
    rows = _load_rows(cfg["eval_data_path"], text_col, label_col)

    by_label: dict[int, list[str]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r["text"])
    labels = sorted(by_label)

    # Stratified anchors/queries split (disjoint per class).
    anchors: dict[int, list[str]] = {}
    queries: list[dict] = []
    skipped_classes: list[int] = []
    for label in labels:
        texts = list(by_label[label])
        rng.shuffle(texts)
        n_anchor = min(anchors_per_class, max(1, len(texts) - 1)) if len(texts) > 1 else len(texts)
        anchors[label] = texts[:n_anchor]
        for t in texts[n_anchor:]:
            queries.append({"text": t, "label": label})
        if len(texts) <= 1:
            skipped_classes.append(label)  # single-example class: anchor only, no query

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

    confusion: dict[int, dict[int, int]] = {i: {j: 0 for j in labels} for i in labels}
    correct = 0
    abstained = 0
    results = []
    for q, emb in zip(queries, query_emb):
        scores: dict[int, float] = {}
        for label in labels:
            sims = anchor_emb[label] @ emb
            k = min(top_k, len(sims))
            scores[label] = float(np.sort(sims)[-k:].mean())
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        abstain = best_score < tau
        if abstain:
            abstained += 1
        pred_label = best_label  # for accuracy/confusion, an abstain still counts as best_label
        confusion[q["label"]][pred_label] += 1
        is_correct = pred_label == q["label"]
        correct += int(is_correct)
        results.append({
            "text": q["text"],
            "actual": q["label"],
            "actual_name": name(q["label"]),
            "predicted": FALLBACK if abstain else pred_label,
            "predicted_name": FALLBACK if abstain else name(pred_label),
            "score": round(best_score, 4),
            "correct": is_correct,
        })

    total = len(queries)
    accuracy = correct / total if total else 0.0
    macro_f1, per_class_f1 = _macro_f1(confusion, labels)

    metrics = {
        "eval_mode": "classification",
        "results": {
            # Mirror the generative runner so on_success + UI reuse the shape.
            "scores": {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4)},
            "model": {"num_samples": total},
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": {name(i): round(per_class_f1[i], 4) for i in labels},
            "confusion": {
                name(i): {name(j): confusion[i][j] for j in labels} for i in labels
            },
            "num_classes": len(labels),
            "num_anchors_per_class": {name(i): len(anchors[i]) for i in labels},
            "num_queries": total,
            "abstained": abstained,
            "tau": tau,
            "top_k": top_k,
            "skipped_single_example_classes": [name(i) for i in skipped_classes],
        },
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(
        f"classification eval: accuracy={accuracy:.2%} macro_f1={macro_f1:.4f} "
        f"over {total} queries in {len(labels)} classes "
        f"({abstained} abstained at tau={tau})"
    )


if __name__ == "__main__":
    main()
