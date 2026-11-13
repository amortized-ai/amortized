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


def _head_kind(model_path: str, trust_remote_code: bool = False) -> str | None:
    """Which classification head a model carries: "sequence", "other" (multiple-choice /
    token head), or None for an embedding model (sentence-transformers modules, no head).

    A classifier_sft output ships label_mapping.json plus a config.json naming the head
    architecture. A base-model baseline may be a Hugging Face id, so resolve its config
    remotely when there is no local directory."""
    archs: list[str] = []
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isdir(model_path):
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    archs = list(json.load(f).get("architectures") or [])
            except (ValueError, OSError):
                archs = []
        if not archs and os.path.exists(os.path.join(model_path, "label_mapping.json")):
            return "sequence"  # classifier_sft output with an unnamed architecture
    else:
        try:
            from transformers import AutoConfig

            archs = list(
                AutoConfig.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                ).architectures
                or []
            )
        except Exception:
            return None
    arch = " ".join(archs)
    if "ForSequenceClassification" in arch:
        return "sequence"
    if "ForMultipleChoice" in arch or "ForTokenClassification" in arch:
        return "other"
    return None


def _ece(probs, correct, bins: int = 15) -> float:

    conf = probs.max(1)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.sum() > 0:
            e += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


# Temperature-scaling calibration: fit T on a seeded random slice of the eval set and
# measure ECE before/after only on the rest, so the reported gain is out-of-sample.
# Skipped (and reported as skipped, never as T=1.0) when the held-out slice is too small.
_CAL_FIT_MAX = 1500
_CAL_MIN_HELDOUT = 200


def _fit_temperature(logits, y, seed: int) -> dict:
    """Fit a softmax temperature on part of (logits, y) and score ECE on the remainder.

    Returns {"ece_before", "ece_after", "temperature", "fit_rows", "heldout_rows"} or
    {"skipped": reason}. Rows whose label the model doesn't know (y < 0) can't be fit."""
    import numpy as np
    import torch

    known = np.flatnonzero(y >= 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(known)
    n_fit = min(_CAL_FIT_MAX, len(known) // 2)
    fit, held = known[:n_fit], known[n_fit:]
    if len(held) < _CAL_MIN_HELDOUT or n_fit == 0:
        return {
            "skipped": (
                f"{len(known)} rows with labels the model knows; calibration needs at least"
                f" {2 * _CAL_MIN_HELDOUT} (half to fit T, half to measure ECE)"
            )
        }
    # Fit log T so T stays positive; LBFGS with a line search converges reliably.
    log_t = torch.zeros(1, requires_grad=True)
    fit_logits = torch.tensor(logits[fit], dtype=torch.float64)
    fit_labels = torch.tensor(y[fit])
    opt = torch.optim.LBFGS([log_t], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(fit_logits / log_t.exp(), fit_labels)
        loss.backward()
        return loss

    opt.step(_closure)
    temp = float(log_t.detach().exp())
    if not np.isfinite(temp) or temp <= 0:
        return {"skipped": f"temperature fit did not converge (T={temp})"}

    def _softmax(x):
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    held_logits = logits[held]
    correct = (held_logits.argmax(1) == y[held]).astype(float)
    return {
        "ece_before": round(_ece(_softmax(held_logits), correct), 4),
        "ece_after": round(_ece(_softmax(held_logits / temp), correct), 4),
        "temperature": round(temp, 3),
        "fit_rows": int(n_fit),
        "heldout_rows": len(held),
    }


def _head_classifier_eval(cfg: dict, rows: list[dict], output_dir: str, name) -> None:
    """Evaluate a classifier_sft head model: argmax predictions on the held-out set, with
    accuracy / macro-F1 / per-class F1 / confusion + temperature-scaled calibration (ECE).

    Unlike the embedding centroid proxy, this is a *true* held-out classifier accuracy.
    ``name`` maps a label to its display name (class_labels), applied to both the ground
    truth and the predictions so they share one label space. The router-only knobs
    (anchors_per_class, top_k, tau) don't apply here."""
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    mp = cfg["model_path"]
    max_len = int(cfg.get("max_length") or 512)
    trust = bool(cfg.get("trust_remote_code"))
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=trust)
    model = AutoModelForSequenceClassification.from_pretrained(mp, trust_remote_code=trust).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token  # decoder backbones ship without a pad token
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {str(k): int(v) for k, v in (model.config.label2id or {}).items()}

    def _label_name(raw) -> str:
        # Integer-like checkpoint labels ("3") map through class_labels like int ground truth.
        if isinstance(raw, str) and raw.lstrip("-").isdigit():
            raw = int(raw)
        return name(raw)

    texts = [r["text"] for r in rows]
    raw_true = [str(r["label"]) for r in rows]
    true = [_label_name(r["label"]) for r in rows]
    logits_list = []
    for i in range(0, len(texts), 128):
        enc = tok(
            texts[i : i + 128],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            logits_list.append(model(**enc).logits.float().cpu().numpy())
    logits = np.concatenate(logits_list)
    if not np.isfinite(logits).all():
        raise ValueError(
            "the classifier produced non-finite logits (NaN/inf) — the checkpoint is broken"
            " (e.g. diverged training); retrain before evaluating"
        )
    pred_idx = logits.argmax(1)
    pred = [_label_name(id2label.get(int(i), i)) for i in pred_idx]

    accuracy = accuracy_score(true, pred)
    labels_sorted = sorted(set(true) | set(pred))
    macro_f1 = f1_score(true, pred, average="macro", labels=labels_sorted, zero_division=0)
    per_class = f1_score(true, pred, average=None, labels=labels_sorted, zero_division=0)

    y_idx = np.array([label2id.get(t, -1) for t in raw_true])
    calibration = _fit_temperature(logits, y_idx, int(cfg.get("seed", 42)))

    confusion = {t: {p: 0 for p in labels_sorted} for t in labels_sorted}
    for t, p in zip(true, pred, strict=False):
        confusion[t][p] += 1

    metrics = {
        "eval_mode": "classification",
        "results": {
            "scores": {
                "accuracy": round(float(accuracy), 4),
                "macro_f1": round(float(macro_f1), 4),
            },
            "model": {"num_samples": len(rows)},
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_class_f1": {
                lbl: round(float(f), 4)
                for lbl, f in zip(labels_sorted, per_class, strict=False)
            },
            "confusion": confusion,
            "num_classes": len(labels_sorted),
            "num_samples": len(rows),
            "calibration": calibration,
            "metric_meaning": (
                "True held-out classifier accuracy: argmax of a trained classification head on "
                "the held-out set (not a separability proxy). Calibration = temperature-scaled "
                "ECE, with T fit on one random part of the eval set and ECE measured on the rest."
            ),
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
        for t, p in zip(true, pred, strict=False):
            f.write(json.dumps({"actual_name": t, "predicted_name": p, "correct": t == p}) + "\n")
    if "skipped" in calibration:
        cal_msg = f"calibration skipped: {calibration['skipped']}"
    else:
        cal_msg = (
            f"ECE {calibration['ece_before']:.3f} -> {calibration['ece_after']:.3f}"
            f" (T={calibration['temperature']:.2f}, on {calibration['heldout_rows']} held-out rows)"
        )
    print(
        f"classification eval (head): accuracy={accuracy:.2%} macro_f1={macro_f1:.4f} "
        f"over {len(rows)} examples in {len(labels_sorted)} classes; {cal_msg}"
    )


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

    # A classifier_sft head model is evaluated by direct argmax (true held-out accuracy +
    # calibration); an embedding model falls through to the nearest-centroid separability proxy.
    kind = _head_kind(cfg["model_path"], bool(cfg.get("trust_remote_code")))
    if kind == "other":
        raise ValueError(
            "this model has a multiple-choice or token-classification head; classification"
            " eval scores sequence-classification heads only (one text -> one label)"
        )
    if kind == "sequence":
        _head_classifier_eval(cfg, rows, output_dir, name)
        return

    by_label: dict[int, list[str]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r["text"])
    labels = sorted(by_label)

    # Display name for the abstain bucket in the emitted metrics/results. The
    # internal key is the _ABSTAIN sentinel (so scoring never collides), but the
    # output uses a string — pick one that doesn't clash with a real class named
    # "__fallback__", or its confusion column would overwrite that class's.
    class_display_names = {name(i) for i in labels}
    abstain_name = FALLBACK
    while abstain_name in class_display_names:
        abstain_name = "_" + abstain_name

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
            "predicted": abstain_name if abstain else best_label,
            "predicted_name": abstain_name if abstain else name(best_label),
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
                    # common tau=0 output stays a plain NxN matrix. abstain_name is
                    # collision-free even if a real class is named "__fallback__".
                    **({abstain_name: confusion[i][_ABSTAIN]} if abstained else {}),
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
