"""Turn a training dataset into the single JSONL file ``thub lora-grpo`` (verl backend) reads.

training-hub's verl data prep opens ``data_path`` as one JSONL file, but amortized datasets
arrive as a directory of SDG parquet/JSONL files. A HuggingFace dataset ID (e.g.
``HuggingFaceH4/MATH-500``) is also accepted: its ``train`` split (else its only/first split)
is downloaded with ``datasets``.

- ``math``: each row becomes ``problem``/``answer`` (what training-hub's math reward reads),
  taken from the prompt and reference-answer columns.
- ``tool_call``: rows are copied unchanged — training-hub reads the tool-call trace fields
  (question/tools/target_tool_name/target_arguments, or multi-turn messages) itself.

Usage: python3 prepare_grpo_data.py <math|tool_call> <data_path> <out.jsonl>
           [question_column] [answer_column]
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from typing import Any

QUESTION_COLUMNS = ("question", "problem", "prompt", "input", "text")
ANSWER_COLUMNS = ("ground_truth", "answer", "label", "category", "output")


def _data_files(path: str) -> list[str]:
    exts = (".parquet", ".jsonl", ".json")
    if os.path.isfile(path):
        return [path]
    found: list[str] = []
    for root, _dirs, files in os.walk(path):
        found.extend(os.path.join(root, f) for f in files if f.endswith(exts))
    return sorted(found)


def _read_rows(file: str) -> Iterator[dict[str, Any]]:
    if file.endswith(".parquet"):
        import pyarrow.parquet as pq

        yield from pq.read_table(file).to_pylist()
        return
    with open(file) as f:
        if file.endswith(".json"):
            data = json.load(f)
            yield from data if isinstance(data, list) else [data]
            return
        for line in f:
            if line.strip():
                yield json.loads(line)


def _hf_rows(dataset_id: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    splits = load_dataset(dataset_id)
    split = "train" if "train" in splits else next(iter(splits))
    print(f"prepare_grpo_data: using split {split!r} of {dataset_id}")
    yield from splits[split]


def _rows(data_path: str) -> Iterator[dict[str, Any]]:
    # Not a generator itself: the input files are listed before the caller opens the output
    # file, which may live in the same directory.
    if not data_path.startswith("/"):
        return _hf_rows(data_path)
    files = _data_files(data_path)
    if not files:
        raise SystemExit(f"prepare_grpo_data: no parquet/jsonl/json files under {data_path}")
    return (row for file in files for row in _read_rows(file))


def _pick(row: dict[str, Any], explicit: str, candidates: tuple[str, ...]) -> str | None:
    if explicit:
        return explicit if explicit in row else None
    return next((c for c in candidates if c in row), None)


def _math_record(row: dict[str, Any], question_column: str, answer_column: str) -> dict[str, str]:
    q_col = _pick(row, question_column, QUESTION_COLUMNS)
    a_col = _pick(row, answer_column, ANSWER_COLUMNS)
    if q_col is None or a_col is None:
        raise SystemExit(
            "prepare_grpo_data: rows need a prompt column"
            f" ({question_column or ' / '.join(QUESTION_COLUMNS)}) and a reference-answer"
            f" column ({answer_column or ' / '.join(ANSWER_COLUMNS)}); got {sorted(row)}"
        )
    return {"problem": str(row[q_col]), "answer": str(row[a_col])}


def convert(
    reward_type: str,
    data_path: str,
    out_path: str,
    question_column: str = "",
    answer_column: str = "",
) -> int:
    if reward_type not in ("math", "tool_call"):
        raise SystemExit(f"prepare_grpo_data: unknown reward_type {reward_type!r}")
    written = 0
    rows = _rows(data_path)
    with open(out_path, "w") as out:
        for row in rows:
            record = (
                _math_record(row, question_column, answer_column) if reward_type == "math" else row
            )
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            written += 1
    if written == 0:
        raise SystemExit(f"prepare_grpo_data: {data_path} has no rows")
    return written


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    args = [*sys.argv[1:], "", ""]
    n = convert(args[0], args[1], args[2], args[3], args[4])
    print(f"prepare_grpo_data: wrote {n} rows to {args[2]}")
