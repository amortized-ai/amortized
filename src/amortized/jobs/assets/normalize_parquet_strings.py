"""Rewrite parquet files whose string columns are Arrow ``large_string`` as ``string``.

Data Designer writes SDG output with ``large_string`` columns. training-hub's
embedding_sft encodes string labels with ``datasets.map``, which keeps the
column's ``large_string`` type, so the encoded labels stay strings and
sentence-transformers fails with "too many dimensions 'str'". With plain
``string`` columns the same encoding yields int64 labels.

Usage: python3 normalize_parquet_strings.py <data_path>
"""

from __future__ import annotations

import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def _parquet_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path] if path.endswith(".parquet") else []
    found: list[str] = []
    for root, _dirs, files in os.walk(path):
        found.extend(os.path.join(root, name) for name in files if name.endswith(".parquet"))
    return sorted(found)


def normalize(path: str) -> list[str]:
    """Cast top-level large_string columns to string in place; return the rewritten files."""
    rewritten = []
    for file in _parquet_files(path):
        table = pq.read_table(file)
        if not any(pa.types.is_large_string(field.type) for field in table.schema):
            continue
        schema = pa.schema(
            [
                pa.field(field.name, pa.string(), field.nullable)
                if pa.types.is_large_string(field.type)
                else field
                for field in table.schema
            ]
        )
        pq.write_table(table.cast(schema), file)
        rewritten.append(file)
    return rewritten


def main() -> int:
    target = sys.argv[1]
    # A non-local data_path (e.g. a HuggingFace dataset id) is loaded by training-hub itself.
    if not os.path.exists(target):
        print(f"normalize_parquet_strings: {target} is not a local path, skipping")
        return 0
    for file in normalize(target):
        print(f"normalize_parquet_strings: cast large_string columns to string in {file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
