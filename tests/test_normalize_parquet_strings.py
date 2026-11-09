"""embedding_sft data normalization: large_string parquet columns become string.

Data Designer writes large_string columns; with them, training-hub's label
encoding keeps the label column a string and training crashes with
"too many dimensions 'str'".
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

import amortized.jobs

ASSET = Path(amortized.jobs.__file__).parent / "assets" / "normalize_parquet_strings.py"


def _asset() -> ModuleType:
    spec = importlib.util.spec_from_file_location("normalize_parquet_strings", ASSET)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sdg_like(data_dir: Path) -> Path:
    data_dir.mkdir()
    path = data_dir / "batch_00000.parquet"
    table = pa.table(
        {
            "category": pa.array(
                ["praise", "bug_report", "praise", "complaint"], pa.large_string()
            ),
            "text": pa.array(["love it", "it crashes", "great", "too slow"], pa.large_string()),
        }
    )
    pq.write_table(table, path)
    return path


def test_casts_large_string_columns_to_string(tmp_path: Path) -> None:
    path = _write_sdg_like(tmp_path / "generated_data")
    assert _asset().normalize(str(tmp_path / "generated_data")) == [str(path)]
    schema = pq.read_schema(path)
    assert schema.field("category").type == pa.string()
    assert schema.field("text").type == pa.string()
    assert pq.read_table(path).column("category").to_pylist()[1] == "bug_report"


def test_label_encoding_yields_ints_after_normalize(tmp_path: Path) -> None:
    data_dir = tmp_path / "generated_data"
    _write_sdg_like(data_dir)
    _asset().normalize(str(data_dir))
    # Same steps as training-hub 0.11.1 embedding_sft._load_dataset for string labels.
    ds = load_dataset(str(data_dir), split="train", cache_dir=str(tmp_path / "cache"))
    ds = ds.rename_column("category", "label")
    mapping = {value: idx for idx, value in enumerate(sorted(set(ds["label"])))}
    ds = ds.map(lambda batch: {"label": [mapping[v] for v in batch["label"]]}, batched=True)
    assert ds.features["label"].dtype == "int64"
    assert ds[0]["label"] == mapping["praise"]


def test_plain_string_files_are_left_alone(tmp_path: Path) -> None:
    data_dir = tmp_path / "split"
    data_dir.mkdir()
    pq.write_table(pa.table({"category": ["a"], "text": ["x"]}), data_dir / "batch_00000.parquet")
    assert _asset().normalize(str(data_dir)) == []


def test_non_local_path_is_skipped() -> None:
    proc = subprocess.run(
        [sys.executable, str(ASSET), "org/some-hf-dataset"], capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert "skipping" in proc.stdout
