"""SDG uploads the directory Data Designer actually wrote the dataset to.

drop_columns edits the main dataset in place (its processors-files folder stays
empty); only schema_transform writes a new dataset under processors-files/<name>.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from amortized.jobs import sdg as sdg_builder

PARQUET = "/amortized/work/dataset/parquet-files"
DROP = {"name": "drop_feature", "processor_type": "drop_columns", "column_names": ["feature"]}
TRANSFORM = {"name": "to_messages", "processor_type": "schema_transform", "template": {}}


@pytest.mark.parametrize(
    ("processors", "expected"),
    [
        ([], PARQUET),
        ([DROP], PARQUET),
        ([TRANSFORM], "/amortized/work/dataset/processors-files/to_messages"),
        ([TRANSFORM, DROP], "/amortized/work/dataset/processors-files/to_messages"),
        ([DROP, TRANSFORM], "/amortized/work/dataset/processors-files/to_messages"),
    ],
)
def test_output_dir_follows_schema_transform_only(processors: list, expected: str) -> None:
    assert sdg_builder.sdg_output_dir({"processors": processors}) == expected


@pytest.mark.asyncio
async def test_build_checks_output_before_upload_for_drop_columns() -> None:
    config = {
        "columns": [
            {
                "name": "category",
                "column_type": "sampler",
                "sampler_type": "category",
                "params": {"values": ["a", "b"]},
            },
        ],
        "processors": [DROP],
        "mode": "preview",
        "num_records": 5,
    }
    result = await sdg_builder.build({"id": "job-drop"}, config, {})
    check, upload = result.post_commands
    assert check == sdg_builder.require_nonempty_dir_cmd(PARQUET)
    assert f"-l {PARQUET}" in upload
    assert "-a generated_data" in upload


def _run(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["sh", "-c", f"{cmd} && echo uploaded"], capture_output=True, text=True)


def test_nonempty_check_fails_on_empty_and_missing_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    for path in (empty, tmp_path / "missing"):
        proc = _run(sdg_builder.require_nonempty_dir_cmd(str(path)))
        assert proc.returncode != 0
        assert "uploaded" not in proc.stdout
        assert "SDG produced no output files" in proc.stderr


def test_nonempty_check_passes_with_files(tmp_path: Path) -> None:
    (tmp_path / "batch_00000.parquet").write_bytes(b"PAR1")
    proc = _run(sdg_builder.require_nonempty_dir_cmd(str(tmp_path)))
    assert proc.returncode == 0
    assert "uploaded" in proc.stdout
