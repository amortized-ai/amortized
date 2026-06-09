"""Tests for the container runner RunContext."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "containers"))

from shared.artifacts import save_to_local
from shared.context import RunContext
from shared.events import emit_stdout


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    return work


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config = {
        "config": {
            "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
            "data_path": "./data.jsonl",
            "ckpt_output_dir": "./outputs",
        },
        "artifacts": {"dataset": "/shared/datasets/train.jsonl"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


class TestRunContextFromEnvironment:
    def test_reads_config_and_env(self, tmp_work_dir: Path, config_file: Path) -> None:
        env = {
            "AMORTIZED_JOB_ID": "job-123",
            "AMORTIZED_WORK_DIR": str(tmp_work_dir),
            "AMORTIZED_CONFIG_PATH": str(config_file),
        }
        with patch.dict(os.environ, env, clear=False):
            ctx = RunContext.from_environment()

        assert ctx.job_id == "job-123"
        assert ctx.work_dir == tmp_work_dir
        assert ctx.config["model_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert ctx.artifacts["dataset"] == "/shared/datasets/train.jsonl"

    def test_events_url_optional(self, tmp_work_dir: Path, config_file: Path) -> None:
        env = {
            "AMORTIZED_JOB_ID": "job-456",
            "AMORTIZED_WORK_DIR": str(tmp_work_dir),
            "AMORTIZED_CONFIG_PATH": str(config_file),
        }
        with patch.dict(os.environ, env, clear=False):
            ctx = RunContext.from_environment()

        assert ctx._events_url is None

    def test_events_url_set(self, tmp_work_dir: Path, config_file: Path) -> None:
        env = {
            "AMORTIZED_JOB_ID": "job-789",
            "AMORTIZED_WORK_DIR": str(tmp_work_dir),
            "AMORTIZED_CONFIG_PATH": str(config_file),
            "AMORTIZED_EVENTS_URL": "http://localhost:9400/events",
        }
        with patch.dict(os.environ, env, clear=False):
            ctx = RunContext.from_environment()

        assert ctx._events_url == "http://localhost:9400/events"


class TestRunContextEmit:
    def test_emit_writes_to_stdout(
        self, tmp_work_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ctx = RunContext(
            job_id="job-emit",
            work_dir=tmp_work_dir,
            config={},
        )
        ctx.emit("progress", {"message": "hello"})

        captured = capsys.readouterr()
        line = json.loads(captured.out.strip())
        assert line["type"] == "progress"
        assert line["job_id"] == "job-emit"
        assert line["data"]["message"] == "hello"
        assert "timestamp" in line


class TestRunContextSaveArtifact:
    def test_saves_file_artifact(self, tmp_work_dir: Path, tmp_path: Path) -> None:
        source = tmp_path / "model.bin"
        source.write_bytes(b"fake model weights")

        ctx = RunContext(job_id="job-art", work_dir=tmp_work_dir, config={})
        ctx.save_artifact("model", source)

        dest = tmp_work_dir / "artifacts" / "model"
        assert dest.exists()
        assert dest.read_bytes() == b"fake model weights"

    def test_saves_directory_artifact(self, tmp_work_dir: Path, tmp_path: Path) -> None:
        source_dir = tmp_path / "checkpoint"
        source_dir.mkdir()
        (source_dir / "weights.bin").write_bytes(b"weights")
        (source_dir / "config.json").write_text('{"r": 16}')

        ctx = RunContext(job_id="job-dir", work_dir=tmp_work_dir, config={})
        ctx.save_artifact("checkpoint", source_dir)

        dest = tmp_work_dir / "artifacts" / "checkpoint"
        assert dest.is_dir()
        assert (dest / "weights.bin").read_bytes() == b"weights"
        assert (dest / "config.json").read_text() == '{"r": 16}'


class TestRunContextCancellation:
    def test_not_cancelled_by_default(self, tmp_work_dir: Path) -> None:
        ctx = RunContext(
            job_id="job-cancel",
            work_dir=tmp_work_dir,
            config={},
            _cancel_file=tmp_work_dir / ".cancel",
        )
        assert ctx.is_cancelled() is False

    def test_cancelled_when_file_exists(self, tmp_work_dir: Path) -> None:
        cancel = tmp_work_dir / ".cancel"
        cancel.touch()

        ctx = RunContext(
            job_id="job-cancel2",
            work_dir=tmp_work_dir,
            config={},
            _cancel_file=cancel,
        )
        assert ctx.is_cancelled() is True


class TestEmitStdout:
    def test_writes_json_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        event = {"type": "test", "data": {"key": "value"}}
        emit_stdout(event)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed == event


class TestSaveToLocal:
    def test_copies_file(self, tmp_path: Path) -> None:
        source = tmp_path / "src" / "data.jsonl"
        source.parent.mkdir()
        source.write_text('{"text": "hello"}')

        storage = tmp_path / "storage"
        storage.mkdir()

        dest = save_to_local("data.jsonl", source, storage)
        assert dest == storage / "data.jsonl"
        assert dest.read_text() == '{"text": "hello"}'

    def test_copies_directory(self, tmp_path: Path) -> None:
        source = tmp_path / "src" / "model"
        source.mkdir(parents=True)
        (source / "weights.bin").write_bytes(b"w")

        storage = tmp_path / "storage"
        storage.mkdir()

        dest = save_to_local("model", source, storage)
        assert dest.is_dir()
        assert (dest / "weights.bin").read_bytes() == b"w"
