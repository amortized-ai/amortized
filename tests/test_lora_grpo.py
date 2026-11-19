"""lora_grpo (training-hub verl backend): config validation, thub config, data prep, API checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

import amortized.jobs
from amortized.jobs import training as training_builder
from amortized.jobs.training import _training_hub_config_yaml
from amortized.models import TrainingJobRequest

ASSET = Path(amortized.jobs.__file__).parent / "assets" / "prepare_grpo_data.py"


def _asset() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_grpo_data", ASSET)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grpo(**kw: Any) -> TrainingJobRequest:
    return TrainingJobRequest(algorithm="lora_grpo", model_name_or_path="m", **kw)


class TestGRPOValidation:
    def test_defaults_accepted(self) -> None:
        assert _grpo().reward_type is None

    @pytest.mark.parametrize("reward", ["math", "tool_call"])
    def test_training_hub_reward_types_accepted(self, reward: str) -> None:
        assert _grpo(reward_type=reward).reward_type == reward

    @pytest.mark.parametrize("reward", ["verl", "custom", "python"])
    def test_other_reward_types_rejected(self, reward: str) -> None:
        with pytest.raises(ValidationError):
            _grpo(reward_type=reward)

    def test_rollouts_must_split_across_workers(self) -> None:
        with pytest.raises(ValidationError, match="divisible"):
            _grpo(prompt_batch_size=3, group_size=3)
        with pytest.raises(ValidationError, match="divisible"):
            _grpo(prompt_batch_size=10, group_size=4, nproc_per_node=8)
        assert _grpo(prompt_batch_size=16, group_size=4, nproc_per_node=2).group_size == 4

    def test_other_algorithms_ignore_grpo_checks(self) -> None:
        req = TrainingJobRequest(
            algorithm="lora_sft", model_name_or_path="m", group_size=3, prompt_batch_size=3
        )
        assert req.algorithm == "lora_sft"


_GRPO_BASE: dict[str, Any] = {
    "algorithm": "lora_grpo",
    "model_name_or_path": "Qwen/Qwen3-0.6B",
    "data_path": "/amortized/work/grpo_data.jsonl",
}


class TestGRPOThubConfig:
    def _parsed(self, **kw: Any) -> dict[str, Any]:
        return dict(yaml.safe_load(_training_hub_config_yaml("lora_grpo", {**_GRPO_BASE, **kw})))

    def test_defaults(self) -> None:
        parsed = self._parsed()
        assert parsed["model_path"] == "Qwen/Qwen3-0.6B"
        assert parsed["ckpt_output_dir"] == "/amortized/work/output"
        assert parsed["reward_type"] == "math"
        assert parsed["backend"] == "verl"
        assert parsed["n_gpus"] == 1
        assert parsed["max_prompt_length"] == 2048
        assert parsed["n_val"] == 0
        for key in ("data_output_dir", "max_seq_len", "micro_batch_size", "nproc_per_node"):
            assert key not in parsed

    def test_maps_amortized_fields(self) -> None:
        parsed = self._parsed(
            num_train_epochs=3,
            nproc_per_node=4,
            per_device_train_batch_size=2,
            max_length=4096,
            gradient_accumulation_steps=2,
            text_column="text",
            label_column="category",
            lora_r=32,
            group_size=4,
        )
        assert parsed["num_iterations"] == 3
        assert parsed["n_gpus"] == 4
        assert parsed["lora_r"] == 32
        assert parsed["group_size"] == 4
        for key in (
            "num_epochs",
            "micro_batch_size",
            "max_seq_len",
            "gradient_accumulation_steps",
            "text_column",
            "label_column",
        ):
            assert key not in parsed

    def test_explicit_values_win(self) -> None:
        parsed = self._parsed(reward_type="tool_call", max_prompt_length=4096, n_val=32)
        assert parsed["reward_type"] == "tool_call"
        assert parsed["max_prompt_length"] == 4096
        assert parsed["n_val"] == 32


class TestGRPOBuild:
    @pytest.mark.asyncio
    async def test_converts_downloaded_dataset(self) -> None:
        config = {
            "algorithm": "lora_grpo",
            "model_name_or_path": "m",
            "data_path": "/amortized/work/data/generated_data",
            "label_column": "category",
            "nproc_per_node": 2,
        }
        files: dict[str, str] = {}
        result = await training_builder.build({"id": "g1", "type": "training"}, config, files)
        assert result.command == ["thub", "lora-grpo", "--config", "/amortized/config.yaml"]
        assert result.resources.gpus == 2
        assert "prepare_grpo_data.py" in files
        assert result.pre_commands[-1] == (
            "python3 /amortized/prepare_grpo_data.py math /amortized/work/data/generated_data"
            " /amortized/work/grpo_data.jsonl '' category"
        )
        parsed = yaml.safe_load(files["config.yaml"])
        assert parsed["data_path"] == "/amortized/work/grpo_data.jsonl"
        assert parsed["n_gpus"] == 2
        # The persisted config keeps the real source so clones re-download the dataset.
        assert result.resolved_config["data_path"] == "/amortized/work/data/generated_data"
        # Exported like lora_sft: only the last verl adapter dir becomes model/.
        assert result.post_commands == [
            "ADAPTER=$(ls -d /amortized/work/output/checkpoints/global_step_*/actor/lora_adapter"
            ' | sort -V | tail -1) && test -n "$ADAPTER"'
            ' && mlflow artifacts log-artifacts -l "$ADAPTER" -r $MLFLOW_RUN_ID -a model'
        ]

    @pytest.mark.asyncio
    async def test_last_adapter_selected_in_shell(self, tmp_path: Path) -> None:
        import subprocess

        for step in (2, 10, 9):
            adapter = tmp_path / "checkpoints" / f"global_step_{step}" / "actor" / "lora_adapter"
            adapter.mkdir(parents=True)
        config = {
            "algorithm": "lora_grpo",
            "model_name_or_path": "m",
            "data_path": "org/ds",
            "output_dir": str(tmp_path),
        }
        result = await training_builder.build({"id": "g3", "type": "training"}, config, {})
        script = result.post_commands[0].split(" && mlflow")[0] + ' && echo "$ADAPTER"'
        out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=True)
        assert out.stdout.strip().endswith("global_step_10/actor/lora_adapter")

    @pytest.mark.asyncio
    async def test_math_hf_dataset_id_is_converted(self) -> None:
        # training-hub's math prep only opens local files, so an HF ID goes through the asset.
        config = {"algorithm": "lora_grpo", "model_name_or_path": "m", "data_path": "org/ds"}
        files: dict[str, str] = {}
        result = await training_builder.build({"id": "g2", "type": "training"}, config, files)
        assert "prepare_grpo_data.py" in files
        assert result.pre_commands[-1].startswith(
            "python3 /amortized/prepare_grpo_data.py math org/ds "
        )
        parsed = yaml.safe_load(files["config.yaml"])
        assert parsed["data_path"] == "/amortized/work/grpo_data.jsonl"

    @pytest.mark.asyncio
    async def test_tool_call_hf_dataset_id_passes_through(self) -> None:
        # training-hub loads HF tool-call datasets itself.
        config = {
            "algorithm": "lora_grpo",
            "model_name_or_path": "m",
            "data_path": "org/ds",
            "reward_type": "tool_call",
        }
        files: dict[str, str] = {}
        result = await training_builder.build({"id": "g4", "type": "training"}, config, files)
        assert "prepare_grpo_data.py" not in files
        assert yaml.safe_load(files["config.yaml"])["data_path"] == "org/ds"
        assert not result.pre_commands


class TestPrepareGRPOData:
    def _read(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_math_parquet_dir_auto_columns(self, tmp_path: Path) -> None:
        data = tmp_path / "generated_data"
        data.mkdir()
        table = pa.table(
            {
                "text": pa.array(["refund please", "great app"], pa.large_string()),
                "category": pa.array(["billing", "praise"], pa.large_string()),
            }
        )
        pq.write_table(table, data / "batch_00000.parquet")
        out = tmp_path / "out.jsonl"
        assert _asset().convert("math", str(data), str(out)) == 2
        assert self._read(out)[0] == {"problem": "refund please", "answer": "billing"}

    def test_math_explicit_columns_and_non_string_answer(self, tmp_path: Path) -> None:
        src = tmp_path / "d.jsonl"
        src.write_text(json.dumps({"q": "1+1?", "a": 2, "question": "ignored"}) + "\n")
        out = tmp_path / "out.jsonl"
        _asset().convert("math", str(src), str(out), "q", "a")
        assert self._read(out)[0] == {"problem": "1+1?", "answer": "2"}

    def test_math_missing_answer_column_fails(self, tmp_path: Path) -> None:
        src = tmp_path / "d.jsonl"
        src.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
        with pytest.raises(SystemExit, match="reference-answer column"):
            _asset().convert("math", str(src), str(tmp_path / "out.jsonl"))

    def test_tool_call_rows_pass_through(self, tmp_path: Path) -> None:
        data = tmp_path / "generated_data"
        data.mkdir()
        row = {
            "question": "weather in Paris?",
            "tools": [{"name": "get_weather", "parameters": {"city": "string"}}],
            "target_tool_name": "get_weather",
            "target_arguments": {"city": "Paris"},
        }
        pq.write_table(pa.Table.from_pylist([row]), data / "batch_00000.parquet")
        out = tmp_path / "out.jsonl"
        _asset().convert("tool_call", str(data), str(out))
        assert self._read(out) == [row]

    def test_math_hf_dataset_uses_train_else_first_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import datasets

        def fake_load(dataset_id: str) -> dict[str, list[dict[str, str]]]:
            assert dataset_id == "HuggingFaceH4/MATH-500"
            return {"test": [{"problem": "1+1?", "answer": "2", "solution": "..."}]}

        monkeypatch.setattr(datasets, "load_dataset", fake_load)
        out = tmp_path / "out.jsonl"
        _asset().convert("math", "HuggingFaceH4/MATH-500", str(out))
        assert self._read(out) == [{"problem": "1+1?", "answer": "2"}]

    def test_empty_dir_fails(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="no parquet"):
            _asset().convert("math", str(tmp_path), str(tmp_path / "out.jsonl"))


class TestGRPOApiChecks:
    @staticmethod
    async def _violations(monkeypatch: pytest.MonkeyPatch, row: dict, config: dict) -> list[str]:
        from amortized.api import datasets as datasets_api
        from amortized.api import jobs as jobs_api

        async def _find(_client: object, _run: object) -> list[str]:
            return ["data.jsonl"]

        class _Client:
            async def get_artifact(self, _run: object, _path: object) -> bytes:
                return b""

        monkeypatch.setattr(datasets_api, "_mlflow_client", lambda: _Client())
        monkeypatch.setattr(datasets_api, "_find_dataset_artifacts", _find)
        monkeypatch.setattr(datasets_api, "_parse_records", lambda _p, _b: [row])
        return await jobs_api._training_shape_violations(
            {"data_run_id": "r" * 32, "algorithm": "lora_grpo", **config}, "", None
        )

    @pytest.mark.asyncio
    async def test_shape_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not await self._violations(monkeypatch, {"question": "q", "answer": "a"}, {})
        assert not await self._violations(monkeypatch, {"text": "t", "category": "c"}, {})
        errors = await self._violations(monkeypatch, {"messages": []}, {})
        assert errors and "reference-answer" in errors[0]
        errors = await self._violations(
            monkeypatch, {"text": "t", "category": "c"}, {"label_column": "gold"}
        )
        assert errors and "gold" in errors[0]

    @pytest.mark.asyncio
    async def test_tool_call_shape_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = {"reward_type": "tool_call"}
        assert not await self._violations(monkeypatch, {"messages": []}, cfg)
        single = {"question": "q", "tools": [], "target_tool_name": "f", "target_arguments": {}}
        assert not await self._violations(monkeypatch, single, cfg)
        errors = await self._violations(monkeypatch, {"question": "q", "answer": "a"}, cfg)
        assert errors and "tool-call traces" in errors[0]

    @pytest.mark.asyncio
    async def test_callable_params_rejected(self) -> None:
        from amortized.api import jobs as jobs_api

        config = {
            "algorithm": "lora_grpo",
            "model_name_or_path": "m",
            "reward_fn": "evil.module.fn",
        }
        errors = await jobs_api._validate_training_data(config, "", None)  # type: ignore[arg-type]
        assert any("reward_fn" in e and "arbitrary code" in e for e in errors)


class TestDemoPayload:
    PAYLOAD = (
        Path(__file__).resolve().parent.parent
        / "agents"
        / "training"
        / "skills"
        / "grpo"
        / "reference-payload.json"
    )

    def _config(self) -> dict[str, Any]:
        config = json.loads(self.PAYLOAD.read_text())
        config.pop("_meta")
        return dict(config)

    def test_payload_validates(self) -> None:
        req = TrainingJobRequest(**self._config())
        assert req.algorithm == "lora_grpo"
        assert req.reward_type == "math"

    @pytest.mark.asyncio
    async def test_payload_builds(self) -> None:
        files: dict[str, str] = {}
        result = await training_builder.build(
            {"id": "demo", "type": "training"}, self._config(), files
        )
        parsed = yaml.safe_load(files["config.yaml"])
        assert parsed["model_path"] == "Qwen/Qwen3.5-2B"
        assert parsed["data_path"] == "/amortized/work/grpo_data.jsonl"
        assert parsed["num_iterations"] == 1
        assert parsed["n_train"] == 512
        assert parsed["language_model_only"] is True
        assert "HuggingFaceH4/MATH-500" in result.pre_commands[-1]
        assert result.resources.gpus == 1
