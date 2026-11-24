"""lora_grpo / grpo (training-hub verl backend): validation, thub config, data prep, API checks."""

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

ASSETS = Path(amortized.jobs.__file__).parent / "assets"
ASSET = ASSETS / "prepare_grpo_data.py"
_LMO_CMD = (
    "python3 /amortized/grpo_language_model_only.py"
    " /amortized/config.yaml /amortized/work/config.yaml"
)


def _asset(path: Path = ASSET) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
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
        assert result.command == ["thub", "lora-grpo", "--config", "/amortized/work/config.yaml"]
        assert result.resources.gpus == 2
        assert "prepare_grpo_data.py" in files
        assert result.pre_commands[-1] == _LMO_CMD
        assert result.pre_commands[-2] == (
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
        assert result.pre_commands[-2].startswith(
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
        assert result.pre_commands == [_LMO_CMD]


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


# The math demo config (Qwen3.5-2B on MATH-500): no SDG job or upload needed.
_DEMO_CONFIG: dict[str, Any] = {
    "algorithm": "lora_grpo",
    "model_name_or_path": "Qwen/Qwen3.5-2B",
    "data_path": "HuggingFaceH4/MATH-500",
    "reward_type": "math",
    "n_train": 512,
    "num_train_epochs": 1,
    "prompt_batch_size": 64,
    "group_size": 8,
    "max_prompt_length": 512,
    "max_tokens": 1024,
    "temperature": 1.0,
    "learning_rate": 0.0001,
    "lora_r": 32,
    "lora_alpha": 64,
    "gpu_memory_utilization": 0.4,
    "language_model_only": True,
    "nproc_per_node": 1,
    "topic": "math reasoning grpo demo",
}


class TestDemoPayload:
    def _config(self) -> dict[str, Any]:
        return dict(_DEMO_CONFIG)

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
        assert "HuggingFaceH4/MATH-500" in result.pre_commands[-2]
        assert result.resources.gpus == 1


class TestFullGRPO:
    """algorithm="grpo": full-parameter GRPO (`thub grpo`), same data and rewards as lora_grpo."""

    def test_rollout_split_check_applies(self) -> None:
        with pytest.raises(ValidationError, match="divisible"):
            TrainingJobRequest(
                algorithm="grpo", model_name_or_path="m", prompt_batch_size=3, group_size=3
            )

    def test_thub_config_drops_lora_and_backend(self) -> None:
        parsed = yaml.safe_load(
            _training_hub_config_yaml(
                "grpo",
                {
                    **_GRPO_BASE,
                    "algorithm": "grpo",
                    "num_train_epochs": 2,
                    "nproc_per_node": 4,
                    "lora_r": 16,
                    "max_grad_norm": 1.0,
                    "backend": "art",
                    "concurrency": 8,
                },
            )
        )
        assert parsed["num_iterations"] == 2
        assert parsed["n_gpus"] == 4
        assert parsed["reward_type"] == "math"
        assert parsed["max_prompt_length"] == 2048
        assert parsed["n_val"] == 0
        for key in ("backend", "lora_r", "max_grad_norm", "concurrency", "num_epochs"):
            assert key not in parsed

    @pytest.mark.asyncio
    async def test_build_converts_data_and_exports_hf_format(self) -> None:
        config = {
            "algorithm": "grpo",
            "model_name_or_path": "m",
            "data_path": "/amortized/work/data/generated_data",
            "nproc_per_node": 2,
        }
        files: dict[str, str] = {}
        result = await training_builder.build({"id": "f1", "type": "training"}, config, files)
        assert result.command == ["thub", "grpo", "--config", "/amortized/work/config.yaml"]
        assert result.resources.gpus == 2
        assert "prepare_grpo_data.py" in files
        assert yaml.safe_load(files["config.yaml"])["data_path"] == (
            "/amortized/work/grpo_data.jsonl"
        )
        post = result.post_commands[0]
        assert "python3 -m verl.model_merger merge --backend fsdp" in post
        assert "lora_adapter" not in post
        assert post.endswith(
            "mlflow artifacts log-artifacts -l /amortized/work/grpo_export"
            " -r $MLFLOW_RUN_ID -a model"
        )

    @pytest.mark.asyncio
    async def test_merges_last_checkpoint_into_hf_format(self, tmp_path: Path) -> None:
        import subprocess

        for step in (2, 10, 9):
            (tmp_path / "checkpoints" / f"global_step_{step}" / "actor").mkdir(parents=True)
        config = {
            "algorithm": "grpo",
            "model_name_or_path": "m",
            "data_path": "org/ds",
            "reward_type": "tool_call",
            "output_dir": str(tmp_path),
        }
        result = await training_builder.build({"id": "f2", "type": "training"}, config, {})
        merge = result.post_commands[0].split(" && mlflow")[0]
        # Swap the merger for echo to see which checkpoint and target the shell resolves.
        script = merge.replace("python3 -m verl.model_merger", "echo")
        out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=True)
        assert f"--local_dir {tmp_path}/checkpoints/global_step_10/actor" in out.stdout
        assert "--target_dir /amortized/work/grpo_export/hf_format/global_step_10" in out.stdout

    @pytest.mark.asyncio
    async def test_lora_knobs_rejected(self) -> None:
        from amortized.api import jobs as jobs_api

        config = {"algorithm": "grpo", "model_name_or_path": "m", "lora_r": 16}
        errors = await jobs_api._validate_training_data(config, "", None)  # type: ignore[arg-type]
        assert any("lora_r" in e and "lora_grpo" in e for e in errors)
        config = {"algorithm": "lora_grpo", "model_name_or_path": "m", "lora_r": 16}
        errors = await jobs_api._validate_training_data(config, "", None)  # type: ignore[arg-type]
        assert not any("lora_r" in e for e in errors)

    @pytest.mark.asyncio
    async def test_shape_check_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        errors = await TestGRPOApiChecks._violations(
            monkeypatch, {"messages": []}, {"algorithm": "grpo"}
        )
        assert errors and errors[0].startswith("training data: grpo needs")


class TestGRPOLanguageModelOnly:
    """Multimodal checkpoints (e.g. Qwen3.5) get language_model_only, detected in the pod."""

    @staticmethod
    def _fake_transformers(monkeypatch: pytest.MonkeyPatch, model_config: object) -> None:
        import sys
        from types import SimpleNamespace

        class LlamaConfig:
            pass

        class Qwen35TextConfig:
            pass

        class Qwen35Config:
            text_config = Qwen35TextConfig()

        class Gemma3Config:
            text_config = Qwen35TextConfig()

        class Qwen3VLConfig:
            pass

        def causal(name: str) -> type:
            return type(name, (), {})

        mapping = {
            LlamaConfig: causal("LlamaForCausalLM"),
            Qwen35TextConfig: causal("Qwen3_5ForCausalLM"),
            Gemma3Config: causal("Gemma3ForConditionalGeneration"),
        }
        image_text = {Qwen3VLConfig: causal("Qwen3VLForConditionalGeneration")}
        configs = {
            "llama": LlamaConfig(),
            "qwen3.5": Qwen35Config(),
            "gemma3": Gemma3Config(),
            "qwen3-vl": Qwen3VLConfig(),
        }
        auto = SimpleNamespace(
            MODEL_FOR_CAUSAL_LM_MAPPING=mapping,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING=image_text,
        )
        fake = SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda path, **_kw: configs[str(model_config or path)]
            ),
            models=SimpleNamespace(auto=auto),
        )
        monkeypatch.setitem(sys.modules, "transformers", fake)
        monkeypatch.setitem(sys.modules, "transformers.models", fake.models)
        monkeypatch.setitem(sys.modules, "transformers.models.auto", auto)

    def _patched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, **extra: Any
    ) -> dict[str, Any]:
        self._fake_transformers(monkeypatch, None)
        src, dst = tmp_path / "in.yaml", tmp_path / "out.yaml"
        src.write_text(yaml.safe_dump({"model_path": model, "reward_type": "math", **extra}))
        _asset(ASSETS / "grpo_language_model_only.py").patch(str(src), str(dst))
        return dict(yaml.safe_load(dst.read_text()))

    @pytest.mark.parametrize("model", ["qwen3.5", "gemma3", "qwen3-vl"])
    def test_multimodal_gets_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        patched = self._patched(tmp_path, monkeypatch, model)
        assert patched["language_model_only"] is True
        assert patched["reward_type"] == "math"

    def test_text_model_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert "language_model_only" not in self._patched(tmp_path, monkeypatch, "llama")

    def test_explicit_value_kept(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patched = self._patched(tmp_path, monkeypatch, "qwen3.5", language_model_only=False)
        assert patched["language_model_only"] is False

    @pytest.mark.asyncio
    async def test_build_ships_asset_for_both_algorithms(self) -> None:
        for algorithm in ("lora_grpo", "grpo"):
            config = {"algorithm": algorithm, "model_name_or_path": "Qwen/Qwen3.5-2B"}
            files: dict[str, str] = {}
            result = await training_builder.build({"id": "l1", "type": "training"}, config, files)
            assert "grpo_language_model_only.py" in files
            assert result.pre_commands[-1] == _LMO_CMD
            assert result.command[-1] == "/amortized/work/config.yaml"

    @pytest.mark.asyncio
    async def test_other_algorithms_untouched(self) -> None:
        config = {"algorithm": "lora_sft", "model_name_or_path": "Qwen/Qwen3.5-2B"}
        files: dict[str, str] = {}
        result = await training_builder.build({"id": "l2", "type": "training"}, config, files)
        assert "grpo_language_model_only.py" not in files
        assert result.command[-1] == "/amortized/config.yaml"
