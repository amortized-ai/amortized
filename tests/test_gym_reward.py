"""reward_type=gym GRPO reward: Gym verifier sidecar, reward function, data prep, API checks."""

from __future__ import annotations

import importlib.util
import inspect
import json
import textwrap
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

import amortized.config as config_mod
import amortized.jobs
from amortized.backends import JobSpec, Resources, Sidecar
from amortized.backends.kubernetes import KubernetesBackend
from amortized.gym_verifiers import GYM_VERIFIERS
from amortized.jobs import training as training_builder
from amortized.models import TrainingJobRequest

ASSETS = Path(amortized.jobs.__file__).parent / "assets"
_GYM_SIDECAR = Sidecar(name="gym", image="reg/gym:t", port=8000, env={"GYM_SERVER": "mcqa"})


def _asset(name: str) -> ModuleType:
    path = ASSETS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


# --- Kubernetes backend -------------------------------------------------------------------


class TestSidecarPodSpec:
    @staticmethod
    def _pod(sidecars: list[Sidecar]) -> Any:
        backend = KubernetesBackend(namespace="test-jobs", image_pull_policy="IfNotPresent")
        spec = JobSpec(
            job_id="j1", command=["true"], resources=Resources(gpus=1), sidecars=sidecars
        )
        return backend._build_pod_spec(spec, "amortized-j1")

    def test_no_sidecars_leaves_pod_unchanged(self) -> None:
        pod = self._pod([])
        assert pod.init_containers is None
        assert [c.name for c in pod.containers] == ["job"]

    def test_sidecar_is_a_native_sidecar_with_startup_probe(self) -> None:
        pod = self._pod([_GYM_SIDECAR])
        [sidecar] = pod.init_containers
        assert sidecar.name == "gym"
        assert sidecar.image == "reg/gym:t"
        # restartPolicy Always on an init container = native sidecar: started before the job
        # container, stopped after it exits, so the Job still completes.
        assert sidecar.restart_policy == "Always"
        assert sidecar.image_pull_policy == "IfNotPresent"
        assert sidecar.startup_probe.tcp_socket.port == 8000
        env = {e.name: e.value for e in sidecar.env}
        assert env == {"GYM_SERVER": "mcqa", "NVIDIA_VISIBLE_DEVICES": "void"}
        assert sidecar.resources.requests == {"cpu": "250m", "memory": "256Mi"}
        assert sidecar.resources.limits == {"memory": "512Mi"}
        assert sidecar.security_context.allow_privilege_escalation is False
        # The job container keeps the GPU.
        assert pod.containers[0].resources.limits == {"nvidia.com/gpu": "1"}


def _pending_pod(init_waiting: str | None, main_waiting: str | None = None) -> SimpleNamespace:
    def status(name: str, reason: str | None) -> SimpleNamespace:
        waiting = SimpleNamespace(reason=reason, message="detail") if reason else None
        return SimpleNamespace(
            name=name,
            image=f"reg/{name}:t",
            state=SimpleNamespace(waiting=waiting, terminated=None),
        )

    return SimpleNamespace(
        status=SimpleNamespace(
            phase="Pending",
            init_container_statuses=[status("gym", init_waiting)],
            container_statuses=[status("job", main_waiting)],
        )
    )


class TestStuckSidecar:
    @staticmethod
    def _reason(monkeypatch: pytest.MonkeyPatch, pod: SimpleNamespace) -> str | None:
        import kubernetes_asyncio.client as k8s_client

        class _Core:
            async def list_namespaced_pod(self, namespace: str, label_selector: str) -> Any:
                return SimpleNamespace(items=[pod])

        monkeypatch.setattr(k8s_client, "CoreV1Api", lambda api_client: _Core())
        backend = KubernetesBackend(namespace="test-jobs")
        reason: str | None = _run(backend._get_stuck_pod_reason("job1", api_client=None))
        return reason

    def test_sidecar_image_pull_failure_is_stuck(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reason = self._reason(monkeypatch, _pending_pod("ImagePullBackOff"))
        assert reason is not None and "reg/gym:t" in reason and "ImagePullBackOff" in reason

    def test_crashing_sidecar_is_stuck(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reason = self._reason(monkeypatch, _pending_pod("CrashLoopBackOff"))
        assert reason is not None and "'gym' keeps crashing" in reason

    def test_starting_sidecar_is_not_stuck(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._reason(monkeypatch, _pending_pod(None, "PodInitializing")) is None


class TestOtherBackendsRejectSidecars:
    def test_local(self, tmp_path: Path) -> None:
        from amortized.backends.local import LocalBackend

        spec = JobSpec(job_id="j1", command=["true"], work_dir=str(tmp_path / "w"))
        spec.sidecars = [_GYM_SIDECAR]
        with pytest.raises(RuntimeError, match="only the kubernetes backend"):
            _run(LocalBackend().submit(spec))
        assert not (tmp_path / "w").exists()

    def test_ssh(self) -> None:
        from amortized.backends.ssh import SSHBackend

        spec = JobSpec(job_id="j1", command=["true"], sidecars=[_GYM_SIDECAR])
        backend = SSHBackend(host="h", user="u")
        with pytest.raises(RuntimeError, match="only the kubernetes backend"):
            _run(backend.submit(spec))


# --- Training builder ---------------------------------------------------------------------


_GYM_CONFIG: dict[str, Any] = {
    "algorithm": "lora_grpo",
    "model_name_or_path": "m",
    "data_path": "/amortized/work/data/generated_data",
    "reward_type": "gym",
    "gym_verifier": "mcqa",
}


class TestGymBuild:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("algorithm", ["lora_grpo", "grpo"])
    async def test_wires_custom_reward_and_sidecar(
        self, algorithm: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config_mod.settings, "job_image_tag", "sha-abc")
        monkeypatch.setattr(config_mod.settings, "gym_image_tag", "")
        files: dict[str, str] = {}
        result = await training_builder.build(
            {"id": "g1", "type": "training"}, {**_GYM_CONFIG, "algorithm": algorithm}, files
        )
        subcommand = algorithm.replace("_", "-")
        assert result.command == ["thub", subcommand, "--config", "/amortized/work/config.yaml"]
        parsed = yaml.safe_load(files["config.yaml"])
        assert parsed["reward_type"] == "custom"
        assert parsed["reward_fn"] == "gym_reward.compute_score"
        assert parsed["system_prompt"] == GYM_VERIFIERS["mcqa"].system_prompt
        assert parsed["data_path"] == "/amortized/work/grpo_data.jsonl"
        assert "gym_verifier" not in parsed  # amortized-side only
        assert files["gym_reward.py"] == (ASSETS / "gym_reward.py").read_text()
        assert files["check_gym_data.py"] == (ASSETS / "check_gym_data.py").read_text()
        # prepare -> verifier data check -> language_model_only patch -> thub
        assert result.pre_commands[-3].startswith("python3 /amortized/prepare_grpo_data.py gym ")
        assert result.pre_commands[-2] == (
            "python3 /amortized/check_gym_data.py /amortized/work/grpo_data.jsonl 20"
        )
        assert result.env["PYTHONPATH"] == "/amortized"
        assert result.env["GYM_VERIFIER_URL"] == "http://localhost:8000/verify"
        [sidecar] = result.sidecars
        assert sidecar.image == "ghcr.io/amortized-ai/gym:sha-abc"
        assert sidecar.port == 8000
        assert sidecar.env == {"GYM_SERVER": "mcqa"}

    @pytest.mark.asyncio
    async def test_gym_image_tag_override_and_user_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config_mod.settings, "gym_image_tag", "xingliu-dev")
        files: dict[str, str] = {}
        config = {**_GYM_CONFIG, "system_prompt": "Reply with \\boxed{LETTER}."}
        result = await training_builder.build({"id": "g1", "type": "training"}, config, files)
        assert result.sidecars[0].image == "ghcr.io/amortized-ai/gym:xingliu-dev"
        prompt = yaml.safe_load(files["config.yaml"])["system_prompt"]
        assert prompt == "Reply with \\boxed{LETTER}."

    @pytest.mark.asyncio
    async def test_hf_dataset_id_is_converted(self) -> None:
        files: dict[str, str] = {}
        config = {**_GYM_CONFIG, "data_path": "org/mcqa-set"}
        result = await training_builder.build({"id": "g1", "type": "training"}, config, files)
        assert any("prepare_grpo_data.py gym org/mcqa-set" in c for c in result.pre_commands)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reward", ["math", "tool_call"])
    async def test_other_rewards_have_no_sidecar(self, reward: str) -> None:
        files: dict[str, str] = {}
        config = {**_GYM_CONFIG, "reward_type": reward}
        config.pop("gym_verifier")
        result = await training_builder.build({"id": "g1", "type": "training"}, config, files)
        assert result.sidecars == []
        assert "gym_reward.py" not in files
        assert "check_gym_data.py" not in files
        assert "PYTHONPATH" not in result.env
        assert "reward_fn" not in yaml.safe_load(files["config.yaml"])


class TestGymSchema:
    @staticmethod
    def _req(**kw: Any) -> TrainingJobRequest:
        return TrainingJobRequest(algorithm="lora_grpo", model_name_or_path="m", **kw)

    def test_verifier_accepted(self) -> None:
        req = self._req(reward_type="gym", gym_verifier="mcqa")
        assert (req.reward_type, req.gym_verifier) == ("gym", "mcqa")

    def test_gym_needs_a_verifier(self) -> None:
        with pytest.raises(ValidationError, match="needs gym_verifier"):
            self._req(reward_type="gym")

    def test_unknown_verifier_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"unknown gym_verifier 'nope'.*mcqa"):
            self._req(reward_type="gym", gym_verifier="nope")

    def test_verifier_without_gym_reward_rejected(self) -> None:
        with pytest.raises(ValidationError, match="only applies with reward_type=gym"):
            self._req(reward_type="math", gym_verifier="mcqa")

    def test_registry_matches_the_image_verifier_list(self) -> None:
        # The agent may only pick verifiers the gym image installs: the manifest synced from
        # the image (scripts/sync_gym_verifiers.sh) must list exactly containers/gym/verifiers.txt.
        repo = Path(__file__).resolve().parents[1]
        lines = (repo / "containers/gym/verifiers.txt").read_text().splitlines()
        shipped = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
        assert shipped == set(GYM_VERIFIERS)
        assert len(shipped) > 1

    def test_manifest_entries_are_complete(self) -> None:
        for v in GYM_VERIFIERS.values():
            assert set(v.required) <= set(v.fields), v.name
            assert "capture_rollout_id" not in v.fields  # Gym base-request bookkeeping
        assert (
            GYM_VERIFIERS["mcqa"].system_prompt
            and "\\boxed{}" in GYM_VERIFIERS["mcqa"].system_prompt
        )


# --- Data prep ----------------------------------------------------------------------------


class TestPrepareGym:
    @staticmethod
    def _convert(tmp_path: Path, src_name: str, question_column: str = "") -> list[dict[str, Any]]:
        out = tmp_path / "out.jsonl"
        _asset("prepare_grpo_data.py").convert(
            "gym", str(tmp_path / src_name), str(out), question_column, ""
        )
        records = [json.loads(line) for line in out.read_text().splitlines()]
        for r in records:
            r["ground_truth"] = json.loads(r["ground_truth"])
        return records

    def test_other_columns_pass_through_as_verifier_fields(self, tmp_path: Path) -> None:
        row = {
            "question": "Which test?\nA: Karyotyping\nB: PCR",
            "options": [{"A": "Karyotyping"}, {"B": "PCR"}],
            "expected_answer": "B",
        }
        (tmp_path / "in.jsonl").write_text(json.dumps(row) + "\n")
        [rec] = self._convert(tmp_path, "in.jsonl")
        assert rec["question"] == row["question"]
        assert rec["ground_truth"] == {
            "prompt": row["question"],
            "fields": {"options": row["options"], "expected_answer": "B"},
        }

    def test_explicit_question_column(self, tmp_path: Path) -> None:
        (tmp_path / "in.jsonl").write_text(json.dumps({"stem": "Q", "x": 1}) + "\n")
        [rec] = self._convert(tmp_path, "in.jsonl", question_column="stem")
        assert rec["ground_truth"] == {"prompt": "Q", "fields": {"x": 1}}

    def test_parquet_struct_padding_is_dropped(self, tmp_path: Path) -> None:
        # Parquet pads a list of single-key dicts into structs with every key: A=None, B=...
        table = pa.Table.from_pylist(
            [{"question": "Q", "options": [{"A": "x"}, {"B": "y"}], "expected_answer": "B"}]
        )
        pq.write_table(table, tmp_path / "in.parquet")
        [rec] = self._convert(tmp_path, "in.parquet")
        assert rec["ground_truth"]["fields"]["options"] == [{"A": "x"}, {"B": "y"}]

    def test_missing_question_fails(self, tmp_path: Path) -> None:
        (tmp_path / "in.jsonl").write_text(json.dumps({"options": [], "x": 1}) + "\n")
        with pytest.raises(SystemExit, match="need a prompt column"):
            self._convert(tmp_path, "in.jsonl")


# --- Reward function ----------------------------------------------------------------------


@pytest.fixture
def verifier() -> Iterator[tuple[str, list[dict[str, Any]], Callable[[int, dict[str, Any]], None]]]:
    """A stand-in /verify endpoint: records request bodies, replies with a set status/body."""
    seen: list[dict[str, Any]] = []
    reply: dict[str, Any] = {"status": 200, "body": {"reward": 1.0}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            payload = json.dumps(reply["body"]).encode()
            self.send_response(reply["status"])
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def set_reply(status: int, body: dict[str, Any]) -> None:
        reply.update(status=status, body=body)

    yield f"http://127.0.0.1:{server.server_port}/verify", seen, set_reply
    server.shutdown()


def _copied_compute_score() -> Callable[..., float]:
    """compute_score as verl loads it: training-hub copies only the function's source."""
    namespace: dict[str, Any] = {}
    source = inspect.getsource(_asset("gym_reward.py").compute_score)
    exec(textwrap.dedent(source), namespace)
    fn: Callable[..., float] = namespace["compute_score"]
    return fn


_GT = json.dumps({"prompt": "Q?\nA: x\nB: y", "fields": {"expected_answer": "B", "options": []}})


class TestGymRewardFunction:
    def test_posts_rollout_and_returns_reward(
        self, verifier: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, seen, set_reply = verifier
        monkeypatch.setenv("GYM_VERIFIER_URL", url)
        set_reply(200, {"reward": 1.0, "extracted_answer": "B"})
        assert _copied_compute_score()("gym", "\\boxed{B}", _GT, {"index": 3}) == 1.0
        [body] = seen
        assert body["expected_answer"] == "B"
        assert body["responses_create_params"]["input"][0]["content"] == "Q?\nA: x\nB: y"
        assert body["response"]["output"][0]["content"][0]["text"] == "\\boxed{B}"

    def test_unreadable_ground_truth_scores_zero(self) -> None:
        assert _copied_compute_score()("gym", "\\boxed{B}", "not json") == 0.0

    def test_rejected_request_raises(self, verifier: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        url, seen, set_reply = verifier
        monkeypatch.setenv("GYM_VERIFIER_URL", url)
        set_reply(422, {"detail": "bad options"})
        with pytest.raises(RuntimeError, match="rejected the request \\(422\\)"):
            _copied_compute_score()("gym", "x", _GT)
        assert len(seen) == 1  # no retries on a 4xx

    def test_unreachable_verifier_raises_after_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        monkeypatch.setattr(time, "sleep", lambda _s: None)
        monkeypatch.setenv("GYM_VERIFIER_URL", "http://127.0.0.1:1/verify")
        with pytest.raises(RuntimeError, match="failed after 4 attempts"):
            _copied_compute_score()("gym", "x", _GT)


# --- API checks ---------------------------------------------------------------------------


class TestGymApiChecks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["local", "ssh"])
    async def test_non_kubernetes_backend_rejected(
        self, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from amortized.api import jobs as jobs_api

        monkeypatch.setattr(config_mod.settings, "default_backend", backend)
        errors = await jobs_api._validate_training_data(dict(_GYM_CONFIG), "", None)  # type: ignore[arg-type]
        assert any("needs the kubernetes compute backend" in e for e in errors)

    @pytest.mark.asyncio
    async def test_kubernetes_backend_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from amortized.api import jobs as jobs_api

        monkeypatch.setattr(config_mod.settings, "default_backend", "kubernetes")
        errors = await jobs_api._validate_training_data(dict(_GYM_CONFIG), "", None)  # type: ignore[arg-type]
        assert not any("kubernetes" in e for e in errors)

    @staticmethod
    async def _violations(
        monkeypatch: pytest.MonkeyPatch, row: dict[str, Any], verifier: str = "mcqa"
    ) -> list[str]:
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
        # The shape check only inspects MLflow datasets — a direct data_path is skipped.
        config = {k: v for k, v in _GYM_CONFIG.items() if k != "data_path"}
        config["data_run_id"] = "r" * 32
        config["gym_verifier"] = verifier
        violations: list[str] = await jobs_api._training_shape_violations(config, "", None)
        return violations

    @pytest.mark.asyncio
    async def test_shape_check_uses_the_verifiers_required_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert GYM_VERIFIERS["equivalence_rule"].required == ("expected_answer",)
        ok = {"question": "Q", "expected_answer": "42"}
        assert not await self._violations(monkeypatch, ok, "equivalence_rule")
        errors = await self._violations(monkeypatch, {"question": "Q"}, "equivalence_rule")
        assert errors and "needs expected_answer" in errors[0]
        # Optional-only fields (mcqa) are not enforced here — the in-pod check covers them.
        assert not await self._violations(monkeypatch, {"question": "Q", "answer": "B"})
        errors = await self._violations(monkeypatch, {"options": [], "expected_answer": "A"})
        assert errors and "a prompt column" in errors[0]


# --- Pre-training data check --------------------------------------------------------------


class TestCheckGymData:
    @staticmethod
    def _check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: int, limit: int) -> int:
        import sys

        monkeypatch.syspath_prepend(str(ASSETS))  # in the pod: PYTHONPATH=/amortized
        sys.modules.pop("gym_reward", None)
        data = tmp_path / "grpo_data.jsonl"
        line = json.dumps({"question": "Q", "ground_truth": _GT}) + "\n"
        data.write_text(line * rows)
        checked: int = _asset("check_gym_data.py").check(str(data), limit)
        return checked

    def test_accepted_rows(
        self, verifier: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, seen, _ = verifier
        monkeypatch.setenv("GYM_VERIFIER_URL", url)
        assert self._check(tmp_path, monkeypatch, rows=30, limit=20) == 20
        assert len(seen) == 20
        assert seen[0]["response"]["output"][0]["content"][0]["text"] == ""

    def test_rejected_row_stops_the_job(
        self, verifier: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, _, set_reply = verifier
        monkeypatch.setenv("GYM_VERIFIER_URL", url)
        set_reply(422, {"detail": "options: Input should be a valid list"})
        with pytest.raises(SystemExit, match=r"row 0 does not fit the gym verifier.*valid list"):
            self._check(tmp_path, monkeypatch, rows=3, limit=20)


# --- Starter recipe -----------------------------------------------------------------------


class TestGymRecipe:
    @staticmethod
    def _recipe() -> dict[str, Any]:
        from amortized.core.recipes import list_starter_templates

        use_case = "verifiable-tasks-grpo-multiple-choice"
        [recipe] = [t for t in list_starter_templates() if t["use_case"] == use_case]
        config: dict[str, Any] = recipe["config"]
        return config

    def test_recipe_passes_the_api_schema(self) -> None:
        config = self._recipe()
        req = TrainingJobRequest(**config)
        assert (req.reward_type, req.gym_verifier) == ("gym", "mcqa")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("algorithm", ["lora_grpo", "grpo"])
    async def test_recipe_builds_with_the_gym_sidecar(self, algorithm: str) -> None:
        config = {**self._recipe(), "data_path": "/amortized/work/data/x", "algorithm": algorithm}
        config.pop("parent_job_id")
        if algorithm == "grpo":  # the recipe's documented full fine-tuning switch
            config.pop("lora_r")
            config.pop("lora_alpha")
            config["learning_rate"] = 1e-06
            TrainingJobRequest(**config)
        files: dict[str, str] = {}
        result = await training_builder.build({"id": "r1", "type": "training"}, config, files)
        assert [s.env for s in result.sidecars] == [{"GYM_SERVER": "mcqa"}]
        assert yaml.safe_load(files["config.yaml"])["reward_fn"] == "gym_reward.compute_score"


# --- Reward-function map (bundled with the GRPO skill) -----------------------------------


class TestRewardFunctionMap:
    _PATH = (
        Path(__file__).resolve().parent.parent
        / "agents/training/skills/verifiable-tasks/grpo/reward-functions.json"
    )
    _KEYS = frozenset(
        {"reward_type", "summary", "reward", "answer_format", "row_fields", "good_for"}
        | {"caveats", "general_purpose"}
    )

    def _map(self) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = json.loads(self._PATH.read_text())["reward_functions"]
        return entries

    def test_covers_every_reward_and_shipped_verifier(self) -> None:
        entries = self._map()
        gym = {n for n, e in entries.items() if e["reward_type"] == "gym"}
        builtin = {n for n, e in entries.items() if e["reward_type"] != "gym"}
        assert gym == set(GYM_VERIFIERS)
        assert builtin == {"math", "tool_call"}
        assert all(e["reward_type"] == n for n, e in entries.items() if n in builtin)

    def test_entries_are_complete(self) -> None:
        for name, entry in self._map().items():
            assert entry.keys() >= self._KEYS, name
            assert isinstance(entry["general_purpose"], bool), name
            if entry["reward_type"] == "gym":
                assert entry["doc"].endswith(f"/resources_servers/{name}"), name

    def test_every_map_entry_passes_the_api_schema(self) -> None:
        for name, entry in self._map().items():
            extra = {"gym_verifier": name} if entry["reward_type"] == "gym" else {}
            TrainingJobRequest(
                algorithm="lora_grpo",
                model_name_or_path="Qwen/Qwen3-0.6B",
                reward_type=entry["reward_type"],
                data_path="/amortized/work/data/x",
                **extra,
            )
