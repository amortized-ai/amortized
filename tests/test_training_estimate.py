"""VRAM estimation (api/costs.py) and model architecture resolution (core/model_arch.py)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from amortized.api import costs
from amortized.api.costs import TrainingEstimateRequest, estimate_training_resources
from amortized.core import model_arch
from amortized.core.model_arch import KNOWN_ARCHS, arch_from_hf_config, resolve_arch

GIB = 1024**3


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test reaches the real hub; tests that need it patch fetch_hf_arch."""

    async def _blocked(model_id: str) -> None:
        raise AssertionError(f"unexpected HuggingFace fetch for {model_id}")

    monkeypatch.setattr(model_arch, "fetch_hf_arch", _blocked)
    model_arch._hf_cache.clear()


async def _estimate(**kwargs: Any) -> costs.TrainingEstimateResponse:
    return await estimate_training_resources(TrainingEstimateRequest(**kwargs))


class TestArchFromHfConfig:
    def test_dense(self) -> None:
        config = {
            "num_hidden_layers": 36,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "vocab_size": 151936,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "tie_word_embeddings": False,
        }
        # training-hub's AutoModel count for Qwen3-8B (no lm_head): 7,568,405,504.
        approx = arch_from_hf_config(config, num_params=None)
        assert abs(approx.num_params - 7_568_405_504) / 7_568_405_504 < 0.001
        assert approx.source == "huggingface"
        # The hub's safetensors total includes the stored lm_head, which is dropped.
        exact = arch_from_hf_config(config, num_params=8_190_735_360)
        assert exact.num_params == 7_568_405_504

    def test_tied_model_that_still_stores_lm_head(self) -> None:
        # Qwen3-0.6B: tie_word_embeddings=True, but its safetensors still hold lm_head.
        config = {
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "tie_word_embeddings": True,
        }
        assert arch_from_hf_config(config, num_params=751_632_384).num_params == 596_049_920

    def test_nested_text_config_and_moe(self) -> None:
        config = {
            "model_type": "gemma4",
            "tie_word_embeddings": True,
            "text_config": {
                "num_hidden_layers": 30,
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "vocab_size": 262144,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "num_experts": 128,
                "moe_intermediate_size": 704,
            },
        }
        arch = arch_from_hf_config(config, num_params=25_805_936_206)
        assert arch.num_hidden_layers == 30
        assert arch.vocab_size == 262144
        assert arch.num_experts == 128
        assert arch.num_params == 25_805_936_206
        assert arch.label == "25.8B"

    def test_shared_experts(self) -> None:
        config = {
            "num_hidden_layers": 52,
            "hidden_size": 2688,
            "vocab_size": 131072,
            "num_attention_heads": 32,
            "n_routed_experts": 128,
            "n_shared_experts": 1,
            "moe_intermediate_size": 1856,
        }
        arch = arch_from_hf_config(config, num_params=1)
        assert arch.num_experts == 128
        assert arch.shared_expert_intermediate_size == 1856
        assert arch.num_kv_heads == 32  # defaults to num_attention_heads

    def test_missing_fields(self) -> None:
        with pytest.raises(ValueError):
            arch_from_hf_config({"hidden_size": 1}, num_params=None)


class TestResolveArch:
    async def test_catalog_is_case_insensitive_and_offline(self) -> None:
        arch = await resolve_arch("Qwen/Qwen3.5-4B")
        assert arch is KNOWN_ARCHS["qwen/qwen3.5-4b"]
        assert arch.source == "catalog"

    async def test_hf_fetch_for_unknown_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fetched = arch_from_hf_config(
            {"num_hidden_layers": 2, "hidden_size": 8, "vocab_size": 10}, num_params=123
        )

        async def _fetch(model_id: str) -> model_arch.ModelArch:
            assert model_id == "org/custom-model"
            return fetched

        monkeypatch.setattr(model_arch, "fetch_hf_arch", _fetch)
        assert await resolve_arch("org/custom-model") is fetched

    async def test_unresolvable_raises_instead_of_guessing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _offline(model_id: str) -> None:
            return None

        monkeypatch.setattr(model_arch, "fetch_hf_arch", _offline)
        with pytest.raises(ValueError, match="HF_TOKEN"):
            await resolve_arch("meta-llama/Llama-3.1-8B-Instruct")


class TestFetchHfArch:
    async def test_parses_config_and_param_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/config.json"):
                return httpx.Response(
                    200, json={"num_hidden_layers": 2, "hidden_size": 8, "vocab_size": 10}
                )
            return httpx.Response(200, json={"safetensors": {"total": 4242}})

        real_client = httpx.AsyncClient
        monkeypatch.undo()  # lift the autouse block so the real fetch_hf_arch runs
        monkeypatch.setattr(
            model_arch.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        )
        model_arch._hf_cache.clear()
        arch = await model_arch.fetch_hf_arch("org/model")
        assert arch is not None
        assert arch.num_params == 4242 - 10 * 8  # stored lm_head (vocab x hidden) dropped

    async def test_rejects_non_hub_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()
        assert await model_arch.fetch_hf_arch("/local/path/model") is None
        assert await model_arch.fetch_hf_arch("../../etc/passwd") is None

    async def test_unreachable_hub_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401)

        real_client = httpx.AsyncClient
        monkeypatch.undo()
        monkeypatch.setattr(
            model_arch.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        )
        model_arch._hf_cache.clear()
        assert await model_arch.fetch_hf_arch("org/gated") is None


# Bytes returned by the latest training-hub memory_estimator.estimate() for Qwen/Qwen3-8B with
# batch_size=8, max_seq_len=2048, max_tokens_per_gpu=4096 (sft/osft), unfreeze_rank_ratio=0.2,
# lora_r=32 — produced by running training-hub's own code on the model.
TRAINING_HUB_QWEN3_8B = {
    ("sft", 1): (130148592298, 143163451528, 169193169988),
    ("sft", 4): (39327726250, 43260498875, 51126044125),
    ("osft", 1): (109743603712, 120717964083, 142666684825),
    ("osft", 4): (27798288793, 30578117672, 36137775431),
    ("lora", 1): (29093064601, 33686706380, 36749134233),
    ("lora", 4): (7273266150, 8421676595, 9187283558),
    ("qlora", 1): (18308086758, 21198837299, 25053171353),
    ("qlora", 4): (1797496307, 2081311513, 2459731788),
}


# training-hub's OSFTEstimator defaults to use_liger=True (no logits term), but amortized
# jobs run without Liger, so osft must equal what training-hub reports with use_liger=False:
# the golden value plus tokens x fp32 x vocab x 7/3 (4096 tokens/GPU here), scaled by the
# same low/mid/high overheads.
def _training_hub_expected(method: str, num_gpus: int) -> tuple[float, ...]:
    golden = TRAINING_HUB_QWEN3_8B[(method, num_gpus)]
    if method != "osft":
        return golden
    out = 4096 * 4 * 151936 * 7 / 3
    return tuple(g + out * k for g, k in zip(golden, (1, 1.1, 1.3), strict=True))


class TestMatchesTrainingHub:
    @pytest.mark.parametrize(("method", "num_gpus"), list(TRAINING_HUB_QWEN3_8B))
    def test_catalog_model(self, method: str, num_gpus: int) -> None:
        body = TrainingEstimateRequest(
            model_name_or_path="Qwen/Qwen3-8B",
            method=method,
            effective_batch_size=8,
            max_seq_len=2048,
            max_tokens_per_gpu=4096,
            unfreeze_rank_ratio=0.2,
            lora_r=32,
        )
        info = KNOWN_ARCHS["qwen/qwen3-8b"]
        assert costs._estimate(method, info, body, num_gpus) == pytest.approx(
            _training_hub_expected(method, num_gpus), rel=1e-9
        )

    @pytest.mark.parametrize(("method", "num_gpus"), list(TRAINING_HUB_QWEN3_8B))
    def test_config_derived_model(self, method: str, num_gpus: int) -> None:
        # Same model, but every statistic derived from config.json + the hub's param count.
        config = {
            "num_hidden_layers": 36,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "vocab_size": 151936,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "tie_word_embeddings": False,
        }
        info = arch_from_hf_config(config, num_params=8_190_735_360)
        body = TrainingEstimateRequest(
            model_name_or_path="org/derived",
            method=method,
            max_tokens_per_gpu=4096,
            unfreeze_rank_ratio=0.2,
        )
        assert costs._estimate(method, info, body, num_gpus) == pytest.approx(
            _training_hub_expected(method, num_gpus), rel=1e-9
        )


class TestHelpers:
    def test_derived_stats_match_catalog(self) -> None:
        # Qwen3-0.6B: 16 heads x 128 head_dim = 2048 != hidden 1024.
        catalog = KNOWN_ARCHS["qwen/qwen3-0.6b"]
        derived = replace(
            catalog,
            weight_size_total=None,
            osft_params=None,
            intermediate_size=3072,
            num_attention_heads=16,
            num_kv_heads=8,
            head_dim=128,
        )
        assert costs._calc_weight_size_total(derived) == catalog.weight_size_total
        assert costs._calc_osft_params(derived) == catalog.osft_params

    def test_moe_experts_are_not_targets(self) -> None:
        moe = arch_from_hf_config(
            {
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "vocab_size": 10,
                "num_attention_heads": 2,
                "head_dim": 4,
                "num_experts": 64,
                "moe_intermediate_size": 16,
            },
            num_params=1000,
        )
        assert costs._calc_weight_size_total(moe) == 2 * 4 * (8 + 8)  # q/k/v/o only

    def test_resolve_tokens_per_gpu(self) -> None:
        assert costs._resolve_tokens_per_gpu("sft", 1, 8, 2048, 4096) == 4096
        assert costs._resolve_tokens_per_gpu("sft", 8, 8, 2048, 4096) == 2048
        assert costs._resolve_tokens_per_gpu("osft", 1, 2, 512, 4096) == 2 * 512
        # LoRA ignores max_tokens_per_gpu and splits the batch across GPUs.
        assert costs._resolve_tokens_per_gpu("lora", 4, 8, 2048, 100) == 2 * 2048

    def test_defaults_match_training_py(self) -> None:
        req = TrainingEstimateRequest(model_name_or_path="Qwen/Qwen3-8B")
        assert req.effective_batch_size == 8  # micro_batch_size 2 x 4

    def test_no_gpu_count_input(self) -> None:
        assert "num_gpus" not in TrainingEstimateRequest.model_fields


class TestPerGpu:
    """per_gpu_vram_gb_by_gpu_count: FSDP2 for sft/osft, model splitting for lora/qlora."""

    @staticmethod
    def _body(method: str, model: str = "Qwen/Qwen3-8B") -> TrainingEstimateRequest:
        return TrainingEstimateRequest(
            model_name_or_path=model, method=method, max_tokens_per_gpu=4096
        )

    @pytest.mark.parametrize("method", ["sft", "osft", "lora", "qlora"])
    def test_one_gpu_is_training_hub(self, method: str) -> None:
        info = KNOWN_ARCHS["qwen/qwen3-8b"]
        body = self._body(method)
        assert (
            costs._estimate_per_gpu(method, info, body, 1)
            == costs._estimate(method, info, body, 1)[1]
        )

    @pytest.mark.parametrize("method", ["sft", "osft", "lora", "qlora"])
    def test_monotonic_and_less_optimistic(self, method: str) -> None:
        info = KNOWN_ARCHS["qwen/qwen3-8b"]
        body = self._body(method)
        per_gpu = [costs._estimate_per_gpu(method, info, body, n) for n in (1, 2, 4, 8)]
        assert per_gpu == sorted(per_gpu, reverse=True)
        for n, value in zip((2, 4, 8), per_gpu[1:], strict=True):
            assert value > costs._estimate(method, info, body, n)[1]  # never below even split

    def test_fsdp_adds_gathered_unit(self) -> None:
        # Qwen3-8B (untied): root unit = embeddings + lm_head = 2 x 151936 x 4096 params.
        info = KNOWN_ARCHS["qwen/qwen3-8b"]
        assert costs._calc_largest_fsdp_unit(info) == 2 * 151936 * 4096
        body = self._body("sft")
        extra = (
            costs._estimate_per_gpu("sft", info, body, 4) - costs._estimate("sft", info, body, 4)[1]
        )
        assert extra == int(4 * 2 * 151936 * 4096 * 1.1)

    def test_tied_root_unit_counts_embeddings_once(self) -> None:
        info = KNOWN_ARCHS["qwen/qwen3-0.6b"]
        assert info.tie_word_embeddings
        block = (info.num_params - 151936 * 1024) / 28
        assert costs._calc_largest_fsdp_unit(info) == max(151936 * 1024, block)

    def test_model_split_keeps_full_logits_on_one_gpu(self) -> None:
        info = KNOWN_ARCHS["qwen/qwen3-8b"]
        body = self._body("lora")
        logits = costs._calc_outputs(info, 8 * 2048, costs.FP16, 2)
        assert costs._estimate_per_gpu("lora", info, body, 8) > logits * 1.1

    def test_qlora_floor_never_lowers_estimate(self) -> None:
        info = KNOWN_ARCHS["google/gemma-4-31b-it"]
        body = self._body("qlora", "google/gemma-4-31B-it")
        per_gpu = [costs._estimate_per_gpu("qlora", info, body, n) for n in (1, 2, 4, 8)]
        assert per_gpu == sorted(per_gpu, reverse=True)
        assert per_gpu[-1] >= info.num_params * costs.FP8 / 8


class TestEndpoint:
    async def test_response_shape(self) -> None:
        resp = await _estimate(
            model_name_or_path="Qwen/Qwen3.5-4B", method="osft", max_tokens_per_gpu=15000
        )
        assert resp.model == "Qwen/Qwen3.5-4B"
        assert resp.arch_source == "catalog"
        assert resp.model_size == "4.5B"  # training-hub's count: 4,539,265,536
        assert resp.vram_gb["low"] < resp.vram_gb["expected"] < resp.vram_gb["high"]
        assert set(resp.per_gpu_vram_gb_by_gpu_count) == {"1", "2", "4", "8"}
        assert resp.per_gpu_vram_gb_by_gpu_count["1"] == resp.vram_gb["expected"]

    async def test_reports_gpu_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(costs.settings, "gpu_budget", 3)
        resp = await _estimate(model_name_or_path="Qwen/Qwen3.5-4B", method="lora")
        assert resp.gpu_budget == 3

    async def test_sft_activations_do_not_shard(self) -> None:
        resp = await _estimate(
            model_name_or_path="Qwen/Qwen3.5-9B", method="sft", max_tokens_per_gpu=15000
        )
        by_count = resp.per_gpu_vram_gb_by_gpu_count
        assert by_count["1"] > by_count["2"] > by_count["4"] > by_count["8"]
        assert by_count["2"] > by_count["1"] / 2

    async def test_method_aliases(self) -> None:
        resp = await _estimate(model_name_or_path="Qwen/Qwen3-8B", method="lora_sft")
        assert resp.method == "lora_sft"

    async def test_unknown_method(self) -> None:
        with pytest.raises(HTTPException) as exc:
            await _estimate(model_name_or_path="Qwen/Qwen3-8B", method="dpo")
        assert exc.value.status_code == 400

    async def test_unresolvable_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _offline(model_id: str) -> None:
            return None

        monkeypatch.setattr(model_arch, "fetch_hf_arch", _offline)
        with pytest.raises(HTTPException) as exc:
            await _estimate(model_name_or_path="org/unknown-model")
        assert exc.value.status_code == 400

    def test_requires_a_model(self) -> None:
        with pytest.raises(ValidationError):
            TrainingEstimateRequest(method="lora")  # type: ignore[call-arg]


class TestRegressionsFromRealJobs:
    """Configs whose real peak memory we observed on 80 GB GPUs."""

    def test_osft_counts_logits_qwen35_9b(self) -> None:
        # Job 9f069b2a OOMed on 4x80 GB (a 13 GiB fp32-logits allocation on top of 66 GiB).
        info = KNOWN_ARCHS["qwen/qwen3.5-9b"]
        body = TrainingEstimateRequest(
            model_name_or_path="Qwen/Qwen3.5-9B",
            method="osft",
            effective_batch_size=32,
            max_seq_len=11000,
            max_tokens_per_gpu=15000,
        )
        assert costs._estimate_per_gpu("osft", info, body, num_gpus=4) / 2**30 > 0.9 * 79

    def test_lora_uses_micro_batch_qwen35_9b(self) -> None:
        # Job dceac892 succeeded at ~40 GB/GPU on 4 GPUs (micro-batch 32 // 8 = 4).
        info = KNOWN_ARCHS["qwen/qwen3.5-9b"]
        body = TrainingEstimateRequest(
            model_name_or_path="Qwen/Qwen3.5-9B",
            method="lora",
            effective_batch_size=32,
            gradient_accumulation_steps=8,
            max_seq_len=8192,
            lora_r=16,
        )
        assert 35 < costs._estimate_per_gpu("lora", info, body, num_gpus=4) / 2**30 < 50
