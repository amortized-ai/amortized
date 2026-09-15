"""Generic vLLM serve wrapper for Training Hub text-backbone exports.

Training Hub (mini_trainer) extracts the CausalLM text backbone from VLM
models for training; its hf_format exports carry the bare text config
(model_type like "qwen3_5_text", no "architectures"). vLLM ships native
implementations for many of these backbones (e.g. Qwen3_5ForCausalLM) but
does not register them, because HF only publishes the multimodal wrappers.

This wrapper registers every unregistered *ForCausalLM class found in
vllm.model_executor.models, then delegates to the vLLM CLI. Registration
runs at module level so multiprocessing-spawned engine workers (which
re-import __main__) get it too.

Usage: python3 serve_vllm.py <vllm serve args...>
"""

from __future__ import annotations


def _mrope_input_positions_text_only(
    self,
    input_tokens: list[int],
    mm_features: list,
) -> tuple:
    """Text-only M-RoPE positions: plain 1D positions replicated 3x.

    Every VLM family computes exactly this for its text tokens (see e.g.
    Qwen3VLForConditionalGeneration._get_mrope_input_positions); with all
    three sections at the same position, M-RoPE reduces to standard RoPE.
    """
    import numpy as np
    import torch

    num_tokens = len(input_tokens)
    positions = np.tile(
        np.arange(num_tokens, dtype=np.int64), (3, 1)
    )
    return torch.from_numpy(positions), 0


def _add_text_mrope_interface(cls: type) -> None:
    """Satisfy vLLM's SupportsMRoPE protocol on a text-backbone class.

    Text-backbone exports keep their family's mrope rope_parameters, so
    vLLM computes 3D positions for them and asserts the model implements
    the SupportsMRoPE interface. The classes (e.g. Qwen3_5ForCausalLM) use
    MRotaryEmbedding internally and handle 3D positions, but some vLLM
    versions don't declare the interface. The protocol is runtime-checkable
    (isinstance = attribute presence), so attaching the text-only positions
    method is sufficient and harmless for models that don't use mrope (the
    interface is only exercised when the config has mrope_section).
    """
    if not hasattr(cls, "get_mrope_input_positions"):
        cls.get_mrope_input_positions = _mrope_input_positions_text_only
    cls.supports_mrope = True


def _borrow_hybrid_interface(cls: type, module, registered: dict) -> None:
    """Make a hybrid (attention + mamba/GDN) text backbone satisfy IsHybrid.

    Families like Qwen3.5 mix full_attention and linear_attention layers;
    vLLM only sets up the mamba cache (mamba_block_size, cache mode) when
    the resolved model class implements the IsHybrid interface. The text
    backbone classes don't declare it, but a registered sibling in the same
    module (e.g. Qwen3_5ForConditionalGeneration) does — and since both
    share the same mamba layers and hf_config keys, its state-shape
    classmethods apply unchanged. Copy them over.
    """
    from vllm.model_executor.models.interfaces import IsHybrid

    if isinstance(cls, IsHybrid):
        return
    members = {}
    for other_name in dir(module):
        other = getattr(module, other_name, None)
        if (
            not isinstance(other, type)
            or other is cls
            or other.__module__ != cls.__module__
            or not isinstance(other, IsHybrid)
        ):
            continue
        for attr in (
            "get_mamba_state_shape_from_config",
            "get_mamba_state_dtype_from_config",
            "get_mamba_state_copy_func",
        ):
            fn = getattr(other, attr, None)
            if fn is not None:
                members[attr] = fn
        break
    if len(members) < 2:  # shape + copy funcs are both required
        return
    cls.is_hybrid = True
    for attr, fn in members.items():
        setattr(cls, attr, fn)


def _register_text_backbone_archs() -> None:
    import logging
    import pkgutil

    import vllm.model_executor.models as vllm_models
    from vllm.model_executor.models.registry import ModelRegistry

    logger = logging.getLogger("serve_vllm")
    registered = ModelRegistry.models
    package_name = vllm_models.__name__
    count = 0
    for mod_info in pkgutil.iter_modules(vllm_models.__path__):
        if mod_info.name.startswith("_"):
            continue
        try:
            module = __import__(f"{package_name}.{mod_info.name}", fromlist=["*"])
        except Exception:
            continue
        module_name = f"{package_name}.{mod_info.name}"
        for cls_name in dir(module):
            if not cls_name.endswith("ForCausalLM") or cls_name in registered:
                continue
            cls = getattr(module, cls_name, None)
            if not isinstance(cls, type):
                continue
            # Only register classes *defined* in this module; modules
            # re-export other modules' classes, and registering through a
            # re-exporter would borrow the hybrid interface from the wrong
            # sibling set (and race the class's home-module processing).
            if cls.__module__ != module_name:
                continue
            try:
                _add_text_mrope_interface(cls)
                _borrow_hybrid_interface(cls, module, registered)
                ModelRegistry.register_model(cls_name, cls)
                count += 1
            except Exception:
                logger.debug("Could not register %s", cls_name, exc_info=True)
    if count:
        logger.info("Registered %d text-backbone architectures in vLLM", count)


_register_text_backbone_archs()


def _ignore_nonpersistent_buffers() -> None:
    """Make vLLM's AutoWeightsLoader ignore non-persistent buffer weights.

    HF training exports persist buffers like ``rotary_emb.inv_freq`` that
    vLLM recomputes instead of loading — their presence in the checkpoint
    otherwise aborts weight loading with "There is no module or parameter
    named 'rotary_emb'". Ignoring them is safe for any model.
    """
    from vllm.model_executor.models.utils import AutoWeightsLoader

    extra_prefixes = ("rotary_emb.", "position_embeddings.")

    orig_init = AutoWeightsLoader.__init__

    def patched_init(self, *args, **kwargs):
        ignore = list(kwargs.pop("ignore_unexpected_prefixes", None) or [])
        ignore.extend(p for p in extra_prefixes if p not in ignore)
        kwargs["ignore_unexpected_prefixes"] = ignore
        orig_init(self, *args, **kwargs)

    AutoWeightsLoader.__init__ = patched_init


_ignore_nonpersistent_buffers()


# Engine-config sections describing HOW the engine is hosted (weight
# loading, compilation, profiling, metrics) — auto-derived from hardware
# and vLLM version, vary per run, and never distinguish eval configs.
_SECTION_DENY = frozenset(
    {
        "load_config",
        "offload_config",
        "compilation_config",
        "profiler_config",
        "observability_config",
    }
)

# Fields dropped from every kept section: model identity/paths/secrets
# (the model id is already the comparison key), per-run KV capacity
# (auto-sized from free GPU memory), and host/worker runtime
# coordinates. Anything NOT on these lists fingerprints the config, so
# an arg we did not think of distinguishes runs (visible false split)
# instead of silently merging them.
_FIELD_DENY = frozenset(
    {
        # model identity, paths, secrets
        "model",
        "tokenizer",
        "model_weights",
        "hf_config",
        "hf_text_config",
        "hf_config_path",
        "hf_token",
        "instance_id",
        # per-run KV capacity
        "gpu_memory_utilization",
        "num_gpu_blocks",
        "num_cpu_blocks",
        "num_gpu_blocks_override",
        "kv_cache_size_tokens",
        "kv_cache_memory_bytes",
        "kv_cache_max_concurrency",
        "kv_offloading_size",
        # host runtime coordinates
        "data_parallel_master_ip",
        "data_parallel_rpc_port",
        "data_parallel_master_port",
        "data_parallel_rank",
        "data_parallel_rank_local",
        "data_parallel_index",
        "master_addr",
        "master_port",
        "node_rank",
        "rank",
        "world_size",
        "_data_parallel_master_port_list",
        "_coord_store_port",
        "_api_process_count",
        "_api_process_rank",
        "assigned_physical_gpu_ids",
    }
)


def _dump_resolved_args() -> None:
    """Write the RESOLVED engine config to a sidecar JSON.

    The eval fingerprint uses this instead of the raw CLI args so that an
    arg explicitly set to its default ("--max-model-len <native length>")
    and the same arg left unset produce the SAME resolved value and merge
    in the comparison table. The full engine config is serialized minus a
    denylist of hosting/capacity/identity fields (see above) so any
    unknown user-set arg participates in the fingerprint. Best-effort:
    on any failure the sidecar is not written and the fingerprint falls
    back to the raw args.
    """
    import argparse
    import dataclasses
    import json
    import os
    import sys

    out_path = os.environ.get(
        "VLLM_RESOLVED_ARGS_JSON", "/amortized/work/resolved_vllm_args.json"
    )
    try:
        from vllm.engine.arg_utils import EngineArgs

        # argv: [serve_vllm.py, serve, <model>, ...engine+server flags]
        args_after_sub = sys.argv[2:] if len(sys.argv) > 2 else []
        if args_after_sub and not args_after_sub[0].startswith("-"):
            # "serve <model>" passes the model positionally; EngineArgs'
            # parser only knows --model, and its default would resolve
            # the WRONG model's config here.
            args_after_sub = ["--model", args_after_sub[0], *args_after_sub[1:]]
        p = argparse.ArgumentParser()
        EngineArgs.add_cli_args(p)
        ns, _unknown = p.parse_known_args(args_after_sub)  # skip server flags
        cfg = EngineArgs.from_cli_args(ns).create_engine_config()
        resolved = {}
        for f in dataclasses.fields(cfg):
            if f.name in _SECTION_DENY:
                continue
            v = getattr(cfg, f.name)
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                section = dataclasses.asdict(v)
                resolved[f.name] = {
                    k: v2 for k, v2 in section.items() if k not in _FIELD_DENY
                }
            elif f.name not in _FIELD_DENY:
                resolved[f.name] = str(v) if v is not None else None
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(resolved, f, sort_keys=True, default=str)
    except SystemExit:
        # argparse rejects e.g. --help; the CLI delegation handles it
        pass
    except Exception:
        import traceback

        print(
            "serve_vllm: resolved-args dump failed (fingerprint falls back"
            " to raw args):\n" + traceback.format_exc(),
            file=sys.stderr,
        )


def main() -> None:
    import runpy
    import sys

    _dump_resolved_args()

    # Delegate to vLLM's CLI; argv[0] is replaced with the expected "vllm"
    sys.argv = ["vllm", *sys.argv[1:]]
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")


if __name__ == "__main__":
    main()
