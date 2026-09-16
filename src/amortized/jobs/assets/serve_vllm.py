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


def main() -> None:
    import runpy
    import sys

    # Delegate to vLLM's CLI; argv[0] is replaced with the expected "vllm"
    sys.argv = ["vllm", *sys.argv[1:]]
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")


if __name__ == "__main__":
    main()
