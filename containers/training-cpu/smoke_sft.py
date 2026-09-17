#!/usr/bin/env python3
"""Build-time EXECUTION smoke for the CPU SFT runner.

``py_compile`` only checks syntax — it cannot catch runtime defects such as
the missing-LoRA-``target_modules`` failure (issue #442): PEFT raises
``ValueError: Please specify target_modules`` at adapter injection for model
types missing from its auto-inference mapping, which is exactly the situation
for the catalog's Qwen3.5 models (``model_type='qwen3_5'``). This smoke
therefore runs the ACTUAL dispatched command end-to-end on a tiny
random-weights model with that same unmapped architecture:

- 1-layer, 32-hidden Qwen3.5 config built in-memory (no network, no download)
- a 4-sample chat-format JSONL dataset and a minimal thub-format config.yaml
- ``python3 /usr/local/bin/cpu_sft.py --config <cfg>`` as a subprocess
- must complete training, save ``final/adapter_config.json``, and record
  non-empty ``target_modules`` in it

Runs in seconds on 1 CPU; it fails the container build on any regression.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5TextConfig,
)

RUNNER = "/usr/local/bin/cpu_sft.py"
WORK = Path(tempfile.mkdtemp(prefix="cpu-sft-smoke-"))
MODEL_DIR = WORK / "model"
DATA = WORK / "data.jsonl"
CONFIG = WORK / "config.yaml"
OUT = WORK / "out"

# --- tiny random-weights Qwen3.5-architecture model (model_type 'qwen3_5') ---
# Qwen3_5Config is a composite (text+vision): the size kwargs must go to the
# nested text config or the huge defaults (~3B params) blow up the build.
text_config = Qwen3_5TextConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    layer_types=["full_attention"],
    head_dim=8,
    max_position_embeddings=128,
)
model = AutoModelForCausalLM.from_config(Qwen3_5Config(text_config=text_config))
model.save_pretrained(MODEL_DIR)

# --- offline tiny tokenizer with a minimal chat template ---
vocab = {tok: i for i, tok in enumerate(
    ["<pad>", "<unk>", "a", "b", "c", "d", "e", "f", "g", "h"]
)}
backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<pad>",
    eos_token="<unk>",
    chat_template=(
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        "{{ eos_token }}"
    ),
)
tokenizer.save_pretrained(MODEL_DIR)

# --- 4-sample chat-format dataset (same shape the builder emits) ---
with DATA.open("w") as f:
    for prompt, completion in (
        ("a b c d", "h g f e"),
        ("e f g h", "a c e g"),
        ("a c e g", "b d f h"),
        ("b d f h", "e d c b"),
    ):
        f.write(json.dumps({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        }) + "\n")

# --- minimal thub-format config (mirrors the builder's CPU defaults) ---
CONFIG.write_text(
    f"model_path: {MODEL_DIR}\n"
    f"data_path: {DATA}\n"
    f"ckpt_output_dir: {OUT}\n"
    "lora_r: 2\n"
    "lora_alpha: 4\n"
    "learning_rate: 2e-4\n"
    "num_epochs: 1\n"
    "effective_batch_size: 2\n"
    "max_seq_len: 32\n"
)

# --- run the exact dispatched command ---
proc = subprocess.run(
    [sys.executable, RUNNER, "--config", str(CONFIG)],
    capture_output=True,
    text=True,
)
sys.stdout.write(proc.stdout)
sys.stderr.write(proc.stderr)
if proc.returncode != 0:
    sys.exit(f"smoke FAILED: {RUNNER} exited {proc.returncode}")

adapter_config = OUT / "final" / "adapter_config.json"
if not adapter_config.is_file():
    sys.exit(f"smoke FAILED: {adapter_config} was not saved")
target_modules = json.loads(adapter_config.read_text()).get("target_modules")
if not target_modules:
    sys.exit("smoke FAILED: adapter_config.json has no target_modules")

print(f"cpu_sft.py execution smoke OK (LoRA on {len(target_modules)} target modules)")
