# Training Recipes Upgrade Plan

Upgrade the training recipes in `recipes/training/` and `recipes/models/` to cover the full set of TRL-supported algorithms, with proper parameterization and model-specific presets.

Amortized uses TRL as its training backend. All training configs are submitted as amortized job configs with `type: training`. The `config` block maps to TRL trainer arguments.

Reference: Oumi training configs at `/Users/shiv/workspace/oumi/configs/recipes/` for field names and defaults.

---

## Current State

### `recipes/training/` (4 base templates)

| File | Algorithm | Fields |
|---|---|---|
| `full-sft.yaml` | SFT (all weights) | algorithm, num_train_epochs, learning_rate, max_length, bf16, gradient_checkpointing, report_to |
| `lora-sft.yaml` | LoRA SFT | Same + use_peft, lora_r, lora_alpha |
| `grpo.yaml` | GRPO | algorithm, learning_rate, num_generations, max_completion_length, temperature, beta, epsilon, use_peft, lora_r, lora_alpha |
| `osft.yaml` | Full SFT (large) | Same as full-sft + gradient_accumulation_steps, lower lr |

### `recipes/models/` (4 model presets)

| File | Model | Extends |
|---|---|---|
| `llama3-8b-lora.yaml` | Llama 3.1 8B Instruct | training/lora-sft |
| `qwen-1.5b-lora.yaml` | Qwen 2.5 1.5B Instruct | training/lora-sft |
| `qwen-7b-sft.yaml` | Qwen 2.5 7B Instruct | training/full-sft |
| `qwen3-4b-grpo.yaml` | Qwen3 4B | training/grpo |

---

## Problems

1. **Missing algorithms** — No DPO, KTO, QLoRA, GKD, or GOLD templates. TRL supports all of these.
2. **Too few fields** — The templates are bare-minimum. Missing critical training params: optimizer, lr_scheduler, warmup, weight_decay, max_steps, save_steps, logging_steps, eval_strategy, gradient_accumulation, max_grad_norm.
3. **No data section** — None of the configs specify dataset format or how to load training data. Users need to know where to put their data_path.
4. **No compute hints** — No indication of GPU requirements (VRAM, count) for each template.
5. **`osft.yaml` is confusing** — Name suggests a different algorithm but it's just full-sft with different hyperparams. Either rename or remove.
6. **Model presets are stale** — Only 4 models, no Qwen3 family (except GRPO), no small models for testing.
7. **No `description` field on base templates** — The SDG recipes have descriptions, training recipes should too.

---

## Upgrade Plan

### Phase 1: Upgrade Base Training Templates

#### 1a. `recipes/training/full-sft.yaml` — UPGRADE

Full parameter fine-tuning via TRL SFTTrainer. All model weights are updated. Requires the most VRAM.

```yaml
type: training
description: Full SFT — all weights updated via TRL SFTTrainer
config:
  algorithm: sft
  
  # Data
  data_path: training_data.jsonl
  max_length: 4096
  
  # Training
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-05
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Precision
  bf16: true
  
  # Memory
  gradient_checkpointing: true
  
  # Logging
  logging_steps: 10
  save_steps: 500
  report_to: none
  
  # Output
  output_dir: output/full-sft

compute:
  gpus: 1
  min_vram_gb: 40
```

Changes from current: Added lr_scheduler, warmup, weight_decay, max_grad_norm, gradient_accumulation, per_device_train_batch_size, logging_steps, save_steps, output_dir, data_path, compute.min_vram_gb.

#### 1b. `recipes/training/lora-sft.yaml` — UPGRADE

LoRA parameter-efficient fine-tuning. Only adapter weights are trained. Works on consumer GPUs.

```yaml
type: training
description: LoRA SFT — parameter-efficient fine-tuning via TRL
config:
  algorithm: sft
  
  # Data
  data_path: training_data.jsonl
  max_length: 2048
  
  # Training
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-04
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Precision
  bf16: true
  
  # LoRA
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  
  # Logging
  logging_steps: 10
  save_steps: 500
  report_to: none
  
  # Output
  output_dir: output/lora-sft

compute:
  gpus: 1
  min_vram_gb: 16
```

Changes from current: Added lora_dropout, lora_target_modules (critical — users need to know which modules to target), lr_scheduler, warmup, weight_decay, data_path, compute hints.

#### 1c. `recipes/training/qlora-sft.yaml` — NEW

QLoRA — 4-bit quantized LoRA. Smallest VRAM footprint. Can fine-tune 7B models on a single 24GB GPU.

```yaml
type: training
description: QLoRA SFT — 4-bit quantized LoRA for minimal VRAM
config:
  algorithm: sft
  
  # Data
  data_path: training_data.jsonl
  max_length: 2048
  
  # Training
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-04
  lr_scheduler_type: cosine
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Precision
  bf16: true
  
  # QLoRA
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  qlora: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  
  # Memory
  gradient_checkpointing: true
  
  # Logging
  logging_steps: 10
  save_steps: 500
  report_to: none
  
  # Output
  output_dir: output/qlora-sft

compute:
  gpus: 1
  min_vram_gb: 10
```

#### 1d. `recipes/training/dpo.yaml` — NEW

Direct Preference Optimization. Trains on preference pairs (chosen vs rejected). Requires a preference dataset with chosen/rejected columns.

```yaml
type: training
description: DPO — direct preference optimization via TRL
config:
  algorithm: dpo
  
  # Data — must be preference pairs with chosen/rejected columns
  data_path: preference_data.jsonl
  max_length: 2048
  max_prompt_length: 1024
  
  # Training
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-07
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # DPO params
  beta: 0.1
  loss_type: sigmoid
  
  # Precision
  bf16: true
  
  # LoRA (DPO typically uses LoRA)
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  
  # Logging
  logging_steps: 10
  save_steps: 500
  report_to: none
  
  # Output
  output_dir: output/dpo

compute:
  gpus: 1
  min_vram_gb: 24
```

#### 1e. `recipes/training/kto.yaml` — NEW

Kahneman-Tversky Optimization. Like DPO but only needs binary feedback (good/bad), not preference pairs. Simpler data requirements.

```yaml
type: training
description: KTO — binary feedback optimization via TRL
config:
  algorithm: kto
  
  # Data — binary feedback with label column (true/false)
  data_path: feedback_data.jsonl
  max_length: 2048
  max_prompt_length: 1024
  
  # Training
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-07
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Precision
  bf16: true
  
  # LoRA
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  
  # Logging
  logging_steps: 10
  save_steps: 500
  report_to: none
  
  # Output
  output_dir: output/kto

compute:
  gpus: 1
  min_vram_gb: 24
```

#### 1f. `recipes/training/grpo.yaml` — UPGRADE

Group Relative Policy Optimization. Online RL — generates completions and scores them with a reward function. Most computationally expensive.

```yaml
type: training
description: GRPO — group relative policy optimization via TRL
config:
  algorithm: grpo
  
  # Data
  data_path: prompts.jsonl
  max_prompt_length: 512
  max_completion_length: 256
  
  # Training
  max_steps: 500
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-06
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # GRPO params
  num_generations: 8
  temperature: 0.9
  beta: 0.0
  epsilon: 0.2
  
  # Precision
  bf16: true
  
  # LoRA (GRPO typically uses LoRA to save VRAM for generation)
  use_peft: true
  lora_r: 16
  lora_alpha: 8
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  
  # Memory
  gradient_checkpointing: true
  
  # Logging
  logging_steps: 10
  save_steps: 100
  report_to: none
  
  # Output
  output_dir: output/grpo

compute:
  gpus: 1
  min_vram_gb: 24
```

Changes from current: Added max_prompt_length, lr_scheduler, warmup, weight_decay, max_grad_norm, lora_target_modules, gradient_checkpointing, max_steps (instead of epochs — GRPO uses steps), data_path, compute hints. Removed reward_model field (reward functions are defined differently in TRL).

#### 1g. `recipes/training/gkd.yaml` — NEW

Generalized Knowledge Distillation. Trains a small student model to mimic a larger teacher model. Uses on-policy (student-generated) and off-policy (dataset) data mixed via `lmbda`. Requires two models loaded simultaneously.

TRL class: `trl.experimental.gkd.GKDTrainer` + `GKDConfig`

```yaml
type: training
description: GKD — generalized knowledge distillation via TRL (experimental)
config:
  algorithm: gkd
  
  # Data — chat-format messages
  data_path: training_data.jsonl
  
  # Student model (the model being trained)
  model_name_or_path: Qwen/Qwen3-0.6B
  
  # Teacher model (the larger model to distill from)
  teacher_model_name_or_path: Qwen/Qwen3-4B
  
  # Training
  max_steps: 500
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-04
  lr_scheduler_type: cosine
  warmup_steps: 50
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # GKD params
  temperature: 0.9
  lmbda: 0.5          # 0=all off-policy (dataset), 1=all on-policy (student rollouts)
  beta: 0.5           # 0=forward KL, 1=reverse KL, 0.5=symmetric JSD
  max_new_tokens: 512
  disable_dropout: true
  
  # Precision
  bf16: true
  
  # Logging
  logging_steps: 10
  save_steps: 100
  report_to: none
  
  # Output
  output_dir: output/gkd

compute:
  gpus: 1
  min_vram_gb: 24
```

Key notes for implementer:
- GKD requires loading both student AND teacher models. The training container must handle `teacher_model_name_or_path` and pass it to `GKDTrainer(teacher_model=...)`.
- Import path: `from trl.experimental.gkd import GKDConfig, GKDTrainer`
- Dataset format: standard chat messages format (`messages` column with role/content dicts).
- `lmbda` controls the mix: 0.5 means 50% of batches use student-generated rollouts, 50% use dataset. Start with 0.5.
- `beta` controls the divergence: 0=forward KL, 1=reverse KL, 0.5=symmetric JSD. Paper recommends 0.5.

#### 1h. `recipes/training/gold.yaml` — NEW

General Online Logit Distillation. Superset of GKD — adds cross-tokenizer distillation via ULD (Universal Logit Distillation). Enables distilling from a teacher with a DIFFERENT tokenizer (e.g., Qwen teacher → Llama student).

TRL class: `trl.experimental.gold.GOLDTrainer` + `GOLDConfig` (also importable from `trl` directly)

```yaml
type: training
description: GOLD — online logit distillation with cross-tokenizer support via TRL (experimental)
config:
  algorithm: gold
  
  # Data — chat-format messages
  data_path: training_data.jsonl
  
  # Student model
  model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
  
  # Teacher model (can be a different architecture/tokenizer than student)
  teacher_model_name_or_path: Qwen/Qwen2.5-7B-Instruct
  teacher_tokenizer_name_or_path: Qwen/Qwen2.5-7B-Instruct
  
  # Training
  max_steps: 500
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-04
  lr_scheduler_type: cosine
  warmup_steps: 50
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # GOLD params (inherits GKD params)
  temperature: 0.9
  lmbda: 0.5
  beta: 0.5
  max_completion_length: 512
  disable_dropout: true
  
  # ULD — cross-tokenizer distillation
  use_uld_loss: true
  uld_use_hybrid_loss: true
  
  # Precision
  bf16: true
  
  # Logging
  logging_steps: 10
  save_steps: 100
  report_to: none
  
  # Output
  output_dir: output/gold

compute:
  gpus: 1
  min_vram_gb: 40
```

Key notes for implementer:
- GOLD inherits from GKD. GOLDTrainer is a subclass of SFTTrainer with GKD scheduling.
- Import path: `from trl.experimental.gold import GOLDConfig, GOLDTrainer` or `from trl import GOLDConfig, GOLDTrainer`
- `teacher_tokenizer_name_or_path` is required when teacher and student use different tokenizers.
- `use_uld_loss: true` enables Universal Logit Distillation for cross-tokenizer setups. Without it, GOLD behaves like GKD.
- `uld_use_hybrid_loss: true` combines matched-token and unmatched-token distillation.
- Higher VRAM requirement than GKD because cross-tokenizer alignment is more expensive.
- When student and teacher share the same tokenizer, you can set `use_uld_loss: false` and GOLD reduces to GKD.

#### 1i. DELETE `recipes/training/osft.yaml`

This is just full-sft with slightly different hyperparams. It adds confusion without value. The `full-sft.yaml` template with model-specific overrides in `recipes/models/` covers the same use case.

---

### Phase 2: Upgrade Model Presets

Model presets use `extends:` to inherit from a base template and override model-specific fields. Keep this pattern — it's clean.

#### 2a. `recipes/models/llama3-8b-lora.yaml` — UPGRADE

```yaml
extends: training/lora-sft
description: Llama 3.1 8B Instruct — LoRA SFT
config:
  model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
  max_length: 8192
  per_device_train_batch_size: 2
  gradient_checkpointing: true

compute:
  min_vram_gb: 24
```

Changes: Added max_length override (Llama supports 8k), batch_size adjustment, gradient_checkpointing (needed at 8B), compute hint.

#### 2b. `recipes/models/qwen-1.5b-lora.yaml` — UPGRADE

```yaml
extends: training/lora-sft
description: Qwen 2.5 1.5B Instruct — LoRA SFT (lightweight)
config:
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  max_length: 4096
  lora_r: 32
  lora_alpha: 64
  learning_rate: 1.0e-04
  per_device_train_batch_size: 8

compute:
  min_vram_gb: 8
```

Minor: Added compute hint, batch_size override (small model fits more).

#### 2c. `recipes/models/qwen-7b-sft.yaml` — UPGRADE

```yaml
extends: training/full-sft
description: Qwen 2.5 7B Instruct — full SFT
config:
  model_name_or_path: Qwen/Qwen2.5-7B-Instruct
  max_length: 4096
  gradient_checkpointing: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

compute:
  gpus: 1
  min_vram_gb: 40
```

Changes: Added batch_size/accumulation overrides for 7B model, compute hint.

#### 2d. `recipes/models/qwen3-4b-grpo.yaml` — UPGRADE

```yaml
extends: training/grpo
description: Qwen3 4B — GRPO reinforcement learning
config:
  model_name_or_path: Qwen/Qwen3-4B
  max_prompt_length: 512
  max_completion_length: 256
  num_generations: 4
  gradient_checkpointing: true

compute:
  gpus: 1
  min_vram_gb: 24
```

Changes: Added max_prompt_length, max_completion_length, reduced num_generations (4B model), compute hint.

#### 2e. `recipes/models/qwen3-0.6b-lora.yaml` — NEW

Small model for quick testing and prototyping. Users should start here.

```yaml
extends: training/lora-sft
description: Qwen3 0.6B — LoRA SFT (fastest, for prototyping)
config:
  model_name_or_path: Qwen/Qwen3-0.6B
  max_length: 2048
  per_device_train_batch_size: 8
  num_train_epochs: 3
  learning_rate: 3.0e-04

compute:
  gpus: 1
  min_vram_gb: 8
```

#### 2f. `recipes/models/qwen3-4b-lora.yaml` — NEW

Sweet spot for task models — small enough for single GPU, large enough for good quality.

```yaml
extends: training/lora-sft
description: Qwen3 4B — LoRA SFT (recommended for task models)
config:
  model_name_or_path: Qwen/Qwen3-4B
  max_length: 4096
  per_device_train_batch_size: 2
  gradient_checkpointing: true

compute:
  gpus: 1
  min_vram_gb: 16
```

#### 2g. `recipes/models/llama3-8b-qlora.yaml` — NEW

QLoRA variant for Llama — fits on consumer GPUs.

```yaml
extends: training/qlora-sft
description: Llama 3.1 8B Instruct — QLoRA SFT (low VRAM)
config:
  model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
  max_length: 4096
  per_device_train_batch_size: 2
  gradient_checkpointing: true

compute:
  gpus: 1
  min_vram_gb: 12
```

#### 2h. `recipes/models/qwen3-4b-dpo.yaml` — NEW

DPO model preset — demonstrates how to use DPO with a specific model.

```yaml
extends: training/dpo
description: Qwen3 4B — DPO preference tuning
config:
  model_name_or_path: Qwen/Qwen3-4B
  max_length: 4096
  gradient_checkpointing: true

compute:
  gpus: 1
  min_vram_gb: 24
```

#### 2i. `recipes/models/qwen3-0.6b-gkd.yaml` — NEW

GKD distillation — Qwen3 4B teacher distilling into Qwen3 0.6B student. Same tokenizer, so no ULD needed.

```yaml
extends: training/gkd
description: Qwen3 0.6B ← Qwen3 4B — GKD distillation (same-tokenizer)
config:
  model_name_or_path: Qwen/Qwen3-0.6B
  teacher_model_name_or_path: Qwen/Qwen3-4B
  max_new_tokens: 512
  per_device_train_batch_size: 4

compute:
  gpus: 1
  min_vram_gb: 16
```

#### 2j. `recipes/models/llama-1b-gold.yaml` — NEW

GOLD cross-tokenizer distillation — Qwen teacher into Llama student. Demonstrates ULD for cross-architecture distillation.

```yaml
extends: training/gold
description: Llama 3.2 1B ← Qwen 2.5 7B — GOLD cross-tokenizer distillation
config:
  model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
  teacher_model_name_or_path: Qwen/Qwen2.5-7B-Instruct
  teacher_tokenizer_name_or_path: Qwen/Qwen2.5-7B-Instruct
  use_uld_loss: true
  uld_use_hybrid_loss: true
  max_completion_length: 512

compute:
  gpus: 1
  min_vram_gb: 40
```

---

### Phase 3: Verify `extends` Mechanism

The `extends:` field in model presets references a base template by relative path. Verify that the amortized CLI/server correctly resolves and merges these. The behavior should be:

1. Load the base template (e.g., `recipes/training/lora-sft.yaml`)
2. Deep-merge the model preset's `config` on top
3. Model preset values override base template values
4. Arrays (like `lora_target_modules`) are replaced, not appended

If `extends` is not yet implemented, the model presets should be standalone (duplicate the full config). Note this in a comment at the top of each model preset file.

---

## File Summary

### Base Templates (`recipes/training/`)

| File | Action | Algorithm | TRL Class |
|---|---|---|---|
| `full-sft.yaml` | **Upgrade** — add missing training params | SFT (full weights) | `SFTTrainer` |
| `lora-sft.yaml` | **Upgrade** — add lora_target_modules, training params | SFT (LoRA) | `SFTTrainer` + PEFT |
| `qlora-sft.yaml` | **New** — 4-bit quantized LoRA | SFT (QLoRA) | `SFTTrainer` + PEFT + BnB |
| `dpo.yaml` | **New** — direct preference optimization | DPO | `DPOTrainer` |
| `kto.yaml` | **New** — binary feedback optimization | KTO | `KTOTrainer` |
| `grpo.yaml` | **Upgrade** — add missing params | GRPO | `GRPOTrainer` |
| `gkd.yaml` | **New** — generalized knowledge distillation | GKD | `trl.experimental.gkd.GKDTrainer` |
| `gold.yaml` | **New** — online logit distillation (cross-tokenizer) | GOLD | `trl.experimental.gold.GOLDTrainer` |
| `osft.yaml` | **Delete** — redundant with full-sft | — | — |

### Model Presets (`recipes/models/`)

| File | Action | Model | Algorithm |
|---|---|---|---|
| `llama3-8b-lora.yaml` | **Upgrade** — add overrides | Llama 3.1 8B | LoRA SFT |
| `qwen-1.5b-lora.yaml` | **Upgrade** — add compute hint | Qwen 2.5 1.5B | LoRA SFT |
| `qwen-7b-sft.yaml` | **Upgrade** — add batch/accumulation | Qwen 2.5 7B | Full SFT |
| `qwen3-4b-grpo.yaml` | **Upgrade** — add GRPO params | Qwen3 4B | GRPO |
| `qwen3-0.6b-lora.yaml` | **New** — prototyping model | Qwen3 0.6B | LoRA SFT |
| `qwen3-4b-lora.yaml` | **New** — recommended task model | Qwen3 4B | LoRA SFT |
| `llama3-8b-qlora.yaml` | **New** — QLoRA variant | Llama 3.1 8B | QLoRA SFT |
| `qwen3-4b-dpo.yaml` | **New** — DPO preset | Qwen3 4B | DPO |
| `qwen3-0.6b-gkd.yaml` | **New** — same-tokenizer distillation | Qwen3 0.6B ← 4B | GKD |
| `llama-1b-gold.yaml` | **New** — cross-tokenizer distillation | Llama 1B ← Qwen 7B | GOLD |

### NOT included (out of scope for now)

- **veRL GRPO** — Optional scale backend for multi-node GRPO. Add when multi-node training is needed.
- **FSDP/DeepSpeed configs** — Multi-GPU distribution strategy configs. Add when multi-node training is supported.
- **Vision model presets** — amortized focuses on text task models. Add VLM support later.

---

## Key Decisions for the Implementer

1. **`algorithm` field** — This maps to TRL trainer classes. Valid values: `sft`, `dpo`, `kto`, `grpo`, `gkd`, `gold`. The worker/container needs to route these to the right TRL Trainer class:
   - `sft` → `trl.SFTTrainer`
   - `dpo` → `trl.DPOTrainer`
   - `kto` → `trl.KTOTrainer`
   - `grpo` → `trl.GRPOTrainer`
   - `gkd` → `trl.experimental.gkd.GKDTrainer` (experimental)
   - `gold` → `trl.experimental.gold.GOLDTrainer` (experimental)
   
   Check that the training container handles all 6. GKD and GOLD require loading a teacher model alongside the student — the container must extract `teacher_model_name_or_path` from the config and pass it to the trainer.

2. **`data_path` field** — This tells the trainer where to find training data. Expected formats per algorithm:
   - SFT: JSONL with `messages` column (chat format: `[{role, content}, ...]`)
   - DPO: JSONL with `chosen`/`rejected` columns (preference pairs)
   - KTO: JSONL with `completion` and `label` (true/false) columns
   - GRPO: JSONL with `prompt` column
   - GKD: JSONL with `messages` column (same as SFT)
   - GOLD: JSONL with `messages` column (same as SFT)

3. **`lora_target_modules`** — The list `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` works for Llama, Qwen, and most transformer architectures. For models with different module names, the model preset should override this list.

4. **`compute` section** — This is metadata for the amortized scheduler, not a TRL field. `min_vram_gb` helps the control plane pick appropriate GPU nodes. `gpus` indicates how many GPUs are needed. These should be ignored by the TRL runner and only used by the job dispatcher.

5. **`report_to: none`** — Default to `report_to: none`. On OpenShift AI with MLflow deployed, amortized will override this to `report_to: mlflow` by injecting `MLFLOW_TRACKING_URI` env var into training containers. Users can also explicitly set `report_to: wandb` or `report_to: tensorboard`.

## Reference

Oumi training configs for parameter reference:
- SFT full: `/Users/shiv/workspace/oumi/configs/recipes/smollm/sft/135m/train.yaml`
- LoRA: `/Users/shiv/workspace/oumi/configs/recipes/phi3/sft/lora_train.yaml`
- QLoRA: `/Users/shiv/workspace/oumi/configs/recipes/llama3_1/sft/8b_qlora/train.yaml`
- DPO: `/Users/shiv/workspace/oumi/configs/recipes/phi3/dpo/train.yaml`
- KTO: `/Users/shiv/workspace/oumi/configs/recipes/phi3/kto/train.yaml`
- GRPO: `/Users/shiv/workspace/oumi/configs/examples/grpo_tldr/train.yaml`
- GKD: `/Users/shiv/workspace/oumi/configs/examples/gkd/train.yaml`
- GOLD: `/Users/shiv/workspace/oumi/configs/examples/gold/train.yaml`
