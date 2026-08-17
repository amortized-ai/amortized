# Training Skill Guidance

## Sub-Skill Guides

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| knowledge-ingestion/osft | `skills/training/knowledge-ingestion/osft/` | Knowledge QA, FAQ bots, doc-grounded tasks |

## Algorithm Selection

The platform supports multiple training algorithms. Choose based on the
user's task pattern and data format:

| Algorithm | Best For | Data Needed |
|-----------|----------|-------------|
| `osft` | Knowledge QA, grounded tasks — 30%+ better than SFT for open-book settings | SFT messages |
| `sft` | General supervised fine-tuning — solid default for most task patterns | SFT messages |
| `lora_sft` | Same as SFT but parameter-efficient — lower VRAM, faster training | SFT messages |
| `dpo` | Preference alignment — when you have chosen/rejected response pairs | Preference pairs |
| `kto` | Binary preference — simpler than DPO, only needs good/bad labels | Binary labels |
| `grpo` / `lora_grpo` | Reinforcement from rewards — when you have a reward signal or verifier | Reward scores |
| `gkd` | Knowledge distillation — compress a large model's behavior into a smaller one | Teacher/student |

**Recommended defaults by task pattern:**
- Knowledge QA → `osft` (validated advantage for grounded tasks)
- Classification, routing → `lora_sft` (straightforward pattern, VRAM-efficient)
- Summarization, transformation → `sft` or `lora_sft` (generation tasks)
- Extraction → `sft` with structured output training
- When the user has preference data → `dpo` or `kto`
- When the user wants to distill a frontier model → `gkd`

Aliases: `lora` → `lora_sft`, `qlora` → `lora_sft` (with `load_in_4bit: true`).

**If a sub-skill guide exists** for the pattern (e.g. OSFT for knowledge
QA), load it for hyperparameter guidance.

**If no guide exists**, use the defaults from the algorithm table and
adapt based on model size and dataset. The training config accepts
arbitrary extra fields — any Training Hub parameter can be passed through.

## Student Model Selection

You MUST show VRAM estimates before presenting model options. The user
needs to see GPU memory requirements before choosing.

1. Call `estimate_training_resources` for EACH candidate model size
   with the default method (lora)
2. Call `show_vram_estimate` with ALL collected estimates
3. THEN call `present_options` with model choices

## Training Method Selection

You MUST show VRAM estimates before presenting method options.

1. Call `estimate_training_resources` with the selected model size for
   EACH method (lora, qlora, osft, sft)
2. Call `show_vram_estimate` with ALL collected estimates
3. THEN call `present_options` with method choices

## Training Confirmation

Before calling `validate_training_job`, call
`estimate_training_resources` with the final model size and method,
then call `show_vram_estimate` so the user sees what they are
committing to.

## Job Chaining

Set `parent_job_id` to the SDG job ID. The worker resolves the SDG
output from MLflow and sets `data_path` automatically. No manual
data path configuration needed.

## Hyperparameter Guidance (General)

When no sub-skill guide covers the algorithm, use these starting points:

| Param | Guidance |
|-------|----------|
| `num_train_epochs` | 3–5 for <1000 samples, 2–3 for 1000–5000, 1–2 for 5000+ |
| `learning_rate` | 2e-5 for 9B, 5e-5 for 4B, 1e-4 for 0.8B–2B |
| `effective_batch_size` | 32 (1 GPU × 32 per-GPU batch) |
| `max_length` | Match to the longest example in the training data |
| `nproc_per_node` | Always 1 |
| `bf16` | true for Ampere+ GPUs (A100, H100) |
| `topic` | Required. 1–5 word description of the task. |

More data → fewer epochs. Larger models → lower learning rate. When in
doubt, start conservative and iterate.
