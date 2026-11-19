---
name: training-grpo
description: >-
  LoRA GRPO (reinforcement learning with a verifiable reward) via training-hub's verl
  backend. Use when each prompt has a checkable reference answer (math, short labels,
  tool calls) rather than a full target response. Ships a fixed math demo payload
  (Qwen3.5-2B on MATH-500) for a quick end-to-end test.
metadata:
  stage: training
---

# GRPO (LoRA) — Training Guide

`algorithm="lora_grpo"` runs `thub lora-grpo` (training-hub, verl backend). For every prompt
the model samples `group_size` answers, each is scored by the reward, and the LoRA adapter is
updated toward the better-scoring answers. The result is exported like `lora_sft` (a PEFT
adapter under MLflow `model/`), so eval and serving handle it the same way.

## Reward (`reward_type`)

- `math` (default) — the answer is read from `\boxed{}` or `<answer></answer>` and scores 1.0
  on an exact or math-equivalent match with the reference answer, else 0. Works for short
  labels too. Data needs a prompt column (`question`/`problem`/`prompt`/`input`/`text`, or
  `text_column`) and a reference-answer column (`ground_truth`/`answer`/`label`/`category`/
  `output`, or `label_column`).
- `tool_call` — scores tool-call name and arguments. Data must hold tool-call traces:
  `question`, `tools`, `target_tool_name`, `target_arguments`, or multi-turn `messages`.

Custom Python rewards are not accepted (`reward_fn` is rejected).

## Data

Any one of: `parent_job_id` (SDG job), `data_run_id` (uploaded dataset / split), or
`data_path` — a HuggingFace dataset ID (its `train` split, else its only split) or a local path.

## Key knobs

| Field | Default | Notes |
|---|---|---|
| `num_train_epochs` | 15 | passes over the data (training-hub `num_iterations`) |
| `prompt_batch_size` | 100 | prompts per step; `prompt_batch_size x group_size` must divide by `4 x nproc_per_node` |
| `group_size` | 8 | answers sampled per prompt |
| `max_prompt_length` | 2048 | longer prompts are dropped |
| `max_tokens` | 512 | max answer length |
| `n_train` | 5000 | max training rows |
| `nproc_per_node` | 1 | GPUs (rollout and training share them) |
| `gpu_memory_utilization` | 0.45 | GPU fraction for the vLLM rollout engine |
| `language_model_only` | — | set `true` for Qwen3.5 (VLM checkpoints) so vLLM loads only the text model |

## Quick end-to-end test

`reference-payload.json` is a fixed demo: Qwen3.5-2B, MATH-500 (500 rows, so `n_train: 512`
uses all of them), math reward, 1 epoch of 16 steps on 1 GPU. It needs no SDG job or upload.

From the repo root, with the server on `:8000`:

```bash
jq 'del(._meta)' agents/training/skills/grpo/reference-payload.json \
  | curl -sS -X POST http://localhost:8000/api/v1/jobs/training/validate \
      -H 'Content-Type: application/json' -d @- | jq

jq 'del(._meta)' agents/training/skills/grpo/reference-payload.json \
  | curl -sS -X POST http://localhost:8000/api/v1/jobs/training \
      -H 'Content-Type: application/json' -d @- | jq '{id, status}'
```

Or pick the **GRPO LoRA (math demo)** starter template in Studio.

Check: the job succeeds, the MLflow run shows reward metrics per step, and `model/` holds
`adapter_config.json` + `adapter_model.safetensors`.
