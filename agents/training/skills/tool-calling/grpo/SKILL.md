---
name: training-grpo
description: >-
  Train a small model with GRPO (reinforcement learning with a verifiable reward) via
  training-hub's verl backend. Recommended for tool calling: the model learns to pick the
  right tool and arguments, scored against a target call. Also covers short checkable
  answers (math, short-answer QA) with the math reward, and requests for GRPO / RL.
  LoRA (lora_grpo, default) or full-weight (grpo). Trains tool-call traces from an SDG
  job (parent_job_id) or an uploaded dataset / split (data_run_id). Covers model
  selection, GPUs, reward choice, and the GRPO hyperparameters. Bundles the
  supported-model list and a config template.
metadata:
  stage: training
---

# Tool-Calling GRPO Training Guide

GRPO (Group Relative Policy Optimization) is the recommended training method for tool
calling. Instead of imitating a target response, the model samples several answers per
prompt, a reward scores each one against the target tool call, and training moves the
model toward the higher-scoring answers — so it learns to call the right tool with the
right arguments, not just to copy a format.

> **Scope.** This guide is written around **tool calling** (`reward_type: "tool_call"`):
> each training row is a prompt, the available tools, and the one tool call the model
> should make. **Only single-turn tool calling is supported for now** — one user request,
> one tool call. Multi-turn tasks (a conversation with several tool calls and tool
> results, or an agent loop) are not supported: tell the user so, and don't train on
> multi-turn `messages` traces. GRPO also covers any task where every prompt has a **short, checkable
> reference answer** — math, short-answer QA, short labels — via the `math` reward (see
> Reward Functions); the model, GPU, and hyperparameter guidance still apply. (Note: for
> imitating full responses — chat, doc-grounded QA, style — use the
> **`training-knowledge-ingestion-osft`** skill; for classifying text into a fixed set of
> categories, use **`training-classifier`** — both are faster and cheaper than RL for
> those tasks.)

## How This Works

You will gather requirements and call `validate_training_job` with the appropriate
parameters. The tool validates all fields and rejects invalid combinations (e.g. a
dataset without the columns the reward needs, `prompt_batch_size × group_size` not
divisible by `4 × nproc_per_node`, or LoRA fields with full-weight `grpo`).

**VRAM estimates.** The workflow's estimate tools model GRPO: set `method` to `lora_grpo`
or `grpo` and pass the config's `max_prompt_length`, `max_tokens`,
`gpu_memory_utilization` and (`lora_grpo`) `lora_r`, plus `gpu_memory_gb` for the GPU type
from the Compute budget question (80 for an H100). Unset fields use the job defaults. The
vLLM rollout engine and the trainer share each GPU, and vLLM reserves
`gpu_memory_utilization` of it, so the per-GPU need drops only a little with more GPUs for
`lora_grpo`; full `grpo` shards its larger trainer state and drops more. Show the VRAM card
for the model options (step 1) and for the final config (Phase 3). Skip the workflow's
Training Method Selection VRAM step: the method is decided in step 5.

Every parameter is adjustable:
- Model selection based on available compute
- GPU count based on model size and method
- Reward based on the task and the data
- LoRA vs full-weight training
- Rollout length and batch sizes based on the data

## Requirement Gathering

Ask the user, one question per message:

1. **What model?** — Read `supported_models.json` (bundled with this skill) for the list
   of instruct models with tool-calling support, sized for GRPO. **Recommend the
   `default` entry first** and present the options with their sizes and descriptions;
   never auto-select. Start at ≤4B — RL on top of a model that already gets some calls
   right learns fastest. Show the VRAM comparison card first, as in the workflow's Student
   Model Selection, with `method` `lora_grpo` and the default lengths (the data isn't chosen
   yet). Multimodal checkpoints such as Qwen3.5 are handled automatically: the job detects them and sets `language_model_only`.

   **If the user wants a model that is NOT in the list**, follow this flow:

   a. **Validate the model.** Check three things:
      - **Valid HuggingFace model ID** (e.g. `org/model-name`) that can be resolved and
        downloaded.
      - **Generative instruct model.** GRPO samples answers, so it needs a decoder
        (chat/instruct) model — not an encoder or embedding model.
      - **Tool-calling support** (for `tool_call`): the model's chat template supports
        tools. Base (non-instruct) models rarely produce a parseable call.

      If the model passes all three, accept it and proceed.

   b. **If the model looks risky** (fails a check, or it can't be determined), warn the
      user and suggest a similar model from the list. If the user insists, allow it —
      state clearly: "This model is not in our verified list and may not be compatible.
      We will proceed with your choice, but if the job fails, consider switching to one
      of the supported models." Then continue normally.

   The model choice affects all downstream parameters — gather them after.

2. **GPUs** — Set `nproc_per_node` to the smallest GPU count whose per-GPU estimate fits
   the GPU's memory. Stay within the compute budget gathered in
   the workflow (H100 80 GB by default) and briefly explain the choice. Keep
   `prompt_batch_size × group_size` divisible by `4 × nproc_per_node` when you change it.

3. **Training data** — `parent_job_id` (SDG job) or `data_run_id` (uploaded dataset /
   split), as in the workflow's Job Chaining. Ask for it if not already in the
   conversation. Check the rows match the Data Format for the reward (below) and confirm
   the record count like any other training job. For `tool_call`, every row must be
   **single-turn** (`question`, `tools`, `target_tool_name`, `target_arguments`). If the
   dataset holds multi-turn `messages` traces instead, stop: say multi-turn tool calling
   isn't supported yet and hand back for a single-turn dataset — don't submit it.

4. **Reward** — `tool_call` for tool-calling data (the default for this skill), `math`
   for short checkable answers. Pick it from the task and the data (see Reward
   Functions); only ask when the task fits both, otherwise state the choice in one line.
   For `math`, set `text_column` / `label_column` to the prompt and reference-answer
   columns.

5. **Method** — `lora_grpo` (LoRA adapter, default, recommended) unless the user asks for
   full fine-tuning, then `grpo` (all weights; much more GPU memory — re-estimate with `grpo` and revisit step 2).
   Don't ask otherwise.

Don't ask about the GRPO hyperparameters unless the user brings them up.

## Tool Parameters

Call `validate_training_job` (as a single `config_json`, per the workflow) with these
parameters — `training-config-template.json` (bundled with this skill) holds the same
starting point:

```json
{
  "algorithm": "lora_grpo",
  "model_name_or_path": "<model>",
  "reward_type": "tool_call",
  "num_train_epochs": 1,
  "group_size": 8,
  "prompt_batch_size": 64,
  "max_prompt_length": 2048,
  "max_tokens": 512,
  "temperature": 1.0,
  "learning_rate": 0.0001,
  "lora_r": 32,
  "lora_alpha": 64,
  "gpu_memory_utilization": 0.4,
  "nproc_per_node": 1,
  "parent_job_id": "<SDG_JOB_ID>",
  "topic": "<1-5 word topic>"
}
```

For the `math` reward, set `"reward_type": "math"`, add `text_column` / `label_column`,
and raise `max_tokens` to ~1024 for step-by-step answers. For full-weight training, set
`"algorithm": "grpo"`, drop `lora_r` / `lora_alpha`, and lower `learning_rate` (5e-5 for a ~4B model).

### Key Parameters — Adjust Based on Model Size, Data, and Compute

Do NOT use static defaults blindly. Adapt each parameter to the user's dataset, model
choice, and GPU count.

| Param | How to set |
|-------|-----------|
| `model_name_or_path` | User's chosen model |
| `algorithm` | `lora_grpo` (default). `grpo` only when the user asks for full fine-tuning |
| `reward_type` | `tool_call` for tool-call traces; `math` for short checkable answers. See Reward Functions |
| `num_train_epochs` | Passes over the data. 1 for a first run or 2000+ samples; 2–3 for <1000 samples |
| `learning_rate` | For a ~4B model: 1e-4 for `lora_grpo`, 5e-5 for `grpo`. Roughly double for ≤2B, halve for 7–9B; scale by √(rollouts per step ÷ 512) when you change `prompt_batch_size` × `group_size`. Higher than usual GRPO recipes because the verl backend averages a whole batch into one optimizer step (when `prompt_batch_size` ≤ 256) rather than several mini-batch steps. Halve it if the reward collapses |
| `group_size` | Answers sampled per prompt (8). More gives a steadier signal but costs rollout time |
| `prompt_batch_size` | Prompts per step (64). `prompt_batch_size × group_size` must be divisible by `4 × nproc_per_node` |
| `max_prompt_length` | Must fit the longest prompt **including the tool definitions** (they are added to the system prompt); longer prompts are **dropped**. 2048 for a few small tools; raise it for many or large tool schemas |
| `max_tokens` | Longest answer. 512 for a single tool call (thinking models may need 1024); ~1024 for step-by-step math |
| `temperature` | Rollout sampling temperature. 1.0 keeps answers varied enough to learn from |
| `lora_r` / `lora_alpha` | `lora_grpo` only: 32 / 64. Not allowed with `grpo` |
| `n_train` | Cap on training samples (default 5000). Raise it to use a larger dataset fully |
| `nproc_per_node` | From step 2; rollout and training share the GPUs |
| `gpu_memory_utilization` | GPU fraction for the vLLM rollout engine (0.4). Lower it on out-of-memory |
| `language_model_only` | Leave unset: the job sets it automatically for multimodal checkpoints (e.g. Qwen3.5) so vLLM loads only the text model. Set `false` only to override |
| `topic` | **Required.** 1–5 word description of what the model does (e.g. "calendar tool calling"). Infer it from the task if not stated |

Only state a value to the user if it is set in the config (the workflow's Phase 3 rule),
and say WHY you chose it: "2 epochs because the dataset is small (400 samples)" or
"max_prompt_length 4096 because the tool schemas are large".

### Job Chaining

Set `parent_job_id` to the SDG job ID, or `data_run_id` to an uploaded dataset or split —
never both. The worker resolves the dataset, downloads it, and flattens it into the
single JSONL training-hub reads. No manual data path configuration needed.

## Data Format

### `tool_call` — single-turn only

One row per prompt, all four fields required. `tools` uses the OpenAI function-schema format; `target_arguments`
is an object (a JSON string is also accepted):

```json
{"question": "What's the weather in Paris tomorrow?", "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the forecast", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "day": {"type": "string"}}, "required": ["city"]}}}], "target_tool_name": "get_weather", "target_arguments": {"city": "Paris", "day": "tomorrow"}}
```

**Multi-turn tool calling is not supported for now.** Don't use conversations with
several tool calls and tool results (`messages` traces with `tool_calls` / tool-role
turns), even though training-hub can read them — that path isn't supported on the
platform yet. Rows missing any of the four fields are skipped during training, so a
mostly-incomplete dataset trains on far fewer samples than its record count.

### `math`

A prompt column and a reference-answer column, e.g.:

```json
{"question": "What is 15% of 80?", "answer": "12"}
```

A `messages`-only (chat SFT) dataset has no separate reference answer, so validation
rejects it for `math`.

## Reward Functions

The reward decides what a "better answer" is. Only the rewards below exist; set one via
`reward_type`.

| `reward_type` | Scores | Data needs |
|---|---|---|
| `tool_call` | 1.0 = right tool and arguments; 0.5 = right tool, wrong arguments; 0 = wrong or no call | Single-turn tool calls (Data Format above); multi-turn not supported |
| `math` | 1.0 = final answer matches the reference (exact or math-equivalent); else 0 | A prompt column and a reference-answer column |

**`tool_call`** parses the model's call from `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
or a bare `{"name": ..., "arguments": ...}` JSON object (thinking blocks are ignored). Tool
names match loosely (case, `-`/`.`/`_`, and a namespace prefix are ignored); arguments must match exactly after
normalization (key order doesn't matter). `text_column` / `label_column` don't apply.

**`math`** reads the answer from `\boxed{}` or `<answer></answer>`. Despite the name it
works for any short answer — numbers, expressions, single words, labels. The default
system prompt asks for the final answer in `\boxed{}`: leave `system_prompt` unset unless
the user wants different instructions, and if they set one it must still ask for
`\boxed{}`.
- `text_column` — prompt column. Unset: first of question, problem, prompt, input, text.
- `label_column` — reference-answer column. Unset: first of ground_truth, answer, label,
  category, output.

**Custom rewards are not available yet.** Validation rejects `reward_fn` (and
`rollout_fn` / `iteration_callback`): training-hub would import arbitrary code inside the
training pod. Never put code or a module path in a job config. If no reward above fits
the task (e.g. open-ended writing, rubric scoring), say GRPO can't score it today and
offer OSFT on reference responses instead.

## Compute Requirements

- GRPO needs at least 1 GPU; there is no CPU path
- If a run runs out of memory, lower `gpu_memory_utilization`, `max_tokens`, or
  `prompt_batch_size` (keeping the divisibility rule), or add GPUs
- Checkpoints are saved during training; only the final one is exported

## After Training

The trained model is registered in MLflow's model registry. The user can deploy it via the serve pipeline or download the adapter weights.