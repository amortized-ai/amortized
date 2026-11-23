---
name: training-verifiable-tasks-grpo
description: >-
  Train a small model with GRPO (reinforcement learning with a verifiable reward) via
  training-hub's verl backend, for verifiable tasks: every prompt has an answer a reward
  function can check automatically. Covers tool calling (right tool and arguments), short
  checkable answers (math, short-answer QA), and anything a NVIDIA NeMo Gym verifier can
  score (multiple choice, string match, instruction following, structured output,
  long-context retrieval, code with unit tests, ...), plus requests for GRPO / RL. LoRA
  (lora_grpo, default) or full-weight (grpo). Trains on an SDG job (parent_job_id) or an
  uploaded dataset / split (data_run_id). Covers model selection, GPUs, reward selection,
  and the GRPO hyperparameters. Bundles the reward-function map, the supported-model list
  and a config template.
metadata:
  stage: training
---

# Verifiable Tasks — GRPO Training Guide

GRPO (Group Relative Policy Optimization) trains a model on tasks whose answers can be
checked automatically. Instead of imitating a target response, the model samples several
answers per prompt, a reward function scores each one (against a reference answer or a
rule), and training moves the model toward the higher-scoring answers. It learns to get
the answer right, not just to copy a format.

> **Scope.** A verifiable task is one where a reward function in `reward-functions.json`
> can score each answer: a tool call against a target call, a short answer against a
> reference, a letter against the correct option, text against checkable constraints,
> JSON against required keys, code against unit tests, ... **Only single-turn tasks are
> supported for now** — one prompt, one answer. Multi-turn tasks (a conversation with
> several tool calls and tool results, or an agent loop) are not supported: tell the user
> so, and don't train on multi-turn `messages` traces.
>
> GRPO is not always the right method for a verifiable task:
> - **The output is one label from a fixed set** (intent, topic, sentiment) → use
>   **`training-classifier`**: a classification head is faster and better calibrated.
> - **The goal is to imitate full responses** (chat, doc-grounded QA, style), or there is
>   no automatic check → use **`training-knowledge-ingestion-osft`**.
> - **The base model gets almost no answers right** (e.g. it doesn't know the output
>   format yet) → every rollout scores 0 and GRPO learns nothing. Say so, and suggest
>   supervised training on reference answers first (OSFT), or a stronger base model.

## How This Works

You will gather requirements and call `validate_training_job` with the appropriate
parameters. The tool validates all fields and rejects invalid combinations (e.g. a
dataset without the columns the reward needs, `prompt_batch_size × group_size` not
divisible by `4 × nproc_per_node`, or LoRA fields with full-weight `grpo`).

Unlike the OSFT/SFT/LoRA sub-skills, the VRAM estimator does **not** model GRPO (the vLLM
rollout engine and the trainer share the GPUs), so there is **no VRAM card**: skip the
workflow's Training Method Selection VRAM step, When the Config Doesn't Fit, and Phase 3's
resource estimation. GRPO still needs a GPU, so the Compute budget question applies.

Every parameter is adjustable:
- Model selection based on available compute
- GPU count based on model size and method
- Reward based on the task and the data
- LoRA vs full-weight training
- Rollout length and batch sizes based on the data

## Requirement Gathering

Ask the user, one question per message:

1. **What model?** — Read `supported_models.json` (bundled with this skill) for the list
   of instruct models with tool-calling support, sized for GRPO (they suit the other
   rewards too). **Recommend the `default` entry first** and present the options with
   their sizes and descriptions; never auto-select. Start at ≤4B — RL on top of a model
   that already gets some answers right learns fastest. No VRAM comparison card. Multimodal
   checkpoints such as Qwen3.5 are handled automatically: the job detects them and sets `language_model_only`.

   **If the user wants a model that is NOT in the list**, follow this flow:

   a. **Validate the model.** Check three things:
      - **Valid HuggingFace model ID** (e.g. `org/model-name`) that can be resolved and
        downloaded.
      - **Generative instruct model.** GRPO samples answers, so it needs a decoder
        (chat/instruct) model — not an encoder or embedding model.
      - **Tool-calling support** (only when the task is tool calling): the model's chat
        template supports tools. Base (non-instruct) models rarely produce a parseable
        call.

      If the model passes the checks that apply, accept it and proceed.

   b. **If the model looks risky** (fails a check, or it can't be determined), warn the
      user and suggest a similar model from the list. If the user insists, allow it —
      state clearly: "This model is not in our verified list and may not be compatible.
      We will proceed with your choice, but if the job fails, consider switching to one
      of the supported models." Then continue normally.

   The model choice affects all downstream parameters — gather them after.

2. **GPUs** — Set `nproc_per_node` from the model size and the method (no estimator call;
   see Compute Requirements). `lora_grpo` on a ≤4B model fits one 80 GB GPU; use 2+ GPUs
   for 8–9B models or for full-weight `grpo`. Stay within the compute budget gathered in
   the workflow (H100 80 GB by default) and briefly explain the choice. Keep
   `prompt_batch_size × group_size` divisible by `4 × nproc_per_node` when you change it.

3. **Training data** — `parent_job_id` (SDG job) or `data_run_id` (uploaded dataset /
   split), as in the workflow's Job Chaining. Ask for it if not already in the
   conversation. Confirm the record count like any other training job and note the
   columns — the reward step matches them against what each reward needs. For
   tool-calling data, every row must be **single-turn** (`question`, `tools`,
   `target_tool_name`, `target_arguments`). If the dataset holds multi-turn `messages`
   traces instead, stop: say multi-turn tool calling isn't supported yet and hand back
   for a single-turn dataset — don't submit it.

4. **Reward** — decide what a "better answer" is, from the task and the training data:
   the reward must match the columns the dataset holds (or SDG can produce), and decides
   whether GRPO fits the task at all.

   a. **Analyse the need.** From the handoff and the conversation, work out what the
      model must produce (a tool call, a short answer, a letter, JSON, text that follows
      rules, ...), how a correct answer can be checked against a reference, and what the
      dataset from step 3 holds (its columns and a few rows).

   b. **Match it against the map.** Read `reward-functions.json` (bundled with this
      skill) — every reward function with what it checks, how it scores, the answer
      format it parses, the row fields it needs, what it fits and its caveats. Shortlist
      the 2–4 entries that genuinely fit the task and whose `row_fields` the dataset has
      (or could have, if the user is willing to regenerate it). Prefer entries with
      `"general_purpose": true`; only shortlist a benchmark-specific one when the user's
      task or data is that benchmark's format. Drop any whose `caveats` rule it out for
      this task (audio, multi-turn, agent trajectories, ...).

   c. **Present the shortlist** with `present_options`, best fit first. For each option
      give the name and one plain line on why it fits and what each row will need. Keep
      the options visually equal: no "(recommended)" / "(default)" label, no bold or
      star on one of them, no "we suggest" lead-in — the order already carries your
      view. Add one short sentence on the trade-off between the top options if it is not
      obvious. Then STOP and wait for the user's pick; never choose silently, even when
      one option clearly fits best.

   d. **If nothing in the map fits** (e.g. open-ended writing with no checkable
      reference), say GRPO can't score it today and offer OSFT on reference responses
      instead.

   e. **If the user picks a reward the dataset can't feed** (it lacks some of that
      entry's `row_fields`), say which columns are missing. You don't generate data
      yourself: hand back with `request_orchestrator_help` — CALL the tool. Set `need`
      to a dataset for the task with exactly those columns (names and value shapes from
      the map), and `context` to the chosen reward and that you resume with the new SDG
      job's dataset. Never rename or invent columns to make a dataset fit.

   Map the pick to the config: built-in entries set `reward_type` to their name
   (`tool_call`, `math`; for `math` also set `text_column` / `label_column`); Gym
   entries set `"reward_type": "gym"` and `"gym_verifier": "<name>"`.

5. **Method** — `lora_grpo` (LoRA adapter, default, recommended) unless the user asks for
   full fine-tuning, then `grpo` (all weights; much more GPU memory — revisit step 2).
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
  "learning_rate": 5e-06,
  "lora_r": 32,
  "lora_alpha": 64,
  "gpu_memory_utilization": 0.4,
  "nproc_per_node": 1,
  "parent_job_id": "<SDG_JOB_ID>",
  "topic": "<1-5 word topic>"
}
```

For the `math` reward, set `"reward_type": "math"`, add `text_column` / `label_column`,
and raise `max_tokens` to ~1024 for step-by-step answers. For a Gym verifier, set
`"reward_type": "gym"` and `"gym_verifier": "<name>"`, set `system_prompt` when its
answer format needs one (see Reward Functions), and size `max_tokens` to its answer
(~1024 when the model reasons before answering; more for long documents or code). For full-weight training, set
`"algorithm": "grpo"`, drop `lora_r` / `lora_alpha`, and lower `learning_rate`.

### Key Parameters — Adjust Based on Model Size, Data, and Compute

Do NOT use static defaults blindly. Adapt each parameter to the user's dataset, model
choice, and GPU count.

| Param | How to set |
|-------|-----------|
| `model_name_or_path` | User's chosen model |
| `algorithm` | `lora_grpo` (default). `grpo` only when the user asks for full fine-tuning |
| `reward_type` | The reward the user picked in step 4: `tool_call`, `math`, or `gym` + `gym_verifier`. See Reward Functions |
| `num_train_epochs` | Passes over the data. 1 for a first run or 2000+ samples; 2–3 for <1000 samples |
| `learning_rate` | 5e-6 for `lora_grpo`; 1e-6 for `grpo`. RL is sensitive to high learning rates |
| `group_size` | Answers sampled per prompt (8). More gives a steadier signal but costs rollout time |
| `prompt_batch_size` | Prompts per step (64). `prompt_batch_size × group_size` must be divisible by `4 × nproc_per_node` |
| `max_prompt_length` | Must fit the longest prompt, **including the tool definitions** for tool calling (they are added to the system prompt) and the options / context for other tasks; longer prompts are **dropped**. 2048 for short prompts or a few small tools; raise it for many or large tool schemas, or long contexts |
| `max_tokens` | Longest answer. 512 for a single tool call or a short answer (thinking models may need 1024); ~1024 for step-by-step reasoning; more for long structured output or code |
| `temperature` | Rollout sampling temperature. 1.0 keeps answers varied enough to learn from |
| `lora_r` / `lora_alpha` | `lora_grpo` only: 32 / 64. Not allowed with `grpo` |
| `n_train` | Cap on training samples (default 5000). Raise it to use a larger dataset fully |
| `nproc_per_node` | From step 2; rollout and training share the GPUs |
| `gpu_memory_utilization` | GPU fraction for the vLLM rollout engine (0.4). Lower it on out-of-memory |
| `language_model_only` | Leave unset: the job sets it automatically for multimodal checkpoints (e.g. Qwen3.5) so vLLM loads only the text model. Set `false` only to override |
| `topic` | **Required.** 1–5 word description of what the model does (e.g. "calendar tool calling", "chemistry multiple choice"). Infer it from the task if not stated |

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

### `gym`

A prompt column plus the chosen verifier's fields under their exact names (`row_fields`
in `reward-functions.json`), e.g. for `mcqa`:

```json
{"question": "Which technique amplifies DNA?\nA: PCR\nB: FISH", "options": [{"A": "PCR"}, {"B": "FISH"}], "expected_answer": "A"}
```

**Only the prompt column reaches the model**; the verifier fields go to the verifier
alone. So the question text must hold everything the model needs to answer — the
lettered options, the constraints to follow, the function signature to implement, the
context to search — and SDG must write it that way.

## Reward Functions

The reward decides what a "better answer" is. Only the rewards below exist; set one via
`reward_type`. `reward-functions.json` holds the full map (built-ins and every Gym
verifier) that step 4 shortlists from.

| `reward_type` | Scores | Data needs |
|---|---|---|
| `tool_call` | 1.0 = right tool and arguments; 0.5 = right tool, wrong arguments; 0 = wrong or no call | Single-turn tool calls (Data Format above); multi-turn not supported |
| `math` | 1.0 = final answer matches the reference (exact or math-equivalent); else 0 | A prompt column and a reference-answer column |
| `gym` | Whatever the chosen `gym_verifier` checks (see below) | A prompt column plus that verifier's fields |

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

**`gym`** — the reward pool for everything else. NVIDIA NeMo Gym verifiers run as a
sidecar next to training (**Kubernetes compute backend only** — validation rejects other
backends) and score each answer; the user picks one in step 4 and it goes in
`gym_verifier`. `reward-functions.json` describes every shipped verifier (multiple choice,
exact/string match, instruction following, long-context retrieval, proof selection,
structured output, ...) with a link to its NeMo Gym docs.
- Each row is the question (`text_column`, default as for `math`) plus the verifier's
  fields under their exact names (`row_fields` in the map), e.g. `mcqa`: `options` =
  `[{"A": "..."}, {"B": "..."}]` and `expected_answer` = `"B"`. Plan SDG to produce those
  columns, with the question text repeating whatever the model must see (Data Format).
- `system_prompt`: only `mcqa` has a default that asks for its answer format. For any
  other verifier whose `answer_format` the bare question would not reveal (a `\boxed{}`
  answer, a last-word label, a JSON shape, ...), set `system_prompt` to ask for exactly
  that format — otherwise every rollout scores 0.
- Starter recipe: **Gym Reward GRPO (LoRA)** (`mcqa` example) — swap `gym_verifier` and the
  data columns for other verifiers.
- Before training, the first rows are checked against the verifier; a field mismatch fails
  the job in seconds with the verifier's error.
- Never put code or a module path (`reward_fn`, etc.) in a job config — validation
  rejects it.

## Compute Requirements

- GRPO needs at least 1 GPU; there is no CPU path
- If a run runs out of memory, lower `gpu_memory_utilization`, `max_tokens`, or
  `prompt_batch_size` (keeping the divisibility rule), or add GPUs
- Checkpoints are saved during training; only the final one is exported

## After Training

The trained model is registered in MLflow's model registry. The user can deploy it via the serve pipeline or download the adapter weights.