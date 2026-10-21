### Step 2 — Choose the model source
Call `signal_phase` this response with the step for the phase you are in — this phase: `step-2-model-source`.

The eval job serves the model itself: when the config names a model
(`training_job_id` for a tuned model, or `model_name_or_path` for an HF
id), the job pod starts vLLM on a GPU, waits for it to be healthy, runs
the eval, and exits when scores are done — no separate serving step,
nothing to poll, no endpoint URLs involved.

**If Step 0 already settled the subject, do NOT re-ask the model source.**
When the user picked a training job or named their fine-tuned model, the
source is resolved: set `training_job_id` (or `model_name_or_path` for a
named HF model) and go straight to Step 3 (`step-3-metrics`). If they named
a gateway model (no endpoint URL), resolve it WITHOUT re-asking which model:
call `get_eval_endpoint_suggestions`, match the name against
`known_endpoints`, and set the eval's `endpoint` (base_url + model) — then
go to Step 3. Only present the options below when the subject is still open
— the user said "evaluate a model" without naming one, or wants a
non-trained endpoint.

Present the model-source options as clickable choices:

1. **The trained model** — when the user picked a training job, the
   default: the eval config gets `training_job_id` set. Tell the user
   the model will be served inside the eval job (loading takes a few
   minutes; the job's monitor card shows the progress).
2. **A model by name** — any HF hub id (e.g. Qwen/Qwen3.5-4B): the
   config gets `model_name_or_path` set.
3. **A gateway model** — models served through the platform gateway
   (from `get_eval_endpoint_suggestions` → `known_endpoints`), e.g.
   "gpt-oss (openai/gpt-oss-120b) via gateway" — selecting one fills in
   the eval's `endpoint` (base_url + model). Gateway evals need no GPU.
4. "I'll give the endpoint URL" — an external OpenAI-compatible
   endpoint: ask for the URL and the model name it serves.

When the subject is still open, do NOT pre-decide — show the options and
let the user pick. (If Step 0 already resolved the model, you skipped
this per the note above.) Never mention endpoint URLs or utilization
numbers for models the eval serves itself; that is internal plumbing.

Options 1 and 2 use the same GPU budget scheme as training jobs: the
eval pod requests GPUs and the user's namespace quota bounds it. No
GPU availability check is needed before creating the job — if the
quota is exhausted the job stays queued/fails at scheduling, and you
can relay that error to the user when it happens.
