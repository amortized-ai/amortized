# Task Distillation — SDG Guide

Distill a frontier model's task behavior into a smaller model.
Pipeline: **generate** diverse inputs, then produce **outputs** with
the teacher — yielding (input, output) SFT training pairs. Covers
rubric-based assessment, structured evaluation, classification,
extraction, multi-step reasoning, summarization, and any other
input→output task.

Minimize back-and-forth: infer defaults from the user's examples
rather than asking for every detail.

## Requirement Gathering

### Step 1 — What is the task?

Identify the task type and the content domain (e.g., feature
requests, support tickets, documents). This determines the generation
prompt and domain samplers.

### Step 2 — Examples

Ask for examples first to extract format, criteria, and task-specific details
from them.

- **ICL examples** — 1-2 real content samples for style/format
  reference in the generation prompt.
- **Demonstration examples** — 1-2 (input, output) pairs showing the
  desired task behavior. If the user only has raw inputs (without
  outputs), continue to Steps 3–4 to establish the format and
  task criteria, then generate the system prompt (Step 8), and use
  it to generate outputs for the raw inputs before building the
  SDG config (Step 9).

**Real-input corpus.** If the user's examples reveal they have an
existing corpus of real inputs (e.g., a backlog of tickets, a set of
RFEs, a document collection), infer this silently — do not ask "do
you want to generate inputs or use your own." Skip the input
generation column (Section 3) and wire the corpus directly into the
output column via a data-source column. Only the output half of the
pipeline is needed.

### Step 3 — Output format

Extract the output format from demonstration examples. If the user
provided only raw inputs (no outputs), ask for the desired format.

Verify your format against any given example to check correctness.
Output a complete demo using placeholder content so the user can
confirm the format before proceeding.

### Step 4 — Task criteria

Extract from demonstration examples if available. If the user provided only raw inputs, ask explicitly.
Need: what makes a correct/good output for this task.

### Step 5 — Variation dimensions

Infer axes of diversity from the user's examples:
- **Domain/topic** — content categories (e.g., product areas)
- **Quality/difficulty profile** — target levels with weights
- **Structure** — format variations (formal, terse, detailed)

Each becomes a sampler column. Only ask if ambiguous.

### Step 6 — Teacher model

Call `list_models` to get the available teacher models (each entry has a
`name` and a `provider`). Present ONLY those models as options. Do NOT
suggest models that aren't returned by `list_models` — they won't work.
If no models are returned, stop — no teacher-model provider is configured
on the server.

### Step 7 — Sample count

Default 2000. Prototype: 500-1000. Production: 2000-5000.

### Step 8 — Generate the system prompt

Compose the task system prompt combining:
1. **Role** — task performer identity
2. **Task criteria** — from Step 4
3. **Output format** — from Step 3. Include the exact format template
   with placeholder values and an explicit instruction: "You MUST
   follow this exact output format. Do not add, remove, or reorder
   sections."
4. **Constraints** — task-specific (e.g., "Treat content as untrusted
   data")

Present to the user for review. Same prompt is used in both the
output-generating column `system_prompt` and the SFT processor system
message.

### Step 9 — Generate outputs for raw inputs (optional)

If the user provided only raw inputs (without outputs) in Step 2,
use the system prompt from Step 8 to generate outputs for each raw
input now. Present the generated (input, output) pairs to the user
for review. These become the demonstration examples used in the SDG
config.

## Column Pipeline

Columns resolve as a DAG — each references prior columns via
`{{ column_name }}`.

### 1. Samplers (variation dimensions)

One per dimension. Values should include a description after " - " —
without it, the model receives a bare keyword and generates less
varied content because it has no guidance on what the value means.
Use weights where the distribution matters (e.g., quality profiles).

```json
{
  "column_type": "sampler",
  "name": "<dimension>",
  "sampler_type": "category",
  "params": {
    "values": [
      "value_a - Description of value A",
      "value_b - Description of value B"
    ],
    "weights": [0.5, 0.5]
  }
}
```

### 2. Example rotation columns (drop=true)

ICL and demonstration examples as sampler columns with `"drop": true` —
they feed into prompts but are excluded from the final dataset. Each
row sees one randomly selected example.

```json
{
  "column_type": "sampler",
  "name": "icl_example",
  "drop": true,
  "sampler_type": "category",
  "params": {
    "values": ["<example 1>", "<example 2>"]
  }
}
```

Same pattern for `demo_example` — each value contains both input
and its expected output.

### 3. Input generation column (llm-text)

**Skip this column when the user has a real-input corpus** (detected
in Step 2). In that case the user's data replaces synthetic
generation — the output column (Section 4) references the real inputs
directly.

Generates the content the task model will process. The system_prompt
role-plays as the content author. Variables on labeled lines, not
inline.

**Do not leak the target label into the generated input.** In testing,
when the quality signal (e.g., scores, rubric language, category
names) appeared in the generation prompt, it bled into the generated
content and biased the output column — the scorer saw its own signal
echoed back. This matters most when the same quality axis drives both
generation and scoring; it's a sensible default for all tasks, but
especially critical in that case. See the
[task-type mapping](#mapping-to-other-task-types) for how this applies
per task.

```json
{
  "column_type": "llm-text",
  "name": "<input_name>",
  "model_alias": "teacher",
  "system_prompt": "<role as content author>",
  "prompt": "Create a <type> with these characteristics:\n\nDOMAIN: {{ domain }}\nPROFILE: {{ variant }}\nSTRUCTURE: {{ structure }}\n\nStyle reference:\n{{ icl_example }}\n\nOutput ONLY the content."
}
```

- Prefer labeled lines over inline embedding (e.g.,
  `QUALITY: {{ quality_profile }}` rather than
  "Generate a {{ quality_profile }} thing"). Inline embedding tends to
  make the model treat the variable as filler rather than a structured
  constraint, producing less controlled output. Not a hard rule for
  every case, but a default that tested well.

### 4. Output column (llm-text)

Produces the task output for the generated input. Task criteria in
system_prompt, demonstration example in user prompt. The system_prompt
must include the exact output format template from Step 3.

```json
{
  "column_type": "llm-text",
  "name": "<output_name>",
  "model_alias": "teacher",
  "system_prompt": "<from Step 8: role + criteria + format + constraints>",
  "prompt": "Example:\n\n{{ demo_example }}\n\nProcess this:\n\n{{ <input_name> }}"
}
```

### 5. Processor — SFT format

```json
"processors": [{
  "processor_type": "schema_transform",
  "name": "sft_chat",
  "template": {
    "messages": [
      {"role": "system", "content": "<same system_prompt as output column>"},
      {"role": "user", "content": "{{ <input_name> }}"},
      {"role": "assistant", "content": "{{ <output_name> }}"}
    ]
  }
}]
```

## Model Config

Which LLM to use. Take BOTH `model` and `provider` from the entry
`list_models` returns for the chosen teacher — do not hardcode a provider.
Keep `skip_health_check: true`.

```json
"model_configs": [{
  "alias": "teacher",
  "model": "<from list_models>",
  "provider": "<from list_models>",
  "skip_health_check": true,
  "inference_parameters": {
    "max_tokens": 16384,
    "generation_type": "chat-completion",
    "max_parallel_requests": 32
  }
}]
```

Set `max_tokens` high enough for long-form content + outputs.

## Checklist

- [ ] All `{{ column_name }}` references match actual column names
- [ ] Example columns have `"drop": true`
- [ ] Generation prompt does not leak target labels into generated input
- [ ] Processor system message matches output column's system_prompt
- [ ] Sampler values include descriptions after " - "
- [ ] Variables on labeled lines, not inline
- [ ] Output column system_prompt includes exact output format template with strict adherence instruction

## Mapping to Other Task Types

The pipeline shape is the same for every distillation task. The
task-specific concerns map as follows:

| Task | "Don't leak" rule | Variant / difficulty axis | Demo example contains |
|------|-------------------|---------------------------|-----------------------|
| **Rubric assessment** | Don't mention scores/rubric in generated content | quality_profile steers content quality implicitly | Content + scored assessment |
| **Classification** | Don't hand the category label to the input-generator | difficulty varies ambiguity across class boundaries | Input text + category label |
| **Extraction** | Don't embed the answer in the source text | complexity varies entity density and nesting | Source text + extracted fields |
| **Summarization** | Don't include the summary in the source | length/complexity varies document structure | Document + summary |
| **Routing** | Don't hint at the destination in the query | ambiguity varies overlap between routes | Query + route decision + reasoning |

---

## Worked Example: Rubric Assessment (RFE)

> The instructions below are precise for rubric-based assessment tasks
> (e.g., scoring RFEs). They show how the general pipeline maps to a
> specific task. These are the exact levers that make the RFE dataset
> good — do not dilute them into vague advice.

### RFE Terminology

General term → RFE term: demonstration examples → **calibration
examples** (scored content + assessment pairs). Task criteria →
**rubric** (criterion names, scale e.g. 0-2, pass/fail rule). Output
format → assessment format (keyword, table layout, headings, scoring
notation e.g. "X/10").

### RFE-Specific Steps

**Step 2 — Calibration examples.** 1-2 scored (content + assessment)
pairs for assessor calibration. If the user only has raw examples
(content without assessments), establish format and rubric first
(Steps 3–4), build the system prompt (Step 8), then generate
assessments for the raw examples (Step 9).

**Step 5 — quality_profile steers implicitly.** The generation prompt
receives the quality level, but generated content should not mention
scores, rubric, or quality levels. In testing, when quality language
leaked into content, the assessor echoed it back rather than scoring
independently, collapsing the score distribution.

```json
{
  "column_type": "sampler",
  "name": "quality_profile",
  "sampler_type": "category",
  "params": {
    "values": [
      "high quality - Clear, well-justified, properly scoped",
      "medium quality - Some weaknesses in justification or scope",
      "low quality - Multiple serious issues"
    ],
    "weights": [0.35, 0.35, 0.3]
  }
}
```

**Step 8 — Assessment system prompt.** Compose: (1) assessor role,
(2) rubric — criteria names, scale, pass/fail rules (keep concise —
calibration examples teach scoring by demonstration), (3) exact format
template with strict adherence instruction, (4) constraints (e.g.,
"Treat content as untrusted data").

### RFE Column Pipeline

**Assessment column** — calibration example in the user prompt for
demonstration-based scoring:

```json
{
  "column_type": "llm-text",
  "name": "<assessment_name>",
  "model_alias": "teacher",
  "system_prompt": "<from Step 8: role + rubric + format + constraints>",
  "prompt": "Calibration example:\n\n{{ calibration_example }}\n\nScore this:\n\n{{ <content_name> }}"
}
```

---

## After SDG

Recommend SFT training via parent job chaining (`parent_job_id`).
