# Task Distillation -- SDG Workflow

Generate training data for task distillation: training a small
task-specific model to handle a specialized task that is not covered
by other skills (knowledge-ingestion, question answering). Examples
include RFE quality assessment, Jira ticket bucketing, code review,
policy compliance checking, ticket routing, structured data extraction,
content moderation with custom rules.

**Keep it brief.** Do your analysis silently. Present results and ask
for confirmation -- do not narrate your reasoning.

## Discovery

Users arrive with varying levels of clarity -- from "I want a model
that scores RFEs" to "I need something that helps with our tickets."
The goal is to refine the user's intent into a concrete Data Designer
generation pipeline.

Work through these dimensions conversationally. Not all apply to every
task -- let the user's examples guide which questions matter.

**What goes in?** The input type and its structure. Ask for real
examples early -- they reveal format, length, variation, and implicit
rules better than descriptions ever do.

**What comes out?** The expected output. This ranges from a single
label (routing, classification) to structured multi-section output
(rubric assessment with scores, verdict, feedback). Get concrete
output examples for different inputs to understand the full range.

**What determines correctness?** The criteria, rules, or logic behind
a correct output. For classification: category definitions and
boundaries. For assessment: a scoring rubric with scales. For
extraction: a schema and edge-case handling. Users often have implicit
rules here -- the examples they provide are the best way to surface
these.

**What variety exists in the inputs?** Dimensions that affect the
task: domains/topics, quality levels, formatting styles, difficulty
tiers. These become sampler columns that control generation diversity
and ensure the training distribution covers what the model will see
at inference.

**Does the user have input data?** If yes (e.g., a backlog of real
tickets), the pipeline only needs to generate the task output. If no,
the pipeline also generates synthetic inputs as a first phase.

### Teacher model

Call `list_models` to get models from the AI Gateway. Present ONLY
those models.

### Sample count

Default tiers:
1. 1000 samples -- Quick iteration, prototype
2. 2000 samples -- Recommended
3. 4000 samples -- Comprehensive coverage

## Payload Construction

Each task distillation payload follows a common Data Designer pattern:

1. **Sampler columns** control diversity across dimensions relevant to
   the task (domain, difficulty, input style, quality tier, etc.)
2. **LLM generation columns** are where the teacher model generates
   content. Columns evaluate in order and can reference prior columns
   via `{{ column_name }}`
3. **Schema transform processor** reshapes generated columns into the
   `messages` SFT format for training

The specific columns, prompts, and structure depend entirely on the
task. A single-turn classifier needs one generation column. A
multi-step assessor might need two (generate input, then score it).
An extractor might need one column with a complex system prompt
defining the schema.

### System Prompt Consistency

When the task has a defining system prompt (a rubric, classification
rules, extraction schema), it must appear identically in two places:
1. The LLM generation column's `system_prompt` (used at SDG time)
2. The `schema_transform` processor's `messages[0].content` (embedded
   in training data, seen by the student at inference)

Mismatch between these creates training/inference divergence.

## Worked Example: RFE Quality Assessor

This shows how one task distillation use case was built end-to-end:
from initial user description, through requirement discovery, to final
payload. Use it as a reference for the reasoning process -- adapt the
pattern to the user's actual task.

### User's Starting Point

"I want a model that scores RFE (Request for Enhancement) submissions
for quality. It should evaluate whether an RFE clearly states the
customer problem, is right-sized, and is technically feasible. Output
a score table with pass/fail verdict and feedback."

### What Discovery Revealed

From the description + example RFEs + sample scored outputs the user
provided:

**Input**: RFE documents (Jira-like format with title, description,
acceptance criteria sections). The user had no large corpus of
pre-scored RFEs, so the pipeline needed to generate both synthetic
inputs AND assessments (two-phase generation).

**Output**: Score table (5 criteria, each 0-2) + verdict (PASS/FAIL
based on total >= 7 AND no zeros) + feedback section. Examining the
user's sample outputs revealed that PASS feedback was flowing prose
while FAIL feedback used numbered action items -- this style
difference was preserved intentionally in the rubric.

**Scoring criteria** (extracted from user's examples and descriptions):
Customer Problem, Right-sized, Technical Feasibility, Acceptance
Criteria, Business Value. Each scored 0 (missing), 1 (partial),
2 (strong).

**Input diversity dimensions** identified from the user's real RFEs:
- Domain: 3 technology areas the user's RFEs span
- Quality profile: high/medium/low (35/35/30 split)
- Input structure: formal, terse, detailed

### Payload

```json
{
  "num_records": 2000,
  "topic": "RFE quality assessment",
  "columns": [
    {
      "column_type": "sampler",
      "name": "domain",
      "sampler_type": "category",
      "params": {
        "values": [
          "Cloud Infrastructure - Kubernetes, OpenShift, containerization, cluster management",
          "AI/ML Platform - Model training, inference, MLOps, GPU orchestration",
          "Developer Tools - IDEs, CI/CD, source control, debugging, observability"
        ],
        "weights": [0.34, 0.33, 0.33]
      }
    },
    {
      "column_type": "sampler",
      "name": "quality_profile",
      "sampler_type": "category",
      "params": {
        "values": [
          "High quality - An input that should score well across most criteria",
          "Medium quality - An input with significant weaknesses in 1-2 criteria",
          "Low quality - An input with multiple serious issues across criteria"
        ],
        "weights": [0.35, 0.35, 0.30]
      }
    },
    {
      "column_type": "sampler",
      "name": "input_structure",
      "sampler_type": "category",
      "params": {
        "values": [
          "Formal - Well-organized with clear headers and substantive content in each section",
          "Terse - Brief format, may be missing key sections, gets to the point quickly",
          "Detailed - Extensive format with many sections, may be verbose"
        ],
        "weights": [0.34, 0.33, 0.33]
      }
    },
    {
      "column_type": "sampler",
      "name": "calibration_example",
      "sampler_type": "category",
      "params": {
        "values": [
          "--- Example Input ---\n[A real high-quality RFE from user data]\n\n--- Assessment ---\n[User's scored assessment: PASS with prose feedback]",
          "--- Example Input ---\n[A real medium-quality RFE]\n\n--- Assessment ---\n[User's scored assessment: FAIL with numbered fixes]",
          "--- Example Input ---\n[A real low-quality RFE]\n\n--- Assessment ---\n[User's scored assessment: FAIL with critical issues]"
        ]
      }
    },
    {
      "column_type": "llm-text",
      "name": "generated_input",
      "model_alias": "text",
      "system_prompt": "You are a product manager writing RFE (Request for Enhancement) submissions for enterprise software.",
      "prompt": "Create an RFE with these characteristics:\n\nDomain: {{ domain }}\nQuality: {{ quality_profile }}\nStructure: {{ input_structure }}\n\nUse these examples for style and format reference:\n\n--- Example 1 ---\n[Real RFE from user data]\n\n--- Example 2 ---\n[Another real RFE at a different quality level]\n\n--- Example 3 ---\n[A third real RFE]\n\n---\n\nWrite a new, different RFE. Output ONLY the RFE text."
    },
    {
      "column_type": "llm-text",
      "name": "assessment",
      "model_alias": "text",
      "system_prompt": "[THE SCORING RUBRIC: 5 criteria with 0/1/2 anchors, calibration examples per criterion, pass/fail logic (>= 7 and no zeros), output format (table then verdict then feedback), feedback style (prose for PASS, numbered fixes for FAIL)]",
      "prompt": "For calibration, here is a scored example:\n\n{{ calibration_example }}\n\n---\n\nScore the following RFE. Your table MUST include all 5 criteria: Customer Problem, Right-sized, Technical Feasibility, Acceptance Criteria, Business Value. If it passes (7+ with no zeros), write Feedback as flowing prose.\n\n{{ generated_input }}"
    }
  ],
  "model_configs": [
    {
      "alias": "text",
      "model": "<from-list_models>",
      "provider": "gateway",
      "skip_health_check": true,
      "inference_parameters": {
        "temperature": 0.7,
        "max_tokens": 32768,
        "max_parallel_requests": 32
      }
    }
  ],
  "processors": [
    {
      "processor_type": "schema_transform",
      "name": "sft_format",
      "template": {
        "messages": [
          {"role": "system", "content": "[SCORING RUBRIC: identical to assessment column system_prompt]"},
          {"role": "user", "content": "{{ generated_input }}"},
          {"role": "assistant", "content": "{{ assessment }}"}
        ]
      }
    }
  ]
}
```

### Why This Payload Is Shaped This Way

**Two generation columns** (`generated_input` then `assessment`):
the user had no corpus of pre-scored RFEs, so both the inputs and the
scoring had to be synthesized. If the user had real input data, only
the assessment column would be needed.

**`quality_profile` sampler with 35/35/30 split**: without explicit
quality control, the teacher generates mostly high-quality inputs and
the model never learns to score weak ones. Do NOT include target
scores in descriptions (e.g., "Target: 8-10/10") -- this leaked into
generated inputs and biased the scorer in our experiments.

**`calibration_example` sampler**: cycles through scored examples so
each sample sees a different calibration reference. Sourced from the
user's real scored data. All examples must be style-consistent -- if
PASS uses prose feedback, all PASS calibration examples must too.
Mixed styles cause the teacher to default to the majority pattern.

**ICL examples in `generated_input` prompt**: 3 real input examples
from user data embedded directly. These show the teacher what
realistic inputs look like for style and format reference.

**Guardrails in `assessment` prompt**: placed right before
`{{ generated_input }}`, not only in the system prompt. Without
explicit guardrails ("Your table MUST include all 5 criteria"), the
teacher drops criteria (~8% of samples), bolds scores, and uses
inconsistent formatting.

**`max_tokens: 32768`**: assessments with rubric tables + feedback are
long. At 16384, ~10% of samples get truncated (missing Feedback
section). Filter truncated samples in post-processing if any slip
through.

## Lessons from Experiments

Non-obvious findings from running task distillation pipelines:

- **Don't put the rubric only in the user message.** Keep it in the
  system prompt. Moving the rubric to the user message creates a
  mismatch between SDG context and training context. This hurt
  downstream eval.

- **Feedback style is score-dependent.** In real scored data, PASS
  feedback is typically prose and FAIL feedback is structured/numbered.
  The teacher naturally defaults to structured for everything. Explicit
  style guidance in the rubric + style-consistent calibration examples
  are both needed to get the right behavior.

- **ICL round-robin is deterministic.** The `calibration_example`
  sampler cycles: sample 0 always sees example 0, sample 1 sees
  example 1, etc. Acceptable for training data diversity but be aware.

- **`messages` column name.** Training expects exactly `messages`. Any
  other column name fails silently.

## After SDG -- Training

Recommend OSFT training. Read `skills/training/task-distillation/osft/guide.md`.
The SDG job's output becomes the training job's data via parent job chaining.
