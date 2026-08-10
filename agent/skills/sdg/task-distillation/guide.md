# Task Distillation -- SDG Workflow

Generate scored assessment training data for task distillation: training
a smaller model to replicate a frontier model's scoring, classification,
or structured assessment behavior (e.g., RFE quality assessors, code
reviewers, compliance checkers).

Two-phase generation: Phase 1 generates synthetic inputs guided by
sampled attributes, Phase 2 scores them using a rubric. The teacher
model's assessment is the training signal the student learns to
replicate.

**Keep it brief.** Do your analysis silently. Present results and ask
for confirmation -- do not narrate your reasoning.

## Workflow

Ask the user these questions one at a time using `present_options`.

### Step 1 -- Task description

What is the model scoring? What input type does it process?

Collect:
- Input type (e.g., Jira issues, RFEs, support tickets, code reviews)
- What the model produces (e.g., rubric score + verdict + feedback)
- Output format preference (table + verdict + feedback is the default)

### Step 2 -- Scoring criteria

What dimensions does the rubric evaluate?

Collect for each criterion:
- Name (e.g., "Customer Problem", "Right-sized", "Technical Feasibility")
- Scale (e.g., 0-2 where 0=missing, 1=partial, 2=strong)
- What each score level looks like (calibration anchors)

Also collect:
- Pass/fail threshold (e.g., "total >= 7 AND no zeros")
- Total possible score

### Step 3 -- Sample data

Ask the user for real examples. Minimum requirements:
- 3 input examples at different quality levels (Phase 1 style reference)
- 3 input + assessment pairs at different quality levels (Phase 2
  calibration)

More examples = better variety. 9 examples (3 per quality tier) is
ideal for both pools.

Check that calibration examples are style-consistent: if PASS feedback
uses prose and FAIL uses numbered fixes, all calibration ICLs should
follow this pattern. Mixed styles cause the teacher to default to the
majority pattern.

### Step 4 -- Domain areas

What subject areas should generated inputs cover? These become the
`domain` sampler values.

Populate based on the real-world distribution of inputs the model will
see at inference. Equal weights unless the user specifies otherwise.

### Step 5 -- Teacher model

Call `list_models` to get models from the AI Gateway. Present ONLY
those models. If none returned, direct user to Settings -> AI Gateway.

### Step 6 -- Sample count

Present three tiers:
1. 1000 samples -- Quick iteration, prototype
2. 2000 samples -- Recommended
3. 4000 samples -- Comprehensive coverage

### Step 7 -- Rubric and confirmation

Build the scoring rubric system prompt from Steps 2 + 3. It must
include:
- Scoring criteria with explicit scale definitions (what 0, 1, 2 means)
- Calibration examples for each criterion showing score levels
- Pass/fail logic
- Output format template (table, verdict, feedback sections)
- Feedback style guidance (prose for PASS, numbered fixes for FAIL)

Present a summary table of all parameters + the constructed rubric
for user review before submitting.

## Reference Payload

Use this as the base for `create_sdg_job()`. Replace `[PLACEHOLDERS]`
with values from the workflow. Set `model` to the user's choice from
`list_models`.

The payload has two LLM generation phases evaluated in column order:
`generated_input` creates synthetic inputs, then `assessment` scores
them.

```json
{
  "num_records": 2000,
  "topic": "[TASK_NAME] assessment",
  "columns": [
    {
      "column_type": "sampler",
      "name": "domain",
      "sampler_type": "category",
      "params": {
        "values": [
          "Domain Area 1 - Description of the first domain area",
          "Domain Area 2 - Description of the second domain area",
          "Domain Area 3 - Description of the third domain area"
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
          "--- Example Input ---\n[CALIBRATION_INPUT_1: real input from user data]\n\n--- Assessment ---\n[CALIBRATION_ASSESSMENT_1: the scored assessment for this input]",
          "--- Example Input ---\n[CALIBRATION_INPUT_2]\n\n--- Assessment ---\n[CALIBRATION_ASSESSMENT_2]",
          "--- Example Input ---\n[CALIBRATION_INPUT_3]\n\n--- Assessment ---\n[CALIBRATION_ASSESSMENT_3]"
        ]
      }
    },
    {
      "column_type": "llm-text",
      "name": "generated_input",
      "model_alias": "text",
      "system_prompt": "You are a [ROLE] writing realistic [INPUT_TYPE] for [DOMAIN]. Create examples that match the specified quality level and structure.",
      "prompt": "Create a [INPUT_TYPE] with these characteristics:\n\nDomain: {{ domain }}\nQuality: {{ quality_profile }}\nStructure: {{ input_structure }}\n\nUse these examples for style and format reference:\n\n--- Example 1 ---\n[ICL_EXAMPLE_1: real input from user data]\n\n--- Example 2 ---\n[ICL_EXAMPLE_2]\n\n--- Example 3 ---\n[ICL_EXAMPLE_3]\n\n---\n\nWrite a new, different [INPUT_TYPE]. Output ONLY the [INPUT_TYPE] text."
    },
    {
      "column_type": "llm-text",
      "name": "assessment",
      "model_alias": "text",
      "system_prompt": "[SCORING_RUBRIC: built in Step 7. Include criteria with scale definitions, calibration examples per criterion, pass/fail logic, output format, feedback style guidance. This prompt appears identically in the training data.]",
      "prompt": "For calibration, here is a scored example:\n\n{{ calibration_example }}\n\n---\n\nScore the following input. [GUARDRAILS: e.g. 'Your table MUST include all N criteria. If it passes (7+ with no zeros), write Feedback as flowing prose, not numbered lists.']\n\n{{ generated_input }}"
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
          {"role": "system", "content": "[SCORING_RUBRIC: must be identical to the assessment column system_prompt]"},
          {"role": "user", "content": "{{ generated_input }}"},
          {"role": "assistant", "content": "{{ assessment }}"}
        ]
      }
    }
  ]
}
```

### Adapting the Payload

- **`num_records`**: from Step 6 tier choice
- **`columns[0].params.values`**: domain areas from Step 4
- **`columns[3].params.values`**: calibration input+assessment pairs from Step 3.
  Each value is a complete calibration example string. The sampler cycles
  through them so different samples see different calibration references
- **`generated_input` column prompt**: embed the Phase 1 ICL examples (input-only
  examples from Step 3) directly in the prompt text
- **`assessment` column system_prompt`**: the scoring rubric built in Step 7
- **`assessment` column prompt**: add guardrails right before `{{ generated_input }}`
  (e.g., "Your table MUST include all N criteria")
- **`sft_format` processor `messages[0].content`**: must be identical to the
  `assessment` column `system_prompt` -- any mismatch creates
  training/inference divergence
- **`model_configs[0].model`**: the model chosen in Step 5
- **`max_tokens`**: 32768 prevents assessment truncation. At 16384, expect
  ~10% truncation rate for long rubrics

### Quality Profile

Do NOT include target scores in quality profile descriptions.
"High quality" must describe characteristics ("should score well across
most criteria"), not targets ("Target: 8-10/10"). Score leakage into
generated inputs biases the scorer.

### ICL Guidelines

- Source all examples from real user-provided data
- Ensure diversity across score ranges (high/medium/low quality inputs)
- Phase 1 ICLs: 3 input-only examples embedded in the `generated_input`
  prompt. Show the teacher what realistic inputs look like
- Phase 2 calibration: input+assessment pairs in the `calibration_example`
  sampler. Show the teacher how to score. Each sample sees one example,
  cycling through the pool
- Style consistency within the pool matters. If PASS feedback is prose
  and FAIL feedback is numbered fixes, select ICLs that demonstrate this
  pattern

### Output Format Constraints

Place these in the `assessment` prompt as guardrails:
- "Your table MUST include all N criteria: [list them]"
- "Follow this format exactly. Do not bold the TITLE label or scores."
- "Write scores as plain text (e.g. 2/2, 1/2, 0/2), not bold."

Without these, the teacher model drops criteria (~8%), bolds scores,
and uses inconsistent formatting.

## After SDG -- Training

Recommend OSFT training. Read `skills/training/task-distillation/osft/guide.md`.
The SDG job's output becomes the training job's data via parent job chaining.
