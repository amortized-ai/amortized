# SDG Recipes Upgrade Plan

Upgrade the 7 existing SDG recipes in `recipes/sdg/` to match Oumi's quality, and add 3 missing templates. The Oumi originals are at `/Users/shiv/workspace/oumi/configs/examples/synthesis/`.

All recipes use the asynth config format, NOT Oumi's. The key difference is the inference config block — see the format section below.

## Config Format

Every SDG recipe is an amortized job config with `type: sdg`. The `config` block maps to asynth's `SynthesisConfig`.

### Inference config format (asynth, NOT Oumi)

Oumi uses a nested `inference_config` with `model.model_name`, `engine` enum, `generation.*`, `remote_params.*`. We use asynth's flat `LiteLLMInferenceConfig` fields at the top level of `config`:

```yaml
# CORRECT — asynth format (what we use)
type: sdg
config:
  model: anthropic/claude-sonnet-4-20250514    # litellm model string
  num_samples: 100
  temperature: 0.7
  max_tokens: 8192
  max_concurrency: 10
  strategy_params:
    ...

# WRONG — Oumi format (do NOT use)
type: sdg
config:
  inference_config:
    model:
      model_name: claude-sonnet-4-20250514
    engine: ANTHROPIC
    generation:
      max_new_tokens: 8192
      temperature: 0.7
    remote_params:
      num_workers: 50
```

### Field mapping from Oumi to asynth

| Oumi field | asynth field |
|---|---|
| `inference_config.model.model_name` | `model` (prefixed with provider: `anthropic/`, `openai/`) |
| `inference_config.engine` | Not needed — litellm routes by model prefix |
| `inference_config.generation.max_new_tokens` | `max_tokens` |
| `inference_config.generation.temperature` | `temperature` |
| `inference_config.generation.top_p` | `top_p` (optional) |
| `inference_config.remote_params.num_workers` | `max_concurrency` |
| `num_samples` | `num_samples` |

### Default model

Use `openai/gpt-4o-mini` as the default model in all templates. It's cheap, fast, and good enough for SDG. Users will swap in their preferred model.

---

## Upgrades to Existing Recipes

### 1. `recipes/sdg/conversation.yaml`

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/conversation_synth.yaml`

**Current problems:**
- No `model` field
- Only 3 scenarios (Oumi has 6)
- No `customer_type` attribute (Oumi has 7 values with sample_rates)
- No `customer_interaction` attribute (Oumi has 4 values with sample_rates)
- Only 2-turn output (system+user+assistant). Oumi chains 4 generated attributes to build a 4-turn conversation (opener → agent response → followup → final response)
- No postprocessing on generated attributes
- No sample_rates on any attributes

**What to do:**

1. Add `model: openai/gpt-4o-mini` and `max_tokens: 8192`
2. Expand `scenario` to 6 values: account_issue, order_issue, billing_issue, product_issue, technical_issue, refund_request
3. Add `customer_type` sampled attribute with 7 values and weighted sample_rates (concise 0.4, friendly 0.05, frustrated 0.2, confused 0.1, demanding 0.1, curious 0.05, skeptical 0.1)
4. Add `customer_interaction` sampled attribute with 4 values and sample_rates (cooperative 0.6, escalated 0.2, incomplete 0.1, difficult 0.1)
5. Add `system_instruction` sampled attribute with 1 value — the CareBot persona with ACTION block format (CLARIFY, LOOKUP_ORDER, INITIATE_RETURN, SEARCH_PRODUCT, ESCALATE). Copy this from Oumi's config.
6. Chain 4 generated attributes instead of 2:
   - `customer_opener` with postprocessing → `cleaned_opener`
   - `agent_response` (references `{cleaned_opener}`, `{system_instruction}`) with postprocessing → `cleaned_agent_response`
   - `customer_followup` (references `{cleaned_opener}`, `{cleaned_agent_response}`) with postprocessing → `cleaned_followup`
   - `final_agent_response` (references full 4-message context) with postprocessing → `cleaned_final_response`
7. Update the chat transform to 5 messages: SYSTEM + USER + ASSISTANT + USER + ASSISTANT
8. Update passthrough_attributes to include all cleaned attributes plus metadata

Copy the actual prompt text from Oumi's `conversation_synth.yaml` — the instruction_messages content is well-crafted.

### 2. `recipes/sdg/customer-support-classifier.yaml`

**Source:** Our original `new-oumi.yaml` (not from Oumi)

**Current status:** This is already the best recipe. Keep it mostly as-is.

**Minor fixes:**
1. Change `model: openai/gpt-5.4` to `model: openai/gpt-4o-mini` (gpt-5.4 may not be available to all users)
2. Add `keep_original_text_attribute: true` to `postprocessing_params` (preserves raw output alongside cleaned version, useful for debugging)

### 3. `recipes/sdg/data-augmentation.yaml`

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/data_augmentation_synth.yaml`

**Current problems:**
- Doesn't actually augment existing data — generates its own "original_instruction" from scratch
- Missing `input_data` block that loads from an existing JSONL file
- Missing `combination_sampling` for controlling augmentation type × style distributions
- No `phrasing_style` attribute (Oumi has 4 styles)
- No `specificity_level` attribute (Oumi has 3 levels)
- No model field
- No postprocessing

**What to do:**

1. Add `model: openai/gpt-4o-mini`, `max_tokens: 4096`
2. Add `input_data` block that loads from a JSONL file:
   ```yaml
   input_data:
     - path: data.jsonl
       attribute_map:
         instruction: original_instruction
         input: original_input
         output: original_response
   ```
3. Remove the `original_instruction` generated attribute — it should come from input_data
4. Add `phrasing_style` sampled attribute with 4 values (direct, conversational, formal, context_rich) at equal 0.25 rates
5. Add `specificity_level` sampled attribute with 3 values (general, specific, precise)
6. Rename `augmentation_type` values to match Oumi: rephrase (0.3) and related_task (0.7)
7. Add `combination_sampling` with 4 combos (copy from Oumi):
   ```yaml
   combination_sampling:
     - attributes: {augmentation_type: rephrase, phrasing_style: conversational, specificity_level: general}
       sample_rate: 0.15
     - attributes: {augmentation_type: rephrase, phrasing_style: formal, specificity_level: precise}
       sample_rate: 0.15
     - attributes: {augmentation_type: related_task, phrasing_style: direct, specificity_level: specific}
       sample_rate: 0.1
     - attributes: {augmentation_type: related_task, phrasing_style: context_rich, specificity_level: general}
       sample_rate: 0.1
   ```
8. Add postprocessing to both generated attributes (strip_whitespace: true)
9. Add chat transform for output formatting
10. Update passthrough to include original_instruction, original_input, original_response

Copy prompt text from Oumi — the augmentation prompts reference `{augmentation_type}`, `{phrasing_style}`, `{specificity_level}`, and `{original_instruction}`.

### 4. `recipes/sdg/domain-qa.yaml`

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/domain_qa_synth.yaml`

**Current problems:**
- Only 3 specialties (Oumi has 6)
- No `context_type` attribute (Oumi has 5: patient_education, diagnosis_support, treatment_guidance, prevention_advice, parent_guidance)
- No `complexity_level` attribute (Oumi has 3 with sample_rates: basic 0.4, intermediate 0.4, professional 0.2)
- No few-shot `input_examples`
- No postprocessing
- No model field

**What to do:**

1. Add `model: openai/gpt-4o-mini`, `max_tokens: 4096`
2. Expand `specialty` to 6 values: cardiology, dermatology, pediatrics, neurology, orthopedics, endocrinology
3. Add `context_type` sampled attribute with 5 values (copy descriptions from Oumi)
4. Add `complexity_level` sampled attribute with sample_rates (basic 0.4, intermediate 0.4, professional 0.2)
5. Add `input_examples` with 3 seed examples (copy from Oumi — cardiology, dermatology, pediatrics examples with fields: example_specialty, example_specialty_description, example_context_type, example_context_type_description, example_complexity_level, example_complexity_level_description, example_question)
6. Rewrite generated attributes to use few-shot pattern: USER with example → ASSISTANT with example answer → USER with actual attributes
7. Add postprocessing with `cut_prefix`, `cut_suffix`, `strip_whitespace` → `cleaned_question`, `cleaned_answer`
8. Add chat transform: USER (cleaned_question) + ASSISTANT (cleaned_answer)
9. Update passthrough

### 5. `recipes/sdg/instruction-following.yaml`

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/instruction_following_synth.yaml`

**Current problems:**
- 4 domains (Oumi has 6: writing, analysis, coding, math, science, business)
- No few-shot `input_examples`
- No `combination_sampling`
- No postprocessing
- No model field
- `task_format` attribute missing (Oumi has 5: explain, create, analyze, solve, summarize)

**What to do:**

1. Add `model: openai/gpt-4o-mini`, `max_tokens: 4096`
2. Expand `domain` to 6 values: writing, analysis, coding, math, science, business
3. Add `task_format` sampled attribute with 5 values (explain, create, analyze, solve, summarize)
4. Add `input_examples` with 3 seeds (copy from Oumi — creative writing, analysis, programming examples)
5. Rewrite generated attributes to use few-shot pattern
6. Add `combination_sampling` with 3 combos (copy from Oumi):
   ```yaml
   combination_sampling:
     - attributes: {domain: coding, complexity: advanced, task_format: solve}
       sample_rate: 0.15
     - attributes: {domain: science, complexity: intermediate, task_format: explain}
       sample_rate: 0.1
     - attributes: {domain: writing, complexity: basic, task_format: create}
       sample_rate: 0.1
   ```
7. Add postprocessing → `cleaned_instruction`, `cleaned_response`
8. Keep existing chat transform but use cleaned attributes

### 6. `recipes/sdg/multiturn-conversation.yaml`

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/multiturn_conversation_synth.yaml`

**Current problems:**
- Only 3 scenarios (Oumi has 6)
- No `customer_type` attribute (Oumi has 7 values with sample_rates)
- No `customer_interaction` attribute (Oumi has 4 values with sample_rates)
- No `issue_detail` generated attribute
- Minimal role instructions (Oumi has detailed per-role instructions)
- No `output_system_prompt` in multiturn config
- No model field

**What to do:**

1. Add `model: openai/gpt-4o-mini`, `max_tokens: 8192`
2. Expand `scenario` to 6 values (same as conversation.yaml)
3. Add `customer_type` sampled attribute with 7 weighted values
4. Add `customer_interaction` sampled attribute with 4 weighted values
5. Add `issue_detail` generated attribute that creates a specific issue detail for the scenario
6. Expand `role_instruction_messages` with detailed per-role instructions:
   - USER: references `{customer_name}`, `{customer_type}`, `{customer_interaction}`, `{issue_detail}`, `{scenario}`
   - ASSISTANT: full CareBot persona with ACTION block format
7. Add `output_system_prompt` to the multiturn config
8. Add `support_conversation_plan` to passthrough

Copy the detailed role instruction text from Oumi's `multiturn_conversation_synth.yaml`.

### 7. `recipes/sdg/question-answer.yaml`

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/question_answer_synth.yaml`

**Current problems:**
- No few-shot `input_examples`
- No postprocessing
- Different topics than Oumi (science/history/technology vs capitals/physical/countries/climate)
- No model field

**What to do:**

1. Add `model: openai/gpt-4o-mini`, `max_tokens: 1024`
2. Keep the current topics (science, history, technology are more useful than geography-only)
3. Add `input_examples` with 3 seeds — adapt Oumi's pattern but use science/history/technology examples instead of geography
4. Rewrite generated attributes to use few-shot pattern (USER→ASSISTANT→USER)
5. Add postprocessing → `cleaned_question`, `cleaned_answer`
6. Add chat transform: USER (cleaned_question) + ASSISTANT (cleaned_answer)

---

## New Templates to Add

### 8. `recipes/sdg/dynamic-few-shot.yaml` (NEW)

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/dynamic_few_shot_synth.yaml`

**Purpose:** Demonstrate the `num_shots` feature — randomly sample N examples from a pool for each synthesis sample. Useful for generating diverse instruction-following data.

**What to create:**

```yaml
type: sdg
description: Generate diverse tasks using dynamic few-shot sampling from an example pool
config:
  model: openai/gpt-4o-mini
  num_samples: 100
  temperature: 0.9
  max_tokens: 1024
  max_concurrency: 10
  strategy_params:
    input_examples:
      - id: few_shot_examples
        num_shots: 3
        examples:
          - {task_type: summarization, example_input: "Summarize this article about climate change..."}
          - {task_type: translation, example_input: "Translate this paragraph to French..."}
          - {task_type: question_answering, example_input: "Based on the passage, what caused..."}
          - {task_type: sentiment_analysis, example_input: "Determine the sentiment of this review..."}
          - {task_type: extraction, example_input: "Extract all dates and locations from..."}
          - {task_type: rewriting, example_input: "Rewrite this paragraph in a formal tone..."}
    generated_attributes:
      - id: instruction
        instruction_messages:
          - role: user
            content: >-
              Here are example tasks for reference:
              1. [{few_shot_examples[0].task_type}]: {few_shot_examples[0].example_input}
              2. [{few_shot_examples[1].task_type}]: {few_shot_examples[1].example_input}
              3. [{few_shot_examples[2].task_type}]: {few_shot_examples[2].example_input}

              Generate a new, different task instruction inspired by these examples.
              The task should be specific and actionable. Just output the instruction.
      - id: response
        instruction_messages:
          - role: user
            content: "Complete this task:\n\n{instruction}"
    transformed_attributes:
      - id: conversation
        transformation_strategy:
          type: chat
          chat_transform:
            messages:
              - role: user
                content: "{instruction}"
              - role: assistant
                content: "{response}"
    passthrough_attributes:
      - conversation
```

Adapt the bracket notation `{few_shot_examples[0].task_type}` from Oumi. Verify this syntax works with asynth before finalizing.

### 9. `recipes/sdg/tool-use-deterministic.yaml` (NEW)

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/library_tool_use_synth.yaml`

**Purpose:** Generate multi-turn conversations with deterministic (lookup-table) tool use. A library patron interacts with LibBot which has 2 tools backed by a hardcoded book catalog.

**What to create:**

Copy the full structure from Oumi's `library_tool_use_synth.yaml` but adapt:
- Replace inference config with asynth format: `model: openai/gpt-4o-mini`, `temperature: 0.4`, `max_tokens: 1024`, `max_concurrency: 10`
- Keep the entire `environment_config` block as-is (deterministic env with lookup_table, tool definitions, grounding config)
- Keep the `multiturn_attributes` block with `available_environments: [library]`, `conversation_planner`, `role_instruction_messages`, `output_system_prompt`
- Keep the 3 `user_persona` sampled values (anxious_parent, enthusiastic_reader, busy_professional)
- Keep the 8-book lookup table with statuses

Read the full Oumi file at `/Users/shiv/workspace/oumi/configs/examples/synthesis/library_tool_use_synth.yaml` and adapt. The only changes are the inference config format.

### 10. `recipes/sdg/tool-use-synthetic.yaml` (NEW)

**Source:** `/Users/shiv/workspace/oumi/configs/examples/synthesis/mcp_docs_lookup_synth.yaml`

**Purpose:** Generate multi-turn conversations with LLM-simulated tool use. A developer interacts with DocBot which has 3 MCP-style tools (resolve_library_id, query_docs, search_examples) — tool outputs are generated by the LLM.

**What to create:**

Copy the full structure from Oumi's `mcp_docs_lookup_synth.yaml` but adapt:
- Replace inference config with asynth format: `model: openai/gpt-4o-mini`, `temperature: 0.7`, `max_tokens: 2048`, `max_concurrency: 10`
- Keep the entire `environment_config` block as-is (synthetic env with system_prompt, cache_by_input, 3 tool definitions with parameters AND output_schema)
- Keep the `multiturn_attributes` block with `available_environments: [docs_lookup]`, `available_tools`, `max_consecutive_tool_turns`, `conversation_planner`, `role_instruction_messages`
- Keep all 3 sampled attributes: `developer_experience` (4 values with sample_rates), `query_intent` (7 values), `tone` (5 values)
- Keep the `developer_name` generated attribute
- Set `num_samples: 50` (lower than Oumi's 150 for a template default)

Read the full Oumi file at `/Users/shiv/workspace/oumi/configs/examples/synthesis/mcp_docs_lookup_synth.yaml` and adapt.

---

## Verification

After implementing all changes:

1. **Syntax check** — Each YAML file should parse cleanly: `python3 -c "import yaml; yaml.safe_load(open('recipes/sdg/X.yaml'))"`
2. **Schema check** — Each config's `strategy_params` should match asynth's `GeneralSynthesisParams` field names (sampled_attributes, generated_attributes, transformed_attributes, passthrough_attributes, input_data, input_examples, combination_sampling, multiturn_attributes)
3. **Dry run** — If possible, run the classifier recipe end-to-end with `num_samples: 5` to verify the format works: `amortized submit recipes/sdg/customer-support-classifier.yaml --confirm`

## File Summary

| File | Action |
|---|---|
| `recipes/sdg/conversation.yaml` | Upgrade — expand attributes, add chained generation, postprocessing |
| `recipes/sdg/customer-support-classifier.yaml` | Minor fix — change model to gpt-4o-mini |
| `recipes/sdg/data-augmentation.yaml` | Major rewrite — add input_data, combination_sampling, new attributes |
| `recipes/sdg/domain-qa.yaml` | Upgrade — expand attributes, add few-shot, postprocessing |
| `recipes/sdg/instruction-following.yaml` | Upgrade — expand attributes, add few-shot, combination_sampling |
| `recipes/sdg/multiturn-conversation.yaml` | Upgrade — expand attributes, detailed role instructions |
| `recipes/sdg/question-answer.yaml` | Upgrade — add few-shot, postprocessing |
| `recipes/sdg/dynamic-few-shot.yaml` | **New** — dynamic num_shots sampling |
| `recipes/sdg/tool-use-deterministic.yaml` | **New** — library tool-use with lookup table |
| `recipes/sdg/tool-use-synthetic.yaml` | **New** — MCP docs lookup with LLM-simulated tools |

## Reference Files

All Oumi originals to copy prompts/attributes from:
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/conversation_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/multiturn_conversation_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/domain_qa_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/question_answer_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/data_augmentation_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/instruction_following_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/dynamic_few_shot_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/library_tool_use_synth.yaml`
- `/Users/shiv/workspace/oumi/configs/examples/synthesis/mcp_docs_lookup_synth.yaml`
