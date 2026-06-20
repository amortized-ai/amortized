# End-to-End Examples Upgrade Plan

Upgrade the 2 existing projects in `examples/` and add new ones. Each project bundles `synth.yaml` + `train.yaml` + `eval.yaml` + `README.md` — a complete pipeline from data generation to trained, evaluated task model.

Reference: Oumi projects at `/Users/shiv/workspace/oumi/configs/projects/`, Oumi prompt library at https://docs.oumi.ai/reference/prompts, Oumi quickstart (Banking77 classifier) at https://docs.oumi.ai/guides/quickstart.

---

## Current State

### `examples/ticket-classifier/`

| File | Status | Issues |
|---|---|---|
| `synth.yaml` | Good | `model: openai/gpt-5.4` should be `openai/gpt-4o-mini`. Otherwise solid — urgency × topic, postprocessing, chat transform |
| `train.yaml` | Bare minimum | 10 lines. Missing: model_name_or_path, data_path, lora_target_modules, lr_scheduler, warmup, weight_decay, batch_size, gradient_checkpointing, logging_steps, save_steps, output_dir |
| `eval.yaml` | Too vague | Generic LLM judge prompt. `max_samples: 10` too few. `model: openai/gpt-5.4`. No structured scoring, no per-class breakdown, no eval dataset specified |
| `README.md` | Good structure | Pipeline steps are clear. Has expected results. Missing: prereqs, time estimates, GPU requirements, how to interpret results |

### `examples/entity-extractor/`

| File | Status | Issues |
|---|---|---|
| `synth.yaml` | Good | Same model issue. Good entity_type × text_type design with JSON extraction |
| `train.yaml` | Bare minimum | Same issues as ticket-classifier |
| `eval.yaml` | Too vague | Judge prompt just asks "are the extractions correct?" No precision/recall, no partial match scoring, no structured eval |
| `README.md` | Incomplete | No expected results section, no prereqs |

---

## Upgrades to Existing Projects

### 1. `examples/ticket-classifier/`

#### 1a. `synth.yaml` — minor fix

Change `model: openai/gpt-5.4` to `model: openai/gpt-4o-mini`. Keep everything else — this config is already good.

#### 1b. `train.yaml` — rewrite

Replace with a complete config that matches the upgraded `recipes/training/lora-sft.yaml` template:

```yaml
type: training
description: LoRA SFT for ticket classification — Qwen 2.5 1.5B
config:
  algorithm: sft
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct

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
  output_dir: output/ticket-classifier

compute:
  gpus: 1
  min_vram_gb: 8
```

#### 1c. `eval.yaml` — rewrite

Replace with a structured eval config that does both deterministic matching AND LLM-as-judge:

```yaml
type: eval
description: Evaluate ticket classifier — deterministic accuracy + LLM judge
config:
  # Test data
  test_data_path: test_data.jsonl
  max_samples: 50

  # Deterministic scoring — exact match on structured output
  deterministic_checks:
    - field: urgency
      type: exact_match
      expected_from: label
    - field: topic
      type: exact_match
      expected_from: label

  # LLM-as-judge — assess overall quality
  judge:
    model: openai/gpt-4o-mini
    temperature: 0.0
    template: generic/instruction_following
    prompt: >-
      A customer support ticket was classified by urgency and topic.
      Evaluate whether the classification is reasonable given the ticket content.

      Ticket: {request}

      Expected classification:
      Urgency: {expected_urgency}
      Topic: {expected_topic}

      Model's classification:
      {response}

      Consider:
      1. Is the urgency level appropriate for the customer's situation?
      2. Is the topic correctly identified?
      3. Would a human support agent agree with this classification?

  # Reporting
  metrics:
    - overall_accuracy
    - per_class_accuracy
    - confusion_matrix
    - judge_pass_rate
```

#### 1d. `README.md` — expand

```markdown
# Ticket Classifier

Fine-tune a small model to classify customer support tickets by urgency and topic,
replacing expensive frontier model calls.

## What You'll Build

A task model that takes a customer support ticket and outputs:
- **Urgency**: low, medium, high, critical
- **Topic**: orders, shipping, returns, payments, product_questions, account_issues

## Prerequisites

- Amortized server running (`amortized up`)
- Compute backend configured (`amortized config`)
- API key for an LLM provider (OpenAI, Anthropic, etc.) set as env var
- ~30 minutes total (5 min synth, 15 min training, 5 min eval)

## Pipeline

### Step 1: Generate training data (100 labeled tickets)

```bash
amortized submit examples/ticket-classifier/synth.yaml --confirm
```

This generates 100 realistic customer support tickets with controlled
urgency/topic distributions, formatted as SFT training conversations.

### Step 2: Fine-tune with LoRA SFT

```bash
amortized submit examples/ticket-classifier/train.yaml \
  --set config.data_path=<sdg-output-path> --confirm
```

Trains a Qwen 2.5 1.5B model with LoRA. Takes ~15 minutes on a single GPU.

### Step 3: Serve the fine-tuned model

```bash
amortized submit recipes/serve/adapter.yaml \
  --set config.model=Qwen/Qwen2.5-1.5B-Instruct \
  --set config.adapter=<training-output-path> --confirm
```

### Step 4: Evaluate

```bash
amortized submit examples/ticket-classifier/eval.yaml \
  --set config.test_data_path=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```

## Expected Results

| Metric | Base Model | Fine-tuned |
|--------|-----------|------------|
| Urgency accuracy | ~60% | ~90%+ |
| Topic accuracy | ~80% | ~95%+ |
| Judge pass rate | ~70% | ~95%+ |

## Customization

- **More data**: Increase `num_samples` in `synth.yaml` to 500-1000 for better quality
- **Different model**: Change `model_name_or_path` in `train.yaml` (try `Qwen/Qwen3-4B` for higher accuracy)
- **Different categories**: Edit the `possible_values` in `synth.yaml` to match your ticket taxonomy
- **Harder task**: Add more sampled attributes (e.g., language, department, priority)

## GPU Requirements

| Stage | GPU | VRAM | Time |
|-------|-----|------|------|
| Synth | None (API calls) | 0 | ~5 min |
| Training | 1x GPU | 8 GB+ | ~15 min |
| Serving | 1x GPU | 4 GB+ | — |
| Eval | None (API calls) | 0 | ~2 min |
```

### 2. `examples/entity-extractor/`

#### 2a. `synth.yaml` — minor fix

Change `model: openai/gpt-5.4` to `model: openai/gpt-4o-mini`.

Also improve the extraction prompt — the current prompt asks the LLM to "Return a JSON array" but doesn't use postprocessing to clean the output. Add postprocessing:

```yaml
      - id: extraction
        instruction_messages:
          - role: user
            content: >-
              Extract all {entity_type} entities from this text. Return ONLY a JSON
              array of objects with "entity" and "type" fields.

              Entities:
              <json>
              Text:
              {source_text}
        postprocessing_params:
          id: extraction_clean
          cut_prefix: "Entities:"
          strip_whitespace: true
```

#### 2b. `train.yaml` — rewrite

Same pattern as ticket-classifier but with `max_length: 4096` (extraction inputs are longer):

```yaml
type: training
description: LoRA SFT for entity extraction — Qwen 2.5 1.5B
config:
  algorithm: sft
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct

  # Data
  data_path: training_data.jsonl
  max_length: 4096

  # Training
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
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

  # Memory
  gradient_checkpointing: true

  # Logging
  logging_steps: 10
  save_steps: 500
  report_to: none

  # Output
  output_dir: output/entity-extractor

compute:
  gpus: 1
  min_vram_gb: 8
```

#### 2c. `eval.yaml` — rewrite

Entity extraction needs precision/recall evaluation, not just "is it correct":

```yaml
type: eval
description: Evaluate entity extraction — precision, recall, and LLM judge
config:
  test_data_path: test_data.jsonl
  max_samples: 50

  # Deterministic scoring — JSON array comparison
  deterministic_checks:
    - field: entities
      type: json_array_match
      expected_from: label
      scoring: f1

  # LLM-as-judge — assess extraction quality
  judge:
    model: openai/gpt-4o-mini
    temperature: 0.0
    prompt: >-
      An entity extraction task was performed on the following text.

      Source text: {request}

      Expected entities: {expected}
      Extracted entities: {response}

      Evaluate the extraction quality:
      1. Are all expected entities found? (recall)
      2. Are there any false positives — entities extracted that shouldn't be? (precision)
      3. Are entity boundaries correct (full names vs partial)?
      4. Are entity types correctly labeled?

  # Reporting
  metrics:
    - precision
    - recall
    - f1
    - judge_pass_rate
    - per_entity_type_f1
```

#### 2d. `README.md` — expand

Follow the same pattern as ticket-classifier README above. Add:
- Prerequisites section
- GPU requirements table
- Expected results table (precision/recall/F1 for base vs fine-tuned)
- Customization section (different entity types, different text domains)
- Time estimates

---

## New Projects to Add

### 3. `examples/intent-router/` — NEW

**Use case:** Classify user messages into intents for routing (like Oumi's Banking77 quickstart). This is the canonical task-model use case — replace a frontier model doing intent classification with a tiny fine-tuned model.

**Why:** Oumi's quickstart is exactly this. It's the most compelling demo of the "task model beats frontier model" story.

#### 3a. `synth.yaml`

```yaml
type: sdg
description: Generate intent classification training data for a support routing system
config:
  model: openai/gpt-4o-mini
  num_samples: 200
  temperature: 0.8
  max_tokens: 2048
  max_concurrency: 10
  strategy_params:
    sampled_attributes:
      - id: intent
        name: Intent
        description: The user's intent category
        possible_values:
          - id: billing_inquiry
            name: Billing Inquiry
            description: Questions about charges, invoices, payment methods
            sample_rate: 0.15
          - id: technical_support
            name: Technical Support
            description: Product bugs, errors, how-to questions
            sample_rate: 0.2
          - id: account_management
            name: Account Management
            description: Password reset, profile updates, account deletion
            sample_rate: 0.15
          - id: order_status
            name: Order Status
            description: Where is my order, tracking, delivery ETA
            sample_rate: 0.15
          - id: refund_request
            name: Refund Request
            description: Return items, request refund, dispute charge
            sample_rate: 0.1
          - id: feature_request
            name: Feature Request
            description: Suggestions for new features or improvements
            sample_rate: 0.05
          - id: complaint
            name: Complaint
            description: Dissatisfaction with service, escalation
            sample_rate: 0.1
          - id: general_inquiry
            name: General Inquiry
            description: Pricing, availability, business hours, policies
            sample_rate: 0.1
      - id: tone
        name: Tone
        description: The user's communication tone
        possible_values:
          - id: polite
            name: Polite
            description: Friendly, patient, uses pleasantries
            sample_rate: 0.4
          - id: neutral
            name: Neutral
            description: Direct and factual, no strong emotion
            sample_rate: 0.3
          - id: frustrated
            name: Frustrated
            description: Annoyed, impatient, may use strong language
            sample_rate: 0.2
          - id: urgent
            name: Urgent
            description: Time-pressured, needs immediate help
            sample_rate: 0.1
    generated_attributes:
      - id: user_message
        instruction_messages:
          - role: system
            content: >-
              Generate a realistic customer message for a SaaS product's support system.
              The message should feel authentic — vary length (1 sentence to a short
              paragraph), include typos occasionally, and match the specified tone.
              Just write the customer's message, nothing else.
          - role: user
            content: >-
              Generate a customer message with:
              - Intent: {intent} — {intent.description}
              - Tone: {tone} — {tone.description}
    transformed_attributes:
      - id: messages
        transformation_strategy:
          type: chat
          chat_transform:
            messages:
              - role: system
                content: >-
                  You are an intent classifier for a customer support system.
                  Classify the user's message into exactly one intent category.
                  Respond with only the intent label, nothing else.
                  Valid intents: billing_inquiry, technical_support, account_management,
                  order_status, refund_request, feature_request, complaint, general_inquiry
              - role: user
                content: "{user_message}"
              - role: assistant
                content: "{intent}"
    passthrough_attributes:
      - intent
      - tone
      - user_message
      - messages
```

#### 3b. `train.yaml`

```yaml
type: training
description: LoRA SFT for intent routing — Qwen3 0.6B (ultra-fast inference)
config:
  algorithm: sft
  model_name_or_path: Qwen/Qwen3-0.6B

  data_path: training_data.jsonl
  max_length: 1024

  num_train_epochs: 5
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 3.0e-04
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0

  bf16: true

  use_peft: true
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

  logging_steps: 10
  save_steps: 500
  report_to: none
  output_dir: output/intent-router

compute:
  gpus: 1
  min_vram_gb: 8
```

Note: Uses Qwen3-0.6B deliberately — intent routing should be as small and fast as possible since it's on the hot path of every request.

#### 3c. `eval.yaml`

```yaml
type: eval
description: Evaluate intent router accuracy
config:
  test_data_path: test_data.jsonl
  max_samples: 100

  deterministic_checks:
    - field: intent
      type: exact_match
      expected_from: label

  metrics:
    - overall_accuracy
    - per_class_accuracy
    - confusion_matrix
```

Note: Intent routing is a pure classification task — deterministic exact match is sufficient, no LLM-as-judge needed.

#### 3d. `README.md`

Follow the ticket-classifier README pattern. Highlight:
- Uses Qwen3-0.6B (smallest model) for minimal latency
- 8 intent categories (customizable)
- Expected accuracy: base ~50%, fine-tuned ~85%+
- Inference latency: <50ms per request
- Use case: replace frontier model doing intent classification at every API call

### 4. `examples/summarizer/` — NEW

**Use case:** Condense customer support conversations into structured summaries. Train a model to extract key information (issue, resolution, action items) from multi-turn conversations.

#### 4a. `synth.yaml`

```yaml
type: sdg
description: Generate conversation summarization training data
config:
  model: openai/gpt-4o-mini
  num_samples: 100
  temperature: 0.7
  max_tokens: 4096
  max_concurrency: 10
  strategy_params:
    sampled_attributes:
      - id: domain
        name: Domain
        description: Business domain of the conversation
        possible_values:
          - id: support
            name: Customer Support
            description: Technical support, billing, account issues
            sample_rate: 0.4
          - id: sales
            name: Sales
            description: Product inquiries, pricing, demos
            sample_rate: 0.2
          - id: hr
            name: HR
            description: Employee onboarding, benefits, policies
            sample_rate: 0.2
          - id: legal
            name: Legal
            description: Contract review, compliance, terms
            sample_rate: 0.2
      - id: length
        name: Conversation Length
        description: How long the conversation is
        possible_values:
          - id: short
            name: Short
            description: 2-4 turns, simple issue
            sample_rate: 0.3
          - id: medium
            name: Medium
            description: 5-8 turns, moderate complexity
            sample_rate: 0.5
          - id: long
            name: Long
            description: 9-15 turns, complex multi-issue conversation
            sample_rate: 0.2
    generated_attributes:
      - id: conversation
        instruction_messages:
          - role: system
            content: >-
              Generate a realistic {length} ({length.description}) business
              conversation in the {domain} domain. Format as alternating
              Customer/Agent lines. Include specific details — names, order
              numbers, dates, product names. Make it feel like a real
              conversation transcript.
          - role: user
            content: "Generate a {length} {domain} conversation."
      - id: summary
        instruction_messages:
          - role: system
            content: >-
              You are a conversation summarizer. Given a business conversation,
              produce a structured summary with exactly these fields:
              - Issue: One sentence describing the customer's problem
              - Resolution: One sentence describing how it was resolved (or "Unresolved" if not)
              - Action Items: Bullet list of follow-up actions (or "None")
              - Sentiment: positive, neutral, or negative
          - role: user
            content: >-
              Summarize this conversation:

              {conversation}
    transformed_attributes:
      - id: messages
        transformation_strategy:
          type: chat
          chat_transform:
            messages:
              - role: system
                content: >-
                  Summarize the conversation. Output exactly:
                  Issue: <one sentence>
                  Resolution: <one sentence or "Unresolved">
                  Action Items: <bullet list or "None">
                  Sentiment: <positive/neutral/negative>
              - role: user
                content: "{conversation}"
              - role: assistant
                content: "{summary}"
    passthrough_attributes:
      - domain
      - length
      - conversation
      - summary
      - messages
```

#### 4b. `train.yaml`

```yaml
type: training
description: LoRA SFT for conversation summarization — Qwen 2.5 1.5B
config:
  algorithm: sft
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct

  data_path: training_data.jsonl
  max_length: 4096

  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-04
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0

  bf16: true

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

  gradient_checkpointing: true

  logging_steps: 10
  save_steps: 500
  report_to: none
  output_dir: output/summarizer

compute:
  gpus: 1
  min_vram_gb: 8
```

#### 4c. `eval.yaml`

```yaml
type: eval
description: Evaluate summarization quality — structure compliance + LLM judge
config:
  test_data_path: test_data.jsonl
  max_samples: 50

  # Deterministic checks — does the output have the right structure?
  deterministic_checks:
    - field: output
      type: contains
      values: ["Issue:", "Resolution:", "Action Items:", "Sentiment:"]
    - field: sentiment
      type: enum_match
      allowed: [positive, neutral, negative]

  # LLM-as-judge — assess summary quality
  judge:
    model: openai/gpt-4o-mini
    temperature: 0.0
    prompt: >-
      A conversation was summarized. Evaluate the summary quality.

      Original conversation:
      {request}

      Summary:
      {response}

      Evaluate:
      1. Does the Issue field accurately capture the customer's problem?
      2. Does the Resolution field correctly describe the outcome?
      3. Are the Action Items complete and actionable?
      4. Is the Sentiment classification correct?
      5. Is the summary concise without losing critical information?

  metrics:
    - structure_compliance_rate
    - judge_pass_rate
    - per_field_accuracy
```

#### 4d. `README.md`

Follow the ticket-classifier README pattern. Highlight:
- Structured output (Issue, Resolution, Action Items, Sentiment)
- 4 business domains (support, sales, HR, legal)
- Variable conversation lengths (2-15 turns)
- Expected results: base model produces verbose summaries, fine-tuned produces structured summaries consistently

### 5. `examples/content-moderator/` — NEW

**Use case:** Binary classification — classify user-generated content as safe or unsafe. The simplest possible task model, good as a first example for new users.

#### 5a. `synth.yaml`

```yaml
type: sdg
description: Generate content moderation training data — safe vs unsafe classification
config:
  model: openai/gpt-4o-mini
  num_samples: 200
  temperature: 0.8
  max_tokens: 2048
  max_concurrency: 10
  strategy_params:
    sampled_attributes:
      - id: safety_label
        name: Safety Label
        description: Whether the content is safe or unsafe
        possible_values:
          - id: safe
            name: Safe
            description: Appropriate content that follows community guidelines
            sample_rate: 0.6
          - id: unsafe
            name: Unsafe
            description: Content that violates community guidelines
            sample_rate: 0.4
      - id: content_type
        name: Content Type
        description: The type of user-generated content
        possible_values:
          - id: comment
            name: Comment
            description: A comment on a post or article
            sample_rate: 0.3
          - id: review
            name: Review
            description: A product or service review
            sample_rate: 0.3
          - id: message
            name: Message
            description: A direct message or chat message
            sample_rate: 0.2
          - id: post
            name: Post
            description: A social media post or forum post
            sample_rate: 0.2
      - id: violation_type
        name: Violation Type
        description: Type of guideline violation (only applies when unsafe)
        possible_values:
          - id: harassment
            name: Harassment
            description: Targeted abuse, bullying, or intimidation
            sample_rate: 0.25
          - id: hate_speech
            name: Hate Speech
            description: Discrimination based on protected characteristics
            sample_rate: 0.2
          - id: misinformation
            name: Misinformation
            description: Deliberately false or misleading claims
            sample_rate: 0.2
          - id: spam
            name: Spam
            description: Unsolicited commercial content or scams
            sample_rate: 0.2
          - id: self_harm
            name: Self Harm
            description: Content promoting or glorifying self-harm
            sample_rate: 0.15
    generated_attributes:
      - id: content
        instruction_messages:
          - role: system
            content: >-
              You are generating synthetic content moderation training data. Generate
              a realistic piece of user-generated content. For safe content, write
              normal, appropriate messages. For unsafe content, write content that
              is clearly problematic but not graphic — it should be detectable by
              a classifier. Do not include labels or metadata. Just write the content.
          - role: user
            content: >-
              Generate a {content_type} that is {safety_label}.
              {safety_label == "unsafe" ? "Violation type: {violation_type} — {violation_type.description}" : ""}
    transformed_attributes:
      - id: messages
        transformation_strategy:
          type: chat
          chat_transform:
            messages:
              - role: system
                content: >-
                  You are a content moderator. Classify the following content as
                  either "safe" or "unsafe". Respond with exactly one word.
              - role: user
                content: "{content}"
              - role: assistant
                content: "{safety_label}"
    passthrough_attributes:
      - safety_label
      - content_type
      - content
      - messages
```

Note: The conditional `{safety_label == "unsafe" ? ...}` may not be supported by asynth. If not, split into two generated attributes or use the violation_type for all samples and ignore it for safe ones.

#### 5b. `train.yaml`

```yaml
type: training
description: LoRA SFT for content moderation — Qwen3 0.6B
config:
  algorithm: sft
  model_name_or_path: Qwen/Qwen3-0.6B

  data_path: training_data.jsonl
  max_length: 1024

  num_train_epochs: 5
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 3.0e-04
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0

  bf16: true

  use_peft: true
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

  logging_steps: 10
  save_steps: 500
  report_to: none
  output_dir: output/content-moderator

compute:
  gpus: 1
  min_vram_gb: 8
```

Note: Uses Qwen3-0.6B — binary classification is simple enough for the smallest model, and latency matters for moderation (blocking pipeline).

#### 5c. `eval.yaml`

```yaml
type: eval
description: Evaluate content moderator — precision, recall, F1
config:
  test_data_path: test_data.jsonl
  max_samples: 100

  deterministic_checks:
    - field: safety_label
      type: exact_match
      expected_from: label

  metrics:
    - overall_accuracy
    - precision
    - recall
    - f1
    - false_positive_rate
    - false_negative_rate
```

No LLM-as-judge needed — binary classification is pure deterministic eval.

#### 5d. `README.md`

Highlight:
- Simplest possible task model (binary classification)
- Good first project for new users
- Uses Qwen3-0.6B for minimal latency
- Critical metric: false negative rate (missing unsafe content is worse than false positives)
- Expected: base model ~75%, fine-tuned ~95%+

### 6. `examples/distillation/` — NEW

**Use case:** Distill a frontier model's classification ability into a tiny model using GKD. Demonstrates the distillation workflow — no synthetic data needed, just a teacher and student.

#### 6a. `synth.yaml`

Same as ticket-classifier synth config — generates labeled tickets as training data.

Or alternatively, skip synth and use the teacher model to generate completions on unlabeled data (true distillation). In that case, this file would be a "generate completions" config that runs the teacher on raw prompts.

#### 6b. `train.yaml`

```yaml
type: training
description: GKD distillation — Qwen3 4B teacher → Qwen3 0.6B student
config:
  algorithm: gkd
  model_name_or_path: Qwen/Qwen3-0.6B
  teacher_model_name_or_path: Qwen/Qwen3-4B

  data_path: training_data.jsonl

  max_steps: 500
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-04
  lr_scheduler_type: cosine
  warmup_steps: 50
  weight_decay: 0.01
  max_grad_norm: 1.0

  temperature: 0.9
  lmbda: 0.5
  beta: 0.5
  max_new_tokens: 256
  disable_dropout: true

  bf16: true

  logging_steps: 10
  save_steps: 100
  report_to: none
  output_dir: output/distillation

compute:
  gpus: 1
  min_vram_gb: 16
```

#### 6c. `eval.yaml`

Same structure as ticket-classifier eval — compare student accuracy vs teacher accuracy.

#### 6d. `README.md`

Highlight:
- No synthetic data generation step — distillation uses the teacher model directly
- Teacher: Qwen3 4B, Student: Qwen3 0.6B
- Expected: student reaches ~90% of teacher accuracy at 1/7th the size and inference cost
- Use case: when you already have a working larger model and want to compress it

---

## File Summary

| Project | Action | Use Case |
|---|---|---|
| `examples/ticket-classifier/` | **Upgrade** — fix model, expand train/eval/README | Multi-label classification |
| `examples/entity-extractor/` | **Upgrade** — fix model, expand train/eval/README | Structured extraction |
| `examples/intent-router/` | **New** | Single-label classification (routing) |
| `examples/summarizer/` | **New** | Structured summarization |
| `examples/content-moderator/` | **New** | Binary classification |
| `examples/distillation/` | **New** | Model compression via GKD |

## Verification

After implementing:
1. All YAML files parse cleanly
2. Each project's synth.yaml can run with `amortized submit <path> --confirm` (at least dry-run)
3. Each README has: prereqs, pipeline steps, expected results, GPU requirements, customization tips
4. No config references `gpt-5.4` — all should use `gpt-4o-mini`
