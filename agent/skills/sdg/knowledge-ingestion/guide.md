# Knowledge Ingestion -- SDG Workflow

Generate QA training data from uploaded documents for FAQ bots,
QA assistants, and doc-grounded chat.

**Document analysis workflow:** Use `get_document_chunks(doc_id)` to
get chunks with token counts and headings. Use `get_document_content`
if you need full text for prompt writing.

**Keep it brief.** Do your analysis silently. Present results and ask
for confirmation -- do not narrate your reasoning.

## Workflow

Ask the user these questions one at a time using `present_options`.

### Step 1 -- Documents

Check if they've uploaded documents via the Documents page. Use
`list_documents` to show available documents with their IDs. If not
uploaded yet, guide them to upload first.

### Step 2 -- Question types

What kinds of questions should the model handle?

Default distribution: factual (25%), procedural (35%),
troubleshooting (25%), comparison (15%). Adapt to the domain -- a
troubleshooting guide needs more troubleshooting questions, a
reference manual needs more factual ones.

### Step 3 -- Difficulty levels

Default: basic (35%), intermediate (45%), advanced (20%).

### Step 4 -- Teacher model

Call `list_models` to get models from the AI Gateway. Present ONLY
those models. If none returned, direct user to Settings -> AI Gateway.

### Step 5 -- Sample count

Use `get_document_chunks(doc_id)` for chunk count and token stats.
Use ALL chunks -- the worker sends every chunk to Data Designer,
so do not filter or exclude any.

Compute samples from chunk statistics:

```
avg_qa_tokens ~ 200  (empirical median; ranges 80-300)
mean_chunk_tokens = mean of num_tokens across ALL chunks
multiplier = coverage x mean_chunk_tokens / avg_qa_tokens
num_samples = multiplier x num_chunks
```

Present three tiers with computed sample counts:

1. 3x source coverage -- Good starting point
2. 5x source coverage -- Recommended
3. 8x source coverage -- Best quality, thorough coverage

Example: 50 chunks, mean 400 tokens:
- 3x: multiplier = 3 x 400 / 200 = 6 -> 300 samples
- 5x: multiplier = 5 x 400 / 200 = 10 -> 500 samples
- 8x: multiplier = 8 x 400 / 200 = 16 -> 800 samples

Show the actual numbers, multiplier, and coverage tier.

### Step 6 -- System prompt

Do NOT ask the user to write a system prompt. Generate a
domain-specific default based on document content. Mention it in
the confirmation table so the user can adjust.

## Reference Payload

Use this as the base for `create_sdg_job()`. Replace `[DOMAIN]` with
the actual product/technology name. Set `model` to the user's choice
from `list_models`. Adapt sampler weights to the user's distribution
preferences from the workflow.

```json
{
  "num_records": 500,
  "document_ids": ["<doc-id-from-list_documents>"],
  "topic": "[DOMAIN] knowledge QA",
  "columns": [
    {
      "column_type": "sampler",
      "name": "difficulty",
      "sampler_type": "category",
      "params": {
        "values": [
          "Basic - Simple factual question, single concept",
          "Intermediate - Requires connecting multiple concepts or steps",
          "Advanced - Multi-step reasoning, edge cases, comparing approaches"
        ],
        "weights": [0.35, 0.45, 0.20]
      }
    },
    {
      "column_type": "sampler",
      "name": "question_type",
      "sampler_type": "category",
      "params": {
        "values": [
          "Factual - Understanding what something is, why it works, or how components relate",
          "Procedural - Step-by-step question about accomplishing a specific task",
          "Troubleshooting - Diagnosing and resolving a specific error or unexpected behavior",
          "Comparison - Choosing between options, understanding trade-offs, or comparing configurations"
        ],
        "weights": [0.25, 0.35, 0.25, 0.15]
      }
    },
    {
      "column_type": "llm-text",
      "name": "question",
      "model_alias": "text",
      "system_prompt": "You generate specific, realistic questions about [DOMAIN] documentation. The question MUST be fully answerable using ONLY the provided documentation context. Do not ask about concepts that are merely mentioned but not explained in the context. Output ONLY the question text, nothing else.",
      "prompt": "Documentation context:\n{{ content }}\n\nDifficulty: {{ difficulty }}\nQuestion type: {{ question_type }}"
    },
    {
      "column_type": "llm-text",
      "name": "answer",
      "model_alias": "text",
      "system_prompt": "You are a knowledgeable documentation assistant for [DOMAIN]. Answer ONLY using information from the provided documentation context. Do not add commands, procedures, or details that are not explicitly present in the context. Be concise but thorough. The question is answerable from the provided context, so read it carefully and answer from it.",
      "prompt": "Documentation context:\n{{ content }}\n\nQuestion: {{ question }}"
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
          {"role": "system", "content": "You are a knowledgeable documentation assistant for [DOMAIN]. Answer accurately based on the official documentation. Be concise and provide step-by-step instructions when appropriate."},
          {"role": "user", "content": "{{ question }}"},
          {"role": "assistant", "content": "{{ answer }}"}
        ]
      }
    }
  ]
}
```

### Adapting the Payload

- **`num_records`**: computed from Step 5 formula, not a static default
- **`document_ids`**: from `list_documents` in Step 1
- **`columns[*].params.weights`**: match the user's distribution choices
- **`model_configs[0].model`**: the model chosen in Step 4
- **System prompts**: replace `[DOMAIN]` with the actual product name
- **No `topic` sampler**: chunk content (`{{ content }}`) determines subject matter -- a topic sampler creates mismatches where the LLM ignores the context and hallucinates based on the topic
- **Question system prompt**: must include the answerability constraint ("MUST be fully answerable using ONLY the provided documentation context")
- **Answer system prompt**: must include "Answer ONLY using information from the provided documentation context". Do NOT add "Include specific commands" (encourages fabrication) or "If the documentation does not cover the topic, say so" (teaches refusal)

## After SDG -- Training

Recommend OSFT training. Read `skills/training/knowledge-ingestion/osft/guide.md`.
The SDG job's output becomes the training job's data via parent job chaining.
