# Knowledge Ingestion — SDG Guide

Use this guide when building FAQ bots, QA assistants, document-grounded chat,
or RAG-deployed knowledge models. The template and guidelines here are
suggestions — the user has the final call on each knob.

## Requirement Gathering

Ask the user these questions (one at a time, with numbered options):

1. **What documents?** — What documentation or knowledge base should the
   model learn from? (PDF, DOCX, TXT, or a URL)
2. **What topics?** — What are the main topic areas in the documentation?
   Suggest 5-9 topics weighted by documentation depth per section.
3. **RAG or closed-book?** — Will this model be deployed with a RAG system
   (retrieval-augmented generation) or standalone?
4. **Question types** — What kinds of questions should it handle?
   Default: factual (25%), procedural (35%), troubleshooting (25%),
   comparison (15%)
5. **Difficulty levels** — Default: basic (35%), intermediate (45%),
   advanced (20%)
6. **How many samples?** — Suggest based on document size
7. **"I don't know" behavior** — Should the model say "I don't know" for
   questions outside its knowledge, or always attempt an answer?

## SDG Recipe Configuration

### Document Segmentation

| Knob | What it does | Recommended |
|------|-------------|-------------|
| `segmentation_params.segment_length` | Token count per document chunk | 2048 tokens |
| `segmentation_params.segment_overlap` | Overlap between chunks | 256 tokens |
| `segmentation_params.tokenizer` | Tokenizer for counting tokens | `cl100k_base` |

**Prefer segmenting.** Passing the full document to every sample inflates
token usage, slows generation, and produces unfocused answers. Chunked
context produces more focused, grounded answers. Only skip segmentation if
the user has a specific need for full-document context.

### Generation Architecture

**Prefer single-turn QA** — question and answer as separate LLM calls (two
`generated_attributes`). Do NOT use `multiturn_attributes` for document-
grounded QA. Multi-turn generation uses one teacher model to role-play both
user and assistant; when both roles receive document context, the model
generates assistant-style content regardless of role.

**Prefer few-shot examples** — Start with 3 domain-specific Q&A examples in
the generation prompt via `input_examples`. Guides format, tone, and
grounding behavior.

**Prefer postprocessing** — `cut_prefix`/`cut_suffix` on both question and
answer to strip LLM formatting artifacts (e.g., "Question:", "End Question").

**Use `messages` as the transformed attribute ID** — Training Hub expects a
`messages` column. Setting the `transformed_attributes` id to anything else
(e.g., `conversation`) will silently break training.

### Sampled Attributes

Start with 2-3 sampled dimensions that capture the most important axes of
variation. Adding more dimensions increases diversity but can dilute focus.

| Attribute | Purpose | Example values |
|-----------|---------|----------------|
| `question_type` | Covers breadth of user query patterns | factual, procedural, troubleshooting, comparison |
| `topic` | Ensures coverage proportional to content density | Weighted by documentation depth per section |
| `difficulty` | Varies answer depth and complexity | basic, intermediate, advanced |

### RAG Context in Training Data

For RAG-deployed models, include the context segment in the training data.
Put it before the question in the user turn:

```json
{"role": "user", "content": "{context_segment}\n\n{cleaned_question}"}
```

No prefix like "Context:" or "Based on the following:" — the model should
learn to handle context regardless of formatting, since downstream RAG
systems may vary.

Including context improves RAG eval performance but lowers closed-book
performance. This is the correct tradeoff for a RAG-deployed model.
Dynamically choose the data format based on whether the user plans to
deploy with RAG or not.

### Prompt Engineering

**Question generation — enforce answerability:**
Add to the system prompt: "The question MUST be fully answerable using ONLY
the provided documentation context. Do not ask about concepts that are
merely mentioned but not explained in the context."

**Answer generation — trust the context:**
Use: "The question is answerable from the provided context, so read it
carefully and answer from it."

Do NOT use: "If the documentation does not cover the topic, say so." This
encourages disclaimer responses and teaches the model to say "I don't know."
This is sensible as default, but confirm the user's preference on "I don't
know" samples.

### Defaults — What NOT to Do

- **Don't skip segmentation** — full document per sample wastes tokens,
  produces unfocused answers
- **Don't use multi-turn for document QA** — role confusion is structural
- **Don't include "I don't know" training samples** by default — teaches
  refusal behavior. Only include if the user explicitly wants IDK behavior
- **Don't add too many sampled dimensions** — 3 focused dimensions often
  outperform 5+ diluted ones

## Recommended Training Method

**OSFT is the default for knowledge ingestion.** It outperforms standard
SFT by 30+ percentage points in open-book settings. When the user is
ready to train, recommend reading `skills/training/osft/guide.md`.

## Config Template

The config template at `skills/sdg/knowledge-ingestion/sdg-recipe-template.json`
shows the golden parameterization for a knowledge-ingestion SDG job. Use it
as a starting point and customize based on the user's requirements —
specifically the topics, question types, few-shot examples, and system
prompts should be tailored to the user's domain.
