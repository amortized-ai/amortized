# Amortized

Build task models that replace frontier API calls.

---

Take any task your AI agent does with GPT-4o — classification, extraction, summarization, routing — and train a small, fast, cheap model that does it just as well. One pipeline: generate data → train → evaluate → serve.

## The Pipeline

```
1. Generate    →  Create labeled training data using a frontier model
2. Train       →  Fine-tune a small model (Qwen 0.6B–7B) on that data
3. Serve       →  Host the model with an OpenAI-compatible API
4. Evaluate    →  Compare accuracy: base vs fine-tuned vs frontier
```

Each step is a job submitted through the CLI. Templates handle the config. You bring the task definition.

## Quickstart

```bash
pip install -e .
amortized config     # set up your GPU backend
amortized up         # start the server
```

### Run the ticket classifier example

```bash
# Generate 100 labeled support tickets
amortized submit sdg --recipe examples/ticket-classifier/synth \
  --set model=openai/gpt-4o-mini --confirm

# Train Qwen 1.5B with LoRA
amortized submit training --recipe examples/ticket-classifier/train \
  --data <artifact-id> --confirm

# Serve the fine-tuned model
amortized submit serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter <model-artifact-id> --confirm

# Evaluate
amortized submit eval --recipe examples/ticket-classifier/eval \
  --data <artifact-id> --serve <serve-job-id> --confirm
```

## Examples

| Project | Task | Model |
|---------|------|-------|
| [ticket-classifier](examples/ticket-classifier/) | Classify support tickets by urgency + topic | Qwen 1.5B |
| [intent-router](examples/intent-router/) | Route user messages to the right handler | Qwen3 0.6B |
| [entity-extractor](examples/entity-extractor/) | Extract structured entities from text | Qwen 1.5B |
| [summarizer](examples/summarizer/) | Condense conversations into structured summaries | Qwen 1.5B |
| [content-moderator](examples/content-moderator/) | Binary safe/unsafe classification | Qwen3 0.6B |
| [distillation](examples/distillation/) | Compress a 4B model into 0.6B via GKD | Qwen3 0.6B |

## Templates

Reusable configs for each step of the pipeline:

- **`templates/sdg/`** — 10 data generation templates (classification, extraction, conversation, Q&A, tool-use, ...)
- **`templates/training/`** — 8 training algorithms (LoRA SFT, QLoRA, DPO, KTO, GRPO, GKD, GOLD) + 10 model presets
- **`templates/eval/`** — 19 evaluation judges (safety, truthfulness, instruction-following, code quality, ...)

## Architecture

```
amortized CLI
    ↓
FastAPI server (jobs, artifacts, events)
    ↓
compute backends (SSH → podman containers on GPU nodes)
    ↓
┌──────────────┬──────────────┬──────────────┐
│  asynth      │  TRL         │  vLLM        │
│  (SDG+eval)  │  (training)  │  (serving)   │
└──────────────┴──────────────┴──────────────┘
```

No custom containers. Official upstream images:
- Training: `docker.io/huggingface/trl`
- Serving: `docker.io/vllm/vllm-openai`
- SDG + Eval: `ghcr.io/amortized-ai/asynth`

## Why "Amortized"

Amortization spreads a large cost over time. This project spreads the capability of an expensive frontier model across many cheap inferences by a smaller fine-tuned model — reducing per-call cost while preserving quality.

## License

[Apache 2.0](LICENSE)
