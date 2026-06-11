# Amortized — Architecture

## Three Layers

```
┌───────────────────────────────────────────────────────────┐
│                       Interfaces                          │
│                                                           │
│   CLI              MCP Server           REST API          │
│   amortized        auto-generated       /api/v1/*         │
│   submit/jobs/     from OpenAPI         direct HTTP       │
│   logs/cancel      (Claude Code)                          │
│                                                           │
└────────────────────────┬──────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────┐
│                    Control Plane                           │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  FastAPI Server (main.py)                            │ │
│  │  16 API routes · Agent chat · MCP · WebSocket · Auth │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Core Domain Logic                                   │ │
│  │  jobs.py    — state machine (queued→running→done)    │ │
│  │  recipes.py — YAML templates with inheritance        │ │
│  │  artifacts.py — output registration + resolution     │ │
│  │  events.py  — job event stream (SSE)                 │ │
│  │  judge_templates.py — 19 eval judge configs          │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Worker (worker.py)                                  │ │
│  │                                                      │ │
│  │  Picks up queued jobs and generates configs:          │ │
│  │  • Training → TRL YAML config                        │ │
│  │  • SDG      → Python script (asynth synthesize)      │ │
│  │  • Eval     → Python script (inference + judge)      │ │
│  │  • Serve    → vLLM YAML config                       │ │
│  │                                                      │ │
│  │  Then dispatches to compute backend,                  │ │
│  │  polls until completion, fetches outputs,             │ │
│  │  registers artifacts.                                 │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  SQLite (aiosqlite)                                  │ │
│  │  jobs · artifacts · events · api_keys · evaluators   │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
└────────────────────────┬──────────────────────────────────┘
                         │ SSH + podman secrets
                         │ config.yaml / run.py written to remote
                         │ podman run -d --gpus all {image} {cmd}
                         │
┌────────────────────────▼──────────────────────────────────┐
│                       Compute                              │
│                   (GPU node via SSH)                       │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ Training    │  │ SDG + Eval  │  │ Serving          │  │
│  │             │  │             │  │                  │  │
│  │ trl sft     │  │ asynth      │  │ vllm serve      │  │
│  │ --config    │  │ synthesize  │  │ --config         │  │
│  │             │  │ + judges    │  │                  │  │
│  │ huggingface │  │ amortized-  │  │ vllm/            │  │
│  │ /trl:1.5.0  │  │ ai/asynth  │  │ vllm-openai     │  │
│  └─────────────┘  └─────────────┘  └──────────────────┘  │
│                                                           │
│  All official upstream container images.                   │
│  No custom Dockerfiles. No baked-in runner code.          │
│  Worker generates configs, containers execute them.        │
│                                                           │
│  ~/amortized-jobs/{job-id}/                               │
│    config.json · config.yaml / run.py · outputs           │
│    mounted at /amortized/work inside containers           │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Pipeline Flow

```
User describes task
        │
        ▼
┌─ SDG ──────────────────────────────────────┐
│ Frontier model generates labeled data       │
│ Input: recipe template + model              │
│ Output: training_data.jsonl (artifact)      │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌─ Train ────────────────────────────────────┐
│ Fine-tune small model on generated data     │
│ Input: data artifact + model + algorithm    │
│ Output: model artifact (adapter weights)    │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌─ Serve ────────────────────────────────────┐
│ Deploy fine-tuned model via vLLM            │
│ Input: base model + adapter artifact        │
│ Output: OpenAI-compatible API on port 8000  │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌─ Eval ─────────────────────────────────────┐
│ Run inference + compute metrics + LLM judge │
│ Input: test data + served model endpoint    │
│ Output: eval_results.json (artifact)        │
│                                             │
│ Deterministic: exact match, contains, enum  │
│ LLM judge: 19 templates (safety,            │
│   truthfulness, classification, code, ...)  │
└─────────────────────────────────────────────┘
```

## What Ships

```
templates/
  sdg/       10 data generation templates
  training/   8 algorithms + 10 model presets
  eval/      19 judge templates

examples/
  ticket-classifier/    synth + train + eval
  intent-router/        synth + train + eval
  entity-extractor/     synth + train + eval
  summarizer/           synth + train + eval
  content-moderator/    synth + train + eval
  distillation/         synth + train + eval (GKD)
```
