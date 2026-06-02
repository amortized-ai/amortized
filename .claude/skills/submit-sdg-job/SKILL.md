---
description: Submit a synthetic data generation (SDG) job to the Amortized runtime API
invoke-on-demand: true
---

Submit an SDG job to generate training data using a teacher model.

## Required parameters
- `flow_id` — SDG flow identifier (use list-flows to discover available flows)
- `dataset_path` — path to input dataset
- `model` — teacher model ID (e.g. `openai/gpt-4o`, `hosted_vllm/meta-llama/Llama-3.3-70B-Instruct`)

## Optional parameters
- `api_base` — teacher model API base URL (default: `http://localhost:8101/v1`)
- `api_key` — API key for the teacher model provider
- `runtime_params` — per-block parameter overrides (e.g. `{"gen_qa_pairs": {"n": 50, "temperature": 0.7}}`)

## API call

```bash
curl -X POST http://localhost:8000/api/v1/jobs/sdg \
  -H "Content-Type: application/json" \
  -d '{
    "flow_id": "epic-jade-656",
    "dataset_path": "./input_data.jsonl",
    "model": "openai/gpt-4o",
    "api_key": "sk-...",
    "runtime_params": {}
  }'
```

The response contains a `job_id` that can be used to check status.
