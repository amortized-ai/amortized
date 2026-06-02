---
description: List all training and SDG jobs with optional status/type filters
invoke-on-demand: true
---

List all jobs in the Amortized runtime.

## API call

```bash
# List all jobs
curl http://localhost:8000/api/v1/jobs

# Filter by status
curl "http://localhost:8000/api/v1/jobs?status=running"

# Filter by type
curl "http://localhost:8000/api/v1/jobs?type=training"

# Filter by both
curl "http://localhost:8000/api/v1/jobs?status=completed&type=sdg"
```

## Status values
- `pending` — queued, not yet started
- `running` — currently executing
- `completed` — finished successfully
- `failed` — exited with error
- `cancelled` — stopped by user

## Type values
- `training` — LoRA SFT training job
- `sdg` — synthetic data generation job
