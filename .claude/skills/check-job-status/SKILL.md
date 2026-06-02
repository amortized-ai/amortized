---
description: Check the status and metrics of a running or completed job
invoke-on-demand: true
---

Check job status, training metrics, and output artifacts.

## Get job status

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

Returns job details including `status` (pending, running, completed, failed, cancelled), timestamps, and configuration.

## Get training metrics (training jobs only)

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}/metrics
```

Returns per-step metrics: `step`, `loss`, `epoch`, `learning_rate`, `max_steps`.

## Get job artifacts

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}/artifacts
```

Returns list of output artifacts (model checkpoints, generated data files) with paths and sizes.

## Cancel a job

```bash
curl -X DELETE http://localhost:8000/api/v1/jobs/{job_id}
```
