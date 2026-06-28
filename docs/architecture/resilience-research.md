# Resilience & Retry — Deep Research Summary

Adversarially verified research across 102 agents, 20 sources, 24/25 claims confirmed.

## Key Findings

### 1. NCCL Errors — Root Cause is Counterintuitive

**>60% of NCCL watchdog timeouts are caused by CPU-side issues, NOT GPU/network hardware.** (Meta fleet data, PyTorch blog March 2026)

Root causes: data loading stalls, checkpointing, PT2 compilation, cross-rank code path divergence. Network/hardware is only 20-30%.

**Detection:** PyTorch Flight Recorder (`TORCH_NCCL_DUMP_ON_TIMEOUT=1`) dumps per-rank ring buffers with operation state, sequence IDs, timing, stack frames. Default watchdog timeout: 10 minutes (600,000ms).

**For amortized:** Most NCCL errors at our scale (single or few-node) will be code/data issues, not hardware. Retry with same config will likely fail again. Log the stderr pattern and surface it to the user rather than blind retry.

### 2. Retry Strategies — Platform Comparison

| Platform | Retry Unit | Config | Checkpoint Resume | Node Exclusion |
|---|---|---|---|---|
| **Ray Train** | Full worker group restart | `FailureConfig(max_failures=N)`, -1 for unlimited | Auto via `get_checkpoint()` from `storage_path` | Not built-in |
| **SkyPilot** | Full cluster teardown + reprovision | `job_recovery.max_restarts_on_errors`, `recover_on_exit_codes` | User must implement (mount NFS/bucket) | Cross-region failover |
| **veRL** | Checkpoint resume | `resume_mode: auto\|resume_path\|disable` | `latest_checkpointed_iteration.txt` tracker file | Not built-in |
| **MegaScale** | K8s node eviction + restart | Automatic, ~90% of recoveries automated | Two-stage async (GPU→host memory→HDFS) | K8s-based node replacement |
| **ByteRobust** | Hierarchical 4-stage | diagnose→reattempt→rollback→dual-phase replay | Async every-step with dual-buffer | Faulty node isolation |

**Pattern:** Every platform uses full-group restart (not individual worker replacement). Checkpoint-based resume is universal but varies in automation.

### 3. Checkpoint-Based Resume

**veRL pattern** (most practical for amortized):
- `find_latest_ckpt_path()` reads `latest_checkpointed_iteration.txt` → returns directory like `global_step_200`
- Saves: per-rank model/optimizer states, LR scheduler, all RNG states, tokenizer on rank 0
- `resume_mode: auto` does this automatically

**TRL/HuggingFace pattern:**
- `resume_from_checkpoint=True` in TrainingArguments → auto-finds latest `checkpoint-N` directory
- Saves: model, optimizer, scheduler, RNG, trainer state
- Works with LoRA: saves only adapter weights

**For amortized:** Add `resume_from_checkpoint` support to training configs. On retry, point to the last checkpoint directory in the output path.

### 4. Hung Job Detection

**ByteRobust:** Stack-trace aggregation every 10s over 5 rounds. NIC crash detected in 30s, driver hang in 10s.

**MegaScale:** RDMA traffic heartbeat monitoring. Decline → alert. Cessation → auto-recovery.

**Current amortized:** Heartbeat timeout at 300s (5 minutes) → probe backend → mark failed. This is adequate for single-node but should be configurable.

### 5. Spot/Preemption Recovery

**SkyPilot:** Tear down entire cluster → search across regions/clouds for available instances → reprovision → restart from scratch (or checkpoint if user implemented it).

**Key:** Exit code 137 is reserved by SkyPilot for internal Ray task cancellation — never use it for application errors.

### 6. OOM — Research Gap

No platform has automated GPU OOM recovery (auto-reduce batch size). OOM is typically a configuration error that needs human intervention. The practical pattern:
- Detect OOM via exit code or stderr (`CUDA out of memory`, `RuntimeError: CUDA error: out of memory`)
- Don't retry with same config — it will OOM again
- Surface the error with a recommendation: "Reduce batch_size, enable gradient_checkpointing, or use QLoRA"

### 7. Network Failures — Research Gap

Beyond NCCL timeout mechanisms, no confirmed patterns for SSH disconnect or control-plane-to-compute partition handling. The practical pattern:
- Heartbeat-based detection (amortized already has this at 300s)
- On heartbeat timeout: probe backend status
- If backend says alive: re-establish monitoring
- If backend says dead: mark failed, offer retry

## What Amortized Should Implement

### Phase 1: Smart Failure Classification (now)

Parse stderr on job failure and classify the error:

```python
FAILURE_PATTERNS = {
    "oom": [r"CUDA out of memory", r"RuntimeError.*out of memory", r"OOM"],
    "nccl": [r"NCCL timeout", r"NCCL error", r"watchdog timeout"],
    "checkpoint": [r"FileNotFoundError.*checkpoint", r"safetensors.*corrupt"],
    "import": [r"ModuleNotFoundError", r"ImportError"],
    "config": [r"ValueError", r"KeyError.*config"],
}
```

Return the classification with the failure reason so the UI can show actionable messages ("OOM detected — try reducing batch_size or enabling gradient_checkpointing").

### Phase 2: Configurable Retry (next)

```yaml
compute:
  retry:
    max_retries: 3
    retryable_exit_codes: [137, 139, 143]  # SIGKILL, SIGSEGV, SIGTERM
    non_retryable_patterns: ["OOM", "ImportError", "ValueError"]
    backoff_seconds: [30, 60, 120]
```

Don't retry OOM or config errors. Do retry signal-based kills (spot preemption, pod eviction).

### Phase 3: Checkpoint Resume (later)

Add `resume_from_checkpoint` to training configs. On manual retry, pass `--resume` flag:
```bash
amortized submit training --resume <failed-job-id>
```

This resolves the last checkpoint from the failed job's output directory and passes `resume_from_checkpoint=<path>` to TRL.

## Sources

- PyTorch Flight Recorder: https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/
- ByteRobust (SOSP 2025): https://arxiv.org/abs/2509.16293
- MegaScale (NSDI 2024): https://arxiv.org/html/2402.15627v1
- Ray Train fault tolerance: https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html
- SkyPilot managed jobs: https://docs.skypilot.co/en/latest/examples/managed-jobs.html
- veRL checkpointing: https://verl.readthedocs.io/en/latest/advance/checkpoint.html
