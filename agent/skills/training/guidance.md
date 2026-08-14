# Training Skill Guidance

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| knowledge-ingestion/osft | `skills/training/knowledge-ingestion/osft/` | Knowledge ingestion, FAQ bots, doc-grounded QA |

## How to Choose

- **Knowledge ingestion** → OSFT (default, recommended)

## Student Model Selection

You MUST show VRAM estimates before presenting model options. The user
needs to see GPU memory requirements before choosing.

1. Estimate training resources for EACH candidate model size with the
   default method (lora)
2. Show a VRAM comparison card with ALL collected estimates
3. THEN present model options

## Training Method Selection

You MUST show VRAM estimates before presenting method options.

1. Estimate training resources with the selected model size for EACH
   method (lora, qlora, osft, sft)
2. Show a VRAM comparison card with ALL collected estimates
3. THEN present method options

## Training Confirmation

Before submitting, estimate training resources with the final model
size and method, then show the VRAM card so the user sees what they
are committing to.

## Job Chaining

Set `parent_job_id` to the SDG job ID. The worker resolves the SDG
output from MLflow and sets `data_path` automatically. No manual
data path configuration needed.
