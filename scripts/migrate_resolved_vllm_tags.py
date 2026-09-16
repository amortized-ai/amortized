"""One-time migration: backfill resolved_vllm_args tags (new full-dump
format) for succeeded eval runs that used embedded vLLM serving but
predate the fingerprint recording.

For each unique (model source, served name) it replays the exact
production serve-side resolution — download the training export (or let
vLLM fetch the hub model), patch architectures, then run
serve_vllm.py's own _dump_resolved_args with the same argv shape the
job used — and stamps the resulting fingerprint onto each target run.
"""

import json
import os
import subprocess
import sys
import urllib.request

TRACKING = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")

# (training mlflow run id or "", hub model id or "", served name, eval run ids)
COMBOS = [
    ("2c14f2765032480082b7dc258ae056ae", "", "mdl-redolent-skunk-338",
     ["563de28813e745eaa66abfd83c84a48e", "d5693cd9ce194cab9ac5ff9cbbfb8102"]),
    ("bccf6e25a3fb492a8b516c73cd209edf", "", "mdl-brawny-jay-896",
     ["f31fc30a4cf54b53952f4ae4a3c309c0", "114aff8370e74e08b65b8098f9d67987"]),
    ("", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B",
     ["4f62a95a19e746d1b85ed0bd885da8ee"]),
]


def resolve_model_dir(train_run: str, hub_model: str) -> str:
    if hub_model:
        return hub_model  # vLLM/HF fetches the config itself
    local = "/amortized/work/served_model"
    subprocess.run(
        ["mlflow", "artifacts", "download", "-r", train_run,
         "-a", "model/hf_format", "-d", local],
        check=True, capture_output=True, text=True,
    )
    dirs = sorted(
        (d for d in os.listdir(f"{local}/hf_format")
         if os.path.isdir(f"{local}/hf_format/{d}")),
    )
    model_dir = f"{local}/hf_format/{dirs[-1]}"
    subprocess.run(
        ["python3", "/amortized/patch_model_config.py", model_dir],
        check=True, capture_output=True, text=True,
    )
    return model_dir


def fingerprint_for(model_dir: str, served_name: str) -> str:
    # Reproduce the job's serve argv; _dump_resolved_args reads sys.argv.
    sys.argv = [
        "/amortized/serve_vllm.py", "serve", model_dir,
        "--served-model-name", served_name,
        "--port", "8000",
        "--gpu-memory-utilization", "0.9",  # denylisted — value irrelevant
    ]
    os.environ["VLLM_RESOLVED_ARGS_JSON"] = "/amortized/work/resolved.json"
    sys.path.insert(0, "/amortized")
    import serve_vllm
    serve_vllm._dump_resolved_args()
    with open("/amortized/work/resolved.json") as f:
        resolved = json.load(f)
    return json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)


def set_tag(run_id: str, value: str) -> None:
    req = urllib.request.Request(
        f"{TRACKING}/api/2.0/mlflow/runs/set-tag",
        data=json.dumps({"run_id": run_id, "key": "resolved_vllm_args",
                         "value": value}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=15)


def main() -> int:
    for train_run, hub_model, served_name, eval_runs in COMBOS:
        print(f"== {served_name} -> {len(eval_runs)} run(s)")
        model_dir = resolve_model_dir(train_run, hub_model)
        print(f"   model dir: {model_dir}")
        payload = fingerprint_for(model_dir, served_name)
        print(f"   fingerprint: {len(payload)} bytes")
        for rid in eval_runs:
            set_tag(rid, payload)
            print(f"   tagged {rid[:8]}")
    print("migration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
