"""Upload Data Designer output to MLflow artifact store."""

import glob
import os
import sys

import mlflow


def main() -> None:
    dataset_dir = sys.argv[1]
    proc_subdir = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ""

    if proc_subdir:
        proc_path = os.path.join(dataset_dir, proc_subdir)
        files = glob.glob(os.path.join(proc_path, "*.parquet"))
    else:
        files = []

    if not files:
        files = glob.glob(os.path.join(dataset_dir, "*.parquet"))
    if not files:
        files = glob.glob(os.path.join(dataset_dir, "*.jsonl"))

    if not files:
        print("ERROR: no dataset files found")
        sys.exit(1)

    with mlflow.start_run() as run:
        for f in files:
            mlflow.log_artifact(f, "generated_data")
        print(f"AMORTIZED_MLFLOW_RUN_ID={run.info.run_id}")
        print(f"Uploaded {len(files)} file(s) to MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
