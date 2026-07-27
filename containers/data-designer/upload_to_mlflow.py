"""Upload Data Designer output to MLflow artifact store."""

import os
import sys

import mlflow
import pandas as pd


def main() -> None:
    dataset_dir = sys.argv[1]
    proc_subdir = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ""

    if proc_subdir:
        upload_dir = os.path.join(dataset_dir, proc_subdir)
    else:
        upload_dir = os.path.join(dataset_dir, "parquet-files")

    if not os.path.isdir(upload_dir):
        print(f"ERROR: {upload_dir} not found")
        sys.exit(1)

    parquet_files = [
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir)
        if f.endswith(".parquet")
    ]
    if not parquet_files:
        print(f"ERROR: no parquet files in {upload_dir}")
        sys.exit(1)

    df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    jsonl_path = os.path.join("/tmp", "generated_data.jsonl")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"Converted {len(df)} records from parquet to JSONL")

    with mlflow.start_run() as run:
        mlflow.log_artifact(jsonl_path, "generated_data")
        print(f"AMORTIZED_MLFLOW_RUN_ID={run.info.run_id}")
        print(f"Uploaded to MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
