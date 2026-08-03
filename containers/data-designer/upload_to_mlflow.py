"""Upload Data Designer output to MLflow artifact store."""

import os
import sys

import mlflow


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

    raw_dir = os.path.join(dataset_dir, "parquet-files")

    with mlflow.start_run() as run:
        mlflow.log_artifacts(upload_dir, "generated_data")
        if proc_subdir and os.path.isdir(raw_dir) and raw_dir != upload_dir:
            mlflow.log_artifacts(raw_dir, "raw_data")
        print(f"AMORTIZED_MLFLOW_RUN_ID={run.info.run_id}")
        print(f"Uploaded {upload_dir} to MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
