# Synthetic data generation container
# Used for SDG jobs — generates training data via teacher LLMs
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install asynth (includes litellm, mlflow, boto3, s3fs, etc.)
# google-cloud-aiplatform provides the vertexai SDK that litellm needs
# when calling Vertex AI models (e.g. vertex_ai/claude-haiku-4-5-20251001)
#
# Pin s3fs/fsspec to resolve datasets vs s3fs version conflict:
#   datasets 5.0.0 requires fsspec<=2026.4.0
#   s3fs 2026.6.0 requires fsspec>=2026.6.0
RUN pip install --no-cache-dir \
    "asynth>=0.1.10" \
    "google-cloud-aiplatform>=1.38" \
    "s3fs==2026.4.0" \
    "fsspec==2026.4.0"

WORKDIR /app
CMD ["asynth"]
