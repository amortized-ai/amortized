# Evaluation container
# Used for eval jobs — runs judge models against generated/training data
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install asynth (includes litellm, mlflow, boto3, s3fs, etc.)
# google-cloud-aiplatform provides the vertexai SDK that litellm needs
# when calling Vertex AI models (e.g. vertex_ai/claude-haiku-4-5-20251001)
RUN pip install --no-cache-dir \
    "asynth>=0.1.10" \
    "google-cloud-aiplatform>=1.38"

WORKDIR /app
CMD ["asynth"]
