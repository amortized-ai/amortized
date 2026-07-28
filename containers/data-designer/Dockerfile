FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache data-designer pandas mlflow boto3
RUN echo "y" | data-designer plugin install retrieval-sdg

COPY upload_to_mlflow.py /usr/local/bin/upload_to_mlflow.py

RUN mkdir -p /.data-designer /amortized/work && chmod 777 /.data-designer /amortized/work
RUN cat > /.data-designer/model_providers.yaml << 'EOF'
providers:
- name: gateway
  endpoint: http://mlflow.amortized.svc.cluster.local:5000/gateway/mlflow/v1
  provider_type: openai
  api_key: DD_API_KEY
EOF
ENV DATA_DESIGNER_HOME=/.data-designer

WORKDIR /amortized
ENTRYPOINT ["data-designer"]
