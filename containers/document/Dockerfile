FROM ghcr.io/docling-project/docling-serve:latest

USER root
RUN pip install --no-cache-dir "chonkie[api]>=1.7.0" tiktoken mlflow boto3 greenlet aiosqlite sqlalchemy
ENV XDG_CACHE_HOME=/opt/cache TORCH_HOME=/opt/cache/torch TORCHINDUCTOR_CACHE_DIR=/opt/cache/torch HOME=/opt/cache
RUN mkdir -p /opt/cache/torch && \
    python3 -c "from chonkie import TokenChunker; TokenChunker(chunk_size=64, tokenizer='cl100k_base')" && \
    chmod -R a+rw /opt/cache

COPY process_document.py /app/process_document.py
USER 1000

WORKDIR /app
