FROM ghcr.io/docling-project/docling-serve:latest

USER root
RUN pip install --no-cache-dir "chonkie[api]>=1.7.0" greenlet aiosqlite sqlalchemy tiktoken
ENV XDG_CACHE_HOME=/opt/cache
RUN mkdir -p /opt/cache && \
    python3 -c "from chonkie import TokenChunker; TokenChunker(chunk_size=64, tokenizer='cl100k_base')" && \
    chmod -R a+rw /opt/cache
USER 1000

COPY entrypoint.sh /entrypoint.sh

EXPOSE 5001 8900

ENTRYPOINT ["/entrypoint.sh"]
