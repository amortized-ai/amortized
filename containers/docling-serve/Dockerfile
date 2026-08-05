FROM ghcr.io/docling-project/docling-serve:latest

USER root
RUN pip install --no-cache-dir "chonkie[api]>=1.7.0" greenlet aiosqlite sqlalchemy tiktoken
RUN python3 -c "from chonkie import TokenChunker; TokenChunker(chunk_size=64, tokenizer='cl100k_base')"
RUN chmod -R a+r /root/.cache && cp -r /root/.cache/huggingface /tmp/hf_cache && chmod -R a+rw /tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
USER 1000

COPY entrypoint.sh /entrypoint.sh

EXPOSE 5001 8900

ENTRYPOINT ["/entrypoint.sh"]
