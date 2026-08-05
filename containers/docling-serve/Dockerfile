FROM ghcr.io/docling-project/docling-serve:latest

USER root
RUN pip install --no-cache-dir "chonkie[api]>=1.7.0" greenlet aiosqlite sqlalchemy
USER 1000

COPY entrypoint.sh /entrypoint.sh

EXPOSE 5001 8900

ENTRYPOINT ["/entrypoint.sh"]
