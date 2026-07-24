FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache data-designer pandas
RUN echo "y" | data-designer plugin install retrieval-sdg

RUN mkdir -p /.data-designer /tmp/dd-artifacts && chmod 777 /.data-designer /tmp/dd-artifacts
ENV DATA_DESIGNER_HOME=/.data-designer
