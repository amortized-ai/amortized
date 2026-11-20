"""Model-liveness check: probe the LLM endpoints a job will call before it is created.

Self-contained — it imports nothing else from ``amortized`` — so it runs both in-process in the
server and as the standalone service in its own image (containers/model-check).
"""
