"""FastAPI application entry point."""

import logging
from datetime import UTC, datetime

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("amortized_runtime")

app = FastAPI(
    title="Amortized Runtime",
    description="AI model customization runtime API",
    version="0.1.0",
)


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
    }
