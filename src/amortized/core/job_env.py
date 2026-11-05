"""The environment a job container receives from the control plane.

Shared by the worker (which builds the job's env) and the model-liveness check (which must
resolve an endpoint's credential exactly as the workload will), so the two can't diverge.
"""

from __future__ import annotations

import os

import amortized.config as config_mod


def forwarded_env() -> dict[str, str]:
    """Server env vars listed in ``settings.forward_env`` that are set and non-empty."""
    env: dict[str, str] = {}
    for name in config_mod.settings.forward_env:
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env
