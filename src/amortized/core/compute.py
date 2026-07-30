"""Compute backend registry — no HTTP imports."""

from __future__ import annotations

import logging

from amortized.backends import Capability, ComputeBackend

logger = logging.getLogger("amortized.core.compute")

_backends: dict[str, ComputeBackend] = {}


class MissingCapabilityError(Exception):
    def __init__(self, backend_name: str, missing: set[Capability]) -> None:
        self.backend_name = backend_name
        self.missing = missing
        names = ", ".join(sorted(c.value for c in missing))
        super().__init__(f"Backend {backend_name!r} lacks required capabilities: {names}")


def register_backend(backend: ComputeBackend) -> None:
    caps = sorted(c.value for c in backend.capabilities())
    logger.info("Registered backend %r capabilities=%s", backend.name, ",".join(caps))
    _backends[backend.name] = backend


def get_backend(name: str) -> ComputeBackend:
    try:
        return _backends[name]
    except KeyError:
        logger.error("Backend %r not found, registered=%s", name, ",".join(_backends))
        raise KeyError(f"Unknown compute backend: {name!r}") from None


def check_capabilities(backend: ComputeBackend, required: set[Capability]) -> None:
    available = backend.capabilities()
    missing = required - available
    if missing:
        names = ",".join(c.value for c in missing)
        logger.warning("Backend %r missing capabilities: %s", backend.name, names)
        raise MissingCapabilityError(backend.name, missing)


def get_all_backends() -> dict[str, ComputeBackend]:
    return dict(_backends)


def list_backends() -> list[dict[str, object]]:
    return [
        {
            "name": b.name,
            "capabilities": sorted(c.value for c in b.capabilities()),
        }
        for b in _backends.values()
    ]


def unregister_backend(name: str) -> bool:
    removed = _backends.pop(name, None) is not None
    if removed:
        logger.info("Unregistered backend %r", name)
    return removed


def reset() -> None:
    _backends.clear()
