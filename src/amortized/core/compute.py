"""Compute backend registry — no HTTP imports."""

from __future__ import annotations

from amortized.backends import Capability, ComputeBackend

_backends: dict[str, ComputeBackend] = {}


class MissingCapabilityError(Exception):
    def __init__(self, backend_name: str, missing: set[Capability]) -> None:
        self.backend_name = backend_name
        self.missing = missing
        names = ", ".join(sorted(c.value for c in missing))
        super().__init__(f"Backend {backend_name!r} lacks required capabilities: {names}")


def register_backend(backend: ComputeBackend) -> None:
    _backends[backend.name] = backend


def get_backend(name: str) -> ComputeBackend:
    try:
        return _backends[name]
    except KeyError:
        raise KeyError(f"Unknown compute backend: {name!r}") from None


def check_capabilities(backend: ComputeBackend, required: set[Capability]) -> None:
    available = backend.capabilities()
    missing = required - available
    if missing:
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
    return _backends.pop(name, None) is not None


def reset() -> None:
    _backends.clear()
