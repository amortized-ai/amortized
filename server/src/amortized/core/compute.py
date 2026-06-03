"""Compute backend registry — no HTTP imports."""

from __future__ import annotations

from amortized.backends import ComputeBackend

_backends: dict[str, ComputeBackend] = {}


def register_backend(backend: ComputeBackend) -> None:
    _backends[backend.name] = backend


def get_backend(name: str) -> ComputeBackend:
    try:
        return _backends[name]
    except KeyError:
        raise KeyError(f"Unknown compute backend: {name!r}") from None


def list_backends() -> list[dict[str, object]]:
    return [
        {
            "name": b.name,
            "capabilities": sorted(c.value for c in b.capabilities()),
        }
        for b in _backends.values()
    ]


def reset() -> None:
    _backends.clear()
