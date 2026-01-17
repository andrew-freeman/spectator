from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class Backend(Protocol):
    def complete(self, prompt: str, params: dict[str, Any] | None = None) -> str:
        ...


_BACKENDS: dict[str, Callable[..., Backend]] = {}


def register_backend(name: str, factory: Callable[..., Backend]) -> None:
    key = name.lower()
    if key in _BACKENDS:
        raise ValueError(f"Backend '{name}' is already registered")
    _BACKENDS[key] = factory


def get_backend(name: str, **kwargs: Any) -> Backend:
    key = name.lower()
    factory = _BACKENDS.get(key)
    if factory is None:
        available = ", ".join(list_backends())
        raise ValueError(f"Unknown backend '{name}'. Available backends: {available}")
    return factory(**kwargs)


def list_backends() -> list[str]:
    return sorted(_BACKENDS)
