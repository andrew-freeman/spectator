"""State tracking utilities for the hierarchical reasoning system."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


class StateManager:
    """Maintains shared state and reasoning logs for the API layer."""

    def __init__(self, initial_state: Dict[str, Any] | None = None):
        self._state: Dict[str, Any] = initial_state or {}
        self._history: List[Dict[str, Any]] = []
        self._cycle_index: int = 0

    @property
    def cycle_index(self) -> int:
        return self._cycle_index

    def read(self) -> Dict[str, Any]:
        return deepcopy(self._state)

    def update(self, delta: Dict[str, Any]) -> Dict[str, Any]:
        self._state.update(delta)
        return self.read()

    def log_cycle(self, payload: Dict[str, Any]) -> None:
        record = {"cycle": self._cycle_index, **payload}
        self._history.append(record)
        self._cycle_index += 1

    def history(self) -> List[Dict[str, Any]]:
        return deepcopy(self._history)

    def update_last_cycle(self, delta: Dict[str, Any]) -> None:
        if not self._history:
            return
        self._history[-1].update(delta)


__all__ = ["StateManager"]
