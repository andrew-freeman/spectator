from __future__ import annotations

from typing import Iterable

from spectator.core.types import State

REQUEST_PREFIX = "request_permission:"
GRANT_PREFIX = "grant_permission:"


def _remove_value(items: Iterable[str], value: str) -> list[str]:
    return [item for item in items if item != value]


def normalize_capabilities(state: State) -> None:
    granted = set(state.capabilities_granted)
    if not granted:
        return
    state.capabilities_pending = [cap for cap in state.capabilities_pending if cap not in granted]


def request_permission(state: State, cap: str) -> bool:
    if cap in state.capabilities_granted:
        return False
    if cap in state.capabilities_pending:
        return False
    state.capabilities_pending.append(cap)
    return True


def grant_permission(state: State, cap: str) -> bool:
    changed = False
    if cap not in state.capabilities_granted:
        state.capabilities_granted.append(cap)
        changed = True
    if cap in state.capabilities_pending:
        state.capabilities_pending = _remove_value(state.capabilities_pending, cap)
        changed = True
    return changed


def revoke_permission(state: State, cap: str) -> bool:
    if cap not in state.capabilities_granted:
        return False
    state.capabilities_granted = _remove_value(state.capabilities_granted, cap)
    return True


def clear_pending(state: State) -> bool:
    if not state.capabilities_pending:
        return False
    state.capabilities_pending = []
    return True


def apply_permission_actions(state: State, actions: Iterable[str]) -> dict[str, object]:
    before = {
        "granted": list(state.capabilities_granted),
        "pending": list(state.capabilities_pending),
    }
    applied: list[str] = []
    ignored: list[dict[str, str]] = []

    for action in actions:
        if action.startswith(REQUEST_PREFIX):
            cap = action[len(REQUEST_PREFIX) :]
            if not cap:
                ignored.append({"action": action, "reason": "empty_capability"})
                continue
            if request_permission(state, cap):
                applied.append(action)
        elif action.startswith(GRANT_PREFIX):
            cap = action[len(GRANT_PREFIX) :]
            if not cap:
                ignored.append({"action": action, "reason": "empty_capability"})
                continue
            if grant_permission(state, cap):
                applied.append(action)
        else:
            ignored.append({"action": action, "reason": "unknown_action"})

    normalize_capabilities(state)

    after = {
        "granted": list(state.capabilities_granted),
        "pending": list(state.capabilities_pending),
    }
    return {"before": before, "after": after, "applied": applied, "ignored": ignored}
