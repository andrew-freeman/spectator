from spectator.core.types import State
from spectator.runtime.capabilities import (
    apply_permission_actions,
    clear_pending,
    grant_permission,
    revoke_permission,
)


def test_request_adds_pending() -> None:
    state = State()

    apply_permission_actions(state, ["request_permission:net"])

    assert state.capabilities_granted == []
    assert state.capabilities_pending == ["net"]


def test_request_noop_when_granted() -> None:
    state = State(capabilities_granted=["net"])

    apply_permission_actions(state, ["request_permission:net"])

    assert state.capabilities_granted == ["net"]
    assert state.capabilities_pending == []


def test_grant_moves_pending_to_granted() -> None:
    state = State(capabilities_pending=["net"])

    apply_permission_actions(state, ["grant_permission:net"])

    assert state.capabilities_granted == ["net"]
    assert state.capabilities_pending == []


def test_grant_idempotent() -> None:
    state = State(capabilities_granted=["net"], capabilities_pending=["net:example.com"])

    apply_permission_actions(state, ["grant_permission:net"])

    assert state.capabilities_granted == ["net"]
    assert state.capabilities_pending == ["net:example.com"]


def test_pending_never_contains_granted() -> None:
    state = State(capabilities_granted=["net"], capabilities_pending=["net", "net:example.com"])

    apply_permission_actions(state, ["request_permission:net:example.com"])

    assert state.capabilities_granted == ["net"]
    assert state.capabilities_pending == ["net:example.com"]


def test_clear_pending() -> None:
    state = State(capabilities_pending=["net", "net:example.com"])

    assert clear_pending(state)
    assert state.capabilities_pending == []


def test_revoke_removes_granted_without_pending() -> None:
    state = State(capabilities_granted=["net", "net:example.com"])

    assert revoke_permission(state, "net")
    assert state.capabilities_granted == ["net:example.com"]
    assert state.capabilities_pending == []


def test_grant_helper_clears_pending_match() -> None:
    state = State(capabilities_granted=["net"], capabilities_pending=["net"])

    assert grant_permission(state, "net")
    assert state.capabilities_granted == ["net"]
    assert state.capabilities_pending == []
