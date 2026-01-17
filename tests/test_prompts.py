from __future__ import annotations

from pathlib import Path

import pytest

from spectator.backends.fake import FakeBackend
from spectator.prompts import get_role_prompt, load_prompt
from spectator.runtime import controller


def test_load_prompt_reads_role_file() -> None:
    content = load_prompt("roles/reflection.txt")
    assert "Reflect on the request." in content


@pytest.mark.parametrize("role", ["reflection", "planner", "critic", "governor"])
def test_get_role_prompt_returns_nonempty(role: str) -> None:
    content = get_role_prompt(role)
    assert content.strip()


def test_controller_default_roles_use_file_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_roles = []

    def fake_run_pipeline(
        checkpoint, user_text, roles, backend, tool_executor=None, tracer=None
    ):
        captured_roles.extend(list(roles))
        return "final", [], checkpoint

    monkeypatch.setattr(controller, "run_pipeline", fake_run_pipeline)
    backend = FakeBackend()

    controller.run_turn("session-prompts", "hello", backend, base_dir=tmp_path)

    assert captured_roles
    expected = {
        "reflection": get_role_prompt("reflection"),
        "planner": get_role_prompt("planner"),
        "critic": get_role_prompt("critic"),
        "governor": get_role_prompt("governor"),
    }
    for role in captured_roles:
        assert role.system_prompt == expected[role.name]
