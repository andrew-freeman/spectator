import json

from spectator.backends.fake import FakeBackend
from spectator.core.types import ChatMessage, Checkpoint, State
from spectator.runtime.controller import run_turn
from spectator.runtime.pipeline import RoleSpec, _format_history, run_pipeline


def test_format_history_bounds_messages_and_chars() -> None:
    messages = [
        ChatMessage(role="system", content="ignore"),
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="second"),
        ChatMessage(role="user", content="third"),
        ChatMessage(role="assistant", content="fourth"),
    ]

    formatted = _format_history(messages, max_messages=2, max_chars=2000)
    assert json.loads(formatted) == [
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
    ]

    truncated = _format_history(messages, max_messages=8, max_chars=10)
    assert len(truncated) <= 10
    json.loads(truncated)


def test_prompt_includes_history_for_previous_turn(tmp_path) -> None:
    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["r1", "r2"])
    backend.extend_role_responses("planner", ["p1", "p2"])
    backend.extend_role_responses("critic", ["c1", "c2"])
    backend.extend_role_responses("governor", ["g1", "g2"])

    run_turn("history-session", "Hello", backend=backend, base_dir=tmp_path)
    run_turn(
        "history-session",
        "What was my previous question?",
        backend=backend,
        base_dir=tmp_path,
    )

    governor_prompts = [
        call["prompt"] for call in backend.calls if call["params"].get("role") == "governor"
    ]
    assert "HISTORY_JSON:" in governor_prompts[-1]
    history_block = governor_prompts[-1].split("HISTORY_JSON:\n", 1)[1].split("\n\n", 1)[0]
    history_payload = json.loads(history_block)
    assert {"role": "user", "content": "Hello"} in history_payload


def test_identity_response_avoids_vendor_names() -> None:
    checkpoint = Checkpoint(session_id="id-1", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("governor", ["<think>I am OpenAI</think> I am Spectator."])

    roles = [RoleSpec(name="governor", system_prompt="Decide.")]

    final_text, _results, _updated = run_pipeline(checkpoint, "Who are you?", roles, backend)

    assert "Spectator" in final_text
    for vendor in ("Anthropic", "OpenAI", "Alibaba", "Qwen"):
        assert vendor not in final_text
