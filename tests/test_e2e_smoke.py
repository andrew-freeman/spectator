from __future__ import annotations

import json
from pathlib import Path

from spectator.backends.fake import FakeBackend
from spectator.runtime import controller
from spectator.runtime.tool_calls import END_MARKER, START_MARKER


def test_e2e_smoke_controller(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    sandbox_root = base_dir / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    (sandbox_root / "hello.txt").write_text("hello", encoding="utf-8")

    tool_calls = [{"id": "t1", "tool": "fs.read_text", "args": {"path": "hello.txt"}}]
    response_1 = (
        "Need to read the file.\n"
        f"{START_MARKER}\n"
        f"{json.dumps(tool_calls)}\n"
        f"{END_MARKER}\n"
    )
    response_2 = "Smoke run complete."

    backend = FakeBackend()
    backend.extend_role_responses("reflection", ["Noted."])
    backend.extend_role_responses("planner", ["Plan drafted."])
    backend.extend_role_responses("critic", ["Looks good."])
    backend.extend_role_responses("governor", [response_1, response_2])

    final_text = controller.run_turn("smoke-1", "hi", backend, base_dir=base_dir)

    assert final_text == "Smoke run complete."

    checkpoint_path = base_dir / "checkpoints" / "smoke-1.json"
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "smoke-1"
    assert len(payload["recent_messages"]) == 2

    trace_path = base_dir / "traces" / "smoke-1__rev-1.jsonl"
    assert trace_path.exists()
    trace_kinds = {
        json.loads(line)["kind"] for line in trace_path.read_text(encoding="utf-8").splitlines()
    }
    for required in {"llm_req", "llm_done", "tool_plan", "tool_start", "tool_done"}:
        assert required in trace_kinds
