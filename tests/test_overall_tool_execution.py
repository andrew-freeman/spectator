import json
from pathlib import Path

from spectator.backends.fake import FakeBackend
from spectator.core.tracing import TraceWriter
from spectator.core.types import Checkpoint, State
from spectator.runtime.pipeline import RoleSpec, run_pipeline
from spectator.tools.executor import ToolExecutor
from spectator.tools.fs_tools import list_dir_handler
from spectator.tools.registry import ToolRegistry
from spectator.tools.settings import default_tool_settings
from spectator.tools.shell_tool import shell_exec_handler


def _http_stub_handler(args: dict[str, object], _context) -> dict[str, object]:
    # Use a deterministic stub to avoid network access in tests.
    url = args.get("url")
    return {"url": url, "status": 200, "text": "stubbed http payload", "cache_hit": False}


def _build_executor(root: Path) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register("fs.list_dir", list_dir_handler(root))
    registry.register("shell.exec", shell_exec_handler(root))
    registry.register("http.get", _http_stub_handler)
    settings = default_tool_settings(root)
    return ToolExecutor(root, registry, settings)


def _load_events(path: Path) -> list[dict[str, object]]:
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in raw if line.strip()]


def _find_events(events: list[dict[str, object]], kind: str) -> list[dict[str, object]]:
    return [event for event in events if event.get("kind") == kind]


def _assert_tool_events(
    events: list[dict[str, object]],
    tool: str,
    expected_args: dict[str, object],
) -> None:
    starts = [event for event in events if event.get("kind") == "tool_start"]
    dones = [event for event in events if event.get("kind") == "tool_done"]
    assert any(start.get("data", {}).get("tool") == tool for start in starts)
    assert any(done.get("data", {}).get("tool") == tool for done in dones)
    for start in starts:
        if start.get("data", {}).get("tool") == tool:
            assert start.get("data", {}).get("args") == expected_args
            break
    start_index = next(
        idx for idx, event in enumerate(events)
        if event.get("kind") == "tool_start" and event.get("data", {}).get("tool") == tool
    )
    done_index = next(
        idx for idx, event in enumerate(events)
        if event.get("kind") == "tool_done" and event.get("data", {}).get("tool") == tool
    )
    assert start_index < done_index


def _assert_no_tool_json_leak(visible_output: str) -> None:
    lowered = visible_output.lower()
    assert "fs." not in lowered
    assert "shell." not in lowered
    assert "http." not in lowered
    assert "{\"name\"" not in lowered
    assert "{\"tool\"" not in lowered


def test_tool_execution_fs_fake_backend(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "hello.txt").write_text("hello", encoding="utf-8")
    executor = _build_executor(sandbox)

    response_1 = "{\"name\":\"fs.list_dir\",\"arguments\":{\"path\":\".\"}}"
    response_2 = "Listed sandbox entries: {{TOOL_OUTPUT}}"

    backend = FakeBackend()
    backend.set_role_responses("governor", [response_1, response_2])
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]
    tracer = TraceWriter("tool-fs", base_dir=tmp_path / "traces")

    checkpoint = Checkpoint(session_id="tool-fs", revision=0, updated_ts=0.0, state=State())
    final_text, _results, _checkpoint = run_pipeline(
        checkpoint,
        "List the sandbox contents.",
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    events = _load_events(tracer.path)
    _assert_tool_events(events, "fs.list_dir", {"path": "."})

    assert len(backend.calls) == 2
    assert "TOOL_RESULTS:" in backend.calls[1]["prompt"]
    assert "entries" in backend.calls[1]["prompt"]
    assert "hello.txt" in backend.calls[1]["prompt"]

    visible_events = _find_events(events, "visible_response")
    visible_output = visible_events[-1]["data"]["visible_response"]
    _assert_no_tool_json_leak(visible_output)
    assert "hello.txt" in final_text


def test_tool_execution_shell_fake_backend(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    executor = _build_executor(sandbox)

    tool_calls = [{"id": "call-1", "tool": "shell.exec", "args": {"cmd": "echo hello-from-shell"}}]
    response_1 = (
        "Run shell.\n"
        "<<<TOOL_CALLS_JSON>>>\n"
        f"{json.dumps(tool_calls)}\n"
        "<<<END_TOOL_CALLS_JSON>>>\n"
    )
    response_2 = "Shell output captured: {{TOOL_OUTPUT}}"

    backend = FakeBackend()
    backend.set_role_responses("governor", [response_1, response_2])
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]
    tracer = TraceWriter("tool-shell", base_dir=tmp_path / "traces")

    checkpoint = Checkpoint(session_id="tool-shell", revision=0, updated_ts=0.0, state=State())
    final_text, _results, _checkpoint = run_pipeline(
        checkpoint,
        "Run a safe shell command.",
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    events = _load_events(tracer.path)
    _assert_tool_events(events, "shell.exec", {"cmd": "echo hello-from-shell"})

    assert len(backend.calls) == 2
    assert "TOOL_RESULTS:" in backend.calls[1]["prompt"]
    assert "hello-from-shell" in backend.calls[1]["prompt"]

    visible_events = _find_events(events, "visible_response")
    visible_output = visible_events[-1]["data"]["visible_response"]
    _assert_no_tool_json_leak(visible_output)
    assert "hello-from-shell" in final_text


def test_tool_execution_http_fake_backend(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    executor = _build_executor(sandbox)

    response_1 = "{\"tool\":\"http.get\",\"args\":{\"url\":\"https://example.invalid\"}}"
    response_2 = "HTTP response summarized: {{TOOL_OUTPUT}}"

    backend = FakeBackend()
    backend.set_role_responses("governor", [response_1, response_2])
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]
    tracer = TraceWriter("tool-http", base_dir=tmp_path / "traces")

    state = State()
    state.capabilities_granted = ["net"]
    checkpoint = Checkpoint(session_id="tool-http", revision=0, updated_ts=0.0, state=state)
    final_text, _results, _checkpoint = run_pipeline(
        checkpoint,
        "Fetch the HTTP response.",
        roles,
        backend,
        tool_executor=executor,
        tracer=tracer,
    )

    events = _load_events(tracer.path)
    _assert_tool_events(events, "http.get", {"url": "https://example.invalid"})
    done_events = _find_events(events, "tool_done")
    http_done = next(
        event for event in done_events if event.get("data", {}).get("tool") == "http.get"
    )
    assert http_done["data"]["url"] == "https://example.invalid"
    assert http_done["data"]["cache_hit"] is False

    assert len(backend.calls) == 2
    assert "TOOL_RESULTS:" in backend.calls[1]["prompt"]
    assert "stubbed http payload" in backend.calls[1]["prompt"]

    visible_events = _find_events(events, "visible_response")
    visible_output = visible_events[-1]["data"]["visible_response"]
    _assert_no_tool_json_leak(visible_output)
    assert "stubbed http payload" in final_text
