import json
from pathlib import Path

from spectator.analysis.autopsy import autopsy_from_trace


def _write_trace(path: Path, events: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(event) for event in events) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_autopsy_detects_tools_and_sanitizer_warning(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    events = [
        {
            "ts": 1.0,
            "kind": "llm_req",
            "data": {"role": "governor", "prompt": "USER:\nhello"},
        },
        {
            "ts": 2.0,
            "kind": "tool_start",
            "data": {"role": "governor", "id": "t1", "tool": "fs.list_dir", "args": {"path": "."}},
        },
        {
            "ts": 3.0,
            "kind": "tool_done",
            "data": {
                "role": "governor",
                "id": "t1",
                "tool": "fs.list_dir",
                "ok": True,
                "error": None,
                "duration_ms": 12.3,
                "args": {"path": "."},
            },
        },
        {
            "ts": 4.0,
            "kind": "sanitize_warning",
            "data": {"role": "governor", "message": "visible output empty after sanitization"},
        },
        {
            "ts": 5.0,
            "kind": "llm_done",
            "data": {"role": "governor", "response": "All set."},
        },
        {
            "ts": 6.0,
            "kind": "visible_response",
            "data": {"role": "governor", "visible_response": "All set."},
        },
    ]
    _write_trace(trace_path, events)

    report = autopsy_from_trace(trace_path)

    assert report["tools"]
    assert report["tools"][0]["tool"] == "fs.list_dir"
    assert report["sanitizer"]["warnings"]
    assert report["recommendations"]


def test_autopsy_flags_visible_tool_json_leak(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    events = [
        {
            "ts": 1.0,
            "kind": "visible_response",
            "data": {
                "role": "governor",
                "visible_response": "{\"name\":\"fs.list_dir\",\"arguments\":{}}",
            },
        }
    ]
    _write_trace(trace_path, events)

    report = autopsy_from_trace(trace_path)

    anomalies = report["anomalies"]
    assert any(item["code"] == "visible_tool_json_leak" for item in anomalies)
    assert report["recommendations"]
