from __future__ import annotations

import re

from spectator.tools import build_default_registry


def test_system_time_tool_registered(tmp_path) -> None:
    registry, _executor = build_default_registry(tmp_path)
    assert registry.get("system.time") is not None


def test_system_time_tool_output_format(tmp_path) -> None:
    registry, _executor = build_default_registry(tmp_path)
    tool = registry.get("system.time")
    assert tool is not None
    payload = tool.handler({})
    assert isinstance(payload, dict)
    assert isinstance(payload.get("utc"), str)
    assert isinstance(payload.get("local"), str)
    assert isinstance(payload.get("epoch_s"), (int, float))
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", payload["utc"])
