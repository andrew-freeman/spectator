from __future__ import annotations

from spectator.tools.registry import ToolRegistry


def test_tool_registry_register_and_lookup() -> None:
    registry = ToolRegistry()

    def handler(args: dict[str, str]) -> dict[str, str]:
        return args

    registry.register("fs.read_text", handler)

    spec = registry.get("fs.read_text")
    assert spec is not None
    assert spec.name == "fs.read_text"
    assert spec.handler is handler
    assert [tool.name for tool in registry.list_tools()] == ["fs.read_text"]
