from __future__ import annotations

from spectator.backends.llama_server import LlamaServerBackend
from spectator.prompts import load_prompt


def test_llama_backend_build_payload_injects_system_rules() -> None:
    backend = LlamaServerBackend()
    payload = backend._build_payload("hello", {})

    assert payload["messages"][0]["role"] == "system"
    assert load_prompt("system/llama_rules.txt") in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "hello"}


def test_llama_backend_build_payload_respects_messages_override() -> None:
    backend = LlamaServerBackend()
    custom_messages = [{"role": "system", "content": "custom"}]
    payload = backend._build_payload("hello", {"messages": custom_messages})

    assert payload["messages"] == custom_messages


def test_llama_backend_rules_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SPECTATOR_LLAMA_RULES_PROMPT", "system/test_llama_rules.txt")
    backend = LlamaServerBackend()
    payload = backend._build_payload("hello", {})

    assert payload["messages"][0]["role"] == "system"
    assert load_prompt("system/test_llama_rules.txt") in payload["messages"][0]["content"]
