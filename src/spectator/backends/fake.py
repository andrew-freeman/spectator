from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List

from spectator.backends.registry import register_backend

_TOOL_RESULTS_MARKER = "TOOL_RESULTS:\n"


@dataclass(slots=True)
class FakeBackend:
    responses: List[str] = field(default_factory=list)
    role_responses: dict[str, List[str]] = field(default_factory=dict)
    calls: List[dict[str, Any]] = field(default_factory=list)
    supports_messages: bool = False

    def complete(self, prompt: str, params: dict[str, Any] | None = None) -> str:
        payload = {"prompt": prompt, "params": params or {}}
        self.calls.append(payload)
        role = payload["params"].get("role")
        if role and role in self.role_responses and self.role_responses[role]:
            response = self.role_responses[role].pop(0)
            return _render_response(response, prompt)
        if self.responses:
            response = self.responses.pop(0)
            return _render_response(response, prompt)
        return ""

    def extend_responses(self, responses: Iterable[str]) -> None:
        self.responses.extend(responses)

    def set_responses(self, responses: Iterable[str]) -> None:
        self.responses = list(responses)

    def extend_role_responses(self, role: str, responses: Iterable[str]) -> None:
        self.role_responses.setdefault(role, []).extend(responses)

    def set_role_responses(self, role: str, responses: Iterable[str]) -> None:
        self.role_responses[role] = list(responses)


def _render_response(response: str, prompt: str) -> str:
    if not isinstance(response, str):
        return response
    if "{{TOOL_OUTPUT}}" not in response:
        return response
    tool_output = _select_tool_output(_extract_tool_results(prompt))
    return response.replace("{{TOOL_OUTPUT}}", tool_output)


def _extract_tool_results(prompt: str) -> list[dict[str, Any]]:
    start = prompt.find(_TOOL_RESULTS_MARKER)
    if start == -1:
        return []
    tail = prompt[start + len(_TOOL_RESULTS_MARKER):]
    results: list[dict[str, Any]] = []
    for line in tail.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def _select_tool_output(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    output = results[0].get("output")
    if isinstance(output, dict):
        stdout = output.get("stdout")
        if isinstance(stdout, str):
            return stdout.strip()
        text = output.get("text")
        if isinstance(text, str):
            return text.strip()
        entries = output.get("entries")
        if isinstance(entries, list):
            return ", ".join(str(entry) for entry in entries)
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=True)


def _load_env_json_list(env_value: str) -> list[str]:
    data = json.loads(env_value)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("fake responses must be a JSON list of strings")
    return data


def _load_env_json_role_map(env_value: str) -> dict[str, list[str]]:
    data = json.loads(env_value)
    if not isinstance(data, dict):
        raise ValueError("fake role responses must be a JSON object")
    role_responses: dict[str, list[str]] = {}
    for role, responses in data.items():
        if not isinstance(role, str) or not isinstance(responses, list):
            raise ValueError("fake role responses must map role -> list[str]")
        if not all(isinstance(item, str) for item in responses):
            raise ValueError("fake role responses must map role -> list[str]")
        role_responses[role] = list(responses)
    return role_responses


def _factory(**_kwargs: Any) -> "FakeBackend":
    backend = FakeBackend()
    responses_json = os.getenv("SPECTATOR_FAKE_RESPONSES")
    if responses_json:
        backend.responses = _load_env_json_list(responses_json)
    role_responses_json = os.getenv("SPECTATOR_FAKE_ROLE_RESPONSES")
    if role_responses_json:
        backend.role_responses = _load_env_json_role_map(role_responses_json)
    return backend


register_backend("fake", _factory)
