from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List

from spectator.backends.registry import register_backend


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
            return self.role_responses[role].pop(0)
        if self.responses:
            return self.responses.pop(0)
        return ""

    def extend_responses(self, responses: Iterable[str]) -> None:
        self.responses.extend(responses)

    def extend_role_responses(self, role: str, responses: Iterable[str]) -> None:
        self.role_responses.setdefault(role, []).extend(responses)


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
