from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List


@dataclass(slots=True)
class FakeBackend:
    responses: List[str] = field(default_factory=list)
    role_responses: dict[str, List[str]] = field(default_factory=dict)
    calls: List[dict[str, Any]] = field(default_factory=list)

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
