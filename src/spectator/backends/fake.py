from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass(slots=True)
class FakeBackend:
    responses: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return ""

    def extend_responses(self, responses: Iterable[str]) -> None:
        self.responses.extend(responses)
