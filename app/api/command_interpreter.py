"""Natural language command interpreter for Spectator."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Protocol


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


PROMPT_TEMPLATE = (
    "You are the command interpreter for a cybernetic meta-cognitive system.\n"
    "Convert the USER message into a structured reasoning request.\n"
    "Do NOT hallucinate tools.\n"
    "Only produce valid JSON with: objectives, context, memory_snippets.\n"
    "USER: {user_block}\n"
)

QUERY_STARTERS = ("what", "why", "how", "when", "who")
QUERY_PHRASES = ("what is your", "previous", "history", "explain", "describe")
COMMAND_VERBS = (
    "set",
    "adjust",
    "apply",
    "change",
    "increase",
    "decrease",
    "run",
    "execute",
)


class CommandInterpreter:
    """Converts free-form instructions into structured reasoning objectives."""

    def __init__(self, client: SupportsGenerate):
        self._client = client

    def interpret(self, message: str) -> Dict[str, Any]:
        """Return structured objectives/context/memory for a user message."""

        mode = classify(message)
        if mode == "query":
            return {"objectives": [], "context": {}, "memory_snippets": []}

        user_block = json.dumps(message, ensure_ascii=False)
        prompt = PROMPT_TEMPLATE.format(user_block=user_block)
        raw = self._client.generate(prompt, stop=None)
        payload = _parse_json(raw)
        context = dict(payload.get("context") or {})
        context.setdefault("mode", "command")
        return {
            "objectives": _ensure_string_list(payload.get("objectives")),
            "context": context,
            "memory_snippets": _ensure_string_list(payload.get("memory_snippets")),
            "force_action": True,
        }


def classify(message: str) -> str:
    """Heuristic classifier that routes messages to query or command mode."""

    text = (message or "").strip().lower()
    if not text:
        return "query"

    for starter in QUERY_STARTERS:
        if text.startswith(starter):
            return "query"

    if any(phrase in text for phrase in QUERY_PHRASES):
        return "query"

    if any(verb in text for verb in COMMAND_VERBS):
        return "command"

    return "query"


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Command interpreter returned invalid JSON: {exc}: {raw!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Command interpreter returned non-object payload: {data!r}")
    return data


def _ensure_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


__all__ = ["CommandInterpreter", "classify"]

