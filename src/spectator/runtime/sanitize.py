from __future__ import annotations

import re

from spectator.runtime.notes import END_MARKER as NOTES_END
from spectator.runtime.notes import START_MARKER as NOTES_START
from spectator.runtime.tool_calls import END_MARKER as TOOLS_END
from spectator.runtime.tool_calls import START_MARKER as TOOLS_START


_PROTECTED_PATTERN = re.compile(
    f"{re.escape(NOTES_START)}.*?{re.escape(NOTES_END)}"
    f"|{re.escape(TOOLS_START)}.*?{re.escape(TOOLS_END)}",
    re.DOTALL,
)

_REASONING_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<<<THOUGHTS>>>.*?<<<END_THOUGHTS>>>", re.DOTALL),
    re.compile(r"=== REASONING ===.*?=== END REASONING ===", re.DOTALL),
]


def _strip_reasoning_wrappers(text: str) -> str:
    sanitized = text
    for pattern in _REASONING_PATTERNS:
        sanitized = pattern.sub("", sanitized)
    return sanitized


def sanitize_visible_text(text: str) -> str:
    if not text:
        return text

    placeholders: dict[str, str] = {}
    segments: list[str] = []
    last_index = 0
    for idx, match in enumerate(_PROTECTED_PATTERN.finditer(text)):
        placeholder = f"<<<SPECTATOR_BLOCK_{idx}>>>"
        placeholders[placeholder] = match.group(0)
        segments.append(text[last_index : match.start()])
        segments.append(placeholder)
        last_index = match.end()
    segments.append(text[last_index:])

    protected = "".join(segments)
    sanitized = _strip_reasoning_wrappers(protected)
    for placeholder, original in placeholders.items():
        sanitized = sanitized.replace(placeholder, original)
    return sanitized
