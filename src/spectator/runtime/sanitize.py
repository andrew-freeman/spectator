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
_TOOLS_BLOCK_PATTERN = re.compile(
    f"{re.escape(TOOLS_START)}.*?{re.escape(TOOLS_END)}",
    re.DOTALL,
)
_NOTES_BLOCK_PATTERN = re.compile(
    f"{re.escape(NOTES_START)}.*?{re.escape(NOTES_END)}",
    re.DOTALL,
)

_REASONING_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<<<THOUGHTS>>>.*?<<<END_THOUGHTS>>>", re.DOTALL),
    re.compile(r"=== REASONING ===.*?=== END REASONING ===", re.DOTALL),
]

_SCAFFOLD_HEADERS = {
    "HISTORY:": "HISTORY",
    "HISTORY_JSON:": "HISTORY",
    "STATE:": "STATE",
    "UPSTREAM:": "UPSTREAM",
    "USER:": "USER",
    "TOOL_RESULTS:": "TOOL_RESULTS",
    "reflection:": "ROLE_TRANSCRIPT",
    "planner:": "ROLE_TRANSCRIPT",
    "critic:": "ROLE_TRANSCRIPT",
    "assistant:": "ROLE_TRANSCRIPT",
}
_RETRIEVED_START = "=== RETRIEVED_MEMORY ==="
_RETRIEVED_END = "=== END_RETRIEVED_MEMORY ==="
_RETRIEVAL_START = "=== RETRIEVAL ==="
_RETRIEVAL_END = "=== END RETRIEVAL ==="
_RETRIEVAL_BLOCK_PATTERN = re.compile(
    f"{re.escape(_RETRIEVED_START)}.*?{re.escape(_RETRIEVED_END)}"
    f"|{re.escape(_RETRIEVAL_START)}.*?{re.escape(_RETRIEVAL_END)}",
    re.DOTALL,
)


def _strip_reasoning_wrappers(text: str) -> str:
    sanitized = text
    for pattern in _REASONING_PATTERNS:
        sanitized = pattern.sub("", sanitized)
    return sanitized


def _strip_leading_scaffolding(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    working = text
    while True:
        stripped = working.lstrip()
        if not stripped:
            return "", removed
        if stripped.startswith((_RETRIEVED_START, _RETRIEVAL_START)):
            if stripped.startswith(_RETRIEVAL_START):
                end_marker = _RETRIEVAL_END
            else:
                end_marker = _RETRIEVED_END
            end_index = stripped.find(end_marker)
            cut_index = end_index + len(end_marker) if end_index != -1 else len(stripped)
            working = stripped[cut_index:]
            if "RETRIEVED_MEMORY" not in removed:
                removed.append("RETRIEVED_MEMORY")
            continue
        matched = False
        for header, label in _SCAFFOLD_HEADERS.items():
            if stripped.startswith(header):
                block_end = stripped.find("\n\n")
                working = stripped[block_end + 2 :] if block_end != -1 else ""
                if label not in removed:
                    removed.append(label)
                matched = True
                break
        if not matched:
            return working, removed


def _strip_trailing_scaffolding(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    working = text
    while True:
        stripped = working.rstrip()
        if not stripped:
            return "", removed
        if stripped.endswith((_RETRIEVED_END, _RETRIEVAL_END)):
            if stripped.endswith(_RETRIEVAL_END):
                start_marker = _RETRIEVAL_START
                end_marker = _RETRIEVAL_END
            else:
                start_marker = _RETRIEVED_START
                end_marker = _RETRIEVED_END
            start_index = stripped.rfind(start_marker)
            if start_index != -1:
                working = stripped[:start_index]
                if "RETRIEVED_MEMORY" not in removed:
                    removed.append("RETRIEVED_MEMORY")
                continue
        last_break = stripped.rfind("\n\n")
        if last_break == -1:
            last_block = stripped
            prefix = ""
        else:
            last_block = stripped[last_break + 2 :]
            prefix = stripped[:last_break]
        last_block_stripped = last_block.lstrip()
        matched = False
        for header, label in _SCAFFOLD_HEADERS.items():
            if last_block_stripped.startswith(header):
                working = prefix
                if label not in removed:
                    removed.append(label)
                matched = True
                break
        if not matched:
            return working, removed


def _strip_dangling_markers(text: str) -> tuple[str, bool]:
    """Remove stray tool/notes markers that are not part of protected blocks."""
    sanitized = text
    removed = False
    for marker in (TOOLS_START, TOOLS_END, NOTES_START, NOTES_END):
        if marker in sanitized:
            sanitized = sanitized.replace(marker, "")
            removed = True
    return sanitized, removed


def _strip_tool_notes_blocks(text: str) -> tuple[str, list[str]]:
    sanitized = text
    removed: list[str] = []
    if _TOOLS_BLOCK_PATTERN.search(sanitized):
        sanitized = _TOOLS_BLOCK_PATTERN.sub("", sanitized)
        removed.append("TOOL_BLOCK_STRIPPED")
    if _NOTES_BLOCK_PATTERN.search(sanitized):
        sanitized = _NOTES_BLOCK_PATTERN.sub("", sanitized)
        removed.append("NOTES_BLOCK_STRIPPED")
    sanitized, stripped_markers = _strip_dangling_markers(sanitized)
    if stripped_markers:
        removed.append("MARKER_POLLUTION")
    return sanitized, removed


def _strip_retrieval_blocks(text: str) -> tuple[str, bool]:
    if _RETRIEVAL_BLOCK_PATTERN.search(text):
        return _RETRIEVAL_BLOCK_PATTERN.sub("", text), True
    return text, False


def sanitize_visible_text_with_report(text: str) -> tuple[str, list[str], bool]:
    if not text:
        return text, [], False

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
    sanitized, leading_removed = _strip_leading_scaffolding(sanitized)
    sanitized, trailing_removed = _strip_trailing_scaffolding(sanitized)
    sanitized, retrieval_removed = _strip_retrieval_blocks(sanitized)
    sanitized, stripped_markers = _strip_dangling_markers(sanitized)
    for placeholder, original in placeholders.items():
        sanitized = sanitized.replace(placeholder, original)
    sanitized, block_removed = _strip_tool_notes_blocks(sanitized)
    removed = []
    for label in (*leading_removed, *trailing_removed):
        if label not in removed:
            removed.append(label)
    if retrieval_removed and "RETRIEVED_MEMORY" not in removed:
        removed.append("RETRIEVED_MEMORY")
    if stripped_markers and "MARKER_POLLUTION" not in removed:
        removed.append("MARKER_POLLUTION")
    for label in block_removed:
        if label not in removed:
            removed.append(label)
    if not sanitized.strip():
        return "...", removed, True
    return sanitized, removed, False


def sanitize_visible_text(text: str) -> str:
    """
    Remove prompt scaffolding and other internal-only sections from user-visible output.
    NOTE: tool/notes markers are removed from the final visible output.
    """
    sanitized, _removed, _empty = sanitize_visible_text_with_report(text)
    return sanitized
