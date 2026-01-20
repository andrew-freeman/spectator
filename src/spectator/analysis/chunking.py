from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Chunk:
    id: str
    title: str
    strategy: str
    start_line: int
    end_line: int
    text: str


def chunk_file(
    path: str,
    text: str,
    strategy: str = "auto",
    max_chars: int = 40000,
    overlap_chars: int = 0,
) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    normalized = _normalize_newlines(text)
    if normalized == "":
        return []
    resolved_strategy = _resolve_strategy(path, strategy)
    if resolved_strategy == "headings":
        chunks = _chunk_by_headings(path, normalized, max_chars)
    elif resolved_strategy == "python_ast":
        chunks = _chunk_by_python_ast(path, normalized, max_chars)
    elif resolved_strategy == "log":
        chunks = _chunk_by_log(path, normalized, max_chars)
    elif resolved_strategy == "fixed":
        chunks = _chunk_fixed(path, normalized, max_chars, overlap_chars)
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'")
    for chunk in chunks:
        chunk.strategy = resolved_strategy
    return chunks


def _normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _resolve_strategy(path: str, strategy: str) -> str:
    lowered = (strategy or "auto").lower()
    if lowered != "auto":
        return lowered
    suffix = Path(path).suffix.lower()
    if suffix in {".log", ".jsonl", ".txt"}:
        return "log"
    if suffix in {".md", ".rst"}:
        return "headings"
    if suffix == ".py":
        return "python_ast"
    return "fixed"


def _chunk_by_headings(path: str, text: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    headings = _extract_headings(lines)
    sections: list[tuple[int, int, str]] = []
    if headings:
        first_line, _ = headings[0]
        if first_line > 1:
            sections.append((1, first_line - 1, "preamble"))
        for idx, (line_no, title) in enumerate(headings):
            next_line = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines) + 1
            sections.append((line_no, next_line - 1, title))
    else:
        sections.append((1, len(lines), "document"))
    chunks: list[Chunk] = []
    for start_line, end_line, title in sections:
        if end_line < start_line:
            continue
        section_text = "".join(lines[start_line - 1 : end_line])
        chunks.extend(
            _split_oversize(
                path,
                title,
                start_line,
                end_line,
                section_text,
                max_chars,
            )
        )
    return chunks


def _extract_headings(lines: list[str]) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    md_re = re.compile(r"^(#{1,6})\s+(.*)$")
    underline_re = re.compile(r"^[=\-]{3,}\s*$")
    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip("\n")
        match = md_re.match(line)
        if match:
            title = match.group(2).strip() or "heading"
            headings.append((idx + 1, title))
            idx += 1
            continue
        if idx + 1 < len(lines):
            underline = lines[idx + 1].rstrip("\n")
            if underline_re.match(underline) and line.strip():
                headings.append((idx + 1, line.strip()))
                idx += 2
                continue
        idx += 1
    return headings


def _chunk_by_python_ast(path: str, text: str, max_chars: int) -> list[Chunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _chunk_fixed(path, text, max_chars, overlap_chars=0)
    lines = text.splitlines(keepends=True)
    nodes: list[tuple[int, int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if getattr(node, "lineno", None) is None or getattr(node, "end_lineno", None) is None:
                return _chunk_fixed(path, text, max_chars, overlap_chars=0)
            title = _title_for_node(node)
            nodes.append((node.lineno, node.end_lineno, title))
    chunks: list[Chunk] = []
    if not nodes:
        return _split_oversize(path, "module", 1, len(lines), text, max_chars)
    first_start = nodes[0][0]
    if first_start > 1:
        pre_text = "".join(lines[: first_start - 1])
        chunks.extend(
            _split_oversize(path, "module", 1, first_start - 1, pre_text, max_chars)
        )
    for start_line, end_line, title in nodes:
        if end_line < start_line:
            continue
        section_text = "".join(lines[start_line - 1 : end_line])
        chunks.extend(
            _split_oversize(
                path,
                title,
                start_line,
                end_line,
                section_text,
                max_chars,
            )
        )
    return chunks


def _title_for_node(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if isinstance(node, ast.AsyncFunctionDef):
        return f"async def {node.name}"
    if isinstance(node, ast.FunctionDef):
        return f"def {node.name}"
    return "module"


def _chunk_fixed(
    path: str, text: str, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[Chunk] = []
    start_line = 1
    buf: list[str] = []
    buf_len = 0
    for idx, line in enumerate(lines, start=1):
        line_len = len(line)
        if line_len > max_chars:
            if buf:
                chunk_text = "".join(buf)
                end_line = idx - 1
                chunks.append(_build_chunk(path, "chunk", start_line, end_line, chunk_text))
                buf = []
                buf_len = 0
            chunks.extend(_split_long_line(path, "chunk", idx, line, max_chars))
            start_line = idx + 1
            continue
        if buf and buf_len + line_len > max_chars:
            chunk_text = "".join(buf)
            end_line = idx - 1
            chunks.append(_build_chunk(path, "chunk", start_line, end_line, chunk_text))
            overlap, overlap_lines = _compute_overlap(buf, overlap_chars)
            buf = overlap
            buf_len = sum(len(part) for part in buf)
            if overlap_lines:
                start_line = end_line - overlap_lines + 1
            else:
                start_line = idx
        buf.append(line)
        buf_len += line_len
    if buf:
        end_line = start_line + len(buf) - 1
        chunks.append(_build_chunk(path, "chunk", start_line, end_line, "".join(buf)))
    return chunks


def _compute_overlap(lines: list[str], overlap_chars: int) -> tuple[list[str], int]:
    if overlap_chars <= 0 or not lines:
        return [], 0
    total = 0
    overlap: list[str] = []
    for line in reversed(lines):
        if total + len(line) > overlap_chars and overlap:
            break
        overlap.append(line)
        total += len(line)
    overlap.reverse()
    return overlap, len(overlap)


def _split_oversize(
    path: str,
    title: str,
    start_line: int,
    end_line: int,
    text: str,
    max_chars: int,
) -> list[Chunk]:
    if len(text) <= max_chars:
        return [_build_chunk(path, title, start_line, end_line, text)]
    lines = text.splitlines(keepends=True)
    parts: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    part_start_line = start_line
    for idx, line in enumerate(lines, start=start_line):
        line_len = len(line)
        if line_len > max_chars:
            if buf:
                part_text = "".join(buf)
                part_end_line = idx - 1
                parts.append(_build_chunk(path, title, part_start_line, part_end_line, part_text))
                buf = []
                buf_len = 0
            parts.extend(_split_long_line(path, title, idx, line, max_chars))
            part_start_line = idx + 1
            continue
        if buf and buf_len + line_len > max_chars:
            part_text = "".join(buf)
            part_end_line = idx - 1
            parts.append(_build_chunk(path, title, part_start_line, part_end_line, part_text))
            buf = []
            buf_len = 0
            part_start_line = idx
        buf.append(line)
        buf_len += line_len
    if buf:
        part_end_line = part_start_line + len(buf) - 1
        parts.append(_build_chunk(path, title, part_start_line, part_end_line, "".join(buf)))
    if len(parts) == 1:
        return parts
    total_parts = len(parts)
    labeled: list[Chunk] = []
    for idx, part in enumerate(parts, start=1):
        labeled_title = f"{title} (part {idx}/{total_parts})"
        labeled.append(
            Chunk(
                id=_chunk_id(path, part.start_line, part.end_line, labeled_title),
                title=labeled_title,
                strategy=part.strategy,
                start_line=part.start_line,
                end_line=part.end_line,
                text=part.text,
            )
        )
    return labeled


def _build_chunk(
    path: str, title: str, start_line: int, end_line: int, text: str
) -> Chunk:
    return Chunk(
        id=_chunk_id(path, start_line, end_line, title),
        title=title,
        strategy="",
        start_line=start_line,
        end_line=end_line,
        text=text,
    )


def _chunk_id(path: str, start_line: int, end_line: int, title: str) -> str:
    payload = f"{path}:{start_line}:{end_line}:{title}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _split_long_line(
    path: str,
    title: str,
    line_no: int,
    line: str,
    max_chars: int,
) -> list[Chunk]:
    parts: list[Chunk] = []
    start = 0
    while start < len(line):
        end = min(start + max_chars, len(line))
        segment = line[start:end]
        parts.append(_build_chunk(path, title, line_no, line_no, segment))
        start = end
    return parts


def _chunk_by_log(path: str, text: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    tail_count = min(200, len(lines))
    tail_start = len(lines) - tail_count + 1
    main_lines = lines[: tail_start - 1]
    blocks: list[tuple[int, int, str, str]] = []
    if main_lines:
        current_kind: bool | None = None
        buf: list[str] = []
        buf_start = 1
        for idx, line in enumerate(main_lines, start=1):
            kind = _classify_log_line(line)
            if kind is None:
                kind = current_kind if current_kind is not None else False
            if current_kind is None:
                current_kind = kind
                buf_start = idx
            elif kind != current_kind:
                if buf:
                    blocks.append((buf_start, idx - 1, _block_title(current_kind), "".join(buf)))
                buf = []
                buf_start = idx
                current_kind = kind
            buf.append(line)
        if buf:
            blocks.append((buf_start, len(main_lines), _block_title(current_kind), "".join(buf)))
    chunks: list[Chunk] = []
    log_index = 0
    nonlog_index = 0
    for start_line, end_line, kind_title, block_text in blocks:
        if kind_title == "log":
            log_index += 1
            title = f"log block {log_index}"
        else:
            nonlog_index += 1
            title = f"non-log block {nonlog_index}"
        chunks.extend(_split_oversize(path, title, start_line, end_line, block_text, max_chars))
    tail_text = "".join(lines[tail_start - 1 :])
    if tail_text:
        chunks.extend(
            _split_oversize(
                path,
                "tail",
                tail_start,
                len(lines),
                tail_text,
                max_chars,
            )
        )
    return chunks


def _block_title(is_log: bool | None) -> str:
    if is_log:
        return "log"
    return "non-log"


def _classify_log_line(line: str) -> bool | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("{") and stripped.endswith("}"):
        return True
    if _LOG_LINE_RE.match(stripped):
        return True
    if _PREFIX_RE.match(stripped):
        return True
    ratio = _symbol_ratio(stripped)
    if ratio >= 0.35:
        return True
    if _looks_like_log_prefix(stripped):
        return True
    return False


def _symbol_ratio(text: str) -> float:
    symbols = 0
    non_space = 0
    for ch in text:
        if ch.isspace():
            continue
        non_space += 1
        if ch.isdigit() or ch in _SYMBOL_CHARS:
            symbols += 1
    if non_space == 0:
        return 0.0
    return symbols / non_space


def _looks_like_log_prefix(text: str) -> bool:
    if ":" not in text:
        return False
    head, _sep, _rest = text.partition(":")
    if 2 <= len(head) <= 32 and head.replace("_", "").replace("-", "").isalnum():
        return True
    return False


_LOG_LINE_RE = re.compile(
    r"^(?:"
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"|"
    r"\d{2}:\d{2}:\d{2}"
    r"|"
    r"(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b"
    r"|"
    r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r")"
)
_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]{2,}:\s")
_SYMBOL_CHARS = set("[]{}()=:+-_/\\|<>.,'\"")
