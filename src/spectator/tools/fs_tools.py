from __future__ import annotations

from pathlib import Path
from typing import Any

from spectator.tools.sandbox import resolve_under_root


def _get_path(root: Path, user_path: str) -> Path:
    if "\x00" in user_path:
        raise ValueError("path contains NUL byte")
    if user_path == "/sandbox":
        user_path = "."
    elif user_path.startswith("/sandbox/"):
        user_path = user_path[len("/sandbox/"):]
    resolved = resolve_under_root(root, user_path)
    if resolved is None:
        raise ValueError("path escapes sandbox")
    return resolved


def read_text_handler(root: Path) -> Any:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        max_bytes = args.get("max_bytes", 20000)
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")

        resolved = _get_path(root, path)
        if not resolved.is_file():
            raise ValueError("path is not a file")

        with resolved.open("rb") as handle:
            data = handle.read(max_bytes)
        text = data.decode("utf-8", errors="replace")
        return {"path": path, "text": text}

    return handler


def list_dir_handler(root: Path) -> Any:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path", ".")
        max_entries = args.get("max_entries", 200)
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")

        resolved = _get_path(root, path)
        if not resolved.is_dir():
            raise ValueError("path is not a directory")

        entries = sorted(entry.name for entry in resolved.iterdir())
        return {"path": path, "entries": entries[:max_entries]}

    return handler


def write_text_handler(root: Path) -> Any:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        text = args.get("text")
        overwrite = args.get("overwrite", False)
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not isinstance(overwrite, bool):
            raise ValueError("overwrite must be a boolean")

        resolved = _get_path(root, path)
        if resolved.exists() and not overwrite:
            raise ValueError("refusing to overwrite existing file")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text, encoding="utf-8")
        return {"path": path, "bytes": len(text.encode("utf-8"))}

    return handler
