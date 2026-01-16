from __future__ import annotations

from pathlib import Path


def _is_within_root(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_under_root(root: Path, user_path: str) -> Path | None:
    root = root.resolve()
    candidate = Path(user_path)
    if candidate.is_absolute():
        return None

    current = root
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            return None
        current = current / part
        if current.exists() and current.is_symlink():
            resolved = current.resolve()
            if not _is_within_root(root, resolved):
                return None
            current = resolved

    resolved_path = (root / candidate).resolve()
    if not _is_within_root(root, resolved_path):
        return None
    return resolved_path


def validate_shell_cmd(
    cmd: str, allow_prefixes: list[str], deny_substrings: list[str]
) -> tuple[bool, str | None]:
    stripped = cmd.lstrip()
    allowed = any(
        stripped == prefix or stripped.startswith(f"{prefix} ")
        for prefix in allow_prefixes
    )
    if not allowed:
        return False, "command not allowed"

    for deny in deny_substrings:
        if deny in cmd:
            return False, "command contains denied substring"

    return True, None
