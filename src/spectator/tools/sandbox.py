from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

def _is_within_root(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_under_root(root: Path, user_path: str) -> Optional[Path]:
    """
    Return an absolute Path under `root` corresponding to `user_path`,
    or None if it escapes the sandbox.

    Rules:
      - no absolute paths
      - normalize and resolve to remove '..'
      - final path must be within root (or equal to root)
    """
    if not isinstance(user_path, str) or not user_path:
        return None

    root_abs = root.resolve(strict=False)

    # Reject absolute paths early (covers /etc/passwd, C:\..., etc)
    p = Path(user_path)
    if p.is_absolute():
        return None

    # Build candidate then resolve (strict=False so non-existent files still work)
    cand = (root_abs / p).resolve(strict=False)

    try:
        # Raises ValueError if cand is not under root_abs
        cand.relative_to(root_abs)
    except ValueError:
        return None

    return cand


def validate_shell_cmd(
    cmd: str,
    allowed_prefixes: Iterable[str],
    deny_substrings: Iterable[str],
) -> Tuple[bool, Optional[str]]:
    if not isinstance(cmd, str):
        return False, "cmd must be a string"

    s = cmd.strip()
    if not s:
        return False, "empty command"

    # Cheap reject of obvious shell metacharacters (tighten as needed)
    # This prevents `rm -rf /; echo ok`, pipes, redirects, subshells, etc.
    forbidden_chars = ["|", ";", "&", ">", "<", "`", "$(", "\n"]
    if any(ch in s for ch in forbidden_chars):
        return False, "shell metacharacters are not allowed"

    # Parse tokens safely; if it fails, reject
    try:
        tokens = shlex.split(s)
    except ValueError:
        return False, "failed to parse command"

    if not tokens:
        return False, "empty command"

    first = tokens[0]
    allowed = set(allowed_prefixes)
    if first not in allowed:
        return False, f"command '{first}' not allowed"

    lower = s.lower()
    for bad in deny_substrings:
        if bad.lower() in lower:
            return False, f"disallowed substring: {bad}"

    return True, None
