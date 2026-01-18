from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple
import shlex  

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
    """
    Validate a shell command against a conservative safety policy.

    Rules:
      - cmd must be a non-empty string
      - no shell metacharacters (; | & > < ` $() newlines)
      - command must parse cleanly via shlex
      - first token must be in allowed_prefixes
      - deny_substrings are matched against *tokens*, not raw text
    """

    if not isinstance(cmd, str):
        return False, "cmd must be a string"

    s = cmd.strip()
    if not s:
        return False, "empty command"

    # Reject obvious shell metacharacters early (outside of quotes where relevant)
    # Prevents chaining, pipes, redirects, subshells, etc.
    in_quote: str | None = None
    for ch in s:
        if ch in {"'", '"'}:
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
            continue
        if ch in {"$", "`", "\n"}:
            return False, "shell metacharacters are not allowed"
        if ch in {"|", "&", ">", "<"}:
            return False, "shell metacharacters are not allowed"
        if ch == ";" and in_quote is None:
            return False, "shell metacharacters are not allowed"

    # Parse command safely
    try:
        tokens = shlex.split(s)
    except Exception:
        return False, "failed to parse command"

    if not tokens:
        return False, "empty command"

    # Validate command name
    first = tokens[0]
    allowed = {p for p in allowed_prefixes}
    if first not in allowed:
        return False, f"command '{first}' not allowed"

    # Normalize deny rules
    deny_set = {d.lower() for d in deny_substrings}
    token_lowers = [t.lower() for t in tokens]

    # Reject explicitly denied tokens
    for tok in token_lowers:
        if tok in deny_set:
            return False, f"disallowed token: {tok}"

    # Extra safety: reject tokens that *start* with dangerous prefixes
    # (e.g. rm -rf, dd if=, mkfs.ext4, etc.)
    for tok in token_lowers:
        for bad in deny_set:
            if tok.startswith(bad):
                return False, f"disallowed token prefix: {bad}"

    return True, None
