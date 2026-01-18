from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from spectator.tools.sandbox import validate_shell_cmd

ALLOWED_PREFIXES = [
    "ls",
    "cat",
    "echo",
    "pwd",
    "python",
    "pytest",
    "rg",
    "grep",
    "sed",
    "head",
    "tail",
]
DENY_SUBSTRINGS = [
    "rm ",
    " rm",
    "sudo",
    "chmod",
    "chown",
    "mkfs",
    "dd ",
    ":(){",
    ">/dev/sd",
    "curl",
    "wget",
]

MAX_OUTPUT_CHARS = 20000


def shell_exec_handler(root: Path) -> Any:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        cmd = args.get("cmd")
        timeout_s = args.get("timeout_s", 20)
        if not isinstance(cmd, str):
            raise ValueError("cmd must be a string")
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        ok, reason = validate_shell_cmd(cmd, ALLOWED_PREFIXES, DENY_SUBSTRINGS)
        if not ok:
            raise ValueError(reason or "command rejected")

        try:
            completed = subprocess.run(
                cmd,
                shell=True,
                cwd=root,
                timeout=timeout_s,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("command timed out") from exc

        stdout = completed.stdout[:MAX_OUTPUT_CHARS]
        stderr = completed.stderr[:MAX_OUTPUT_CHARS]
        return {
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    return handler
