from __future__ import annotations

"""Test artifact utilities.

Enable per-request logging for llama harness runs by setting:
- SPECTATOR_TEST_OUTDIR to the llama_test_artifacts/<SESSION_ID> directory
- SPECTATOR_TEST_CASE_ID to the current case id
Optionally set SPECTATOR_TEST_REDACT=1 to redact auth/token fields.
"""

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REDACT_KEYS = ("authorization", "api_key", "apikey", "token", "access_token")
_COUNTER = itertools.count(1)


@dataclass(frozen=True, slots=True)
class TestArtifactContext:
    enabled: bool
    outdir: Path | None
    case_id: str | None
    session_id: str | None
    redact: bool


def get_test_context() -> TestArtifactContext:
    outdir_raw = os.getenv("SPECTATOR_TEST_OUTDIR")
    case_id = os.getenv("SPECTATOR_TEST_CASE_ID")
    redact = os.getenv("SPECTATOR_TEST_REDACT") == "1"
    if outdir_raw and case_id:
        outdir = Path(outdir_raw)
        session_id = outdir.name
        return TestArtifactContext(True, outdir, case_id, session_id, redact)
    return TestArtifactContext(False, None, None, None, redact)


def next_sequence() -> int:
    return next(_COUNTER)


def artifact_paths(outdir: Path, case_id: str) -> tuple[Path, Path, Path]:
    case_root = outdir / "cases" / case_id
    return (
        case_root / "requests",
        case_root / "responses",
        case_root / "meta",
    )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def write_bytes(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            handle.write(data)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def write_json(path: Path, obj: Any) -> None:
    write_bytes(path, json_bytes(obj))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _should_redact(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _REDACT_KEYS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("Bearer "):
        return "Bearer ***REDACTED***"
    return "***REDACTED***"


def redact_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted: dict[str, Any] = {}
        for key, value in obj.items():
            if _should_redact(str(key)):
                redacted[key] = _redact_value(value)
            else:
                redacted[key] = redact_obj(value)
        return redacted
    if isinstance(obj, list):
        return [redact_obj(item) for item in obj]
    return obj


def maybe_redact(obj: Any, enabled: bool) -> Any:
    if not enabled:
        return obj
    return redact_obj(obj)
