from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ModelInfo:
    path: str
    size_bytes: int
    mtime: float


def list_models(model_root: Path, suffixes: Iterable[str] | None = None) -> list[ModelInfo]:
    if suffixes is None:
        suffixes = {".gguf"}
    root = model_root.resolve()
    if not root.exists():
        return []
    items: list[ModelInfo] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(root)
        stat = path.stat()
        items.append(
            ModelInfo(
                path=str(rel),
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    items.sort(key=lambda item: item.path)
    return items


def resolve_model_path(model_root: Path, model_path: str) -> Path:
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model must be a non-empty string")
    candidate = Path(model_path)
    resolved = (model_root / candidate).resolve()
    try:
        resolved.relative_to(model_root.resolve())
    except ValueError as exc:
        raise ValueError("model path escapes model root") from exc
    if not resolved.is_file():
        raise ValueError("model path does not exist")
    return resolved
