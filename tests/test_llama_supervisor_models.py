from __future__ import annotations

from pathlib import Path

import pytest

from llama_supervisor.models import list_models, resolve_model_path


def test_model_listing_and_resolve(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    model = root / "sample.gguf"
    model.write_text("data", encoding="utf-8")
    other = root / "skip.bin"
    other.write_text("nope", encoding="utf-8")

    items = list_models(root)
    assert [item.path for item in items] == ["sample.gguf"]

    resolved = resolve_model_path(root, "sample.gguf")
    assert resolved == model.resolve()


def test_model_path_traversal_blocked(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside.gguf"
    outside.write_text("data", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_model_path(root, "../outside.gguf")
