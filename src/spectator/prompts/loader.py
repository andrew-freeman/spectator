from __future__ import annotations

from importlib import resources

_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(rel_path: str) -> str:
    if rel_path in _PROMPT_CACHE:
        return _PROMPT_CACHE[rel_path]
    content = resources.files(__package__).joinpath(rel_path).read_text(encoding="utf-8")
    _PROMPT_CACHE[rel_path] = content
    return content


def get_role_prompt(role: str) -> str:
    return load_prompt(f"roles/{role}.txt")
