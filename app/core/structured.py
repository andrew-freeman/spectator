"""Helpers for schema-driven LLM prompting with automatic repair."""
from __future__ import annotations

import json
from typing import Callable, Iterable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


ModelT = TypeVar("ModelT", bound=BaseModel)


def assemble_prompt(*sections: str) -> str:
    return "\n".join(part for part in sections if part)


def wrap_json_content(content: str) -> str:
    text = content.strip()
    if not text:
        return "{}"
    prefix = "" if text.startswith("{") else "{\n"
    suffix = "" if text.endswith("}") else "\n}"
    return f"{prefix}{text}{suffix}"


def generate_structured_object(
    client: SupportsGenerate,
    prompt: str,
    schema: type[ModelT],
    fallback: Callable[[Exception | None], ModelT],
) -> ModelT:
    raw = client.generate(prompt, stop=None)
    return _decode_with_repair(client, schema, raw, fallback)


def _decode_with_repair(
    client: SupportsGenerate,
    schema: type[ModelT],
    raw: str,
    fallback: Callable[[Exception | None], ModelT],
) -> ModelT:
    candidate = wrap_json_content(raw)
    try:
        return schema.model_validate_json(candidate)
    except (ValidationError, ValueError) as exc:
        repaired = _repair_json(client, schema, candidate)
        if repaired is not None:
            return repaired
        return fallback(exc)


def _repair_json(
    client: SupportsGenerate,
    schema: type[ModelT],
    invalid_json: str,
) -> ModelT | None:
    schema_block = json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=False)
    instructions = assemble_prompt(
        "Role: JSON repair assistant.",
        "Constraints:",
        "- Fix the provided JSON so it validates against the schema.",
        "- Preserve the meaning and key values.",
        "- Return only JSON content (no braces, no prose).",
        "Schema description:",
        schema_block,
        "Original JSON object:",
        invalid_json,
    )
    raw = client.generate(instructions, stop=None)
    try:
        return schema.model_validate_json(wrap_json_content(raw))
    except (ValidationError, ValueError):
        return None


__all__ = ["assemble_prompt", "generate_structured_object", "wrap_json_content"]
