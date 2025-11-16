"""Unified registry for declaring runtime tools and their schemas."""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Mapping, Type

from pydantic import BaseModel, create_model


class ToolRegistry:
    """Tracks callable tools and exposes schema metadata."""

    def __init__(self) -> None:
        self.declared_tools: Dict[str, Callable[..., Any]] = {}
        self.argument_models: Dict[str, Type[BaseModel]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        argument_model: Type[BaseModel] | None = None,
    ) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Tool name cannot be empty")
        model = argument_model or self._derive_model(clean_name, handler)
        self.declared_tools[clean_name] = handler
        self.argument_models[clean_name] = model

    def describe(self) -> str:
        lines = []
        for name, model in sorted(self.argument_models.items()):
            fields = model.model_fields
            if not fields:
                lines.append(f"- {name}: no arguments")
                continue
            field_descriptions = []
            for field_name, info in fields.items():
                annotation = getattr(info.annotation, "__name__", str(info.annotation))
                required_checker = getattr(info, "is_required", None)
                if callable(required_checker):
                    required = required_checker()
                else:
                    required = getattr(info, "is_required", False)
                default_marker = "required" if required else "optional"
                field_descriptions.append(f"{field_name} ({annotation}, {default_marker})")
            joined = ", ".join(field_descriptions)
            lines.append(f"- {name}: {joined}")
        return "\n".join(lines)

    def validate_arguments(self, name: str, payload: Mapping[str, Any] | None) -> Dict[str, Any]:
        model = self.argument_models.get(name)
        if model is None:
            raise KeyError(name)
        data = payload or {}
        return model.model_validate(dict(data)).model_dump()

    def _derive_model(self, name: str, handler: Callable[..., Any]) -> Type[BaseModel]:
        sig = inspect.signature(handler)
        hints = inspect.getfullargspec(handler).annotations
        fields: Dict[str, tuple[type[Any], Any]] = {}
        for index, (param_name, parameter) in enumerate(sig.parameters.items()):
            if param_name == "self" and index == 0:
                continue
            if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            annotation = hints.get(param_name, Any)
            default = parameter.default
            default_value = default if default is not inspect._empty else ...
            fields[param_name] = (annotation, default_value)
        model_name = f"{name.title().replace('_', '')}Args"
        return create_model(model_name, **fields)  # type: ignore[arg-type]


__all__ = ["ToolRegistry"]
