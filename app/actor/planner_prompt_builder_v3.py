# app/actor/planner_prompt_builder_v3.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.core.schemas import ReflectionOutput
from app.core.tool_registry import READ_TOOLS, CONTROL_TOOLS

from .actor_prompt_v3 import PLANNER_PROMPT


def _serialize(obj: Any) -> str:
    """Serialize Python objects into pretty JSON blocks for the context sections."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_planner_prompt_v3(
    reflection: ReflectionOutput,
    current_state: Dict[str, Any],
    *,
    memory_context: Optional[List[str]] = None,
    identity: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the V3 planner prompt by replacing sentinel tokens."""

    reflection_block = _serialize(reflection.to_dict())
    state_block = _serialize(current_state or {})
    memory_block = _serialize(memory_context or [])
    identity_block = _serialize(identity or {})
    policy_block = _serialize(policy or {})

    prompt = PLANNER_PROMPT

    prompt = prompt.replace("<<ALLOWED_READ_TOOLS>>", ", ".join(sorted(READ_TOOLS)))
    prompt = prompt.replace("<<ALLOWED_CONTROL_TOOLS>>", ", ".join(sorted(CONTROL_TOOLS)))

    prompt = prompt.replace("<<REFLECTION>>", reflection_block)
    prompt = prompt.replace("<<STATE>>", state_block)
    prompt = prompt.replace("<<MEMORY>>", memory_block)
    prompt = prompt.replace("<<IDENTITY>>", identity_block)
    prompt = prompt.replace("<<POLICY>>", policy_block)

    return prompt