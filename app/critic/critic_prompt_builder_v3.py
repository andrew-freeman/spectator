# app/critic/critic_prompt_builder_v3.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from app.core.schemas import PlannerPlan, ReflectionOutput, ToolResult

from .critic_prompt_v3 import CRITIC_PROMPT


def _serialize(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_critic_prompt_v3(
    reflection: ReflectionOutput,
    plan: PlannerPlan,
    tool_results: Sequence[ToolResult],
    current_state: Dict[str, Any],
    *,
    identity: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the V3 critic prompt by replacing sentinel tokens."""

    reflection_block = _serialize(reflection.to_dict())
    plan_block = _serialize(plan.to_dict())
    tool_results_block = _serialize([tr.to_dict() for tr in tool_results])
    state_block = _serialize(current_state or {})
    identity_block = _serialize(identity or {})
    policy_block = _serialize(policy or {})

    prompt = CRITIC_PROMPT
    prompt = prompt.replace("<<REFLECTION>>", reflection_block)
    prompt = prompt.replace("<<PLAN>>", plan_block)
    prompt = prompt.replace("<<TOOL_RESULTS>>", tool_results_block)
    prompt = prompt.replace("<<STATE>>", state_block)
    prompt = prompt.replace("<<IDENTITY>>", identity_block)
    prompt = prompt.replace("<<POLICY>>", policy_block)

    return prompt