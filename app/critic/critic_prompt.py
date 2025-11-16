"""Prompt utilities for the base-layer critic."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List, Optional


def build_critic_prompt(
    plan_payload: Dict[str, Any],
    safety_policies: Optional[List[str]] = None,
    *,
    identity: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> str:
    """Create the critic instruction prompt using strict JSON formatting."""

    policies = safety_policies or [
        "Do not execute unsafe actuator commands without confirming sensor data.",
        "Respect user privacy and avoid exfiltrating confidential data.",
        "Flag any contradictions or missing context in the actor plan.",
    ]

    prompt = dedent(
        f"""
        You are the *Critic* in a hierarchical reasoning system. Review the
        actor output carefully and respond with **valid JSON only**.

        ## Identity & environment
        {json.dumps(identity or {}, indent=2, ensure_ascii=False)}

        ## Thermal policy guidance
        {json.dumps(policy or {}, indent=2, ensure_ascii=False)}

        ## Planner submission
        {json.dumps(plan_payload, indent=2, ensure_ascii=False)}

        ## Safety policies
        {json.dumps(policies, indent=2, ensure_ascii=False)}

        ## Output JSON schema
        {{
          "risk": "low | medium | high | unsafe",
          "issues": ["List specific problems or contradictions"],
          "suggestions": ["Optional actionable improvements"],
          "adjusted_steps": ["Optional revised steps"],
          "adjusted_tool_calls": [{{"name": "tool", "arguments": {{}}}}],
          "confidence": 0.0-1.0 number showing confidence in this review,
          "notes": "Short natural-language summary"
        }}

        Rules:
        - If the actor plan may cause harm or violates policies, set risk_level
          to "unsafe".
        - Provide concrete references to steps or tool calls when flagging issues.
        - When you are uncertain, lower the confidence score.
        """
    ).strip()

    return prompt


__all__ = ["build_critic_prompt"]
