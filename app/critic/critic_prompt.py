"""Prompt utilities for the base-layer critic."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List, Optional


def build_critic_prompt(
    actor_output: Dict[str, Any],
    safety_policies: Optional[List[str]] = None,
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

        ## Actor submission
        {json.dumps(actor_output, indent=2, ensure_ascii=False)}

        ## Safety policies
        {json.dumps(policies, indent=2, ensure_ascii=False)}

        ## Output JSON schema
        {{
          "evaluation": "Short summary of alignment with objectives and policies",
          "detected_issues": ["List specific problems or contradictions"],
          "risk_level": "low | medium | high | unsafe",
          "confidence": 0.0-1.0 number showing confidence in this review,
          "recommendations": ["Optional actionable improvements"]
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
