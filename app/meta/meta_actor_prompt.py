"""Prompt builder for the meta-layer actor."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List, Optional


def build_meta_actor_prompt(
    current_params: Dict[str, Any],
    recent_decisions: List[Dict[str, Any]],
    meta_cycle: int,
    system_limits: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the meta-actor instruction prompt with JSON schema requirements."""

    prompt = dedent(
        f"""
        You are the *Meta-Actor* coordinating cognitive strategy updates for a
        hierarchical reasoning system. Respond with **valid JSON only**.

        ## Current cognitive parameters
        {json.dumps(current_params, indent=2, ensure_ascii=False)}

        ## Recent governor decisions
        {json.dumps(recent_decisions, indent=2, ensure_ascii=False)}

        ## System limits
        {json.dumps(system_limits or {}, indent=2, ensure_ascii=False)}

        ## Meta cycle
        {meta_cycle}

        ## Output JSON schema
        {{
          "meta_thoughts": ["Observations about current performance"],
          "cognitive_strategy": {{
              "priorities": ["Key focus areas"],
              "cycle_adjustments": "How to adjust evaluation cadence"
          }},
          "parameter_adjustments": {{
              "<param_name>": {{"delta": number between -0.05 and 0.05, "justification": "reason"}}
          }},
          "meta_improvements": ["Process improvements to apply"],
          "assumptions": ["Explicit assumptions enabling these updates"]
        }}

        Rules:
        - Ensure deltas respect absolute limit of 0.05 per update.
        - Prefer incremental refinements over sweeping changes.
        - Highlight uncertainty via the assumptions field.
        """
    ).strip()

    return prompt


__all__ = ["build_meta_actor_prompt"]
