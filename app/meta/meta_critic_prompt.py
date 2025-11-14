"""Prompt factory for the meta-layer critic."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict


def build_meta_critic_prompt(meta_actor_payload: Dict[str, Any], current_params: Dict[str, Any]) -> str:
    """Create the prompt instructing the meta-critic to review proposed changes."""

    prompt = dedent(
        f"""
        You are the *Meta-Critic* ensuring stability of a hierarchical cognition
        stack. Respond with **valid JSON only**.

        ## Meta-actor proposal
        {json.dumps(meta_actor_payload, indent=2, ensure_ascii=False)}

        ## Current parameters
        {json.dumps(current_params, indent=2, ensure_ascii=False)}

        ## Output JSON schema
        {{
          "meta_evaluation": "Summary of proposal viability",
          "meta_issues": ["Specific risks or contradictions"],
          "risk_rating": "low | medium | high | unsafe",
          "confidence": 0.0-1.0 number indicating trust in this assessment,
          "meta_improvements": ["Suggested refinements"],
          "stability_notes": "Optional notes about long-term impacts"
        }}

        Rules:
        - Flag proposals exceeding ±0.05 delta as unstable.
        - Encourage logging and memory hygiene when uncertainty is high.
        - If unsure, lower confidence rather than approving risky changes.
        """
    ).strip()

    return prompt


__all__ = ["build_meta_critic_prompt"]
