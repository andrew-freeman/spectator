"""Prompt construction utilities for the base-layer actor."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List, Optional


def build_actor_prompt(
    objectives: List[str],
    context: Optional[Dict[str, Any]] = None,
    memory_snippets: Optional[List[str]] = None,
) -> str:
    """Return a prompt instructing the actor model to produce structured JSON.

    The actor is responsible for proposing plans, tool invocations, and
    highlighting any information gaps that should be resolved in later cycles.
    """

    context_block = json.dumps(context or {}, indent=2, ensure_ascii=False)
    memory_block = "\n".join(memory_snippets or []) or "(no recent memory)"

    instructions = dedent(
        f"""
        You are the *Actor* inside a hierarchical cognitive system. Follow the
        specification precisely and respond with **valid JSON** only.

        ## Objectives
        {json.dumps(objectives, indent=2, ensure_ascii=False)}

        ## Context
        {context_block}

        ## Memory Snippets
        {memory_block}

        ## Output JSON schema
        {{
          "analysis": "short paragraphs explaining the situation",
          "plan": ["ordered plan steps"],
          "tool_calls": [
            {{
              "tool_name": "read_sensors | set_fan_speed | read_state | update_state | append_memory | query_memory",
              "arguments": {{"...": "tool specific arguments"}}
            }}
          ],
          "information_gaps": ["questions to resolve"],
          "confidence": 0.0-1.0 number summarising how confident you are
        }}

        *Important rules*
        - Never include trailing comments.
        - Omit optional arrays when empty.
        - Ensure tool call arguments are fully specified JSON objects.
        - Prefer monitoring tools before actuators when unsure.
        """
    ).strip()

    return instructions


__all__ = ["build_actor_prompt"]
