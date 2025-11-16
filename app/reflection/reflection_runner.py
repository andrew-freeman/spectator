"""Reflection runner responsible for pre-processing user intent (V2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Protocol


class SupportsGenerate(Protocol):
    """Protocol describing the shared LLM client interface."""

    def generate(self, prompt: str, *, stop: Optional[Iterable[str]] = None) -> str:
        ...


REFLECTION_PROMPT = """
You are a REFLECTION module that classifies a user message BEFORE any tools or planning.

You MUST respond with a SINGLE JSON object with these fields:

- mode: one of ["chat", "knowledge", "reasoning", "world_query", "world_control"]
- goal: short natural-language description of the primary goal
- context: JSON object with any extra flags or hints (or {{}})
- needs_clarification: boolean
- reflection_notes: short natural-language explanation of how you classified the message

CLASSIFICATION RULES (VERY IMPORTANT):

1. mode = "chat"
   - Greetings, small talk, identity questions:
     - "Who are you?"
     - "Are you there?"
     - "What can you do?"
   - No external tools required.

2. mode = "knowledge"
   - Pure Q&A about general information or math NOT tied to local machine state:
     - "How much is 2+2?"
     - "What is Ohm's law?"
     - "Explain entropy."
   - The answer comes from general knowledge or simple reasoning.
   - DO NOT set world_query here.

3. mode = "reasoning"
   - Logic puzzles, story problems, physical reasoning NOT requiring local machine state:
     - "I have a cup with a nut inside. I go to the kitchen, put the cup on the countertop and flip it upside down, then carry the cup back to the living room. Where is the nut?"
     - "If a train leaves A at 10:00 and another leaves B at 11:00..."
     - Riddles, 'where is X now', mental simulations.
   - The answer should be inferential, not a system command.

4. mode = "world_query"
   - Questions about REAL STATE of the local machine or environment:
     - "What are the readings from nvidia-smi?"
     - "What are the GPU temperatures?"
     - "What is the CPU load?"
   - Requires tools such as read_gpu_temps, read_system_load, etc.

5. mode = "world_control"
   - Requests to CHANGE the environment:
     - "Lower GPU temps below 55C, noise is fine."
     - "Increase fan speed to 60%."
   - Requires control tools, e.g. set_fan_speed.

NEVER mention tools by name in this reflection output. You only classify intent.

EXAMPLES:

Input: "Who are you?"
Output:
{{
  "mode": "chat",
  "goal": "Answer identity / capabilities question",
  "context": {{"chat_mode": true}},
  "needs_clarification": false,
  "reflection_notes": "User is asking who the agent is."
}}

Input: "How much is 2+2?"
Output:
{{
  "mode": "knowledge",
  "goal": "Answer a simple arithmetic question",
  "context": {{"query_mode": false}},
  "needs_clarification": false,
  "reflection_notes": "Simple math; no tools required."
}}

Input: "What are the readings from nvidia-smi?"
Output:
{{
  "mode": "world_query",
  "goal": "Retrieve current GPU readings via nvidia-smi",
  "context": {{"query_mode": true}},
  "needs_clarification": false,
  "reflection_notes": "Needs live GPU state."
}}

Input: "Please lower GPU temps below 55C, noise is fine."
Output:
{{
  "mode": "world_control",
  "goal": "Adjust cooling to bring GPUs below 55C",
  "context": {{"target_temp": 55, "noise_sensitive": false}},
  "needs_clarification": false,
  "reflection_notes": "Direct request to change hardware behaviour."
}}

Input: "I have a cup with a nut inside. I go to the kitchen, put the cup on the countertop and flip it upside down. Then I take the cup and go back to living room. Where is the nut?"
Output:
{{
  "mode": "reasoning",
  "goal": "Solve a physical reasoning puzzle about object location",
  "context": {{}},
  "needs_clarification": false,
  "reflection_notes": "This is a story puzzle; answer by reasoning, not tools."
}}

Now process the USER MESSAGE below and respond with EXACTLY one JSON object, no extra text.

USER MESSAGE:
{message}
""".strip()


@dataclass
class ReflectionOutput:
    mode: str = "chat"
    goal: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    reflection_notes: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ReflectionOutput":
        mode = str(payload.get("mode", "chat")).strip().lower()
        if mode not in {"chat", "knowledge", "reasoning", "world_query", "world_control"}:
            mode = "chat"
        goal = str(payload.get("goal", "")).strip()
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            context = {}
        needs_clarification = bool(payload.get("needs_clarification", False))
        reflection_notes = str(payload.get("reflection_notes", "")).strip()
        return cls(
            mode=mode,
            goal=goal,
            context=context,
            needs_clarification=needs_clarification,
            reflection_notes=reflection_notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReflectionRunner:
    """Generate a structured reflection summary prior to full reasoning."""

    def __init__(self, client: SupportsGenerate, identity_profile: Optional[Dict[str, Any]] = None):
        self._client = client
        self._identity = identity_profile or {}

    def run(self, message: str) -> ReflectionOutput:
        prompt = REFLECTION_PROMPT.format(message=message)
        try:
            raw = self._client.generate(prompt, stop=None)
            # Try to strip any junk around JSON
            raw_stripped = raw.strip()
            first_brace = raw_stripped.find("{")
            last_brace = raw_stripped.rfind("}")
            if first_brace != -1 and last_brace != -1:
                raw_stripped = raw_stripped[first_brace : last_brace + 1]
            payload = json.loads(raw_stripped)
            return ReflectionOutput.from_payload(payload)
        except Exception:
            # Defensive fallback keeps pipeline resilient.
            return ReflectionOutput(
                mode="chat",
                goal=message.strip(),
                context={},
                needs_clarification=False,
                reflection_notes="Reflection fallback invoked; defaulting to chat mode.",
            )