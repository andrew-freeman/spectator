import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.reflection.reflection_runner import REFLECTION_PROMPT, ReflectionRunner


class _DummyClient:
    def __init__(self, payload: dict):
        self._payload = payload
        self.last_prompt = ""

    def generate(self, prompt: str, *, stop=None) -> str:  # pragma: no cover - simple stub
        self.last_prompt = prompt
        return json.dumps(self._payload)


def test_reflection_prompt_handles_literal_braces():
    prompt = REFLECTION_PROMPT.format(message="test", identity_block="{}")
    assert '"intent"' in prompt
    assert '"chat_mode"' in prompt


def test_reflection_runner_injects_identity_and_chat_mode():
    payload = {
        "intent": "chat",
        "refined_objectives": [],
        "context": {},
        "needs_clarification": False,
        "reflection_notes": "",
    }
    client = _DummyClient(payload)
    runner = ReflectionRunner(client, identity_profile={"name": "Spectator"})
    result = runner.run("Who are you?")
    assert result["context"].get("chat_mode") is True
    assert "Spectator" in client.last_prompt
