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


def test_reflection_prompt_lists_constraints():
    prompt = REFLECTION_PROMPT.format(message="test")
    assert "Role: Spectator reflection classifier." in prompt
    assert "Constraints:" in prompt


def test_reflection_runner_parses_json_payload():
    payload = {
        "mode": "knowledge",
        "goal": "Compute 2+2",
        "context": {},
    }
    client = _DummyClient(payload)
    runner = ReflectionRunner(client, identity_profile={"name": "Spectator"})
    result = runner.run("How much is 2+2?")
    assert result.mode == "knowledge"
    assert result.goal == "Compute 2+2"
