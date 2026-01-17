from spectator.backends.fake import FakeBackend
from spectator.core.types import Checkpoint, State
from spectator.runtime.pipeline import RoleSpec, run_pipeline


def test_pipeline_sanitizes_visible_output() -> None:
    checkpoint = Checkpoint(session_id="s-sanitize", revision=0, updated_ts=0.0, state=State())
    backend = FakeBackend()
    backend.extend_role_responses("governor", ["STATE:\n{...}"])
    roles = [RoleSpec(name="governor", system_prompt="Decide.")]

    final_text, results, _updated = run_pipeline(checkpoint, "hello", roles, backend)

    assert final_text
    assert not final_text.startswith("STATE:")
    assert results[0].text == final_text
