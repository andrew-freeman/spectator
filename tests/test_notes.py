import json

from spectator.runtime import notes


def test_extract_notes_parses_patch_and_strips_block() -> None:
    payload = {
        "set_goals": ["ship step 0"],
        "add_open_loops": ["docs"],
        "close_open_loops": [],
        "add_decisions": ["use src layout"],
        "add_constraints": ["python 3.12"],
        "set_episode_summary": "bootstrapped",
        "add_memory_tags": ["step0"],
        "actions": ["condense_now"],
    }
    text = (
        "Intro text\n"
        f"{notes.START_MARKER}\n"
        f"{json.dumps(payload)}\n"
        f"{notes.END_MARKER}\n"
        "Outro text"
    )

    visible, patch = notes.extract_notes(text)

    assert "NOTES_JSON" not in visible
    assert visible.strip().startswith("Intro text")
    assert visible.strip().endswith("Outro text")
    assert patch is not None
    assert patch.set_goals == ["ship step 0"]
    assert patch.add_open_loops == ["docs"]
    assert patch.add_decisions == ["use src layout"]
    assert patch.actions == ["condense_now"]


def test_extract_notes_rejects_malformed_json() -> None:
    text = (
        "Before\n"
        f"{notes.START_MARKER}\n"
        "{not-json]\n"
        f"{notes.END_MARKER}\n"
        "After"
    )

    visible, patch = notes.extract_notes(text)

    assert visible == text
    assert patch is None
