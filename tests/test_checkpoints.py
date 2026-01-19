import json
import time
from pathlib import Path

import pytest

from spectator.core.types import ChatMessage, Checkpoint, State
from spectator.runtime import checkpoints


def test_load_or_create_defaults(tmp_path: Path) -> None:
    checkpoint = checkpoints.load_or_create("session-1", base_dir=tmp_path)

    assert checkpoint.session_id == "session-1"
    assert checkpoint.revision == 0
    assert isinstance(checkpoint.updated_ts, float)
    assert checkpoint.state == State()
    assert checkpoint.recent_messages == []
    assert checkpoint.trace_tail == []


def test_save_checkpoint_increments_revision(tmp_path: Path) -> None:
    checkpoint = checkpoints.load_or_create("session-2", base_dir=tmp_path)
    first_updated = checkpoint.updated_ts

    checkpoints.save_checkpoint(checkpoint, base_dir=tmp_path)

    assert checkpoint.revision == 1
    assert checkpoint.updated_ts >= first_updated

    second_updated = checkpoint.updated_ts
    checkpoints.save_checkpoint(checkpoint, base_dir=tmp_path)

    assert checkpoint.revision == 2
    assert checkpoint.updated_ts >= second_updated


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    checkpoint = Checkpoint(
        session_id="session-3",
        revision=0,
        updated_ts=time.time(),
        state=State(
            goals=["goal"],
            open_loops=["loop"],
            decisions=["decision"],
            constraints=["constraint"],
            episode_summary="summary",
            memory_tags=["tag"],
            memory_refs=["ref"],
            capabilities_granted=["net"],
            capabilities_pending=["net:example.com"],
        ),
        recent_messages=[ChatMessage(role="user", content="Hello")],
        trace_tail=["trace.jsonl"],
    )

    checkpoints.save_checkpoint(checkpoint, base_dir=tmp_path)
    loaded = checkpoints.load_latest("session-3", base_dir=tmp_path)

    assert loaded is not None
    assert loaded == checkpoint


def test_load_latest_rejects_invalid_state_types(tmp_path: Path) -> None:
    path = tmp_path / "session-bad.json"
    payload = {
        "session_id": "session-bad",
        "revision": 1,
        "updated_ts": time.time(),
        "state": {
            "goals": "not-a-list",
            "open_loops": [],
            "decisions": [],
            "constraints": [],
            "episode_summary": 123,
            "memory_tags": [],
            "memory_refs": [],
            "capabilities_granted": [],
            "capabilities_pending": [],
        },
        "recent_messages": [],
        "trace_tail": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint state field"):
        checkpoints.load_latest("session-bad", base_dir=tmp_path)
