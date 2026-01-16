from spectator.memory.retrieval import format_retrieval_block
from spectator.memory.vector_store import MemoryRecord


def test_format_retrieval_block_no_matches() -> None:
    block = format_retrieval_block([])

    assert "(no matches)" in block


def test_format_retrieval_block_truncates_preview() -> None:
    record = MemoryRecord(
        id="rec-1",
        ts=123.0,
        text="word " * 100,
    )

    block = format_retrieval_block([(record, 0.9876)])
    lines = block.splitlines()
    entry_line = next(line for line in lines if line.startswith("[1]"))

    assert entry_line.endswith("...")
    assert len(entry_line) <= 220
