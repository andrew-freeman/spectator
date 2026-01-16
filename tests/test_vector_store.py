from pathlib import Path

from spectator.memory.embeddings import HashEmbedder
from spectator.memory.vector_store import MemoryRecord, SQLiteVectorStore


def test_vector_store_add_and_query(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "memory.sqlite")
    embedder = HashEmbedder(dim=32)
    records = [
        MemoryRecord(id="one", ts=1.0, text="remember this"),
        MemoryRecord(id="two", ts=2.0, text="something else"),
    ]
    vectors = embedder.embed([record.text for record in records])

    store.add(records, vectors)

    query_vector = embedder.embed(["remember this"])[0]
    results = store.query(query_vector, top_k=1)

    assert results
    assert results[0][0].id == "one"
