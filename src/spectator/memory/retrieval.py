from __future__ import annotations

from typing import Iterable

from spectator.memory.embeddings import Embedder
from spectator.memory.vector_store import MemoryRecord, SQLiteVectorStore


def retrieve(
    query_text: str,
    store: SQLiteVectorStore,
    embedder: Embedder,
    top_k: int = 5,
) -> list[tuple[MemoryRecord, float]]:
    vectors = embedder.embed([query_text])
    if not vectors:
        return []
    return store.query(vectors[0], top_k=top_k)


def format_retrieval_block(results: Iterable[tuple[MemoryRecord, float]]) -> str:
    lines = ["=== RETRIEVAL ==="]
    count = 0
    for record, score in results:
        count += 1
        preview = _truncate_preview(record.text)
        lines.append(f"[{count}] score={score:.3f} id={record.id} text={preview}")
    if count == 0:
        lines.append("(no matches)")
    lines.append("=== END RETRIEVAL ===")
    return "\n".join(lines)


def _truncate_preview(text: str, limit: int = 160) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return f"{flattened[:limit - 3]}..."
