import math

from spectator.memory.embeddings import HashEmbedder


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashEmbedder(dim=64)
    vectors_first = embedder.embed(["hello", "world"])
    vectors_second = embedder.embed(["hello", "world"])

    assert vectors_first == vectors_second

    for vector in vectors_first:
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm > 0.0
        assert math.isclose(norm, 1.0, rel_tol=1e-6)
