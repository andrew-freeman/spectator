from __future__ import annotations

from spectator.analysis.chunking import chunk_file


def test_chunk_headings_markdown() -> None:
    text = (
        "# Intro\n"
        "Alpha\n"
        "## Details\n"
        "More\n"
        "### Appendix\n"
        "End\n"
    )
    chunks = chunk_file("doc.md", text, strategy="headings", max_chars=200)
    titles = [chunk.title for chunk in chunks]
    assert len(chunks) >= 3
    assert any("Intro" in title for title in titles)
    assert any("Details" in title for title in titles)
    assert any("Appendix" in title for title in titles)


def test_chunk_python_ast() -> None:
    text = (
        "import os\n"
        "\n"
        "def foo():\n"
        "    return 1\n"
        "\n"
        "class Bar:\n"
        "    def baz(self):\n"
        "        return 2\n"
        "\n"
        "def qux():\n"
        "    return 3\n"
    )
    chunks = chunk_file("sample.py", text, strategy="python_ast", max_chars=200)
    titles = [chunk.title for chunk in chunks]
    assert "module" in titles
    assert "def foo" in titles
    assert "class Bar" in titles
    assert "def qux" in titles


def test_chunk_oversize_function_splits() -> None:
    body = "    value = 1\n" * 200
    text = f"def big():\n{body}"
    chunks = chunk_file("big.py", text, strategy="python_ast", max_chars=200)
    part_titles = [chunk.title for chunk in chunks if "part" in chunk.title]
    assert len(chunks) > 1
    assert part_titles
    assert all("def big (part" in title for title in part_titles)


def test_chunk_auto_falls_back_to_fixed() -> None:
    text = ("line\n" * 50).strip() + "\n"
    chunks_a = chunk_file("data.bin", text, strategy="auto", max_chars=50)
    chunks_b = chunk_file("data.bin", text, strategy="auto", max_chars=50)
    assert chunks_a
    assert all(chunk.strategy == "fixed" for chunk in chunks_a)
    assert [chunk.id for chunk in chunks_a] == [chunk.id for chunk in chunks_b]
