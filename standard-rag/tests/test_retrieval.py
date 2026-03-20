"""Unit tests for FAISS index, retriever, and passage loader.

Note: the faiss-cpu PyPI wheel crashes on macOS ARM64 with Python ≥3.12.
FAISS tests are therefore skipped on non-Linux platforms; they pass on EC2.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from rag.retrieval.index import load_passages

faiss_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="faiss-cpu PyPI wheel is incompatible with macOS ARM64 (Python ≥3.12)",
)


def _write_tsv(path, rows: list[tuple[str, str, str]]) -> None:
    with open(path, "w") as f:
        f.write("id\ttext\ttitle\n")
        for pid, text, title in rows:
            f.write(f"{pid}\t{text}\t{title}\n")


class TestLoadPassages:
    def test_basic_parse(self, tmp_path):
        _write_tsv(tmp_path / "p.tsv", [("1", "hello world", "Doc A")])
        _, texts, titles = load_passages(tmp_path / "p.tsv")
        assert texts == ["hello world"]
        assert titles == ["Doc A"]

    def test_header_skipped(self, tmp_path):
        _write_tsv(tmp_path / "p.tsv", [("1", "text", "title")])
        _, texts, _ = load_passages(tmp_path / "p.tsv")
        assert len(texts) == 1

    def test_multiple_rows(self, tmp_path):
        rows = [(str(i), f"text {i}", f"title {i}") for i in range(5)]
        _write_tsv(tmp_path / "p.tsv", rows)
        _, texts, titles = load_passages(tmp_path / "p.tsv")
        assert len(texts) == 5
        assert texts[2] == "text 2"
        assert titles[4] == "title 4"

    def test_max_passages_cap(self, tmp_path):
        rows = [(str(i), f"text {i}", "") for i in range(10)]
        _write_tsv(tmp_path / "p.tsv", rows)
        _, texts, _ = load_passages(tmp_path / "p.tsv", max_passages=3)
        assert len(texts) == 3

    def test_missing_title_defaults_empty(self, tmp_path):
        with open(tmp_path / "p.tsv", "w") as f:
            f.write("id\ttext\ttitle\n")
            f.write("1\tsome text\n")  # no title column
        _, texts, titles = load_passages(tmp_path / "p.tsv")
        assert texts == ["some text"]
        assert titles == [""]

    def test_malformed_line_skipped(self, tmp_path):
        with open(tmp_path / "p.tsv", "w") as f:
            f.write("id\ttext\ttitle\n")
            f.write("bad line with no tabs\n")
            f.write("1\tgood text\ttitle\n")
        _, texts, _ = load_passages(tmp_path / "p.tsv")
        assert texts == ["good text"]

    def test_ids_returned(self, tmp_path):
        _write_tsv(tmp_path / "p.tsv", [("42", "text", "title")])
        ids, _, _ = load_passages(tmp_path / "p.tsv")
        assert ids == ["42"]


pytestmark_faiss = faiss_only


def _make_index(n: int = 10, dim: int = 768):
    import faiss

    from rag.retrieval.index import FAISSIndex

    vecs = np.random.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    texts = [f"passage {i}" for i in range(n)]
    titles = [f"title {i}" for i in range(n)]
    return FAISSIndex(index, texts, titles), vecs


@faiss_only
class TestFAISSIndex:
    def test_search_returns_top_k(self):
        idx, _ = _make_index(n=20)
        q = np.random.randn(1, 768).astype(np.float32)
        q /= np.linalg.norm(q)
        assert len(idx.search(q, top_k=5)) == 5

    def test_search_returns_strings(self):
        idx, _ = _make_index(n=10)
        q = np.random.randn(1, 768).astype(np.float32)
        q /= np.linalg.norm(q)
        assert all(isinstance(r, str) for r in idx.search(q, top_k=3))

    def test_search_self_is_top_1(self):
        import faiss

        from rag.retrieval.index import FAISSIndex

        dim = 768
        vecs = np.eye(dim, dtype=np.float32)[:5]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        fi = FAISSIndex(index, [f"passage {i}" for i in range(5)], [""] * 5)
        for i in range(5):
            result = fi.search(vecs[i : i + 1], top_k=1)
            assert result[0] == f"passage {i}"

    def test_search_top_k_larger_than_n(self):
        idx, _ = _make_index(n=3)
        q = np.random.randn(1, 768).astype(np.float32)
        q /= np.linalg.norm(q)
        results = idx.search(q, top_k=10)
        assert len(results) == 3

    def test_search_formats_title_and_text(self):
        import faiss

        from rag.retrieval.index import FAISSIndex

        dim = 768
        vec = np.eye(dim, dtype=np.float32)[:1]
        index = faiss.IndexFlatIP(dim)
        index.add(vec)
        fi = FAISSIndex(index, ["body text"], ["My Title"])
        result = fi.search(vec, top_k=1)[0]
        assert "My Title" in result
        assert "body text" in result

    def test_search_no_title_returns_text_only(self):
        import faiss

        from rag.retrieval.index import FAISSIndex

        dim = 768
        vec = np.eye(dim, dtype=np.float32)[:1]
        index = faiss.IndexFlatIP(dim)
        index.add(vec)
        fi = FAISSIndex(index, ["just the text"], [""])
        result = fi.search(vec, top_k=1)[0]
        assert result == "just the text"

    def test_save_load_roundtrip(self, tmp_path):
        idx, _ = _make_index(n=5)
        idx.save(tmp_path / "test.index", tmp_path / "meta.pkl")

        from rag.retrieval.index import FAISSIndex

        loaded = FAISSIndex.load(tmp_path / "test.index", tmp_path / "meta.pkl")
        q = np.random.randn(1, 768).astype(np.float32)
        q /= np.linalg.norm(q)
        assert loaded.search(q, top_k=3) == idx.search(q, top_k=3)
