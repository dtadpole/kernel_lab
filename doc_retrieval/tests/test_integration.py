"""Integration tests for the full doc retrieval pipeline.

Tests HTML parsing → BM25 indexing → search → navigation on real CUDA docs.
Dense search tests require the embedding service and are skipped if unavailable.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "html"

# Use a small doc for fast tests, large doc for coverage
SMALL_SLUG = "blackwell-tuning-guide"
LARGE_SLUG = "cuda-c-programming-guide"


def _needs_html(slug: str):
    """Skip if the HTML data isn't available."""
    path = DATA_DIR / slug / "index.html"
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"HTML data not available: {path}",
    )


@pytest.fixture(scope="module")
def enc():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture(scope="module")
def small_doc_parsed(enc):
    """Parse the small doc once for the module."""
    from doc_retrieval.html_parser import parse_html_doc

    html = (DATA_DIR / SMALL_SLUG / "index.html").read_text(encoding="utf-8")
    toc, sections, chunks = parse_html_doc(
        html, SMALL_SLUG,
        f"https://docs.nvidia.com/cuda/{SMALL_SLUG}/index.html", enc,
    )
    return toc, sections, chunks


@pytest.fixture(scope="module")
def large_doc_parsed(enc):
    """Parse the large doc once for the module."""
    from doc_retrieval.html_parser import parse_html_doc

    html = (DATA_DIR / LARGE_SLUG / "index.html").read_text(encoding="utf-8")
    toc, sections, chunks = parse_html_doc(
        html, LARGE_SLUG,
        f"https://docs.nvidia.com/cuda/{LARGE_SLUG}/index.html", enc,
    )
    return toc, sections, chunks


@pytest.fixture(scope="module")
def runtime_dir(tmp_path_factory, small_doc_parsed, large_doc_parsed):
    """Create a temporary runtime directory with parsed data and BM25 index."""
    from rank_bm25 import BM25Okapi

    tmpdir = tmp_path_factory.mktemp("doc_retrieval_test")
    chunks_dir = tmpdir / "chunks"
    chunks_dir.mkdir()
    index_dir = tmpdir / "index"
    index_dir.mkdir()

    # Combine chunks from both docs
    all_toc = list(small_doc_parsed[0]) + list(large_doc_parsed[0])
    all_sections = list(small_doc_parsed[1]) + list(large_doc_parsed[1])
    all_chunks = list(small_doc_parsed[2]) + list(large_doc_parsed[2])

    # Write JSONL files
    with open(chunks_dir / "toc.jsonl", "w", encoding="utf-8") as f:
        for entry in all_toc:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(chunks_dir / "sections.jsonl", "w", encoding="utf-8") as f:
        for sec in all_sections:
            f.write(json.dumps(sec, ensure_ascii=False) + "\n")

    with open(chunks_dir / "all_chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Build BM25 index
    def _tokenize(text: str) -> list[str]:
        text = re.sub(r"(\w+)\.([xyzw])\b", r"\1_\2", text)
        return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+", text.lower())

    corpus = [_tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(corpus)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # Write chunk_ids.json (needed by FAISS path, even if we don't build FAISS)
    with open(index_dir / "chunk_ids.json", "w") as f:
        json.dump([c["chunk_id"] for c in all_chunks], f)

    return tmpdir


@pytest.fixture(scope="module")
def searcher(runtime_dir):
    """DocSearcher backed by test data."""
    from doc_retrieval.searcher import DocSearcher

    s = DocSearcher(index_dir=runtime_dir / "index")
    s._runtime_root = runtime_dir
    return s


# ===========================================================================
# HTML Parse Pipeline Tests
# ===========================================================================

@_needs_html(SMALL_SLUG)
class TestHTMLParsePipeline:
    """Verify parsing produces correct TOC, sections, and chunks."""

    def test_small_doc_toc_structure(self, small_doc_parsed):
        toc, sections, chunks = small_doc_parsed
        assert len(toc) > 0, "TOC should have entries"
        # Top-level sections should have no parent
        top_level = [t for t in toc if t["parent_section_id"] is None]
        assert len(top_level) >= 1

    def test_small_doc_sections_match_toc(self, small_doc_parsed):
        toc, sections, chunks = small_doc_parsed
        toc_ids = {t["section_id"] for t in toc}
        section_ids = {s["section_id"] for s in sections}
        assert toc_ids == section_ids, "TOC and sections should have same IDs"

    def test_small_doc_chunks_reference_valid_sections(self, small_doc_parsed):
        toc, sections, chunks = small_doc_parsed
        section_ids = {s["section_id"] for s in sections}
        for chunk in chunks:
            assert chunk["section_id"] in section_ids, \
                f"Chunk references unknown section: {chunk['section_id']}"

    def test_small_doc_chunks_have_required_fields(self, small_doc_parsed):
        toc, sections, chunks = small_doc_parsed
        required = {"chunk_id", "doc_id", "section_id", "source_url",
                     "title", "section_path", "chunk_index", "text"}
        for chunk in chunks:
            missing = required - set(chunk.keys())
            assert not missing, f"Chunk missing fields: {missing}"

    def test_small_doc_deep_links_have_anchors(self, small_doc_parsed):
        toc, sections, chunks = small_doc_parsed
        for t in toc:
            assert "#" in t["deep_link"], \
                f"Deep link missing anchor: {t['deep_link']}"

    @_needs_html(LARGE_SLUG)
    def test_large_doc_has_many_sections(self, large_doc_parsed):
        toc, sections, chunks = large_doc_parsed
        assert len(toc) > 500, f"Expected >500 TOC entries, got {len(toc)}"
        assert len(chunks) > 1000, f"Expected >1000 chunks, got {len(chunks)}"

    @_needs_html(LARGE_SLUG)
    def test_large_doc_parent_child_consistency(self, large_doc_parsed):
        toc, sections, chunks = large_doc_parsed
        by_id = {t["section_id"]: t for t in toc}
        for t in toc:
            # Every child should reference this node as parent
            for child_id in t["children"]:
                assert child_id in by_id, \
                    f"Child {child_id} not found in TOC"
                assert by_id[child_id]["parent_section_id"] == t["section_id"], \
                    f"Child {child_id} parent mismatch"

    @_needs_html(LARGE_SLUG)
    def test_content_is_lightweight_html(self, large_doc_parsed):
        toc, sections, chunks = large_doc_parsed
        # Sample some sections — content should have HTML tags but no nav/script
        for sec in sections[:50]:
            content = sec["content"]
            if not content:
                continue
            assert "<script" not in content, "Content should not have <script>"
            assert "<nav" not in content, "Content should not have <nav>"
            assert "<style" not in content, "Content should not have <style>"


# ===========================================================================
# BM25 Search Tests
# ===========================================================================

@_needs_html(LARGE_SLUG)
class TestBM25Search:
    """BM25 keyword search on real parsed data."""

    def test_shared_memory_query(self, searcher):
        results = searcher.search_bm25("shared memory bank conflicts", top_k=5)
        assert len(results) >= 3
        assert results[0].score > 0
        texts = " ".join(r.text.lower() for r in results)
        assert "shared memory" in texts or "bank" in texts

    def test_thread_hierarchy_query(self, searcher):
        results = searcher.search_bm25("thread block cluster hierarchy", top_k=5)
        assert len(results) >= 1
        texts = " ".join(r.text.lower() for r in results)
        assert "thread" in texts

    def test_cuda_keyword_exact_match(self, searcher):
        results = searcher.search_bm25("__syncthreads", top_k=5)
        assert len(results) >= 1
        texts = " ".join(r.text.lower() for r in results)
        assert "__syncthreads" in texts or "syncthreads" in texts

    def test_blackwell_tuning(self, searcher):
        results = searcher.search_bm25("Blackwell tuning guide", top_k=5)
        assert len(results) >= 1
        doc_ids = [r.doc_id for r in results]
        assert SMALL_SLUG in doc_ids, \
            f"Expected {SMALL_SLUG} in results, got {doc_ids}"

    def test_results_have_section_id(self, searcher):
        """Every BM25 result should carry a section_id for navigation."""
        results = searcher.search_bm25("memory hierarchy", top_k=5)
        assert len(results) >= 1
        # Load chunks to check section_id exists
        chunks = searcher._load_chunks()
        chunk_map = {c["chunk_id"]: c for c in chunks}
        for r in results:
            chunk = chunk_map.get(r.chunk_id)
            assert chunk is not None
            assert "section_id" in chunk, \
                f"Chunk {r.chunk_id} missing section_id"

    def test_empty_query_returns_empty(self, searcher):
        results = searcher.search_bm25("", top_k=5)
        assert len(results) == 0


# ===========================================================================
# Navigation Tests on Real Data
# ===========================================================================

@_needs_html(LARGE_SLUG)
class TestNavigationOnRealData:
    """Navigation (browse_toc, read_section) on real parsed CUDA docs."""

    def test_browse_top_level(self, searcher):
        result = searcher.browse_toc(LARGE_SLUG)
        assert isinstance(result, list)
        assert len(result) > 0
        # Top-level entries should have titles
        for entry in result:
            assert "section_id" in entry
            assert "title" in entry
            assert "children_count" in entry

    def test_browse_expand_chapter(self, searcher):
        # First get top-level
        top = searcher.browse_toc(LARGE_SLUG)
        # Find "programming-model" or similar chapter with children
        chapter = next(
            (t for t in top if t["children_count"] > 0), None
        )
        assert chapter is not None, "Should have at least one chapter with children"

        # Expand it
        expanded = searcher.browse_toc(LARGE_SLUG, chapter["section_id"])
        assert "children" in expanded
        assert len(expanded["children"]) == chapter["children_count"]

    def test_browse_deep_expand(self, searcher):
        result = searcher.browse_toc(LARGE_SLUG, depth=3)
        assert isinstance(result, list)
        # At depth=3, some entries should have nested children
        has_nested = False
        for entry in result:
            if "children" in entry:
                for child in entry["children"]:
                    if "children" in child:
                        has_nested = True
                        break
        assert has_nested, "Depth=3 should produce nested children"

    def test_read_section_content(self, searcher):
        result = searcher.read_section(LARGE_SLUG, "programming-model")
        assert result is not None
        assert "content" in result
        assert "title" in result
        assert "nav" in result
        assert "deep_link" in result
        assert len(result["content"]) > 0

    def test_read_section_nav_context(self, searcher):
        """Read a section and verify nav has parent + siblings."""
        # Browse to find a section with siblings
        top = searcher.browse_toc(LARGE_SLUG)
        chapter_with_children = next(
            (t for t in top if t["children_count"] >= 2), None
        )
        assert chapter_with_children is not None

        expanded = searcher.browse_toc(LARGE_SLUG, chapter_with_children["section_id"])
        second_child_id = expanded["children"][1]["section_id"]

        result = searcher.read_section(LARGE_SLUG, second_child_id)
        assert result is not None
        assert result["nav"]["parent"] == chapter_with_children["section_id"]
        assert result["nav"]["prev_sibling"] is not None
        # May or may not have next_sibling depending on position

    def test_read_nonexistent_section(self, searcher):
        result = searcher.read_section(LARGE_SLUG, "this-section-does-not-exist")
        assert result is None

    def test_browse_nonexistent_doc(self, searcher):
        result = searcher.browse_toc("nonexistent-doc-slug")
        assert result == []


# ===========================================================================
# Search → Navigate Workflow Tests
# ===========================================================================

@_needs_html(LARGE_SLUG)
class TestSearchThenNavigate:
    """End-to-end: search → find section_id → browse/read for context."""

    def test_search_then_read_section(self, searcher):
        """Search for a term, then use section_id to read full section."""
        results = searcher.search_bm25("memory hierarchy", top_k=3)
        assert len(results) >= 1

        # Get section_id from the chunk
        chunks = searcher._load_chunks()
        chunk_map = {c["chunk_id"]: c for c in chunks}
        top_chunk = chunk_map[results[0].chunk_id]
        section_id = top_chunk.get("section_id")
        assert section_id is not None

        # Read the full section
        doc_id = top_chunk["doc_id"]
        section = searcher.read_section(doc_id, section_id)
        assert section is not None
        assert len(section["content"]) > 0

    def test_search_then_browse_parent(self, searcher):
        """Search, find section, go up to parent chapter via nav."""
        results = searcher.search_bm25("thread block", top_k=3)
        assert len(results) >= 1

        chunks = searcher._load_chunks()
        chunk_map = {c["chunk_id"]: c for c in chunks}
        top_chunk = chunk_map[results[0].chunk_id]
        section_id = top_chunk.get("section_id")
        doc_id = top_chunk["doc_id"]

        section = searcher.read_section(doc_id, section_id)
        if section and section["nav"]["parent"]:
            parent_toc = searcher.browse_toc(doc_id, section["nav"]["parent"])
            assert "children" in parent_toc or "section_id" in parent_toc

    def test_search_then_read_sibling(self, searcher):
        """Search, find section, navigate to next sibling."""
        results = searcher.search_bm25("performance guidelines", top_k=3)
        assert len(results) >= 1

        chunks = searcher._load_chunks()
        chunk_map = {c["chunk_id"]: c for c in chunks}
        top_chunk = chunk_map[results[0].chunk_id]
        section_id = top_chunk.get("section_id")
        doc_id = top_chunk["doc_id"]

        section = searcher.read_section(doc_id, section_id)
        if section and section["nav"]["next_sibling"]:
            next_sec = searcher.read_section(doc_id, section["nav"]["next_sibling"])
            assert next_sec is not None
            assert next_sec["title"] != section["title"]


# ===========================================================================
# Dense Search Tests (require embedding service)
# ===========================================================================

def _embedding_service_available() -> bool:
    """Check if the local embedding service is running."""
    try:
        import httpx
        resp = httpx.get("http://localhost:46982/v1/models", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not _embedding_service_available(),
    reason="Embedding service not available at localhost:46982",
)
@_needs_html(LARGE_SLUG)
class TestDenseSearch:
    """Dense vector search — requires the embedding service."""

    @pytest.fixture(scope="class")
    def dense_searcher(self, runtime_dir):
        """DocSearcher with FAISS index built from embeddings."""
        import faiss
        import numpy as np
        from doc_retrieval.embeddings import create_client
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=runtime_dir / "index")
        s._runtime_root = runtime_dir

        # Build FAISS index if not already built
        faiss_path = runtime_dir / "index" / "faiss.index"
        if not faiss_path.exists():
            chunks = s._load_chunks()
            client = create_client()
            texts = [c["text"][:500] for c in chunks]  # truncate for speed

            # Embed in batches
            all_vecs = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                vecs = client.embed_batch(batch)
                all_vecs.extend(vecs)

            matrix = np.array(all_vecs, dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            matrix = matrix / norms

            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            faiss.write_index(index, str(faiss_path))

        return s

    def test_semantic_search(self, dense_searcher):
        results = dense_searcher.search_dense(
            "how to optimize GPU memory access patterns", top_k=5
        )
        assert len(results) >= 3
        assert results[0].score > 0.3

    def test_hybrid_search(self, dense_searcher):
        results = dense_searcher.search_hybrid(
            "shared memory bank conflicts", top_k=5
        )
        assert len(results) >= 3
