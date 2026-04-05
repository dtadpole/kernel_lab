"""Integration tests for BM25 retrieval on NVIDIA CUDA docs.

Each test issues a representative query and asserts that the top results
come from the expected documents and/or sections.
"""

from __future__ import annotations

import pytest

from doc_retrieval.searcher import DocSearcher, SearchResult

TOP_K = 5


@pytest.fixture(scope="module")
def searcher() -> DocSearcher:
    """Shared searcher instance (lazy-loads indices once per module)."""
    return DocSearcher()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _top_doc_ids(results: list[SearchResult]) -> list[str]:
    return [r.doc_id for r in results]


def _top_titles(results: list[SearchResult]) -> list[str]:
    return [r.title for r in results]


def _any_section_contains(results: list[SearchResult], substring: str) -> bool:
    sub = substring.lower()
    return any(sub in r.section_path.lower() for r in results)


def _any_text_contains(results: list[SearchResult], substring: str) -> bool:
    sub = substring.lower()
    return any(sub in r.text.lower() for r in results)


# =======================================================================
# BM25 Tests
# =======================================================================

class TestBM25:
    """BM25 keyword search — good for exact API names and identifiers."""

    def test_shared_memory_bank_conflicts(self, searcher: DocSearcher):
        """Querying 'shared memory bank conflicts' should surface the
        Programming Guide or Best Practices Guide sections on shared memory."""
        results = searcher.search("shared memory bank conflicts", TOP_K)

        assert len(results) >= 3, f"Expected >=3 results, got {len(results)}"
        assert results[0].score > 0, "Top result should have positive score"
        assert _any_text_contains(results, "bank"), \
            "At least one result should mention 'bank'"
        assert _any_text_contains(results, "shared memory"), \
            "At least one result should mention 'shared memory'"

    def test_ptx_ld_global(self, searcher: DocSearcher):
        """PTX instruction 'ld.global' should surface the PTX ISA doc."""
        results = searcher.search("ld.global instruction", TOP_K)

        assert len(results) >= 1
        assert _any_text_contains(results, "ld.global") or \
            _any_text_contains(results, "ld."), \
            "At least one result should mention the ld instruction"

    def test_tma_descriptor(self, searcher: DocSearcher):
        """'TMA descriptor' should surface TMA sections from Programming Guide or PTX ISA."""
        results = searcher.search("TMA tensor memory accelerator descriptor", TOP_K)

        assert len(results) >= 1
        doc_ids = _top_doc_ids(results)
        assert "cuda-c-programming-guide" in doc_ids or "parallel-thread-execution" in doc_ids, \
            f"Expected programming guide or PTX ISA in results, got {doc_ids}"

    def test_blackwell_tuning(self, searcher: DocSearcher):
        """'blackwell tuning' should surface the Blackwell Tuning Guide."""
        results = searcher.search("blackwell tuning guide SM120", TOP_K)

        assert len(results) >= 1
        doc_ids = _top_doc_ids(results)
        assert "blackwell-tuning-guide" in doc_ids or "blackwell-compatibility-guide" in doc_ids, \
            f"Expected blackwell doc in results, got {doc_ids}"

    def test_nvcc_compiler_options(self, searcher: DocSearcher):
        """'nvcc -arch sm_90' should surface relevant content."""
        results = searcher.search("nvcc -arch sm_90", TOP_K)

        assert len(results) >= 1
        assert _any_text_contains(results, "nvcc") or \
            _any_text_contains(results, "-arch"), \
            "At least one result should mention nvcc or -arch"
