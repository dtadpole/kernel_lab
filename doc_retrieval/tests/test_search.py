"""Integration tests for BM25 and Dense retrieval on NVIDIA CUDA docs.

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
        results = searcher.search_bm25("shared memory bank conflicts", TOP_K)

        assert len(results) >= 3, f"Expected >=3 results, got {len(results)}"
        assert results[0].score > 0, "Top result should have positive score"
        assert _any_text_contains(results, "bank"), \
            "At least one result should mention 'bank'"
        assert _any_text_contains(results, "shared memory"), \
            "At least one result should mention 'shared memory'"

    def test_cudamemcpyasync(self, searcher: DocSearcher):
        """Exact API name 'cudaMemcpyAsync' should hit the Runtime API doc."""
        results = searcher.search_bm25("cudaMemcpyAsync", TOP_K)

        assert len(results) >= 1
        doc_ids = _top_doc_ids(results)
        assert "cuda-runtime-api" in doc_ids, \
            f"Expected 'cuda-runtime-api' in results, got {doc_ids}"

    def test_ptx_ld_global(self, searcher: DocSearcher):
        """PTX instruction 'ld.global' should surface the PTX ISA doc."""
        results = searcher.search_bm25("ld.global instruction", TOP_K)

        assert len(results) >= 1
        assert _any_text_contains(results, "ld.global") or \
            _any_text_contains(results, "ld."), \
            "At least one result should mention the ld instruction"

    def test_nvcc_compiler_options(self, searcher: DocSearcher):
        """'nvcc -arch sm_90' should surface the NVCC compiler driver doc."""
        results = searcher.search_bm25("nvcc -arch sm_90", TOP_K)

        assert len(results) >= 1
        assert _any_text_contains(results, "nvcc") or \
            _any_text_contains(results, "-arch"), \
            "At least one result should mention nvcc or -arch"

    def test_cublas_gemm(self, searcher: DocSearcher):
        """'cublasSgemm' should surface the cuBLAS library doc."""
        results = searcher.search_bm25("cublasSgemm", TOP_K)

        assert len(results) >= 1
        doc_ids = _top_doc_ids(results)
        assert "cublas-library" in doc_ids, \
            f"Expected 'cublas-library' in results, got {doc_ids}"


# =======================================================================
# Dense (Semantic) Tests
# =======================================================================

class TestDense:
    """Dense vector search — good for conceptual / semantic queries."""

    def test_how_to_avoid_warp_divergence(self, searcher: DocSearcher):
        """Semantic query about warp divergence should surface relevant
        performance guidance from Programming or Best Practices Guide."""
        results = searcher.search_dense(
            "how to avoid warp divergence for better performance", TOP_K
        )

        assert len(results) >= 3
        assert results[0].score > 0.5, \
            f"Top result similarity should be > 0.5, got {results[0].score:.4f}"
        # Should find content about branching/divergence/warps
        assert _any_text_contains(results, "warp") or \
            _any_text_contains(results, "divergen") or \
            _any_text_contains(results, "branch"), \
            "At least one result should discuss warp divergence or branching"

    def test_memory_coalescing_explanation(self, searcher: DocSearcher):
        """Conceptual query about coalesced global memory access should
        surface the relevant guide sections."""
        results = searcher.search_dense(
            "coalesced global memory access pattern alignment requirements",
            TOP_K,
        )

        assert len(results) >= 3
        # Dense results should be about memory access patterns —
        # coalescing, alignment, global memory, or device memory
        assert _any_text_contains(results, "coalesc") or \
            _any_text_contains(results, "global memory") or \
            _any_text_contains(results, "memory access") or \
            _any_text_contains(results, "alignment"), \
            "At least one result should discuss memory access patterns"

    def test_thread_synchronization_primitives(self, searcher: DocSearcher):
        """Semantic query about synchronization should find __syncthreads,
        barriers, or cooperative groups content."""
        results = searcher.search_dense(
            "thread synchronization primitives in CUDA kernels", TOP_K
        )

        assert len(results) >= 3
        assert _any_text_contains(results, "sync") or \
            _any_text_contains(results, "barrier") or \
            _any_text_contains(results, "cooperative"), \
            "At least one result should discuss synchronization"

    def test_occupancy_and_register_pressure(self, searcher: DocSearcher):
        """Query about occupancy should surface tuning or best practices
        content about register usage and occupancy."""
        results = searcher.search_dense(
            "how does register pressure affect occupancy on NVIDIA GPUs", TOP_K
        )

        assert len(results) >= 3
        assert _any_text_contains(results, "occupancy") or \
            _any_text_contains(results, "register"), \
            "At least one result should mention occupancy or registers"

    def test_tensor_core_operations(self, searcher: DocSearcher):
        """Query about tensor cores should surface relevant hardware
        or programming guide content."""
        results = searcher.search_dense(
            "using tensor cores for matrix multiply accumulate", TOP_K
        )

        assert len(results) >= 3
        assert _any_text_contains(results, "tensor") or \
            _any_text_contains(results, "mma") or \
            _any_text_contains(results, "wmma"), \
            "At least one result should discuss tensor cores or MMA"
