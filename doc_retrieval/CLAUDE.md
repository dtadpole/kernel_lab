# CLAUDE.md

This file provides guidance to Claude Code when working with code in this directory.

## What this is

`doc_retrieval` is a document retrieval system for NVIDIA CUDA Toolkit documentation.
It downloads, parses, indexes, and searches docs from `docs.nvidia.com/cuda/`,
providing BM25 (sparse) and dense (FAISS + API embeddings) retrieval with hybrid
fusion via Reciprocal Rank Fusion (RRF).

Designed for integration with `cuda_agent` as MCP tools so the optimization agent
can look up CUDA APIs, best practices, PTX instructions, and memory model semantics.

## Commands

### Download NVIDIA docs
```bash
cd /home/centos/kernel_lab
doc_retrieval/.venv/bin/python -m doc_retrieval download --tier 1
doc_retrieval/.venv/bin/python -m doc_retrieval download --tier all
```

### Parse downloaded docs into chunks
```bash
doc_retrieval/.venv/bin/python -m doc_retrieval parse
```

### Build search indices
```bash
doc_retrieval/.venv/bin/python -m doc_retrieval index
doc_retrieval/.venv/bin/python -m doc_retrieval index --only bm25
doc_retrieval/.venv/bin/python -m doc_retrieval index --only dense
```

### Search (CLI testing)
```bash
doc_retrieval/.venv/bin/python -m doc_retrieval search "shared memory bank conflicts" --mode hybrid
```

## Architecture

### Module responsibilities

- **`downloader.py`** -- Downloads PDFs from `docs.nvidia.com/cuda/pdf/` and crawls HTML pages. Supports tiered downloads (tier 1 = essential, tier 2 = important, tier 3 = remaining).
- **`parser.py`** -- Uses Docling (IBM) to convert PDF/HTML into structured Markdown chunks with metadata (title, section_path, source URL). Handles image extraction.
- **`indexer.py`** -- Builds BM25 index (rank-bm25) and FAISS dense index from parsed chunks. Caches embeddings to avoid re-embedding unchanged content.
- **`searcher.py`** -- Unified search interface: BM25, dense, and hybrid (RRF) modes. Lazy-loads indices on first query.
- **`embeddings.py`** -- Thin API client for embedding providers (OpenAI, Voyage). Batched requests with rate limiting.
- **`config.py`** -- Hydra config loader (same pattern as `cuda_agent/config.py`).
- **`cli.py`** -- CLI subcommands: download, parse, index, search.

### Key design decisions

- **Docling for parsing** -- Single tool handles both PDF and HTML with image support and structured output.
- **API-based embeddings** -- Uses OpenAI `text-embedding-3-small` by default; configurable via Hydra.
- **FAISS for vectors** -- `IndexFlatIP` with normalized vectors for cosine similarity. Corpus is small enough (~50K chunks) for flat index.
- **Hybrid search via RRF** -- Reciprocal Rank Fusion combines BM25 and dense results without score calibration.
- **Lazy loading** -- Indices loaded on first search call to avoid startup cost when tools aren't used.

### Storage layout

```
~/.doc_retrieval/
  raw/pdfs/          # Downloaded PDF files
  raw/html/          # Crawled HTML pages as JSON
  chunks/            # all_chunks.jsonl
  index/             # bm25.pkl, faiss.index, chunk_ids.json, embedding_cache.npz
```
