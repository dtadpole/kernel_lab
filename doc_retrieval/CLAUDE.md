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
- **`parser.py`** -- Orchestrates parsing: PDFs via Docling, HTML via BeautifulSoup (html_parser.py). Outputs all_chunks.jsonl, toc.jsonl, sections.jsonl.
- **`html_parser.py`** -- BeautifulSoup-based parser for Sphinx HTML. Extracts sections with anchor IDs, builds TOC tree, produces lightweight HTML chunks for dual-layer index.
- **`indexer.py`** -- Builds BM25 index (rank-bm25) and FAISS dense index from parsed chunks. Caches embeddings to avoid re-embedding unchanged content.
- **`searcher.py`** -- Unified search interface: BM25, dense, and hybrid (RRF) modes. TOC browsing and section reading for navigation. Lazy-loads all data on first use.
- **`embeddings.py`** -- Thin API client for embedding providers (OpenAI, Voyage). Batched requests with rate limiting.
- **`config.py`** -- Hydra config loader (same pattern as `cuda_agent/config.py`).
- **`cli.py`** -- CLI subcommands: download, parse, index, search, browse, read.

### Key design decisions

- **Dual-layer HTML index** -- Sections (full content, lightweight HTML) for reading + 512-token chunks for BM25/FAISS search. HTML parsed with BeautifulSoup to preserve anchor IDs for navigation. PDFs continue using Docling.
- **API-based embeddings** -- Uses OpenAI `text-embedding-3-small` by default; configurable via Hydra.
- **FAISS for vectors** -- `IndexFlatIP` with normalized vectors for cosine similarity. Corpus is small enough (~50K chunks) for flat index.
- **Hybrid search via RRF** -- Reciprocal Rank Fusion combines BM25 and dense results without score calibration.
- **Lazy loading** -- Indices loaded on first search call to avoid startup cost when tools aren't used.

### Storage layout

Raw source documents live in the **repo** (committed to git).
Derived artifacts live in the **runtime** directory.

```
doc_retrieval/data/raw/              # IN REPO — single source of truth
  pdfs/
    CUDA_C_Programming_Guide.pdf
    ptx_isa_9.2.pdf
    ...
  html/
    cuda-c-programming-guide/        # one folder per doc (slug name)
      index.html                     # full single-page Sphinx HTML
      _images/                       # referenced images
        grid-of-thread-blocks.png
        memory-hierarchy.png
        ...
    parallel-thread-execution/
      index.html
      _images/
        ...

~/.doc_retrieval/                    # RUNTIME — derived artifacts only
  chunks/
    all_chunks.jsonl   # Search-layer chunks (PDF + HTML)
    toc.jsonl          # HTML document TOC trees
    sections.jsonl     # HTML full section content (reading layer)
  index/             # bm25.pkl, faiss.index, chunk_ids.json, embedding_cache.npz
```

Convention:
- **Raw docs (PDFs, HTML+images)** → `doc_retrieval/data/raw/` in repo. Never duplicated to runtime.
- **HTML folder structure** → `{slug}/index.html` + `{slug}/_images/`. One folder per doc.
- **Derived artifacts (chunks, indices)** → `~/.doc_retrieval/` (or `DOC_RETRIEVAL_ROOT` env var). Not committed.
- Downloader writes to `doc_retrieval/data/raw/`. Parser reads from `doc_retrieval/data/raw/`, writes to runtime.
- PDF filenames match NVIDIA's naming (e.g. `CUDA_C_Programming_Guide.pdf`).
- HTML slugs match NVIDIA's URL path (e.g. `cuda-c-programming-guide` from `docs.nvidia.com/cuda/cuda-c-programming-guide/`).
