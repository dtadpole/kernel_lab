# doc_retrieval

This file provides guidance when working with code in this directory.

## What this is

`doc_retrieval` is the search engine behind the `kb` plugin. It downloads, parses,
indexes, and searches NVIDIA CUDA Toolkit documentation from `docs.nvidia.com/cuda/`,
providing BM25 (sparse) and FAISS (dense) retrieval with hybrid fusion via
Reciprocal Rank Fusion (RRF).

The `kb` plugin (`plugins/kb/`) exposes this functionality to Claude Code via three
skills: `docs` (search/read/browse), `index` (download/parse/index), and `service`
(embedding service management). All skills invoke CLI commands — no MCP server.

## Commands

All commands use the project venv:

```bash
cd /home/centos/kernel_lab
```

### Download NVIDIA docs
```bash
doc_retrieval/.venv/bin/python -m doc_retrieval download
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

### Find (search documentation)
```bash
doc_retrieval/.venv/bin/python -m doc_retrieval find "shared memory bank conflicts" --mode hybrid
```

### Browse / Read (navigate documentation)
```bash
doc_retrieval/.venv/bin/python -m doc_retrieval browse cuda-c-programming-guide --depth 1
doc_retrieval/.venv/bin/python -m doc_retrieval read cuda-c-programming-guide shared-memory
```

## Architecture

### Module responsibilities

- **`cli.py`** — CLI subcommands: download, parse, index, find, browse, read.
- **`downloader.py`** — Crawls Sphinx HTML pages from `docs.nvidia.com/cuda/`. Saves to `data/nvidia-docs/html/{slug}/`.
- **`parser.py`** — Parses HTML via BeautifulSoup (html_parser.py). Outputs all_chunks.jsonl, toc.jsonl, sections.jsonl.
- **`html_parser.py`** — BeautifulSoup-based parser for Sphinx HTML. Extracts sections with anchor IDs, builds TOC tree, produces lightweight HTML chunks for dual-layer index.
- **`indexer.py`** — Builds BM25 index (rank-bm25) and FAISS dense index from parsed chunks. Caches embeddings to avoid re-embedding unchanged content.
- **`searcher.py`** — Unified search interface: BM25, dense, and hybrid (RRF) modes. TOC browsing and section reading for navigation. Lazy-loads all data on first use.
- **`embeddings.py`** — httpx-based client for the OpenAI-compatible `/v1/embeddings` endpoint. Connects to a local Qwen3-Embedding-4B server (deployed via `kb:service` skill). Batched requests with caching.
- **`config.py`** — Hydra config loader. Config lives at `conf/doc_retrieval/default.yaml`.

### Key design decisions

- **HTML-only, dual-layer index** — Sections (full content, lightweight HTML) for reading + 512-token chunks for BM25/FAISS search. HTML parsed with BeautifulSoup to preserve anchor IDs for navigation.
- **Local embeddings** — Uses Qwen3-Embedding-4B via a Python embedding server on a remote GPU host (`plugins/kb/embed_server/`), managed by the `kb:service` skill.
- **FAISS for vectors** — `IndexFlatIP` with normalized vectors for cosine similarity. Corpus is small enough (~3.5K chunks) for flat index.
- **Hybrid search via RRF** — Reciprocal Rank Fusion combines BM25 and dense results without score calibration.
- **Lazy loading** — Indices loaded on first search call to avoid startup cost.

### Storage layout

Raw source documents live in the **repo** (committed to git).
Derived artifacts live in the **runtime** directory.

```
data/nvidia-docs/                    # IN REPO — single source of truth
  html/
    cuda-c-programming-guide/        # one folder per doc (slug name)
      index.html                     # full single-page Sphinx HTML
      _images/                       # referenced images
    parallel-thread-execution/
      index.html
      _images/
    ...

~/.doc_retrieval/                    # RUNTIME — derived artifacts (safe to delete)
  chunks/
    all_chunks.jsonl                 # search-layer chunks
    toc.jsonl                        # HTML document TOC trees
    sections.jsonl                   # HTML full section content (reading layer)
  index/
    bm25.pkl                         # BM25 index
    faiss.index                      # FAISS dense index
    chunk_ids.json                   # chunk ID mapping
    embedding_cache.npz              # cached embeddings
    metadata.json                    # index metadata
```

### Relationship to plugins/kb

```
plugins/kb/                          # Claude Code plugin (thin interface)
  skills/docs/SKILL.md              # → calls: find, read, browse
  skills/index/SKILL.md             # → calls: download, parse, index
  skills/service/SKILL.md           # → manages: embedding server on GPU host
  embed_server/                      # → Python embedding server code
  deploy/                            # → deploy CLI + systemd unit

doc_retrieval/                       # Python package (engine)
  cli.py                            # CLI entry point: python -m doc_retrieval <cmd>
  ...
```

Both the `kb` plugin and `cuda_agent` consume `doc_retrieval` through the same
CLI interface (`python -m doc_retrieval`). The plugin provides the skill
definitions; the package provides the implementation.
