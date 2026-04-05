# doc_retrieval

This file provides guidance when working with code in this directory.

## What this is

`doc_retrieval` is the search engine behind the `ik` plugin's docs skill. It downloads,
parses, indexes, and searches NVIDIA CUDA Toolkit documentation from
`docs.nvidia.com/cuda/`, providing BM25 keyword retrieval.

The `ik` plugin (`plugins/ik/`) exposes this functionality to Claude Code via two
skills: `docs` (search/read/browse) and `index` (download/parse/index).
All skills invoke CLI commands — no MCP server.

## Commands

All commands use the project venv:

```bash
cd /home/zhenc/kernel_lab
```

### Download NVIDIA docs
```bash
.venv/bin/python -m doc_retrieval download
```

### Parse downloaded docs into chunks
```bash
.venv/bin/python -m doc_retrieval parse
```

### Build BM25 index
```bash
.venv/bin/python -m doc_retrieval index
```

### Find (search documentation)
```bash
.venv/bin/python -m doc_retrieval find "shared memory bank conflicts"
```

### Browse / Read (navigate documentation)
```bash
.venv/bin/python -m doc_retrieval browse cuda-c-programming-guide --depth 1
.venv/bin/python -m doc_retrieval read cuda-c-programming-guide shared-memory
```

## Architecture

### Module responsibilities

- **`cli.py`** — CLI subcommands: download, parse, index, find, browse, read.
- **`downloader.py`** — Crawls Sphinx HTML pages from `docs.nvidia.com/cuda/`. Saves to `data/nvidia-docs/html/{slug}/`.
- **`parser.py`** — Parses HTML via BeautifulSoup (html_parser.py). Outputs all_chunks.jsonl, toc.jsonl, sections.jsonl.
- **`html_parser.py`** — BeautifulSoup-based parser for Sphinx HTML. Extracts sections with anchor IDs, builds TOC tree, produces lightweight HTML chunks for dual-layer index.
- **`indexer.py`** — Builds BM25 index (rank-bm25) from parsed chunks.
- **`searcher.py`** — BM25 search interface. TOC browsing and section reading for navigation. Lazy-loads all data on first use.
- **`config.py`** — Hydra config loader. Config lives at `conf/doc_retrieval/default.yaml`.

### Key design decisions

- **HTML-only, dual-layer index** — Sections (full content, lightweight HTML) for reading + 512-token chunks for BM25 search. HTML parsed with BeautifulSoup to preserve anchor IDs for navigation.
- **BM25 keyword search** — Uses rank-bm25 with CUDA-aware tokenization (preserves identifiers like `__syncthreads`, `threadIdx.x`).
- **Lazy loading** — Index loaded on first search call to avoid startup cost.

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
```

### Relationship to plugins/ik

```
plugins/ik/                          # Claude Code plugin (thin interface)
  skills/docs/SKILL.md              # → calls: find, read, browse
  skills/index/SKILL.md             # → calls: download, parse, index

doc_retrieval/                       # Python package (engine)
  cli.py                            # CLI entry point: python -m doc_retrieval <cmd>
  ...
```

The `ik` plugin consumes `doc_retrieval` through the CLI interface
(`python -m doc_retrieval`). The plugin provides the skill definitions;
the package provides the implementation.
