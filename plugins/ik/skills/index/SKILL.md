---
name: index
description: Manage the CUDA documentation search index — download, build, rebuild, or nuke
user-invocable: true
---

# KB Index Management

Download NVIDIA HTML docs, parse into chunks, and build BM25 search index.

## Commands

All commands use the project venv:

```bash
cd /home/zhenc/kernel_lab
```

### Full rebuild (download → parse → index)

```bash
.venv/bin/python -m doc_retrieval download && \
.venv/bin/python -m doc_retrieval parse && \
.venv/bin/python -m doc_retrieval index
```

### Download HTML docs from NVIDIA

```bash
.venv/bin/python -m doc_retrieval download
```

Raw docs are saved to `data/nvidia-docs/html/` (in-repo, single source of truth).

### Parse into chunks

```bash
.venv/bin/python -m doc_retrieval parse
```

Parses HTML (via BeautifulSoup) into search chunks, TOC trees, and full sections. Output: `~/.doc_retrieval/chunks/`.

### Build BM25 index

```bash
.venv/bin/python -m doc_retrieval index
```

Output: `~/.doc_retrieval/index/bm25.pkl`.

### Nuke derived artifacts

```bash
rm -rf ~/.doc_retrieval/chunks ~/.doc_retrieval/index
```

This removes parsed chunks and search index. Raw docs in `data/nvidia-docs/` are untouched. Re-run `parse && index` to rebuild.

## Storage Layout

```
data/nvidia-docs/                # in-repo, committed
  html/{slug}/index.html         # Sphinx HTML pages
  html/{slug}/_images/           # referenced images

~/.doc_retrieval/                # runtime, derived (safe to delete)
  chunks/                        # all_chunks.jsonl, toc.jsonl, sections.jsonl
  index/                         # bm25.pkl
```
