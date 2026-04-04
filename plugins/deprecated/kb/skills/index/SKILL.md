---
name: index
description: Manage the CUDA documentation search index — download, build, rebuild, or nuke
user-invocable: true
---

# KB Index Management

Download NVIDIA HTML docs, parse into chunks, and build search indices.

## Commands

All commands use the project venv:

```bash
cd /home/centos/kernel_lab
```

### Full rebuild (download → parse → index)

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval download && \
doc_retrieval/.venv/bin/python -m doc_retrieval parse && \
doc_retrieval/.venv/bin/python -m doc_retrieval index
```

### Download HTML docs from NVIDIA

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval download
```

Raw docs are saved to `data/nvidia-docs/html/` (in-repo, single source of truth).

### Parse into chunks

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval parse
```

Parses HTML (via BeautifulSoup) into search chunks, TOC trees, and full sections. Output: `~/.doc_retrieval/chunks/`.

### Build search indices

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval index               # both BM25 + FAISS
doc_retrieval/.venv/bin/python -m doc_retrieval index --only bm25   # BM25 only (no embeddings needed)
doc_retrieval/.venv/bin/python -m doc_retrieval index --only dense  # FAISS only (requires embedding service)
```

Output: `~/.doc_retrieval/index/`.

### Nuke derived artifacts

```bash
rm -rf ~/.doc_retrieval/chunks ~/.doc_retrieval/index
```

This removes parsed chunks and search indices. Raw docs in `data/nvidia-docs/` are untouched. Re-run `parse && index` to rebuild.

## Storage Layout

```
data/nvidia-docs/                # in-repo, committed
  html/{slug}/index.html         # Sphinx HTML pages
  html/{slug}/_images/           # referenced images

~/.doc_retrieval/                # runtime, derived (safe to delete)
  chunks/                        # all_chunks.jsonl, toc.jsonl, sections.jsonl
  index/                         # bm25.pkl, faiss.index, chunk_ids.json
```
