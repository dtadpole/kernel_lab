---
name: index
description: Manage the CUDA documentation search index — download, build, rebuild, or nuke
user-invocable: true
---

# KB Index Management

Download NVIDIA docs, parse into chunks, and build search indices.

## Commands

All commands use the project venv:

```bash
cd /home/centos/kernel_lab
```

### Full rebuild (download → parse → index)

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval download --tier all && \
doc_retrieval/.venv/bin/python -m doc_retrieval parse && \
doc_retrieval/.venv/bin/python -m doc_retrieval index
```

### Download raw docs from NVIDIA

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval download --tier 1        # essential docs only
doc_retrieval/.venv/bin/python -m doc_retrieval download --tier all      # everything
doc_retrieval/.venv/bin/python -m doc_retrieval download --html-only     # skip PDFs
doc_retrieval/.venv/bin/python -m doc_retrieval download --pdf-only      # skip HTML
```

Tiers: 1 = essential (Programming Guide, PTX ISA, Best Practices), 2 = important (cuBLAS, tuning guides), 3 = remaining.

Raw docs are saved to `doc_retrieval/data/raw/` (in-repo, single source of truth).

### Parse into chunks

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval parse
```

Parses PDFs (via Docling) and HTML (via BeautifulSoup) into search chunks, TOC trees, and full sections. Output: `~/.doc_retrieval/chunks/`.

### Build search indices

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval index               # both BM25 + FAISS
doc_retrieval/.venv/bin/python -m doc_retrieval index --only bm25   # BM25 only (no embeddings needed)
doc_retrieval/.venv/bin/python -m doc_retrieval index --only dense  # FAISS only (requires embedding service)
```

Output: `~/.doc_retrieval/index/`.

### Nuke derived artifacts

```bash
rm -rf ~/.doc_retrieval
```

This removes all chunks and indices. Raw docs in `doc_retrieval/data/raw/` are untouched. Re-run `parse && index` to rebuild.

## Storage Layout

```
doc_retrieval/data/raw/          # in-repo, committed
  pdfs/*.pdf                     # raw PDFs from NVIDIA
  html/{slug}/index.html         # Sphinx HTML pages
  html/{slug}/_images/           # referenced images

~/.doc_retrieval/                # runtime, derived (safe to delete)
  chunks/                        # all_chunks.jsonl, toc.jsonl, sections.jsonl
  index/                         # bm25.pkl, faiss.index, chunk_ids.json
```
