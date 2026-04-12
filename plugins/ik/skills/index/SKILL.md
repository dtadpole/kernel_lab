---
name: index
description: Manage the CUDA documentation search index — download, build, rebuild, or nuke
user-invocable: true
argument-hint: <build|rebuild|nuke>
---

# KB Index Management

Manage the CUDA documentation search index for `ik:docs`.

## Commands

```bash
cd /home/zhenc/kernel_lab
```

### build — download docs, parse, and create index

```bash
.venv/bin/python -m doc_retrieval build
```

Downloads NVIDIA HTML docs, parses into chunks, and builds BM25 search index.
Safe to re-run — skips already-downloaded docs.

**Note:** On proxy-restricted hosts (e.g. Meta devvms), `build` may fail
downloading tiktoken data. Use SSH localhost:
```bash
SSH_AUTH_SOCK=/run/user/$(id -u)/ssh-agent.socket ssh localhost \
  "cd ~/kernel_lab && source .venv/bin/activate && python -m doc_retrieval build"
```

### rebuild — nuke + build

```bash
.venv/bin/python -m doc_retrieval rebuild
```

Deletes all derived artifacts (chunks + index), then does a full build.
Raw HTML docs in `data/nvidia-docs/` are kept.

### nuke — delete derived artifacts

```bash
.venv/bin/python -m doc_retrieval nuke
```

Removes `~/.doc_retrieval/chunks/` and `~/.doc_retrieval/index/`.
Raw HTML docs are untouched. Run `build` to recreate.

## Storage Layout

```
data/nvidia-docs/                # in-repo, committed
  html/{slug}/index.html         # Sphinx HTML pages
  html/{slug}/_images/           # referenced images

~/.doc_retrieval/                # runtime, derived (safe to nuke)
  chunks/                        # all_chunks.jsonl, toc.jsonl, sections.jsonl
  index/                         # bm25.pkl
```
