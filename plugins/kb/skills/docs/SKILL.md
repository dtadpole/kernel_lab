---
name: docs
description: Find and read NVIDIA CUDA Toolkit documentation — programming guides, PTX ISA, best practices, tuning guides
user-invocable: true
argument-hint: <query>
---

# CUDA Documentation

Search, read, and browse indexed NVIDIA CUDA Toolkit documentation via CLI.

## Commands

All commands use the project venv:

```bash
cd /home/centos/kernel_lab
```

### Find — search for documentation

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval find "shared memory bank conflicts"
doc_retrieval/.venv/bin/python -m doc_retrieval find "TMA descriptor" --mode bm25 --top-k 10
```

Modes: `hybrid` (default, best quality), `bm25` (keyword), `dense` (semantic).

Each result includes `doc_id` and `section_id` for follow-up with `read` or `browse`.

### Read — read a full section

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval read cuda-c-programming-guide shared-memory
doc_retrieval/.venv/bin/python -m doc_retrieval read parallel-thread-execution data-movement-and-conversion-instructions-ld
```

Returns full section content with navigation context (`nav.parent`, `nav.prev_sibling`, `nav.next_sibling`).

### Browse — explore document structure

```bash
doc_retrieval/.venv/bin/python -m doc_retrieval browse cuda-c-programming-guide
doc_retrieval/.venv/bin/python -m doc_retrieval browse cuda-c-programming-guide --section-id performance-guidelines --depth 3
```

## Workflow

1. `find` with a natural language query: $ARGUMENTS
2. Pick a result and `read` its full section using doc_id + section_id
3. Use `nav.next_sibling` / `nav.parent` from the read result to continue reading
4. Use `browse` to see the TOC structure around a section

## Available Documents

- `cuda-c-programming-guide` — CUDA C++ Programming Guide
- `parallel-thread-execution` — PTX ISA Reference
- `cuda-c-best-practices-guide` — CUDA C++ Best Practices Guide
- `inline-ptx-assembly` — Inline PTX Assembly in CUDA
- `blackwell-tuning-guide` — Blackwell Tuning Guide
- `blackwell-compatibility-guide` — Blackwell Compatibility Guide
- `nvrtc` — NVRTC (Runtime Compilation)
- `ptx-compiler-api` — PTX Compiler APIs
- `cutile-python` — CuTile Python API
- `tile-ir` — Tile IR Reference
