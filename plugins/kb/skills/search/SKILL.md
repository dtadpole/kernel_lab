---
name: search
description: Search NVIDIA CUDA Toolkit documentation for programming concepts, API references, and optimization techniques
user-invocable: true
argument-hint: <query>
---

# CUDA Documentation Search

Search indexed NVIDIA CUDA Toolkit documentation using the knowledge-search MCP tools.

## Available Tools

- **search_docs** — BM25/dense/hybrid search over all indexed docs
- **lookup_doc_section** — Retrieve full text from a URL (from search results)
- **browse_toc** — Browse document table of contents by doc_id
- **read_section** — Read a specific section with navigation context

## Workflow

1. Start with `search_docs` using a natural language query: $ARGUMENTS
2. Review the results — each has a `url` and `section_path`
3. To read more context, use `read_section` with the doc_id and section anchor
4. To explore nearby sections, use `browse_toc` to see the TOC structure
5. Navigate using `nav.parent`, `nav.prev_sibling`, `nav.next_sibling` from read_section results

## Available Documents

- `cuda-c-programming-guide` — CUDA C++ Programming Guide
- `parallel-thread-execution` — PTX ISA Reference
- `cuda-c-best-practices-guide` — CUDA C++ Best Practices Guide
- `inline-ptx-assembly` — Inline PTX Assembly in CUDA
- `blackwell-tuning-guide` — Blackwell Tuning Guide
- `blackwell-compatibility-guide` — Blackwell Compatibility Guide
