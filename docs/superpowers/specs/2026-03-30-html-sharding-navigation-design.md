# HTML Sharding + Navigation Design

Date: 2026-03-30

## Context

The doc_retrieval system indexes 6 NVIDIA CUDA HTML documents (Sphinx-generated, single-page) plus PDFs. The current parser uses Docling for both PDF and HTML, converting to Markdown then chunking. This loses all HTML anchor IDs (`<section id="xxx">`), making it impossible to deep-link search results or navigate document structure.

We need an Agent that can:
1. **Search then navigate** — find a chunk, then browse surrounding context (parent, siblings, children)
2. **Browse then drill down** — start from a TOC tree, select a chapter, expand into sub-sections

## Decision Record

| Decision | Choice | Rationale |
|----------|--------|-----------|
| HTML parser | BeautifulSoup | Sphinx HTML is clean/predictable; Docling loses anchor IDs; BS4 gives full DOM control |
| PDF parser | Docling (unchanged) | PDF structure needs ML-based layout analysis |
| Chunk text format | Lightweight HTML | HtmlRAG (WWW 2025) shows preserving `<code>`, `<table>`, `<strong>`, `<a>` improves RAG quality vs plain text |
| Index strategy | Dual-layer | Search layer (512-token chunks) for precision; reading layer (full sections) for context |
| Storage format | JSONL files | toc.jsonl + sections.jsonl + all_chunks.jsonl in runtime (`~/.doc_retrieval/`) |
| Agent interface | New MCP tools | `cuda_browse_toc` + `cuda_read_section` alongside existing `cuda_search_docs` |

## Section Size Analysis (from actual data)

| Document | Sections | Median | Mean | Max |
|----------|-------:|-------:|-----:|----:|
| CUDA C Programming Guide | 881 | 169 tok | 276 tok | 3,966 tok |
| PTX ISA | 682 | 249 tok | 461 tok | 10,095 tok |
| Best Practices Guide | 139 | 192 tok | 292 tok | 1,781 tok |

~46% of sections are 128-512 tokens (1-2 pages). Only 1-3% exceed 2K tokens.

## Architecture

```
HTML files (Sphinx)           PDF files
    |                            |
    v                            v
BeautifulSoup parser         Docling parser (unchanged)
    |                            |
    +---> toc.jsonl              |
    +---> sections.jsonl         |
    +---> chunks ----------------+---> all_chunks.jsonl
                                          |
                                    +-----+-----+
                                    v           v
                                BM25 index   FAISS index
```

### Parse pipeline (HTML)

1. BeautifulSoup reads `{slug}/index.html`
2. Traverse `<section id="xxx">` tags recursively
3. For each section:
   - Extract `id`, heading text, heading level (h1-h6)
   - Extract direct content (excluding nested `<section>` children)
   - Clean HTML: strip `<nav>`, `<script>`, `<style>`, class/style attributes; keep `<code>`, `<table>`, `<strong>`, `<em>`, `<a>`, `<ul>`, `<ol>`, `<li>`, `<pre>`, `<p>`
   - Record parent-child relationships from nesting
4. Output toc.jsonl, sections.jsonl
5. Chunk large sections (>512 tokens) at paragraph boundaries with 64-token overlap
6. Merge tiny sections (<128 tokens) with next sibling
7. Each chunk carries `section_id` of its source section
8. Output chunks to all_chunks.jsonl (merged with PDF chunks)

## Data Model

### toc.jsonl

One line per section node. Provides the navigation tree.

```json
{
  "doc_id": "cuda-c-programming-guide",
  "section_id": "thread-hierarchy",
  "title": "Thread Hierarchy",
  "heading_level": 2,
  "parent_section_id": "programming-model",
  "children": ["thread-block-clusters", "..."],
  "deep_link": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy"
}
```

### sections.jsonl

One line per section. Full content for the reading layer.

```json
{
  "doc_id": "cuda-c-programming-guide",
  "section_id": "thread-hierarchy",
  "title": "Thread Hierarchy",
  "heading_level": 2,
  "heading_path": ["Programming Model", "Thread Hierarchy"],
  "content": "<p>CUDA threads are organized into...</p><pre><code>dim3 grid(2,2);</code></pre>...",
  "token_count": 843,
  "deep_link": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy"
}
```

### all_chunks.jsonl (extended schema)

Backward-compatible with existing PDF chunks; HTML chunks add `section_id`.

```json
{
  "chunk_id": "a1b2c3d4",
  "doc_id": "cuda-c-programming-guide",
  "section_id": "thread-hierarchy",
  "source_url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy",
  "title": "CUDA C Programming Guide",
  "section_path": "Programming Model > Thread Hierarchy",
  "chunk_index": 42,
  "text": "<p>CUDA threads are organized into...</p>"
}
```

PDF chunks have `section_id: null` (no anchor navigation available).

## MCP Tools

### Existing (unchanged)

**`cuda_search_docs(query, mode?, top_k?)`** — hybrid BM25+dense search over all_chunks.jsonl. Returns chunks with `section_id` in metadata, enabling follow-up navigation.

### New

**`cuda_browse_toc(doc_id, section_id?, depth?)`**

Browse the document TOC tree.

- No `section_id` → return top-level chapters of that doc
- With `section_id` → return that node + its children
- `depth` (default 2) controls how many levels to expand
- Returns: list of TOC nodes with `{section_id, title, heading_level, children_count}`

```
cuda_browse_toc("cuda-c-programming-guide")
→ [{section_id: "introduction", title: "Introduction", children_count: 3},
   {section_id: "programming-model", title: "Programming Model", children_count: 4},
   ...]

cuda_browse_toc("cuda-c-programming-guide", "programming-model")
→ {section_id: "programming-model", title: "Programming Model",
   children: [{section_id: "kernels", ...}, {section_id: "thread-hierarchy", ...}, ...]}
```

**`cuda_read_section(doc_id, section_id)`**

Read full section content (lightweight HTML).

- Returns: section content + navigation context (parent, previous/next siblings)
- Enables Agent to read then navigate sideways or upward

```
cuda_read_section("cuda-c-programming-guide", "thread-hierarchy")
→ {content: "<p>CUDA threads are organized into...</p>...",
   title: "Thread Hierarchy",
   heading_path: ["Programming Model", "Thread Hierarchy"],
   token_count: 843,
   nav: {parent: "programming-model",
         prev_sibling: "kernels",
         next_sibling: "thread-block-clusters"},
   deep_link: "https://...#thread-hierarchy"}
```

## Agent Navigation Patterns

### Pattern 1: Search → Navigate

```
Agent: cuda_search_docs("shared memory bank conflicts")
  → chunk with section_id="shared-memory"

Agent: cuda_read_section("cuda-c-programming-guide", "shared-memory")
  → full section content + nav context

Agent: cuda_read_section("cuda-c-programming-guide", nav.next_sibling)
  → read next section for more context
```

### Pattern 2: Browse → Drill down

```
Agent: cuda_browse_toc("cuda-c-programming-guide")
  → top-level chapters

Agent: cuda_browse_toc("cuda-c-programming-guide", "performance-guidelines")
  → sub-sections of Performance Guidelines

Agent: cuda_read_section("cuda-c-programming-guide", "device-memory-accesses")
  → read specific section
```

## Files to Create/Modify

### New files
- `doc_retrieval/html_parser.py` — BeautifulSoup-based HTML parser producing toc.jsonl + sections.jsonl + chunks

### Modified files
- `doc_retrieval/parser.py` — call `html_parser.py` for HTML docs instead of Docling; Docling path unchanged for PDFs
- `doc_retrieval/searcher.py` — add `browse_toc()` and `read_section()` methods (lazy-load toc.jsonl and sections.jsonl)
- `doc_retrieval/cli.py` — add `browse` and `read` subcommands for CLI testing
- `cuda_agent/mcp_server.py` — register `cuda_browse_toc` and `cuda_read_section` MCP tools
- `conf/doc_retrieval/default.yaml` — add HTML parsing config (allowed tags, etc.)
- `doc_retrieval/CLAUDE.md` — update architecture docs

## Verification

1. `python -m doc_retrieval parse` produces toc.jsonl, sections.jsonl, and all_chunks.jsonl
2. Every HTML section has a corresponding TOC entry with correct parent/children
3. All chunk `section_id` values exist in sections.jsonl
4. `python -m doc_retrieval browse cuda-c-programming-guide` shows TOC tree
5. `python -m doc_retrieval read cuda-c-programming-guide thread-hierarchy` shows section content
6. `python -m doc_retrieval search "shared memory bank conflicts"` returns chunks with valid `section_id`
7. MCP tools work end-to-end from cuda_agent
