"""FastMCP stdio server for CUDA documentation search and retrieval.

Exposes 4 MCP tools for searching and browsing indexed NVIDIA CUDA Toolkit
documentation:

    search_docs         — BM25/dense/hybrid search over all indexed docs
    lookup_doc_section  — Retrieve full text from a URL (from search results)
    browse_toc          — Browse document table of contents by doc_id
    read_section        — Read a specific section with navigation context

Configuration (environment variables):
    DOC_RETRIEVAL_INDEX_DIR    Path to the doc_retrieval index directory.
                               Forwarded to DocSearcher; see doc_retrieval
                               for defaults.
    DOC_RETRIEVAL_DOCS_DIR     Path to the parsed HTML docs directory.
                               Forwarded to DocSearcher; see doc_retrieval
                               for defaults.
"""

from __future__ import annotations

import json
import sys
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("kb")

# ---------------------------------------------------------------------------
# Document retrieval tools
# ---------------------------------------------------------------------------

_doc_searcher = None


def _get_doc_searcher():
    """Lazy-load the document searcher singleton."""
    global _doc_searcher
    if _doc_searcher is None:
        try:
            from doc_retrieval.searcher import DocSearcher

            _doc_searcher = DocSearcher()
        except Exception as exc:
            return None, str(exc)
    return _doc_searcher, None


@mcp.tool()
async def search_docs(
    query: Annotated[
        str,
        Field(description="Natural language search query about CUDA programming"),
    ],
    mode: Annotated[
        Literal["bm25", "dense", "hybrid"],
        Field(description="Search mode: bm25 (keyword), dense (semantic), hybrid (combined)"),
    ] = "hybrid",
    top_k: Annotated[
        int,
        Field(description="Number of results to return (1-20)", ge=1, le=20),
    ] = 5,
) -> str:
    """Search NVIDIA CUDA Toolkit documentation.

    Searches the indexed CUDA documentation corpus using the specified
    retrieval mode.  Returns matching chunks with section metadata.

    Example queries:
        - "shared memory bank conflicts"
        - "warp divergence impact on performance"
        - "atomicCAS signature and semantics"
        - "cudaMemcpyAsync stream synchronization"
        - "PTX ld.global instruction"

    Each result includes a ``url`` field with an anchor (e.g.
    ``...index.html#thread-hierarchy``).  Extract the doc slug and
    anchor to follow up:

        result = search_docs("shared memory bank conflicts")
        # Found section_path "Performance Guidelines > Device Memory Accesses"
        # url ends with .../cuda-c-programming-guide/index.html#device-memory-accesses

        # Read the full section for more context:
        read_section("cuda-c-programming-guide", "device-memory-accesses")

        # Or browse the parent chapter's TOC:
        browse_toc("cuda-c-programming-guide", "performance-guidelines")
    """
    searcher, err = _get_doc_searcher()
    if searcher is None:
        return json.dumps({"error": f"Doc searcher unavailable: {err}"})

    if mode == "bm25":
        results = searcher.search_bm25(query, top_k)
    elif mode == "dense":
        results = searcher.search_dense(query, top_k)
    else:
        results = searcher.search_hybrid(query, top_k)

    return json.dumps(
        [
            {
                "title": r.title,
                "section_path": r.section_path,
                "url": r.source_url,
                "text": r.text[:2000],  # truncate for context limits
                "score": round(r.score, 4),
            }
            for r in results
        ],
        indent=2,
    )


@mcp.tool()
async def lookup_doc_section(
    url: Annotated[
        str,
        Field(description="URL of the CUDA documentation page (from a prior search result)"),
    ],
    section: Annotated[
        str | None,
        Field(description="Section heading to filter to (optional)"),
    ] = None,
) -> str:
    """Retrieve full text from a specific CUDA documentation page or section.

    Given a URL from a prior search_docs result, retrieves all
    chunks from that page.  Optionally filter to a specific section.

    Prefer ``read_section`` for HTML docs — it returns structured
    content with navigation context (parent, siblings).  Use this tool
    when you have a URL but not a doc_id/section_id, or for PDF docs
    which don't have anchor-based navigation.
    """
    searcher, err = _get_doc_searcher()
    if searcher is None:
        return json.dumps({"error": f"Doc searcher unavailable: {err}"})

    chunks = searcher._load_chunks()
    matching = [c for c in chunks if c["source_url"] == url]

    if section:
        section_lower = section.lower()
        matching = [
            c for c in matching
            if section_lower in c["section_path"].lower()
        ]

    if not matching:
        return json.dumps({"error": f"No chunks found for url={url}, section={section}"})

    # Concatenate chunk texts in order
    matching.sort(key=lambda c: c["chunk_index"])
    text = "\n\n".join(c["text"] for c in matching)

    # Truncate to ~8000 chars to stay within context limits
    if len(text) > 8000:
        text = text[:8000] + "\n\n[... truncated ...]"

    return json.dumps(
        {
            "url": url,
            "section": section,
            "num_chunks": len(matching),
            "text": text,
        },
        indent=2,
    )


@mcp.tool()
async def browse_toc(
    doc_id: Annotated[str, Field(description=(
        "Document slug, e.g. 'cuda-c-programming-guide', 'parallel-thread-execution'"
    ))],
    section_id: Annotated[str | None, Field(description=(
        "Section anchor ID to expand. Omit for top-level chapters."
    ))] = None,
    depth: Annotated[int, Field(description="Expansion depth", ge=1, le=5)] = 2,
) -> str:
    """Browse the table of contents of a CUDA documentation page.

    Available doc_ids:
        cuda-c-programming-guide, parallel-thread-execution,
        cuda-c-best-practices-guide, inline-ptx-assembly,
        blackwell-tuning-guide, blackwell-compatibility-guide

    Usage patterns:

        # List top-level chapters:
        browse_toc("cuda-c-programming-guide")

        # Expand a chapter to see its sub-sections:
        browse_toc("cuda-c-programming-guide", "performance-guidelines")

        # Deep expand (3 levels):
        browse_toc("cuda-c-programming-guide", "programming-model", depth=3)

        # Then read a specific section:
        read_section("cuda-c-programming-guide", "device-memory-accesses")
    """
    searcher, err = _get_doc_searcher()
    if err:
        return err
    result = searcher.browse_toc(doc_id=doc_id, section_id=section_id, depth=depth)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def read_section(
    doc_id: Annotated[str, Field(description=(
        "Document slug, e.g. 'cuda-c-programming-guide', 'parallel-thread-execution'"
    ))],
    section_id: Annotated[str, Field(description=(
        "Section anchor ID from TOC or search result, e.g. 'thread-hierarchy'"
    ))],
) -> str:
    """Read the full content of a specific documentation section.

    Returns the section content as lightweight HTML with navigation
    context.  The response includes a ``nav`` object with
    ``parent``, ``prev_sibling``, and ``next_sibling`` section IDs
    for continued browsing.

    Usage patterns:

        # After searching — expand a result to full section:
        results = search_docs("shared memory bank conflicts")
        # url = ".../cuda-c-programming-guide/index.html#shared-memory"
        section = read_section("cuda-c-programming-guide", "shared-memory")

        # Read the next section using nav context:
        read_section("cuda-c-programming-guide", section.nav.next_sibling)

        # Go up to the parent chapter:
        browse_toc("cuda-c-programming-guide", section.nav.parent)

        # After browsing TOC — read a section you picked:
        toc = browse_toc("parallel-thread-execution", "instruction-set")
        read_section("parallel-thread-execution", "data-movement-and-conversion-instructions-ld")
    """
    searcher, err = _get_doc_searcher()
    if err:
        return err
    result = searcher.read_section(doc_id=doc_id, section_id=section_id)
    if result is None:
        return json.dumps({"error": f"Section '{section_id}' not found in '{doc_id}'"})
    if len(result.get("content", "")) > 8000:
        result["content"] = result["content"][:8000] + "\n... [truncated]"
    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
