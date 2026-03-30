# HTML Sharding + Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse Sphinx HTML docs with BeautifulSoup into a dual-layer index (sections for reading, chunks for search) with anchor-based navigation via TOC browsing and section reading MCP tools.

**Architecture:** BeautifulSoup extracts `<section id>` elements recursively from Sphinx HTML, producing three JSONL outputs: toc.jsonl (navigation tree), sections.jsonl (full section content as lightweight HTML), and chunks in all_chunks.jsonl (512-token pieces for BM25+FAISS search). Two new MCP tools (`cuda_browse_toc`, `cuda_read_section`) expose navigation to the Agent.

**Tech Stack:** BeautifulSoup4, tiktoken, existing Hydra config, existing MCP server (FastMCP)

**Spec:** `docs/superpowers/specs/2026-03-30-html-sharding-navigation-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `doc_retrieval/html_parser.py` | Create | BeautifulSoup-based HTML parser: section extraction, HTML cleaning, TOC building, chunking |
| `doc_retrieval/tests/test_html_parser.py` | Create | Unit tests for html_parser |
| `doc_retrieval/parser.py` | Modify | Route HTML docs through html_parser instead of Docling |
| `doc_retrieval/searcher.py` | Modify | Add `browse_toc()` and `read_section()` methods |
| `doc_retrieval/tests/test_navigation.py` | Create | Tests for browse_toc and read_section |
| `doc_retrieval/cli.py` | Modify | Add `browse` and `read` subcommands |
| `cuda_agent/mcp_server.py` | Modify | Register `cuda_browse_toc` and `cuda_read_section` tools |
| `conf/doc_retrieval/default.yaml` | Modify | Add `html_parsing` config section |
| `doc_retrieval/CLAUDE.md` | Modify | Update architecture docs |

---

### Task 1: HTML section extraction with tests

**Files:**
- Create: `doc_retrieval/tests/test_html_parser.py`
- Create: `doc_retrieval/html_parser.py`

- [ ] **Step 1: Write test fixture and section extraction tests**

```python
# doc_retrieval/tests/test_html_parser.py
"""Tests for BeautifulSoup-based HTML parser."""

import tiktoken
import pytest

FIXTURE_HTML = """\
<!DOCTYPE html>
<html>
<body>
<section id="overview">
  <h1>Overview</h1>
  <p>This is the overview.</p>
  <section id="introduction">
    <h2>Introduction</h2>
    <p>Intro content with <code>code_example()</code> and <strong>bold text</strong>.</p>
    <pre><code>int x = 42;</code></pre>
  </section>
  <section id="getting-started">
    <h2>Getting Started</h2>
    <p>Getting started content.</p>
    <section id="installation">
      <h3>Installation</h3>
      <p>Install instructions.</p>
    </section>
    <section id="first-steps">
      <h3>First Steps</h3>
      <p>First steps content.</p>
    </section>
  </section>
</section>
<section id="programming-model">
  <h1>Programming Model</h1>
  <p>Programming model content.</p>
  <table><tr><th>Feature</th><th>Support</th></tr><tr><td>CUDA</td><td>Yes</td></tr></table>
</section>
</body>
</html>
"""


@pytest.fixture
def enc():
    return tiktoken.get_encoding("cl100k_base")


class TestExtractSections:
    def test_finds_all_sections(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        ids = [s["section_id"] for s in sections]
        assert "overview" in ids
        assert "introduction" in ids
        assert "getting-started" in ids
        assert "installation" in ids
        assert "programming-model" in ids
        assert len(sections) == 6

    def test_heading_levels(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["overview"]["heading_level"] == 1
        assert by_id["introduction"]["heading_level"] == 2
        assert by_id["installation"]["heading_level"] == 3

    def test_parent_child_relationships(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["introduction"]["parent_section_id"] == "overview"
        assert by_id["installation"]["parent_section_id"] == "getting-started"
        assert by_id["overview"]["parent_section_id"] is None
        assert "introduction" in by_id["overview"]["children"]
        assert "getting-started" in by_id["overview"]["children"]

    def test_deep_link(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["introduction"]["deep_link"] == "https://example.com/test-doc/index.html#introduction"

    def test_heading_path(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["installation"]["heading_path"] == ["Overview", "Getting Started", "Installation"]

    def test_content_excludes_nested_sections(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        # overview's direct content should NOT contain "Intro content"
        assert "Intro content" not in by_id["overview"]["content"]
        assert "This is the overview" in by_id["overview"]["content"]

    def test_content_preserves_semantic_html(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        content = by_id["introduction"]["content"]
        assert "<code>" in content
        assert "<strong>" in content
        assert "<pre>" in content

    def test_content_preserves_tables(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        content = by_id["programming-model"]["content"]
        assert "<table>" in content
        assert "<th>" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doc_retrieval/.venv/bin/python -m pytest doc_retrieval/tests/test_html_parser.py -v 2>&1 | head -20`
Expected: `ModuleNotFoundError: No module named 'doc_retrieval.html_parser'`

- [ ] **Step 3: Implement `extract_sections` in html_parser.py**

```python
# doc_retrieval/html_parser.py
"""BeautifulSoup-based parser for Sphinx HTML documentation.

Extracts sections with anchor IDs, builds TOC trees, cleans HTML content,
and produces chunks for the dual-layer index (sections for reading, chunks
for search).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

# Tags to keep in cleaned HTML content.
_KEEP_TAGS = frozenset({
    "p", "code", "pre", "table", "tr", "td", "th", "thead", "tbody",
    "strong", "em", "a", "ul", "ol", "li", "br", "div", "span",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "dl", "dt", "dd",
})

# Tags to remove entirely (including their content).
_REMOVE_TAGS = frozenset({"nav", "script", "style", "header", "footer"})

# Attributes to keep (tag -> set of allowed attrs).
_KEEP_ATTRS = {"a": {"href"}}


def _clean_element(tag: Tag) -> str:
    """Clean an HTML element: keep semantic tags, strip attributes."""
    if isinstance(tag, NavigableString):
        return str(tag)

    if tag.name in _REMOVE_TAGS:
        return ""

    if tag.name == "section":
        # Should not happen (sections are excluded before calling), but safety
        return ""

    # Recurse into children
    inner = "".join(_clean_element(child) for child in tag.children)

    if tag.name in _KEEP_TAGS:
        # Build tag with allowed attributes only
        allowed = _KEEP_ATTRS.get(tag.name, set())
        attrs = " ".join(
            f'{k}="{v}"' for k, v in tag.attrs.items()
            if k in allowed and isinstance(v, str)
        )
        open_tag = f"<{tag.name} {attrs}>" if attrs else f"<{tag.name}>"
        return f"{open_tag}{inner}</{tag.name}>"

    # Unknown tag: pass through content only
    return inner


def _extract_direct_content(section_tag: Tag) -> str:
    """Extract cleaned HTML content from a section, excluding nested sections."""
    parts = []
    for child in section_tag.children:
        if isinstance(child, Tag):
            if child.name == "section":
                continue  # skip nested sections
            if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                continue  # skip the heading itself (stored as title)
            # Also skip anchor spans that are just navigation targets
            if child.name == "span" and child.get("id") and not child.get_text(strip=True):
                continue
            parts.append(_clean_element(child))
        elif isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
    return "".join(parts).strip()


def extract_sections(
    html: str,
    doc_id: str,
    base_url: str,
) -> list[dict]:
    """Extract all sections from Sphinx HTML with hierarchy and cleaned content.

    Args:
        html: Raw HTML string.
        doc_id: Document slug (e.g. "cuda-c-programming-guide").
        base_url: Base URL for deep links (e.g. "https://.../index.html").

    Returns:
        List of section dicts with: section_id, doc_id, title, heading_level,
        heading_path, parent_section_id, children, content, deep_link.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    def _walk(section_tag: Tag, parent_id: str | None, path: list[str]) -> None:
        section_id = section_tag.get("id")
        if not section_id:
            return

        # Find heading
        heading = section_tag.find(re.compile(r"^h[1-6]$"), recursive=False)
        if heading is None:
            # Try first child that contains a heading
            heading = section_tag.find(re.compile(r"^h[1-6]$"))
        if heading is None:
            title = section_id.replace("-", " ").title()
            level = 1
        else:
            title = heading.get_text(strip=True)
            # Remove trailing permalink characters
            title = re.sub(r"\s*[#¶]\s*$", "", title)
            # Remove leading section numbers like "5.3. "
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title)
            level = int(heading.name[1])

        current_path = path + [title]

        # Find direct child sections
        child_sections = section_tag.find_all("section", id=True, recursive=False)
        child_ids = [cs.get("id") for cs in child_sections if cs.get("id")]

        # Extract direct content (excluding nested sections and heading)
        content = _extract_direct_content(section_tag)

        results.append({
            "doc_id": doc_id,
            "section_id": section_id,
            "title": title,
            "heading_level": level,
            "heading_path": current_path,
            "parent_section_id": parent_id,
            "children": child_ids,
            "content": content,
            "deep_link": f"{base_url}#{section_id}",
        })

        # Recurse into child sections
        for child_sec in child_sections:
            _walk(child_sec, section_id, current_path)

    # Find top-level sections
    for top_section in soup.find_all("section", id=True, recursive=False):
        _walk(top_section, None, [])

    # If no top-level sections found, try inside body
    if not results:
        body = soup.find("body")
        if body:
            for top_section in body.find_all("section", id=True, recursive=False):
                _walk(top_section, None, [])

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/centos/kernel_lab/.claude/worktrees/doc-retrieval && doc_retrieval/.venv/bin/python -m pytest doc_retrieval/tests/test_html_parser.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add doc_retrieval/html_parser.py doc_retrieval/tests/test_html_parser.py
git commit -m "feat: add HTML section extraction with BeautifulSoup"
```

---

### Task 2: HTML section chunking + parse_html_doc entry point

**Files:**
- Modify: `doc_retrieval/html_parser.py`
- Modify: `doc_retrieval/tests/test_html_parser.py`

- [ ] **Step 1: Write chunking and entry point tests**

Add to `doc_retrieval/tests/test_html_parser.py`:

```python
LARGE_SECTION_HTML = """\
<!DOCTYPE html>
<html><body>
<section id="big-section">
  <h1>Big Section</h1>
  {paragraphs}
</section>
</body></html>
""".format(paragraphs="".join(f"<p>Paragraph {i} with enough words to take up tokens. " * 8 + "</p>" for i in range(30)))

TINY_SECTIONS_HTML = """\
<!DOCTYPE html>
<html><body>
<section id="parent">
  <h1>Parent</h1>
  <p>Parent content.</p>
  <section id="tiny-a">
    <h2>Tiny A</h2>
    <p>Short.</p>
  </section>
  <section id="tiny-b">
    <h2>Tiny B</h2>
    <p>Also short.</p>
  </section>
  <section id="normal">
    <h2>Normal Section</h2>
    <p>This section has enough content to stand alone with many words repeated several times to ensure it exceeds the minimum token threshold for chunking purposes in our system.</p>
  </section>
</section>
</body></html>
"""


class TestChunkSections:
    def test_large_section_is_split(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            LARGE_SECTION_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
            max_tokens=512, min_tokens=128, overlap_tokens=64,
        )
        # Should produce multiple chunks from the one large section
        big_chunks = [c for c in chunks if c["section_id"] == "big-section"]
        assert len(big_chunks) > 1
        # All chunks carry the section_id
        assert all(c["section_id"] == "big-section" for c in big_chunks)

    def test_tiny_sections_merged(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            TINY_SECTIONS_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
            max_tokens=512, min_tokens=128, overlap_tokens=64,
        )
        # tiny-a and tiny-b should be merged into one chunk
        chunk_texts = " ".join(c["text"] for c in chunks)
        assert "Tiny A" in chunk_texts
        assert "Tiny B" in chunk_texts

    def test_toc_output(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            FIXTURE_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
        )
        assert len(toc) == 6
        by_id = {t["section_id"]: t for t in toc}
        assert "children" in by_id["overview"]
        assert by_id["overview"]["parent_section_id"] is None

    def test_sections_output(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            FIXTURE_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
        )
        assert len(sections) > 0
        sec = next(s for s in sections if s["section_id"] == "introduction")
        assert "content" in sec
        assert "token_count" in sec
        assert sec["token_count"] > 0

    def test_chunks_have_section_id(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            FIXTURE_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
        )
        for chunk in chunks:
            assert "section_id" in chunk
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert chunk["doc_id"] == "test-doc"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doc_retrieval/.venv/bin/python -m pytest doc_retrieval/tests/test_html_parser.py::TestChunkSections -v 2>&1 | head -10`
Expected: FAIL — `parse_html_doc` not defined

- [ ] **Step 3: Implement `parse_html_doc` and chunking in html_parser.py**

Add to the end of `doc_retrieval/html_parser.py`:

```python
def _split_html_at_paragraphs(
    html_content: str,
    max_tokens: int,
    overlap_tokens: int,
    enc,
) -> list[str]:
    """Split HTML content at <p> boundaries respecting token limits."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Split at top-level elements (paragraphs, pre, table, ul, ol, etc.)
    elements = [str(el) for el in soup.children if isinstance(el, Tag) or str(el).strip()]
    elements = [e for e in elements if e.strip()]

    if not elements:
        return [html_content] if html_content.strip() else []

    chunks = []
    current = []
    current_tokens = 0

    for elem in elements:
        elem_tokens = len(enc.encode(elem))
        if current_tokens + elem_tokens > max_tokens and current:
            chunks.append("".join(current))
            # Overlap: keep last element(s) up to overlap_tokens
            overlap = []
            overlap_tok = 0
            for prev in reversed(current):
                pt = len(enc.encode(prev))
                if overlap_tok + pt > overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_tok += pt
            current = overlap
            current_tokens = overlap_tok
        current.append(elem)
        current_tokens += elem_tokens

    if current:
        chunks.append("".join(current))

    return chunks


def parse_html_doc(
    html: str,
    doc_id: str,
    base_url: str,
    enc,
    max_tokens: int = 512,
    min_tokens: int = 128,
    overlap_tokens: int = 64,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse Sphinx HTML into TOC entries, sections, and chunks.

    Args:
        html: Raw HTML string (or Path will be read).
        doc_id: Document slug.
        base_url: Base URL for deep links.
        enc: tiktoken encoding instance.
        max_tokens: Maximum tokens per chunk.
        min_tokens: Minimum tokens per chunk (smaller sections get merged).
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        Tuple of (toc_entries, sections, chunks).
    """
    all_sections = extract_sections(html, doc_id, base_url)

    # --- TOC entries ---
    toc_entries = []
    for sec in all_sections:
        toc_entries.append({
            "doc_id": sec["doc_id"],
            "section_id": sec["section_id"],
            "title": sec["title"],
            "heading_level": sec["heading_level"],
            "parent_section_id": sec["parent_section_id"],
            "children": sec["children"],
            "deep_link": sec["deep_link"],
        })

    # --- Sections (full content for reading layer) ---
    section_entries = []
    for sec in all_sections:
        token_count = len(enc.encode(sec["content"])) if sec["content"] else 0
        section_entries.append({
            "doc_id": sec["doc_id"],
            "section_id": sec["section_id"],
            "title": sec["title"],
            "heading_level": sec["heading_level"],
            "heading_path": sec["heading_path"],
            "content": sec["content"],
            "token_count": token_count,
            "deep_link": sec["deep_link"],
        })

    # --- Chunks (search layer) ---
    chunks: list[dict] = []
    chunk_index = 0

    # Group sections for potential merging of tiny ones
    # Process in document order (as returned by extract_sections)
    pending_merge: list[dict] = []
    pending_tokens = 0

    def _flush_pending():
        nonlocal chunk_index, pending_merge, pending_tokens
        if not pending_merge:
            return
        # Use the first section's metadata for the merged chunk
        first = pending_merge[0]
        merged_content = " ".join(s["content"] for s in pending_merge if s["content"])
        section_path = " > ".join(first["heading_path"])

        if pending_tokens <= max_tokens:
            chunk_id = hashlib.sha256(
                f"{doc_id}:{chunk_index}:{merged_content[:100]}".encode()
            ).hexdigest()[:16]
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "section_id": first["section_id"],
                "source_url": first["deep_link"],
                "title": first["title"],
                "section_path": section_path,
                "chunk_index": chunk_index,
                "text": merged_content,
            })
            chunk_index += 1
        else:
            # Split oversized merged content
            parts = _split_html_at_paragraphs(merged_content, max_tokens, overlap_tokens, enc)
            for part in parts:
                chunk_id = hashlib.sha256(
                    f"{doc_id}:{chunk_index}:{part[:100]}".encode()
                ).hexdigest()[:16]
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "section_id": first["section_id"],
                    "source_url": first["deep_link"],
                    "title": first["title"],
                    "section_path": section_path,
                    "chunk_index": chunk_index,
                    "text": part,
                })
                chunk_index += 1

        pending_merge = []
        pending_tokens = 0

    for sec in all_sections:
        content = sec["content"]
        if not content.strip():
            continue
        token_count = len(enc.encode(content))

        if token_count < min_tokens:
            # Accumulate for merging
            pending_merge.append(sec)
            pending_tokens += token_count
            if pending_tokens >= min_tokens:
                _flush_pending()
        else:
            # Flush any pending tiny sections first
            _flush_pending()

            if token_count <= max_tokens:
                # Section fits in one chunk
                section_path = " > ".join(sec["heading_path"])
                chunk_id = hashlib.sha256(
                    f"{doc_id}:{chunk_index}:{content[:100]}".encode()
                ).hexdigest()[:16]
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "section_id": sec["section_id"],
                    "source_url": sec["deep_link"],
                    "title": sec["title"],
                    "section_path": " > ".join(sec["heading_path"]),
                    "chunk_index": chunk_index,
                    "text": content,
                })
                chunk_index += 1
            else:
                # Split large section
                section_path = " > ".join(sec["heading_path"])
                parts = _split_html_at_paragraphs(content, max_tokens, overlap_tokens, enc)
                for part in parts:
                    chunk_id = hashlib.sha256(
                        f"{doc_id}:{chunk_index}:{part[:100]}".encode()
                    ).hexdigest()[:16]
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "section_id": sec["section_id"],
                        "source_url": sec["deep_link"],
                        "title": sec["title"],
                        "section_path": section_path,
                        "chunk_index": chunk_index,
                        "text": part,
                    })
                    chunk_index += 1

    # Flush remaining pending
    _flush_pending()

    return toc_entries, section_entries, chunks
```

- [ ] **Step 4: Run all tests**

Run: `doc_retrieval/.venv/bin/python -m pytest doc_retrieval/tests/test_html_parser.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add doc_retrieval/html_parser.py doc_retrieval/tests/test_html_parser.py
git commit -m "feat: add HTML chunking and parse_html_doc entry point"
```

---

### Task 3: Integrate html_parser into parser.py

**Files:**
- Modify: `doc_retrieval/parser.py:117-198`

- [ ] **Step 1: Replace `_parse_html` and update `parse_docs`**

In `doc_retrieval/parser.py`, replace the `_parse_html` function (lines 117-140) and update `parse_docs` (lines 143-198):

Replace lines 117-140 (`_parse_html`) with:

```python
def _parse_html_doc(html_dir_path: Path, enc) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse a single HTML doc folder using BeautifulSoup.

    Returns (toc_entries, section_entries, chunks).
    """
    from doc_retrieval.html_parser import parse_html_doc

    slug = html_dir_path.name
    index_path = html_dir_path / "index.html"
    base_url = f"https://docs.nvidia.com/cuda/{slug}/index.html"

    cfg = load_config()
    cc = cfg.doc_retrieval.chunking

    logger.info("Parsing HTML: %s", slug)

    html = index_path.read_text(encoding="utf-8")
    toc, sections, chunks = parse_html_doc(
        html, slug, base_url, enc,
        max_tokens=cc.max_chunk_tokens,
        min_tokens=cc.min_chunk_tokens,
        overlap_tokens=cc.overlap_tokens,
    )

    logger.info("  -> %d sections, %d chunks from %s", len(sections), len(chunks), slug)
    return toc, sections, chunks
```

Replace lines 143-198 (`parse_docs`) with:

```python
def parse_docs(
    with_images: bool = False,
    vlm_captions: bool = False,
) -> None:
    """Parse all downloaded docs into chunks, sections, and TOC.

    PDFs are parsed with Docling. HTML docs are parsed with BeautifulSoup
    to preserve anchor IDs for navigation.

    Args:
        with_images: Extract images (reserved for future).
        vlm_captions: Generate VLM captions for images (reserved for future).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    enc = tiktoken.get_encoding(cfg.doc_retrieval.chunking.tokenizer)

    raw = _raw_root()
    pdf_dir = raw / "pdfs"
    html_dir = raw / "html"
    runtime = _runtime_root()
    chunks_dir = runtime / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    all_toc: list[dict] = []
    all_sections: list[dict] = []

    # --- PDFs via Docling ---
    if pdf_dir.exists():
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()

        pdfs = sorted(pdf_dir.glob("*.pdf"))
        logger.info("Found %d PDFs to parse", len(pdfs))
        for pdf in pdfs:
            all_chunks.extend(_parse_pdf(pdf, converter, enc))

    # --- HTML via BeautifulSoup ---
    if html_dir.exists():
        html_folders = sorted(
            d for d in html_dir.iterdir()
            if d.is_dir() and (d / "index.html").exists()
        )
        logger.info("Found %d HTML docs to parse", len(html_folders))
        for html_folder in html_folders:
            toc, sections, chunks = _parse_html_doc(html_folder, enc)
            all_toc.extend(toc)
            all_sections.extend(sections)
            all_chunks.extend(chunks)

    # --- Write outputs ---
    out_chunks = chunks_dir / "all_chunks.jsonl"
    with open(out_chunks, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    out_toc = chunks_dir / "toc.jsonl"
    with open(out_toc, "w", encoding="utf-8") as f:
        for entry in all_toc:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    out_sections = chunks_dir / "sections.jsonl"
    with open(out_sections, "w", encoding="utf-8") as f:
        for sec in all_sections:
            f.write(json.dumps(sec, ensure_ascii=False) + "\n")

    logger.info(
        "Total: %d chunks from %d documents -> %s",
        len(all_chunks),
        len(set(c["doc_id"] for c in all_chunks)),
        out_chunks,
    )
    logger.info("TOC: %d entries -> %s", len(all_toc), out_toc)
    logger.info("Sections: %d entries -> %s", len(all_sections), out_sections)
```

Also remove the `import tempfile` from the imports at the top (line 14) since it's no longer used.

- [ ] **Step 2: Run a quick smoke test with one HTML doc**

Run: `doc_retrieval/.venv/bin/python -c "
from pathlib import Path
import tiktoken
from doc_retrieval.html_parser import parse_html_doc
enc = tiktoken.get_encoding('cl100k_base')
html = Path('doc_retrieval/data/raw/html/blackwell-tuning-guide/index.html').read_text()
toc, secs, chunks = parse_html_doc(html, 'blackwell-tuning-guide', 'https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html', enc)
print(f'TOC: {len(toc)} entries')
print(f'Sections: {len(secs)} entries')
print(f'Chunks: {len(chunks)} chunks')
for t in toc[:5]:
    print(f'  {t[\"section_id\"]}: {t[\"title\"]} (level {t[\"heading_level\"]})')
"`

Expected: TOC/sections/chunks with correct structure.

- [ ] **Step 3: Commit**

```bash
git add doc_retrieval/parser.py
git commit -m "feat: integrate BeautifulSoup HTML parser into parse pipeline"
```

---

### Task 4: Add browse_toc and read_section to searcher.py

**Files:**
- Create: `doc_retrieval/tests/test_navigation.py`
- Modify: `doc_retrieval/searcher.py`

- [ ] **Step 1: Write navigation tests**

```python
# doc_retrieval/tests/test_navigation.py
"""Tests for TOC browsing and section reading."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def nav_dir(tmp_path):
    """Create test TOC and sections JSONL files."""
    toc = [
        {"doc_id": "test-doc", "section_id": "intro", "title": "Introduction",
         "heading_level": 1, "parent_section_id": None,
         "children": ["basics", "advanced"],
         "deep_link": "https://example.com/test-doc/index.html#intro"},
        {"doc_id": "test-doc", "section_id": "basics", "title": "Basics",
         "heading_level": 2, "parent_section_id": "intro",
         "children": [],
         "deep_link": "https://example.com/test-doc/index.html#basics"},
        {"doc_id": "test-doc", "section_id": "advanced", "title": "Advanced",
         "heading_level": 2, "parent_section_id": "intro",
         "children": [],
         "deep_link": "https://example.com/test-doc/index.html#advanced"},
        {"doc_id": "other-doc", "section_id": "overview", "title": "Overview",
         "heading_level": 1, "parent_section_id": None,
         "children": [],
         "deep_link": "https://example.com/other-doc/index.html#overview"},
    ]
    sections = [
        {"doc_id": "test-doc", "section_id": "intro", "title": "Introduction",
         "heading_level": 1, "heading_path": ["Introduction"],
         "content": "<p>Welcome to the intro.</p>", "token_count": 10,
         "deep_link": "https://example.com/test-doc/index.html#intro"},
        {"doc_id": "test-doc", "section_id": "basics", "title": "Basics",
         "heading_level": 2, "heading_path": ["Introduction", "Basics"],
         "content": "<p>Basic content here.</p>", "token_count": 8,
         "deep_link": "https://example.com/test-doc/index.html#basics"},
        {"doc_id": "test-doc", "section_id": "advanced", "title": "Advanced",
         "heading_level": 2, "heading_path": ["Introduction", "Advanced"],
         "content": "<p>Advanced content here.</p>", "token_count": 8,
         "deep_link": "https://example.com/test-doc/index.html#advanced"},
        {"doc_id": "other-doc", "section_id": "overview", "title": "Overview",
         "heading_level": 1, "heading_path": ["Overview"],
         "content": "<p>Other doc overview.</p>", "token_count": 8,
         "deep_link": "https://example.com/other-doc/index.html#overview"},
    ]

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    with open(chunks_dir / "toc.jsonl", "w") as f:
        for entry in toc:
            f.write(json.dumps(entry) + "\n")
    with open(chunks_dir / "sections.jsonl", "w") as f:
        for sec in sections:
            f.write(json.dumps(sec) + "\n")
    # Empty chunks file (not needed for navigation tests)
    (chunks_dir / "all_chunks.jsonl").touch()
    # Empty index dir
    (tmp_path / "index").mkdir()

    return tmp_path


class TestBrowseToc:
    def test_top_level(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir  # override for test
        result = s.browse_toc("test-doc")
        assert len(result) == 1
        assert result[0]["section_id"] == "intro"
        assert result[0]["children_count"] == 2

    def test_expand_section(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.browse_toc("test-doc", section_id="intro")
        assert result["section_id"] == "intro"
        assert len(result["children"]) == 2
        assert result["children"][0]["section_id"] == "basics"

    def test_unknown_doc(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.browse_toc("nonexistent")
        assert result == []


class TestReadSection:
    def test_read_existing(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.read_section("test-doc", "basics")
        assert result["content"] == "<p>Basic content here.</p>"
        assert result["title"] == "Basics"
        assert result["nav"]["parent"] == "intro"

    def test_sibling_navigation(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.read_section("test-doc", "basics")
        assert result["nav"]["prev_sibling"] is None
        assert result["nav"]["next_sibling"] == "advanced"

    def test_not_found(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.read_section("test-doc", "nonexistent")
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doc_retrieval/.venv/bin/python -m pytest doc_retrieval/tests/test_navigation.py -v 2>&1 | head -15`
Expected: FAIL — `browse_toc` not defined

- [ ] **Step 3: Implement browse_toc and read_section in searcher.py**

Add these methods to the `DocSearcher` class in `doc_retrieval/searcher.py`, after the existing `search_hybrid` method. Also add a `_runtime_root` attribute and lazy-loading for toc/sections:

First, add to `__init__` (after the existing lazy-loaded state):
```python
        # Navigation data (lazy-loaded)
        self._toc: list[dict] | None = None
        self._sections_data: list[dict] | None = None
        self._runtime_root: Path | None = None
```

Then add these methods:

```python
    def _get_runtime_root(self) -> Path:
        if self._runtime_root is not None:
            return self._runtime_root
        cfg = load_config()
        return Path(cfg.doc_retrieval.storage.runtime_root).expanduser()

    def _load_toc(self) -> list[dict]:
        if self._toc is not None:
            return self._toc
        toc_path = self._get_runtime_root() / "chunks" / "toc.jsonl"
        if not toc_path.exists():
            self._toc = []
            return self._toc
        self._toc = []
        with open(toc_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._toc.append(json.loads(line))
        return self._toc

    def _load_sections_data(self) -> list[dict]:
        if self._sections_data is not None:
            return self._sections_data
        sec_path = self._get_runtime_root() / "chunks" / "sections.jsonl"
        if not sec_path.exists():
            self._sections_data = []
            return self._sections_data
        self._sections_data = []
        with open(sec_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._sections_data.append(json.loads(line))
        return self._sections_data

    def browse_toc(
        self,
        doc_id: str,
        section_id: str | None = None,
        depth: int = 2,
    ) -> list[dict] | dict:
        """Browse the document TOC tree.

        Args:
            doc_id: Document slug.
            section_id: If None, return top-level sections. If set, return
                that node with its children expanded.
            depth: How many levels to expand (default 2).

        Returns:
            List of TOC nodes (top-level) or single node dict (with children).
        """
        toc = self._load_toc()
        doc_toc = [t for t in toc if t["doc_id"] == doc_id]

        if not doc_toc:
            return []

        by_id = {t["section_id"]: t for t in doc_toc}

        def _node_summary(entry: dict, remaining_depth: int) -> dict:
            node = {
                "section_id": entry["section_id"],
                "title": entry["title"],
                "heading_level": entry["heading_level"],
                "children_count": len(entry.get("children", [])),
            }
            if remaining_depth > 1 and entry.get("children"):
                node["children"] = [
                    _node_summary(by_id[cid], remaining_depth - 1)
                    for cid in entry["children"]
                    if cid in by_id
                ]
            return node

        if section_id is None:
            # Return top-level sections
            top_level = [t for t in doc_toc if t["parent_section_id"] is None]
            return [_node_summary(t, depth) for t in top_level]

        if section_id not in by_id:
            return []

        return _node_summary(by_id[section_id], depth)

    def read_section(
        self,
        doc_id: str,
        section_id: str,
    ) -> dict | None:
        """Read full section content with navigation context.

        Returns:
            Dict with content, title, heading_path, token_count, nav, deep_link.
            None if section not found.
        """
        sections = self._load_sections_data()
        toc = self._load_toc()

        sec = next(
            (s for s in sections if s["doc_id"] == doc_id and s["section_id"] == section_id),
            None,
        )
        if sec is None:
            return None

        # Find navigation context from TOC
        toc_entry = next(
            (t for t in toc if t["doc_id"] == doc_id and t["section_id"] == section_id),
            None,
        )

        parent_id = toc_entry["parent_section_id"] if toc_entry else None

        # Find siblings
        prev_sibling = None
        next_sibling = None
        if toc_entry and parent_id:
            parent = next(
                (t for t in toc if t["doc_id"] == doc_id and t["section_id"] == parent_id),
                None,
            )
            if parent and parent.get("children"):
                siblings = parent["children"]
                try:
                    idx = siblings.index(section_id)
                    if idx > 0:
                        prev_sibling = siblings[idx - 1]
                    if idx < len(siblings) - 1:
                        next_sibling = siblings[idx + 1]
                except ValueError:
                    pass

        return {
            "content": sec["content"],
            "title": sec["title"],
            "heading_path": sec["heading_path"],
            "token_count": sec["token_count"],
            "nav": {
                "parent": parent_id,
                "prev_sibling": prev_sibling,
                "next_sibling": next_sibling,
            },
            "deep_link": sec["deep_link"],
        }
```

Also add `import json` to the imports if not already present, and add `from pathlib import Path` if needed.

- [ ] **Step 4: Run navigation tests**

Run: `doc_retrieval/.venv/bin/python -m pytest doc_retrieval/tests/test_navigation.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add doc_retrieval/searcher.py doc_retrieval/tests/test_navigation.py
git commit -m "feat: add browse_toc and read_section navigation methods"
```

---

### Task 5: Add CLI subcommands

**Files:**
- Modify: `doc_retrieval/cli.py`

- [ ] **Step 1: Add browse and read commands**

Add to `doc_retrieval/cli.py`:

After `cmd_search` function, add:

```python
def cmd_browse(args: argparse.Namespace) -> None:
    from doc_retrieval.searcher import DocSearcher
    import json

    searcher = DocSearcher()
    result = searcher.browse_toc(
        doc_id=args.doc_id,
        section_id=args.section_id,
        depth=args.depth,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_read(args: argparse.Namespace) -> None:
    from doc_retrieval.searcher import DocSearcher
    import json

    searcher = DocSearcher()
    result = searcher.read_section(
        doc_id=args.doc_id,
        section_id=args.section_id,
    )
    if result is None:
        print(f"Section '{args.section_id}' not found in '{args.doc_id}'")
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

In `main()`, after the search subparser registration, add:

```python
    # --- browse ---
    br = sub.add_parser("browse", help="Browse document table of contents")
    br.add_argument("doc_id", help="Document ID (slug)")
    br.add_argument("--section-id", default=None, help="Section to expand")
    br.add_argument("--depth", type=int, default=2, help="Expansion depth (default: 2)")
    br.set_defaults(func=cmd_browse)

    # --- read ---
    rd = sub.add_parser("read", help="Read a document section")
    rd.add_argument("doc_id", help="Document ID (slug)")
    rd.add_argument("section_id", help="Section ID (anchor)")
    rd.set_defaults(func=cmd_read)
```

- [ ] **Step 2: Verify CLI help**

Run: `doc_retrieval/.venv/bin/python -m doc_retrieval --help`
Expected: Shows `browse` and `read` subcommands

- [ ] **Step 3: Commit**

```bash
git add doc_retrieval/cli.py
git commit -m "feat: add browse and read CLI subcommands"
```

---

### Task 6: Register MCP tools

**Files:**
- Modify: `cuda_agent/mcp_server.py`

- [ ] **Step 1: Add MCP tool registrations**

In `cuda_agent/mcp_server.py`, after the `cuda_lookup_doc_section` tool (around line 857), add:

```python
@mcp.tool()
async def cuda_browse_toc(
    doc_id: Annotated[str, Field(description=(
        "Document slug, e.g. 'cuda-c-programming-guide', 'parallel-thread-execution'"
    ))],
    section_id: Annotated[str | None, Field(description=(
        "Section anchor ID to expand. Omit for top-level chapters."
    ))] = None,
    depth: Annotated[int, Field(description="Expansion depth", ge=1, le=5)] = 2,
) -> str:
    """Browse the table of contents of a CUDA documentation page.

    Use without section_id to see top-level chapters.
    Use with section_id to expand a specific section and see its children.
    """
    searcher, err = _get_doc_searcher()
    if err:
        return err
    result = searcher.browse_toc(doc_id=doc_id, section_id=section_id, depth=depth)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def cuda_read_section(
    doc_id: Annotated[str, Field(description=(
        "Document slug, e.g. 'cuda-c-programming-guide', 'parallel-thread-execution'"
    ))],
    section_id: Annotated[str, Field(description=(
        "Section anchor ID from TOC or search result, e.g. 'thread-hierarchy'"
    ))],
) -> str:
    """Read the full content of a specific documentation section.

    Returns the section content as lightweight HTML with navigation context
    (parent section, previous/next siblings) for further browsing.

    Use after cuda_search_docs to read the full context of a search result,
    or after cuda_browse_toc to read a section you found in the TOC.
    """
    searcher, err = _get_doc_searcher()
    if err:
        return err
    result = searcher.read_section(doc_id=doc_id, section_id=section_id)
    if result is None:
        return json.dumps({"error": f"Section '{section_id}' not found in '{doc_id}'"})
    # Truncate very large sections to ~8000 chars
    if len(result.get("content", "")) > 8000:
        result["content"] = result["content"][:8000] + "\n... [truncated]"
    return json.dumps(result, indent=2, ensure_ascii=False)
```

- [ ] **Step 2: Commit**

```bash
git add cuda_agent/mcp_server.py
git commit -m "feat: register cuda_browse_toc and cuda_read_section MCP tools"
```

---

### Task 7: Update config and docs, end-to-end verification

**Files:**
- Modify: `conf/doc_retrieval/default.yaml`
- Modify: `doc_retrieval/CLAUDE.md`

- [ ] **Step 1: Add html_parsing config section**

In `conf/doc_retrieval/default.yaml`, after the chunking section (line 88), add:

```yaml
# --- HTML parsing settings ---
html_parsing:
  keep_tags:
    - "p"
    - "code"
    - "pre"
    - "table"
    - "tr"
    - "td"
    - "th"
    - "thead"
    - "tbody"
    - "strong"
    - "em"
    - "a"
    - "ul"
    - "ol"
    - "li"
    - "blockquote"
    - "dl"
    - "dt"
    - "dd"
  remove_tags:
    - "nav"
    - "script"
    - "style"
    - "header"
    - "footer"
```

- [ ] **Step 2: Update CLAUDE.md module responsibilities and architecture**

In `doc_retrieval/CLAUDE.md`, add `html_parser.py` to the module responsibilities section:

```
- **`html_parser.py`** -- BeautifulSoup-based parser for Sphinx HTML docs. Extracts sections with anchor IDs, builds TOC tree, produces lightweight HTML chunks for dual-layer index (sections for reading, chunks for search).
```

Add to the architecture section or key design decisions:

```
- **Dual-layer HTML index** -- Sections (full content, lightweight HTML) for reading + 512-token chunks for BM25/FAISS search. HTML parsed with BeautifulSoup to preserve anchor IDs for navigation. PDFs continue using Docling.
```

Update storage layout to include toc.jsonl and sections.jsonl:

```
~/.doc_retrieval/                    # RUNTIME — derived artifacts only
  chunks/
    all_chunks.jsonl   # Search-layer chunks (PDF + HTML)
    toc.jsonl          # HTML document TOC trees
    sections.jsonl     # HTML full section content (reading layer)
  index/             # bm25.pkl, faiss.index, chunk_ids.json, embedding_cache.npz
```

- [ ] **Step 3: Run end-to-end parse on one HTML doc**

Run: `doc_retrieval/.venv/bin/python -c "
from pathlib import Path
import tiktoken
from doc_retrieval.html_parser import parse_html_doc
enc = tiktoken.get_encoding('cl100k_base')
html = Path('doc_retrieval/data/raw/html/cuda-c-programming-guide/index.html').read_text()
toc, secs, chunks = parse_html_doc(html, 'cuda-c-programming-guide', 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html', enc)
print(f'TOC: {len(toc)} entries')
print(f'Sections: {len(secs)} entries')
print(f'Chunks: {len(chunks)} chunks')
print()
print('Sample TOC (first 5):')
for t in toc[:5]:
    indent = '  ' * (t['heading_level'] - 1)
    print(f'  {indent}{t[\"section_id\"]}: {t[\"title\"]} (h{t[\"heading_level\"]}, {len(t[\"children\"])} children)')
print()
print('Sample chunk:')
c = chunks[10]
print(f'  section_id: {c[\"section_id\"]}')
print(f'  section_path: {c[\"section_path\"]}')
print(f'  text[:200]: {c[\"text\"][:200]}')
"`

Expected: ~881 TOC entries, ~881 sections, several hundred chunks with correct section_ids.

- [ ] **Step 4: Run all tests**

Run: `doc_retrieval/.venv/bin/python -m pytest doc_retrieval/tests/test_html_parser.py doc_retrieval/tests/test_navigation.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add conf/doc_retrieval/default.yaml doc_retrieval/CLAUDE.md
git commit -m "docs: update config and architecture for HTML sharding + navigation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | HTML section extraction with tests | html_parser.py, test_html_parser.py |
| 2 | HTML chunking + parse_html_doc entry point | html_parser.py, test_html_parser.py |
| 3 | Integrate into parser.py | parser.py |
| 4 | browse_toc + read_section in searcher.py | searcher.py, test_navigation.py |
| 5 | CLI subcommands | cli.py |
| 6 | MCP tool registration | mcp_server.py |
| 7 | Config, docs, end-to-end verification | default.yaml, CLAUDE.md |
