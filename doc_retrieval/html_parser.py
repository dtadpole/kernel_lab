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
                continue
            if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                continue
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

        heading = section_tag.find(re.compile(r"^h[1-6]$"), recursive=False)
        if heading is None:
            heading = section_tag.find(re.compile(r"^h[1-6]$"))
        if heading is None:
            title = section_id.replace("-", " ").title()
            level = 1
        else:
            title = heading.get_text(strip=True)
            title = re.sub(r"\s*[#¶]\s*$", "", title)
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title)
            level = int(heading.name[1])

        current_path = path + [title]

        child_sections = section_tag.find_all("section", id=True, recursive=False)
        child_ids = [cs.get("id") for cs in child_sections if cs.get("id")]

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

        for child_sec in child_sections:
            _walk(child_sec, section_id, current_path)

    for top_section in soup.find_all("section", id=True, recursive=False):
        _walk(top_section, None, [])

    if not results:
        body = soup.find("body")
        if body:
            for top_section in body.find_all("section", id=True, recursive=False):
                _walk(top_section, None, [])

    return results


_BLOCK_TAGS = frozenset({
    "p", "pre", "table", "ul", "ol", "dl", "blockquote", "div",
    "h1", "h2", "h3", "h4", "h5", "h6",
})


def _split_html_at_paragraphs(
    html_content: str,
    max_tokens: int,
    overlap_tokens: int,
    enc,
) -> list[str]:
    """Split HTML content at block element boundaries respecting token limits.

    Tries top-level children first; falls back to all block-level descendants
    when the top level has only one element (e.g. a single wrapping tag).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    top_elements = [str(el) for el in soup.children if isinstance(el, Tag) or str(el).strip()]
    top_elements = [e for e in top_elements if e.strip()]

    if not top_elements:
        return [html_content] if html_content.strip() else []

    # If the entire content is wrapped in a single top-level element, flatten
    # to block-level descendants so we can split at meaningful boundaries.
    if len(top_elements) == 1:
        block_tags = soup.find_all(_BLOCK_TAGS)
        if len(block_tags) > 1:
            elements = [str(t) for t in block_tags if str(t).strip()]
        else:
            elements = top_elements
    else:
        elements = top_elements

    if not elements:
        return [html_content] if html_content.strip() else []

    chunks = []
    current = []
    current_tokens = 0

    for elem in elements:
        elem_tokens = len(enc.encode(elem))
        if current_tokens + elem_tokens > max_tokens and current:
            chunks.append("".join(current))
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

    pending_merge: list[dict] = []
    pending_tokens = 0

    def _flush_pending():
        nonlocal chunk_index, pending_merge, pending_tokens
        if not pending_merge:
            return
        first = pending_merge[0]
        merged_content = " ".join(
            f"<h{s['heading_level']}>{s['title']}</h{s['heading_level']}> {s['content']}"
            if s["content"] else f"<h{s['heading_level']}>{s['title']}</h{s['heading_level']}>"
            for s in pending_merge
        )
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
            pending_merge.append(sec)
            pending_tokens += token_count
            if pending_tokens >= min_tokens:
                _flush_pending()
        else:
            _flush_pending()

            if token_count <= max_tokens:
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

    _flush_pending()

    return toc_entries, section_entries, chunks
