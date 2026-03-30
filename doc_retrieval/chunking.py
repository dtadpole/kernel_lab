"""Chunking strategies for NVIDIA CUDA documentation.

Two strategies based on document type:

- **Narrative** (Programming Guides, Tuning Guides, etc.):
  Split by heading, then enforce a minimum chunk size by merging
  short consecutive sections within the same parent heading.

- **API reference** (Runtime API, Driver API, Library docs, etc.):
  Group content into complete API entries (function/enum/typedef),
  merging the sub-headings (Parameters, Returns, See also) into
  the parent entry.
"""

from __future__ import annotations

import re

import tiktoken

# Documents where each top-level entry is a self-contained API reference.
# These get the "API entry grouping" strategy.
API_REFERENCE_DOCS = {
    "cuda-runtime-api",
    "cuda-driver-api",
    "cuda-math-api",
    "cuda-debugger-api",
    "libdevice-users-guide",
    "libnvvm-api",
    "cublas-library",
    "cusparse-library",
    "cusolver-library",
    "curand-library",
    "cufft-library",
    "npp-library",
    "nvblas-library",
    "nvjpeg",
    "nvjitlink",
    "nvfatbin",
    "cudla-api",
    "ptx-compiler-api",
}

# Sub-heading patterns in API docs that belong to the parent entry
_API_SUB_HEADINGS = re.compile(
    r"^(Parameters|Returns|Return values|Return Value|Description|"
    r"See also|See Also|Deprecated|Example|Note|Notes|"
    r"enumerator|Enumerations|Functions|Typedefs|"
    r"Target ISA Notes|Restrictions|Semantics|"
    r"Synopsis|Members|Values|Details|Overview)\b",
    re.IGNORECASE,
)


def _parse_heading(line: str) -> tuple[int, str] | None:
    """Parse a markdown heading line. Returns (level, title) or None."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    level = 0
    for ch in stripped:
        if ch == "#":
            level += 1
        else:
            break
    title = stripped[level:].strip().strip("[]()# ")
    if title:
        return level, title
    return None


def _split_by_headings(markdown: str) -> list[dict]:
    """Split markdown into raw sections by heading boundaries.

    Returns list of {level, title, text} where text includes the heading.
    """
    lines = markdown.split("\n")
    sections: list[dict] = []
    current_lines: list[str] = []
    current_level = 0
    current_title = "Introduction"

    def _flush():
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({
                "level": current_level,
                "title": current_title,
                "text": text,
            })

    for line in lines:
        parsed = _parse_heading(line)
        if parsed:
            _flush()
            current_lines = [line]
            current_level, current_title = parsed
        else:
            current_lines.append(line)

    _flush()
    return sections


def _is_api_sub_heading(title: str) -> bool:
    """Check if a heading title is a sub-part of an API entry."""
    return bool(_API_SUB_HEADINGS.match(title))


def chunk_narrative(
    markdown: str,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> list[dict]:
    """Chunk narrative-style documents.

    Strategy: split by heading, then merge consecutive short sections
    that share the same parent heading until min_tokens is reached.
    Sections exceeding max_tokens are split at paragraph boundaries.
    """
    sections = _split_by_headings(markdown)

    # Build section_path for each section
    path_stack: list[str] = []
    for sec in sections:
        level = sec["level"]
        while len(path_stack) >= level and path_stack:
            path_stack.pop()
        path_stack.append(sec["title"])
        sec["section_path"] = " > ".join(path_stack)

    # Merge short consecutive sections
    merged: list[dict] = []
    buffer_text = ""
    buffer_path = ""

    for sec in sections:
        sec_tokens = len(enc.encode(sec["text"]))

        if not buffer_text:
            buffer_text = sec["text"]
            buffer_path = sec["section_path"]
            continue

        buffer_tokens = len(enc.encode(buffer_text))

        if buffer_tokens < min_tokens:
            # Merge into buffer
            buffer_text = buffer_text + "\n\n" + sec["text"]
        else:
            # Flush buffer, start new
            merged.append({"section_path": buffer_path, "text": buffer_text})
            buffer_text = sec["text"]
            buffer_path = sec["section_path"]

    if buffer_text:
        merged.append({"section_path": buffer_path, "text": buffer_text})

    # Split oversized chunks at paragraph boundaries
    final: list[dict] = []
    for sec in merged:
        tokens = len(enc.encode(sec["text"]))
        if tokens <= max_tokens:
            final.append(sec)
        else:
            sub_chunks = _split_at_paragraphs(
                sec["text"], max_tokens, overlap_tokens, enc
            )
            for text in sub_chunks:
                final.append({"section_path": sec["section_path"], "text": text})

    return final


def chunk_api_reference(
    markdown: str,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> list[dict]:
    """Chunk API reference documents.

    Strategy: group sub-headings (Parameters, Returns, See also, enumerator,
    etc.) into the parent API entry. Then merge any remaining short entries
    with their neighbours.
    """
    sections = _split_by_headings(markdown)

    # Phase 1: Group sub-headings into parent entries
    groups: list[dict] = []
    for sec in sections:
        is_sub = _is_api_sub_heading(sec["title"])
        if is_sub and groups:
            # Append to previous group
            groups[-1]["text"] += "\n\n" + sec["text"]
        else:
            groups.append({
                "title": sec["title"],
                "level": sec["level"],
                "text": sec["text"],
            })

    # Build section paths
    path_stack: list[str] = []
    for g in groups:
        level = g["level"]
        while len(path_stack) >= level and path_stack:
            path_stack.pop()
        path_stack.append(g["title"])
        g["section_path"] = " > ".join(path_stack)

    # Phase 2: Merge remaining short entries
    merged: list[dict] = []
    buffer_text = ""
    buffer_path = ""

    for g in groups:
        if not buffer_text:
            buffer_text = g["text"]
            buffer_path = g["section_path"]
            continue

        buffer_tokens = len(enc.encode(buffer_text))

        if buffer_tokens < min_tokens:
            buffer_text = buffer_text + "\n\n" + g["text"]
        else:
            merged.append({"section_path": buffer_path, "text": buffer_text})
            buffer_text = g["text"]
            buffer_path = g["section_path"]

    if buffer_text:
        merged.append({"section_path": buffer_path, "text": buffer_text})

    # Phase 3: Split oversized chunks
    final: list[dict] = []
    for sec in merged:
        tokens = len(enc.encode(sec["text"]))
        if tokens <= max_tokens:
            final.append(sec)
        else:
            sub_chunks = _split_at_paragraphs(
                sec["text"], max_tokens, overlap_tokens, enc
            )
            for text in sub_chunks:
                final.append({"section_path": sec["section_path"], "text": text})

    return final


def _split_at_paragraphs(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    """Split text at paragraph boundaries respecting token limits."""
    tokens = len(enc.encode(text))
    if tokens <= max_tokens:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))
        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Overlap: keep trailing parts
            overlap_parts: list[str] = []
            overlap_count = 0
            for p in reversed(current_parts):
                p_tok = len(enc.encode(p))
                if overlap_count + p_tok > overlap_tokens:
                    break
                overlap_parts.insert(0, p)
                overlap_count += p_tok
            current_parts = overlap_parts
            current_tokens = overlap_count

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks
