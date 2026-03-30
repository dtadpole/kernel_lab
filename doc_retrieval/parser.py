"""Parse downloaded NVIDIA CUDA docs into structured text chunks using Docling.

Uses type-aware chunking from ``chunking.py``: narrative strategy for
programming guides and tutorials, API-reference strategy for API docs
and library references.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path

import tiktoken

from doc_retrieval.chunking import (
    API_REFERENCE_DOCS,
    chunk_api_reference,
    chunk_narrative,
)
from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)


def _storage_root() -> Path:
    cfg = load_config()
    return Path(cfg.doc_retrieval.storage.root).expanduser()


def _make_doc_id(filename: str) -> str:
    """Derive a stable doc_id from filename (stem, lowered, hyphened)."""
    return Path(filename).stem.lower().replace("_", "-")


def _chunk_markdown(doc_id: str, markdown: str, enc: tiktoken.Encoding) -> list[dict]:
    """Chunk markdown using the appropriate strategy for the document type."""
    cfg = load_config()
    cc = cfg.doc_retrieval.chunking
    max_tokens = cc.max_chunk_tokens
    min_tokens = cc.min_chunk_tokens
    overlap = cc.overlap_tokens

    if doc_id in API_REFERENCE_DOCS:
        return chunk_api_reference(markdown, max_tokens, min_tokens, overlap, enc)
    else:
        return chunk_narrative(markdown, max_tokens, min_tokens, overlap, enc)


def _extract_title(markdown: str, fallback: str) -> str:
    """Extract title from first heading or use fallback."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()
    return fallback


def _sections_to_chunks(
    doc_id: str,
    source_url: str,
    title: str,
    sections: list[dict],
) -> list[dict]:
    """Convert chunked sections into final chunk dicts with metadata."""
    chunks = []
    for idx, sec in enumerate(sections):
        chunk_id = hashlib.sha256(
            f"{doc_id}:{idx}:{sec['text'][:100]}".encode()
        ).hexdigest()[:16]
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_url": source_url,
            "title": title,
            "section_path": sec["section_path"],
            "chunk_index": idx,
            "text": sec["text"],
        })
    return chunks


def _parse_pdf(pdf_path: Path, converter, enc: tiktoken.Encoding) -> list[dict]:
    """Parse a single PDF and return chunks with metadata."""
    doc_id = _make_doc_id(pdf_path.name)
    source_url = f"https://docs.nvidia.com/cuda/pdf/{pdf_path.name}"

    logger.info("Parsing PDF: %s (strategy: %s)",
                pdf_path.name,
                "api_reference" if doc_id in API_REFERENCE_DOCS else "narrative")
    try:
        result = converter.convert(str(pdf_path))
    except Exception:
        logger.exception("Failed to parse %s", pdf_path.name)
        return []

    md = result.document.export_to_markdown()
    title = _extract_title(md, pdf_path.stem.replace("_", " "))
    sections = _chunk_markdown(doc_id, md, enc)
    chunks = _sections_to_chunks(doc_id, source_url, title, sections)

    logger.info("  -> %d chunks from %s", len(chunks), pdf_path.name)
    return chunks


def _parse_html(html_json_path: Path, converter, enc: tiktoken.Encoding) -> list[dict]:
    """Parse a single crawled HTML page and return chunks with metadata."""
    data = json.loads(html_json_path.read_text(encoding="utf-8"))
    slug = data["slug"]
    source_url = data["url"]
    html_content = data["html"]
    doc_id = slug

    logger.info("Parsing HTML: %s (strategy: %s)",
                slug,
                "api_reference" if doc_id in API_REFERENCE_DOCS else "narrative")

    with tempfile.NamedTemporaryFile(
        suffix=".html", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(html_content)
        tmp_path = Path(f.name)

    try:
        result = converter.convert(str(tmp_path))
    except Exception:
        logger.exception("Failed to parse HTML %s", slug)
        return []
    finally:
        tmp_path.unlink(missing_ok=True)

    md = result.document.export_to_markdown()
    title = _extract_title(md, slug.replace("-", " ").title())
    sections = _chunk_markdown(doc_id, md, enc)
    chunks = _sections_to_chunks(doc_id, source_url, title, sections)

    logger.info("  -> %d chunks from %s", len(chunks), slug)
    return chunks


def parse_docs(
    with_images: bool = False,
    vlm_captions: bool = False,
) -> None:
    """Parse all downloaded docs into chunks and write to all_chunks.jsonl.

    Args:
        with_images: Extract images (currently not wired — reserved for future).
        vlm_captions: Generate VLM captions for images (reserved for future).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    cfg = load_config()
    enc = tiktoken.get_encoding(cfg.doc_retrieval.chunking.tokenizer)

    root = _storage_root()
    pdf_dir = root / "raw" / "pdfs"
    html_dir = root / "raw" / "html"
    chunks_dir = root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []

    if pdf_dir.exists():
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        logger.info("Found %d PDFs to parse", len(pdfs))
        for pdf in pdfs:
            all_chunks.extend(_parse_pdf(pdf, converter, enc))

    if html_dir.exists():
        htmls = sorted(html_dir.glob("*.json"))
        logger.info("Found %d HTML pages to parse", len(htmls))
        for html_file in htmls:
            all_chunks.extend(_parse_html(html_file, converter, enc))

    out_path = chunks_dir / "all_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        "Total: %d chunks from %d documents -> %s",
        len(all_chunks),
        len(set(c["doc_id"] for c in all_chunks)),
        out_path,
    )
