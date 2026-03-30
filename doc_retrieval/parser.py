"""Parse downloaded NVIDIA CUDA docs into structured text chunks.

PDFs are parsed with Docling. HTML docs are parsed with BeautifulSoup
(via ``html_parser.py``) to preserve anchor IDs for navigation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import tiktoken

from doc_retrieval.chunking import (
    API_REFERENCE_DOCS,
    chunk_api_reference,
    chunk_narrative,
)
from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)


def _raw_root() -> Path:
    """Return the raw data root (in-repo)."""
    cfg = load_config()
    p = Path(cfg.doc_retrieval.storage.raw_root).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    return p


def _runtime_root() -> Path:
    """Return the runtime storage root (for derived artifacts)."""
    cfg = load_config()
    return Path(cfg.doc_retrieval.storage.runtime_root).expanduser()


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
