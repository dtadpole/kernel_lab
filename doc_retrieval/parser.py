"""Parse downloaded NVIDIA CUDA HTML docs into structured text chunks.

HTML docs are parsed with BeautifulSoup (via ``html_parser.py``) to preserve
anchor IDs for navigation.  Produces three output files:

- ``all_chunks.jsonl`` — 512-token chunks for search
- ``toc.jsonl`` — document TOC trees for browsing
- ``sections.jsonl`` — full section content for reading
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


def parse_docs() -> None:
    """Parse all downloaded HTML docs into chunks, sections, and TOC.

    HTML docs are parsed with BeautifulSoup to preserve anchor IDs
    for navigation.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import tiktoken

    cfg = load_config()
    enc = tiktoken.get_encoding(cfg.doc_retrieval.chunking.tokenizer)

    raw = _raw_root()
    html_dir = raw / "html"
    runtime = _runtime_root()
    chunks_dir = runtime / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    all_toc: list[dict] = []
    all_sections: list[dict] = []

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
