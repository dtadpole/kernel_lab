"""Download NVIDIA CUDA Toolkit PDFs and crawl HTML documentation pages."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; doc_retrieval/1.0; CUDA docs indexer)"
)


def _storage_root() -> Path:
    cfg = load_config()
    root = cfg.doc_retrieval.storage.root
    return Path(root).expanduser()


def _pdf_list(tier: str) -> list[str]:
    """Return the list of PDF filenames for the requested tier."""
    cfg = load_config()
    dl = cfg.doc_retrieval.download
    if tier == "1":
        return list(dl.tier1)
    if tier == "2":
        return list(dl.tier1) + list(dl.tier2)
    if tier == "3" or tier == "all":
        return list(dl.tier1) + list(dl.tier2) + list(dl.tier3)
    return list(dl.tier1)


def _html_pages() -> list[dict]:
    """Return the list of HTML pages to crawl."""
    cfg = load_config()
    return [
        {"url": p.url, "slug": p.slug}
        for p in cfg.doc_retrieval.download.html_pages
    ]


async def _download_pdf(
    client: httpx.AsyncClient,
    base_url: str,
    filename: str,
    out_dir: Path,
    delay: float,
) -> bool:
    """Download a single PDF. Returns True on success."""
    dest = out_dir / filename
    if dest.exists():
        logger.info("Already downloaded: %s", filename)
        return True

    url = f"{base_url}{filename}"
    logger.info("Downloading %s ...", url)
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size_mb = len(resp.content) / (1024 * 1024)
        logger.info("  -> saved %s (%.1f MB)", dest.name, size_mb)
        await asyncio.sleep(delay)
        return True
    except httpx.HTTPStatusError as exc:
        logger.warning("  -> HTTP %d for %s", exc.response.status_code, url)
        return False
    except httpx.RequestError as exc:
        logger.warning("  -> Request error for %s: %s", url, exc)
        return False


async def _download_pdfs(tier: str) -> None:
    """Download all PDFs for the given tier."""
    cfg = load_config()
    dl = cfg.doc_retrieval.download
    base_url = dl.pdf_base_url
    delay = dl.request_delay

    out_dir = _storage_root() / "raw" / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _pdf_list(tier)
    logger.info("Downloading %d PDFs (tier %s) ...", len(pdfs), tier)

    limits = httpx.Limits(
        max_connections=dl.max_connections,
        max_keepalive_connections=dl.max_connections,
    )
    async with httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(60.0, connect=10.0),
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        ok = 0
        for filename in pdfs:
            if await _download_pdf(client, base_url, filename, out_dir, delay):
                ok += 1

    logger.info("PDF download complete: %d/%d succeeded", ok, len(pdfs))


async def _crawl_html_page(
    client: httpx.AsyncClient,
    url: str,
    slug: str,
    out_dir: Path,
    delay: float,
) -> bool:
    """Crawl a single HTML page. Returns True on success."""
    dest = out_dir / f"{slug}.json"
    if dest.exists():
        logger.info("Already crawled: %s", slug)
        return True

    logger.info("Crawling %s ...", url)
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        data = {
            "url": str(resp.url),
            "slug": slug,
            "html": html,
            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("  -> saved %s (%d chars)", dest.name, len(html))
        await asyncio.sleep(delay)
        return True
    except httpx.HTTPStatusError as exc:
        logger.warning("  -> HTTP %d for %s", exc.response.status_code, url)
        return False
    except httpx.RequestError as exc:
        logger.warning("  -> Request error for %s: %s", url, exc)
        return False


async def _crawl_html_pages() -> None:
    """Crawl all configured HTML-only doc pages."""
    cfg = load_config()
    dl = cfg.doc_retrieval.download
    delay = dl.request_delay

    out_dir = _storage_root() / "raw" / "html"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = _html_pages()
    logger.info("Crawling %d HTML pages ...", len(pages))

    limits = httpx.Limits(
        max_connections=dl.max_connections,
        max_keepalive_connections=dl.max_connections,
    )
    async with httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(60.0, connect=10.0),
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        ok = 0
        for page in pages:
            if await _crawl_html_page(
                client, page["url"], page["slug"], out_dir, delay
            ):
                ok += 1

    logger.info("HTML crawl complete: %d/%d succeeded", ok, len(pages))


def download_docs(
    tier: str = "all",
    pdf_only: bool = False,
    html_only: bool = False,
) -> None:
    """Download NVIDIA CUDA documentation (PDFs and/or HTML pages).

    Args:
        tier: PDF download tier ("1", "2", "3", or "all").
        pdf_only: If True, skip HTML crawling.
        html_only: If True, skip PDF downloads.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    async def _run() -> None:
        if not html_only:
            await _download_pdfs(tier)
        if not pdf_only:
            await _crawl_html_pages()

    asyncio.run(_run())
