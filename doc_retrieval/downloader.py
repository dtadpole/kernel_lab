"""Download NVIDIA CUDA Toolkit HTML documentation pages.

Crawls Sphinx HTML pages from ``docs.nvidia.com/cuda/`` into the repo at
``data/nvidia-docs/html/{slug}/index.html`` with referenced ``_images/``
alongside.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import httpx

from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; doc_retrieval/1.0; CUDA docs indexer)"
)


def _raw_root() -> Path:
    """Return the raw data root (in-repo by default)."""
    cfg = load_config()
    raw = cfg.doc_retrieval.storage.raw_root
    p = Path(raw).expanduser()
    if not p.is_absolute():
        # Relative paths are relative to repo root (parent of doc_retrieval/)
        p = Path(__file__).resolve().parents[1] / p
    return p


def _html_pages() -> list[dict]:
    """Return the list of HTML pages to crawl."""
    cfg = load_config()
    return [
        {"url": p.url, "slug": p.slug}
        for p in cfg.doc_retrieval.download.html_pages
    ]


def _extract_image_refs(html: str) -> list[str]:
    """Extract ``_images/`` relative paths from HTML ``src`` attributes."""
    return sorted(set(re.findall(r'src=["\'](_images/[^"\' ]+)["\']', html)))


# ---------------------------------------------------------------------------
# HTML crawl with images
# ---------------------------------------------------------------------------

async def _download_images(
    client: httpx.AsyncClient,
    slug: str,
    image_refs: list[str],
    images_dir: Path,
    delay: float,
) -> int:
    """Download referenced images for an HTML doc. Returns success count."""
    base = f"https://docs.nvidia.com/cuda/{slug}/"
    ok = 0
    for ref in image_refs:
        filename = ref.split("/", 1)[1]  # strip "_images/" prefix
        dest = images_dir / filename
        if dest.exists():
            ok += 1
            continue
        url = base + ref
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.debug("  -> image: %s (%d bytes)", filename, len(resp.content))
            ok += 1
            await asyncio.sleep(delay * 0.2)
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("  -> Failed image %s: %s", url, exc)
    return ok


async def _crawl_html_page(
    client: httpx.AsyncClient,
    url: str,
    slug: str,
    out_dir: Path,
    delay: float,
) -> bool:
    """Crawl a single HTML page and download its images. Returns True on success."""
    slug_dir = out_dir / slug
    index_path = slug_dir / "index.html"

    if index_path.exists():
        logger.info("Already crawled: %s", slug)
        return True

    logger.info("Crawling %s ...", url)
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        slug_dir.mkdir(parents=True, exist_ok=True)
        index_path.write_text(html, encoding="utf-8")
        logger.info("  -> saved %s/index.html (%d chars)", slug, len(html))

        image_refs = _extract_image_refs(html)
        if image_refs:
            images_dir = slug_dir / "_images"
            images_dir.mkdir(exist_ok=True)
            ok = await _download_images(client, slug, image_refs, images_dir, delay)
            logger.info("  -> downloaded %d/%d images for %s", ok, len(image_refs), slug)

        await asyncio.sleep(delay)
        return True
    except httpx.HTTPStatusError as exc:
        logger.warning("  -> HTTP %d for %s", exc.response.status_code, url)
        return False
    except httpx.RequestError as exc:
        logger.warning("  -> Request error for %s: %s", url, exc)
        return False


async def _crawl_html_pages() -> None:
    """Crawl all configured HTML doc pages with images."""
    cfg = load_config()
    dl = cfg.doc_retrieval.download
    delay = dl.request_delay

    out_dir = _raw_root() / "html"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = _html_pages()
    logger.info("Crawling %d HTML pages to %s ...", len(pages), out_dir)

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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def download_docs() -> None:
    """Download NVIDIA CUDA HTML documentation pages.

    Raw docs are written to ``data/nvidia-docs/html/`` in the repo.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(_crawl_html_pages())
