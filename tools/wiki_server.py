#!/usr/bin/env python3
"""Lightweight wiki HTTP server for kernel_lab_kb.

Serves Markdown files from ~/kernel_lab_kb/wiki/ as rendered HTML.
Resolves [[wiki-links]] to clickable <a> tags.

Dependencies: PyYAML, markdown-it-py (both in .venv)

Usage:
    python tools/wiki_server.py [--port 45678] [--wiki-dir ~/kernel_lab_kb/wiki]
"""

import argparse
import html
import os
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

import yaml
from markdown_it import MarkdownIt

DEFAULT_PORT = 45678
DEFAULT_WIKI_DIR = Path.home() / "kernel_lab_kb" / "wiki"
CATEGORIES = ["concepts", "patterns", "problems", "decisions", "references"]


# ---------------------------------------------------------------------------
# Markdown rendering (markdown-it-py)
# ---------------------------------------------------------------------------

_md = MarkdownIt("commonmark", {"html": True}).enable("table")


def _render_markdown(text: str) -> str:
    """Render Markdown to HTML using markdown-it-py (CommonMark + tables)."""
    return _md.render(text)


# ---------------------------------------------------------------------------
# Wiki logic
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, parts[2].strip()
    return {}, text


def _find_page(wiki_dir: Path, slug: str) -> Path | None:
    """Find a wiki page by slug. Search order:
    1. <slug>.md in root
    2. <slug>/_index.md in root (category index)
    3. <category>/<slug>.md in each category
    """
    # Root level
    root_file = wiki_dir / f"{slug}.md"
    if root_file.exists():
        return root_file

    # Category index
    cat_index = wiki_dir / slug / "_index.md"
    if cat_index.exists():
        return cat_index

    # Search all categories
    for cat in CATEGORIES:
        cat_file = wiki_dir / cat / f"{slug}.md"
        if cat_file.exists():
            return cat_file

    return None


def _resolve_wiki_links(text: str, wiki_dir: Path) -> str:
    """Replace [[slug]] with HTML links. Blue if page exists, red if not."""
    def _replace(m):
        slug = m.group(1)
        display = slug.replace("-", " ").title()
        exists = _find_page(wiki_dir, slug) is not None
        if exists:
            return f'<a href="/wiki/{slug}" class="wiki-link">{display}</a>'
        else:
            return f'<a href="/wiki/{slug}" class="wiki-link broken">{display}</a>'
    return re.sub(r"\[\[([^\]]+)\]\]", _replace, text)


def _find_backlinks(wiki_dir: Path, target_slug: str) -> list[tuple[str, str]]:
    """Find all pages that link to target_slug. Returns [(slug, title), ...]."""
    backlinks = []
    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_") and md_file.name != "_index.md":
            continue
        rel = md_file.relative_to(wiki_dir)
        if str(rel).startswith("_proposals"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        if f"[[{target_slug}]]" in content:
            meta, _ = _parse_frontmatter(content)
            if md_file.name == "_index.md":
                slug = md_file.parent.name
            else:
                slug = md_file.stem
            title = meta.get("title", slug)
            backlinks.append((slug, title))
    return backlinks


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 40px;
    line-height: 1.6;
    color: #24292e;
    background: #fff;
}
nav {
    border-bottom: 1px solid #e1e4e8;
    padding: 10px 0;
    margin-bottom: 20px;
    font-size: 14px;
}
nav a { margin-right: 16px; text-decoration: none; color: #0366d6; }
nav a:hover { text-decoration: underline; }
nav a.active { font-weight: bold; }
h1, h2, h3, h4 { margin-top: 24px; margin-bottom: 8px; }
h1 { font-size: 2em; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px; }
h2 { font-size: 1.5em; border-bottom: 1px solid #e1e4e8; padding-bottom: 6px; }
pre {
    background: #f6f8fa;
    border-radius: 6px;
    padding: 16px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.45;
}
code {
    background: #f6f8fa;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 85%;
}
pre code { background: none; padding: 0; }
.wiki-table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
}
.wiki-table th, .wiki-table td {
    border: 1px solid #e1e4e8;
    padding: 8px 12px;
    text-align: left;
}
.wiki-table th { background: #f6f8fa; font-weight: 600; }
.wiki-table tr:nth-child(even) { background: #fafbfc; }
.wiki-link { color: #0366d6; text-decoration: none; }
.wiki-link:hover { text-decoration: underline; }
.wiki-link.broken { color: #cb2431; }
.meta {
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 13px;
}
.meta .title { font-size: 11px; color: #586069; text-transform: uppercase; }
.meta .tags span {
    display: inline-block;
    background: #e1e4e8;
    color: #24292e;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
}
.backlinks {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #e1e4e8;
    font-size: 13px;
    color: #586069;
}
.backlinks a { color: #0366d6; }
blockquote {
    border-left: 4px solid #dfe2e5;
    padding: 0 16px;
    color: #6a737d;
    margin: 16px 0;
}
hr { border: none; border-top: 1px solid #e1e4e8; margin: 24px 0; }
"""


def _render_page(wiki_dir: Path, slug: str) -> str | None:
    """Render a wiki page to full HTML. Returns None if not found."""
    page_path = _find_page(wiki_dir, slug)
    if page_path is None:
        return None

    raw = page_path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    # Resolve wiki links before rendering markdown
    body = _resolve_wiki_links(body, wiki_dir)
    body_html = _render_markdown(body)

    title = meta.get("title", slug)
    category = meta.get("category", "")
    tags = meta.get("tags", [])
    status = meta.get("status", "")
    created = meta.get("created", "")
    updated = meta.get("updated", "")
    sources = meta.get("sources", [])

    # Meta block
    meta_html = ""
    if any([category, tags, created]):
        parts = []
        if category:
            parts.append(f'<span class="title">Category:</span> <a href="/wiki/{category}">{category}</a>')
        if status:
            parts.append(f'<span class="title">Status:</span> {html.escape(str(status))}')
        if created:
            parts.append(f'<span class="title">Created:</span> {html.escape(str(created))}')
        if updated and str(updated) != str(created):
            parts.append(f'<span class="title">Updated:</span> {html.escape(str(updated))}')
        if tags:
            tag_html = " ".join(f"<span>{html.escape(t)}</span>" for t in tags)
            parts.append(f'<span class="title">Tags:</span> <span class="tags">{tag_html}</span>')
        if sources:
            src_html = ", ".join(html.escape(str(s)) for s in sources[:5])
            parts.append(f'<span class="title">Sources:</span> {src_html}')
        meta_html = '<div class="meta">' + " &nbsp;|&nbsp; ".join(parts) + "</div>"

    # Backlinks
    backlinks = _find_backlinks(wiki_dir, slug)
    backlinks_html = ""
    if backlinks:
        links = ", ".join(f'<a href="/wiki/{s}">{html.escape(t)}</a>' for s, t in backlinks)
        backlinks_html = f'<div class="backlinks"><strong>Backlinks:</strong> {links}</div>'

    # Navigation
    nav_items = ['<a href="/wiki/">Home</a>']
    for cat in CATEGORIES:
        active = ' class="active"' if cat == category else ""
        nav_items.append(f'<a href="/wiki/{cat}"{active}>{cat}</a>')
    nav_html = " ".join(nav_items)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)} — Kernel Lab Wiki</title>
<style>{CSS}</style>
</head>
<body>
<nav>{nav_html}</nav>
{meta_html}
{body_html}
{backlinks_html}
</body>
</html>"""


def _render_404(slug: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Not Found</title><style>{CSS}</style></head>
<body>
<nav><a href="/wiki/">Home</a></nav>
<h1>Page Not Found</h1>
<p>No wiki page found for <code>{html.escape(slug)}</code>.</p>
<p><a href="/wiki/">← Back to index</a></p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class WikiHandler(BaseHTTPRequestHandler):
    wiki_dir: Path = DEFAULT_WIKI_DIR

    def do_GET(self):
        path = unquote(self.path).rstrip("/")

        # Root → redirect to wiki index
        if path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/wiki/")
            self.end_headers()
            return

        # /wiki/ or /wiki/<slug>
        if path.startswith("/wiki"):
            slug = path[5:].strip("/")  # remove /wiki prefix
            if not slug:
                slug = "_index"

            page_html = _render_page(self.wiki_dir, slug)
            if page_html:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page_html.encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_render_404(slug).encode("utf-8"))
            return

        # Everything else → 404
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Quieter logging
        pass


def main():
    parser = argparse.ArgumentParser(description="Wiki HTTP server for kernel_lab_kb")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--wiki-dir", type=str, default=str(DEFAULT_WIKI_DIR), help="Wiki directory")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki_dir).expanduser().resolve()
    if not wiki_dir.exists():
        print(f"Error: wiki directory not found: {wiki_dir}")
        return

    WikiHandler.wiki_dir = wiki_dir

    server = HTTPServer(("0.0.0.0", args.port), WikiHandler)
    print(f"Wiki server running at http://0.0.0.0:{args.port}/")
    print(f"Wiki directory: {wiki_dir}")
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
