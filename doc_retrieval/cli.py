"""CLI subcommands for doc_retrieval: download, parse, index, find, browse, read."""

from __future__ import annotations

import argparse
import sys


def cmd_download(args: argparse.Namespace) -> None:
    from doc_retrieval.downloader import download_docs

    download_docs()


def cmd_parse(args: argparse.Namespace) -> None:
    from doc_retrieval.parser import parse_docs

    parse_docs()


def cmd_index(args: argparse.Namespace) -> None:
    from doc_retrieval.indexer import build_indices

    build_indices(only=args.only)


def cmd_find(args: argparse.Namespace) -> None:
    from doc_retrieval.searcher import cli_find

    cli_find(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doc_retrieval",
        description="NVIDIA CUDA Toolkit document retrieval system",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    dl = sub.add_parser("download", help="Download NVIDIA CUDA HTML documentation")
    dl.set_defaults(func=cmd_download)

    # --- parse ---
    pa = sub.add_parser("parse", help="Parse downloaded HTML docs into chunks")
    pa.set_defaults(func=cmd_parse)

    # --- index ---
    ix = sub.add_parser("index", help="Build search indices from parsed chunks")
    ix.add_argument(
        "--only",
        choices=["bm25", "dense"],
        default=None,
        help="Build only one index type (default: both)",
    )
    ix.set_defaults(func=cmd_index)

    # --- find ---
    sr = sub.add_parser("find", help="Search CUDA documentation")
    sr.add_argument("query", help="Search query")
    sr.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    sr.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    sr.set_defaults(func=cmd_find)

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

    args = parser.parse_args()
    args.func(args)
