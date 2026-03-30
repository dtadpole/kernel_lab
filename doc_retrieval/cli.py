"""CLI subcommands for doc_retrieval: download, parse, index, search."""

from __future__ import annotations

import argparse
import sys


def cmd_download(args: argparse.Namespace) -> None:
    from doc_retrieval.downloader import download_docs

    download_docs(
        tier=args.tier,
        pdf_only=args.pdf_only,
        html_only=args.html_only,
    )


def cmd_parse(args: argparse.Namespace) -> None:
    from doc_retrieval.parser import parse_docs

    parse_docs(
        with_images=args.with_images,
        vlm_captions=args.vlm_captions,
    )


def cmd_index(args: argparse.Namespace) -> None:
    from doc_retrieval.indexer import build_indices

    build_indices(only=args.only)


def cmd_search(args: argparse.Namespace) -> None:
    from doc_retrieval.searcher import cli_search

    cli_search(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doc_retrieval",
        description="NVIDIA CUDA Toolkit document retrieval system",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    dl = sub.add_parser("download", help="Download NVIDIA CUDA documentation")
    dl.add_argument(
        "--tier",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which tier of PDFs to download (default: all)",
    )
    dl.add_argument("--pdf-only", action="store_true", help="Skip HTML crawling")
    dl.add_argument("--html-only", action="store_true", help="Skip PDF downloads")
    dl.set_defaults(func=cmd_download)

    # --- parse ---
    pa = sub.add_parser("parse", help="Parse downloaded docs into chunks")
    pa.add_argument(
        "--with-images", action="store_true", help="Extract images from documents"
    )
    pa.add_argument(
        "--vlm-captions",
        action="store_true",
        help="Generate image captions with Docling VLM",
    )
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

    # --- search ---
    sr = sub.add_parser("search", help="Search CUDA documentation")
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
    sr.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)
