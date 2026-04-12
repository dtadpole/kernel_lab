"""CLI for doc_retrieval: build, rebuild, nuke, find, browse, read.

Uses Hydra compose API for configuration. Subcommand is the first positional arg.

Usage:
    python -m doc_retrieval build
    python -m doc_retrieval rebuild
    python -m doc_retrieval nuke
    python -m doc_retrieval find query="shared memory bank conflicts" top_k=10
    python -m doc_retrieval browse doc_id=cuda-c-programming-guide depth=1
    python -m doc_retrieval read doc_id=cuda-c-programming-guide section_id=shared-memory
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def _load_cfg(overrides: list[str]):
    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _do_build() -> None:
    """Download HTML docs, parse into chunks, build BM25 index."""
    from doc_retrieval.downloader import download_docs
    from doc_retrieval.parser import parse_docs
    from doc_retrieval.indexer import build_index
    download_docs()
    parse_docs()
    build_index()


def _do_nuke() -> None:
    """Delete derived artifacts (chunks + index). Raw HTML is kept."""
    import shutil
    chunks_dir = Path.home() / ".doc_retrieval" / "chunks"
    index_dir = Path.home() / ".doc_retrieval" / "index"
    for d in (chunks_dir, index_dir):
        if d.exists():
            shutil.rmtree(d)
            logging.getLogger(__name__).info(f"Removed {d}")


def cmd_build(cfg) -> None:
    _do_build()


def cmd_rebuild(cfg) -> None:
    _do_nuke()
    _do_build()


def cmd_nuke(cfg) -> None:
    _do_nuke()


def cmd_find(cfg, overrides: dict) -> None:
    from doc_retrieval.searcher import cli_find

    query = overrides.get("query", "")
    top_k = int(overrides.get("top_k", cfg.doc_retrieval.search.default_top_k))

    if not query:
        print("Error: query is required. Usage: find query=\"your search query\"", file=sys.stderr)
        sys.exit(1)

    cli_find(query=query, top_k=top_k)


def cmd_browse(cfg, overrides: dict) -> None:
    from doc_retrieval.searcher import DocSearcher

    doc_id = overrides.get("doc_id", "")
    section_id = overrides.get("section_id")
    depth = int(overrides.get("depth", 4))

    if not doc_id:
        print("Error: doc_id is required. Usage: browse doc_id=cuda-c-programming-guide", file=sys.stderr)
        sys.exit(1)

    searcher = DocSearcher()
    result = searcher.browse_toc(doc_id=doc_id, section_id=section_id, depth=depth)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_read(cfg, overrides: dict) -> None:
    from doc_retrieval.searcher import DocSearcher

    doc_id = overrides.get("doc_id", "")
    section_id = overrides.get("section_id", "")

    if not doc_id or not section_id:
        print("Error: doc_id and section_id required. Usage: read doc_id=... section_id=...", file=sys.stderr)
        sys.exit(1)

    searcher = DocSearcher()
    result = searcher.read_section(doc_id=doc_id, section_id=section_id)
    if result is None:
        print(f"Section '{section_id}' not found in '{doc_id}'")
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


COMMANDS = {
    "build": cmd_build,
    "rebuild": cmd_rebuild,
    "nuke": cmd_nuke,
    "find": cmd_find,
    "browse": cmd_browse,
    "read": cmd_read,
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(f"Usage: python -m doc_retrieval <command> [key=value ...]", file=sys.stderr)
        print(f"Commands: {', '.join(COMMANDS)}", file=sys.stderr)
        sys.exit(1)

    command = args[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}. Available: {', '.join(COMMANDS)}", file=sys.stderr)
        sys.exit(1)

    # Separate Hydra overrides (key=value) from command-specific args
    hydra_overrides = [a for a in args[1:] if "=" in a and not a.startswith("-")]
    # Parse key=value pairs for command-specific use
    cmd_overrides = {}
    for ov in hydra_overrides:
        key, _, value = ov.partition("=")
        cmd_overrides[key] = value

    cfg = _load_cfg([])  # Load base config (no Hydra overrides needed for most commands)

    handler = COMMANDS[command]
    if command in ("build", "rebuild", "nuke"):
        handler(cfg)
    else:
        handler(cfg, cmd_overrides)
