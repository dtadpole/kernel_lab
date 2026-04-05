"""Build BM25 index from parsed document chunks."""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)


def _runtime_root() -> Path:
    cfg = load_config()
    return Path(cfg.doc_retrieval.storage.runtime_root).expanduser()


def _load_chunks() -> list[dict]:
    """Load all chunks from the JSONL file."""
    chunks_path = _runtime_root() / "chunks" / "all_chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. Run 'parse' first."
        )
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25, preserving CUDA identifiers.

    Keeps underscored identifiers like __syncthreads, threadIdx.x,
    __shared__ intact rather than splitting on underscores.
    """
    # Replace common CUDA dotted identifiers with underscored versions
    # so they stay as single tokens (e.g., threadIdx.x -> threadIdx_x)
    text = re.sub(r"(\w+)\.([xyzw])\b", r"\1_\2", text)

    # Split on whitespace and punctuation, but keep underscored words
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+", text.lower())
    return tokens


def build_index() -> None:
    """Build BM25 search index from parsed chunks."""
    from rank_bm25 import BM25Okapi

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    chunks = _load_chunks()

    index_dir = _runtime_root() / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Tokenizing %d chunks for BM25...", len(chunks))
    corpus = [_tokenize_for_bm25(c["text"]) for c in chunks]

    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(corpus)

    out_path = index_dir / "bm25.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 index saved to %s", out_path)
