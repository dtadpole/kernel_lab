"""Build BM25 and FAISS indices from parsed document chunks."""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np

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


def _build_bm25(chunks: list[dict]) -> None:
    """Build and save a BM25 index."""
    from rank_bm25 import BM25Okapi

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


def _build_dense(chunks: list[dict]) -> None:
    """Build and save a FAISS dense index with API embeddings."""
    import faiss

    from doc_retrieval.embeddings import EmbeddingCache, create_client

    index_dir = _runtime_root() / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    client = create_client()
    cache = EmbeddingCache(index_dir / "embedding_cache.npz")

    # Determine which chunks need embedding
    texts_to_embed: list[str] = []
    indices_to_embed: list[int] = []

    for i, chunk in enumerate(chunks):
        cached = cache.get(chunk["text"])
        if cached is None:
            texts_to_embed.append(chunk["text"])
            indices_to_embed.append(i)

    logger.info(
        "Need to embed %d/%d chunks (%d cached)",
        len(texts_to_embed),
        len(chunks),
        len(chunks) - len(texts_to_embed),
    )

    # Embed uncached chunks
    if texts_to_embed:
        new_embeddings = client.embed_texts(texts_to_embed)
        for idx, emb in zip(indices_to_embed, new_embeddings):
            cache.put(chunks[idx]["text"], emb)
        cache.save()

    # Collect all embeddings in order
    all_embeddings = np.zeros(
        (len(chunks), client.dimensions), dtype=np.float32
    )
    for i, chunk in enumerate(chunks):
        emb = cache.get(chunk["text"])
        if emb is not None:
            all_embeddings[i] = emb

    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_embeddings = all_embeddings / norms

    # Build FAISS flat index
    logger.info("Building FAISS IndexFlatIP (dim=%d)...", client.dimensions)
    index = faiss.IndexFlatIP(client.dimensions)
    index.add(all_embeddings)

    # Save
    faiss.write_index(index, str(index_dir / "faiss.index"))
    chunk_ids = [c["chunk_id"] for c in chunks]
    with open(index_dir / "chunk_ids.json", "w") as f:
        json.dump(chunk_ids, f)

    metadata = {
        "num_chunks": len(chunks),
        "embedding_base_url": client.base_url,
        "embedding_model": client.model,
        "dimensions": client.dimensions,
    }
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "FAISS index saved: %d vectors, dim=%d",
        index.ntotal,
        client.dimensions,
    )


def build_indices(only: str | None = None) -> None:
    """Build search indices from parsed chunks.

    Args:
        only: If "bm25" or "dense", build only that index. None = both.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    chunks = _load_chunks()

    if only is None or only == "bm25":
        _build_bm25(chunks)

    if only is None or only == "dense":
        _build_dense(chunks)
