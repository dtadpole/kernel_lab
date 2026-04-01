"""API-based embedding client for dense retrieval.

Talks directly to an OpenAI-compatible /v1/embeddings endpoint
(e.g. HuggingFace TEI with Qwen3-Embedding-4B) via httpx.
Batches requests and caches embeddings to avoid re-computing
unchanged content.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx
import numpy as np

from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)

# Generous timeout: large batches on first load can be slow.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class EmbeddingClient:
    """Thin wrapper around an OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:46982/v1",
        model: str = "Qwen/Qwen3-Embedding-4B",
        dimensions: int = 2560,
        batch_size: int = 32,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)
        return self._client

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dimensions) float32 array."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embs = self._embed_batch(batch)
            all_embeddings.append(embs)
            if len(texts) > self.batch_size:
                logger.info(
                    "  Embedded %d/%d texts",
                    min(i + self.batch_size, len(texts)),
                    len(texts),
                )
        return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (dimensions,) float32 array."""
        return self.embed_texts([query])[0]

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        resp = client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = [d["embedding"] for d in data]
        return np.array(vecs, dtype=np.float32)


def create_client() -> EmbeddingClient:
    """Create an EmbeddingClient from Hydra config."""
    cfg = load_config()
    emb = cfg.doc_retrieval.embedding
    return EmbeddingClient(
        base_url=emb.base_url,
        model=emb.model,
        dimensions=emb.dimensions,
        batch_size=emb.batch_size,
    )


class EmbeddingCache:
    """Cache embeddings by content hash to avoid re-embedding unchanged chunks."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            data = np.load(str(cache_path), allow_pickle=True)
            self._cache = dict(data["cache"].item())
            logger.info("Loaded %d cached embeddings", len(self._cache))

    def get(self, text: str) -> np.ndarray | None:
        h = hashlib.sha256(text.encode()).hexdigest()[:32]
        return self._cache.get(h)

    def put(self, text: str, embedding: np.ndarray) -> None:
        h = hashlib.sha256(text.encode()).hexdigest()[:32]
        self._cache[h] = embedding

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.cache_path), cache=self._cache)
        logger.info("Saved %d embeddings to cache", len(self._cache))
