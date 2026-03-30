"""API-based embedding client for dense retrieval.

Supports OpenAI-compatible APIs (including local services like
HuggingFace TEI with Qwen3-Embedding-4B) and Voyage. Batches
requests and caches embeddings to avoid re-computing unchanged content.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Thin wrapper around an embedding API provider."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._base_url = base_url
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.provider in ("openai", "local"):
                from openai import OpenAI
                kwargs = {}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                elif self.provider == "local":
                    kwargs["api_key"] = "not-needed"
                self._client = OpenAI(**kwargs)
            elif self.provider == "voyage":
                import voyageai
                self._client = voyageai.Client()
            else:
                raise ValueError(f"Unknown embedding provider: {self.provider}")
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
        if self.provider in ("openai", "local"):
            kwargs = {"model": self.model, "input": texts}
            # Only pass dimensions for OpenAI proper (not local TEI)
            if self.provider == "openai":
                kwargs["dimensions"] = self.dimensions
            resp = client.embeddings.create(**kwargs)
            vecs = [d.embedding for d in resp.data]
            return np.array(vecs, dtype=np.float32)
        elif self.provider == "voyage":
            resp = client.embed(texts, model=self.model)
            return np.array(resp.embeddings, dtype=np.float32)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def create_client() -> EmbeddingClient:
    """Create an EmbeddingClient from Hydra config."""
    cfg = load_config()
    emb = cfg.doc_retrieval.embedding
    return EmbeddingClient(
        provider=emb.provider,
        model=emb.model,
        dimensions=emb.dimensions,
        batch_size=emb.batch_size,
        base_url=getattr(emb, "base_url", None),
        api_key=getattr(emb, "api_key", None),
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
