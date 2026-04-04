"""Minimal OpenAI-compatible embedding server using Qwen3-Embedding.

Exposes /v1/embeddings (OpenAI-compatible) and /health endpoints.
Designed to run on a GPU host behind a systemd user unit.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("embed_server")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_model_id: str = ""


def _load_model(model_id: str) -> None:
    """Load model and tokenizer onto GPU."""
    global _model, _tokenizer, _model_id
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading tokenizer: %s", model_id)
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info("Loading model: %s", model_id)
    _model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda().eval()

    _model_id = model_id
    logger.info("Model loaded on GPU: %s", model_id)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    model_id = os.environ.get("EMBED_MODEL_ID", "Qwen/Qwen3-Embedding-4B")
    _load_model(model_id)
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = ""


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")

    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(400, "Empty input")

    encoded = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = _model(**encoded)
        # Use last_hidden_state with mean pooling (masked)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        # L2-normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    data = [
        EmbeddingData(embedding=emb.tolist(), index=i)
        for i, emb in enumerate(embeddings.cpu())
    ]

    return EmbeddingResponse(
        data=data,
        model=_model_id,
        usage={"prompt_tokens": int(encoded["attention_mask"].sum()), "total_tokens": int(encoded["attention_mask"].sum())},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=46982)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)
