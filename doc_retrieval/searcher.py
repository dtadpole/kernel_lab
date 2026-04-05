"""BM25 search interface for CUDA documentation.

When a matched chunk is shorter than ``min_context_tokens`` (default 300),
the searcher automatically expands it by merging adjacent chunks from the
same document until the target length is reached.  This keeps the index
granular for matching while returning useful context to the caller.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

from doc_retrieval.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    source_url: str
    title: str
    section_path: str
    text: str
    score: float


class DocSearcher:
    """Lazy-loading document searcher using BM25."""

    def __init__(self, index_dir: Path | None = None):
        cfg = load_config()
        if index_dir is None:
            index_dir = (
                Path(cfg.doc_retrieval.storage.runtime_root).expanduser() / "index"
            )
        self._index_dir = index_dir

        # Lazy-loaded state
        self._chunks: list[dict] | None = None
        self._chunk_id_to_idx: dict[str, int] = {}
        self._bm25 = None

        # Navigation data (lazy-loaded)
        self._toc: list[dict] | None = None
        self._sections_data: list[dict] | None = None
        self._runtime_root: Path | None = None

    def _load_chunks(self) -> list[dict]:
        if self._chunks is None:
            chunks_dir = self._index_dir.parent / "chunks"
            chunks_path = chunks_dir / "all_chunks.jsonl"
            self._chunks = []
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._chunks.append(json.loads(line))
            # Build chunk_id -> index lookup
            self._chunk_id_to_idx = {
                c["chunk_id"]: i for i, c in enumerate(self._chunks)
            }
        return self._chunks

    def _load_bm25(self):
        if self._bm25 is None:
            bm25_path = self._index_dir / "bm25.pkl"
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            logger.info("BM25 index loaded from %s", bm25_path)
        return self._bm25

    @staticmethod
    def _top_section(section_path: str) -> str:
        """Extract the top-level section (first component of the path)."""
        return section_path.split(" > ")[0] if " > " in section_path else section_path

    def _is_same_segment(self, a: int, b: int) -> bool:
        """Check if two chunk indices belong to the same segment.

        Same segment = same doc_id AND same top-level section.
        """
        chunks = self._chunks
        if chunks[a]["doc_id"] != chunks[b]["doc_id"]:
            return False
        return self._top_section(chunks[a]["section_path"]) == self._top_section(chunks[b]["section_path"])

    def _with_neighbors(self, idx: int) -> str:
        """Return the chunk text with 1 neighbor merged on each side.

        Stops at segment boundaries (different doc or top-level section).
        """
        chunks = self._chunks
        parts = []

        if idx - 1 >= 0 and self._is_same_segment(idx - 1, idx):
            parts.append(chunks[idx - 1]["text"])

        parts.append(chunks[idx]["text"])

        if idx + 1 < len(chunks) and self._is_same_segment(idx, idx + 1):
            parts.append(chunks[idx + 1]["text"])

        return "\n\n".join(parts)

    def _chunk_to_result(self, chunk: dict, score: float) -> SearchResult:
        idx = self._chunk_id_to_idx.get(chunk["chunk_id"])
        if idx is not None:
            text = self._with_neighbors(idx)
        else:
            text = chunk["text"]
        return SearchResult(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            source_url=chunk["source_url"],
            title=chunk["title"],
            section_path=chunk["section_path"],
            text=text,
            score=score,
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 keyword search."""
        import re

        import numpy as np

        chunks = self._load_chunks()
        bm25 = self._load_bm25()

        # Tokenize query same way as corpus
        query_tokens = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+", query.lower()
        )
        scores = bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            self._chunk_to_result(chunks[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def _get_runtime_root(self) -> Path:
        if self._runtime_root is not None:
            return self._runtime_root
        cfg = load_config()
        return Path(cfg.doc_retrieval.storage.runtime_root).expanduser()

    def _load_toc(self) -> list[dict]:
        if self._toc is not None:
            return self._toc
        toc_path = self._get_runtime_root() / "chunks" / "toc.jsonl"
        if not toc_path.exists():
            self._toc = []
            return self._toc
        self._toc = []
        with open(toc_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._toc.append(json.loads(line))
        return self._toc

    def _load_sections_data(self) -> list[dict]:
        if self._sections_data is not None:
            return self._sections_data
        sec_path = self._get_runtime_root() / "chunks" / "sections.jsonl"
        if not sec_path.exists():
            self._sections_data = []
            return self._sections_data
        self._sections_data = []
        with open(sec_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._sections_data.append(json.loads(line))
        return self._sections_data

    def browse_toc(
        self,
        doc_id: str,
        section_id: str | None = None,
        depth: int = 2,
    ) -> list[dict] | dict:
        """Browse the document TOC tree."""
        toc = self._load_toc()
        doc_toc = [t for t in toc if t["doc_id"] == doc_id]

        if not doc_toc:
            return []

        by_id = {t["section_id"]: t for t in doc_toc}

        def _node_summary(entry: dict, remaining_depth: int) -> dict:
            node = {
                "section_id": entry["section_id"],
                "title": entry["title"],
                "heading_level": entry["heading_level"],
                "children_count": len(entry.get("children", [])),
            }
            if remaining_depth > 1 and entry.get("children"):
                node["children"] = [
                    _node_summary(by_id[cid], remaining_depth - 1)
                    for cid in entry["children"]
                    if cid in by_id
                ]
            return node

        if section_id is None:
            top_level = [t for t in doc_toc if t["parent_section_id"] is None]
            return [_node_summary(t, depth) for t in top_level]

        if section_id not in by_id:
            return []

        return _node_summary(by_id[section_id], depth)

    def read_section(
        self,
        doc_id: str,
        section_id: str,
    ) -> dict | None:
        """Read full section content with navigation context."""
        sections = self._load_sections_data()
        toc = self._load_toc()

        sec = next(
            (s for s in sections if s["doc_id"] == doc_id and s["section_id"] == section_id),
            None,
        )
        if sec is None:
            return None

        toc_entry = next(
            (t for t in toc if t["doc_id"] == doc_id and t["section_id"] == section_id),
            None,
        )

        parent_id = toc_entry["parent_section_id"] if toc_entry else None

        prev_sibling = None
        next_sibling = None
        if toc_entry and parent_id:
            parent = next(
                (t for t in toc if t["doc_id"] == doc_id and t["section_id"] == parent_id),
                None,
            )
            if parent and parent.get("children"):
                siblings = parent["children"]
                try:
                    idx = siblings.index(section_id)
                    if idx > 0:
                        prev_sibling = siblings[idx - 1]
                    if idx < len(siblings) - 1:
                        next_sibling = siblings[idx + 1]
                except ValueError:
                    pass

        return {
            "content": sec["content"],
            "title": sec["title"],
            "heading_path": sec["heading_path"],
            "token_count": sec["token_count"],
            "nav": {
                "parent": parent_id,
                "prev_sibling": prev_sibling,
                "next_sibling": next_sibling,
            },
            "deep_link": sec["deep_link"],
        }


def cli_find(query: str, top_k: int = 5) -> None:
    """Run a search from the CLI and print results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    searcher = DocSearcher()
    results = searcher.search(query, top_k)

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Mode: bm25 | Results: {len(results)}")
    print(f"{'='*80}")

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.score:.4f}) ---")
        print(f"Title: {r.title}")
        print(f"Section: {r.section_path}")
        print(f"URL: {r.source_url}")
        print(f"Text ({len(r.text)} chars):")
        # Show first 500 chars
        text_preview = r.text[:500]
        if len(r.text) > 500:
            text_preview += "..."
        print(text_preview)
