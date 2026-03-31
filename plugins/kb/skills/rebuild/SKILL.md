---
name: rebuild
description: Rebuild the entire knowledge base search index from scratch
user-invocable: true
---

# TODO: Rebuild Skill

Rebuild the entire BM25 + FAISS search index from all parsed documents.

## Planned Workflow

1. Re-parse all raw documents in `doc_retrieval/data/raw/`
2. Re-chunk all parsed content
3. Rebuild BM25 and FAISS indices from scratch

## MCP Tools Needed

- May need new MCP tools or use shell commands:
  - `python -m doc_retrieval parse --all`
  - `python -m doc_retrieval index`

## Status: NOT IMPLEMENTED
