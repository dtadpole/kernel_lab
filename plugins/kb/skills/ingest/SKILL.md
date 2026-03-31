---
name: ingest
description: Add new documents to the knowledge base search index
user-invocable: true
argument-hint: <path-or-url>
---

# TODO: Ingest Skill

Add new documents (PDF or HTML) to the knowledge base index.

## Planned Workflow

1. Accept a file path or URL as input
2. Run the doc_retrieval parser (Docling for PDF, BeautifulSoup for HTML)
3. Chunk the parsed content
4. Append chunks to the existing index (BM25 + FAISS)

## MCP Tools Needed

- May need new MCP tools or use shell commands to invoke `doc_retrieval` CLI:
  - `python -m doc_retrieval parse <path>`
  - `python -m doc_retrieval index --append`

## Status: NOT IMPLEMENTED
