# doc_retrieval/Makefile
#
# Usage:  make -C doc_retrieval <target>
#
#   make -C doc_retrieval ingest                  # download + parse + index
#   make -C doc_retrieval rebuild                 # re-parse + re-index (keeps downloads)
#   make -C doc_retrieval test                    # all search tests
#   make -C doc_retrieval search Q="query"        # hybrid search

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

REPO_ROOT  := $(shell git -C $(dir $(abspath $(lastword $(MAKEFILE_LIST)))) rev-parse --show-toplevel)
PYTHON     := PYTHONPATH=$(REPO_ROOT) $(REPO_ROOT)/doc_retrieval/.venv/bin/python
VENV       := $(REPO_ROOT)/doc_retrieval/.venv
DATA       := $(REPO_ROOT)/doc_retrieval/data

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: $(VENV)/bin/python
$(VENV)/bin/python:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(VENV)/bin/python \
		-r $(REPO_ROOT)/doc_retrieval/requirements.txt pytest

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

download: venv ## Download NVIDIA CUDA PDFs + HTML pages
	$(PYTHON) -m doc_retrieval download --tier all

ingest: download venv ## Full pipeline: download → parse → index
	$(PYTHON) -m doc_retrieval parse
	$(PYTHON) -m doc_retrieval index
	@$(MAKE) -s status

rebuild: venv ## Re-parse + re-index from existing downloads
	rm -f $(DATA)/chunks/* $(DATA)/index/*
	$(PYTHON) -m doc_retrieval parse
	$(PYTHON) -m doc_retrieval index
	@$(MAKE) -s status

rebuild-bm25: venv ## Rebuild BM25 index only
	rm -f $(DATA)/index/bm25.pkl
	$(PYTHON) -m doc_retrieval index --only bm25

rebuild-dense: venv ## Rebuild FAISS dense index only
	rm -f $(DATA)/index/faiss.index $(DATA)/index/chunk_ids.json
	$(PYTHON) -m doc_retrieval index --only dense

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: venv ## Run all search tests (BM25 + Dense)
	$(PYTHON) -m pytest $(REPO_ROOT)/doc_retrieval/tests/ -v

test-bm25: venv ## Run BM25 tests only
	$(PYTHON) -m pytest $(REPO_ROOT)/doc_retrieval/tests/test_search.py -v -k "TestBM25"

test-dense: venv ## Run Dense tests only
	$(PYTHON) -m pytest $(REPO_ROOT)/doc_retrieval/tests/test_search.py -v -k "TestDense"

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

Q ?= shared memory bank conflicts

search: venv ## Hybrid search (Q="your query")
	$(PYTHON) -m doc_retrieval search "$(Q)" --mode hybrid

search-bm25: venv ## BM25 keyword search (Q="your query")
	$(PYTHON) -m doc_retrieval search "$(Q)" --mode bm25

search-dense: venv ## Dense semantic search (Q="your query")
	$(PYTHON) -m doc_retrieval search "$(Q)" --mode dense

# ---------------------------------------------------------------------------
# Status / Cleanup
# ---------------------------------------------------------------------------

status: ## Show current data stats
	@echo "PDFs:    $$(ls $(DATA)/raw/pdfs/*.pdf 2>/dev/null | wc -l) files ($$(du -sh $(DATA)/raw/pdfs/ 2>/dev/null | cut -f1))"
	@echo "HTML:    $$(ls $(DATA)/raw/html/*.json 2>/dev/null | wc -l) files"
	@echo "Chunks:  $$(wc -l < $(DATA)/chunks/all_chunks.jsonl 2>/dev/null || echo 0)"
	@echo "BM25:    $$(test -f $(DATA)/index/bm25.pkl && du -sh $(DATA)/index/bm25.pkl | cut -f1 || echo 'not built')"
	@echo "FAISS:   $$(test -f $(DATA)/index/faiss.index && du -sh $(DATA)/index/faiss.index | cut -f1 || echo 'not built')"

cleanup: ## Remove all generated data (keeps source code)
	rm -rf $(DATA)/raw/pdfs/* $(DATA)/raw/html/* $(DATA)/chunks/* $(DATA)/index/*
	@echo "All data removed. Run 'make ingest' to rebuild."

.PHONY: help venv download ingest rebuild rebuild-bm25 rebuild-dense \
        test test-bm25 test-dense search search-bm25 search-dense \
        status cleanup
