# doc_retrieval/Makefile
#
# Targets map 1:1 to the kb plugin skills:
#
#   kb:index skill            kb:docs skill           kb:service skill
#   ─────────────────         ─────────────────        ─────────────────
#   make download             make find Q="..."        (see plugins/kb/deploy/cli.py)
#   make parse                make browse DOC=...
#   make index                make read DOC=... SEC=...
#   make rebuild
#   make cleanup

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

REPO_ROOT  := $(shell git -C $(dir $(abspath $(lastword $(MAKEFILE_LIST)))) rev-parse --show-toplevel)
PYTHON     := PYTHONPATH=$(REPO_ROOT) $(REPO_ROOT)/doc_retrieval/.venv/bin/python
VENV       := $(REPO_ROOT)/doc_retrieval/.venv
RAW_DATA   := $(REPO_ROOT)/data/nvidia-docs
RUNTIME    := $(HOME)/.doc_retrieval

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: $(VENV)/bin/python
$(VENV)/bin/python:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(VENV)/bin/python \
		-r $(REPO_ROOT)/doc_retrieval/requirements.txt pytest

# ===========================================================================
# kb:index — download, parse, index, rebuild, cleanup
# ===========================================================================

download: venv ## Download NVIDIA CUDA HTML docs
	$(PYTHON) -m doc_retrieval download

parse: venv ## Parse HTML docs into chunks, TOC, and sections
	$(PYTHON) -m doc_retrieval parse

index: venv ## Build search indices (BM25 + FAISS)
	$(PYTHON) -m doc_retrieval index

index-bm25: venv ## Build BM25 index only
	$(PYTHON) -m doc_retrieval index --only bm25

index-dense: venv ## Build FAISS dense index only
	$(PYTHON) -m doc_retrieval index --only dense

rebuild: venv ## Full pipeline: download → parse → index
	$(PYTHON) -m doc_retrieval download
	$(PYTHON) -m doc_retrieval parse
	$(PYTHON) -m doc_retrieval index
	@$(MAKE) -s status

cleanup: ## Nuke chunks + indices. Raw docs untouched.
	rm -rf $(RUNTIME)/chunks $(RUNTIME)/index
	@echo "Chunks and indices removed. Run 'make rebuild' to regenerate."

status: ## Show current data stats
	@echo "HTML:    $$(ls -d $(RAW_DATA)/html/*/ 2>/dev/null | wc -l) docs"
	@echo "Chunks:  $$(wc -l < $(RUNTIME)/chunks/all_chunks.jsonl 2>/dev/null || echo 0)"
	@echo "TOC:     $$(wc -l < $(RUNTIME)/chunks/toc.jsonl 2>/dev/null || echo 0) entries"
	@echo "Sections:$$(wc -l < $(RUNTIME)/chunks/sections.jsonl 2>/dev/null || echo 0) entries"
	@echo "BM25:    $$(test -f $(RUNTIME)/index/bm25.pkl && du -sh $(RUNTIME)/index/bm25.pkl | cut -f1 || echo 'not built')"
	@echo "FAISS:   $$(test -f $(RUNTIME)/index/faiss.index && du -sh $(RUNTIME)/index/faiss.index | cut -f1 || echo 'not built')"

# ===========================================================================
# kb:docs — find, browse, read
# ===========================================================================

Q   ?= shared memory bank conflicts
DOC ?= cuda-c-programming-guide
SEC ?=
DEPTH ?= 2
MODE ?= bm25
TOP_K ?= 5

find: venv ## Search docs (Q="query" [MODE=hybrid] [TOP_K=5])
	$(PYTHON) -m doc_retrieval find "$(Q)" --mode $(MODE) --top-k $(TOP_K)

browse: venv ## Browse TOC (DOC=slug [SEC=section-id] [DEPTH=2])
	$(PYTHON) -m doc_retrieval browse $(DOC) $(if $(SEC),--section-id $(SEC)) --depth $(DEPTH)

read: venv ## Read a section (DOC=slug SEC=section-id)
	@if [ -z "$(SEC)" ]; then echo "Usage: make read DOC=<slug> SEC=<section-id>"; exit 1; fi
	$(PYTHON) -m doc_retrieval read $(DOC) $(SEC)

# ===========================================================================
# Testing
# ===========================================================================

test: venv ## Run all tests
	$(PYTHON) -m pytest $(REPO_ROOT)/doc_retrieval/tests/ -v

test-bm25: venv ## Run BM25 tests only
	$(PYTHON) -m pytest $(REPO_ROOT)/doc_retrieval/tests/test_search.py -v -k "TestBM25"

test-dense: venv ## Run Dense tests only
	$(PYTHON) -m pytest $(REPO_ROOT)/doc_retrieval/tests/test_search.py -v -k "TestDense"

.PHONY: help venv download parse index index-bm25 index-dense rebuild cleanup status \
        find browse read test test-bm25 test-dense
