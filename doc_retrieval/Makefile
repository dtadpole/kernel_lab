# doc_retrieval/Makefile
#
# Targets map 1:1 to the ik plugin skills:
#
#   ik:index skill            ik:docs skill
#   ─────────────────         ─────────────────
#   make download             make find Q="..."
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
# ik:index — download, parse, index, rebuild, cleanup
# ===========================================================================

download: venv ## Download NVIDIA CUDA HTML docs
	$(PYTHON) -m doc_retrieval download

parse: venv ## Parse HTML docs into chunks, TOC, and sections
	$(PYTHON) -m doc_retrieval parse

index: venv ## Build BM25 search index
	$(PYTHON) -m doc_retrieval index

rebuild: venv ## Full pipeline: download → parse → index
	$(PYTHON) -m doc_retrieval download
	$(PYTHON) -m doc_retrieval parse
	$(PYTHON) -m doc_retrieval index
	@$(MAKE) -s status

cleanup: ## Nuke chunks + index. Raw docs untouched.
	rm -rf $(RUNTIME)/chunks $(RUNTIME)/index
	@echo "Chunks and index removed. Run 'make rebuild' to regenerate."

status: ## Show current data stats
	@echo "HTML:    $$(ls -d $(RAW_DATA)/html/*/ 2>/dev/null | wc -l) docs"
	@echo "Chunks:  $$(wc -l < $(RUNTIME)/chunks/all_chunks.jsonl 2>/dev/null || echo 0)"
	@echo "TOC:     $$(wc -l < $(RUNTIME)/chunks/toc.jsonl 2>/dev/null || echo 0) entries"
	@echo "Sections:$$(wc -l < $(RUNTIME)/chunks/sections.jsonl 2>/dev/null || echo 0) entries"
	@echo "BM25:    $$(test -f $(RUNTIME)/index/bm25.pkl && du -sh $(RUNTIME)/index/bm25.pkl | cut -f1 || echo 'not built')"

# ===========================================================================
# ik:docs — find, browse, read
# ===========================================================================

Q   ?= shared memory bank conflicts
DOC ?= cuda-c-programming-guide
SEC ?=
DEPTH ?= 2
TOP_K ?= 5

find: venv ## Search docs (Q="query" [TOP_K=5])
	$(PYTHON) -m doc_retrieval find "$(Q)" --top-k $(TOP_K)

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

.PHONY: help venv download parse index rebuild cleanup status \
        find browse read test
