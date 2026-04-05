# cuda_exec/scripts/Makefile — Manual CLI for the compile → evaluate → profile pipeline
#
# Output convention:
#   All files land in the standard turn-root layout:
#     $(CUDA_EXEC_ROOT)/$(RUN_TAG)/$(VERSION)/$(DIRECTION_ID)_$(DIRECTION_SLUG)/turn_$(TURN)/
#       workspace/inputs/{reference,generated}/
#       artifacts/
#       logs/
#       state/
#
# Quick smoke test (default: vecadd):
#   make test
#   make test KERNEL=matmul
#
# Manual single-config run:
#   make compile evaluate KERNEL=matmul CONFIG_SLUG=mat-1024x1024
#
# Full pipeline:
#   make clean compile evaluate profile-ncu-generated KERNEL=matmul

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT     := $(shell git -C $(dir $(abspath $(lastword $(MAKEFILE_LIST)))) rev-parse --show-toplevel)
SCRIPTS       := $(REPO_ROOT)/cuda_exec/scripts
TESTS         := $(REPO_ROOT)/cuda_exec/tests
VENV_DIR      := $(REPO_ROOT)/.venv
VENV_PYTHON   := $(VENV_DIR)/bin/python
COMPILE_SH    := $(SCRIPTS)/compile.sh
EVALUATE_PY   := $(SCRIPTS)/trial.py
PROFILE_SH    := $(SCRIPTS)/profile.sh
PROFILE_REF   := $(SCRIPTS)/profile_reference.py
FMT_EVAL_PY   := $(SCRIPTS)/fmt_eval.py
EVAL_HARNESS  := $(SCRIPTS)/eval_harness.cu

# ---------------------------------------------------------------------------
# Kernel selector — switch everything with one variable
#   make test KERNEL=vecadd        (default)
#   make test KERNEL=matmul
# ---------------------------------------------------------------------------
KERNEL ?= vecadd

# Per-kernel defaults: DIRECTION_ID, DIRECTION_SLUG, CONFIG_SLUG, CONFIG_JSON
# These are set via conditional blocks so KERNEL= is the only thing to change.
ifeq ($(KERNEL),vecadd)
  _KERNEL_DIRECTION_ID   := 1
  _KERNEL_DIRECTION_SLUG := vecadd
  _KERNEL_CONFIG_SLUG    := vec1d-n16777216
  _KERNEL_CONFIG_JSON    := {"family":"integration-vector-add","rank":1,"shape":[16777216],"input_size":16777216,"shape_kind":"1d"}
  _KERNEL_NCU_FILTER     := regex:"vecadd"
else ifeq ($(KERNEL),matmul)
  _KERNEL_DIRECTION_ID   := 2
  _KERNEL_DIRECTION_SLUG := matmul
  _KERNEL_CONFIG_SLUG    := mat-4096x4096
  _KERNEL_CONFIG_JSON    := {"family":"integration-matrix-multiplication","rank":2,"shape":[4096,4096],"input_size":16777216,"shape_kind":"2d"}
  _KERNEL_NCU_FILTER     := regex:"matmul"
else ifeq ($(KERNEL),fa4)
  _KERNEL_DIRECTION_ID   := 3
  _KERNEL_DIRECTION_SLUG := fa4
  _KERNEL_CONFIG_SLUG    := mha-causal-b2-s16384
  _KERNEL_CONFIG_JSON    := {"family":"fa4-mha","batch_size":2,"seq_len":16384,"num_heads":16,"num_kv_heads":16,"head_dim":128,"causal":true,"total_tokens":32768,"input_size":67108864,"rank":4,"shape":[2,16384,16,128],"shape_kind":"4d"}
  _KERNEL_NCU_FILTER     := regex:"fa4"
  export CUDA_EXEC_HARNESS_NUM_INPUTS := 3
else
  # Unknown kernel — assume ref at data/ref/$(KERNEL)/, gen at data/gen/$(ARCH)/$(KERNEL)/
  _KERNEL_DIRECTION_ID   := 99
  _KERNEL_DIRECTION_SLUG := $(KERNEL)
  _KERNEL_CONFIG_SLUG    := default
  _KERNEL_CONFIG_JSON    := {}
  _KERNEL_NCU_FILTER     := regex:".*"
endif

# ---------------------------------------------------------------------------
# Config selector — pick a config slug from data/configs/$(KERNEL).json
#   make evaluate KERNEL=matmul CONFIG=mat-1024x1024
#   make evaluate KERNEL=fa4 CONFIG=mha-causal-b2-s16384
# If CONFIG is not set, uses the per-kernel default from _KERNEL_CONFIG_SLUG.
# ---------------------------------------------------------------------------
_CONFIG_FILE = $(REPO_ROOT)/data/configs/$(KERNEL).json

# ---------------------------------------------------------------------------
# Metadata — derived from KERNEL, overridable on command line
# ---------------------------------------------------------------------------
CUDA_EXEC_ROOT ?= $(HOME)/.cuda_exec
RUN_TAG        ?= manual-cli-$(KERNEL)
VERSION        ?= v1
DIRECTION_ID   ?= $(_KERNEL_DIRECTION_ID)
DIRECTION_SLUG ?= $(_KERNEL_DIRECTION_SLUG)
TURN           ?= 0
CONFIG_SLUG    ?= $(if $(CONFIG),$(CONFIG),$(_KERNEL_CONFIG_SLUG))
CONFIG_JSON    ?= $(if $(CONFIG),$(shell $(VENV_PYTHON) -c "import json,sys;d=json.load(open('$(_CONFIG_FILE)'));print(json.dumps(d['$(CONFIG)']))" 2>/dev/null),$(_KERNEL_CONFIG_JSON))

# ---------------------------------------------------------------------------
# Host-specific settings — auto-detected from conf/hosts/default.yaml
# ---------------------------------------------------------------------------
# All build env vars come from the host config. Never hardcode them.
_HOST_CFG    := $(REPO_ROOT)/conf/hosts/default.yaml
_HOST_SCRIPT := import yaml,socket,sys;\
c=yaml.safe_load(open('$(_HOST_CFG)'));\
f=socket.getfqdn();s=socket.gethostname();\
h=[v for _,v in c.get('hosts',{}).items() if f.startswith(v.get('ssh_host','').rstrip('.')) or s in v.get('ssh_host','')];\
e=h[0].get('env',{}) if h else {};\
print(e.get(sys.argv[1],''))

override CUDA_HOME  := $(or $(shell $(VENV_PYTHON) -c "$(_HOST_SCRIPT)" cuda_home 2>/dev/null),/usr/local/cuda)
override LD_PRELOAD := $(shell $(VENV_PYTHON) -c "$(_HOST_SCRIPT)" ld_preload 2>/dev/null)
export CUDA_HOME
export LD_PRELOAD

# ---------------------------------------------------------------------------
# Input files — derived from KERNEL and host config
# ---------------------------------------------------------------------------
# ARCH: directory-style arch name (e.g. sm90) — from host config torch_cuda_arch
# COMPILE_ARCH: nvcc/ptxas -arch flag (e.g. sm_90a) — always accelerated variant
_HOST_ARCH    := $(shell $(VENV_PYTHON) -c "$(_HOST_SCRIPT)" torch_cuda_arch 2>/dev/null)
ARCH          ?= sm$(subst .,,$(or $(_HOST_ARCH),$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)))
COMPILE_ARCH  ?= $(subst sm,sm_,$(ARCH))a

# --- New data layout: data/{ref,gen,configs} ---
# ref/  = arch-agnostic library implementations (cublas.py, cudnn.py, cutedsl.py)
# gen/  = arch-specific generated code (cutedsl.py, cuda.cu)
# configs/ = arch-agnostic benchmark configs (matmul.json, fa4.json)
#
# IMPL selects which implementation to evaluate:
#   IMPL=ref/cublas       → data/ref/$(KERNEL)/cublas.py
#   IMPL=gen/cutedsl      → data/gen/$(ARCH)/$(KERNEL)/cutedsl.py
#   IMPL=gen/cuda         → data/gen/$(ARCH)/$(KERNEL)/cuda.cu
#
# Harness auto-detects by extension: .py → Python harness, .cu → C harness
REF_DIR       := $(REPO_ROOT)/data/ref/$(KERNEL)
GEN_DIR       := $(REPO_ROOT)/data/gen/$(ARCH)/$(KERNEL)

# Default: first .py in ref/ as reference, first .cu in gen/ as generated
REFERENCE_PY  ?= $(firstword $(wildcard $(REF_DIR)/*.py))
CUDNN_PY      ?= $(wildcard $(REF_DIR)/cublas.py $(REF_DIR)/cudnn.py)
GENERATED_CU  ?= $(wildcard $(GEN_DIR)/cuda.cu)

# Backward compat aliases
FIXTURE_DIR   ?= $(REF_DIR)

# Pass COMPILE_ARCH to compile.sh
export PTXAS_ARCH ?= $(COMPILE_ARCH)

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
TIMEOUT                ?= 300
SEED                   ?= 42
NUM_WARMUPS            ?= 5
NUM_PERF_TRIALS        ?= 10
NUM_CORRECTNESS_TRIALS ?= 3

# ---------------------------------------------------------------------------
# Execute target parameters
# ---------------------------------------------------------------------------
EXEC_CMD ?= /usr/local/cuda/bin/nvcc --version

# ---------------------------------------------------------------------------
# Derived paths — the standard turn-root layout
# ---------------------------------------------------------------------------
TURN_ROOT  = $(CUDA_EXEC_ROOT)/$(RUN_TAG)/$(VERSION)/$(DIRECTION_ID)_$(DIRECTION_SLUG)/turn_$(TURN)
WORKSPACE  = $(TURN_ROOT)/workspace
ARTIFACTS  = $(TURN_ROOT)/artifacts
LOGS       = $(TURN_ROOT)/logs
STATE      = $(TURN_ROOT)/state

# ---------------------------------------------------------------------------
# Test defaults — derived from KERNEL
# ---------------------------------------------------------------------------
TEST_RUN_TAG = manual-cli-test-$(KERNEL)
TEST_TURN    = 0

# ---------------------------------------------------------------------------
# Common metadata CLI args
# ---------------------------------------------------------------------------
_META_ARGS = \
	--run-tag $(RUN_TAG) \
	--version $(VERSION) \
	--direction-id $(DIRECTION_ID) \
	--direction-slug $(DIRECTION_SLUG) \
	--turn $(TURN)

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
define _require_input_files
	$(if $(wildcard $(GEN_DIR)/*),,$(error No gen/ files found at $(GEN_DIR)))
endef

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: ensure-venv
ensure-venv: ## Create the Python venv with uv if it does not exist
	@if [ ! -f "$(VENV_PYTHON)" ]; then \
		echo "--- ensure-venv: creating venv at $(VENV_DIR) ---"; \
		uv venv "$(VENV_DIR)" --python 3.12; \
		uv pip install --python "$(VENV_PYTHON)" \
			-r "$(REPO_ROOT)/cuda_exec/requirements.txt"; \
		echo "--- ensure-venv: done ---"; \
	fi

.PHONY: help
help: ## Show available targets
	@echo "cuda_exec manual CLI"
	@echo ""
	@echo "KERNEL=$(KERNEL)  CONFIG=$(if $(CONFIG),$(CONFIG),<default>)"
	@echo "Turn root: \$$(CUDA_EXEC_ROOT)/\$$(RUN_TAG)/\$$(VERSION)/\$$(DIRECTION_ID)_\$$(DIRECTION_SLUG)/turn_\$$(TURN)/"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-z][a-z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  %-24s %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:"
	@echo "  make test                                    # smoke test vecadd (default)"
	@echo "  make test KERNEL=matmul                      # smoke test matmul"
	@echo "  make test KERNEL=matmul CONFIG=mat-1024x1024 # matmul with specific config"
	@echo "  make clean-test                              # clean up smoke test output"
	@echo ""
	@echo "CONFIG= picks a slug from conf/fixtures/\$$(KERNEL)/configs.json."
	@echo "CONFIG=all runs evaluate over every config in configs.json."
	@echo "If omitted, uses the per-kernel default."

.PHONY: show
show: ## Print resolved turn-root path and key variables
	@echo "KERNEL         : $(KERNEL)"
	@echo "ARCH           : $(ARCH)"
	@echo "COMPILE_ARCH   : $(COMPILE_ARCH)"
	@echo "CUDA_HOME      : $(CUDA_HOME)"
	@echo "LD_PRELOAD     : $(LD_PRELOAD)"
	@echo "CUDA_EXEC_ROOT : $(CUDA_EXEC_ROOT)"
	@echo "RUN_TAG        : $(RUN_TAG)"
	@echo "VERSION        : $(VERSION)"
	@echo "DIRECTION_ID   : $(DIRECTION_ID)"
	@echo "DIRECTION_SLUG : $(DIRECTION_SLUG)"
	@echo "TURN           : $(TURN)"
	@echo "TURN_ROOT      : $(TURN_ROOT)"
	@echo "CONFIG_SLUG    : $(CONFIG_SLUG)"
	@echo "CONFIG_JSON    : $(CONFIG_JSON)"
	@echo "REF_DIR        : $(REF_DIR)"
	@echo "GEN_DIR        : $(GEN_DIR)"
	@echo "REFERENCE_PY   : $(REFERENCE_PY)"
	@echo "GENERATED_CU   : $(GENERATED_CU)"
	@echo ""
	@if [ -d "$(TURN_ROOT)" ]; then \
		echo "Turn root exists. Contents:"; \
		find "$(TURN_ROOT)" -type f | sort | head -40; \
	else \
		echo "Turn root does not exist yet."; \
	fi

.PHONY: clean
clean: ## Remove the entire turn root for current metadata
	@if [ -d "$(TURN_ROOT)" ]; then \
		echo "Removing $(TURN_ROOT)"; \
		rm -rf "$(TURN_ROOT)"; \
	else \
		echo "Nothing to clean: $(TURN_ROOT) does not exist"; \
	fi

.PHONY: compile
compile: ensure-venv ## Stage reference + generated files and run the 5-step compile pipeline
	$(call _require_input_files)
	@set -euo pipefail
	# Create turn-root directories
	mkdir -p "$(WORKSPACE)/inputs/reference" "$(WORKSPACE)/inputs/generated" "$(WORKSPACE)/inputs/cudnn" "$(ARTIFACTS)" "$(LOGS)" "$(STATE)"
	# Stage ref/ implementations:
	#   cutedsl.py → inputs/reference/ (CuTe DSL library reference)
	#   cublas.py, cudnn.py → inputs/cudnn/ (vendor baseline)
	for f in $(REF_DIR)/*.py; do \
		[ ! -f "$$f" ] && continue; \
		case "$$(basename $$f)" in \
			cutedsl.py) cp "$$f" "$(WORKSPACE)/inputs/reference/" ;; \
			*)          cp "$$f" "$(WORKSPACE)/inputs/cudnn/" ;; \
		esac; \
	done
	# Stage gen/ Python implementations (cutedsl.py + helpers) → inputs/reference/
	# If ref/ already provided cutedsl.py, gen/ cutedsl.py overwrites it
	for f in $(GEN_DIR)/*.py; do \
		[ -f "$$f" ] && cp "$$f" "$(WORKSPACE)/inputs/reference/" 2>/dev/null || true; \
	done
	# Stage gen/ CUDA implementation → inputs/generated/
	if [ -f "$(GEN_DIR)/cuda.cu" ]; then \
		cp "$(GEN_DIR)/cuda.cu" "$(WORKSPACE)/inputs/generated/generated.cu"; \
	fi
	# Run the 5-step compile pipeline
	echo "--- compile [$(KERNEL)]: staging done, running compile.sh ---"
	HARNESS_FLAG="" && \
	if grep -q 'kernel_run' "$(WORKSPACE)/inputs/generated/generated.cu" 2>/dev/null; then \
		HARNESS_FLAG="--harness $(EVAL_HARNESS)"; \
	fi && \
	CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
	bash "$(COMPILE_SH)" \
		--source "$(WORKSPACE)/inputs/generated/generated.cu" \
		--output "$(ARTIFACTS)/compile.attempt_001.generated.bin" \
		--arch "$(COMPILE_ARCH)" \
		$$HARNESS_FLAG
	# Write the compile manifest so evaluate/profile can find the binary
	cat > "$(STATE)/compile.attempt_001.json" <<'MANIFEST_EOF'
	{
	  "stage": "compile",
	  "attempt": 1,
	  "status": "ok",
	  "selected_source": "inputs/generated/generated.cu",
	  "primary_artifact_id": "compile:primary_binary",
	  "artifacts": [
	    {"artifact_id": "compile:primary_binary", "kind": "binary", "path": "artifacts/compile.attempt_001.generated.bin"},
	    {"artifact_id": "compile:primary_ptx", "kind": "ptx", "path": "artifacts/compile.attempt_001.generated.ptx"},
	    {"artifact_id": "compile:primary_cubin", "kind": "cubin", "path": "artifacts/compile.attempt_001.generated.cubin"},
	    {"artifact_id": "compile:resource_usage", "kind": "report", "path": "artifacts/compile.attempt_001.generated.resource-usage.txt"},
	    {"artifact_id": "compile:sass_nvdisasm", "kind": "sass", "path": "artifacts/compile.attempt_001.generated.nvdisasm.sass"}
	  ]
	}
	MANIFEST_EOF
	echo "--- compile [$(KERNEL)]: done ---"
	echo "Turn root: $(TURN_ROOT)"
	@ls -lh "$(ARTIFACTS)/"

.PHONY: evaluate
evaluate: ensure-venv ## Run evaluate for one config (CONFIG=slug) or all (CONFIG=all)
	@export LD_LIBRARY_PATH=$(VENV_DIR)/lib/python3.12/site-packages/nvidia/cublas/lib:$(VENV_DIR)/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64:$$LD_LIBRARY_PATH
	@set -euo pipefail
	if [ "$(CONFIG_SLUG)" = "all" ]; then
		SLUGS=$$($(VENV_PYTHON) -c "import json; print(' '.join(json.load(open('$(_CONFIG_FILE)')).keys()))")
		for slug in $$SLUGS; do
			echo ""
			$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) evaluate \
				KERNEL="$(KERNEL)" \
				CONFIG="$$slug" \
				CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
				RUN_TAG="$(RUN_TAG)" \
				VERSION="$(VERSION)" \
				TURN="$(TURN)" \
				TIMEOUT="$(TIMEOUT)" \
				SEED="$(SEED)" \
				NUM_WARMUPS="$(NUM_WARMUPS)" \
				NUM_PERF_TRIALS="$(NUM_PERF_TRIALS)" \
				NUM_CORRECTNESS_TRIALS="$(NUM_CORRECTNESS_TRIALS)"
		done
	else
		cd "$(SCRIPTS)" && \
		FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		$(VENV_PYTHON) "$(EVALUATE_PY)" \
			$(_META_ARGS) \
			--config-slug "$(CONFIG_SLUG)" \
			--config-json '$(CONFIG_JSON)' \
			--timeout $(TIMEOUT) \
			--seed $(SEED) \
			--num-warmups $(NUM_WARMUPS) \
			--num-perf-trials $(NUM_PERF_TRIALS) \
			--num-correctness-trials $(NUM_CORRECTNESS_TRIALS) \
		| $(VENV_PYTHON) "$(FMT_EVAL_PY)"
	fi

# ---------------------------------------------------------------------------
# Profile — Nsight Compute hardware-level profiling (requires sudo)
# ---------------------------------------------------------------------------
NCU_KERNEL_FILTER ?= $(_KERNEL_NCU_FILTER)
_NCU_CONFIG_ENV = \
	CUDA_EXEC_CONFIG_ID="$(CONFIG_SLUG)" \
	CUDA_EXEC_PARAM_INPUT_SIZE="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.load(sys.stdin)["input_size"])')" \
	CUDA_EXEC_PARAM_RANK="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.load(sys.stdin)["rank"])')" \
	CUDA_EXEC_PARAM_SHAPE_KIND="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.load(sys.stdin)["shape_kind"])')" \
	CUDA_EXEC_PARAM_SHAPE="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.dumps(json.load(sys.stdin)["shape"]))')"

.PHONY: profile-ncu-generated
profile-ncu-generated: ensure-venv ## NCU profile the generated CUDA binary for one config
	@set -euo pipefail
	BINARY="$(ARTIFACTS)/compile.attempt_001.generated.bin"
	if [[ ! -x "$$BINARY" ]]; then
		echo "ERROR: binary not found at $$BINARY — run compile first" >&2
		exit 1
	fi
	NCU_PREFIX="$(ARTIFACTS)/ncu.$(CONFIG_SLUG)"
	echo "--- profile-ncu-generated [$(KERNEL)]: $(CONFIG_SLUG) ---"
	$(_NCU_CONFIG_ENV) \
	CUDA_EXEC_CONFIG_JSON='$(CONFIG_JSON)' \
	CUDA_EXEC_NUM_WARMUPS=2 \
	CUDA_EXEC_NUM_TRIALS=1 \
	sudo --preserve-env bash "$(PROFILE_SH)" \
		--target "$$BINARY" \
		--export-prefix "$$NCU_PREFIX" \
		--set detailed \
		--kernel-name $(NCU_KERNEL_FILTER) \
		> "$(LOGS)/ncu.$(CONFIG_SLUG).stdout.log"
	echo "--- profile-ncu-generated [$(KERNEL)]: done ---"
	echo "NCU report: $$NCU_PREFIX.ncu-rep"
	echo ""
	/usr/local/cuda/bin/ncu --import "$$NCU_PREFIX.ncu-rep" --print-details all

.PHONY: profile-ncu-reference
profile-ncu-reference: ensure-venv ## NCU profile the reference Python/CuTe DSL kernel for one config
	@set -euo pipefail
	REFERENCE_PY_FILE="$(WORKSPACE)/inputs/reference/reference.py"
	if [[ ! -f "$$REFERENCE_PY_FILE" ]]; then
		echo "ERROR: reference.py not found at $$REFERENCE_PY_FILE — run compile first (it stages inputs)" >&2
		exit 1
	fi
	NCU_PREFIX="$(ARTIFACTS)/ncu-reference.$(CONFIG_SLUG)"
	echo "--- profile-ncu-reference [$(KERNEL)]: $(CONFIG_SLUG) ---"
	$(_NCU_CONFIG_ENV) \
	CUDA_EXEC_CONFIG_JSON='{"slug":"$(CONFIG_SLUG)","params":$(CONFIG_JSON)}' \
	CUDA_EXEC_NUM_WARMUPS=2 \
	CUDA_EXEC_NUM_TRIALS=1 \
	sudo --preserve-env bash "$(PROFILE_SH)" \
		--target $(VENV_PYTHON) "$(PROFILE_REF)" "$$REFERENCE_PY_FILE" \
		--export-prefix "$$NCU_PREFIX" \
		--set detailed \
		--kernel-name $(NCU_KERNEL_FILTER) \
		> "$(LOGS)/ncu-reference.$(CONFIG_SLUG).stdout.log"
	echo "--- profile-ncu-reference [$(KERNEL)]: done ---"
	echo "NCU report: $$NCU_PREFIX.ncu-rep"
	echo ""
	/usr/local/cuda/bin/ncu --import "$$NCU_PREFIX.ncu-rep" --print-details all

.PHONY: execute
execute: ensure-venv ## Run a generic command in the workspace (override EXEC_CMD="cmd arg1 arg2")
	@set -euo pipefail
	mkdir -p "$(WORKSPACE)" "$(LOGS)"
	echo "--- execute: running in $(WORKSPACE) ---"
	cd "$(WORKSPACE)" && \
	CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
	$(EXEC_CMD) 2>&1 | tee "$(LOGS)/execute.manual.log"
	echo "--- execute: done ---"
	echo "Log: $(LOGS)/execute.manual.log"

# ---------------------------------------------------------------------------
# Smoke test — full pipeline (compile → evaluate → profile)
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run full pipeline smoke test (default: KERNEL=vecadd, or KERNEL=matmul)
	@set -euo pipefail
	echo "=== cuda_exec smoke test: KERNEL=$(KERNEL) ==="
	echo ""
	echo "--- [1/4] clean ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) clean \
		KERNEL="$(KERNEL)" \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		TURN="$(TEST_TURN)"
	echo ""
	echo "--- [2/4] compile ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) compile \
		KERNEL="$(KERNEL)" \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		TURN="$(TEST_TURN)"
	echo ""
	echo "--- [3/4] evaluate ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) evaluate \
		KERNEL="$(KERNEL)" \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		TURN="$(TEST_TURN)"
	echo ""
	echo "--- [4/4] profile-ncu-generated ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) profile-ncu-generated \
		KERNEL="$(KERNEL)" \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		TURN="$(TEST_TURN)"
	echo ""
	echo "=== smoke test [$(KERNEL)] passed ==="

.PHONY: clean-test
clean-test: ## Clean the smoke test turn root
	@$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) clean \
		KERNEL="$(KERNEL)" \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		TURN="$(TEST_TURN)"
