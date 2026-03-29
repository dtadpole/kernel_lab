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
# Quick smoke test with bundled vector-add fixtures:
#   make test
#
# Manual single-config run (requires REFERENCE_PY and GENERATED_CU):
#   make compile evaluate \
#     REFERENCE_PY=/path/to/reference.py GENERATED_CU=/path/to/generated.cu \
#     DIRECTION_ID=7 DIRECTION_SLUG=my-kernel CONFIG_SLUG=shape-1d-1048576 \
#     CONFIG_JSON='{"shape":[1048576],"rank":1,"input_size":1048576,"shape_kind":"1d"}'
#
# Full pipeline:
#   make clean compile evaluate profile-reference profile-generated

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT     := $(shell git -C $(dir $(abspath $(lastword $(MAKEFILE_LIST)))) rev-parse --show-toplevel)
SCRIPTS       := $(REPO_ROOT)/cuda_exec/scripts
TESTS         := $(REPO_ROOT)/cuda_exec/tests
VENV_PYTHON   := $(REPO_ROOT)/cuda_exec/.venv/bin/python
COMPILE_SH    := $(SCRIPTS)/compile.sh
EVALUATE_PY   := $(SCRIPTS)/evaluate.py
PROFILE_PY    := $(SCRIPTS)/profile.py
PROFILE_SH    := $(SCRIPTS)/profile.sh
FMT_EVAL_PY   := $(SCRIPTS)/fmt_eval.py
EVAL_HARNESS  := $(SCRIPTS)/eval_harness.cu

# ---------------------------------------------------------------------------
# Metadata — override on command line as needed
# ---------------------------------------------------------------------------
CUDA_EXEC_ROOT ?= $(HOME)/.cuda_exec
RUN_TAG        ?= manual-cli
VERSION        ?= v1
DIRECTION_ID   ?= 1
DIRECTION_SLUG ?= vector-add
TURN           ?= 0
CONFIG_SLUG    ?= vec1d-n65536

# ---------------------------------------------------------------------------
# Input files — default to vector-add fixtures
# ---------------------------------------------------------------------------
REFERENCE_PY  ?= $(REPO_ROOT)/conf/fixtures/reference/reference.py
GENERATED_CU  ?= $(REPO_ROOT)/conf/fixtures/generated/generated.cu

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
CONFIG_JSON            ?= {"family":"integration-vector-add","rank":1,"shape":[65536],"input_size":65536,"shape_kind":"1d"}
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
# Test fixture defaults (vector-add)
# ---------------------------------------------------------------------------
TEST_FIXTURES     = $(REPO_ROOT)/conf/fixtures
TEST_REFERENCE_PY = $(TEST_FIXTURES)/reference/reference.py
TEST_GENERATED_CU = $(TEST_FIXTURES)/generated/generated.cu
TEST_RUN_TAG      = manual-cli-test
TEST_DIRECTION_ID = 1
TEST_DIRECTION_SLUG = vector-add
TEST_TURN         = 0
TEST_CONFIG_SLUG  = vec1d-n65536
TEST_CONFIG_JSON  = {"family":"integration-vector-add","rank":1,"shape":[65536],"input_size":65536,"shape_kind":"1d"}

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
	$(if $(REFERENCE_PY),,$(error REFERENCE_PY is required — path to reference.py))
	$(if $(GENERATED_CU),,$(error GENERATED_CU is required — path to generated.cu))
	$(if $(wildcard $(REFERENCE_PY)),,$(error REFERENCE_PY not found: $(REFERENCE_PY)))
	$(if $(wildcard $(GENERATED_CU)),,$(error GENERATED_CU not found: $(GENERATED_CU)))
endef

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show available targets
	@echo "cuda_exec manual CLI"
	@echo ""
	@echo "Turn root: \$$(CUDA_EXEC_ROOT)/\$$(RUN_TAG)/\$$(VERSION)/\$$(DIRECTION_ID)_\$$(DIRECTION_SLUG)/turn_\$$(TURN)/"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-z][a-z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  %-24s %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:"
	@echo "  make test                              # smoke test with vector-add fixtures"
	@echo "  make clean-test                        # clean up smoke test output"
	@echo ""
	@echo "Custom kernel:"
	@echo "  make compile REFERENCE_PY=ref.py GENERATED_CU=gen.cu DIRECTION_SLUG=my-kernel"
	@echo "  make evaluate CONFIG_SLUG=cfg1 CONFIG_JSON='{...}' DIRECTION_SLUG=my-kernel"

.PHONY: show
show: ## Print resolved turn-root path and key variables
	@echo "CUDA_EXEC_ROOT : $(CUDA_EXEC_ROOT)"
	@echo "RUN_TAG        : $(RUN_TAG)"
	@echo "VERSION        : $(VERSION)"
	@echo "DIRECTION_ID   : $(DIRECTION_ID)"
	@echo "DIRECTION_SLUG : $(DIRECTION_SLUG)"
	@echo "TURN           : $(TURN)"
	@echo "TURN_ROOT      : $(TURN_ROOT)"
	@echo "CONFIG_SLUG    : $(CONFIG_SLUG)"
	@echo "CONFIG_JSON    : $(CONFIG_JSON)"
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
compile: ## Stage reference + generated files and run the 5-step compile pipeline
	$(call _require_input_files)
	@set -euo pipefail
	# Create turn-root directories
	mkdir -p "$(WORKSPACE)/inputs/reference" "$(WORKSPACE)/inputs/generated" "$(ARTIFACTS)" "$(LOGS)" "$(STATE)"
	# Stage input files
	cp "$(REFERENCE_PY)" "$(WORKSPACE)/inputs/reference/reference.py"
	cp "$(GENERATED_CU)" "$(WORKSPACE)/inputs/generated/generated.cu"
	# Run the 5-step compile pipeline
	echo "--- compile: staging done, running compile.sh ---"
	HARNESS_FLAG="" && \
	if grep -q 'kernel_run' "$(WORKSPACE)/inputs/generated/generated.cu" 2>/dev/null; then \
		HARNESS_FLAG="--harness $(EVAL_HARNESS)"; \
	fi && \
	CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
	bash "$(COMPILE_SH)" \
		--source "$(WORKSPACE)/inputs/generated/generated.cu" \
		--output "$(ARTIFACTS)/compile.attempt_001.generated.bin" \
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
	echo "--- compile: done ---"
	echo "Turn root: $(TURN_ROOT)"
	@ls -lh "$(ARTIFACTS)/"

.PHONY: evaluate
evaluate: ## Run evaluate (reference vs generated comparison) for one config
	@set -euo pipefail
	cd "$(SCRIPTS)" && \
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

.PHONY: profile-reference
profile-reference: ## Profile reference side only for one config
	@set -euo pipefail
	cd "$(SCRIPTS)" && \
	CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
	$(VENV_PYTHON) "$(PROFILE_PY)" \
		$(_META_ARGS) \
		--config-slug "$(CONFIG_SLUG)" \
		--config-json '$(CONFIG_JSON)' \
		--mode reference_only \
		--timeout $(TIMEOUT)

.PHONY: profile-generated
profile-generated: ## Profile generated side only for one config
	@set -euo pipefail
	cd "$(SCRIPTS)" && \
	CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
	$(VENV_PYTHON) "$(PROFILE_PY)" \
		$(_META_ARGS) \
		--config-slug "$(CONFIG_SLUG)" \
		--config-json '$(CONFIG_JSON)' \
		--mode generated_only \
		--timeout $(TIMEOUT)

.PHONY: profile-dual
profile-dual: ## Profile both sides with speedup comparison for one config
	@set -euo pipefail
	cd "$(SCRIPTS)" && \
	CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
	$(VENV_PYTHON) "$(PROFILE_PY)" \
		$(_META_ARGS) \
		--config-slug "$(CONFIG_SLUG)" \
		--config-json '$(CONFIG_JSON)' \
		--mode dual \
		--timeout $(TIMEOUT)

.PHONY: profile-ncu
profile-ncu: ## Profile generated binary with NCU (Nsight Compute) for one config
	@set -euo pipefail
	BINARY="$(ARTIFACTS)/compile.attempt_001.generated.bin"
	if [[ ! -x "$$BINARY" ]]; then
		echo "ERROR: binary not found at $$BINARY — run compile first" >&2
		exit 1
	fi
	NCU_PREFIX="$(ARTIFACTS)/ncu.$(CONFIG_SLUG)"
	echo "--- profile-ncu: $(CONFIG_SLUG) ---"
	CUDA_EXEC_CONFIG_ID="$(CONFIG_SLUG)" \
	CUDA_EXEC_CONFIG_JSON='$(CONFIG_JSON)' \
	CUDA_EXEC_PARAM_INPUT_SIZE="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.load(sys.stdin)["input_size"])')" \
	CUDA_EXEC_PARAM_RANK="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.load(sys.stdin)["rank"])')" \
	CUDA_EXEC_PARAM_SHAPE_KIND="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.load(sys.stdin)["shape_kind"])')" \
	CUDA_EXEC_PARAM_SHAPE="$(shell echo '$(CONFIG_JSON)' | $(VENV_PYTHON) -c 'import json,sys;print(json.dumps(json.load(sys.stdin)["shape"]))')" \
	CUDA_EXEC_NUM_WARMUPS=2 \
	CUDA_EXEC_NUM_TRIALS=1 \
	sudo --preserve-env bash "$(PROFILE_SH)" \
		--target "$$BINARY" \
		--export-prefix "$$NCU_PREFIX" \
		--set detailed \
		> "$(LOGS)/ncu.$(CONFIG_SLUG).stdout.log"
	echo "--- profile-ncu: done ---"
	echo "NCU report: $$NCU_PREFIX.ncu-rep"
	echo ""
	/usr/local/cuda/bin/ncu --import "$$NCU_PREFIX.ncu-rep" --print-details all

.PHONY: execute
execute: ## Run a generic command in the workspace (override EXEC_CMD="cmd arg1 arg2")
	@set -euo pipefail
	mkdir -p "$(WORKSPACE)" "$(LOGS)"
	echo "--- execute: running in $(WORKSPACE) ---"
	cd "$(WORKSPACE)" && \
	CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
	$(EXEC_CMD) 2>&1 | tee "$(LOGS)/execute.manual.log"
	echo "--- execute: done ---"
	echo "Log: $(LOGS)/execute.manual.log"

# ---------------------------------------------------------------------------
# Smoke test — full pipeline with vector-add fixtures
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run full pipeline smoke test with vector-add fixtures (compile → evaluate → profile)
	@set -euo pipefail
	echo "=== cuda_exec manual CLI smoke test ==="
	echo ""
	echo "--- [1/5] clean ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) clean \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		DIRECTION_ID="$(TEST_DIRECTION_ID)" \
		DIRECTION_SLUG="$(TEST_DIRECTION_SLUG)" \
		TURN="$(TEST_TURN)"
	echo ""
	echo "--- [2/5] compile ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) compile \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		DIRECTION_ID="$(TEST_DIRECTION_ID)" \
		DIRECTION_SLUG="$(TEST_DIRECTION_SLUG)" \
		TURN="$(TEST_TURN)" \
		REFERENCE_PY="$(TEST_REFERENCE_PY)" \
		GENERATED_CU="$(TEST_GENERATED_CU)"
	echo ""
	echo "--- [3/5] evaluate ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) evaluate \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		DIRECTION_ID="$(TEST_DIRECTION_ID)" \
		DIRECTION_SLUG="$(TEST_DIRECTION_SLUG)" \
		TURN="$(TEST_TURN)" \
		CONFIG_SLUG="$(TEST_CONFIG_SLUG)" \
		CONFIG_JSON='$(TEST_CONFIG_JSON)'
	echo ""
	echo "--- [4/5] profile-reference ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) profile-reference \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		DIRECTION_ID="$(TEST_DIRECTION_ID)" \
		DIRECTION_SLUG="$(TEST_DIRECTION_SLUG)" \
		TURN="$(TEST_TURN)" \
		CONFIG_SLUG="$(TEST_CONFIG_SLUG)" \
		CONFIG_JSON='$(TEST_CONFIG_JSON)'
	echo ""
	echo "--- [5/5] profile-generated ---"
	$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) profile-generated \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		DIRECTION_ID="$(TEST_DIRECTION_ID)" \
		DIRECTION_SLUG="$(TEST_DIRECTION_SLUG)" \
		TURN="$(TEST_TURN)" \
		CONFIG_SLUG="$(TEST_CONFIG_SLUG)" \
		CONFIG_JSON='$(TEST_CONFIG_JSON)'
	echo ""
	echo "=== smoke test passed ==="
	echo "Turn root: $(CUDA_EXEC_ROOT)/$(TEST_RUN_TAG)/$(VERSION)/$(TEST_DIRECTION_ID)_$(TEST_DIRECTION_SLUG)/turn_$(TEST_TURN)/"

.PHONY: clean-test
clean-test: ## Clean the smoke test turn root
	@$(MAKE) -f $(abspath $(lastword $(MAKEFILE_LIST))) clean \
		CUDA_EXEC_ROOT="$(CUDA_EXEC_ROOT)" \
		RUN_TAG="$(TEST_RUN_TAG)" \
		VERSION="$(VERSION)" \
		DIRECTION_ID="$(TEST_DIRECTION_ID)" \
		DIRECTION_SLUG="$(TEST_DIRECTION_SLUG)" \
		TURN="$(TEST_TURN)"
