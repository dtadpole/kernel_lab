# kernel_lab

A repository for kernel optimization experiments and related tooling.

## System design

See `docs/SYSTEM_DESIGN.md` for the multi-agent architecture (Supervisor / Solver / Harness / Evaluator / Reflector) and the two-repo split (kernel-lab + kernel-lab-kb).

## Current components

- `cuda_exec/` — CUDA kernel compile/evaluate/profile engine (FastAPI server + direct Python API)
- `doc_retrieval/` — NVIDIA CUDA Toolkit document retrieval system (BM25 + dense search)
- `plugins/ik/` — Unified Claude Code plugin (CLI-based skills, no MCP server)
  - Skills: `exec` (compile/evaluate/profile), `inspect` (review results), `docs` (search NVIDIA docs), `index` (manage search index), `optimize` (autonomous optimization loop)
- `plugins/deprecated/` — Archived plugins (`cuda/`, `kb/`) replaced by `ik`

## Repo-level conventions

### 1. Python environment

- All components share a single `uv`-managed virtual environment at `.venv`
- Dependencies are defined in the root `pyproject.toml`
- Setup: `uv venv .venv --python 3.12 && uv pip install -e "."`
- Plugins use `.venv/bin/python` to run their MCP servers

### 2. Metadata is mandatory

For command-style API requests/responses, `metadata` is required.
Required fields:

- `run_tag`
- `version`
- `direction_id`
- `direction_slug`
- `turn`

### 3. `cuda_exec` workflow is convention-driven

High-level rules:

- compile first
- compile once per turn
- evaluate/profile depend on compile state from the same turn
- evaluate/profile are config-level, not code-level
- new files require a new turn
- old turns are immutable

### 4. Settled `cuda_exec` interface conventions

These are stable project-level decisions and should remain in repo docs, not only in assistant memory.

#### Runtime mental model

- `workspace = inputs + scratch`
- `artifacts = kept results`
- `logs = process output`
- `state = workflow record`

#### Turn-root layout

`cuda_exec` resolves runtime locations by convention from request metadata. It does not accept a caller-specified working directory.

Runtime root:

```text
~/.cuda_exec/<run_tag>/<version>/<direction_id>_<direction_slug>/turn_<turn>/
```

Within each turn:

```text
turn_<turn>/
  workspace/
  artifacts/
  logs/
  state/
```

#### Compile inputs

`compile` takes inline file maps, not file lists:

- `reference_files: Dict[relative_path, content]`
- `generated_files: Dict[relative_path, content]`

All public request/response file names should use relative paths.

#### Stage model

- `compile` is code-level
- `evaluate` / `profile` are runtime-config-level
- one compile may fan out into many configs
- runtime configs are passed as `configs: Dict[config_slug, Dict[str, Any]]`
- the same config slug is the stable identity on both request and response
- the config body is intentionally kernel-specific and flexible; do not overfit it to one workload family
- reference Python code for `evaluate` / `profile` should follow the explicit module contract: `Model(torch.nn.Module)`, `get_init_inputs()`, `get_inputs(config)`
- `cuda_exec` intentionally borrows that kbEval-style reference contract even though the generated side still runs through the compiled artifact path rather than a Python `ModelNew` class

#### Public response boundary

Default public responses should stay small and only expose stage-relevant artifacts/logs.
Internal workflow state is kept for compile/evaluate/profile bookkeeping but is not part of the default public response.
Public request/response file names use relative paths, and public returned files are shaped as relative-path keyed dictionaries.
For evaluate/profile specifically, public responses mirror request shape and use `configs: Dict[config_slug, ...]` instead of result lists.
Top-level public responses use `all_ok` for aggregate success. Per-config outputs use `status` plus structured summaries instead of raw log-only results.

- evaluate config output: `status` + `reference` + `generated` + `correctness` + `performance` + `artifacts` + `logs`
- profile config output: `status` + `summary` + `artifacts` + `logs`
- profile uses Nsight Compute exclusively; callers specify `side: "generated" | "reference"`

#### Execute boundary

`execute` is logs-only from the public API perspective:

- keep `logs/execute.attempt_###.*`
- do not expose execute state in the public response
- let the caller decide what execute outputs matter

### 5. `cuda_exec` documentation split

- `cuda_exec/DESIGN.md` is the source of truth for detailed design
- `cuda_exec/README.md` stays short
- this `AGENTS.md` stays at repo-level only
- `cuda_exec/models.py` documents the public request/response contract
- `cuda_exec/runner.py` documents runtime-layout semantics
- `cuda_exec/main.py` stays thin and keeps only lightweight endpoint/helper docstrings

### 6. `cuda_exec/tests` is integration-only

- `cuda_exec/tests/` is reserved for end-to-end integration tests
- do not put unit tests there
- tests should start a real uvicorn service in a subprocess and call HTTP interfaces with realistic payloads
- tests should isolate runtime side effects via a temporary `CUDA_EXEC_ROOT`
- prefer placing temporary test roots under `~/temp/`
- create one top-level temp directory per integration-suite invocation rather than one sibling temp directory per test class or service lifecycle
- prefix the run directory name with `YYYY-MM-DD-HH-MM-`
- then use a kebab-case slug plus PID in the subfolder name
- if multiple service processes are started during the suite, reuse that same top-level run directory and namespace per-service logs/runtime roots inside it
- preferred isolation direction: provision the uvicorn Python environment from `cuda_exec/requirements.txt` using `uv`, with the environment itself created under a temporary folder for the test run
- when using a temp-folder uv-managed environment, prefer naming it `<temp-run-dir>/.venv`
- preserve run environments and intermediate outputs by default for later inspection; do not rely on immediate deletion after each run
- cleanup should happen via a separate retention process (for example pruning runs older than 7 days)
- the standard helper is `cuda_exec/scripts/prune_temp_runs.py`
- its default behavior is to delete preserved run directories older than 7 days; support `--dry-run`; and skip directories marked for keep
- integration test runs should invoke this helper before starting the temporary uvicorn service
- current tests may still use the repo-local `.venv`, but the temp-folder `uv`-managed `.venv` is the preferred future-tightening path
- expected lower-level CUDA failures are allowed during early integration coverage, as long as the interface behavior itself is exercised
- current integration config coverage should include roughly 4–6 configs spanning multiple 1D sizes plus representative 2D and 3D shape metadata
- prefer storing integration config sets in fixture files under `data/fixtures/` instead of embedding them directly in the main test module
- fixture config slugs should make semantic sense for the sample workload; for vector-add fixtures, prefer size/shape/rank-based slugs rather than unrelated causal/noncausal labels
- for vector-add integration fixtures, the config body itself should stay pertinent: shape/rank/input_size metadata is enough, and unrelated transformer-style fields should be omitted

### 7. `cuda_agent` conventions

- `cuda_agent` does not import from `cuda_exec` — it communicates via HTTP through MCP servers
- `cuda_agent` requires `cuda_exec` to be running separately
- `cuda_agent` requires `ANTHROPIC_API_KEY` in the environment
- `cuda_agent` reads the bearer token from the same key file as `cuda_exec` (`~/.keys/cuda_exec.key` or `CUDA_EXEC_KEY_PATH`)
- `cuda_agent` loads two plugin MCP servers: `kb` (knowledge search) and `cuda` (toolkit execution)
- the agent (`agent.py`) uses `claude-agent-sdk` to run a single long optimization session
- the agent manages its own iteration loop internally — Claude decides when to compile, evaluate, modify, and converge

### 8. Service deployment

#### Infrastructure

- Host inventory and service-to-host mapping live in `conf/hosts/default.yaml` (single source of truth)
- Deploy CLIs: `plugins/deprecated/cuda/deploy/cli.py` (cuda_exec), `plugins/deprecated/kb/deploy/cli.py` (kb_embed)
- Deploy CLIs require PyYAML — run with `.venv/bin/python`, not system `python3` (Meta devvms lack PyYAML and block pip)

#### Host-specific constraints

| Host | Internet | Notes |
|------|----------|-------|
| _one, _two | yes | SSH aliases in personal `~/.ssh/config` only — not resolvable from devvms |
| h8_3 (devvm8490) | yes | fwdproxy works for PyPI and HuggingFace |
| h8_4 (devvm8491) | **no** | fwdproxy blocks CONNECT tunnels — `uv pip install` and HF model downloads fail |

#### Deploying to an online host

```bash
.venv/bin/python plugins/deprecated/cuda/deploy/cli.py deploy <host>
.venv/bin/python plugins/deprecated/cuda/deploy/cli.py start <host>
.venv/bin/python plugins/deprecated/kb/deploy/cli.py deploy <host>
.venv/bin/python plugins/deprecated/kb/deploy/cli.py start <host>
```

#### Deploying to an offline host (e.g., h8_4)

The deploy CLI assumes internet on the target. For offline hosts:

1. Run deploy CLI — code sync succeeds, dependency install fails at step 3:
   `.venv/bin/python plugins/deprecated/cuda/deploy/cli.py deploy h8_4`
2. Rsync venvs from an online host (e.g., h8_3):
   ```bash
   rsync -az ~/.cuda_exec_service/.venv/ devvm8491:~/.cuda_exec_service/.venv/
   rsync -az ~/.kb_embed_service/.venv/ devvm8491:~/.kb_embed_service/.venv/
   ```
3. Rsync HuggingFace model cache for kb_embed:
   ```bash
   rsync -az ~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/ \
     devvm8491:~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/
   ```
4. Add `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to the kb-embed systemd unit on the target
5. Manually install the systemd unit if the CLI didn't complete steps 4-5, then start

#### Pre-flight checks for shared devvms

Before deploying, verify the configured port is not occupied by another user:

```bash
sudo ss -tlnp | grep <port>
sudo lsof -i :<port>
```

If occupied, update the port in `conf/hosts/default.yaml` and redeploy.

Known conflict: port 46982 on h8_4 is used by another user — kb_embed moved to 46984.

#### Health endpoints

- cuda_exec: `GET /healthz` → `{"ok":true,"service":"cuda_exec"}`
- kb_embed: `GET /health` → `{"status":"ok"}`
- kb_embed functional: `POST /v1/embeddings` with `{"input":"test"}` → OpenAI-compatible response

#### SM arch vs SM arch-a (accelerated) compilation targets

NVIDIA GPUs report a base compute capability (e.g. 9.0), but the `nvcc`/`ptxas`
compilation target determines which instructions are available. The `-a` suffix
enables architecture-specific accelerator instructions:

| Target | Architecture | Key accelerated instructions enabled by `-a` |
|--------|-------------|----------------------------------------------|
| `sm_90` | Hopper (base) | Standard CUDA ISA for SM90 |
| `sm_90a` | Hopper (accel) | **WGMMA** (warpgroup MMA), **TMA** (tensor memory accelerator), **mbarrier** async, cluster launch |
| `sm_100` | Blackwell (base) | Standard CUDA ISA for SM100 |
| `sm_100a` | Blackwell (accel) | **TMEM** (tensor memory), **tcgen05** MMA, **TMA multicast**, **2-CTA clusters** |
| `sm_120` | Blackwell GeForce (base) | SM120 base ISA |
| `sm_120a` | Blackwell GeForce (accel) | Subset of SM100a — `mma.sync` but **no WGMMA/tcgen05/TMEM** |

**Rules:**
- All H100 hardware supports `sm_90a` — always compile with `-arch=sm_90a`
- All B200/B100 hardware supports `sm_100a`
- GeForce Blackwell (RTX 5090, RTX PRO 6000) is `sm_120a` — **not** `sm_100a`
- `torch.cuda.get_device_capability()` returns the base version (e.g. `(9, 0)`),
  not the `-a` variant. The `-a` is purely a compilation choice.
- CuTe DSL auto-detects and uses `sm_90a` on H100 (via `CUTE_DSL_ARCH` env var
  or GPU auto-detection). No manual override needed.

#### SM-architecture-specific code and environments

Generated kernels are architecture-specific: `data/generated/<arch>/matmul/generated.cu`.
Each arch folder contains kernels optimized for that SM version — **do not copy between arches**.

| Host | GPU | SM arch | `data/generated/` folder |
|------|-----|---------|--------------------------|
| h8_3, h8_4 | H100 | SM90 (Hopper) | `sm90/` |
| _one, _two | RTX PRO 6000 | SM120 (Blackwell) | `sm120/` |

**Key architecture differences:**
- SM90 (H100): optimal MMA is **WGMMA** (warpgroup, 4 warps, 64×N×K shapes). Per-warp `mma.sync m16n8k16` runs but achieves only ~40% of cuBLAS.
- SM120 (Blackwell GeForce): uses per-warp `mma.sync m16n8k16`. WGMMA not available on GeForce SM120.
- TMA (`cp.async.bulk.tensor`) works on both SM90+ and SM120.

**SM90 generated kernel (data/generated/sm90/matmul/generated.cu):**
- Uses CUTLASS 3.x C++ API (CollectiveBuilder) with `KernelTmaWarpSpecializedCooperative`
- Tile: 128×256×64 (big) / 128×128×64 (small), auto stage count, persistent scheduler
- Compile: `nvcc -arch=sm_90a -std=c++17 -O3 -I<cutlass>/include -I<cutlass>/tools/util/include --expt-relaxed-constexpr -lcuda`
- CUTLASS headers: `/home/zhenc/workspace1/third-party/cutlass/4.3.5/`
- Performance at 8192×8192: **668 TFLOPS (95% of cuBLAS 715)**
- Remaining 5% gap: ptxas `wgmma pipeline crossing function boundary` — CUTLASS template generates cross-function wgmma ops that nvcc can't fully pipeline. cuBLAS is NVIDIA-internal compiled with better instruction scheduling.

**CuTe DSL venv compatibility:**
- The service venv (`~/.cuda_exec_service/.venv`) has `cuda-python==13.2.0` → requires CUDA 13.x driver.
- h8_3 has driver 550.90.07 (CUDA 12.4) → CuTe DSL **fails** with `cudaErrorInsufficientDriver`.
- _one/_two have driver 595.45.04 (CUDA 13.2) → CuTe DSL works natively.

**FA4 CuTe DSL venv on h8_3/h8_4 (driver 550.90.07):**

A dedicated venv at `~/.fa4_venv` is required to run FA4 CuTe DSL on H100 hosts with CUDA 12.x drivers.

Setup (requires internet — run on h8_3, then rsync to h8_4):
```bash
# Create venv
python3.12 -m venv ~/.fa4_venv
# Install compatible packages (order matters)
~/.fa4_venv/bin/pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
~/.fa4_venv/bin/pip install cuda-bindings==12.8.0
~/.fa4_venv/bin/pip install nvidia-cutlass-dsl==4.4.2 --no-deps
~/.fa4_venv/bin/pip install flash-attn-4>=4.0.0b5
```

Key constraints:
- `cuda-bindings==12.8.0` (NOT `cuda-python==13.x` which requires CUDA 13 driver)
- `nvidia-cutlass-dsl==4.4.2 --no-deps` (prevents pulling cuda-python 13.x)
- `torch==2.6.0+cu124` (matches driver's CUDA 12.4)

Usage:
```bash
CUDA_VISIBLE_DEVICES=4 CUTE_DSL_ARCH=sm_90a ~/.fa4_venv/bin/python cutedsl.py
```

Verified: 2026-04-03, h8_3 (devvm8490), GPU 4, FA4 CuTe DSL 397–750 TFLOPS on SM90.

### 9. GPU allocation on multi-GPU hosts (h8_3, h8_4)

Each H100 host has 8 GPUs (0-7). Allocations are fixed to avoid contention:

| GPU(s) | Purpose | Set via |
|--------|---------|---------|
| **7** | `cuda_exec` service (long-running, systemd) | `CUDA_VISIBLE_DEVICES=7` in service unit |
| **6** | `kb_embed` service (embedding server, systemd) | `CUDA_VISIBLE_DEVICES=6` in service unit |
| **4, 5** | Local benchmarks / ad-hoc kernel runs | `CUDA_VISIBLE_DEVICES=4` or `5` |
| **0-3** | Free (other users / experiments) | — |

This applies to **both h8_3 and h8_4** identically. See `conf/hosts/default.yaml` for the canonical config.

**When running benchmarks locally**, always set `CUDA_VISIBLE_DEVICES=4` (or `5`).

**cuDNN requires `libnvrtc.so.12`** for JIT-compiling flash attention kernels.
On h8_3/h8_4, this library is at `/usr/local/cuda-12.8/lib64/` but not on the
default `LD_LIBRARY_PATH`. Without it, cuDNN SDPA silently falls back to a
much slower kernel (181 TFLOPS instead of 600+ TFLOPS). Always set both:

```bash
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH python bench_script.py
```

### 10. Plugin: `ik`

- Single unified plugin at `plugins/ik/` — Claude Code plugin with `.claude-plugin/plugin.json`
- CLI-only: all skills use bash/Python CLI commands, no MCP server
- Skills: `exec`, `inspect`, `docs`, `index`, `optimize`
- Invocation: `/ik:exec`, `/ik:inspect`, `/ik:docs`, `/ik:index`, `/ik:optimize`
- Old plugins (`cuda`, `kb`) archived in `plugins/deprecated/`

### 11. Git worktrees

Worktrees provide isolated copies of the repo for parallel or experimental work without disturbing `main`.

#### Convention

- **Location:** all worktrees live under `.worktrees/` in the project root
- **`.worktrees/`** is git-ignored — ephemeral working copies, not project data
- **Branch naming:** `worktree-<host>-<name>` — include the host short name so branches stay unique across machines (e.g. `worktree-h8_3-matmul`, `worktree-h8_4-fa4-optimize`)
- **Directory naming:** `.worktrees/<name>` — local directory does not need the host prefix (the branch carries it)

Host short names are defined in section 8 (`h8_3`, `h8_4`, `_one`, `_two`, etc.).

#### Creating a worktree

```bash
# From the project root:
mkdir -p .worktrees
git worktree add .worktrees/<name> -b worktree-<host>-<name>
```

Example (on h8_3):

```bash
git worktree add .worktrees/matmul -b worktree-h8_3-matmul
```

#### Listing worktrees

```bash
git worktree list
```

#### Removing a worktree

```bash
git worktree remove .worktrees/<name>
# Then optionally delete the branch:
git branch -d worktree-<host>-<name>
```

#### Rules

- Do not create worktrees outside `.worktrees/` — the old locations (`.claude/worktrees/`, sibling directories like `kernel_lab-worktrees/`) are deprecated
- Worktrees are ephemeral — merge or rebase work back to `main`, then remove the worktree
- Do not commit `.worktrees/` to git — it is in `.gitignore`
- Each worktree has its own working tree but shares the same `.git` object store

### 10. Results file naming

Results are stored under `results/<arch>/<device>/matmul/` (or other kernel family).

**Filename format:** `YYYYMMDD_HHMM_<hash>_<slug>.md`

| Component | Description | Example |
|-----------|-------------|---------|
| Date+time | `YYYYMMDD_HHMM` | `20260403_0200` |
| Hash | Short commit hash | `e732f2d` |
| Slug | Kebab-case description | `sm90-wgmma-benchmark` |
| Extension | Always `.md` | `.md` |

**Example:** `20260403_0200_e732f2d_sm90-wgmma-benchmark.md`

### 11. Data directory layout

```text
data/
├── fixtures/           # Reference implementations and configs, by arch
│   ├── sm80/
│   ├── sm90/
│   ├── sm100/
│   └── sm120/
│       ├── devices.json          # SM120 device registry (RTX 5090 vs RTX PRO 6000)
│       ├── vecadd/               # cutedsl.py, cudnn.py, configs.json
│       ├── matmul/               # cutedsl.py, cute_gemm.py, cudnn.py, configs*.json
│       └── fa4/                  # cutedsl.py, cudnn.py, configs*.json
├── generated/          # Generated CUDA kernels, by arch
│   └── sm120/
│       ├── vecadd/generated.cu
│       ├── matmul/generated.cu
│       └── fa4/generated.cu
└── nvidia-docs/        # Cached NVIDIA documentation

.worktrees/             # Git worktrees for isolated development (git-ignored)
```

- `data/` is tracked in git — project data (fixtures, generated code, docs)
- `.worktrees/` is git-ignored — ephemeral working copies for parallel development
- Fixture entry point files are named `cutedsl.py` (CuTe DSL reference implementations)
- Device-specific configs use `configs_<device>.json` naming (e.g. `configs_rtx5090.json`)

### 12. Benchmarking rules

#### All benchmarks must go through the unified eval harness

All code — cuDNN/cuBLAS, CuTe DSL, and Generated CUDA — **must** be timed through
the unified eval harness. There are two harness implementations that enforce
identical methodology:

1. **C harness** (`eval_harness.cu`): for Generated CUDA kernels via `kernel_run()`
2. **Python harness** (`measure_reference()` in `eval_support.py`): for CuTe DSL
   and cuDNN/cuBLAS Python modules

Both harnesses enforce:
- **Cold-L2 conditions**: L2 flush before each trial
- **Fresh input buffers**: new allocations per trial (new pointers break pointer caches)
- **Standardized timing**: CUDA events, 5 warmup + 10 trials (configurable)
- **Fair comparison**: identical methodology across all three implementations

**Fixture files (`cutedsl.py`, `cudnn.py`) must NOT contain their own timing code.**
They define only `Model`, `get_inputs()`, and `get_init_inputs()` — the harness
provides all measurement infrastructure. Any `main()`, timing loop, or CUDA event
code in fixture files is a bug and must be removed.

Running directly (e.g., `torch.mm()` in a Python loop) inflates numbers by up to 15% at large
sizes due to warm L2, causing unfair comparisons.

#### Never write custom benchmark scripts

**Never** write ad-hoc Python or CUDA benchmark scripts (e.g., `bench_all.py`, standalone
timing loops). All benchmarking **must** go through the existing eval infrastructure:

1. **Generated kernels**: compile via `compile.sh` + run via `eval_harness.cu`
2. **Reference/cuDNN**: run via `evaluate.py` (which calls the harness + reference modules)
3. **Local convenience**: use the Makefile targets or cuda_exec service API

Custom scripts bypass cold-L2 flushing, fresh-pointer allocation, and fair comparison
methodology. Results from custom scripts are unreliable and must not be used for
optimization decisions.

#### Correctness is a hard gate

Never push or commit code when correctness tests are failing. If any config fails correctness,
fix it first. Do not push performance-only improvements that have correctness regressions.

#### cuDNN baseline = `torch.mm()` via cuBLAS

The cuDNN reference (`cudnn.py`) uses `torch.mm()` which dispatches to cuBLAS `cublasGemmEx`.
This is the vendor-optimized baseline. Do **not** use CUTLASS for the cuDNN baseline.

## License

MIT — see `LICENSE`.
