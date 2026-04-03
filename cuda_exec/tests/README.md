# cuda_exec tests

This directory is for **end-to-end integration tests only**.

Do **not** put unit tests here.

These tests should exercise the public service interface by:

- starting a real uvicorn service in a subprocess
- calling HTTP endpoints
- sending realistic request payloads
- validating response shape and high-level behavior

Current fixture direction:

- use one reference CuTeDSL-style vector-add source and one generated inline-PTX CUDA source
- keep evaluate/profile config sets in fixture files rather than embedding them directly in the main integration-test code
- current config fixture: `data/fixtures/sm120/vecadd/configs.json`
- drive evaluate/profile with 4–6 slug-keyed configs
- cover a mix of 1D / 2D / 3D input-shape metadata in those configs
- include multiple 1D sizes, plus at least one 2D case and one 3D case
- keep config slugs semantically meaningful for vector-add fixtures: use size/shape/rank terms, not unrelated concepts like causal/noncausal
- for vector-add fixtures, keep the config body pertinent too: shape/rank/input_size metadata is enough; avoid unrelated transformer-style fields like `num_layers`, `num_heads`, or `embedding_size`

Runtime side effects should be isolated during tests:

- use a temporary runtime root via `CUDA_EXEC_ROOT`
- clean up the subprocess on teardown
- preserve the temporary run directory on teardown for inspection
- prefer placing the temporary suite run directory under `~/temp/`
- create one top-level temp directory per integration-suite invocation, not one sibling temp directory per test class or service lifecycle
- prefix the run directory name with `YYYY-MM-DD-HH-MM-`
- then use a kebab-case slug plus PID, e.g. `2026-03-28-23-27-cuda-exec-integration-12345-...`
- if multiple service processes are started within the suite, reuse that same top-level run directory and give each service instance its own namespaced log/runtime-root paths inside it
- the harness now provisions a temporary uv-managed environment at `<temp-run-dir>/.venv` for the service process and pruning helper
- current transition note: the reference execution stack (`torch` / `cutlass.cute`) is still borrowed from the repo-local `cuda_exec/.venv` via `PYTHONPATH` injection until those dependencies have a fully reproducible temp-env installation path
- practical note for this shape: after `uv venv <temp-run-dir>/.venv`, install service dependencies with `uv pip install --python <temp-run-dir>/.venv/bin/python -r cuda_exec/requirements.txt`
- future tightening beyond the current transition state should replace the borrowed repo-local reference stack with a fully temp-env-managed dependency source

Retention helper for preserved runs:

- helper script: `cuda_exec/scripts/prune_temp_runs.py`
- invoke this helper before starting the temporary uvicorn service so older preserved run directories are pruned
- default behavior: delete preserved run directories older than 7 days
- `--dry-run` shows what would be deleted without removing anything
- directories are kept if their name contains `keep` or if they contain a marker file such as `KEEP`

They are intentionally allowed to observe expected failures from the underlying
CUDA toolchain or sample kernels. The purpose is to keep interface coverage and
workflow coverage in place even before the lower-level implementation is fully
stable.

## Current coverage map and remaining integration-test gaps

Current suite coverage already includes:

- compile request validation, inline file-map handling, second-attempt rejection, and compile artifact exposure
- file-read success/error behavior and max-bytes truncation
- evaluate endpoint shape plus standalone `scripts/evaluate.py` CLI execution after compile
- reference fixture contract enforcement (`Model(nn.Module)`, `get_init_inputs()`, `get_inputs(config)`) and fixture execution from config env
- NCU profile for `side=generated` and `side=reference`
- rejection of invalid `side` values
- execute endpoint logs-first public contract
- temporary runtime-root preservation and temp-venv harness behavior

Most important remaining integration-test gaps:

1. **Negative NCU contract path**
   - We now know one generated fixture falls back with `No kernels were profiled` while another can materialize a report.
   - The suite should still add an explicit assertion that the fallback fixture does *not* publish `.ncu-rep` and does record `ncu_profiled=false` / fallback metadata.

2. **Profile manifest/state contract checks**
   - Current tests mostly assert public response payloads.
   - We do not yet directly assert the kept `state/profile.attempt_###.json` manifest shape for comparison-runtime vs NCU runs.

3. **Execute failure-mode coverage**
   - `execute` is only checked for the success-ish `/usr/local/cuda/bin/nvcc --version` path.
   - We do not yet assert public behavior for a failing command, timeout, or explicit non-zero exit path.

4. **Compile artifact/content contract checks for alternate generated fixture**
   - The runtime-launch generated fixture now exists for NCU verification, but we do not yet assert compile artifact shape/logs for that alternate generated fixture specifically.

5. **Cross-attempt profile file retention sanity**
   - We do not yet assert that repeated profile attempts for the same turn create distinct attempt-tagged artifacts/logs rather than overwriting prior public files.

## Which gaps should be added now vs only documented

Add now:

1. **Negative NCU fallback contract path**
   - High value because we already know both success-ish and fallback NCU shapes exist.
   - Prevents regressions where fallback runs incorrectly claim `.ncu-rep` or omit explicit fallback metadata.

2. **Execute failure-mode coverage**
   - High value because `execute` is a direct public endpoint and its error/timeout contract should stay stable.
   - Cheap to exercise with one or two subprocess-oriented test cases.

3. **Cross-attempt profile retention sanity**
   - High value because attempt-tagged artifact retention is part of the public file/log contract, not just an internal detail.

Document only for now:

1. **Profile manifest/state contract checks**
   - Useful, but lower immediate value than public response/log/artifact regressions.
   - Better treated after the public API surface is stable.

2. **Alternate generated fixture compile artifact contract checks**
   - Nice to have, but the runtime-launch fixture was introduced mainly to unlock real NCU verification rather than to establish a second full compile contract matrix.
   - Can be revisited if alternate-fixture compilation starts to diverge materially.
