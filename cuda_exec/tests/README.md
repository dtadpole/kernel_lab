# cuda_exec tests

This directory is for **end-to-end integration tests only**.

Do **not** put unit tests here.

These tests should exercise the public service interface by:

- starting a real uvicorn service in a subprocess
- calling HTTP endpoints
- sending realistic request payloads
- validating response shape and high-level behavior

Current fixture direction:

- use one original CuTeDSL-style vector-add source and one generated inline-PTX CUDA source
- keep evaluate/profile config sets in fixture files rather than embedding them directly in the main integration-test code
- current config fixture: `tests/fixtures/configs/vector_add_shapes.json`
- drive evaluate/profile with 4–6 slug-keyed configs
- cover a mix of 1D / 2D / 3D input-shape metadata in those configs
- include multiple 1D sizes, plus at least one 2D case and one 3D case
- keep config slugs semantically meaningful for vector-add fixtures: use size/shape/rank terms, not unrelated concepts like causal/noncausal
- for vector-add fixtures, keep the config body pertinent too: shape/rank/input_size metadata is enough; avoid unrelated transformer-style fields like `num_layers`, `num_heads`, or `embedding_size`

Runtime side effects should be isolated during tests:

- use a temporary runtime root via `CUDA_EXEC_ROOT`
- clean up the subprocess on teardown
- preserve the temporary run directory on teardown for inspection
- prefer placing the temporary test root under `~/temp/`
- prefix the run directory name with `YYYY-MM-DD-HH-MM-`
- then use a kebab-case slug plus PID, e.g. `2026-03-28-23-27-cuda-exec-integration-12345-...`
- the current harness uses the repo-local Python environment at `cuda_exec/.venv`
- if the harness is later moved to a fully temporary uv-managed environment, prefer the conventional path `<temp-run-dir>/.venv`
- practical note for that future shape: after `uv venv <temp-run-dir>/.venv`, install dependencies with `uv pip install --python <temp-run-dir>/.venv/bin/python -r cuda_exec/requirements.txt` so the temporary environment is targeted explicitly

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
