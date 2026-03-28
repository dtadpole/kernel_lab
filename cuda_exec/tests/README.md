# cuda_exec tests

This directory is for **end-to-end integration tests only**.

Do **not** put unit tests here.

These tests should exercise the public service interface by:

- starting a real uvicorn service in a subprocess
- calling HTTP endpoints
- sending realistic request payloads
- validating response shape and high-level behavior

Runtime side effects should be isolated during tests:

- use a temporary runtime root via `CUDA_EXEC_ROOT`
- clean up the subprocess on teardown
- prefer placing the temporary test root under `~/temp/`
- prefer a kebab-case subfolder name with a slug plus PID, e.g. `cuda-exec-integration-12345-...`
- preferably also provision the uvicorn Python environment from `cuda_exec/requirements.txt` using `uv` inside a temporary folder, rather than relying forever on a persistent repo-local environment
- when doing so, prefer the conventional path `<temp-run-dir>/.venv`
- practical note: after `uv venv <temp-run-dir>/.venv`, install dependencies with `uv pip install --python <temp-run-dir>/.venv/bin/python -r cuda_exec/requirements.txt` so the temporary environment is targeted explicitly

Retention direction for inspection-friendly runs:

- preserve the run directory, `.venv`, request/response payloads, service logs, and runtime-root contents after the run
- do not depend on immediate deletion of intermediate outputs
- instead, use a separate cleanup/retention process, for example pruning runs older than 7 days

They are intentionally allowed to observe expected failures from the underlying
CUDA toolchain or sample kernels. The purpose is to keep interface coverage and
workflow coverage in place even before the lower-level implementation is fully
stable.
