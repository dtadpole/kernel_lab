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
- remove the temporary runtime directory after the run

They are intentionally allowed to observe expected failures from the underlying
CUDA toolchain or sample kernels. The purpose is to keep interface coverage and
workflow coverage in place even before the lower-level implementation is fully
stable.
