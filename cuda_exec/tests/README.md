# cuda_exec tests

This directory is for **end-to-end integration tests only**.

Do **not** put unit tests here.

These tests should exercise the public service interface by:

- starting the FastAPI service
- calling HTTP endpoints
- sending realistic request payloads
- validating response shape and high-level behavior

They are intentionally allowed to observe expected failures from the underlying
CUDA toolchain or sample kernels. The purpose is to keep interface coverage and
workflow coverage in place even before the lower-level implementation is fully
stable.
