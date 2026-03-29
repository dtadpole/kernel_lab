# cuda_exec Service Contract Audit

This note records the current public contract surface for `/home/centos/kernel_lab/cuda_exec` and checks that the README, FastAPI wiring, and Pydantic models describe the same interface.

## Public endpoints

Defined in `/home/centos/kernel_lab/cuda_exec/main.py`:

- `GET /healthz`
- `POST /compile`
- `POST /files/read`
- `POST /evaluate`
- `POST /profile`
- `POST /execute`

## Request/response model surface

Defined in `/home/centos/kernel_lab/cuda_exec/models.py`:

- `CompileRequest` / `CompileResponse`
- `ReadFileRequest` / `ReadFileResponse`
- `EvaluateRequest` / `EvaluateResponse`
- `ProfileRequest` / `ProfileResponse`
- `ExecuteRequest` / `ExecuteResponse`

### Compile contract

- inputs are inline file maps:
  - `reference_files: Dict[relative_path, content]`
  - `generated_files: Dict[relative_path, content]`
- response includes:
  - `all_ok`
  - `attempt`
  - `artifacts`
  - `logs`
  - stage-local compile metadata

### Evaluate contract

- request uses slug-keyed configs:
  - `configs: Dict[config_slug, Dict[str, Any]]`
- response includes:
  - `all_ok`
  - `attempt`
  - `configs`
- per-config evaluate output includes:
  - `status`
  - `reference`
  - `generated`
  - `correctness`
  - `performance`
  - `artifacts`
  - `logs`

### Profile contract

- request uses slug-keyed configs and supports:
  - `mode`
  - `profiler_backend`
- response includes:
  - `all_ok`
  - `attempt`
  - `configs`
- per-config profile output includes:
  - `status`
  - `summary`
  - `reference`
  - `generated`
  - `reference_summary`
  - `generated_summary`
  - `artifacts`
  - `logs`

### Execute contract

- request accepts:
  - `command`
  - `env`
  - `timeout_seconds`
- current public policy is intentionally restricted:
  - the command must point to a CUDA Toolkit binary
- response includes:
  - `all_ok`
  - `returncode`
  - `artifacts`
  - `logs`

## Consistency check

### README vs models/main

Current README at `/home/centos/kernel_lab/cuda_exec/README.md` is consistent with the live endpoint/model surface in these key places:

- compile uses `reference_files` / `generated_files`
- evaluate responses are config-keyed and carry `reference`, `generated`, `correctness`, `performance`, `artifacts`, and `logs`
- profile supports `mode` + `profiler_backend`
- profile outputs include `reference_summary` / `generated_summary`
- the README explicitly documents the `comparison_runtime` vs `ncu` backend boundary

### Endpoint wiring vs models

`/home/centos/kernel_lab/cuda_exec/main.py` wires each public endpoint to the matching request/response model family in `/home/centos/kernel_lab/cuda_exec/models.py` without an obvious contract mismatch.

## Audit conclusion

The current public service contract is coherent across:

- `/home/centos/kernel_lab/cuda_exec/README.md`
- `/home/centos/kernel_lab/cuda_exec/models.py`
- `/home/centos/kernel_lab/cuda_exec/main.py`

No new contract mismatch was identified during this audit.
