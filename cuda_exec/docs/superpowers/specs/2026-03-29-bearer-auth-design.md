# Bearer Token Authentication for cuda_exec

## Context

`cuda_exec` is a FastAPI service for remote CUDA kernel compilation, evaluation, and profiling. It currently has **no authentication** — all endpoints are publicly accessible. Since the service is designed for external consumption, we need to gate access behind a bearer token to prevent unauthorized use.

## Design

### Key file

- **Default path**: `~/.keys/cuda_exec.key`
- **Override**: `CUDA_EXEC_KEY_PATH` environment variable (mirrors the existing `CUDA_EXEC_ROOT` pattern for test isolation)
- **Format**: Plain text file containing a single token string. Leading/trailing whitespace is stripped.
- **Startup behavior**: The service reads the key file once at import time. If the file is missing, unreadable, or empty after stripping, the service **refuses to start** with a clear error message (fail-fast).

### Auth module: `auth.py`

New module at `cuda_exec/auth.py` with two exports:

1. **`load_key() -> str`** — Called at module level. Reads key file, strips whitespace, validates non-empty, returns the token string. Raises `SystemExit` on failure.

2. **`verify_bearer_token`** — FastAPI dependency using `fastapi.security.HTTPBearer`. Extracts the `Authorization: Bearer <token>` header and compares against the loaded key using `hmac.compare_digest` (constant-time comparison to prevent timing attacks). Returns `None` on success; raises `HTTPException(401, detail="invalid bearer token")` on failure.

### Endpoint integration

| Endpoint | Auth required |
|----------|--------------|
| `GET /healthz` | No — allows LB/monitoring probes without credentials |
| `POST /compile` | Yes |
| `POST /evaluate` | Yes |
| `POST /profile` | Yes |
| `POST /execute` | Yes |
| `POST /files/read` | Yes |

Auth is applied via `Depends(verify_bearer_token)` in each protected endpoint's function signature. No middleware.

### Client usage

```
Authorization: Bearer <token-from-key-file>
```

Requests to protected endpoints without this header, or with an incorrect token, receive:

```json
HTTP 401
{"detail": "invalid bearer token"}
```

Requests missing the `Authorization` header entirely receive FastAPI's built-in `HTTPBearer` response:

```json
HTTP 403
{"detail": "Not authenticated"}
```

### Test changes

**Existing e2e tests** (`tests/test_e2e_service.py`):

- `ServiceProcess.start()` writes a test key file into the instance temp directory and sets `CUDA_EXEC_KEY_PATH` in the service environment.
- `post_json()` and `get_json()` include the `Authorization: Bearer <test-token>` header on all requests.
- No behavioral changes to existing test assertions — they continue to test the same contracts, now with auth.

**New auth-specific tests** (added to `test_e2e_service.py`):

| Test | Behavior |
|------|----------|
| `test_healthz_requires_no_auth` | GET `/healthz` without token returns 200 |
| `test_protected_endpoint_rejects_missing_token` | POST `/compile` without token returns 401/403 |
| `test_protected_endpoint_rejects_wrong_token` | POST `/compile` with bad token returns 401 |
| `test_protected_endpoint_accepts_valid_token` | POST `/compile` with correct token returns 200 (covered by existing compile tests) |

### Files to create or modify

| File | Action |
|------|--------|
| `cuda_exec/auth.py` | **Create** — key loading + bearer token dependency |
| `cuda_exec/main.py` | **Modify** — add `Depends(verify_bearer_token)` to 5 protected endpoints |
| `cuda_exec/tests/test_e2e_service.py` | **Modify** — provision test key, add auth header, add auth-specific tests |
| `cuda_exec/DESIGN.md` | **Modify** — document auth contract in the public API section |

### What this does NOT include

- No RBAC, scopes, or multi-tenant support
- No token rotation or expiry mechanism
- No rate limiting
- No JWT — a static bearer token is sufficient for the current use case
