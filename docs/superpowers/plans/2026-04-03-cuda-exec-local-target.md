# cuda:exec Local + Remote Target Support

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a required `target` parameter to all cuda:exec action tools so each call explicitly specifies local (direct invocation with GPU index) or remote (HTTP proxy with host lookup) execution.

**Architecture:** The MCP server (`mcp_server.py`) gains a dispatch layer. Remote dispatch sends HTTP requests to a host resolved from `conf/hosts/default.yaml`. Local dispatch imports `cuda_exec` endpoint handlers directly (no HTTP, no auth), setting `CUDA_VISIBLE_DEVICES` for GPU selection. A new `dispatch.py` module owns target validation, host resolution, and both dispatch paths. The 4 data-retrieval tools (`get_data_point`, `get_compile_data`, `get_evaluate_data`, `get_profile_data`) are unchanged — they read from the local data store regardless of execution target.

**Tech Stack:** Python, FastMCP, httpx (remote), direct function calls (local), PyYAML (host config)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `plugins/cuda/dispatch.py` | Target validation, host resolution, local dispatch, remote dispatch |
| Modify | `cuda_exec/auth.py` | Make bearer token loading lazy (don't crash at import if key missing) |
| Modify | `plugins/cuda/mcp_server.py` | Add `target` param to 5 action tools, wire dispatch |
| Modify | `plugins/cuda/skills/exec/SKILL.md` | Document target parameter |
| Modify | `plugins/cuda/.claude-plugin/plugin.json` | Update description |
| Create | `plugins/cuda/tests/test_dispatch.py` | Unit tests for dispatch module |
| Modify | `plugins/cuda/tests/test_exec_tools.py` | Update integration tests for target param |
| Modify | `plugins/cuda/tests/test_plugin.py` | Update tool count assertion if needed |

---

### Task 1: Make `cuda_exec/auth.py` Lazy

**Why:** Currently `auth.py` loads the bearer token at module import time and calls `sys.exit(1)` if the key file is missing. When the MCP server imports `cuda_exec.main` for local dispatch, this crashes the entire MCP server even if the user only wants local execution and has no remote key file.

**Files:**
- Modify: `cuda_exec/auth.py`
- Test: `cuda_exec/tests/` (existing tests should still pass)

- [ ] **Step 1: Read the current auth module**

Read `cuda_exec/auth.py` to confirm the eager loading pattern.

- [ ] **Step 2: Write a test for lazy loading**

Create a test that verifies the module can be imported without a key file, and that `verify_bearer_token` only fails when actually called.

```python
# cuda_exec/tests/test_auth_lazy.py
"""Test that auth module loads lazily — no crash at import time."""

import importlib
import os
from unittest.mock import patch

import pytest


def test_auth_imports_without_key_file():
    """Importing auth should NOT crash even if key file is missing."""
    with patch.dict(os.environ, {"CUDA_EXEC_KEY_PATH": "/nonexistent/path.key"}):
        # Force reimport
        import cuda_exec.auth as auth_mod
        importlib.reload(auth_mod)
        # Module loaded — no crash
        assert hasattr(auth_mod, "verify_bearer_token")


def test_auth_raises_on_use_without_key():
    """verify_bearer_token should raise when actually called without a valid key."""
    with patch.dict(os.environ, {"CUDA_EXEC_KEY_PATH": "/nonexistent/path.key"}):
        import cuda_exec.auth as auth_mod
        importlib.reload(auth_mod)
        with pytest.raises(Exception):
            # Simulate FastAPI calling the dependency
            import asyncio
            from unittest.mock import MagicMock
            creds = MagicMock()
            creds.credentials = "fake-token"
            asyncio.run(auth_mod.verify_bearer_token(creds))
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_exec/tests/test_auth_lazy.py -v`
Expected: FAIL — current auth crashes at import time.

- [ ] **Step 4: Make auth loading lazy**

Change `cuda_exec/auth.py` to defer key loading until first use:

```python
"""Bearer-token authentication for cuda_exec.

Token is loaded lazily on first use, NOT at import time. This allows
modules that import auth (e.g. main.py) to be loaded without requiring
the key file to exist — important for local-only execution paths.
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_DEFAULT_KEY_PATH = Path.home() / ".keys" / "cuda_exec.key"
_bearer_scheme = HTTPBearer()

# Lazy-loaded bearer token
_BEARER_TOKEN: str | None = None
_TOKEN_LOADED: bool = False


def _load_bearer_token() -> str:
    """Load bearer token from key file. Raises RuntimeError if missing."""
    key_path = Path(os.environ.get("CUDA_EXEC_KEY_PATH") or str(_DEFAULT_KEY_PATH))
    try:
        token = key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(f"cuda_exec auth: key file not found: {key_path}")
    except OSError as exc:
        raise RuntimeError(f"cuda_exec auth: cannot read key file {key_path}: {exc}")
    if not token:
        raise RuntimeError(f"cuda_exec auth: key file is empty: {key_path}")
    return token


def _get_token() -> str:
    """Get the bearer token, loading it on first call."""
    global _BEARER_TOKEN, _TOKEN_LOADED
    if not _TOKEN_LOADED:
        _BEARER_TOKEN = _load_bearer_token()
        _TOKEN_LOADED = True
    assert _BEARER_TOKEN is not None
    return _BEARER_TOKEN


def load_key() -> str:
    """Public accessor — returns the token, loading lazily."""
    return _get_token()


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = None,
) -> None:
    """FastAPI dependency: verify the bearer token."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = _get_token()
    if not hmac.compare_digest(credentials.credentials, token):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
```

Note: Preserve the existing function signatures that other code depends on. The key change is removing the module-level `_BEARER_TOKEN: str = load_key()` line.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_exec/tests/test_auth_lazy.py -v`
Expected: PASS

- [ ] **Step 6: Run existing tests to check for regressions**

Run: `cd /home/centos/kernel_lab && python -m pytest cuda_exec/tests/ -m quick -v`
Expected: All existing tests PASS.

- [ ] **Step 7: Commit**

```bash
git add cuda_exec/auth.py cuda_exec/tests/test_auth_lazy.py
git commit -m "refactor: make cuda_exec auth lazy-load bearer token

Defer key file loading until first use instead of at import time.
This allows importing cuda_exec.main for local dispatch without
requiring a remote key file to exist."
```

---

### Task 2: Create `plugins/cuda/dispatch.py`

**Why:** This module owns the dispatch decision: validate the target, resolve remote hosts to URLs, call local endpoint handlers directly, or proxy via HTTP. Keeping this out of `mcp_server.py` keeps the MCP tool definitions clean.

**Files:**
- Create: `plugins/cuda/dispatch.py`
- Create: `plugins/cuda/tests/test_dispatch.py`

- [ ] **Step 1: Write tests for target validation**

```python
# plugins/cuda/tests/test_dispatch.py
"""Tests for the dispatch module — target validation and host resolution."""

import pytest

from plugins.cuda.dispatch import validate_target, resolve_host_url


class TestValidateTarget:
    def test_local_valid(self):
        validate_target({"mode": "local", "gpu_index": 0})

    def test_local_gpu_index_required(self):
        with pytest.raises(ValueError, match="gpu_index"):
            validate_target({"mode": "local"})

    def test_local_gpu_index_nonnegative(self):
        with pytest.raises(ValueError, match="gpu_index"):
            validate_target({"mode": "local", "gpu_index": -1})

    def test_remote_valid(self):
        validate_target({"mode": "remote", "host": "_one"})

    def test_remote_host_required(self):
        with pytest.raises(ValueError, match="host"):
            validate_target({"mode": "remote"})

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            validate_target({"mode": "cloud"})

    def test_missing_mode(self):
        with pytest.raises(ValueError, match="mode"):
            validate_target({})


class TestResolveHostUrl:
    def test_resolve_one(self):
        url = resolve_host_url("_one")
        assert "41980" in url

    def test_resolve_two(self):
        url = resolve_host_url("_two")
        assert "42980" in url

    def test_unknown_host(self):
        with pytest.raises(ValueError, match="not found"):
            resolve_host_url("nonexistent_host")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/centos/kernel_lab && python -m pytest plugins/cuda/tests/test_dispatch.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Write the dispatch module**

```python
# plugins/cuda/dispatch.py
"""Dispatch layer for cuda:exec — routes to local or remote execution.

Local dispatch: imports cuda_exec endpoint handlers directly, sets
CUDA_VISIBLE_DEVICES, calls them as plain Python functions (no HTTP,
no auth). The response is the Pydantic model serialized to dict.

Remote dispatch: HTTP POST to the cuda_exec service on a resolved host.
Host lookup uses conf/hosts/default.yaml.

Target format:
    {"mode": "local",  "gpu_index": 0}
    {"mode": "remote", "host": "_one"}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOSTS_CONFIG_PATH = _REPO_ROOT / "conf" / "hosts" / "default.yaml"

_hosts_cache: dict[str, Any] | None = None

_TIMEOUT = httpx.Timeout(
    float(os.environ.get("CUDA_AGENT_MCP_REQUEST_TIMEOUT", "300.0")),
    connect=float(os.environ.get("CUDA_AGENT_MCP_CONNECT_TIMEOUT", "10.0")),
)

# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------


def validate_target(target: dict[str, Any]) -> None:
    """Validate a target dict. Raises ValueError on invalid input."""
    mode = target.get("mode")
    if mode not in ("local", "remote"):
        raise ValueError(
            'target.mode must be "local" or "remote", '
            f"got: {mode!r}"
        )
    if mode == "local":
        gpu_index = target.get("gpu_index")
        if gpu_index is None:
            raise ValueError(
                "target.gpu_index is required for local mode"
            )
        if not isinstance(gpu_index, int) or gpu_index < 0:
            raise ValueError(
                "target.gpu_index must be a non-negative integer, "
                f"got: {gpu_index!r}"
            )
    if mode == "remote":
        host = target.get("host")
        if not host:
            raise ValueError(
                "target.host is required for remote mode"
            )


# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------


def _load_hosts_config() -> dict[str, Any]:
    global _hosts_cache
    if _hosts_cache is None:
        with _HOSTS_CONFIG_PATH.open() as f:
            _hosts_cache = yaml.safe_load(f)
    return _hosts_cache


def resolve_host_url(host_name: str) -> str:
    """Resolve a host name to an HTTP base URL.

    Uses internet_host + port if available, else ssh_host + port.
    """
    config = _load_hosts_config()
    hosts = config.get("hosts", {})
    if host_name not in hosts:
        available = sorted(hosts.keys())
        raise ValueError(
            f"Host {host_name!r} not found in {_HOSTS_CONFIG_PATH}. "
            f"Available: {available}"
        )
    host = hosts[host_name]
    port = host.get("port", 8000)
    hostname = host.get("internet_host") or host.get("ssh_host", "localhost")
    return f"http://{hostname}:{port}"


# ---------------------------------------------------------------------------
# Remote dispatch
# ---------------------------------------------------------------------------


def _load_bearer_token() -> str:
    """Load bearer token for remote auth."""
    default_key = Path.home() / ".keys" / "cuda_exec.key"
    key_path = Path(os.environ.get("CUDA_EXEC_KEY_PATH") or str(default_key))
    token = key_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Key file is empty: {key_path}")
    return token


_bearer_token_cache: str | None = None


def _get_bearer_token() -> str:
    global _bearer_token_cache
    if _bearer_token_cache is None:
        _bearer_token_cache = _load_bearer_token()
    return _bearer_token_cache


async def dispatch_remote(
    endpoint: str,
    body: dict[str, Any],
    *,
    host: str,
) -> dict[str, Any]:
    """POST to a remote cuda_exec service and return the raw response dict."""
    base_url = resolve_host_url(host)
    token = _get_bearer_token()
    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        resp = await client.post(
            endpoint,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Local dispatch
# ---------------------------------------------------------------------------

# Endpoint handler mapping — populated lazily to avoid import-time
# side effects from cuda_exec modules.
_LOCAL_HANDLERS: dict | None = None
_LOCAL_REQUEST_CLASSES: dict | None = None


def _init_local_handlers() -> None:
    """Lazily import cuda_exec endpoint handlers and request models."""
    global _LOCAL_HANDLERS, _LOCAL_REQUEST_CLASSES
    if _LOCAL_HANDLERS is not None:
        return

    from cuda_exec.main import (
        compile_endpoint,
        evaluate_endpoint,
        file_read_endpoint,
        execute_endpoint,
        profile_endpoint,
    )
    from cuda_exec.models import (
        CompileRequest,
        EvaluateRequest,
        ExecuteRequest,
        FileReadRequest,
        ProfileRequest,
    )

    _LOCAL_HANDLERS = {
        "/compile": compile_endpoint,
        "/evaluate": evaluate_endpoint,
        "/profile": profile_endpoint,
        "/execute": execute_endpoint,
        "/files/read": file_read_endpoint,
    }
    _LOCAL_REQUEST_CLASSES = {
        "/compile": CompileRequest,
        "/evaluate": EvaluateRequest,
        "/profile": ProfileRequest,
        "/execute": ExecuteRequest,
        "/files/read": FileReadRequest,
    }


def dispatch_local(
    endpoint: str,
    body: dict[str, Any],
    *,
    gpu_index: int,
) -> dict[str, Any]:
    """Call a cuda_exec endpoint handler directly with GPU selection.

    Sets CUDA_VISIBLE_DEVICES for the duration of the call, then restores it.
    Returns the Pydantic response model serialized to dict.
    """
    _init_local_handlers()
    assert _LOCAL_HANDLERS is not None
    assert _LOCAL_REQUEST_CLASSES is not None

    if endpoint not in _LOCAL_HANDLERS:
        raise ValueError(f"Unknown endpoint: {endpoint}")

    handler = _LOCAL_HANDLERS[endpoint]
    request_cls = _LOCAL_REQUEST_CLASSES[endpoint]
    request = request_cls(**body)

    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    try:
        response = handler(request)
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible

    return response.model_dump()
```

- [ ] **Step 4: Run validation tests**

Run: `cd /home/centos/kernel_lab && python -m pytest plugins/cuda/tests/test_dispatch.py -v -m "not integration and not gpu"`
Expected: PASS for validation and host resolution tests.

- [ ] **Step 5: Commit**

```bash
git add plugins/cuda/dispatch.py plugins/cuda/tests/test_dispatch.py
git commit -m "feat: add dispatch module for local/remote cuda_exec routing

Validates target dicts, resolves host names to URLs from hosts config,
and provides dispatch_local (direct function call) and dispatch_remote
(HTTP proxy) execution paths."
```

---

### Task 3: Wire Dispatch into MCP Server

**Why:** Replace the hardcoded `_post()` HTTP proxy with target-aware dispatch. Every action tool gets a required `target` parameter (no default). The 4 data-retrieval tools remain unchanged.

**Files:**
- Modify: `plugins/cuda/mcp_server.py`

- [ ] **Step 1: Read the current mcp_server.py**

Read `plugins/cuda/mcp_server.py` to confirm current structure before modifying.

- [ ] **Step 2: Replace `_post()` with `_dispatch()`**

In `mcp_server.py`, replace the bearer token loading and `_post()` function. Remove the eager token loading at module level. Replace `_post()` with `_dispatch()` that routes through the dispatch module:

```python
# Replace these lines at module top:
#   _BEARER_TOKEN: str = _load_bearer_token()
#   def _auth_headers() -> ...
#   async def _post(endpoint, body) -> str:

# With:
from plugins.cuda.dispatch import (
    dispatch_local,
    dispatch_remote,
    validate_target,
)

async def _dispatch(endpoint: str, body: dict[str, Any], *, target: dict[str, Any]) -> str:
    """Dispatch to local or remote, save data, compact, return JSON string."""
    validate_target(target)

    if target["mode"] == "local":
        raw = dispatch_local(endpoint, body, gpu_index=target["gpu_index"])
    else:
        raw = await dispatch_remote(endpoint, body, host=target["host"])

    _save_data_point(endpoint, body, raw)
    data = _compact_response(raw)
    return json.dumps(data, indent=2)
```

Also remove: `_load_bearer_token()`, `_BEARER_TOKEN`, `_auth_headers()`, `_BASE_URL`, `_TIMEOUT` (these are now in `dispatch.py`). Keep `_MAX_CONTENT_CHARS`, `_DEFAULT_TOOL_TIMEOUT`, data store code, and compaction code.

- [ ] **Step 3: Add `target` parameter to `compile` tool**

```python
@mcp.tool()
async def compile(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    target: Annotated[dict[str, Any], Field(description='Execution target. REQUIRED. Either {"mode": "local", "gpu_index": N} or {"mode": "remote", "host": "hostname"}')],
    reference_files: Annotated[dict[str, str], Field(description="Map of relative_path -> file content for reference source inputs")],
    generated_files: Annotated[dict[str, str], Field(description="Map of relative_path -> file content; must contain exactly one .cu file")],
    timeout_seconds: Annotated[int, Field(description="Max seconds for compile")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """Compile CUDA source files and produce a binary, PTX, and SASS.
    ...(keep existing docstring)...
    """
    return await _dispatch("/compile", {
        "metadata": metadata,
        "reference_files": reference_files,
        "generated_files": generated_files,
        "timeout_seconds": timeout_seconds,
    }, target=target)
```

- [ ] **Step 4: Add `target` parameter to `evaluate` tool**

Same pattern — add `target` as second parameter (after metadata, before other params), replace `_post()` with `_dispatch()`:

```python
@mcp.tool()
async def evaluate(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    target: Annotated[dict[str, Any], Field(description='Execution target. REQUIRED. Either {"mode": "local", "gpu_index": N} or {"mode": "remote", "host": "hostname"}')],
    configs: Annotated[dict[str, dict[str, Any]], Field(description="Slug-keyed runtime config payloads")],
    timeout_seconds: Annotated[int, Field(description="Max seconds for evaluate")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """Evaluate a compiled CUDA kernel for correctness and performance.
    ...(keep existing docstring)...
    """
    return await _dispatch("/evaluate", {
        "metadata": metadata,
        "configs": configs,
        "timeout_seconds": timeout_seconds,
    }, target=target)
```

- [ ] **Step 5: Add `target` parameter to `profile` tool**

```python
@mcp.tool()
async def profile(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    target: Annotated[dict[str, Any], Field(description='Execution target. REQUIRED. Either {"mode": "local", "gpu_index": N} or {"mode": "remote", "host": "hostname"}')],
    configs: Annotated[dict[str, dict[str, Any]], Field(description="Slug-keyed runtime config payloads")],
    side: Annotated[Literal["generated", "reference"], Field(description="Which side to NCU-profile")] = "generated",
    timeout_seconds: Annotated[int, Field(description="Max seconds for profile")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """NCU-profile a compiled CUDA kernel or reference Python/CuTe DSL kernel.
    ...(keep existing docstring)...
    """
    return await _dispatch("/profile", {
        "metadata": metadata,
        "configs": configs,
        "side": side,
        "timeout_seconds": timeout_seconds,
    }, target=target)
```

- [ ] **Step 6: Add `target` parameter to `execute` tool**

```python
@mcp.tool()
async def execute(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    target: Annotated[dict[str, Any], Field(description='Execution target. REQUIRED. Either {"mode": "local", "gpu_index": N} or {"mode": "remote", "host": "hostname"}')],
    command: Annotated[list[str], Field(description="Executable plus arguments")],
    env: Annotated[dict[str, str], Field(description="Extra environment variables")] = {},
    timeout_seconds: Annotated[int, Field(description="Max seconds for execute")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """Run an ad-hoc CUDA tool command.
    ...(keep existing docstring)...
    """
    return await _dispatch("/execute", {
        "metadata": metadata,
        "command": command,
        "env": env,
        "timeout_seconds": timeout_seconds,
    }, target=target)
```

- [ ] **Step 7: Add `target` parameter to `read_file` tool**

```python
@mcp.tool()
async def read_file(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    target: Annotated[dict[str, Any], Field(description='Execution target. REQUIRED. Either {"mode": "local", "gpu_index": N} or {"mode": "remote", "host": "hostname"}')],
    path: Annotated[str, Field(description="Relative path under the turn root")],
    max_bytes: Annotated[int | None, Field(description="Optional max bytes to read")] = None,
) -> str:
    """Read a file from a specific turn's directory tree.
    ...(keep existing docstring)...
    """
    body: dict[str, Any] = {"metadata": metadata, "path": path}
    if max_bytes is not None:
        body["max_bytes"] = max_bytes
    return await _dispatch("/files/read", body, target=target)
```

- [ ] **Step 8: Verify MCP server imports cleanly**

Run: `cd /home/centos/kernel_lab && CUDA_EXEC_KEY_PATH=/dev/null python -c "import plugins.cuda.mcp_server; print('OK')"`
Expected: OK (should not crash — token loading is now lazy).

- [ ] **Step 9: Commit**

```bash
git add plugins/cuda/mcp_server.py
git commit -m "feat: add required target parameter to all cuda:exec action tools

Every action tool (compile, evaluate, profile, execute, read_file)
now requires an explicit target dict:
  {\"mode\": \"local\", \"gpu_index\": N}
  {\"mode\": \"remote\", \"host\": \"hostname\"}

No default — caller must always specify local or remote.
Data-retrieval tools (get_*) unchanged."
```

---

### Task 4: Update Tests

**Why:** Existing integration tests call MCP tools without `target`. They need updating. Add new unit tests for dispatch wiring.

**Files:**
- Modify: `plugins/cuda/tests/test_exec_tools.py`
- Modify: `plugins/cuda/tests/test_plugin.py`

- [ ] **Step 1: Update `test_plugin.py` tool count**

Read `plugins/cuda/tests/test_plugin.py` and check if `test_cuda_mcp_tools_registered` asserts a specific tool count. The tool count is still 9 (we added no new tools, just new parameters), so this should be fine. Verify and adjust if needed.

- [ ] **Step 2: Update `test_exec_tools.py` integration tests**

Add `target` parameter to existing test calls. These tests are `@pytest.mark.integration` and require a running remote service:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_compile_vecadd(cuda_mcp, sample_metadata, vecadd_fixtures):
    """compile tool returns structured response."""
    result = await cuda_mcp.compile(
        metadata=sample_metadata,
        target={"mode": "remote", "host": "_one"},
        reference_files={"cutedsl.py": vecadd_fixtures["reference"]},
        generated_files={"generated.cu": vecadd_fixtures["generated"]},
    )
    data = json.loads(result)
    assert "all_ok" in data
    assert "artifacts" in data
    assert "tool_outputs" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compile_has_ptx(cuda_mcp, sample_metadata, vecadd_fixtures):
    """compile produces PTX output."""
    result = await cuda_mcp.compile(
        metadata=sample_metadata,
        target={"mode": "remote", "host": "_one"},
        reference_files={"cutedsl.py": vecadd_fixtures["reference"]},
        generated_files={"generated.cu": vecadd_fixtures["generated"]},
    )
    data = json.loads(result)
    ptx = data.get("artifacts", {}).get("ptx", {})
    assert ptx.get("path") is not None, "PTX artifact should have a path"
```

- [ ] **Step 3: Add local dispatch integration test**

Add a test marked `@pytest.mark.gpu` that tests local compile:

```python
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_compile_vecadd_local(cuda_mcp, sample_metadata, vecadd_fixtures):
    """compile tool works with local target."""
    # Use a unique turn to avoid conflict with remote tests
    sample_metadata["turn"] = 9000
    result = await cuda_mcp.compile(
        metadata=sample_metadata,
        target={"mode": "local", "gpu_index": 0},
        reference_files={"cutedsl.py": vecadd_fixtures["reference"]},
        generated_files={"generated.cu": vecadd_fixtures["generated"]},
    )
    data = json.loads(result)
    assert data["all_ok"] is True
    assert "artifacts" in data
```

- [ ] **Step 4: Run quick tests**

Run: `cd /home/centos/kernel_lab && python -m pytest plugins/cuda/tests/test_plugin.py -v`
Expected: PASS (tool count unchanged at 9).

- [ ] **Step 5: Commit**

```bash
git add plugins/cuda/tests/test_exec_tools.py plugins/cuda/tests/test_plugin.py
git commit -m "test: update cuda exec tests for required target parameter"
```

---

### Task 5: Update Skill Documentation

**Why:** The skill SKILL.md tells Claude how to use the tools. It must document the required `target` parameter, the two modes, and how to pick between them.

**Files:**
- Modify: `plugins/cuda/skills/exec/SKILL.md`
- Modify: `plugins/cuda/.claude-plugin/plugin.json`

- [ ] **Step 1: Update SKILL.md**

Replace the current SKILL.md with updated content that documents the target parameter. Key additions:

1. Add a `## Target` section right after `## Tools` explaining the required parameter
2. Update the workflow to mention target
3. Add examples for both modes

```markdown
---
name: exec
description: Compile, evaluate, and profile CUDA kernels locally or on remote GPU hosts
user-invocable: true
argument-hint: <local|remote> <action> [options]
---

# CUDA Kernel Execution

Compile, evaluate, and profile CUDA kernels locally or on remote GPU hosts via the cuda_exec service.

## Target (Required)

Every action tool call requires an explicit `target` parameter. No default — you must always specify where to run.

### Local execution
```json
{"mode": "local", "gpu_index": 0}
```
Runs directly on the local machine. `gpu_index` selects which GPU (maps to `CUDA_VISIBLE_DEVICES`).

### Remote execution
```json
{"mode": "remote", "host": "_one"}
```
Proxies to the cuda_exec HTTP service on the named host. Host names are resolved from `conf/hosts/default.yaml`.

### Skill invocation
- `/cuda:exec local 0 compile ...` → `target={"mode": "local", "gpu_index": 0}`
- `/cuda:exec remote _one compile ...` → `target={"mode": "remote", "host": "_one"}`

### Available remote hosts
| Host | GPU | Description |
|------|-----|-------------|
| `_one` | RTX PRO 6000 Blackwell | 98GB |
| `_two` | RTX PRO 6000 Blackwell | 98GB |
| `h8_3` | 8x NVIDIA H100 | Meta devvm |
| `h8_4` | 8x NVIDIA H100 | Meta devvm |

## Tools

- **health** — Check if the target cuda_exec service is responding (remote only)
- **compile** — Compile CUDA source to binary/PTX/SASS
- **evaluate** — Correctness + performance testing against runtime configs
- **profile** — NCU profiling (generated or reference side)
- **execute** — Ad-hoc command execution (e.g. query device info, toolkit versions)

## Workflow

1. **Target selection**: Determine local vs remote based on user instruction
2. **Health check** (remote only): Verify the target service is reachable
3. **Compile**: Call `compile` with target, metadata, reference_files, and generated_files
4. **Evaluate**: Call `evaluate` with the same target and metadata, plus configs
5. **Profile** (optional): Call `profile` to get NCU hardware metrics
6. **Iterate**: Modify source code → increment `metadata.turn` → compile again

## Kernel Types

The exec workflow supports multiple kernel implementation approaches:

- **Generated CUDA** — Hand-written `.cu` kernel following the `kernel_run` contract
- **CuTe DSL reference** — Python reference using `cutlass.cute` JIT compilation
- **cuDNN** — Native cuDNN kernel implementation (planned)

All types share the same compile → evaluate → profile pipeline. The reference
implementation provides the ground truth for correctness comparison.

## Rules

- Compile exactly once per turn before evaluate or profile
- New source code requires a new turn (increment metadata.turn)
- Old turns are immutable — never recompile on a previous turn number
- One compile fans out to many evaluate/profile calls with different configs
- Use the same target for all calls within a turn

## Metadata

Every tool requires a `metadata` dict:
```json
{
  "run_tag": "optim_001",
  "version": "v1",
  "direction_id": 7,
  "direction_slug": "vector-add",
  "turn": 1
}
```

## File Inputs

### compile

```json
{
  "reference_files": {"cutedsl.py": "<content>"},
  "generated_files": {"generated.cu": "<content>"}
}
```

- `reference_files`: Python reference source (Model class, get_inputs, get_init_inputs)
- `generated_files`: Must contain exactly one `.cu` file with `kernel_run` entry point

### evaluate / profile

```json
{
  "configs": {
    "vec1d-n65536": {"shape": [65536], "rank": 1, "input_size": 65536, "shape_kind": "1d"},
    "tensor2d-1024x1024": {"shape": [1024, 1024], "rank": 2, "input_size": 1048576, "shape_kind": "2d"}
  }
}
```

Configs are kernel-specific. Each config_slug is a stable identifier across evaluate/profile calls.

## Error Handling

- **Compile failure**: Check `all_ok` field. Use `/cuda:inspect` with `get_compile_data` field="tool_outputs" to see nvcc/ptxas errors.
- **Evaluate failure**: Check per-config `status` and `correctness.passed`. Use `/cuda:inspect` to fetch full logs.
- **Profile failure**: NCU profiling requires elevated permissions. Check `status` per config.
- **Local GPU not found**: Check `gpu_index` matches available GPUs (`nvidia-smi`).
- **Remote host unreachable**: Check host is running with `/cuda:service health <host>`.

## Available Fixtures

Test workloads — reference/configs in `data/fixtures/`, generated code in `data/generated/{arch}/`:

| Fixture | Description | Configs |
|---------|-------------|---------|
| `vecadd` | BF16 vector addition | 7 configs (1D/2D/3D shapes) |
| `matmul` | Matrix multiplication (CuTe reference) | varies |
| `fa4` | Flash Attention v4 | varies |

## Reviewing Results

Use `/cuda:inspect` to review compile, evaluate, and profile results from past turns.
```

- [ ] **Step 2: Update plugin.json description**

```json
{
  "name": "cuda",
  "description": "CUDA kernel compilation, evaluation, profiling, and execution — local GPU or remote cuda_exec service",
  "version": "0.2.0",
  "author": {
    "name": "D.Tadpole"
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add plugins/cuda/skills/exec/SKILL.md plugins/cuda/.claude-plugin/plugin.json
git commit -m "docs: update cuda:exec skill for local/remote target parameter

Skill now documents the required target parameter with examples
for both local (gpu_index) and remote (host) modes."
```

---

### Task 6: End-to-End Smoke Test

**Why:** Verify the full chain works — skill invocation through MCP tool through dispatch to actual execution.

**Files:** (no new files — manual verification)

- [ ] **Step 1: Run all quick tests**

Run: `cd /home/centos/kernel_lab && python -m pytest -m quick -v`
Expected: All PASS.

- [ ] **Step 2: Verify MCP server starts without key file**

Run: `cd /home/centos/kernel_lab && CUDA_EXEC_KEY_PATH=/nonexistent/path timeout 3 .venv/bin/python plugins/cuda/mcp_server.py 2>&1 || true`
Expected: Server starts (waits for stdio), does NOT crash with "key file not found".

- [ ] **Step 3: Verify import chain works**

Run: `cd /home/centos/kernel_lab && python -c "from plugins.cuda.dispatch import validate_target, resolve_host_url, dispatch_local, dispatch_remote; print('dispatch OK')"`
Expected: `dispatch OK`

- [ ] **Step 4: Run local GPU test (if GPU available)**

Run: `cd /home/centos/kernel_lab && python -m pytest plugins/cuda/tests/test_exec_tools.py::test_compile_vecadd_local -v -m gpu`
Expected: PASS if local GPU + CUDA toolkit available, SKIP otherwise.

- [ ] **Step 5: Final commit and push**

```bash
git push
```
