"""Dispatch module for cuda MCP plugin — target validation, host resolution,
local dispatch (direct function call), and remote dispatch (HTTP proxy).

Public API:
    validate_target(target)       — Validate a target dict; raises ValueError.
    resolve_host_url(host_name)   — Look up host in conf/hosts/default.yaml.
    dispatch_local(endpoint, body, *, gpu_index)  — Call endpoint handler directly.
    dispatch_remote(endpoint, body, *, host)       — HTTP POST to remote service.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOSTS_YAML = _REPO_ROOT / "conf" / "hosts" / "default.yaml"

# ---------------------------------------------------------------------------
# Lazy caches (module-level globals)
# ---------------------------------------------------------------------------

_hosts_config: dict[str, Any] | None = None
_bearer_token: str | None = None

# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------


def validate_target(target: dict[str, Any]) -> None:
    """Validate a target dict.  Raises ``ValueError`` on invalid input.

    Local target:  ``{"mode": "local", "gpu_index": <non-negative int>}``
    Remote target: ``{"mode": "remote", "host": "<non-empty string>"}``
    """
    mode = target.get("mode")
    if mode not in ("local", "remote"):
        raise ValueError(
            f"target.mode must be 'local' or 'remote', got {mode!r}"
        )

    if mode == "local":
        gpu_index = target.get("gpu_index")
        if gpu_index is None or not isinstance(gpu_index, int) or gpu_index < 0:
            raise ValueError(
                f"target.gpu_index must be a non-negative integer, got {gpu_index!r}"
            )

    if mode == "remote":
        host = target.get("host")
        if not host or not isinstance(host, str):
            raise ValueError(
                f"target.host must be a non-empty string, got {host!r}"
            )


# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------


def _load_hosts_config() -> dict[str, Any]:
    """Load and cache conf/hosts/default.yaml."""
    global _hosts_config
    if _hosts_config is None:
        with _HOSTS_YAML.open() as f:
            _hosts_config = yaml.safe_load(f)
    return _hosts_config


def resolve_host_url(host_name: str) -> str:
    """Resolve a host name to an HTTP base URL.

    Looks up *host_name* in ``conf/hosts/default.yaml``.  If the host entry
    has an ``internet_host`` field, uses that; otherwise falls back to
    ``ssh_host``.  Constructs ``http://{host}:{port}``.

    Raises ``ValueError`` if *host_name* is not found.
    """
    config = _load_hosts_config()
    hosts = config.get("hosts", {})
    entry = hosts.get(host_name)
    if entry is None:
        available = sorted(hosts.keys())
        raise ValueError(
            f"Host {host_name!r} not found in {_HOSTS_YAML}. "
            f"Available hosts: {available}"
        )
    hostname = entry.get("internet_host") or entry["ssh_host"]
    port = entry["port"]
    return f"http://{hostname}:{port}"


# ---------------------------------------------------------------------------
# Bearer token loading (for remote dispatch)
# ---------------------------------------------------------------------------


def _load_bearer_token() -> str:
    """Load and cache the bearer token for remote dispatch."""
    global _bearer_token
    if _bearer_token is None:
        key_path = Path(
            os.environ.get("CUDA_EXEC_KEY_PATH")
            or str(Path.home() / ".keys" / "cuda_exec.key")
        )
        _bearer_token = key_path.read_text(encoding="utf-8").strip()
    return _bearer_token


# ---------------------------------------------------------------------------
# HTTP client config (lazy — httpx may not be installed in all envs)
# ---------------------------------------------------------------------------

_httpx_timeout: Any = None


def _get_timeout() -> Any:
    """Build and cache the httpx.Timeout on first use."""
    global _httpx_timeout
    if _httpx_timeout is None:
        import httpx

        _httpx_timeout = httpx.Timeout(
            float(os.environ.get("CUDA_AGENT_MCP_REQUEST_TIMEOUT", "300.0")),
            connect=float(os.environ.get("CUDA_AGENT_MCP_CONNECT_TIMEOUT", "10.0")),
        )
    return _httpx_timeout


# ---------------------------------------------------------------------------
# Remote dispatch
# ---------------------------------------------------------------------------


def dispatch_remote(
    endpoint: str,
    body: dict[str, Any],
    *,
    host: str,
) -> dict[str, Any]:
    """POST to a remote cuda_exec service and return the response dict.

    Resolves *host* to a base URL, loads the bearer token, and sends an
    HTTP POST with JSON body.  Returns the parsed JSON response.
    """
    import httpx

    base_url = resolve_host_url(host)
    token = _load_bearer_token()
    timeout = _get_timeout()
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        resp = client.post(
            endpoint,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Local dispatch
# ---------------------------------------------------------------------------

# Endpoint → (RequestModelName, handler_function_name)
_ENDPOINT_MAP: dict[str, tuple[str, str]] = {
    "/compile":    ("CompileRequest",   "compile_endpoint"),
    "/evaluate":   ("EvaluateRequest",  "evaluate_endpoint"),
    "/profile":    ("ProfileRequest",   "profile_endpoint"),
    "/execute":    ("ExecuteRequest",   "execute_endpoint"),
    "/files/read": ("FileReadRequest",  "file_read_endpoint"),
}

# Lazy references — populated on first call
_models_module: Any = None
_main_module: Any = None


def _ensure_local_imports() -> None:
    """Lazily import cuda_exec.models and cuda_exec.main."""
    global _models_module, _main_module
    if _models_module is None:
        import cuda_exec.models as m
        _models_module = m
    if _main_module is None:
        import cuda_exec.main as main
        _main_module = main


def dispatch_local(
    endpoint: str,
    body: dict[str, Any],
    *,
    gpu_index: int,
) -> dict[str, Any]:
    """Call a cuda_exec endpoint handler directly (no HTTP).

    Sets ``CUDA_VISIBLE_DEVICES`` to *gpu_index*, constructs the Pydantic
    request model from *body*, calls the handler, and returns the response
    as a plain dict.
    """
    if endpoint not in _ENDPOINT_MAP:
        raise ValueError(
            f"Unknown endpoint {endpoint!r}. "
            f"Valid endpoints: {sorted(_ENDPOINT_MAP.keys())}"
        )

    _ensure_local_imports()

    model_name, handler_name = _ENDPOINT_MAP[endpoint]
    request_cls = getattr(_models_module, model_name)
    handler = getattr(_main_module, handler_name)

    request = request_cls(**body)

    # Set CUDA_VISIBLE_DEVICES for the duration of the call
    old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    try:
        response = handler(request)
    finally:
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value

    return response.model_dump()
