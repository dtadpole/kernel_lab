"""FastMCP stdio server wrapping the cuda_exec HTTP API as MCP tools.

Exposes 5 action tools (compile, evaluate, profile, execute, read_file)
that proxy to the cuda_exec HTTP service, 4 data retrieval tools
(get_compile_data, get_evaluate_data, get_profile_data, get_data_point)
that read from a local data store of raw request/response JSON, and
2 document search tools (search_docs, lookup_doc_section) for querying
indexed NVIDIA CUDA Toolkit documentation.

Every action tool call (except read_file) is persisted to the local
data store before response compaction, so the full unmodified data
is always available for later retrieval.

Configuration (environment variables):
    CUDA_EXEC_URL                       Base URL of the cuda_exec service.
                                        Default: http://127.0.0.1:8000
    CUDA_EXEC_KEY_PATH                  Path to the bearer token key file.
                                        Default: ~/.keys/cuda_exec.key
    CUDA_AGENT_MCP_REQUEST_TIMEOUT      Overall HTTP request timeout (seconds).
                                        Default: 300.0
    CUDA_AGENT_MCP_CONNECT_TIMEOUT      TCP connect timeout (seconds).
                                        Default: 10.0
    CUDA_AGENT_MCP_MAX_CONTENT_CHARS    Max chars for inline content truncation.
                                        Default: 4000
    CUDA_AGENT_MCP_TOOL_TIMEOUT         Default timeout per tool call (seconds).
                                        Default: 180
    CUDA_AGENT_DATA_DIR                 Local directory for raw data store.
                                        Set by agent.py; if unset, data
                                        storage and retrieval are disabled.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_URL = os.environ.get("CUDA_EXEC_URL", "http://127.0.0.1:8000")
_DEFAULT_KEY_PATH = Path.home() / ".keys" / "cuda_exec.key"


def _load_bearer_token() -> str:
    key_path = Path(os.environ.get("CUDA_EXEC_KEY_PATH") or str(_DEFAULT_KEY_PATH))
    try:
        token = key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"cuda_agent mcp_server: key file not found: {key_path}", file=sys.stderr)
        raise SystemExit(1)
    except OSError as exc:
        print(f"cuda_agent mcp_server: cannot read key file {key_path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if not token:
        print(f"cuda_agent mcp_server: key file is empty: {key_path}", file=sys.stderr)
        raise SystemExit(1)
    return token


_BEARER_TOKEN: str = _load_bearer_token()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIMEOUT = httpx.Timeout(
    float(os.environ.get("CUDA_AGENT_MCP_REQUEST_TIMEOUT", "300.0")),
    connect=float(os.environ.get("CUDA_AGENT_MCP_CONNECT_TIMEOUT", "10.0")),
)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_BEARER_TOKEN}"}


_MAX_CONTENT_CHARS = int(os.environ.get("CUDA_AGENT_MCP_MAX_CONTENT_CHARS", "4000"))
_DEFAULT_TOOL_TIMEOUT = int(os.environ.get("CUDA_AGENT_MCP_TOOL_TIMEOUT", "180"))

# ---------------------------------------------------------------------------
# Data store — persist raw request/response for every tool call
# ---------------------------------------------------------------------------

_DATA_DIR: str | None = os.environ.get("CUDA_AGENT_DATA_DIR")

_ENDPOINT_STAGE: dict[str, str] = {
    "/compile": "compile",
    "/evaluate": "evaluate",
    "/profile": "profile",
    "/execute": "execute",
}

# In-memory attempt counters: {(turn, stage): count}
_attempt_counters: dict[tuple[int, str], int] = {}


def _save_data_point(
    endpoint: str,
    request_body: dict[str, Any],
    response_data: Any,
) -> None:
    """Persist raw request + response to the local data store."""
    if not _DATA_DIR:
        return
    stage = _ENDPOINT_STAGE.get(endpoint)
    if stage is None:
        return
    metadata = request_body.get("metadata", {})
    turn = metadata.get("turn")
    if turn is None:
        return

    key = (int(turn), stage)
    _attempt_counters[key] = _attempt_counters.get(key, 0) + 1
    attempt = _attempt_counters[key]

    turn_dir = Path(_DATA_DIR) / f"turn_{turn}"
    turn_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{stage}.attempt_{attempt:03d}"
    (turn_dir / f"{prefix}.request.json").write_text(
        json.dumps(request_body, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (turn_dir / f"{prefix}.response.json").write_text(
        json.dumps(response_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _list_available(turn_dir: Path) -> list[str]:
    """List available data point prefixes in a turn directory."""
    if not turn_dir.is_dir():
        return []
    seen: set[str] = set()
    for f in sorted(turn_dir.glob("*.json")):
        # Strip .request.json / .response.json suffix
        name = f.name
        for suffix in (".request.json", ".response.json"):
            if name.endswith(suffix):
                seen.add(name[: -len(suffix)])
                break
    return sorted(seen)


# Fields to keep in evaluate/profile per-config outputs.  Everything else
# (reference, generated, artifacts, logs) can be huge and is available via
# cuda_read_file on demand.
_EVAL_CONFIG_KEEP = {"status", "correctness", "performance"}
_PROFILE_CONFIG_KEEP = {"status", "summary"}


def _compact_response(obj: Any, *, _endpoint: str = "") -> Any:
    """Shrink response payloads to stay within MCP message size limits.

    - Replaces base64-encoded binary with a placeholder.
    - Truncates large inline text content fields.
    - For evaluate/profile config outputs, keeps only the structured
      summaries and strips raw side payloads, artifacts, and logs.
    """

    if isinstance(obj, dict):
        # Strip base64 binary
        if obj.get("encoding") == "base64" and "content" in obj:
            return {**obj, "content": "<base64-binary-omitted>"}

        # Truncate large inline text content
        if obj.get("inline") and isinstance(obj.get("content"), str):
            content = obj["content"]
            if len(content) > _MAX_CONTENT_CHARS:
                return {
                    **obj,
                    "content": content[:_MAX_CONTENT_CHARS] + f"\n... [truncated, {len(content)} chars total]",
                    "truncated": True,
                }

        # Compact per-config outputs in evaluate/profile responses.
        # Keep only the structured summaries; the agent can use
        # cuda_read_file to fetch full details on demand.
        if "status" in obj and "correctness" in obj:
            # Evaluate config output
            return {k: v for k, v in obj.items() if k in _EVAL_CONFIG_KEEP}
        if "status" in obj and "summary" in obj and "correctness" not in obj:
            # Profile config output
            return {k: v for k, v in obj.items() if k in _PROFILE_CONFIG_KEEP}

        return {k: _compact_response(v, _endpoint=_endpoint) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_compact_response(item, _endpoint=_endpoint) for item in obj]
    return obj


async def _post(endpoint: str, body: dict[str, Any]) -> str:
    """POST to a cuda_exec endpoint and return the response as JSON text."""

    async with httpx.AsyncClient(base_url=_BASE_URL, timeout=_TIMEOUT) as client:
        resp = await client.post(endpoint, json=body, headers=_auth_headers())
        resp.raise_for_status()
        raw = resp.json()
        _save_data_point(endpoint, body, raw)
        data = _compact_response(raw)
        return json.dumps(data, indent=2)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("cuda_toolkit")


@mcp.tool()
async def cuda_compile(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    reference_files: Annotated[dict[str, str], Field(description="Map of relative_path -> file content for reference source inputs")],
    generated_files: Annotated[dict[str, str], Field(description="Map of relative_path -> file content; must contain exactly one .cu file")],
    timeout_seconds: Annotated[int, Field(description="Max seconds for compile")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """Compile CUDA source files and produce a binary, PTX, and SASS.

    Compiles the generated .cu file against the reference Python inputs
    using the nvcc -> ptxas -> cuobjdump -> nvdisasm toolchain.

    Workflow constraints:
    - Must be called exactly once per turn before cuda_evaluate or cuda_profile.
    - New or modified source code requires a new turn (increment metadata.turn).
    - Old turns are immutable — do not recompile on a previous turn number.

    Parameters:
        metadata:        Turn identity dict with keys: run_tag, version,
                         direction_id, direction_slug, turn.
        reference_files: {relative_path: content} map of reference Python
                         source files (e.g. the nn.Module implementation).
        generated_files: {relative_path: content} map containing exactly one
                         .cu file to compile.
        timeout_seconds: Max wall-clock seconds for the compile step.

    Returns JSON with:
        all_ok:       bool — whether compilation succeeded.
        attempt:      int — always 1 (compile runs once per turn).
        artifacts:    Compile outputs (binary, ptx, cubin, resource_usage, sass).
                      Binary/cubin content is base64-encoded and replaced with
                      a placeholder by the MCP server to save context.
        tool_outputs: Inline stdout/stderr from each compile tool stage.
    """

    return await _post("/compile", {
        "metadata": metadata,
        "reference_files": reference_files,
        "generated_files": generated_files,
        "timeout_seconds": timeout_seconds,
    })


@mcp.tool()
async def cuda_evaluate(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    configs: Annotated[dict[str, dict[str, Any]], Field(description="Slug-keyed runtime config payloads, e.g. {'tensor2d-1024x1024': {shape: [1024,1024]}}")],
    timeout_seconds: Annotated[int, Field(description="Max seconds for evaluate")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """Evaluate a compiled CUDA kernel for correctness and performance.

    Runs the compiled binary and the reference Python module side by side
    for each config, then compares outputs numerically.

    Workflow constraints:
    - Requires a successful cuda_compile on the same turn.
    - Can be called multiple times on the same turn (same compiled binary).
    - One compile fans out to many configs.

    Parameters:
        metadata:        Turn identity dict (must match the compile turn).
        configs:         {config_slug: config_body} map. Each config_slug is
                         a stable identifier; config_body is kernel-specific
                         (e.g. {"shape": [1024, 1024]}).
        timeout_seconds: Max wall-clock seconds for the full evaluate pass.

    Returns JSON with:
        all_ok:  bool — true only if every config passed correctness.
        configs: {config_slug: output} where each output contains:
            status:      "ok" | "error"
            correctness: {passed, max_abs_error, mean_abs_error, max_rel_error, ...}
            performance: {latency_ms: {min, median, max, mean}, runs, comparison}

    Note: The MCP server strips large fields (reference, generated, artifacts,
    logs) from each config output to stay within context limits. Use
    cuda_read_file to fetch full details on demand.
    """

    return await _post("/evaluate", {
        "metadata": metadata,
        "configs": configs,
        "timeout_seconds": timeout_seconds,
    })


@mcp.tool()
async def cuda_profile(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    configs: Annotated[dict[str, dict[str, Any]], Field(description="Slug-keyed runtime config payloads")],
    side: Annotated[Literal["generated", "reference"], Field(description="Which side to NCU-profile")] = "generated",
    timeout_seconds: Annotated[int, Field(description="Max seconds for profile")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """NCU-profile a compiled CUDA kernel or reference Python/CuTe DSL kernel.

    Runs Nsight Compute (ncu) with ``--set detailed`` to collect hardware-level
    GPU metrics: roofline, pipe utilization, memory throughput, warp stalls, etc.

    Workflow constraints:
    - Requires a successful cuda_compile on the same turn (stages inputs).
    - For ``side="reference"``, NCU filters by kernel name regex to skip
      PyTorch JIT overhead kernels and capture only the CuTe DSL kernel.

    Parameters:
        metadata:        Turn identity dict (must match the compile turn).
        configs:         {config_slug: config_body} map (same as evaluate).
        side:            "generated" — profile the compiled CUDA binary.
                         "reference" — profile the reference Python/CuTe DSL kernel.
        timeout_seconds: Max wall-clock seconds for the full profile pass.

    Returns JSON with:
        all_ok:  bool — true if profiling succeeded for all configs.
        configs: {config_slug: output} where each output contains:
            status:   "ok" | "error"
            summary:  {side, ncu_profiled, ncu_report_exists, ncu_report_path, ...}

    Note: The MCP server strips large fields (artifacts, logs) from each config
    output. Use cuda_read_file to fetch the .ncu-rep binary report on demand.
    """

    return await _post("/profile", {
        "metadata": metadata,
        "configs": configs,
        "side": side,
        "timeout_seconds": timeout_seconds,
    })


@mcp.tool()
async def cuda_execute(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    command: Annotated[list[str], Field(description="Executable plus arguments, e.g. ['/usr/local/cuda/bin/nvcc', '--version']")],
    env: Annotated[dict[str, str], Field(description="Extra environment variables")] = {},
    timeout_seconds: Annotated[int, Field(description="Max seconds for execute")] = _DEFAULT_TOOL_TIMEOUT,
) -> str:
    """Run an ad-hoc CUDA tool command on the remote service.

    A general-purpose escape hatch for commands that don't fit the
    compile/evaluate/profile workflow (e.g. querying device info,
    checking toolkit versions, or running custom inspection tools).

    Parameters:
        metadata:        Turn identity dict.
        command:         Argv list — executable path plus arguments.
                         Example: ["/usr/local/cuda/bin/nvcc", "--version"]
        env:             Extra environment variables for the subprocess.
        timeout_seconds: Max wall-clock seconds for the command.

    Returns JSON with:
        all_ok:  bool — true if the command exited successfully.
        attempt: int — attempt number.
        logs:    {relative_path: file_payload} — stdout, stderr, and
                 combined log files from the command execution.
    """

    return await _post("/execute", {
        "metadata": metadata,
        "command": command,
        "env": env,
        "timeout_seconds": timeout_seconds,
    })


@mcp.tool()
async def cuda_read_file(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    path: Annotated[str, Field(description="Relative path under the turn root (must start with artifacts/, logs/, or state/)")],
    max_bytes: Annotated[int | None, Field(description="Optional max bytes to read")] = None,
) -> str:
    """Read a file from a specific turn's directory tree.

    Provides on-demand access to any file produced during a turn.  Since
    cuda_evaluate and cuda_profile responses are compacted by the MCP
    server (large fields stripped), this tool is the way to retrieve
    full details when needed.

    The path must be relative to the turn root and start with one of:
    - artifacts/  — compile outputs (PTX, SASS, binaries), profile reports (.ncu-rep)
    - logs/       — stdout/stderr from compile, evaluate, profile, execute stages
    - state/      — workflow manifests and per-config state records

    Example paths:
        artifacts/compile.attempt_001.vector_add.ptx
        logs/compile.attempt_001.nvcc-ptx.log
        logs/evaluate.attempt_001.config_size_1024.log
        state/compile.attempt_001.json

    Parameters:
        metadata:  Turn identity dict (identifies which turn's files to read).
        path:      Relative path under the turn root.
        max_bytes: Optional cap on how many bytes to return (for large files).

    Returns JSON with:
        metadata: Echoed turn identity.
        file:     {path, inline, content, encoding, truncated} — the file
                  payload. Text files use utf8 encoding; binary files use base64.
    """

    body: dict[str, Any] = {"metadata": metadata, "path": path}
    if max_bytes is not None:
        body["max_bytes"] = max_bytes
    return await _post("/files/read", body)


@mcp.tool()
async def cuda_get_data_point(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    stage: Annotated[Literal["compile", "evaluate", "profile", "execute"], Field(description="Which stage to retrieve")],
    attempt: Annotated[int, Field(description="Attempt number (1-based)")] = 1,
    side: Annotated[Literal["request", "response", "both"], Field(description="Which side(s) to return")] = "both",
) -> str:
    """Retrieve the raw input/output from a prior tool call.

    Every cuda_compile, cuda_evaluate, cuda_profile, and cuda_execute
    call saves its full (uncompacted) request and response to a local
    data store.  This tool reads those saved files so the agent can
    re-examine past results without re-calling cuda_exec.

    Parameters:
        metadata: Turn identity dict (identifies which turn to look up).
        stage:    "compile", "evaluate", "profile", or "execute".
        attempt:  1-based attempt number (default 1).  Within a single
                  turn, each stage tracks its own attempt counter.
        side:     "request" — return only the saved request body.
                  "response" — return only the saved response.
                  "both" — return {"request": ..., "response": ...}.

    Returns JSON with the raw (uncompacted) data.  If the requested
    data point does not exist, returns an error listing available
    data points for that turn.
    """

    if not _DATA_DIR:
        return json.dumps({"error": "Data store not configured (CUDA_AGENT_DATA_DIR not set)."})

    turn = metadata.get("turn")
    if turn is None:
        return json.dumps({"error": "metadata.turn is required."})

    turn_dir = Path(_DATA_DIR) / f"turn_{turn}"
    prefix = f"{stage}.attempt_{attempt:03d}"

    def _read_side(s: str) -> dict[str, Any] | None:
        fp = turn_dir / f"{prefix}.{s}.json"
        if fp.is_file():
            return json.loads(fp.read_text(encoding="utf-8"))
        return None

    if side == "both":
        req = _read_side("request")
        resp = _read_side("response")
        if req is None and resp is None:
            available = _list_available(turn_dir)
            return json.dumps({
                "error": f"No data point found for {prefix} in turn {turn}.",
                "available": available,
            }, indent=2)
        result: dict[str, Any] = {}
        if req is not None:
            result["request"] = req
        if resp is not None:
            result["response"] = _compact_response(resp)
        return json.dumps(result, indent=2)

    data = _read_side(side)
    if data is None:
        available = _list_available(turn_dir)
        return json.dumps({
            "error": f"No {side} found for {prefix} in turn {turn}.",
            "available": available,
        }, indent=2)
    if side == "response":
        data = _compact_response(data)
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Typed data retrieval helpers
# ---------------------------------------------------------------------------


def _read_stored_response(turn: int, stage: str, attempt: int) -> dict[str, Any] | None:
    """Read a stored response JSON from the data store.  Returns None if missing."""
    if not _DATA_DIR:
        return None
    fp = Path(_DATA_DIR) / f"turn_{turn}" / f"{stage}.attempt_{attempt:03d}.response.json"
    if fp.is_file():
        return json.loads(fp.read_text(encoding="utf-8"))
    return None


def _not_found_error(turn: int, stage: str, attempt: int) -> str:
    """Standard error response when a data point is not found."""
    turn_dir = Path(_DATA_DIR or "") / f"turn_{turn}"
    return json.dumps({
        "error": f"No {stage}.attempt_{attempt:03d} response found in turn {turn}.",
        "available": _list_available(turn_dir),
    }, indent=2)


def _data_store_not_configured() -> str:
    return json.dumps({"error": "Data store not configured (CUDA_AGENT_DATA_DIR not set)."})


# ---------------------------------------------------------------------------
# Typed per-stage retrieval tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def cuda_get_compile_data(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    attempt: Annotated[int, Field(description="Attempt number (1-based)")] = 1,
    field: Annotated[
        Literal["all", "ptx", "sass", "resource_usage", "tool_outputs"],
        Field(description="Which part of the compile result to return"),
    ] = "all",
) -> str:
    """Retrieve structured compile results from a prior turn.

    Reads the saved raw compile response and extracts the requested
    structured field.

    Parameters:
        metadata: Turn identity dict (identifies which turn to look up).
        attempt:  1-based attempt number (default 1).
        field:    "all"            — full compile result (compacted).
                  "ptx"            — PTX assembly text.
                  "sass"           — SASS disassembly text.
                  "resource_usage" — register/shared-memory usage text.
                  "tool_outputs"   — stdout/stderr from each compile stage
                                     (nvcc, ptxas, cuobjdump, nvdisasm).

    Returns JSON with {all_ok, field_name: ...}.  For text fields (ptx,
    sass, resource_usage), the content string is returned directly.
    """

    if not _DATA_DIR:
        return _data_store_not_configured()
    turn = metadata.get("turn")
    if turn is None:
        return json.dumps({"error": "metadata.turn is required."})

    data = _read_stored_response(int(turn), "compile", attempt)
    if data is None:
        return _not_found_error(int(turn), "compile", attempt)

    all_ok = data.get("all_ok")

    if field == "all":
        return json.dumps(_compact_response(data), indent=2)

    if field in ("ptx", "sass", "resource_usage"):
        artifact = data.get("artifacts", {}).get(field, {})
        content = artifact.get("content", "")
        return json.dumps({"all_ok": all_ok, field: content}, indent=2)

    if field == "tool_outputs":
        raw_outputs = data.get("tool_outputs", {})
        # Extract just the text content from each tool output
        outputs: dict[str, str] = {}
        for name, payload in raw_outputs.items():
            if isinstance(payload, dict):
                outputs[name] = payload.get("content", "")
            else:
                outputs[name] = str(payload)
        return json.dumps({"all_ok": all_ok, "tool_outputs": outputs}, indent=2)

    return json.dumps({"error": f"Unknown field: {field}"})


@mcp.tool()
async def cuda_get_evaluate_data(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    attempt: Annotated[int, Field(description="Attempt number (1-based)")] = 1,
    config_slug: Annotated[str | None, Field(description="Filter to a single config slug (omit for all configs)")] = None,
    field: Annotated[
        Literal["all", "correctness", "performance"],
        Field(description="Which part of the evaluate result to return"),
    ] = "all",
) -> str:
    """Retrieve structured evaluate results from a prior turn.

    Reads the saved raw evaluate response and extracts correctness
    and/or performance data, optionally filtered to a single config.

    Parameters:
        metadata:    Turn identity dict (identifies which turn to look up).
        attempt:     1-based attempt number (default 1).
        config_slug: If provided, return data only for this config.
                     If omitted, return data for all configs.
        field:       "all"         — correctness + performance per config.
                     "correctness" — only correctness metrics per config.
                     "performance" — only performance metrics per config.

    Returns JSON with:
        all_ok:  bool — aggregate success.
        configs: {config_slug: {status, correctness?, performance?}}

    Available configs are listed in the error message if config_slug
    does not match.
    """

    if not _DATA_DIR:
        return _data_store_not_configured()
    turn = metadata.get("turn")
    if turn is None:
        return json.dumps({"error": "metadata.turn is required."})

    data = _read_stored_response(int(turn), "evaluate", attempt)
    if data is None:
        return _not_found_error(int(turn), "evaluate", attempt)

    all_ok = data.get("all_ok")
    configs = data.get("configs", {})

    if config_slug is not None:
        if config_slug not in configs:
            return json.dumps({
                "error": f"Config '{config_slug}' not found.",
                "available_configs": sorted(configs.keys()),
            }, indent=2)
        configs = {config_slug: configs[config_slug]}

    keep_keys = {"status"}
    if field in ("all", "correctness"):
        keep_keys.add("correctness")
    if field in ("all", "performance"):
        keep_keys.add("performance")

    result_configs = {
        slug: {k: v for k, v in cfg.items() if k in keep_keys}
        for slug, cfg in configs.items()
    }

    return json.dumps({"all_ok": all_ok, "configs": result_configs}, indent=2)


@mcp.tool()
async def cuda_get_profile_data(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    attempt: Annotated[int, Field(description="Attempt number (1-based)")] = 1,
    config_slug: Annotated[str | None, Field(description="Filter to a single config slug (omit for all configs)")] = None,
    field: Annotated[
        Literal["all", "summary"],
        Field(description="Which part of the profile result to return"),
    ] = "all",
) -> str:
    """Retrieve structured profile results from a prior turn.

    Reads the saved raw profile response and extracts the NCU
    summary, optionally filtered to a single config.

    Parameters:
        metadata:    Turn identity dict (identifies which turn to look up).
        attempt:     1-based attempt number (default 1).
        config_slug: If provided, return data only for this config.
                     If omitted, return data for all configs.
        field:       "all"     — status + summary per config.
                     "summary" — summary only per config.

    Returns JSON with:
        all_ok:  bool — aggregate success.
        configs: {config_slug: {status, summary?}}

    Available configs are listed in the error message if config_slug
    does not match.
    """

    if not _DATA_DIR:
        return _data_store_not_configured()
    turn = metadata.get("turn")
    if turn is None:
        return json.dumps({"error": "metadata.turn is required."})

    data = _read_stored_response(int(turn), "profile", attempt)
    if data is None:
        return _not_found_error(int(turn), "profile", attempt)

    all_ok = data.get("all_ok")
    configs = data.get("configs", {})

    if config_slug is not None:
        if config_slug not in configs:
            return json.dumps({
                "error": f"Config '{config_slug}' not found.",
                "available_configs": sorted(configs.keys()),
            }, indent=2)
        configs = {config_slug: configs[config_slug]}

    keep_keys = {"status"}
    if field in ("all", "summary"):
        keep_keys.add("summary")

    result_configs = {
        slug: {k: v for k, v in cfg.items() if k in keep_keys}
        for slug, cfg in configs.items()
    }

    return json.dumps({"all_ok": all_ok, "configs": result_configs}, indent=2)


# ---------------------------------------------------------------------------
# Document retrieval tools
# ---------------------------------------------------------------------------

_doc_searcher = None


def _get_doc_searcher():
    """Lazy-load the document searcher singleton."""
    global _doc_searcher
    if _doc_searcher is None:
        try:
            from doc_retrieval.searcher import DocSearcher

            _doc_searcher = DocSearcher()
        except Exception as exc:
            return None, str(exc)
    return _doc_searcher, None


@mcp.tool()
async def cuda_search_docs(
    query: Annotated[
        str,
        Field(description="Natural language search query about CUDA programming"),
    ],
    mode: Annotated[
        Literal["bm25", "dense", "hybrid"],
        Field(description="Search mode: bm25 (keyword), dense (semantic), hybrid (combined)"),
    ] = "hybrid",
    top_k: Annotated[
        int,
        Field(description="Number of results to return (1-20)", ge=1, le=20),
    ] = 5,
) -> str:
    """Search NVIDIA CUDA Toolkit documentation.

    Searches the indexed CUDA documentation corpus using the specified
    retrieval mode.  Useful for looking up API details, best practices,
    optimization techniques, PTX ISA specifics, or memory model semantics.

    Examples of good queries:
        - "shared memory bank conflicts"
        - "warp divergence impact on performance"
        - "atomicCAS signature and semantics"
        - "cudaMemcpyAsync stream synchronization"
        - "PTX ld.global instruction"
    """
    searcher, err = _get_doc_searcher()
    if searcher is None:
        return json.dumps({"error": f"Doc searcher unavailable: {err}"})

    if mode == "bm25":
        results = searcher.search_bm25(query, top_k)
    elif mode == "dense":
        results = searcher.search_dense(query, top_k)
    else:
        results = searcher.search_hybrid(query, top_k)

    return json.dumps(
        [
            {
                "title": r.title,
                "section_path": r.section_path,
                "url": r.source_url,
                "text": r.text[:2000],  # truncate for context limits
                "score": round(r.score, 4),
            }
            for r in results
        ],
        indent=2,
    )


@mcp.tool()
async def cuda_lookup_doc_section(
    url: Annotated[
        str,
        Field(description="URL of the CUDA documentation page (from a prior search result)"),
    ],
    section: Annotated[
        str | None,
        Field(description="Section heading to filter to (optional)"),
    ] = None,
) -> str:
    """Retrieve full text from a specific CUDA documentation page or section.

    Given a URL from a prior cuda_search_docs result, retrieves all
    chunks from that page.  Optionally filter to a specific section.
    Use this to get deeper context after identifying a relevant page.
    """
    searcher, err = _get_doc_searcher()
    if searcher is None:
        return json.dumps({"error": f"Doc searcher unavailable: {err}"})

    chunks = searcher._load_chunks()
    matching = [c for c in chunks if c["source_url"] == url]

    if section:
        section_lower = section.lower()
        matching = [
            c for c in matching
            if section_lower in c["section_path"].lower()
        ]

    if not matching:
        return json.dumps({"error": f"No chunks found for url={url}, section={section}"})

    # Concatenate chunk texts in order
    matching.sort(key=lambda c: c["chunk_index"])
    text = "\n\n".join(c["text"] for c in matching)

    # Truncate to ~8000 chars to stay within context limits
    if len(text) > 8000:
        text = text[:8000] + "\n\n[... truncated ...]"

    return json.dumps(
        {
            "url": url,
            "section": section,
            "num_chunks": len(matching),
            "text": text,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
