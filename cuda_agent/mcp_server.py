"""FastMCP stdio server wrapping the cuda_exec HTTP API as MCP tools.

Each tool maps 1:1 to a cuda_exec endpoint.  The server makes HTTP
requests to the running cuda_exec FastAPI service and returns the JSON
response as text so the calling agent can parse it directly.

Configuration (environment variables):
    CUDA_EXEC_URL      Base URL of the cuda_exec service.
                       Default: http://127.0.0.1:8000
    CUDA_EXEC_KEY_PATH Path to the bearer token key file.
                       Default: ~/.keys/cuda_exec.key
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

_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_BEARER_TOKEN}"}


_MAX_CONTENT_CHARS = 4000

# Fields to keep in evaluate/profile per-config outputs.  Everything else
# (reference, generated, artifacts, logs) can be huge and is available via
# cuda_read_file on demand.
_EVAL_CONFIG_KEEP = {"status", "correctness", "performance"}
_PROFILE_CONFIG_KEEP = {"status", "summary", "reference_summary", "generated_summary"}


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
        if "status" in obj and "summary" in obj and "reference_summary" in obj:
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
        data = _compact_response(resp.json())
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
    timeout_seconds: Annotated[int, Field(description="Max seconds for compile")] = 180,
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
    timeout_seconds: Annotated[int, Field(description="Max seconds for evaluate")] = 180,
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
    mode: Annotated[Literal["generated_only", "reference_only", "dual"], Field(description="Which side(s) to profile")] = "generated_only",
    profiler_backend: Annotated[Literal["comparison_runtime", "ncu"], Field(description="Profile backend; ncu is generated_only only")] = "comparison_runtime",
    timeout_seconds: Annotated[int, Field(description="Max seconds for profile")] = 180,
) -> str:
    """Profile a compiled CUDA kernel to collect detailed latency data.

    Measures execution latency for the generated kernel, the reference
    Python module, or both, depending on the selected mode.

    Workflow constraints:
    - Requires a successful cuda_compile on the same turn.
    - The ncu backend only supports generated_only mode.
    - comparison_runtime backend supports all three modes.

    Parameters:
        metadata:          Turn identity dict (must match the compile turn).
        configs:           {config_slug: config_body} map (same as evaluate).
        mode:              "generated_only" — profile the compiled CUDA kernel.
                           "reference_only" — profile the reference Python module.
                           "dual" — profile both sides for direct comparison.
        profiler_backend:  "comparison_runtime" — built-in timing comparison.
                           "ncu" — NVIDIA Nsight Compute (generated_only only).
        timeout_seconds:   Max wall-clock seconds for the full profile pass.

    Returns JSON with:
        all_ok:  bool — true if profiling succeeded for all configs.
        configs: {config_slug: output} where each output contains:
            status:            "ok" | "error"
            summary:           {latency_ms: {min, median, max, mean}, runs}
            reference_summary: Side-level summary for reference (if applicable).
            generated_summary: Side-level summary for generated kernel.

    Note: The MCP server strips large fields (reference, generated, artifacts,
    logs) from each config output. Use cuda_read_file to fetch full profiling
    reports (e.g. .ncu-rep files) on demand.
    """

    return await _post("/profile", {
        "metadata": metadata,
        "configs": configs,
        "mode": mode,
        "profiler_backend": profiler_backend,
        "timeout_seconds": timeout_seconds,
    })


@mcp.tool()
async def cuda_execute(
    metadata: Annotated[dict[str, Any], Field(description="Turn identity: {run_tag, version, direction_id, direction_slug, turn}")],
    command: Annotated[list[str], Field(description="Executable plus arguments, e.g. ['/usr/local/cuda/bin/nvcc', '--version']")],
    env: Annotated[dict[str, str], Field(description="Extra environment variables")] = {},
    timeout_seconds: Annotated[int, Field(description="Max seconds for execute")] = 180,
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
