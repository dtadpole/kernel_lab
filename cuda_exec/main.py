"""FastAPI entrypoints for cuda_exec.

Keep this module thin.
- models.py documents the public request/response contract.
- runner.py documents runtime-layout semantics.
- DESIGN.md holds the full design rationale.
"""

from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException

from cuda_exec.models import (
    CompileArtifacts,
    CompileRequest,
    CompileResponse,
    CompileSassArtifacts,
    CompileToolOutputs,
    EvaluateConfigOutput,
    EvaluateRequest,
    EvaluateResponse,
    ExecuteRequest,
    ExecuteResponse,
    FileReadRequest,
    FileReadResponse,
    HealthResponse,
    ProfileConfigOutput,
    ProfileRequest,
    ProfileResponse,
    ToolIOPair,
)
from cuda_exec.runner import capture_turn_file, resolve_workspace_bundle
from cuda_exec.tasks import (
    _validate_relative_path,
    run_compile_task,
    run_evaluate_task,
    run_execute_task,
    run_profile_task,
)

app = FastAPI(title="cuda_exec", version="0.1.0")
SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Lightweight service health check."""

    return HealthResponse(ok=True, service="cuda_exec")


def _attempt_tag(attempt: int) -> str:
    return f"attempt_{attempt:03d}"


def _slugify(value: str) -> str:
    cleaned = SAFE_SLUG_RE.sub("_", value).strip("._-")
    return cleaned or "default"


def _config_suffix(config_slug: str) -> str:
    return f"config_{_slugify(config_slug)}"


def _stage_log_paths(stage: str, attempt: int, config_slug: str | None = None) -> list[str]:
    """Return public log relative paths for a stage attempt."""

    base = f"logs/{stage}.{_attempt_tag(attempt)}"
    if config_slug is not None:
        base += f".{_config_suffix(config_slug)}"
    return [f"{base}.log", f"{base}.stdout", f"{base}.stderr"]


def _compile_artifact_map(result: dict) -> dict[str, str]:
    artifact_map: dict[str, str] = {}
    for artifact in result.get("artifacts", []):
        artifact_id = artifact.get("artifact_id")
        if artifact_id == "compile:primary_binary":
            artifact_map["binary"] = artifact["path"]
        elif artifact_id == "compile:primary_ptx":
            artifact_map["ptx"] = artifact["path"]
        elif artifact_id == "compile:primary_cubin":
            artifact_map["cubin"] = artifact["path"]
        elif artifact_id == "compile:resource_usage":
            artifact_map["resource_usage"] = artifact["path"]
        elif artifact_id == "compile:sass_nvdisasm":
            artifact_map["sass_nvdisasm"] = artifact["path"]
    if not artifact_map:
        raise ValueError("compile result missing public compile artifacts")
    return artifact_map


def _capture_public_file(workspace_path: str, rel_path: str, *, inline: bool, max_bytes: int | None = None) -> dict | None:
    return _capture_public_files(workspace_path, [rel_path], inline=inline, max_bytes=max_bytes).get(rel_path)


def _validate_file_read_scope(rel_path: str) -> None:
    _validate_relative_path(rel_path)
    allowed_prefixes = ("artifacts/", "logs/", "state/")
    if not rel_path.startswith(allowed_prefixes):
        raise HTTPException(
            status_code=400,
            detail="file reads are limited to artifacts/, logs/, and state/ paths under the resolved turn root",
        )


def _profile_report_path(config_output: dict) -> str:
    for artifact in config_output.get("artifacts", []):
        if artifact.get("kind") == "profile_report":
            return artifact["path"]
    raise ValueError("profile config output missing report artifact")


def _capture_public_files(workspace_path: str, rel_paths: list[str], *, inline: bool, max_bytes: int | None = None) -> dict[str, dict]:
    """Materialize public response files as relative_path -> FilePayload."""

    payload: dict[str, dict] = {}
    for rel_path in rel_paths:
        item = capture_turn_file(rel_path, workspace_path, max_bytes=max_bytes)
        if not item.get("exists") or item.get("error"):
            continue
        entry = {
            "path": rel_path,
            "inline": inline,
            "truncated": bool(item.get("truncated", False)),
        }
        if inline:
            entry["content"] = item.get("content") or ""
            entry["encoding"] = item.get("encoding") or "utf8"
        payload[rel_path] = entry
    return payload


@app.post("/compile", response_model=CompileResponse)
def compile_endpoint(request: CompileRequest) -> CompileResponse:
    """Compile inline code inputs into kept compile artifacts plus logs."""

    result = run_compile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        reference_files=request.reference_files,
        generated_files=request.generated_files,
    )
    attempt = result["attempt"]
    artifact_map = _compile_artifact_map(result)
    workspace_path = result["workspace_path"]
    return CompileResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        artifacts=CompileArtifacts(
            binary=_capture_public_file(workspace_path, artifact_map["binary"], inline=False) if "binary" in artifact_map else None,
            ptx=_capture_public_file(workspace_path, artifact_map["ptx"], inline=False) if "ptx" in artifact_map else None,
            cubin=_capture_public_file(workspace_path, artifact_map["cubin"], inline=False) if "cubin" in artifact_map else None,
            resource_usage=_capture_public_file(workspace_path, artifact_map["resource_usage"], inline=False) if "resource_usage" in artifact_map else None,
            sass=CompileSassArtifacts(
                nvdisasm=_capture_public_file(workspace_path, artifact_map["sass_nvdisasm"], inline=False) if "sass_nvdisasm" in artifact_map else None,
            ),
        ),
        tool_outputs=CompileToolOutputs(
            nvcc_ptx=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvcc-ptx.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvcc-ptx.stderr", inline=True),
            ),
            ptxas=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.ptxas.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.ptxas.stderr", inline=True),
            ),
            resource_usage=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.resource-usage.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.resource-usage.stderr", inline=True),
            ),
            nvdisasm=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvdisasm.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvdisasm.stderr", inline=True),
            ),
        ),
    )


@app.post("/files/read", response_model=FileReadResponse)
def file_read_endpoint(request: FileReadRequest) -> FileReadResponse:
    """Read one turn-relative file from artifacts/, logs/, or state/."""

    _validate_file_read_scope(request.path)
    workspace = resolve_workspace_bundle(**request.metadata.model_dump())
    file_payload = _capture_public_file(workspace["workspace_path"], request.path, inline=True, max_bytes=request.max_bytes)
    if file_payload is None:
        raise HTTPException(status_code=404, detail=f"file not found for this turn/path: {request.path}")
    return FileReadResponse(metadata=request.metadata, file=file_payload)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate one compiled artifact against slug-keyed runtime configs."""

    result = run_evaluate_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
    )
    attempt = result["attempt"]
    items = {
        config_slug: EvaluateConfigOutput(
            status=item["status"],
            reference=item.get("reference") or {},
            generated=item.get("generated") or {},
            correctness=item.get("correctness", {}),
            performance=item.get("performance", {}),
            artifacts=_capture_public_files(
                result["workspace_path"],
                [artifact["path"] for artifact in item.get("artifacts", [])],
                inline=True,
            ),
            logs=_capture_public_files(
                result["workspace_path"],
                _stage_log_paths("evaluate", attempt, config_slug),
                inline=True,
            ),
        )
        for config_slug, item in result.get("configs", {}).items()
    }
    return EvaluateResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        configs=items,
    )


@app.post("/profile", response_model=ProfileResponse)
def profile_endpoint(request: ProfileRequest) -> ProfileResponse:
    """Profile one compiled artifact against slug-keyed runtime configs."""

    result = run_profile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
        mode=request.mode,
        profiler_backend=request.profiler_backend,
    )
    attempt = result["attempt"]
    items = {
        config_slug: ProfileConfigOutput(
            status=item["status"],
            summary=item.get("summary", {}),
            reference=item.get("reference") or {},
            generated=item.get("generated") or {},
            reference_summary=(item.get("reference") or {}).get("summary", {}),
            generated_summary=(item.get("generated") or {}).get("summary", {}),
            artifacts=_capture_public_files(
                result["workspace_path"],
                [artifact["path"] for artifact in item.get("artifacts", [])],
                inline=True,
            ),
            logs=_capture_public_files(
                result["workspace_path"],
                _stage_log_paths("profile", attempt, config_slug),
                inline=True,
            ),
        )
        for config_slug, item in result.get("configs", {}).items()
    }
    return ProfileResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        configs=items,
    )


@app.post("/execute", response_model=ExecuteResponse)
def execute_endpoint(request: ExecuteRequest) -> ExecuteResponse:
    """Run a generic CUDA-tool command and return logs only."""

    result = run_execute_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        command=request.command,
        env=request.env,
    )
    attempt = result["attempt"]
    return ExecuteResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        logs=_capture_public_files(result["workspace_path"], _stage_log_paths("execute", attempt), inline=True),
    )
