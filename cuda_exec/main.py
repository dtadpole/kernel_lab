"""FastAPI entrypoints for cuda_exec.

Keep this module thin.
- models.py documents the public request/response contract.
- runner.py documents runtime-layout semantics.
- DESIGN.md holds the full design rationale.
"""

from __future__ import annotations

import re

from fastapi import FastAPI

from cuda_exec.models import (
    CompileRequest,
    CompileResponse,
    ConfigStageResult,
    EvaluateRequest,
    EvaluateResponse,
    ExecuteRequest,
    ExecuteResponse,
    HealthResponse,
    ProfileConfigResult,
    ProfileRequest,
    ProfileResponse,
)
from cuda_exec.runner import capture_turn_file
from cuda_exec.tasks import (
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


def _config_suffix(config_id: str) -> str:
    return f"config_{_slugify(config_id)}"


def _stage_log_paths(stage: str, attempt: int, config_id: str | None = None) -> list[str]:
    """Return public log relative paths for a stage attempt."""

    base = f"logs/{stage}.{_attempt_tag(attempt)}"
    if config_id is not None:
        base += f".{_config_suffix(config_id)}"
    return [f"{base}.log", f"{base}.stdout", f"{base}.stderr"]


def _compile_binary_path(result: dict) -> str:
    for artifact in result.get("artifacts", []):
        if artifact.get("artifact_id") == "compile:primary_binary":
            return artifact["path"]
    raise ValueError("compile result missing primary binary artifact")


def _profile_report_path(config_result: dict) -> str:
    for artifact in config_result.get("artifacts", []):
        if artifact.get("kind") == "profile_report":
            return artifact["path"]
    raise ValueError(f"profile result missing report artifact for config {config_result['config']['config_id']}")


def _capture_public_files(workspace_path: str, rel_paths: list[str]) -> dict[str, dict]:
    """Materialize public response files as relative_path -> FilePayload.

    This is where the service converts internal on-disk files into the compact
    public API shape used by artifacts/logs.
    """

    payload: dict[str, dict] = {}
    for rel_path in rel_paths:
        item = capture_turn_file(rel_path, workspace_path)
        if not item.get("exists") or item.get("error"):
            continue
        payload[rel_path] = {
            "content": item.get("content") or "",
            "encoding": item.get("encoding") or "utf8",
            "truncated": bool(item.get("truncated", False)),
        }
    return payload


@app.post("/compile", response_model=CompileResponse)
def compile_endpoint(request: CompileRequest) -> CompileResponse:
    """Compile inline code inputs into kept compile artifacts plus logs."""

    result = run_compile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        original_files=request.original_files,
        generated_files=request.generated_files,
    )
    attempt = result["attempt"]
    return CompileResponse(
        metadata=request.metadata,
        ok=result["ok"],
        attempt=attempt,
        artifacts=_capture_public_files(result["workspace_path"], [_compile_binary_path(result)]),
        logs=_capture_public_files(result["workspace_path"], _stage_log_paths("compile", attempt)),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate one compiled artifact against one or more runtime configs."""

    result = run_evaluate_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
    )
    attempt = result["attempt"]
    items = [
        ConfigStageResult(
            config_id=item["config"]["config_id"],
            ok=item["ok"],
            logs=_capture_public_files(
                result["workspace_path"],
                _stage_log_paths("evaluate", attempt, item["config"]["config_id"]),
            ),
        )
        for item in result.get("config_results", [])
    ]
    return EvaluateResponse(
        metadata=request.metadata,
        ok=result["ok"],
        attempt=attempt,
        results=items,
    )


@app.post("/profile", response_model=ProfileResponse)
def profile_endpoint(request: ProfileRequest) -> ProfileResponse:
    """Profile one compiled artifact against one or more runtime configs."""

    result = run_profile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
    )
    attempt = result["attempt"]
    items = [
        ProfileConfigResult(
            config_id=item["config"]["config_id"],
            ok=item["ok"],
            artifacts=_capture_public_files(result["workspace_path"], [_profile_report_path(item)]),
            logs=_capture_public_files(
                result["workspace_path"],
                _stage_log_paths("profile", attempt, item["config"]["config_id"]),
            ),
        )
        for item in result.get("config_results", [])
    ]
    return ProfileResponse(
        metadata=request.metadata,
        ok=result["ok"],
        attempt=attempt,
        results=items,
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
        ok=result["ok"],
        attempt=attempt,
        logs=_capture_public_files(result["workspace_path"], _stage_log_paths("execute", attempt)),
    )
