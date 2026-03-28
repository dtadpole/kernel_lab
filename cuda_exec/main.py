from __future__ import annotations

import re

from fastapi import FastAPI

from cuda_exec.models import (
    CompileRequest,
    CompileResponse,
    EvaluateRequest,
    EvaluateResponse,
    ExecuteRequest,
    ExecuteResponse,
    HealthResponse,
    ProfileConfigResult,
    ProfileRequest,
    ProfileResponse,
    ConfigStageResult,
)
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
    return HealthResponse(ok=True, service="cuda_exec")


def _attempt_tag(attempt: int) -> str:
    return f"attempt_{attempt:03d}"


def _slugify(value: str) -> str:
    cleaned = SAFE_SLUG_RE.sub("_", value).strip("._-")
    return cleaned or "default"


def _config_suffix(config_id: str) -> str:
    return f"config_{_slugify(config_id)}"


def _stage_state_path(stage: str, attempt: int) -> str:
    return f"state/{stage}.{_attempt_tag(attempt)}.json"


def _stage_log_base(stage: str, attempt: int, config_id: str | None = None) -> str:
    base = f"logs/{stage}.{_attempt_tag(attempt)}"
    if config_id is not None:
        base += f".{_config_suffix(config_id)}"
    return base


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


@app.post("/compile", response_model=CompileResponse)
def compile_endpoint(request: CompileRequest) -> CompileResponse:
    result = run_compile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        original_files=request.original_files,
        generated_files=request.generated_files,
    )
    attempt = result["attempt"]
    log_base = _stage_log_base("compile", attempt)
    return CompileResponse(
        metadata=request.metadata,
        ok=result["ok"],
        attempt=attempt,
        binary_path=_compile_binary_path(result),
        log_path=f"{log_base}.log",
        stdout_path=f"{log_base}.stdout",
        stderr_path=f"{log_base}.stderr",
        state_path=_stage_state_path("compile", attempt),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(request: EvaluateRequest) -> EvaluateResponse:
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
            log_path=f"{_stage_log_base('evaluate', attempt, item['config']['config_id'])}.log",
            stdout_path=f"{_stage_log_base('evaluate', attempt, item['config']['config_id'])}.stdout",
            stderr_path=f"{_stage_log_base('evaluate', attempt, item['config']['config_id'])}.stderr",
            config_state_path=(
                f"state/configs/evaluate.{_attempt_tag(attempt)}.{_config_suffix(item['config']['config_id'])}.json"
            ),
        )
        for item in result.get("config_results", [])
    ]
    return EvaluateResponse(
        metadata=request.metadata,
        ok=result["ok"],
        attempt=attempt,
        state_path=_stage_state_path("evaluate", attempt),
        results=items,
    )


@app.post("/profile", response_model=ProfileResponse)
def profile_endpoint(request: ProfileRequest) -> ProfileResponse:
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
            report_path=_profile_report_path(item),
            log_path=f"{_stage_log_base('profile', attempt, item['config']['config_id'])}.log",
            stdout_path=f"{_stage_log_base('profile', attempt, item['config']['config_id'])}.stdout",
            stderr_path=f"{_stage_log_base('profile', attempt, item['config']['config_id'])}.stderr",
            config_state_path=(
                f"state/configs/profile.{_attempt_tag(attempt)}.{_config_suffix(item['config']['config_id'])}.json"
            ),
        )
        for item in result.get("config_results", [])
    ]
    return ProfileResponse(
        metadata=request.metadata,
        ok=result["ok"],
        attempt=attempt,
        state_path=_stage_state_path("profile", attempt),
        results=items,
    )


@app.post("/execute", response_model=ExecuteResponse)
def execute_endpoint(request: ExecuteRequest) -> ExecuteResponse:
    result = run_execute_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        command=request.command,
        env=request.env,
    )
    attempt = result["attempt"]
    log_base = _stage_log_base("execute", attempt)
    return ExecuteResponse(
        metadata=request.metadata,
        ok=result["ok"],
        attempt=attempt,
        log_path=f"{log_base}.log",
        stdout_path=f"{log_base}.stdout",
        stderr_path=f"{log_base}.stderr",
        state_path=_stage_state_path("execute", attempt),
    )
