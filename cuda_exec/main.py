from __future__ import annotations

from fastapi import FastAPI

from cuda_exec.models import (
    ArtifactRef,
    CommandOutput,
    CommandResponse,
    CompileRequest,
    ConfigResult,
    EvaluateRequest,
    ExecuteRequest,
    HealthResponse,
    ProfileRequest,
    ResponseFile,
)
from cuda_exec.tasks import (
    run_compile_task,
    run_evaluate_task,
    run_execute_task,
    run_profile_task,
)

app = FastAPI(title="cuda_exec", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(ok=True, service="cuda_exec")


def _to_response(metadata, result: dict) -> CommandResponse:
    return CommandResponse(
        metadata=metadata,
        ok=result["ok"],
        kind=result["kind"],
        attempt=result["attempt"],
        command=result["command"],
        turn_root=result["turn_root"],
        workspace_path=result["workspace_path"],
        returncode=result["returncode"],
        duration_seconds=result["duration_seconds"],
        artifacts=[ArtifactRef(**item) for item in result.get("artifacts", [])],
        output=CommandOutput(**result["output"]),
        files=[ResponseFile(**item) for item in result["files"]],
        config_results=[ConfigResult(**item) for item in result.get("config_results", [])],
    )


@app.post("/compile", response_model=CommandResponse)
def compile_endpoint(request: CompileRequest) -> CommandResponse:
    result = run_compile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        original_files=request.original_files,
        generated_files=request.generated_files,
    )
    return _to_response(request.metadata, result)


@app.post("/evaluate", response_model=CommandResponse)
def evaluate_endpoint(request: EvaluateRequest) -> CommandResponse:
    result = run_evaluate_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
    )
    return _to_response(request.metadata, result)


@app.post("/profile", response_model=CommandResponse)
def profile_endpoint(request: ProfileRequest) -> CommandResponse:
    result = run_profile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
    )
    return _to_response(request.metadata, result)


@app.post("/execute", response_model=CommandResponse)
def execute_endpoint(request: ExecuteRequest) -> CommandResponse:
    result = run_execute_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        command=request.command,
        env=request.env,
    )
    return _to_response(request.metadata, result)
