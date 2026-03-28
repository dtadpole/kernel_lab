from __future__ import annotations

from fastapi import FastAPI

from cuda_exec.models import (
    CommandOutput,
    CommandResponse,
    CompileRequest,
    EvaluateRequest,
    ExecuteRequest,
    HealthResponse,
    ProfileRequest,
    ResponseFile,
)
from cuda_exec.runner import (
    resolve_workspace_bundle,
    run_cuda_binary,
    run_generic_command,
    run_profile,
)

app = FastAPI(title="cuda_exec", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(ok=True, service="cuda_exec")


@app.post("/compile", response_model=CommandResponse)
def compile_endpoint(request: CompileRequest) -> CommandResponse:
    workspace = resolve_workspace_bundle(**request.metadata.model_dump())
    result = run_generic_command(
        kind="compile",
        command=request.command,
        workspace_path=workspace["workspace_path"],
        env=request.env,
        timeout_seconds=request.timeout_seconds,
        return_files=[*request.artifacts, *request.return_files],
    )
    return CommandResponse(
        metadata=request.metadata,
        ok=result["ok"],
        kind=result["kind"],
        command=result["command"],
        workspace_path=result["workspace_path"],
        returncode=result["returncode"],
        duration_seconds=result["duration_seconds"],
        output=CommandOutput(**result["output"]),
        files=[ResponseFile(**item) for item in result["files"]],
    )


@app.post("/evaluate", response_model=CommandResponse)
def evaluate_endpoint(request: EvaluateRequest) -> CommandResponse:
    workspace = resolve_workspace_bundle(**request.metadata.model_dump())
    result = run_generic_command(
        kind="evaluate",
        command=request.command,
        workspace_path=workspace["workspace_path"],
        env=request.env,
        timeout_seconds=request.timeout_seconds,
        return_files=request.return_files,
    )
    return CommandResponse(
        metadata=request.metadata,
        ok=result["ok"],
        kind=result["kind"],
        command=result["command"],
        workspace_path=result["workspace_path"],
        returncode=result["returncode"],
        duration_seconds=result["duration_seconds"],
        output=CommandOutput(**result["output"]),
        files=[ResponseFile(**item) for item in result["files"]],
    )


@app.post("/profile", response_model=CommandResponse)
def profile_endpoint(request: ProfileRequest) -> CommandResponse:
    workspace = resolve_workspace_bundle(**request.metadata.model_dump())
    result = run_profile(
        profiler=request.profiler,
        target_command=request.target_command,
        profiler_args=request.profiler_args,
        workspace_path=workspace["workspace_path"],
        env=request.env,
        timeout_seconds=request.timeout_seconds,
        return_files=request.return_files,
    )
    return CommandResponse(
        metadata=request.metadata,
        ok=result["ok"],
        kind=result["kind"],
        command=result["command"],
        workspace_path=result["workspace_path"],
        returncode=result["returncode"],
        duration_seconds=result["duration_seconds"],
        output=CommandOutput(**result["output"]),
        files=[ResponseFile(**item) for item in result["files"]],
    )


@app.post("/execute", response_model=CommandResponse)
def execute_endpoint(request: ExecuteRequest) -> CommandResponse:
    workspace = resolve_workspace_bundle(**request.metadata.model_dump())
    result = run_cuda_binary(
        binary_path=request.binary_path,
        args=request.args,
        workspace_path=workspace["workspace_path"],
        env=request.env,
        timeout_seconds=request.timeout_seconds,
        return_files=request.return_files,
    )
    return CommandResponse(
        metadata=request.metadata,
        ok=result["ok"],
        kind=result["kind"],
        command=result["command"],
        workspace_path=result["workspace_path"],
        returncode=result["returncode"],
        duration_seconds=result["duration_seconds"],
        output=CommandOutput(**result["output"]),
        files=[ResponseFile(**item) for item in result["files"]],
    )
