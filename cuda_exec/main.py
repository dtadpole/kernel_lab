from __future__ import annotations

from fastapi import FastAPI

from cuda_exec.models import (
    CommandResponse,
    CompileRequest,
    EvaluateRequest,
    ExecuteRequest,
    ProfileRequest,
)
from cuda_exec.runner import run_cuda_binary, run_generic_command, run_profile

app = FastAPI(title="cuda_exec", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "service": "cuda_exec"}


@app.post("/compile", response_model=CommandResponse)
def compile_endpoint(request: CompileRequest) -> CommandResponse:
    return CommandResponse(
        **run_generic_command(
            kind="compile",
            command=request.command,
            workdir=request.workdir,
            env=request.env,
            timeout_seconds=request.timeout_seconds,
        )
    )


@app.post("/evaluate", response_model=CommandResponse)
def evaluate_endpoint(request: EvaluateRequest) -> CommandResponse:
    return CommandResponse(
        **run_generic_command(
            kind="evaluate",
            command=request.command,
            workdir=request.workdir,
            env=request.env,
            timeout_seconds=request.timeout_seconds,
        )
    )


@app.post("/profile", response_model=CommandResponse)
def profile_endpoint(request: ProfileRequest) -> CommandResponse:
    return CommandResponse(
        **run_profile(
            profiler=request.profiler,
            target_command=request.target_command,
            profiler_args=request.profiler_args,
            workdir=request.workdir,
            env=request.env,
            timeout_seconds=request.timeout_seconds,
        )
    )


@app.post("/execute", response_model=CommandResponse)
def execute_endpoint(request: ExecuteRequest) -> CommandResponse:
    return CommandResponse(
        **run_cuda_binary(
            binary_path=request.binary_path,
            args=request.args,
            workdir=request.workdir,
            env=request.env,
            timeout_seconds=request.timeout_seconds,
        )
    )
