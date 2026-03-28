from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CommandRequest(BaseModel):
    workdir: str = Field(..., description="Working directory for command execution")
    command: List[str] = Field(..., min_length=1, description="Executable plus arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")
    timeout_seconds: int = Field(default=300, ge=1, le=86400)


class CompileRequest(CommandRequest):
    artifacts: List[str] = Field(default_factory=list, description="Expected output files")


class EvaluateRequest(CommandRequest):
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="Optional expected output markers or artifact names",
    )


class ProfileRequest(BaseModel):
    profiler: Literal["ncu", "nsys"] = Field(..., description="Profiler backend")
    workdir: str = Field(..., description="Working directory for profiling")
    target_command: List[str] = Field(..., min_length=1, description="Command to profile")
    profiler_args: List[str] = Field(default_factory=list, description="Extra profiler arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")
    timeout_seconds: int = Field(default=1800, ge=1, le=86400)


class ExecuteRequest(BaseModel):
    binary_path: str = Field(..., description="Absolute path to a CUDA Toolkit binary")
    args: List[str] = Field(default_factory=list, description="Arguments passed to the binary")
    workdir: Optional[str] = Field(default=None, description="Optional working directory")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")
    timeout_seconds: int = Field(default=300, ge=1, le=86400)


class CommandResponse(BaseModel):
    ok: bool
    kind: str
    command: List[str]
    workdir: str
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
