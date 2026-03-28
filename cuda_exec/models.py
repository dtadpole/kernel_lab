from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    run_tag: str = Field(..., min_length=1, description="Agent run namespace tag")
    version: str = Field(..., min_length=1, description="Agent/API version tag")
    direction_id: int = Field(..., ge=0, description="Stable integer id for a research direction")
    direction_slug: str = Field(
        ...,
        min_length=1,
        description="Readable slug for a research direction",
    )
    turn: int = Field(..., ge=0, description="Turn index within the direction")


class RequestBase(BaseModel):
    metadata: Metadata = Field(..., description="Required agent metadata")
    return_files: List[str] = Field(
        default_factory=list,
        description="Files to load and return in the response after command execution",
    )


class CommandRequest(RequestBase):
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


class ProfileRequest(RequestBase):
    profiler: Literal["ncu", "nsys"] = Field(..., description="Profiler backend")
    target_command: List[str] = Field(..., min_length=1, description="Command to profile")
    profiler_args: List[str] = Field(default_factory=list, description="Extra profiler arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")
    timeout_seconds: int = Field(default=1800, ge=1, le=86400)


class ExecuteRequest(RequestBase):
    binary_path: str = Field(..., description="Absolute path to a CUDA Toolkit binary")
    args: List[str] = Field(default_factory=list, description="Arguments passed to the binary")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")
    timeout_seconds: int = Field(default=300, ge=1, le=86400)


class CommandOutput(BaseModel):
    stdout: str
    stderr: str


class ResponseFile(BaseModel):
    path: str
    name: str
    exists: bool
    size_bytes: Optional[int] = None
    encoding: Optional[Literal["utf8", "base64"]] = None
    truncated: bool = False
    content: Optional[str] = None
    error: Optional[str] = None


class CommandResponse(BaseModel):
    metadata: Metadata = Field(..., description="Required echoed metadata from the request")
    ok: bool
    kind: str
    command: List[str]
    workspace_path: str
    returncode: int
    duration_seconds: float
    output: CommandOutput
    files: List[ResponseFile] = Field(default_factory=list)


class HealthResponse(BaseModel):
    ok: bool
    service: str
