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
    timeout_seconds: int = Field(default=300, ge=1, le=86400)
    return_files: List[str] = Field(
        default_factory=list,
        description="Additional turn-root-relative files to load and return in the response",
    )


class CompileRequest(RequestBase):
    original_files: List[str] = Field(
        default_factory=list,
        description="Original source artifacts for the direction",
    )
    generated_files: List[str] = Field(
        default_factory=list,
        description="Generated/candidate source artifacts for this turn",
    )


class EvaluateRequest(RequestBase):
    target_artifact_id: Optional[str] = Field(
        default=None,
        description="Optional artifact id override. Defaults to compile:primary_binary",
    )


class ProfileRequest(RequestBase):
    target_artifact_id: Optional[str] = Field(
        default=None,
        description="Optional artifact id override. Defaults to compile:primary_binary",
    )


class ExecuteRequest(RequestBase):
    command: List[str] = Field(..., min_length=1, description="Executable plus arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")


class ArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    path: str
    description: Optional[str] = None


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
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    output: CommandOutput
    files: List[ResponseFile] = Field(default_factory=list)


class HealthResponse(BaseModel):
    ok: bool
    service: str
