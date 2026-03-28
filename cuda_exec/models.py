from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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


class RuntimeConfig(BaseModel):
    config_id: str = Field(..., min_length=1, description="Stable identifier for a runtime config")
    num_layers: Optional[int] = Field(default=None, ge=1)
    embedding_size: Optional[int] = Field(default=None, ge=1)
    num_heads: Optional[int] = Field(default=None, ge=1)
    causal: Optional[bool] = Field(default=None)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra config-specific runtime fields",
    )


class RequestBase(BaseModel):
    metadata: Metadata = Field(..., description="Required agent metadata")
    timeout_seconds: int = Field(default=180, ge=1, le=900)


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
    configs: List[RuntimeConfig] = Field(
        ...,
        min_length=1,
        description="Runtime configs to evaluate against the compiled artifact",
    )


class ProfileRequest(RequestBase):
    configs: List[RuntimeConfig] = Field(
        ...,
        min_length=1,
        description="Runtime configs to profile against the compiled artifact",
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


class ConfigResult(BaseModel):
    config: RuntimeConfig
    ok: bool
    command: List[str]
    returncode: int
    duration_seconds: float
    output: CommandOutput
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    files: List[ResponseFile] = Field(default_factory=list)


class CommandResponse(BaseModel):
    metadata: Metadata = Field(..., description="Required echoed metadata from the request")
    ok: bool
    kind: str
    attempt: int
    command: List[str]
    turn_root: str
    workspace_path: str
    returncode: int
    duration_seconds: float
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    output: CommandOutput
    files: List[ResponseFile] = Field(default_factory=list)
    config_results: List[ConfigResult] = Field(default_factory=list)


class HealthResponse(BaseModel):
    ok: bool
    service: str
