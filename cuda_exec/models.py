from __future__ import annotations

from typing import Any, Dict, List, Optional

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


class HealthResponse(BaseModel):
    ok: bool
    service: str


class StageResponseBase(BaseModel):
    metadata: Metadata = Field(..., description="Required echoed metadata from the request")
    ok: bool
    attempt: int


class CompileResponse(StageResponseBase):
    binary_path: str
    log_path: str
    stdout_path: str
    stderr_path: str


class ConfigStageResult(BaseModel):
    config_id: str
    ok: bool
    log_path: str
    stdout_path: str
    stderr_path: str


class EvaluateResponse(StageResponseBase):
    results: List[ConfigStageResult] = Field(default_factory=list)


class ProfileConfigResult(ConfigStageResult):
    report_path: str


class ProfileResponse(StageResponseBase):
    results: List[ProfileConfigResult] = Field(default_factory=list)


class ExecuteResponse(StageResponseBase):
    log_path: str
    stdout_path: str
    stderr_path: str
