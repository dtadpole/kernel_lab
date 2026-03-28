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
    original_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of relative path to file content for original source inputs",
    )
    generated_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of relative path to file content for generated source inputs",
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


class ReturnedFile(BaseModel):
    content: str
    encoding: Literal["utf8", "base64"] = "utf8"
    truncated: bool = False


class StageResponseBase(BaseModel):
    metadata: Metadata = Field(..., description="Required echoed metadata from the request")
    ok: bool
    attempt: int


class CompileResponse(StageResponseBase):
    artifacts: Dict[str, ReturnedFile] = Field(default_factory=dict)
    logs: Dict[str, ReturnedFile] = Field(default_factory=dict)


class ConfigStageResult(BaseModel):
    config_id: str
    ok: bool
    logs: Dict[str, ReturnedFile] = Field(default_factory=dict)


class EvaluateResponse(StageResponseBase):
    results: List[ConfigStageResult] = Field(default_factory=list)


class ProfileConfigResult(BaseModel):
    config_id: str
    ok: bool
    artifacts: Dict[str, ReturnedFile] = Field(default_factory=dict)
    logs: Dict[str, ReturnedFile] = Field(default_factory=dict)


class ProfileResponse(StageResponseBase):
    results: List[ProfileConfigResult] = Field(default_factory=list)


class ExecuteResponse(StageResponseBase):
    logs: Dict[str, ReturnedFile] = Field(default_factory=dict)
