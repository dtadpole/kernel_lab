"""Public API models for cuda_exec.

Documentation placement rule:
- Put request/response contract semantics here.
- Keep runtime-layout semantics in runner.py.
- Keep full design rationale in DESIGN.md.

Important public conventions captured in this file:
- Compile inputs are inline file maps: Dict[relative_path, content].
- Public response files are returned as Dict[relative_path, ReturnedFile].
- Relative paths may include folder names, but must remain relative.
- `artifacts` means kept results.
- `logs` means process output.
- `state` remains internal-first and is not exposed in default public responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Required turn identity for all command-style requests.

    These fields locate the current turn under the runtime root and form the
    stable request context for compile/evaluate/profile/execute.
    """

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
    """Runtime-only config for evaluate/profile fan-out.

    Compile is code-level. Evaluate/profile are config-level. A single compiled
    artifact may therefore be reused across many RuntimeConfig values.
    """

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
    """Shared request fields for command-style endpoints."""

    metadata: Metadata = Field(..., description="Required agent metadata")
    timeout_seconds: int = Field(default=180, ge=1, le=900)


class CompileRequest(RequestBase):
    """Compile request using inline file maps.

    Both `original_files` and `generated_files` are maps of:
        relative_path -> file_content

    Example:
        {
          "kernels/candidate.cu": "...source code..."
        }

    The service writes these under workspace/inputs/... while preserving the
    provided relative paths.
    """

    original_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of relative path to file content for original source inputs",
    )
    generated_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of relative path to file content for generated source inputs",
    )


class EvaluateRequest(RequestBase):
    """Evaluate request over one or more runtime configs."""

    configs: List[RuntimeConfig] = Field(
        ...,
        min_length=1,
        description="Runtime configs to evaluate against the compiled artifact",
    )


class ProfileRequest(RequestBase):
    """Profile request over one or more runtime configs."""

    configs: List[RuntimeConfig] = Field(
        ...,
        min_length=1,
        description="Runtime configs to profile against the compiled artifact",
    )


class ExecuteRequest(RequestBase):
    """Generic CUDA-tool execution request.

    `execute` is intentionally a tool-style path, not a workflow-state stage.
    It is public-logs-first: the service runs the command and returns logs, but
    does not expose execute-specific state in the default public response.
    """

    command: List[str] = Field(..., min_length=1, description="Executable plus arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")


class HealthResponse(BaseModel):
    ok: bool
    service: str


class ReturnedFile(BaseModel):
    """Returned public file payload.

    Public responses use:
        relative_path -> ReturnedFile

    `encoding` is kept minimal but explicit so that both text logs and binary
    artifacts can share the same response shape.
    """

    content: str = Field(..., description="File payload content. Text uses utf8; binary uses base64")
    encoding: Literal["utf8", "base64"] = Field(
        default="utf8",
        description="Payload encoding for `content`",
    )
    truncated: bool = Field(
        default=False,
        description="Whether the returned content was truncated by service limits",
    )


class StageResponseBase(BaseModel):
    """Shared public response fields.

    Public responses stay intentionally small. They describe stage outcome and
    return only stage-relevant artifacts/logs, not internal workflow state.
    """

    metadata: Metadata = Field(..., description="Required echoed metadata from the request")
    ok: bool
    attempt: int


class CompileResponse(StageResponseBase):
    """Minimal compile response.

    - `artifacts` contains kept compile outputs, typically the compiled binary.
    - `logs` contains compile.log/stdout/stderr keyed by relative path.
    """

    artifacts: Dict[str, ReturnedFile] = Field(
        default_factory=dict,
        description="Relative-path keyed kept compile outputs",
    )
    logs: Dict[str, ReturnedFile] = Field(
        default_factory=dict,
        description="Relative-path keyed compile log/stdout/stderr files",
    )


class ConfigStageResult(BaseModel):
    """Per-config public result for evaluate-like stages."""

    config_id: str
    ok: bool
    logs: Dict[str, ReturnedFile] = Field(
        default_factory=dict,
        description="Relative-path keyed per-config log/stdout/stderr files",
    )


class EvaluateResponse(StageResponseBase):
    """Minimal evaluate response with one result per runtime config."""

    results: List[ConfigStageResult] = Field(default_factory=list)


class ProfileConfigResult(BaseModel):
    """Per-config public result for profile.

    Profile returns both logs and kept profiling artifacts such as .ncu-rep.
    """

    config_id: str
    ok: bool
    artifacts: Dict[str, ReturnedFile] = Field(
        default_factory=dict,
        description="Relative-path keyed kept profiling outputs for this config",
    )
    logs: Dict[str, ReturnedFile] = Field(
        default_factory=dict,
        description="Relative-path keyed profile log/stdout/stderr files for this config",
    )


class ProfileResponse(StageResponseBase):
    """Minimal profile response with one result per runtime config."""

    results: List[ProfileConfigResult] = Field(default_factory=list)


class ExecuteResponse(StageResponseBase):
    """Minimal execute response.

    Execute is logs-only in the public API by design.
    Any higher-level meaning of command outputs is left to the caller.
    """

    logs: Dict[str, ReturnedFile] = Field(
        default_factory=dict,
        description="Relative-path keyed execute log/stdout/stderr files",
    )
