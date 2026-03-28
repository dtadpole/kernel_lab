"""Public API models for cuda_exec.

Documentation placement rule:
- Put request/response contract semantics here.
- Keep runtime-layout semantics in runner.py.
- Keep full design rationale in DESIGN.md.

Important public conventions captured in this file:
- Compile inputs are inline file maps: Dict[relative_path, content].
- Evaluate/profile configs are slug-keyed maps: Dict[config_slug, ConfigSpec].
- Public response files are returned as Dict[relative_path, FilePayload].
- Relative paths may include folder names, but must remain relative.
- `artifacts` means kept results.
- `logs` means process output.
- `state` remains internal-first and is not exposed in default public responses.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Required turn identity for all command-style requests."""

    run_tag: str = Field(..., min_length=1, description="Agent run namespace tag")
    version: str = Field(..., min_length=1, description="Agent/API version tag")
    direction_id: int = Field(..., ge=0, description="Stable integer id for a research direction")
    direction_slug: str = Field(..., min_length=1, description="Readable slug for a research direction")
    turn: int = Field(..., ge=0, description="Turn index within the direction")


class ConfigSpec(BaseModel):
    """Runtime config body keyed by a stable config slug."""

    num_layers: Optional[int] = Field(default=None, ge=1)
    embedding_size: Optional[int] = Field(default=None, ge=1)
    num_heads: Optional[int] = Field(default=None, ge=1)
    causal: Optional[bool] = Field(default=None)
    extra: Dict[str, Any] = Field(default_factory=dict, description="Extra config-specific runtime fields")


class RequestBase(BaseModel):
    metadata: Metadata = Field(..., description="Required agent metadata")
    timeout_seconds: int = Field(default=180, ge=1, le=900)


class CompileRequest(RequestBase):
    """Compile request using inline file maps.

    Both `original_files` and `generated_files` are maps of:
        relative_path -> file_content

    Why request-side files stay this simple:
    - compile inputs are expected to be normal text source files
    - the caller already knows the intended relative path
    - request-side inputs do not need response-only metadata like encoding or truncation
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
    """Evaluate request over slug-keyed runtime configs."""

    configs: Dict[str, ConfigSpec] = Field(
        ...,
        min_length=1,
        description="Slug-keyed runtime configs to evaluate against the compiled artifact",
    )


class ProfileRequest(RequestBase):
    """Profile request over slug-keyed runtime configs."""

    configs: Dict[str, ConfigSpec] = Field(
        ...,
        min_length=1,
        description="Slug-keyed runtime configs to profile against the compiled artifact",
    )


class ExecuteRequest(RequestBase):
    """Generic CUDA-tool execution request.

    `execute` is intentionally a tool-style path, not a workflow-state stage.
    It is public-logs-first: the service runs the command and returns logs, but
    does not expose execute-specific state in the default public response.
    """

    command: list[str] = Field(..., min_length=1, description="Executable plus arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")


class HealthResponse(BaseModel):
    ok: bool
    service: str


class FilePayload(BaseModel):
    """Public file payload returned in responses.

    Public responses use the shape:
        relative_path -> FilePayload

    Example:
        {
          "logs/compile.attempt_001.log": {
            "content": "toolkit_command: ...",
            "encoding": "utf8",
            "truncated": false
          }
        }

    The relative path itself is the outer dict key.
    This object only describes the payload stored at that path.
    """

    content: str = Field(..., description="Returned file content. Text uses utf8; binary uses base64.")
    encoding: Literal["utf8", "base64"] = Field(
        default="utf8",
        description="Encoding used for `content` so callers can distinguish text from binary payloads.",
    )
    truncated: bool = Field(
        default=False,
        description="True when the service returned only a prefix of the file because of response size limits.",
    )


class ResponseBase(BaseModel):
    """Shared public response fields.

    `all_ok` is the aggregate stage-level result. Per-config outputs use
    `status` to report each individual config.
    """

    metadata: Metadata = Field(..., description="Required echoed metadata from the request")
    all_ok: bool
    attempt: int


class LatencySummary(BaseModel):
    """Structured latency statistics in milliseconds."""

    min: Optional[float] = None
    median: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None


class CorrectnessSummary(BaseModel):
    """Structured correctness summary for evaluate.

    This is intentionally more structured than raw logs so agents do not need
    to parse log text to understand numerical quality.
    """

    metadata: Dict[str, Any] = Field(default_factory=dict)
    passed: Optional[bool] = None
    max_abs_error: Optional[float] = None
    mean_abs_error: Optional[float] = None
    abs_variance: Optional[float] = None
    max_rel_error: Optional[float] = None
    mean_rel_error: Optional[float] = None
    rel_variance: Optional[float] = None


class PerformanceSummary(BaseModel):
    """Structured performance summary used by evaluate/profile."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: LatencySummary = Field(default_factory=LatencySummary)
    runs: Optional[int] = None


class CompileResponse(ResponseBase):
    artifacts: Dict[str, FilePayload] = Field(default_factory=dict, description="Relative-path keyed kept compile outputs")
    logs: Dict[str, FilePayload] = Field(default_factory=dict, description="Relative-path keyed compile log/stdout/stderr files")


class EvaluateConfigOutput(BaseModel):
    """Public evaluate output for one config slug.

    `status` is the per-config result, while the top-level response uses
    `all_ok` as the aggregate stage result.
    """

    status: Literal["ok", "error", "timeout", "skipped"]
    correctness: CorrectnessSummary = Field(default_factory=CorrectnessSummary)
    performance: PerformanceSummary = Field(default_factory=PerformanceSummary)
    logs: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed per-config evaluate log/stdout/stderr files",
    )


class EvaluateResponse(ResponseBase):
    """Minimal evaluate response keyed by config slug.

    Example:
        {
          "all_ok": true,
          "configs": {
            "fa4-causal-l12-e4096-h32": {
              "status": "ok",
              "correctness": {
                "passed": true,
                "max_abs_error": 1.2e-6,
                "max_rel_error": 2.4e-5
              },
              "performance": {
                "latency_ms": {"min": 0.82, "median": 0.89, "max": 0.97, "mean": 0.89},
                "runs": 100
              },
              "logs": {}
            }
          }
        }
    """

    configs: Dict[str, EvaluateConfigOutput] = Field(default_factory=dict)


class ProfileConfigOutput(BaseModel):
    """Public profile output for one config slug.

    `summary` carries structured performance/profile information, while
    `artifacts` and `logs` carry the raw retained files.
    """

    status: Literal["ok", "error", "timeout", "skipped"]
    summary: PerformanceSummary = Field(default_factory=PerformanceSummary)
    artifacts: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed kept profiling outputs for this config",
    )
    logs: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed profile log/stdout/stderr files for this config",
    )


class ProfileResponse(ResponseBase):
    """Minimal profile response keyed by config slug."""

    configs: Dict[str, ProfileConfigOutput] = Field(default_factory=dict)


class ExecuteResponse(ResponseBase):
    """Minimal execute response.

    Execute is logs-only in the public API by design.
    """

    logs: Dict[str, FilePayload] = Field(default_factory=dict, description="Relative-path keyed execute log/stdout/stderr files")
