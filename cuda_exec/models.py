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


class ConfigSpec(BaseModel):
    """Runtime config body keyed by a stable config slug.

    Evaluate/profile requests and responses both use the config slug as the
    outer dict key. This keeps request/response mentally aligned:
        config_slug -> config spec / config output
    """

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

    Why request-side files stay this simple:
    - compile inputs are expected to be normal text source files
    - the caller already knows the intended relative path
    - request-side inputs do not need response-only metadata like encoding or truncation

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
    """Evaluate request over one or more runtime configs.

    `configs` is a slug-keyed map:
        config_slug -> ConfigSpec

    Example:
        {
          "fa4-causal-l12-e4096-h32": {"num_layers": 12, "embedding_size": 4096, "num_heads": 32, "causal": true}
        }
    """

    configs: Dict[str, ConfigSpec] = Field(
        ...,
        min_length=1,
        description="Slug-keyed runtime configs to evaluate against the compiled artifact",
    )


class ProfileRequest(RequestBase):
    """Profile request over one or more runtime configs.

    `configs` is a slug-keyed map:
        config_slug -> ConfigSpec
    """

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

    Why a payload wrapper is needed on the response side:
    - some returned files are text, but some are binary
    - binary payloads need an explicit encoding marker (`base64`)
    - large files may be truncated by service limits

    Request-side compile inputs do not need this wrapper because they are
    currently modeled as simple text source files, so Dict[path, content] is
    enough there.
    """

    content: str = Field(
        ...,
        description="Returned file content for this relative path. Text uses utf8; binary uses base64.",
    )
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

    This is the response-side counterpart to RequestBase.
    Public responses stay intentionally small: they report stage outcome and
    expose only stage-relevant artifacts/logs, not internal workflow state.
    """

    metadata: Metadata = Field(..., description="Required echoed metadata from the request")
    ok: bool
    attempt: int


class CompileResponse(ResponseBase):
    """Minimal compile response.

    - `artifacts` is a dict of `relative_path -> FilePayload`
    - `logs` is a dict of `relative_path -> FilePayload`
    """

    artifacts: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed kept compile outputs",
    )
    logs: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed compile log/stdout/stderr files",
    )


class EvaluateConfigOutput(BaseModel):
    """Public evaluate output for one config slug.

    Evaluate responses mirror request shape:
        config_slug -> EvaluateConfigOutput
    """

    ok: bool
    logs: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed per-config evaluate log/stdout/stderr files",
    )


class EvaluateResponse(ResponseBase):
    """Minimal evaluate response keyed by config slug.

    Example:
        {
          "configs": {
            "fa4-causal-l12-e4096-h32": {
              "ok": true,
              "logs": {
                "logs/evaluate.attempt_001.config_fa4-causal-l12-e4096-h32.stdout": {
                  "content": "latency_ms=...",
                  "encoding": "utf8",
                  "truncated": false
                }
              }
            }
          }
        }
    """

    configs: Dict[str, EvaluateConfigOutput] = Field(default_factory=dict)


class ProfileConfigOutput(BaseModel):
    """Public profile output for one config slug.

    Profile responses mirror request shape:
        config_slug -> ProfileConfigOutput
    """

    ok: bool
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
    Any higher-level meaning of command outputs is left to the caller.

    `logs` is a dict of `relative_path -> FilePayload`.
    """

    logs: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed execute log/stdout/stderr files",
    )
