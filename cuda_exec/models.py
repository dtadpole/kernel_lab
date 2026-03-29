"""Public API models for cuda_exec.

Documentation placement rule:
- Put request/response contract semantics here.
- Keep runtime-layout semantics in runner.py.
- Keep full design rationale in DESIGN.md.

Important public conventions captured in this file:
- Compile inputs are inline file maps: Dict[relative_path, content].
- Evaluate/profile configs are slug-keyed maps: Dict[config_slug, Dict[str, Any]].
- Public response files are returned as Dict[relative_path, FilePayload].
- Relative paths may include folder names, but must remain relative.
- `reference` means the reference side of a comparison.
- `generated` means the generated or current side under evaluation.
- `artifacts` means kept results.
- `logs` means process output.
- `state` remains internal-first and is not exposed in default public responses.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


class Metadata(BaseModel):
    """Required turn identity for all command-style requests."""

    run_tag: str = Field(..., min_length=1, description="Agent run namespace tag")
    version: str = Field(..., min_length=1, description="Agent/API version tag")
    direction_id: int = Field(..., ge=0, description="Stable integer id for a research direction")
    direction_slug: str = Field(..., min_length=1, description="Readable slug for a research direction")
    turn: int = Field(..., ge=0, description="Turn index within the direction")


class RequestBase(BaseModel):
    metadata: Metadata = Field(..., description="Required agent metadata")
    timeout_seconds: int = Field(default=180, ge=1, le=900)


class CompileRequest(RequestBase):
    """Compile request using inline file maps.

    Both `reference_files` and `generated_files` are maps of:
        relative_path -> file_content

    Compile request contract:
    - `reference_files` must be non-empty
    - `reference_files` must include a file keyed as `reference.py` (the entry point)
    - `generated_files` must be non-empty
    - `generated_files` must contain exactly one `.cu` file, keyed as `generated.cu`
    - `generated_files` may include additional headers or inline helper files
    - `reference_files` may include additional helper files of any type
    - compile may run only once per turn; use a new turn for a different upload set

    Why request-side files stay this simple:
    - compile inputs are expected to be normal text source files
    - the caller already knows the intended relative path
    - request-side inputs do not need response-only metadata like encoding or truncation
    """

    reference_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Non-empty map of relative path to file content; must include reference.py as the entry point",
    )
    generated_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Non-empty map of generated source inputs; must include generated.cu as the single .cu entry file; headers and inline helper files also allowed",
    )


class EvaluateRequest(RequestBase):
    """Evaluate request over slug-keyed runtime configs.

    `configs` is intentionally flexible:
        config_slug -> arbitrary kernel-specific config payload

    The service owns the transport shape of config. The kernel owns the semantic
    shape of config.

    Reference-side contract note:
    - reference Python code is expected to export `Model(torch.nn.Module)`
    - reference Python code is expected to export `get_init_inputs()`
    - reference Python code is expected to export `get_inputs(config)`
    """

    configs: Dict[str, Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Slug-keyed kernel-specific runtime config payloads for evaluate",
    )


class ProfileRequest(RequestBase):
    """Profile request over slug-keyed runtime configs.

    `configs` is intentionally flexible:
        config_slug -> arbitrary kernel-specific config payload

    `mode` controls which side to profile:
    - `generated_only`: profile the compiled/generated side only
    - `reference_only`: profile the reference module side only
    - `dual`: run both sides and include comparison metadata

    `profiler_backend` selects the implementation path:
    - `comparison_runtime`: current behavior-first runtime
    - `ncu`: generated-side Nsight Compute capture path intentionally scoped to `mode="generated_only"`
    """

    mode: Literal["reference_only", "generated_only", "dual"] = Field(
        default="generated_only",
        description="Which side(s) to profile for each config",
    )
    profiler_backend: Literal["comparison_runtime", "ncu"] = Field(
        default="comparison_runtime",
        description="Profile implementation backend",
    )
    configs: Dict[str, Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Slug-keyed kernel-specific runtime config payloads for profile",
    )


class ExecuteRequest(RequestBase):
    """Generic CUDA-tool execution request.

    `execute` is intentionally a tool-style path, not a workflow-state stage.
    It is public-logs-first: the service runs the command and returns logs, but
    does not expose execute-specific state in the default public response.
    """

    command: list[str] = Field(..., min_length=1, description="Executable plus arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")


class FileReadRequest(BaseModel):
    metadata: Metadata = Field(..., description="Required turn identity used to resolve the turn root")
    path: str = Field(..., min_length=1, description="Relative path under the turn root to read")
    max_bytes: int | None = Field(default=None, ge=1, description="Optional maximum number of bytes to inline from the requested file")


class HealthResponse(BaseModel):
    ok: bool
    service: str


class FilePayload(BaseModel):
    """Public file payload returned in responses.

    Public responses use the shape:
        relative_path -> FilePayload

    Example inline payload:
        {
          "logs/compile.attempt_001.ptxas.stderr": {
            "path": "logs/compile.attempt_001.ptxas.stderr",
            "inline": true,
            "content": "ptxas info    : Used 6 registers ...",
            "encoding": "utf8",
            "truncated": false
          }
        }

    Example path-only payload:
        {
          "artifacts/compile.attempt_001.vector_add_inline_ptx.cubin": {
            "path": "artifacts/compile.attempt_001.vector_add_inline_ptx.cubin",
            "inline": false
          }
        }

    The relative path itself is still the outer dict key. `path` repeats it so
    callers can consume either the mapping key or the payload alone.
    """

    path: str = Field(..., description="Relative path for this returned file reference")
    inline: bool = Field(default=True, description="True when file content is inlined in the response; false for path-only references")
    content: str | None = Field(default=None, description="Returned file content when `inline=true`. Text uses utf8; binary uses base64.")
    encoding: Literal["utf8", "base64"] | None = Field(
        default=None,
        description="Encoding used for `content` when present so callers can distinguish text from binary payloads.",
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

    model_config = ConfigDict(exclude_none=True)

    min: float | None = None
    median: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None


class CorrectnessSummary(BaseModel):
    """Structured correctness summary for evaluate.

    This is intentionally more structured than raw logs so agents do not need
    to parse log text to understand numerical quality.
    """

    metadata: Dict[str, Any] = Field(default_factory=dict)
    passed: bool | None = None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    abs_variance: float | None = None
    max_rel_error: float | None = None
    mean_rel_error: float | None = None
    rel_variance: float | None = None
    output_shape: str | None = None
    trials: str | None = None
    total_trials: int | None = None
    passed_trials: int | None = None


class PerformanceSummary(BaseModel):
    """Structured performance summary used by evaluate/profile."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: LatencySummary = Field(default_factory=LatencySummary)
    runs: int | None = None
    comparison: Dict[str, Any] = Field(default_factory=dict)


class SideProfileSummary(BaseModel):
    """Direct side-specific profile summary exposed in the public response.

    Unlike the top-level `summary`, this mirrors the nested side payload and does
    not synthesize a `comparison` object when the side is absent.
    """

    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: LatencySummary = Field(default_factory=LatencySummary)
    runs: int | None = None


class ToolIOPair(BaseModel):
    stdout: FilePayload | None = None
    stderr: FilePayload | None = None


class CompileSassArtifacts(BaseModel):
    nvdisasm: FilePayload | None = None


class CompileToolOutputs(BaseModel):
    nvcc_ptx: ToolIOPair = Field(default_factory=ToolIOPair)
    ptxas: ToolIOPair = Field(default_factory=ToolIOPair)
    resource_usage: ToolIOPair = Field(default_factory=ToolIOPair)
    nvdisasm: ToolIOPair = Field(default_factory=ToolIOPair)


class CompileArtifacts(BaseModel):
    binary: FilePayload | None = None
    ptx: FilePayload | None = None
    cubin: FilePayload | None = None
    resource_usage: FilePayload | None = None
    sass: CompileSassArtifacts = Field(default_factory=CompileSassArtifacts)


class CompileResponse(ResponseBase):
    artifacts: CompileArtifacts = Field(default_factory=CompileArtifacts, description="Structured kept compile outputs")
    tool_outputs: CompileToolOutputs = Field(default_factory=CompileToolOutputs, description="Structured inline stdout/stderr from compile tools")


class FileReadResponse(BaseModel):
    metadata: Metadata = Field(..., description="Echoed turn identity used to resolve the file")
    file: FilePayload = Field(..., description="Inline payload for the requested turn-relative file")


class EvaluateConfigOutput(BaseModel):
    """Public evaluate output for one config slug.

    `status` is the per-config result, while the top-level response uses
    `all_ok` as the aggregate stage result.

    `reference` / `generated` expose the side-specific structured payloads,
    while `correctness` and `performance` provide the compact comparison-facing
    summaries used by the public evaluate contract.
    """

    status: Literal["ok", "error", "timeout", "skipped"]
    reference: Dict[str, Any] = Field(default_factory=dict)
    generated: Dict[str, Any] = Field(default_factory=dict)
    correctness: CorrectnessSummary = Field(default_factory=CorrectnessSummary)
    performance: PerformanceSummary = Field(default_factory=PerformanceSummary)
    artifacts: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed kept evaluate comparison artifacts for this config",
    )
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
            "tensor2d-1024x1024": {
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

    `summary` carries the compact top-level profile result.
    `reference` / `generated` preserve the side-specific raw payloads from the
    runtime.
    `reference_summary` / `generated_summary` expose the side-by-side structured
    summaries directly in the public response model so callers do not need to
    reach into the nested side payloads for the most common fields.
    `artifacts` / `logs` carry the raw retained files.
    """

    status: Literal["ok", "error", "timeout", "skipped"]
    summary: PerformanceSummary = Field(default_factory=PerformanceSummary)
    reference: Dict[str, Any] = Field(default_factory=dict)
    generated: Dict[str, Any] = Field(default_factory=dict)
    reference_summary: SideProfileSummary = Field(default_factory=SideProfileSummary)
    generated_summary: SideProfileSummary = Field(default_factory=SideProfileSummary)
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
