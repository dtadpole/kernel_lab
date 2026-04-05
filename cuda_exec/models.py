"""Public API models for cuda_exec.

Documentation placement rule:
- Put request/response contract semantics here.
- Keep runtime-layout semantics in runner.py.
- Keep full design rationale in DESIGN.md.

Important public conventions captured in this file:
- Compile inputs are inline file maps: Dict[relative_path, content].
- Trial/profile configs are slug-keyed maps: Dict[config_slug, Dict[str, Any]].
- Public response files are returned as Dict[relative_path, FilePayload].
- Relative paths may include folder names, but must remain relative.
- Implementations are keyed by slug: `{source}-{name}` (e.g. `ref-cublas`, `gen-cuda`).
- The first `ref-*` impl is the golden baseline for correctness comparison.
- `artifacts` means kept results.
- `logs` means process output.
- `state` remains internal-first and is not exposed in default public responses.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


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
    """Compile request using impl-keyed file maps.

    `impls` maps implementation slugs to their source files:
        slug -> {filename: content}

    Example:
        {
            "ref-cublas": {"cublas.py": "..."},
            "gen-cutedsl": {"cutedsl.py": "...", "cute_gemm_sm90.py": "..."},
            "gen-cuda": {"cuda.cu": "..."}
        }

    Slugs follow the format: {source}-{name}
    - source: "ref" (reference/baseline) or "gen" (generated/optimized)
    - name: entry point stem (e.g. "cublas", "cutedsl", "cuda")

    .cu impls are compiled via compile.sh + eval_harness.cu.
    .py impls are run via measure_reference() in trial.py.
    """

    impls: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Impl-slug-keyed map: slug → {filename: content}. At least one ref-* impl required.",
    )


class TrialRequest(RequestBase):
    """Trial request over slug-keyed runtime configs.

    `configs` is intentionally flexible:
        config_slug -> arbitrary kernel-specific config payload

    The service owns the transport shape of config. The kernel owns the semantic
    shape of config.

    Implementation contract note:
    - .py impls export `Model(torch.nn.Module)` and `get_init_inputs()`
    - Input generation is handled by the harness (`generate_inputs()`)
    - .cu impls export `extern "C" int kernel_run(...)`
    """

    configs: Dict[str, Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Slug-keyed kernel-specific runtime config payloads for trial",
    )


class ProfileRequest(RequestBase):
    """Profile request over slug-keyed runtime configs using Nsight Compute.

    `configs` is intentionally flexible:
        config_slug -> arbitrary kernel-specific config payload

    `side` selects which kernel to NCU-profile:
    - `generated`: profile the compiled CUDA binary
    - `reference`: profile the reference Python/CuTe DSL kernel (filters by
      ``NCU_KERNEL_FILTER`` regex to skip PyTorch JIT overhead kernels)
    """

    impl: str = Field(
        ...,
        description="Impl slug to NCU-profile (e.g. 'gen-cuda', 'ref-cublas', 'gen-cutedsl'). "
                    "Resolved from inputs/{slug}/ in the workspace.",
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
          "artifacts/compile.attempt_001.generated.cubin": {
            "path": "artifacts/compile.attempt_001.generated.cubin",
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

    min: float | None = None
    median: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None


class CorrectnessSummary(BaseModel):
    """Structured correctness summary for trial.

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
    """Structured performance summary used by trial/profile."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: LatencySummary = Field(default_factory=LatencySummary)
    runs: int | None = None
    comparison: Dict[str, Any] = Field(default_factory=dict)



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


class ImplTrialResult(BaseModel):
    """Per-impl result for one config in a trial."""

    performance: PerformanceSummary = Field(default_factory=PerformanceSummary)
    correctness: CorrectnessSummary | None = Field(
        default=None,
        description="Correctness vs golden (first ref-*). None for the golden impl itself.",
    )


class TrialConfigOutput(BaseModel):
    """Public trial output for one config slug.

    `impls` maps each implementation slug to its trial result (performance +
    optional correctness). The first ref-* impl is the golden baseline — its
    correctness field is None. All other impls (ref-* and gen-*) have
    correctness compared against the golden.

    Example:
        {
            "status": "ok",
            "golden_slug": "ref-cublas",
            "impls": {
                "ref-cublas": {"performance": {...}, "correctness": null},
                "gen-cutedsl": {"performance": {...}, "correctness": {"passed": true, ...}},
                "gen-cuda": {"performance": {...}, "correctness": {"passed": true, ...}}
            }
        }
    """

    status: Literal["ok", "error", "timeout", "skipped"]
    golden_slug: str = Field(default="", description="Which impl is the golden baseline (first ref-*)")
    impls: Dict[str, ImplTrialResult] = Field(
        default_factory=dict,
        description="Impl-slug-keyed trial results: performance + correctness vs golden",
    )
    artifacts: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed kept trial comparison artifacts for this config",
    )
    logs: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed per-config trial log/stdout/stderr files",
    )


class TrialResponse(ResponseBase):
    """Minimal trial response keyed by config slug.

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

    configs: Dict[str, TrialConfigOutput] = Field(default_factory=dict)


class ProfileConfigOutput(BaseModel):
    """Public NCU profile output for one config slug.

    `summary` carries the compact top-level profile result (timing metadata,
    whether NCU collected metrics, and report path).
    `artifacts` / `logs` carry the raw retained files (including `.ncu-rep`).
    """

    status: Literal["ok", "error", "timeout", "skipped"]
    summary: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, FilePayload] = Field(
        default_factory=dict,
        description="Relative-path keyed NCU outputs for this config (including .ncu-rep)",
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
