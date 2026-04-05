from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List


from cuda_exec.models import (
    CompileArtifacts,
    CompileRequest,
    CompileResponse,
    CompileSassArtifacts,
    CompileToolOutputs,
    ExecuteRequest,
    ExecuteResponse,
    FileReadRequest,
    FileReadResponse,
    ProfileConfigOutput,
    ProfileRequest,
    ProfileResponse,
    ToolIOPair,
    TrialConfigOutput,
    TrialRequest,
    TrialResponse,
)
from cuda_exec.runner import (
    capture_turn_file,
    resolve_workspace_bundle,
    run_cuda_command,
    run_generic_command,
)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
COMPILE_SCRIPT = SCRIPTS_DIR / "compile.sh"
TRIAL_SCRIPT = SCRIPTS_DIR / "trial.py"
PROFILE_NCU_SCRIPT = SCRIPTS_DIR / "profile.sh"
NCU_REPORT_SCRIPT = SCRIPTS_DIR / "ncu_report.py"
EVAL_HARNESS = SCRIPTS_DIR / "eval_harness.cu"
DEFAULT_COMPILE_ARTIFACT_ID = "compile:primary_binary"
SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
logger = logging.getLogger(__name__)
WORKFLOW_RULES = {
    "compile_required_first": True,
    "compile_once_per_turn": True,
    "new_inputs_require_new_turn": True,
    "turns_are_immutable": True,
}


def _validate_relative_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path_value:
        raise ValueError("relative path must not be empty")
    if path.is_absolute():
        raise ValueError(f"path must be relative: {path_value}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"path contains invalid relative segments: {path_value}")
    return path


def _write_input_files(files: Dict[str, str], destination_root: Path) -> List[Path]:
    written: List[Path] = []
    destination_root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        relative = _validate_relative_path(rel_path)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(destination)
    return written


def _pick_single_cuda_source(generated: List[Path], reference: List[Path]) -> Path:
    if not reference:
        raise ValueError((
                "compile requires non-empty reference_files and generated_files. "
                "Do not compile with only generated files; upload both file groups for the turn."
            ),
        )
    if not generated:
        raise ValueError((
                "compile requires non-empty reference_files and generated_files. "
                "Do not compile with only reference files; upload both file groups for the turn."
            ),
        )

    generated_cu = [path for path in generated if path.suffix == ".cu"]

    if len(generated_cu) == 1:
        return generated_cu[0]
    if len(generated_cu) > 1:
        raise ValueError(
            "generated_files must contain exactly one .cu file. "
            f"Found {len(generated_cu)}: {[p.name for p in generated_cu]}"
        )
    raise ValueError(
        "generated_files must contain at least one .cu file."
    )


def _unique_paths(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _build_artifact(*, artifact_id: str, kind: str, path: str, description: str | None = None) -> dict:
    payload = {
        "artifact_id": artifact_id,
        "kind": kind,
        "path": path,
    }
    if description:
        payload["description"] = description
    return payload


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _attempt_tag(attempt: int) -> str:
    return f"attempt_{attempt:03d}"


def _stage_manifest_rel(stage: str, attempt: int) -> str:
    return f"state/{stage}.{_attempt_tag(attempt)}.json"


def _stage_manifest_path(workspace: dict, stage: str, attempt: int) -> Path:
    return Path(workspace["root_path"]) / _stage_manifest_rel(stage, attempt)


def _compile_manifest_path(workspace: dict) -> Path:
    return _stage_manifest_path(workspace, "compile", 1)


def _slugify(value: str) -> str:
    cleaned = SAFE_SLUG_RE.sub("_", value).strip("._-")
    return cleaned or "default"


def _config_suffix(config_slug: str) -> str:
    return f"config_{_slugify(config_slug)}"


def _config_state_rel(stage: str, attempt: int, config_slug: str) -> str:
    return f"state/configs/{stage}.{_attempt_tag(attempt)}.{_config_suffix(config_slug)}.json"


def _stage_log_rel(stage: str, attempt: int, config_slug: str | None = None) -> str:
    base = f"logs/{stage}.{_attempt_tag(attempt)}"
    if config_slug is not None:
        base += f".{_config_suffix(config_slug)}"
    return base + ".log"


def _compile_artifact_rel(attempt: int, stem: str, suffix: str) -> str:
    return f"artifacts/compile.{_attempt_tag(attempt)}.{_slugify(stem)}.{suffix}"


def _compile_log_rel(attempt: int, suffix: str) -> str:
    return f"logs/compile.{_attempt_tag(attempt)}.{suffix}.log"


def _config_artifact_rel(stage: str, attempt: int, config_slug: str, suffix: str) -> str:
    return f"artifacts/{stage}.{_attempt_tag(attempt)}.{_config_suffix(config_slug)}.{suffix}"


def _existing_attempts(workspace: dict, stage: str) -> List[int]:
    state_path = Path(workspace["state_path"])
    pattern = re.compile(rf"^{re.escape(stage)}\.attempt_(\d{{3}})\.json$")
    out: List[int] = []
    if not state_path.exists():
        return out
    for item in state_path.iterdir():
        if not item.is_file():
            continue
        match = pattern.match(item.name)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def _existing_log_attempts(workspace: dict, stage: str) -> List[int]:
    logs_path = Path(workspace["logs_path"])
    pattern = re.compile(rf"^{re.escape(stage)}\.attempt_(\d{{3}})\.log$")
    out: List[int] = []
    if not logs_path.exists():
        return out
    for item in logs_path.iterdir():
        if not item.is_file():
            continue
        match = pattern.match(item.name)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def _next_attempt(workspace: dict, stage: str, source: str = "state") -> int:
    attempts = _existing_attempts(workspace, stage) if source == "state" else _existing_log_attempts(workspace, stage)
    return (max(attempts) if attempts else 0) + 1


def _load_compile_manifest(workspace: dict) -> dict:
    manifest_path = _compile_manifest_path(workspace)
    if not manifest_path.exists():
        raise ValueError((
                "Workflow violation: compile must run first for this turn before trial/profile. "
                f"Missing compile state: {manifest_path}. Start with /compile, or use a new turn."
            ),
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _primary_artifact_from_manifest(workspace: dict) -> tuple[Path, dict]:
    manifest = _load_compile_manifest(workspace)
    requested_id = manifest.get("primary_artifact_id") or DEFAULT_COMPILE_ARTIFACT_ID
    for artifact in manifest.get("artifacts", []):
        if artifact.get("artifact_id") == requested_id:
            rel_path = artifact["path"]
            abs_path = (Path(workspace["root_path"]) / rel_path).resolve()
            if not abs_path.exists():
                raise ValueError(f"artifact path recorded in compile state does not exist: {abs_path}",
                )
            return abs_path, artifact
    raise ValueError(f"compile state is missing its primary artifact entry: {requested_id}",
    )


def _append_file_entry(result: dict, rel_path: str) -> None:
    existing_paths = {item["path"] for item in result.get("files", [])}
    file_entry = capture_turn_file(rel_path, result["workspace_path"])
    if file_entry["path"] not in existing_paths:
        result.setdefault("files", []).append(file_entry)


def _workflow_payload(metadata, *, stage: str, attempt: int, status: str, detail: str | None = None) -> dict:
    payload = {
        "metadata": metadata.model_dump(),
        "stage": stage,
        "attempt": attempt,
        "workflow": dict(WORKFLOW_RULES),
        "status": status,
    }
    if detail is not None:
        payload["detail"] = detail
    return payload


def _dedupe_files(files: List[dict]) -> List[dict]:
    seen: set[str] = set()
    out: List[dict] = []
    for item in files:
        path = item["path"]
        if path not in seen:
            out.append(item)
            seen.add(path)
    return out


def _dedupe_artifacts(artifacts: List[dict]) -> List[dict]:
    seen: set[tuple[str, str]] = set()
    out: List[dict] = []
    for item in artifacts:
        key = (item["artifact_id"], item["path"])
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def _status_from_run_result(run_result: dict) -> str:
    return "ok" if run_result["ok"] else "error"


def _parse_structured_stdout(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _config_metadata(config: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in config.items():
        if key == "extra" and isinstance(value, dict):
            meta.update(value)
        else:
            meta[key] = value
    return {k: v for k, v in meta.items() if v is not None}


def _fallback_performance_summary(run_result: dict, *, source: str, config: dict) -> dict:
    duration_ms = run_result["duration_seconds"] * 1000.0
    return {
        "metadata": {"source": source, **_config_metadata(config)},
        "latency_ms": {
            "min": duration_ms,
            "median": duration_ms,
            "max": duration_ms,
            "mean": duration_ms,
        },
        "runs": 1,
    }


def _trial_correctness_summary(run_result: dict, *, config: dict) -> dict:
    payload = _parse_structured_stdout(run_result["output"]["stdout"])
    if payload and isinstance(payload.get("comparison"), dict):
        comparison = payload["comparison"]
        correctness = comparison.get("correctness")
        if isinstance(correctness, dict):
            correctness.setdefault("metadata", {})
            correctness["metadata"] = {**_config_metadata(config), **correctness["metadata"]}
            return correctness
    if payload and isinstance(payload.get("correctness"), dict):
        correctness = payload["correctness"]
        correctness.setdefault("metadata", {})
        correctness["metadata"] = {**_config_metadata(config), **correctness["metadata"]}
        return correctness
    return {"metadata": {"source": "not_provided", **_config_metadata(config)}}


def _trial_performance_summary(run_result: dict, *, config: dict) -> dict:
    payload = _parse_structured_stdout(run_result["output"]["stdout"])
    if payload and isinstance(payload.get("generated"), dict):
        generated = payload["generated"]
        performance = generated.get("performance")
        if isinstance(performance, dict):
            performance.setdefault("metadata", {})
            performance["metadata"] = {**_config_metadata(config), **performance["metadata"]}
            return performance
    if payload and isinstance(payload.get("performance"), dict):
        performance = payload["performance"]
        performance.setdefault("metadata", {})
        performance["metadata"] = {**_config_metadata(config), **performance["metadata"]}
        return performance
    return _fallback_performance_summary(run_result, source="process_duration_fallback", config=config)



def _summarize_config_outputs(config_results: Dict[str, dict]) -> dict:
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    for config_slug, item in config_results.items():
        stdout_parts.append(f"=== config {config_slug} stdout ===\n{item['output']['stdout']}")
        stderr_parts.append(f"=== config {config_slug} stderr ===\n{item['output']['stderr']}")
    return {
        "stdout": "\n\n".join(stdout_parts),
        "stderr": "\n\n".join(stderr_parts),
    }


def _config_payload(config_slug: str, config: dict[str, Any]) -> dict[str, Any]:
    return {"slug": config_slug, "params": config}


def _write_config_record(workspace: dict, stage: str, attempt: int, config_slug: str, config: dict) -> str:
    rel_path = _config_state_rel(stage, attempt, config_slug)
    abs_path = Path(workspace["root_path"]) / rel_path
    _write_manifest(abs_path, {"config": _config_payload(config_slug, config)})
    return rel_path


def _config_env(
    workspace: dict,
    stage: str,
    attempt: int,
    config_slug: str,
    config: dict[str, Any],
    config_rel: str,
) -> Dict[str, str]:
    config_abs = str((Path(workspace["root_path"]) / config_rel).resolve())
    env: Dict[str, str] = {
        "CUDA_EXEC_STAGE": stage,
        "CUDA_EXEC_ATTEMPT": _attempt_tag(attempt),
        "CUDA_EXEC_CONFIG_ID": config_slug,
        "CUDA_EXEC_CONFIG_PATH": config_abs,
        "CUDA_EXEC_CONFIG_JSON": json.dumps(_config_payload(config_slug, config), sort_keys=True),
    }

    for key, value in config.items():
        if key == "extra" and isinstance(value, dict):
            for extra_key, extra_value in value.items():
                env_key = "CUDA_EXEC_EXTRA_" + _slugify(str(extra_key)).upper().replace(".", "_")
                env[env_key] = json.dumps(extra_value) if not isinstance(extra_value, str) else extra_value
            continue

        env_key = "CUDA_EXEC_PARAM_" + _slugify(str(key)).upper().replace(".", "_")
        if isinstance(value, bool):
            env[env_key] = "1" if value else "0"
        elif isinstance(value, (str, int, float)):
            env[env_key] = str(value)
        else:
            env[env_key] = json.dumps(value)

    return env


def _config_result_payload(
    *,
    config_slug: str,
    config: dict,
    run_result: dict,
    artifacts: List[dict] | None = None,
    correctness: dict | None = None,
    performance: dict | None = None,
    summary: dict | None = None,
) -> dict:
    payload = {
        "config": _config_payload(config_slug, config),
        "status": _status_from_run_result(run_result),
        "command": run_result["command"],
        "returncode": run_result["returncode"],
        "duration_seconds": run_result["duration_seconds"],
        "output": run_result["output"],
        "artifacts": list(artifacts or []),
        "files": list(run_result["files"]),
    }
    if correctness is not None:
        payload["correctness"] = correctness
    if performance is not None:
        payload["performance"] = performance
    if summary is not None:
        payload["summary"] = summary
    return payload


def _finalize_stage_result(
    *,
    metadata,
    workspace: dict,
    kind: str,
    attempt: int,
    command: List[str],
    stage_artifacts: List[dict],
    stage_files: List[dict],
    config_results: Dict[str, dict] | None = None,
    duration_seconds: float,
    returncode: int,
    all_ok: bool,
    output: dict,
) -> dict:
    return {
        "metadata": metadata.model_dump(),
        "all_ok": all_ok,
        "kind": kind,
        "attempt": attempt,
        "command": command,
        "turn_root": workspace["root_path"],
        "workspace_path": workspace["workspace_path"],
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "artifacts": _dedupe_artifacts(stage_artifacts),
        "output": output,
        "files": _dedupe_files(stage_files),
        "configs": dict(config_results or {}),
    }


def run_compile_task(
    *,
    metadata,
    timeout_seconds: int,
    reference_files: Dict[str, str],
    generated_files: Dict[str, str],
    cudnn_files: Dict[str, str] | None = None,
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    workspace_path = Path(workspace["workspace_path"])
    attempt = 1
    manifest_path = _compile_manifest_path(workspace)

    if manifest_path.exists() or _existing_attempts(workspace, "compile"):
        raise ValueError((
                "Workflow violation: compile may run only once per turn. "
                "Do not reuse the same turn to upload another file set. If you have new files or want a different compile input set, start a new turn and compile there."
            ),
        )

    _write_manifest(
        manifest_path,
        _workflow_payload(
            metadata,
            stage="compile",
            attempt=attempt,
            status="running",
            detail="compile has started for this turn",
        ),
    )

    started = time.perf_counter()
    try:
        copied_reference = _write_input_files(reference_files, workspace_path / "inputs" / "reference")
        copied_generated = _write_input_files(generated_files, workspace_path / "inputs" / "generated")

        if cudnn_files:
            copied_cudnn = _write_input_files(cudnn_files, workspace_path / "inputs" / "cudnn")
            # Find the .py entry point dynamically — any Python file is accepted
            cudnn_py_files = [p for p in copied_cudnn if p.suffix == ".py"]
            if not cudnn_py_files:
                raise ValueError("cudnn_files must include at least one .py entry point.")

        ref_py_files = [p for p in copied_reference if p.suffix == ".py"]
        if not ref_py_files:
            raise ValueError(
                "reference_files must include at least one .py entry point."
            )

        source = _pick_single_cuda_source(copied_generated, copied_reference)

        binary_rel = _compile_artifact_rel(attempt, source.stem, "bin")
        ptx_rel = _compile_artifact_rel(attempt, source.stem, "ptx")
        cubin_rel = _compile_artifact_rel(attempt, source.stem, "cubin")
        resource_usage_rel = _compile_artifact_rel(attempt, source.stem, "resource-usage.txt")
        sass_nvdisasm_rel = _compile_artifact_rel(attempt, source.stem, "nvdisasm.sass")
        binary_output = Path(workspace["root_path"]) / binary_rel
        command = [
            "/usr/bin/env",
            "bash",
            str(COMPILE_SCRIPT),
            "--source",
            str(source),
            "--output",
            str(binary_output),
        ]
        # Auto-detect harness mode: if the source exports kernel_run, link with harness
        if EVAL_HARNESS.exists() and 'kernel_run' in source.read_text(encoding="utf-8"):
            command.extend(["--harness", str(EVAL_HARNESS)])
        run_result = run_generic_command(
            kind="compile",
            command=command,
            workspace_path=str(workspace_path),
            env={},
            timeout_seconds=timeout_seconds,
            log_file=_stage_log_rel("compile", attempt),
            return_files=[
                ptx_rel,
                cubin_rel,
                resource_usage_rel,
                sass_nvdisasm_rel,
            ],
        )

        artifacts = [
            _build_artifact(
                artifact_id=DEFAULT_COMPILE_ARTIFACT_ID,
                kind="binary",
                path=binary_rel,
                description="Primary runnable binary artifact produced by compile",
            ),
            _build_artifact(
                artifact_id="compile:primary_ptx",
                kind="ptx",
                path=ptx_rel,
                description="Primary PTX artifact produced by the compile front-end",
            ),
            _build_artifact(
                artifact_id="compile:primary_cubin",
                kind="cubin",
                path=cubin_rel,
                description="Primary CUBIN artifact produced by ptxas",
            ),
            _build_artifact(
                artifact_id="compile:resource_usage",
                kind="report",
                path=resource_usage_rel,
                description="Resource-usage report extracted from the compiled CUBIN",
            ),
            _build_artifact(
                artifact_id="compile:sass_nvdisasm",
                kind="sass",
                path=sass_nvdisasm_rel,
                description="SASS dump generated by nvdisasm from the compiled CUBIN",
            ),
            _build_artifact(
                artifact_id="compile:state",
                kind="state",
                path=_stage_manifest_rel("compile", attempt),
                description="Compile manifest for this turn",
            ),
        ]

        manifest = _workflow_payload(
            metadata,
            stage="compile",
            attempt=attempt,
            status="ok" if run_result["ok"] else "error",
        )
        manifest.update(
            {
                "selected_source": str(source.relative_to(workspace_path)),
                "primary_artifact_id": DEFAULT_COMPILE_ARTIFACT_ID,
                "artifacts": artifacts,
            }
        )
        _write_manifest(manifest_path, manifest)
        _append_file_entry(run_result, _stage_manifest_rel("compile", attempt))

        return _finalize_stage_result(
            metadata=metadata,
            workspace=workspace,
            kind="compile",
            attempt=attempt,
            command=run_result["command"],
            stage_artifacts=artifacts,
            stage_files=run_result["files"],
            duration_seconds=time.perf_counter() - started,
            returncode=run_result["returncode"],
            all_ok=run_result["ok"],
            output=run_result["output"],
        )
    except (ValueError, FileNotFoundError) as exc:
        _write_manifest(
            manifest_path,
            _workflow_payload(
                metadata,
                stage="compile",
                attempt=attempt,
                status="error",
                detail=str(exc),
            ),
        )
        raise
    except Exception as exc:
        _write_manifest(
            manifest_path,
            _workflow_payload(
                metadata,
                stage="compile",
                attempt=attempt,
                status="error",
                detail=str(exc),
            ),
        )
        raise


def run_trial_task(
    *,
    metadata,
    timeout_seconds: int,
    configs: Dict[str, dict],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    target_path, target_artifact = _primary_artifact_from_manifest(workspace)
    workspace_path = Path(workspace["workspace_path"])
    attempt = _next_attempt(workspace, "trial")
    started = time.perf_counter()

    config_results: Dict[str, dict] = {}
    stage_files: List[dict] = []
    stage_artifacts: List[dict] = []

    _ts_fmt = "%H:%M:%S"
    for config_slug, config in configs.items():
        config_rel = _write_config_record(workspace, "trial", attempt, config_slug, config)
        comparison_rel = _config_artifact_rel("trial", attempt, config_slug, "comparison.json")
        command = [
            sys.executable,
            str(TRIAL_SCRIPT),
            "--run-tag",
            metadata.run_tag,
            "--version",
            metadata.version,
            "--direction-id",
            str(metadata.direction_id),
            "--direction-slug",
            metadata.direction_slug,
            "--turn",
            str(metadata.turn),
            "--config-slug",
            config_slug,
            "--config-json",
            json.dumps(config),
            "--timeout",
            str(timeout_seconds),
        ]
        cfg_start = datetime.now()
        logger.info("  trial config %s start [%s]", config_slug, cfg_start.strftime(_ts_fmt))
        run_result = run_generic_command(
            kind="trial",
            command=command,
            workspace_path=str(workspace_path),
            env={},
            timeout_seconds=timeout_seconds,
            return_files=[config_rel],
            log_file=_stage_log_rel("trial", attempt, config_slug),
        )
        cfg_end = datetime.now()
        cfg_dur = (cfg_end - cfg_start).total_seconds()
        logger.info("  trial config %s done  [%s] (%.1fs)", config_slug, cfg_end.strftime(_ts_fmt), cfg_dur)
        payload_json = _parse_structured_stdout(run_result["output"]["stdout"]) or {}
        comparison_path = Path(workspace["root_path"]) / comparison_rel
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(payload_json, indent=2) + "\n", encoding="utf-8")
        payload = _config_result_payload(
            config_slug=config_slug,
            config=config,
            run_result=run_result,
            artifacts=[
                _build_artifact(
                    artifact_id=f"trial:comparison:{config_slug}",
                    kind="comparison",
                    path=comparison_rel,
                    description=f"Reference/generated comparison payload for config {config_slug}",
                )
            ],
            correctness=_trial_correctness_summary(run_result, config=config),
            performance=_trial_performance_summary(run_result, config=config),
        )
        payload["comparison"] = payload_json.get("comparison", {}) if isinstance(payload_json, dict) else {}
        payload["reference"] = payload_json.get("reference", {}) if isinstance(payload_json, dict) else {}
        payload["generated"] = payload_json.get("generated", {}) if isinstance(payload_json, dict) else {}
        payload["cudnn"] = payload_json.get("cudnn", {}) if isinstance(payload_json, dict) else {}
        payload["files"].append(capture_turn_file(comparison_rel, str(workspace_path)))
        config_results[config_slug] = payload
        stage_files.extend(payload["files"])
        stage_artifacts.extend(payload["artifacts"])

    manifest = _workflow_payload(
        metadata,
        stage="trial",
        attempt=attempt,
        status="ok" if all(item["status"] == "ok" for item in config_results.values()) else "error",
    )
    manifest.update(
        {
            "input_artifact_id": target_artifact["artifact_id"],
            "input_path": target_artifact["path"],
            "configs": {config_slug: _config_payload(config_slug, config) for config_slug, config in configs.items()},
            "config_results": {
                config_slug: {
                    "status": item["status"],
                    "returncode": item["returncode"],
                    "duration_seconds": item["duration_seconds"],
                    "correctness": item.get("correctness", {}),
                    "performance": item.get("performance", {}),
                }
                for config_slug, item in config_results.items()
            },
            "artifacts": [
                _build_artifact(
                    artifact_id="trial:state",
                    kind="state",
                    path=_stage_manifest_rel("trial", attempt),
                    description="Trial manifest for this turn and attempt",
                )
            ],
        }
    )
    manifest_path = _stage_manifest_path(workspace, "trial", attempt)
    _write_manifest(manifest_path, manifest)
    stage_files.append(capture_turn_file(_stage_manifest_rel("trial", attempt), str(workspace_path)))
    stage_artifacts.extend(manifest["artifacts"])

    output = _summarize_config_outputs(config_results)
    overall_ok = all(item["status"] == "ok" for item in config_results.values())
    overall_returncode = next((item["returncode"] for item in config_results.values() if item["returncode"] != 0), 0)
    return _finalize_stage_result(
        metadata=metadata,
        workspace=workspace,
        kind="trial",
        attempt=attempt,
        command=[str(target_path)],
        stage_artifacts=stage_artifacts,
        stage_files=stage_files,
        config_results=config_results,
        duration_seconds=time.perf_counter() - started,
        returncode=overall_returncode,
        all_ok=overall_ok,
        output=output,
    )


def _strip_output_result(run_result: dict, stdout_log_path: Path | None) -> None:
    """Strip the large output.result array from the binary's JSON in stdout.

    The eval_harness binary prints all kernel result values to stdout.  For
    profiling those values are irrelevant — correctness is trial's job.
    Removes ``output.result`` from the in-memory stdout and the on-disk log.
    """
    stdout = run_result.get("output", {}).get("stdout", "") or ""
    if '"result"' not in stdout:
        return

    lines = stdout.split("\n")
    json_start = None
    json_end = None
    depth = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if json_start is None and stripped.startswith("{"):
            json_start = i
        if json_start is not None:
            depth += stripped.count("{") - stripped.count("}")
            if depth == 0:
                json_end = i
                break

    if json_start is None or json_end is None:
        return

    try:
        json_text = "\n".join(lines[json_start : json_end + 1])
        obj = json.loads(json_text)
        if "output" in obj and isinstance(obj["output"], dict):
            obj["output"].pop("result", None)
        cleaned_json = json.dumps(obj, indent=2)
        lines[json_start : json_end + 1] = [cleaned_json]
        cleaned_stdout = "\n".join(lines)
        run_result["output"]["stdout"] = cleaned_stdout

        # Overwrite the on-disk .stdout log file.
        if stdout_log_path and stdout_log_path.exists():
            old = stdout_log_path.read_text(encoding="utf-8")
            stdout_log_path.write_text(old.replace(json_text, cleaned_json), encoding="utf-8")
        # Also fix the combined .log file.
        if stdout_log_path:
            combined = stdout_log_path.with_suffix(".log")
            if combined.exists():
                old = combined.read_text(encoding="utf-8")
                if json_text in old:
                    combined.write_text(old.replace(json_text, cleaned_json), encoding="utf-8")
    except (json.JSONDecodeError, ValueError):
        pass  # Non-fatal


def run_profile_task(
    *,
    metadata,
    timeout_seconds: int,
    configs: Dict[str, dict],
    side: str = "generated",
) -> dict:
    """Run Nsight Compute profiling on the generated binary or reference Python kernel."""
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    target_path, target_artifact = _primary_artifact_from_manifest(workspace)
    workspace_path = Path(workspace["workspace_path"])
    attempt = _next_attempt(workspace, "profile")
    started = time.perf_counter()

    if side not in {"generated", "reference", "cudnn"}:
        raise ValueError(f"side must be 'generated', 'reference', or 'cudnn', got: {side}")

    config_results: Dict[str, dict] = {}
    stage_files: List[dict] = []
    stage_artifacts: List[dict] = []

    for config_slug, config in configs.items():
        config_rel = _write_config_record(workspace, "profile", attempt, config_slug, config)
        profile_json_rel = _config_artifact_rel("profile", attempt, config_slug, "summary.json")
        export_prefix_rel = _config_artifact_rel("profile", attempt, config_slug, "ncu")
        export_prefix_abs = str((Path(workspace["root_path"]) / export_prefix_rel).resolve())
        ncu_report_rel = f"{export_prefix_rel}.ncu-rep"

        if side == "generated":
            command = [
                "bash",
                str(PROFILE_NCU_SCRIPT),
                "--target", str(target_path),
                "--export-prefix", export_prefix_abs,
                "--set", "detailed",
            ]
        elif side == "reference":
            ref_dir = Path(workspace["workspace_path"]) / "inputs" / "reference"
            # Dynamically find reference entry point
            ref_py_files = sorted(ref_dir.glob("*.py")) if ref_dir.exists() else []
            reference_py = None
            for p in ref_py_files:
                if "class Model" in p.read_text(errors="ignore"):
                    reference_py = p
                    break
            if reference_py is None:
                raise ValueError(f"No reference .py entry point found in {ref_dir} — compile first to stage inputs")
            command = [
                "bash",
                str(PROFILE_NCU_SCRIPT),
                "--target", sys.executable, str(reference_py),
                "--export-prefix", export_prefix_abs,
                "--set", "detailed",
            ]
        else:  # side == "cudnn"
            cudnn_dir = Path(workspace["workspace_path"]) / "inputs" / "cudnn"
            cudnn_py_files = sorted(cudnn_dir.glob("*.py")) if cudnn_dir.exists() else []
            if not cudnn_py_files:
                raise ValueError(f"No .py entry point found in {cudnn_dir} — include cudnn_files in compile request")
            cudnn_entry = cudnn_py_files[0]
            command = [
                "bash",
                str(PROFILE_NCU_SCRIPT),
                "--target", sys.executable, str(cudnn_entry),
                "--export-prefix", export_prefix_abs,
                "--set", "detailed",
            ]

        stage_log_rel = _stage_log_rel("profile", attempt, config_slug)
        run_result = run_generic_command(
            kind="profile",
            command=command,
            workspace_path=str(workspace_path),
            env=_config_env(workspace, "profile", attempt, config_slug, config, config_rel),
            timeout_seconds=timeout_seconds,
            return_files=[config_rel, ncu_report_rel],
            log_file=stage_log_rel,
        )

        ncu_stdout = run_result.get("output", {}).get("stdout", "") or ""
        ncu_profiled = "No kernels were profiled." not in ncu_stdout and "No metrics to collect found in sections." not in ncu_stdout
        ncu_report_abs = Path(workspace["root_path"]) / ncu_report_rel
        ncu_report_exists = ncu_report_abs.exists()

        # --- Post-processing: strip output.result from binary stdout ---
        stdout_log_path = Path(workspace["root_path"]) / stage_log_rel.replace(".log", ".stdout")
        _strip_output_result(run_result, stdout_log_path)

        # --- Post-processing: generate curated NCU text report ---
        ncu_text_rel: str | None = None
        if ncu_report_exists:
            ncu_text_rel = _config_artifact_rel("profile", attempt, config_slug, "ncu-report.txt")
            ncu_text_abs = str((Path(workspace["root_path"]) / ncu_text_rel).resolve())
            try:
                subprocess.run(
                    [sys.executable, str(NCU_REPORT_SCRIPT), "--input", str(ncu_report_abs), "--output", ncu_text_abs],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
            except Exception:
                ncu_text_rel = None  # non-fatal — curated report generation failed

        ncu_summary: dict[str, Any] = {
            "side": side,
            "ncu_profiled": ncu_profiled,
            "ncu_report_exists": ncu_report_exists,
            "duration_seconds": run_result["duration_seconds"],
            "metadata": _config_metadata(config),
        }
        if ncu_report_exists:
            ncu_summary["ncu_report_path"] = ncu_report_rel
        if ncu_text_rel:
            ncu_summary["ncu_text_report_path"] = ncu_text_rel
        if not ncu_profiled:
            ncu_summary["ncu_warning"] = "ncu reported no profiled kernels or no metrics to collect"

        payload_json = {
            "metadata": metadata.model_dump(),
            "config_slug": config_slug,
            "side": side,
            "summary": ncu_summary,
        }
        profile_json_path = Path(workspace["root_path"]) / profile_json_rel
        profile_json_path.parent.mkdir(parents=True, exist_ok=True)
        profile_json_path.write_text(json.dumps(payload_json, indent=2) + "\n", encoding="utf-8")

        config_artifacts = [
            _build_artifact(
                artifact_id=f"profile:summary:{config_slug}",
                kind="profile_summary",
                path=profile_json_rel,
                description=f"NCU profile summary for config {config_slug}",
            ),
        ]
        if ncu_report_exists:
            config_artifacts.append(
                _build_artifact(
                    artifact_id=f"profile:ncu_report:{config_slug}",
                    kind="ncu_report",
                    path=ncu_report_rel,
                    description=f"Nsight Compute .ncu-rep report for config {config_slug}",
                )
            )
        if ncu_text_rel:
            config_artifacts.append(
                _build_artifact(
                    artifact_id=f"profile:ncu_text:{config_slug}",
                    kind="ncu_text_report",
                    path=ncu_text_rel,
                    description=f"Curated NCU text report for config {config_slug} (device__ deduplicated)",
                )
            )

        payload = _config_result_payload(
            config_slug=config_slug,
            config=config,
            run_result=run_result,
            artifacts=config_artifacts,
            summary=ncu_summary,
        )
        payload["files"].append(capture_turn_file(profile_json_rel, str(workspace_path)))
        if ncu_text_rel:
            payload["files"].append(capture_turn_file(ncu_text_rel, str(workspace_path)))
        config_results[config_slug] = payload
        stage_files.extend(payload["files"])
        stage_artifacts.extend(config_artifacts)

    manifest = _workflow_payload(
        metadata,
        stage="profile",
        attempt=attempt,
        status="ok" if all(item["status"] == "ok" for item in config_results.values()) else "error",
    )
    manifest.update(
        {
            "input_artifact_id": target_artifact["artifact_id"],
            "input_path": target_artifact["path"],
            "side": side,
            "configs": {config_slug: _config_payload(config_slug, config) for config_slug, config in configs.items()},
            "config_results": {
                config_slug: {
                    "status": item["status"],
                    "returncode": item["returncode"],
                    "duration_seconds": item["duration_seconds"],
                    "summary": item.get("summary", {}),
                    "artifacts": item["artifacts"],
                }
                for config_slug, item in config_results.items()
            },
            "artifacts": [
                _build_artifact(
                    artifact_id="profile:state",
                    kind="state",
                    path=_stage_manifest_rel("profile", attempt),
                    description="Profile manifest for this turn and attempt",
                ),
                *stage_artifacts,
            ],
        }
    )
    manifest_path = _stage_manifest_path(workspace, "profile", attempt)
    _write_manifest(manifest_path, manifest)
    stage_files.append(capture_turn_file(_stage_manifest_rel("profile", attempt), str(workspace_path)))
    stage_artifacts = manifest["artifacts"]

    output = _summarize_config_outputs(config_results)
    overall_ok = all(item["status"] == "ok" for item in config_results.values())
    overall_returncode = next((item["returncode"] for item in config_results.values() if item["returncode"] != 0), 0)
    return _finalize_stage_result(
        metadata=metadata,
        workspace=workspace,
        kind="profile",
        attempt=attempt,
        command=[str(target_path)],
        stage_artifacts=stage_artifacts,
        stage_files=stage_files,
        config_results=config_results,
        duration_seconds=time.perf_counter() - started,
        returncode=overall_returncode,
        all_ok=overall_ok,
        output=output,
    )


def run_execute_task(
    *,
    metadata,
    timeout_seconds: int,
    command: List[str],
    env: Dict[str, str],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    workspace_path = Path(workspace["workspace_path"])
    attempt = _next_attempt(workspace, "execute", source="logs")
    started = time.perf_counter()

    run_result = run_cuda_command(
        kind="execute",
        command=command,
        workspace_path=str(workspace_path),
        env=env,
        timeout_seconds=timeout_seconds,
        log_file=_stage_log_rel("execute", attempt),
    )

    return _finalize_stage_result(
        metadata=metadata,
        workspace=workspace,
        kind="execute",
        attempt=attempt,
        command=run_result["command"],
        stage_artifacts=[],
        stage_files=list(run_result["files"]),
        duration_seconds=time.perf_counter() - started,
        returncode=run_result["returncode"],
        all_ok=run_result["ok"],
        output=run_result["output"],
    )


# ---------------------------------------------------------------------------
# Public endpoint functions (previously in main.py)
# ---------------------------------------------------------------------------


def _stage_log_paths(stage: str, attempt: int, config_slug: str | None = None) -> list[str]:
    """Return public log relative paths for a stage attempt."""
    base = f"logs/{stage}.{_attempt_tag(attempt)}"
    if config_slug is not None:
        base += f".{_config_suffix(config_slug)}"
    return [f"{base}.log", f"{base}.stdout", f"{base}.stderr"]


def _compile_artifact_map(result: dict) -> dict[str, str]:
    artifact_map: dict[str, str] = {}
    for artifact in result.get("artifacts", []):
        artifact_id = artifact.get("artifact_id")
        if artifact_id == "compile:primary_binary":
            artifact_map["binary"] = artifact["path"]
        elif artifact_id == "compile:primary_ptx":
            artifact_map["ptx"] = artifact["path"]
        elif artifact_id == "compile:primary_cubin":
            artifact_map["cubin"] = artifact["path"]
        elif artifact_id == "compile:resource_usage":
            artifact_map["resource_usage"] = artifact["path"]
        elif artifact_id == "compile:sass_nvdisasm":
            artifact_map["sass_nvdisasm"] = artifact["path"]
    if not artifact_map:
        raise ValueError("compile result missing public compile artifacts")
    return artifact_map


def _capture_public_file(workspace_path: str, rel_path: str, *, inline: bool, max_bytes: int | None = None) -> dict | None:
    return _capture_public_files(workspace_path, [rel_path], inline=inline, max_bytes=max_bytes).get(rel_path)


def _capture_public_files(workspace_path: str, rel_paths: list[str], *, inline: bool, max_bytes: int | None = None) -> dict[str, dict]:
    """Materialize public response files as relative_path -> FilePayload."""
    payload: dict[str, dict] = {}
    for rel_path in rel_paths:
        item = capture_turn_file(rel_path, workspace_path, max_bytes=max_bytes)
        if not item.get("exists") or item.get("error"):
            continue
        entry = {
            "path": rel_path,
            "inline": inline,
            "truncated": bool(item.get("truncated", False)),
        }
        if inline:
            entry["content"] = item.get("content") or ""
            entry["encoding"] = item.get("encoding") or "utf8"
        payload[rel_path] = entry
    return payload


def compile_endpoint(request: CompileRequest) -> CompileResponse:
    """Compile inline code inputs into kept compile artifacts plus logs."""
    result = run_compile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        reference_files=request.reference_files,
        generated_files=request.generated_files,
        cudnn_files=request.cudnn_files or None,
    )
    attempt = result["attempt"]
    artifact_map = _compile_artifact_map(result)
    workspace_path = result["workspace_path"]
    return CompileResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        artifacts=CompileArtifacts(
            binary=_capture_public_file(workspace_path, artifact_map["binary"], inline=False) if "binary" in artifact_map else None,
            ptx=_capture_public_file(workspace_path, artifact_map["ptx"], inline=False) if "ptx" in artifact_map else None,
            cubin=_capture_public_file(workspace_path, artifact_map["cubin"], inline=False) if "cubin" in artifact_map else None,
            resource_usage=_capture_public_file(workspace_path, artifact_map["resource_usage"], inline=False) if "resource_usage" in artifact_map else None,
            sass=CompileSassArtifacts(
                nvdisasm=_capture_public_file(workspace_path, artifact_map["sass_nvdisasm"], inline=False) if "sass_nvdisasm" in artifact_map else None,
            ),
        ),
        tool_outputs=CompileToolOutputs(
            nvcc_ptx=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvcc-ptx.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvcc-ptx.stderr", inline=True),
            ),
            ptxas=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.ptxas.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.ptxas.stderr", inline=True),
            ),
            resource_usage=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.resource-usage.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.resource-usage.stderr", inline=True),
            ),
            nvdisasm=ToolIOPair(
                stdout=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvdisasm.stdout", inline=True),
                stderr=_capture_public_file(workspace_path, f"logs/compile.{_attempt_tag(attempt)}.nvdisasm.stderr", inline=True),
            ),
        ),
    )


def file_read_endpoint(request: FileReadRequest) -> FileReadResponse:
    """Read one turn-relative file from artifacts/, logs/, or state/."""
    _validate_relative_path(request.path)
    allowed_prefixes = ("artifacts/", "logs/", "state/")
    if not request.path.startswith(allowed_prefixes):
        raise ValueError("file reads are limited to artifacts/, logs/, and state/ paths under the resolved turn root")
    workspace = resolve_workspace_bundle(**request.metadata.model_dump())
    file_payload = _capture_public_file(workspace["workspace_path"], request.path, inline=True, max_bytes=request.max_bytes)
    if file_payload is None:
        raise FileNotFoundError(f"file not found for this turn/path: {request.path}")
    return FileReadResponse(metadata=request.metadata, file=file_payload)


def trial_endpoint(request: TrialRequest) -> TrialResponse:
    """Trial one compiled artifact against slug-keyed runtime configs."""
    result = run_trial_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
    )
    attempt = result["attempt"]
    items = {
        config_slug: TrialConfigOutput(
            status=item["status"],
            reference=item.get("reference") or {},
            generated=item.get("generated") or {},
            cudnn=item.get("cudnn") or {},
            correctness=item.get("correctness", {}),
            performance=item.get("performance", {}),
            artifacts=_capture_public_files(
                result["workspace_path"],
                [artifact["path"] for artifact in item.get("artifacts", [])],
                inline=True,
            ),
            logs=_capture_public_files(
                result["workspace_path"],
                _stage_log_paths("trial", attempt, config_slug),
                inline=True,
            ),
        )
        for config_slug, item in result.get("configs", {}).items()
    }
    return TrialResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        configs=items,
    )


def profile_endpoint(request: ProfileRequest) -> ProfileResponse:
    """NCU-profile a compiled artifact or reference kernel against slug-keyed runtime configs."""
    result = run_profile_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        configs=request.configs,
        side=request.side,
    )
    attempt = result["attempt"]
    items = {
        config_slug: ProfileConfigOutput(
            status=item["status"],
            summary=item.get("summary", {}),
            artifacts=_capture_public_files(
                result["workspace_path"],
                [artifact["path"] for artifact in item.get("artifacts", [])],
                inline=True,
            ),
            logs=_capture_public_files(
                result["workspace_path"],
                _stage_log_paths("profile", attempt, config_slug),
                inline=True,
            ),
        )
        for config_slug, item in result.get("configs", {}).items()
    }
    return ProfileResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        configs=items,
    )


def execute_endpoint(request: ExecuteRequest) -> ExecuteResponse:
    """Run a generic CUDA-tool command and return logs only."""
    result = run_execute_task(
        metadata=request.metadata,
        timeout_seconds=request.timeout_seconds,
        command=request.command,
        env=request.env,
    )
    attempt = result["attempt"]
    return ExecuteResponse(
        metadata=request.metadata,
        all_ok=result["all_ok"],
        attempt=attempt,
        logs=_capture_public_files(result["workspace_path"], _stage_log_paths("execute", attempt), inline=True),
    )
