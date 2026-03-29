from __future__ import annotations

import json
import re
import time
from pathlib import Path
import sys
from typing import Any, Dict, List

from fastapi import HTTPException

from cuda_exec.runner import (
    capture_turn_file,
    resolve_workspace_bundle,
    run_cuda_command,
    run_generic_command,
)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
COMPILE_SCRIPT = SCRIPTS_DIR / "compile.sh"
EVALUATE_SCRIPT = SCRIPTS_DIR / "evaluate.py"
PROFILE_SCRIPT = SCRIPTS_DIR / "profile.py"
PROFILE_NCU_SCRIPT = SCRIPTS_DIR / "profile.sh"
EVAL_HARNESS = SCRIPTS_DIR / "eval_harness.cu"
DEFAULT_COMPILE_ARTIFACT_ID = "compile:primary_binary"
SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
WORKFLOW_RULES = {
    "compile_required_first": True,
    "compile_once_per_turn": True,
    "new_inputs_require_new_turn": True,
    "turns_are_immutable": True,
}


def _validate_relative_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path_value:
        raise HTTPException(status_code=400, detail="relative path must not be empty")
    if path.is_absolute():
        raise HTTPException(status_code=400, detail=f"path must be relative: {path_value}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise HTTPException(status_code=400, detail=f"path contains invalid relative segments: {path_value}")
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
        raise HTTPException(
            status_code=400,
            detail=(
                "compile requires non-empty reference_files and generated_files. "
                "Do not compile with only generated files; upload both file groups for the turn."
            ),
        )
    if not generated:
        raise HTTPException(
            status_code=400,
            detail=(
                "compile requires non-empty reference_files and generated_files. "
                "Do not compile with only reference files; upload both file groups for the turn."
            ),
        )

    generated_cu = [path for path in generated if path.suffix == ".cu"]

    if len(generated_cu) == 1:
        if generated_cu[0].name != "generated.cu":
            raise HTTPException(
                status_code=400,
                detail=(
                    "the .cu entry file in generated_files must be named generated.cu. "
                    "Rename your CUDA source to generated.cu and resubmit. "
                    "Additional header or helper files may use any name."
                ),
            )
        return generated_cu[0]
    if len(generated_cu) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "generated_files must contain exactly one .cu file. We recommend a generator. "
                "Include a single generated .cu file for the optimized kernel, plus any number of headers or inline helper files if needed."
            ),
        )
    raise HTTPException(
        status_code=400,
        detail=(
            "generated_files must contain exactly one .cu file. We recommend a generator. "
            "Include one generated .cu file for compile; reference_files may include supporting sources, headers, or non-.cu inputs."
        ),
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
        raise HTTPException(
            status_code=400,
            detail=(
                "Workflow violation: compile must run first for this turn before evaluate/profile. "
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
                raise HTTPException(
                    status_code=400,
                    detail=f"artifact path recorded in compile state does not exist: {abs_path}",
                )
            return abs_path, artifact
    raise HTTPException(
        status_code=400,
        detail=f"compile state is missing its primary artifact entry: {requested_id}",
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


def _evaluate_correctness_summary(run_result: dict, *, config: dict) -> dict:
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


def _evaluate_performance_summary(run_result: dict, *, config: dict) -> dict:
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


def _profile_summary(run_result: dict, *, config: dict) -> dict:
    payload = _parse_structured_stdout(run_result["output"]["stdout"])
    if payload and isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
        summary.setdefault("metadata", {})
        summary["metadata"] = {**_config_metadata(config), **summary["metadata"]}
        return summary
    if payload and isinstance(payload.get("performance"), dict):
        summary = payload["performance"]
        summary.setdefault("metadata", {})
        summary["metadata"] = {**_config_metadata(config), **summary["metadata"]}
        return summary
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
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    workspace_path = Path(workspace["workspace_path"])
    attempt = 1
    manifest_path = _compile_manifest_path(workspace)

    if manifest_path.exists() or _existing_attempts(workspace, "compile"):
        raise HTTPException(
            status_code=409,
            detail=(
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

        if not any(path.name == "reference.py" for path in copied_reference):
            raise HTTPException(
                status_code=400,
                detail=(
                    "reference_files must include a file named reference.py as the entry point. "
                    "Rename your reference module to reference.py and resubmit. "
                    "Additional helper files may use any name."
                ),
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
    except HTTPException as exc:
        _write_manifest(
            manifest_path,
            _workflow_payload(
                metadata,
                stage="compile",
                attempt=attempt,
                status="error",
                detail=str(exc.detail),
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


def run_evaluate_task(
    *,
    metadata,
    timeout_seconds: int,
    configs: Dict[str, dict],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    target_path, target_artifact = _primary_artifact_from_manifest(workspace)
    workspace_path = Path(workspace["workspace_path"])
    attempt = _next_attempt(workspace, "evaluate")
    started = time.perf_counter()

    config_results: Dict[str, dict] = {}
    stage_files: List[dict] = []
    stage_artifacts: List[dict] = []

    for config_slug, config in configs.items():
        config_rel = _write_config_record(workspace, "evaluate", attempt, config_slug, config)
        comparison_rel = _config_artifact_rel("evaluate", attempt, config_slug, "comparison.json")
        command = [
            sys.executable,
            str(EVALUATE_SCRIPT),
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
        run_result = run_generic_command(
            kind="evaluate",
            command=command,
            workspace_path=str(workspace_path),
            env={},
            timeout_seconds=timeout_seconds,
            return_files=[config_rel],
            log_file=_stage_log_rel("evaluate", attempt, config_slug),
        )
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
                    artifact_id=f"evaluate:comparison:{config_slug}",
                    kind="comparison",
                    path=comparison_rel,
                    description=f"Reference/generated comparison payload for config {config_slug}",
                )
            ],
            correctness=_evaluate_correctness_summary(run_result, config=config),
            performance=_evaluate_performance_summary(run_result, config=config),
        )
        payload["comparison"] = payload_json.get("comparison", {}) if isinstance(payload_json, dict) else {}
        payload["reference"] = payload_json.get("reference", {}) if isinstance(payload_json, dict) else {}
        payload["generated"] = payload_json.get("generated", {}) if isinstance(payload_json, dict) else {}
        payload["files"].append(capture_turn_file(comparison_rel, str(workspace_path)))
        config_results[config_slug] = payload
        stage_files.extend(payload["files"])
        stage_artifacts.extend(payload["artifacts"])

    manifest = _workflow_payload(
        metadata,
        stage="evaluate",
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
                    artifact_id="evaluate:state",
                    kind="state",
                    path=_stage_manifest_rel("evaluate", attempt),
                    description="Evaluate manifest for this turn and attempt",
                )
            ],
        }
    )
    manifest_path = _stage_manifest_path(workspace, "evaluate", attempt)
    _write_manifest(manifest_path, manifest)
    stage_files.append(capture_turn_file(_stage_manifest_rel("evaluate", attempt), str(workspace_path)))
    stage_artifacts.extend(manifest["artifacts"])

    output = _summarize_config_outputs(config_results)
    overall_ok = all(item["status"] == "ok" for item in config_results.values())
    overall_returncode = next((item["returncode"] for item in config_results.values() if item["returncode"] != 0), 0)
    return _finalize_stage_result(
        metadata=metadata,
        workspace=workspace,
        kind="evaluate",
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


def run_profile_task(
    *,
    metadata,
    timeout_seconds: int,
    configs: Dict[str, dict],
    mode: str = "generated_only",
    profiler_backend: str = "comparison_runtime",
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    target_path, target_artifact = _primary_artifact_from_manifest(workspace)
    workspace_path = Path(workspace["workspace_path"])
    attempt = _next_attempt(workspace, "profile")
    started = time.perf_counter()

    config_results: Dict[str, dict] = {}
    stage_files: List[dict] = []
    stage_artifacts: List[dict] = []

    if profiler_backend not in {"comparison_runtime", "ncu"}:
        raise HTTPException(status_code=400, detail=f"unsupported profiler_backend: {profiler_backend}")
    if profiler_backend == "ncu" and mode != "generated_only":
        raise HTTPException(
            status_code=400,
            detail="profiler_backend=ncu currently supports only mode=generated_only",
        )

    for config_slug, config in configs.items():
        config_rel = _write_config_record(workspace, "profile", attempt, config_slug, config)
        profile_json_rel = _config_artifact_rel("profile", attempt, config_slug, "summary.json")
        config_artifacts: List[dict] = []

        if profiler_backend == "comparison_runtime":
            command = [
                sys.executable,
                str(PROFILE_SCRIPT),
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
                "--mode",
                mode,
                "--timeout",
                str(timeout_seconds),
            ]
            run_result = run_generic_command(
                kind="profile",
                command=command,
                workspace_path=str(workspace_path),
                env={},
                timeout_seconds=timeout_seconds,
                return_files=[config_rel],
                log_file=_stage_log_rel("profile", attempt, config_slug),
            )
            payload_json = _parse_structured_stdout(run_result["output"]["stdout"]) or {}
            profile_json_path = Path(workspace["root_path"]) / profile_json_rel
            profile_json_path.parent.mkdir(parents=True, exist_ok=True)
            profile_json_path.write_text(json.dumps(payload_json, indent=2) + "\n", encoding="utf-8")
            config_artifacts = [
                _build_artifact(
                    artifact_id=f"profile:summary:{config_slug}",
                    kind="profile_summary",
                    path=profile_json_rel,
                    description=f"Structured profile payload for config {config_slug}",
                )
            ]
            payload = _config_result_payload(
                config_slug=config_slug,
                config=config,
                run_result=run_result,
                artifacts=config_artifacts,
                summary=_profile_summary(run_result, config=config),
            )
            payload["mode"] = payload_json.get("mode", mode) if isinstance(payload_json, dict) else mode
            payload["reference"] = payload_json.get("reference", {}) if isinstance(payload_json, dict) else {}
            payload["generated"] = payload_json.get("generated", {}) if isinstance(payload_json, dict) else {}
        else:
            export_prefix_rel = _config_artifact_rel("profile", attempt, config_slug, "ncu")
            export_prefix_abs = str((Path(workspace["root_path"]) / export_prefix_rel).resolve())
            ncu_report_rel = f"{export_prefix_rel}.ncu-rep"
            command = [
                "bash",
                str(PROFILE_NCU_SCRIPT),
                "--target",
                str(target_path),
                "--export-prefix",
                export_prefix_abs,
            ]
            run_result = run_generic_command(
                kind="profile",
                command=command,
                workspace_path=str(workspace_path),
                env=_config_env(workspace, "profile", attempt, config_slug, config, config_rel),
                timeout_seconds=timeout_seconds,
                return_files=[config_rel, ncu_report_rel],
                log_file=_stage_log_rel("profile", attempt, config_slug),
            )
            ncu_stdout = run_result.get("output", {}).get("stdout", "") or ""
            ncu_profiled = "No kernels were profiled." not in ncu_stdout and "No metrics to collect found in sections." not in ncu_stdout
            ncu_report_abs = Path(workspace["root_path"]) / ncu_report_rel
            ncu_report_exists = ncu_report_abs.exists()
            ncu_summary = _fallback_performance_summary(run_result, source="ncu_process_duration_fallback", config=config)
            ncu_summary["metadata"] = {
                **config,
                **ncu_summary.get("metadata", {}),
                "profiler_backend": "ncu",
                "ncu_profiled": ncu_profiled,
                "ncu_report_exists": ncu_report_exists,
            }
            if not ncu_profiled:
                ncu_summary["metadata"]["ncu_warning"] = "ncu reported no profiled kernels or no metrics to collect"
            payload_json = {
                "metadata": metadata.model_dump(),
                "config_slug": config_slug,
                "mode": mode,
                "profiler_backend": profiler_backend,
                "generated": {
                    "summary": ncu_summary,
                },
                "summary": ncu_summary,
            }
            if ncu_report_exists:
                payload_json["generated"]["ncu_report"] = {"path": ncu_report_rel}
            profile_json_path = Path(workspace["root_path"]) / profile_json_rel
            profile_json_path.parent.mkdir(parents=True, exist_ok=True)
            profile_json_path.write_text(json.dumps(payload_json, indent=2) + "\n", encoding="utf-8")
            config_artifacts = [
                _build_artifact(
                    artifact_id=f"profile:summary:{config_slug}",
                    kind="profile_summary",
                    path=profile_json_rel,
                    description=f"Structured profile payload for config {config_slug}",
                ),
            ]
            if ncu_report_exists:
                config_artifacts.append(
                    _build_artifact(
                        artifact_id=f"profile:ncu_report:{config_slug}",
                        kind="ncu_report",
                        path=ncu_report_rel,
                        description=f"Nsight Compute report for config {config_slug}",
                    )
                )
            payload = _config_result_payload(
                config_slug=config_slug,
                config=config,
                run_result=run_result,
                artifacts=config_artifacts,
                summary=payload_json["summary"],
            )
            payload["mode"] = mode
            payload["profiler_backend"] = profiler_backend
            payload["reference"] = {}
            payload["generated"] = payload_json.get("generated", {})

        payload["profiler_backend"] = profiler_backend
        payload["files"].append(capture_turn_file(profile_json_rel, str(workspace_path)))
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
            "profiler": profiler_backend,
            "mode": mode,
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
