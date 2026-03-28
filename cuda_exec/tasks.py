from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from cuda_exec.models import RuntimeConfig
from cuda_exec.runner import (
    capture_turn_file,
    resolve_workspace_bundle,
    run_cuda_command,
    run_generic_command,
)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
COMPILE_SCRIPT = SCRIPTS_DIR / "compile.sh"
PROFILE_SCRIPT = SCRIPTS_DIR / "profile.sh"
DEFAULT_COMPILE_ARTIFACT_ID = "compile:primary_binary"
SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
WORKFLOW_RULES = {
    "compile_required_first": True,
    "compile_once_per_turn": True,
    "new_inputs_require_new_turn": True,
    "turns_are_immutable": True,
}


def _absolute_input_path(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"input file does not exist: {path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"input path is not a regular file: {path}")
    return path


def _copy_inputs(paths: List[str], destination_root: Path) -> List[Path]:
    copied: List[Path] = []
    destination_root.mkdir(parents=True, exist_ok=True)
    for item in paths:
        source = _absolute_input_path(item)
        destination = destination_root / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _pick_single_cuda_source(generated: List[Path], original: List[Path]) -> Path:
    generated_cu = [path for path in generated if path.suffix == ".cu"]
    original_cu = [path for path in original if path.suffix == ".cu"]

    if len(generated_cu) == 1:
        return generated_cu[0]
    if len(generated_cu) > 1:
        raise HTTPException(
            status_code=400,
            detail="compile expects at most one generated .cu file in this hardened flow",
        )
    if len(original_cu) == 1:
        return original_cu[0]
    if len(original_cu) > 1:
        raise HTTPException(
            status_code=400,
            detail="compile expects at most one original .cu file in this hardened flow",
        )
    raise HTTPException(
        status_code=400,
        detail="compile could not find a single .cu source file in generated_files or original_files",
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


def _config_suffix(config: RuntimeConfig) -> str:
    return f"config_{_slugify(config.config_id)}"


def _config_state_rel(stage: str, attempt: int, config: RuntimeConfig) -> str:
    return f"state/configs/{stage}.{_attempt_tag(attempt)}.{_config_suffix(config)}.json"


def _stage_log_rel(stage: str, attempt: int, config: RuntimeConfig | None = None) -> str:
    base = f"logs/{stage}.{_attempt_tag(attempt)}"
    if config is not None:
        base += f".{_config_suffix(config)}"
    return base + ".log"


def _compile_artifact_rel(attempt: int, stem: str) -> str:
    return f"artifacts/compile.{_attempt_tag(attempt)}.{_slugify(stem)}.bin"


def _config_artifact_rel(stage: str, attempt: int, config: RuntimeConfig, suffix: str) -> str:
    return f"artifacts/{stage}.{_attempt_tag(attempt)}.{_config_suffix(config)}.{suffix}"


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


def _summarize_config_outputs(config_results: List[dict]) -> dict:
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    for item in config_results:
        config_id = item["config"]["config_id"]
        stdout_parts.append(f"=== config {config_id} stdout ===\n{item['output']['stdout']}")
        stderr_parts.append(f"=== config {config_id} stderr ===\n{item['output']['stderr']}")
    return {
        "stdout": "\n\n".join(stdout_parts),
        "stderr": "\n\n".join(stderr_parts),
    }


def _config_payload(config: RuntimeConfig) -> dict:
    return config.model_dump()


def _write_config_record(workspace: dict, stage: str, attempt: int, config: RuntimeConfig) -> str:
    rel_path = _config_state_rel(stage, attempt, config)
    abs_path = Path(workspace["root_path"]) / rel_path
    _write_manifest(abs_path, {"config": _config_payload(config)})
    return rel_path


def _config_env(workspace: dict, stage: str, attempt: int, config: RuntimeConfig, config_rel: str) -> Dict[str, str]:
    config_abs = str((Path(workspace["root_path"]) / config_rel).resolve())
    env: Dict[str, str] = {
        "CUDA_EXEC_STAGE": stage,
        "CUDA_EXEC_ATTEMPT": _attempt_tag(attempt),
        "CUDA_EXEC_CONFIG_ID": config.config_id,
        "CUDA_EXEC_CONFIG_PATH": config_abs,
        "CUDA_EXEC_CONFIG_JSON": json.dumps(_config_payload(config), sort_keys=True),
    }
    if config.num_layers is not None:
        env["CUDA_EXEC_NUM_LAYERS"] = str(config.num_layers)
    if config.embedding_size is not None:
        env["CUDA_EXEC_EMBEDDING_SIZE"] = str(config.embedding_size)
    if config.num_heads is not None:
        env["CUDA_EXEC_NUM_HEADS"] = str(config.num_heads)
    if config.causal is not None:
        env["CUDA_EXEC_CAUSAL"] = "1" if config.causal else "0"
    for key, value in config.extra.items():
        env_key = "CUDA_EXEC_EXTRA_" + _slugify(str(key)).upper().replace(".", "_")
        env[env_key] = json.dumps(value) if not isinstance(value, str) else value
    return env


def _config_result_payload(
    *,
    config: RuntimeConfig,
    run_result: dict,
    artifacts: List[dict] | None = None,
) -> dict:
    return {
        "config": _config_payload(config),
        "ok": run_result["ok"],
        "command": run_result["command"],
        "returncode": run_result["returncode"],
        "duration_seconds": run_result["duration_seconds"],
        "output": run_result["output"],
        "artifacts": list(artifacts or []),
        "files": list(run_result["files"]),
    }


def _finalize_stage_result(
    *,
    metadata,
    workspace: dict,
    kind: str,
    attempt: int,
    command: List[str],
    stage_artifacts: List[dict],
    stage_files: List[dict],
    config_results: List[dict] | None = None,
    duration_seconds: float,
    returncode: int,
    ok: bool,
    output: dict,
) -> dict:
    return {
        "metadata": metadata.model_dump(),
        "ok": ok,
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
        "config_results": list(config_results or []),
    }


def run_compile_task(
    *,
    metadata,
    timeout_seconds: int,
    original_files: List[str],
    generated_files: List[str],
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
                "If you have new files or want to retry with different inputs, start a new turn."
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
        copied_original = _copy_inputs(original_files, workspace_path / "inputs" / "original")
        copied_generated = _copy_inputs(generated_files, workspace_path / "inputs" / "generated")
        source = _pick_single_cuda_source(copied_generated, copied_original)

        binary_rel = _compile_artifact_rel(attempt, source.stem)
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
        run_result = run_generic_command(
            kind="compile",
            command=command,
            workspace_path=str(workspace_path),
            env={},
            timeout_seconds=timeout_seconds,
            log_file=_stage_log_rel("compile", attempt),
        )

        artifacts = [
            _build_artifact(
                artifact_id=DEFAULT_COMPILE_ARTIFACT_ID,
                kind="binary",
                path=binary_rel,
                description="Default executable artifact produced by compile",
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
            ok=run_result["ok"],
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
    configs: List[RuntimeConfig],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    target_path, target_artifact = _primary_artifact_from_manifest(workspace)
    workspace_path = Path(workspace["workspace_path"])
    attempt = _next_attempt(workspace, "evaluate")
    started = time.perf_counter()

    config_results: List[dict] = []
    stage_files: List[dict] = []
    stage_artifacts: List[dict] = []

    for config in configs:
        config_rel = _write_config_record(workspace, "evaluate", attempt, config)
        env = _config_env(workspace, "evaluate", attempt, config, config_rel)
        run_result = run_generic_command(
            kind="evaluate",
            command=[str(target_path)],
            workspace_path=str(workspace_path),
            env=env,
            timeout_seconds=timeout_seconds,
            return_files=[config_rel],
            log_file=_stage_log_rel("evaluate", attempt, config),
        )
        payload = _config_result_payload(config=config, run_result=run_result)
        config_results.append(payload)
        stage_files.extend(payload["files"])

    manifest = _workflow_payload(
        metadata,
        stage="evaluate",
        attempt=attempt,
        status="ok" if all(item["ok"] for item in config_results) else "error",
    )
    manifest.update(
        {
            "input_artifact_id": target_artifact["artifact_id"],
            "input_path": target_artifact["path"],
            "configs": [_config_payload(config) for config in configs],
            "config_results": [
                {
                    "config_id": item["config"]["config_id"],
                    "ok": item["ok"],
                    "returncode": item["returncode"],
                    "duration_seconds": item["duration_seconds"],
                }
                for item in config_results
            ],
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
    overall_ok = all(item["ok"] for item in config_results)
    overall_returncode = next((item["returncode"] for item in config_results if item["returncode"] != 0), 0)
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
        ok=overall_ok,
        output=output,
    )


def run_profile_task(
    *,
    metadata,
    timeout_seconds: int,
    configs: List[RuntimeConfig],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    target_path, target_artifact = _primary_artifact_from_manifest(workspace)
    workspace_path = Path(workspace["workspace_path"])
    attempt = _next_attempt(workspace, "profile")
    started = time.perf_counter()

    config_results: List[dict] = []
    stage_files: List[dict] = []
    stage_artifacts: List[dict] = []

    for config in configs:
        config_rel = _write_config_record(workspace, "profile", attempt, config)
        env = _config_env(workspace, "profile", attempt, config, config_rel)
        report_base_rel = _config_artifact_rel("profile", attempt, config, "ncu")
        report_file_rel = report_base_rel + ".ncu-rep"
        report_prefix = Path(workspace["root_path"]) / report_base_rel
        command = [
            "/usr/bin/env",
            "bash",
            str(PROFILE_SCRIPT),
            "--target",
            str(target_path),
            "--export-prefix",
            str(report_prefix),
        ]
        run_result = run_generic_command(
            kind="profile",
            command=command,
            workspace_path=str(workspace_path),
            env=env,
            timeout_seconds=timeout_seconds,
            return_files=[config_rel],
            log_file=_stage_log_rel("profile", attempt, config),
        )
        config_artifacts = [
            _build_artifact(
                artifact_id=f"profile:report:{config.config_id}",
                kind="profile_report",
                path=report_file_rel,
                description=f"NCU report for config {config.config_id}",
            )
        ]
        payload = _config_result_payload(config=config, run_result=run_result, artifacts=config_artifacts)
        config_results.append(payload)
        stage_files.extend(payload["files"])
        stage_artifacts.extend(config_artifacts)

    manifest = _workflow_payload(
        metadata,
        stage="profile",
        attempt=attempt,
        status="ok" if all(item["ok"] for item in config_results) else "error",
    )
    manifest.update(
        {
            "input_artifact_id": target_artifact["artifact_id"],
            "input_path": target_artifact["path"],
            "profiler": "ncu",
            "configs": [_config_payload(config) for config in configs],
            "config_results": [
                {
                    "config_id": item["config"]["config_id"],
                    "ok": item["ok"],
                    "returncode": item["returncode"],
                    "duration_seconds": item["duration_seconds"],
                    "artifacts": item["artifacts"],
                }
                for item in config_results
            ],
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
    overall_ok = all(item["ok"] for item in config_results)
    overall_returncode = next((item["returncode"] for item in config_results if item["returncode"] != 0), 0)
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
        ok=overall_ok,
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
        ok=run_result["ok"],
        output=run_result["output"],
    )
