from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from fastapi import HTTPException

from cuda_exec.runner import capture_turn_file, resolve_workspace_bundle, run_generic_command

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
COMPILE_SCRIPT = SCRIPTS_DIR / "compile.sh"
PROFILE_SCRIPT = SCRIPTS_DIR / "profile.sh"
DEFAULT_COMPILE_ARTIFACT_ID = "compile:primary_binary"


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


def _load_compile_manifest(workspace: dict) -> dict:
    manifest_path = Path(workspace["state_path"]) / "compile.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "compile state is missing for this turn: "
                f"{manifest_path}. Run compile first or use a different turn."
            ),
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _artifact_path_from_manifest(workspace: dict, artifact_id: str | None) -> tuple[str, Path, dict]:
    manifest = _load_compile_manifest(workspace)
    requested_id = artifact_id or DEFAULT_COMPILE_ARTIFACT_ID
    for artifact in manifest.get("artifacts", []):
        if artifact.get("artifact_id") == requested_id:
            rel_path = artifact["path"]
            abs_path = (Path(workspace["root_path"]) / rel_path).resolve()
            if not abs_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"artifact path recorded in compile state does not exist: {abs_path}",
                )
            return requested_id, abs_path, artifact
    raise HTTPException(
        status_code=400,
        detail=f"artifact_id not found in compile state: {requested_id}",
    )


def _append_file_entry(result: dict, rel_path: str) -> None:
    existing_paths = {item["path"] for item in result.get("files", [])}
    file_entry = capture_turn_file(rel_path, result["workspace_path"])
    if file_entry["path"] not in existing_paths:
        result.setdefault("files", []).append(file_entry)


def run_compile_task(
    *,
    metadata,
    timeout_seconds: int,
    original_files: List[str],
    generated_files: List[str],
    return_files: List[str],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    workspace_path = Path(workspace["workspace_path"])
    outputs_path = Path(workspace["outputs_path"])
    state_path = Path(workspace["state_path"])

    copied_original = _copy_inputs(original_files, workspace_path / "original")
    copied_generated = _copy_inputs(generated_files, workspace_path / "generated")
    source = _pick_single_cuda_source(copied_generated, copied_original)

    binary_output = outputs_path / source.stem
    command = [
        "/usr/bin/env",
        "bash",
        str(COMPILE_SCRIPT),
        "--source",
        str(source),
        "--output",
        str(binary_output),
    ]
    result = run_generic_command(
        kind="compile",
        command=command,
        workspace_path=str(workspace_path),
        env={},
        timeout_seconds=timeout_seconds,
        return_files=_unique_paths([f"outputs/{source.stem}", *(return_files or [])]),
        log_file="logs/compile.log",
    )

    artifacts = [
        _build_artifact(
            artifact_id=DEFAULT_COMPILE_ARTIFACT_ID,
            kind="binary",
            path=f"outputs/{source.stem}",
            description="Default executable artifact produced by compile",
        ),
        _build_artifact(
            artifact_id="compile:log",
            kind="log",
            path="logs/compile.log",
            description="Compile log",
        ),
        _build_artifact(
            artifact_id="compile:state",
            kind="state",
            path="state/compile.json",
            description="Compile manifest for this turn",
        ),
    ]

    manifest = {
        "metadata": metadata.model_dump(),
        "status": "ok" if result["ok"] else "error",
        "primary_artifact_id": DEFAULT_COMPILE_ARTIFACT_ID,
        "selected_source": str(source.relative_to(workspace_path)),
        "artifacts": artifacts,
    }
    manifest_path = state_path / "compile.json"
    _write_manifest(manifest_path, manifest)
    _append_file_entry(result, "state/compile.json")
    result["artifacts"] = artifacts
    return result


def run_evaluate_task(
    *,
    metadata,
    timeout_seconds: int,
    target_artifact_id: str | None,
    return_files: List[str],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    _, target_path, target_artifact = _artifact_path_from_manifest(workspace, target_artifact_id)
    workspace_path = Path(workspace["workspace_path"])
    result = run_generic_command(
        kind="evaluate",
        command=[str(target_path)],
        workspace_path=str(workspace_path),
        env={},
        timeout_seconds=timeout_seconds,
        return_files=_unique_paths([*(return_files or [])]),
        log_file="logs/evaluate.log",
    )

    artifacts = [
        _build_artifact(
            artifact_id="evaluate:log",
            kind="log",
            path="logs/evaluate.log",
            description="Evaluate log",
        ),
        _build_artifact(
            artifact_id="evaluate:state",
            kind="state",
            path="state/evaluate.json",
            description="Evaluate manifest for this turn",
        ),
    ]
    manifest = {
        "metadata": metadata.model_dump(),
        "status": "ok" if result["ok"] else "error",
        "input_artifact_id": target_artifact["artifact_id"],
        "input_path": target_artifact["path"],
        "artifacts": artifacts,
    }
    _write_manifest(Path(workspace["state_path"]) / "evaluate.json", manifest)
    _append_file_entry(result, "state/evaluate.json")
    result["artifacts"] = artifacts
    return result


def run_profile_task(
    *,
    metadata,
    timeout_seconds: int,
    target_artifact_id: str | None,
    return_files: List[str],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    _, target_path, target_artifact = _artifact_path_from_manifest(workspace, target_artifact_id)
    workspace_path = Path(workspace["workspace_path"])
    report_prefix = Path(workspace["profiles_path"]) / f"{target_path.stem}-ncu"
    command = [
        "/usr/bin/env",
        "bash",
        str(PROFILE_SCRIPT),
        "--target",
        str(target_path),
        "--export-prefix",
        str(report_prefix),
    ]
    report_file = f"profiles/{target_path.stem}-ncu.ncu-rep"
    result = run_generic_command(
        kind="profile",
        command=command,
        workspace_path=str(workspace_path),
        env={},
        timeout_seconds=timeout_seconds,
        return_files=_unique_paths([report_file, *(return_files or [])]),
        log_file="logs/profile.log",
    )

    artifacts = [
        _build_artifact(
            artifact_id="profile:ncu_report",
            kind="profile_report",
            path=report_file,
            description="NCU report for the selected executable artifact",
        ),
        _build_artifact(
            artifact_id="profile:log",
            kind="log",
            path="logs/profile.log",
            description="Profile log",
        ),
        _build_artifact(
            artifact_id="profile:state",
            kind="state",
            path="state/profile.json",
            description="Profile manifest for this turn",
        ),
    ]
    manifest = {
        "metadata": metadata.model_dump(),
        "status": "ok" if result["ok"] else "error",
        "input_artifact_id": target_artifact["artifact_id"],
        "input_path": target_artifact["path"],
        "profiler": "ncu",
        "artifacts": artifacts,
    }
    _write_manifest(Path(workspace["state_path"]) / "profile.json", manifest)
    _append_file_entry(result, "state/profile.json")
    result["artifacts"] = artifacts
    return result
