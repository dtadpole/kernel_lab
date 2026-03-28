from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import HTTPException

from cuda_exec.runner import (
    CUDA_TOOLKIT_BIN,
    resolve_workspace_bundle,
    run_cuda_command,
    run_generic_command,
)

NVCC = str(CUDA_TOOLKIT_BIN / "nvcc")
NCU = str(CUDA_TOOLKIT_BIN / "ncu")


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


def _default_executable(outputs_path: Path) -> Path:
    files = [path for path in outputs_path.iterdir() if path.is_file() and path.suffix != ".log"]
    executable_files = [path for path in files if path.suffix == "" and path.stat().st_mode & 0o111]
    if len(executable_files) == 1:
        return executable_files[0]
    if len(executable_files) > 1:
        raise HTTPException(
            status_code=400,
            detail="multiple executable artifacts found in outputs; provide target_files explicitly",
        )
    raise HTTPException(
        status_code=400,
        detail="no executable artifact found in outputs; compile may need to run first or target_files must be provided",
    )


def _resolve_target_files(target_files: List[str], workspace: dict) -> List[Path]:
    if target_files:
        resolved: List[Path] = []
        for value in target_files:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path(workspace["root_path"]) / path
            path = path.resolve()
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"target file does not exist: {path}")
            resolved.append(path)
        return resolved
    return [_default_executable(Path(workspace["outputs_path"]))]


def _unique_paths(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def run_compile_task(
    *,
    metadata,
    timeout_seconds: int,
    original_files: List[str],
    generated_files: List[str],
    artifacts: List[str],
    return_files: List[str],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    workspace_path = Path(workspace["workspace_path"])
    outputs_path = Path(workspace["outputs_path"])

    copied_original = _copy_inputs(original_files, workspace_path / "original")
    copied_generated = _copy_inputs(generated_files, workspace_path / "generated")
    source = _pick_single_cuda_source(copied_generated, copied_original)

    binary_output = outputs_path / source.stem
    command = [
        NVCC,
        "-arch=native",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        str(source),
        "-o",
        str(binary_output),
    ]
    result = run_cuda_command(
        kind="compile",
        command=command,
        workspace_path=str(workspace_path),
        env={},
        timeout_seconds=timeout_seconds,
        return_files=_unique_paths(
            [
                f"outputs/{source.stem}",
                *(artifacts or []),
                *(return_files or []),
            ]
        ),
        log_file="logs/compile_nvcc.log",
    )
    return result


def run_evaluate_task(
    *,
    metadata,
    timeout_seconds: int,
    target_files: List[str],
    return_files: List[str],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    targets = _resolve_target_files(target_files, workspace)
    if len(targets) != 1:
        raise HTTPException(status_code=400, detail="evaluate expects exactly one target file in this hardened flow")
    target = targets[0]
    workspace_path = Path(workspace["workspace_path"])
    return run_generic_command(
        kind="evaluate",
        command=[str(target)],
        workspace_path=str(workspace_path),
        env={},
        timeout_seconds=timeout_seconds,
        return_files=return_files,
        log_file="logs/evaluate.log",
    )


def run_profile_task(
    *,
    metadata,
    timeout_seconds: int,
    target_files: List[str],
    return_files: List[str],
) -> dict:
    workspace = resolve_workspace_bundle(**metadata.model_dump())
    targets = _resolve_target_files(target_files, workspace)
    if len(targets) != 1:
        raise HTTPException(status_code=400, detail="profile expects exactly one target file in this hardened flow")
    target = targets[0]
    workspace_path = Path(workspace["workspace_path"])
    report_prefix = Path(workspace["profiles_path"]) / f"{target.stem}-ncu"
    command = [
        NCU,
        "--set",
        "default",
        "--target-processes",
        "all",
        "--force-overwrite",
        "--export",
        str(report_prefix),
        str(target),
    ]
    report_file = f"profiles/{target.stem}-ncu.ncu-rep"
    return run_cuda_command(
        kind="profile",
        command=command,
        workspace_path=str(workspace_path),
        env={},
        timeout_seconds=timeout_seconds,
        return_files=_unique_paths([report_file, *(return_files or [])]),
        log_file="logs/profile_ncu.log",
    )
