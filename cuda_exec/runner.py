from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

CUDA_TOOLKIT_ROOT = Path("/usr/local/cuda")
CUDA_TOOLKIT_BIN = CUDA_TOOLKIT_ROOT / "bin"


def _merge_env(extra_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(extra_env)
    return env


def _resolve_workdir(workdir: Optional[str]) -> str:
    if workdir is None:
        return str(Path.cwd())
    path = Path(workdir).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"workdir does not exist: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"workdir is not a directory: {path}")
    return str(path)


def _run_command(
    *,
    kind: str,
    command: List[str],
    workdir: str,
    env: Dict[str, str],
    timeout_seconds: int,
) -> dict:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=_merge_env(env),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=408,
            detail=f"{kind} timed out after {timeout_seconds}s",
        ) from exc

    duration = time.perf_counter() - started
    return {
        "ok": completed.returncode == 0,
        "kind": kind,
        "command": command,
        "workdir": workdir,
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_generic_command(
    *,
    kind: str,
    command: List[str],
    workdir: str,
    env: Dict[str, str],
    timeout_seconds: int,
) -> dict:
    return _run_command(
        kind=kind,
        command=command,
        workdir=_resolve_workdir(workdir),
        env=env,
        timeout_seconds=timeout_seconds,
    )


def run_profile(
    *,
    profiler: str,
    target_command: List[str],
    profiler_args: List[str],
    workdir: str,
    env: Dict[str, str],
    timeout_seconds: int,
) -> dict:
    profiler_bin = CUDA_TOOLKIT_BIN / profiler
    if profiler == "nsys":
        profiler_bin = Path("/usr/local/bin/nsys")
    if not profiler_bin.exists():
        raise HTTPException(status_code=400, detail=f"Profiler not found: {profiler_bin}")

    command = [str(profiler_bin), *profiler_args, *target_command]
    return _run_command(
        kind="profile",
        command=command,
        workdir=_resolve_workdir(workdir),
        env=env,
        timeout_seconds=timeout_seconds,
    )


def run_cuda_binary(
    *,
    binary_path: str,
    args: List[str],
    workdir: Optional[str],
    env: Dict[str, str],
    timeout_seconds: int,
) -> dict:
    binary = Path(binary_path).expanduser().resolve()
    toolkit_bin = CUDA_TOOLKIT_BIN.resolve()
    if toolkit_bin not in binary.parents and binary != toolkit_bin:
        raise HTTPException(
            status_code=400,
            detail=(
                "binary_path must point to a CUDA Toolkit binary under "
                f"{toolkit_bin}"
            ),
        )
    if not binary.exists():
        raise HTTPException(status_code=400, detail=f"binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise HTTPException(status_code=400, detail=f"binary is not executable: {binary}")

    command = [str(binary), *args]
    return _run_command(
        kind="execute",
        command=command,
        workdir=_resolve_workdir(workdir),
        env=env,
        timeout_seconds=timeout_seconds,
    )
