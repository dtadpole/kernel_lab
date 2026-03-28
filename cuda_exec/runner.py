from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

CUDA_TOOLKIT_ROOT = Path("/usr/local/cuda")
CUDA_TOOLKIT_BIN = CUDA_TOOLKIT_ROOT / "bin"
MAX_CAPTURE_BYTES = 1024 * 1024


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


def _resolve_return_file(path_value: str, workdir: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path(workdir) / path
    return path.resolve()


def _capture_file(path_value: str, workdir: str) -> dict:
    path = _resolve_return_file(path_value, workdir)
    if not path.exists():
        return {
            "path": str(path),
            "name": path.name,
            "exists": False,
            "size_bytes": None,
            "encoding": None,
            "truncated": False,
            "content": None,
            "error": None,
        }

    if not path.is_file():
        return {
            "path": str(path),
            "name": path.name,
            "exists": True,
            "size_bytes": None,
            "encoding": None,
            "truncated": False,
            "content": None,
            "error": "path exists but is not a regular file",
        }

    raw = path.read_bytes()
    truncated = len(raw) > MAX_CAPTURE_BYTES
    raw_for_response = raw[:MAX_CAPTURE_BYTES]

    try:
        content = raw_for_response.decode("utf-8")
        encoding = "utf8"
    except UnicodeDecodeError:
        content = base64.b64encode(raw_for_response).decode("ascii")
        encoding = "base64"

    return {
        "path": str(path),
        "name": path.name,
        "exists": True,
        "size_bytes": len(raw),
        "encoding": encoding,
        "truncated": truncated,
        "content": content,
        "error": None,
    }


def _collect_files(paths: List[str], workdir: str) -> List[dict]:
    deduped: List[str] = []
    seen: set[str] = set()
    for value in paths:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return [_capture_file(value, workdir) for value in deduped]


def _run_command(
    *,
    kind: str,
    command: List[str],
    workdir: str,
    env: Dict[str, str],
    timeout_seconds: int,
    return_files: Optional[List[str]] = None,
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
    files = _collect_files(return_files or [], workdir)
    return {
        "ok": completed.returncode == 0,
        "kind": kind,
        "command": command,
        "workdir": workdir,
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "output": {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        "files": files,
    }


def run_generic_command(
    *,
    kind: str,
    command: List[str],
    workdir: str,
    env: Dict[str, str],
    timeout_seconds: int,
    return_files: Optional[List[str]] = None,
) -> dict:
    return _run_command(
        kind=kind,
        command=command,
        workdir=_resolve_workdir(workdir),
        env=env,
        timeout_seconds=timeout_seconds,
        return_files=return_files,
    )


def run_profile(
    *,
    profiler: str,
    target_command: List[str],
    profiler_args: List[str],
    workdir: str,
    env: Dict[str, str],
    timeout_seconds: int,
    return_files: Optional[List[str]] = None,
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
        return_files=return_files,
    )


def run_cuda_binary(
    *,
    binary_path: str,
    args: List[str],
    workdir: Optional[str],
    env: Dict[str, str],
    timeout_seconds: int,
    return_files: Optional[List[str]] = None,
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

    resolved_workdir = _resolve_workdir(workdir)
    command = [str(binary), *args]
    return _run_command(
        kind="execute",
        command=command,
        workdir=resolved_workdir,
        env=env,
        timeout_seconds=timeout_seconds,
        return_files=return_files,
    )
