from __future__ import annotations

import base64
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

CUDA_TOOLKIT_ROOT = Path("/usr/local/cuda")
CUDA_TOOLKIT_BIN = CUDA_TOOLKIT_ROOT / "bin"
CODE_EXEC_ROOT = Path.home() / ".code_exec"
MAX_CAPTURE_BYTES = 1024 * 1024
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_component(label: str, value: str) -> str:
    if not value:
        raise HTTPException(status_code=400, detail=f"{label} must not be empty")
    if value in {".", ".."} or "/" in value:
        raise HTTPException(status_code=400, detail=f"{label} contains an invalid path component")
    if not SAFE_COMPONENT_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} must match {SAFE_COMPONENT_RE.pattern} "
                f"and only contain safe path characters"
            ),
        )
    return value


def resolve_workspace_bundle(
    *,
    run_tag: str,
    version: str,
    direction_id: int,
    direction_slug: str,
    turn: int,
) -> dict:
    safe_run_tag = _validate_component("run_tag", run_tag)
    safe_version = _validate_component("version", version)
    safe_direction_slug = _validate_component("direction_slug", direction_slug)
    if direction_id < 0:
        raise HTTPException(status_code=400, detail="direction_id must be >= 0")
    if turn < 0:
        raise HTTPException(status_code=400, detail="turn must be >= 0")

    turn_root = (
        CODE_EXEC_ROOT
        / safe_run_tag
        / safe_version
        / f"{direction_id}_{safe_direction_slug}"
        / f"turn_{turn}"
    )
    bundle = {
        "root_path": str(turn_root),
        "workspace_path": str(turn_root / "workspace"),
        "outputs_path": str(turn_root / "outputs"),
        "logs_path": str(turn_root / "logs"),
        "profiles_path": str(turn_root / "profiles"),
        "state_path": str(turn_root / "state"),
        "tmp_path": str(turn_root / "tmp"),
    }
    for path in bundle.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return bundle


def _merge_env(extra_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(extra_env)
    return env


def _resolve_existing_directory(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"workspace path does not exist: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"workspace path is not a directory: {path}")
    return path


def _turn_root_from_workspace(workspace_path: Path) -> Path:
    return workspace_path.parent


def resolve_turn_artifact_path(path_value: str, workspace_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _turn_root_from_workspace(workspace_path) / path
    return path.resolve()


def capture_turn_file(path_value: str, workspace_path: str) -> dict:
    workspace = _resolve_existing_directory(workspace_path)
    path = resolve_turn_artifact_path(path_value, workspace)
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


def _collect_files(paths: List[str], workspace_path: Path) -> List[dict]:
    deduped: List[str] = []
    seen: set[str] = set()
    for value in paths:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return [capture_turn_file(value, str(workspace_path)) for value in deduped]


def _run_command(
    *,
    kind: str,
    command: List[str],
    workspace_path: str,
    env: Dict[str, str],
    timeout_seconds: int,
    return_files: Optional[List[str]] = None,
    log_file: Optional[str] = None,
) -> dict:
    resolved_workspace = _resolve_existing_directory(workspace_path)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(resolved_workspace),
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

    extra_files = list(return_files or [])
    if log_file:
        log_path = resolve_turn_artifact_path(log_file, resolved_workspace)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_text = (
            f"command: {' '.join(command)}\n"
            f"returncode: {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}\n"
        )
        log_path.write_text(log_text, encoding="utf-8")
        relative_log = str(log_path.relative_to(_turn_root_from_workspace(resolved_workspace)))
        extra_files.append(relative_log)

    duration = time.perf_counter() - started
    files = _collect_files(extra_files, resolved_workspace)
    return {
        "ok": completed.returncode == 0,
        "kind": kind,
        "command": command,
        "workspace_path": str(resolved_workspace),
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
    workspace_path: str,
    env: Dict[str, str],
    timeout_seconds: int,
    return_files: Optional[List[str]] = None,
    log_file: Optional[str] = None,
) -> dict:
    return _run_command(
        kind=kind,
        command=command,
        workspace_path=workspace_path,
        env=env,
        timeout_seconds=timeout_seconds,
        return_files=return_files,
        log_file=log_file,
    )


def run_cuda_command(
    *,
    kind: str,
    command: List[str],
    workspace_path: str,
    env: Dict[str, str],
    timeout_seconds: int,
    return_files: Optional[List[str]] = None,
    log_file: Optional[str] = None,
) -> dict:
    if not command:
        raise HTTPException(status_code=400, detail="command must not be empty")
    executable = Path(command[0]).expanduser().resolve()
    toolkit_bin = CUDA_TOOLKIT_BIN.resolve()
    if toolkit_bin not in executable.parents and executable != toolkit_bin:
        raise HTTPException(
            status_code=400,
            detail=(
                "command[0] must point to a CUDA Toolkit binary under "
                f"{toolkit_bin}"
            ),
        )
    if not executable.exists():
        raise HTTPException(status_code=400, detail=f"binary does not exist: {executable}")
    if not os.access(executable, os.X_OK):
        raise HTTPException(status_code=400, detail=f"binary is not executable: {executable}")

    normalized_command = [str(executable), *command[1:]]
    return _run_command(
        kind=kind,
        command=normalized_command,
        workspace_path=workspace_path,
        env=env,
        timeout_seconds=timeout_seconds,
        return_files=return_files,
        log_file=log_file,
    )
