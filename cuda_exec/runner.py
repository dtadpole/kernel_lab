"""Runtime helpers for cuda_exec.

This module owns runtime-layout semantics.

Mental model:
- workspace = inputs + scratch
- artifacts = kept results
- logs = process output
- state = workflow record

Only four top-level directories are created per revision root:
- workspace/
- artifacts/
- logs/
- state/

Public API code should document request/response contracts in models.py.
Detailed design rationale belongs in DESIGN.md.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CUDA_TOOLKIT_ROOT = Path("/usr/local/cuda")
CUDA_TOOLKIT_BIN = CUDA_TOOLKIT_ROOT / "bin"
CUDA_EXEC_ROOT_ENV = "CUDA_EXEC_ROOT"
MAX_CAPTURE_BYTES = 1024 * 1024
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_component(label: str, value: str) -> str:
    if not value:
        raise ValueError(f"{label} must not be empty")
    if value in {".", ".."} or "/" in value:
        raise ValueError(f"{label} contains an invalid path component")
    if not SAFE_COMPONENT_RE.fullmatch(value):
        raise ValueError((
                f"{label} must match {SAFE_COMPONENT_RE.pattern} "
                f"and only contain safe path characters"
            ),
        )
    return value


def _runtime_root() -> Path:
    """Return the runtime root for revision data.

    Default:
        Path.home() / ".cuda_exec"

    Tests and isolated runs may override this with the CUDA_EXEC_ROOT
    environment variable so integration coverage can run without leaving
    persistent artifacts under the real home directory.
    """

    override = os.environ.get(CUDA_EXEC_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cuda_exec"


def resolve_workspace_bundle(
    *,
    run_tag: str,
    version: str,
    direction_id: int,
    direction_slug: str,
    revision: int,
) -> dict:
    """Resolve and create the per-revision runtime bundle.

    The returned bundle is the concrete on-disk implementation of the four-dir
    runtime model. `workspace_path` is also the initial cwd for launched
    processes. The bundle root is normally under ~/.cuda_exec but may be
    redirected via CUDA_EXEC_ROOT for tests or isolated runs.
    """

    safe_run_tag = _validate_component("run_tag", run_tag)
    safe_version = _validate_component("version", version)
    safe_direction_slug = _validate_component("direction_slug", direction_slug)
    if direction_id < 0:
        raise ValueError("direction_id must be >= 0")
    if revision < 0:
        raise ValueError("revision must be >= 0")

    rev_root = (
        _runtime_root()
        / safe_run_tag
        / safe_version
        / f"{direction_id}_{safe_direction_slug}"
        / f"rev_{revision}"
    )
    bundle = {
        "root_path": str(rev_root),
        "workspace_path": str(rev_root / "workspace"),
        "artifacts_path": str(rev_root / "artifacts"),
        "logs_path": str(rev_root / "logs"),
        "state_path": str(rev_root / "state"),
    }
    for path in bundle.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return bundle


def _merge_env(extra_env: Dict[str, str]) -> Dict[str, str]:
    from cuda_exec.host_env import resolve_host_env

    env = os.environ.copy()
    env.update(resolve_host_env())  # host-resolved CUDA_HOME, LD_PRELOAD
    env.update(extra_env)           # caller overrides win
    return env


def _resolve_existing_directory(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"workspace path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"workspace path is not a directory: {path}")
    return path


def _rev_root_from_workspace(workspace_path: Path) -> Path:
    return workspace_path.parent


def resolve_rev_artifact_path(path_value: str, workspace_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _rev_root_from_workspace(workspace_path) / path
    return path.resolve()


def capture_rev_file(path_value: str, workspace_path: str, max_bytes: int | None = None) -> dict:
    """Read a revision-relative file for public response use.

    The caller provides a relative path within the revision root. The returned dict
    includes content plus minimal encoding/truncation metadata so the same
    helper can serve both text logs and binary artifacts.
    """

    workspace = _resolve_existing_directory(workspace_path)
    path = resolve_rev_artifact_path(path_value, workspace)
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
    limit = max_bytes if max_bytes is not None else MAX_CAPTURE_BYTES
    truncated = len(raw) > limit
    raw_for_response = raw[:limit]

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
    return [capture_rev_file(value, str(workspace_path)) for value in deduped]


def _relative_to_rev_root(path: Path, workspace_path: Path) -> str:
    return str(path.relative_to(_rev_root_from_workspace(workspace_path)))


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
    logger.info("CMD %s", " ".join(command))
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
        raise ValueError(str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{kind} timed out after {timeout_seconds}s") from exc

    extra_files = list(return_files or [])
    duration = time.perf_counter() - started
    end_ts = datetime.now(timezone.utc)
    start_ts_str = (end_ts.timestamp() - duration)
    start_dt = datetime.fromtimestamp(start_ts_str, tz=timezone.utc)
    ts_fmt = "%Y-%m-%d %H:%M:%S UTC"

    if log_file:
        log_path = resolve_rev_artifact_path(log_file, resolved_workspace)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = log_path.with_suffix(".stdout")
        stderr_path = log_path.with_suffix(".stderr")

        log_text = (
            f"started: {start_dt.strftime(ts_fmt)}\n"
            f"finished: {end_ts.strftime(ts_fmt)}\n"
            f"duration: {duration:.2f}s\n"
            f"command: {' '.join(command)}\n"
            f"returncode: {completed.returncode}\n"
            f"stdout_file: {stdout_path.name}\n"
            f"stderr_file: {stderr_path.name}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}\n"
        )
        log_path.write_text(log_text, encoding="utf-8")
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        extra_files.extend(
            [
                _relative_to_rev_root(log_path, resolved_workspace),
                _relative_to_rev_root(stdout_path, resolved_workspace),
                _relative_to_rev_root(stderr_path, resolved_workspace),
            ]
        )

    files = _collect_files(extra_files, resolved_workspace)
    return {
        "ok": completed.returncode == 0,
        "kind": kind,
        "command": command,
        "rev_root": str(_rev_root_from_workspace(resolved_workspace)),
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
        raise ValueError("command must not be empty")
    executable = Path(command[0]).expanduser().resolve()
    from cuda_exec.host_env import resolve_host_env
    cuda_home = resolve_host_env().get("CUDA_HOME", str(CUDA_TOOLKIT_ROOT))
    toolkit_bin = (Path(cuda_home) / "bin").resolve()
    if toolkit_bin not in executable.parents and executable != toolkit_bin:
        raise ValueError((
                "command[0] must point to a CUDA Toolkit binary under "
                f"{toolkit_bin}"
            ),
        )
    if not executable.exists():
        raise ValueError(f"binary does not exist: {executable}")
    if not os.access(executable, os.X_OK):
        raise ValueError(f"binary is not executable: {executable}")

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
