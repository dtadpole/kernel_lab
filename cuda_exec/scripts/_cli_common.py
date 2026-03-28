from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def parse_env_assignments(values: Iterable[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid env assignment: {item!r}. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid env assignment with empty key: {item!r}")
        env[key] = value
    return env


def shell_join(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def print_command_preview(command: List[str], workdir: str) -> None:
    print("toolkit_command:")
    print(f"  cwd: {workdir}")
    print(f"  cmd: {shell_join(command)}")


def emit_result(result: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=False))
        return

    print("result:")
    print(f"  ok: {result['ok']}")
    print(f"  kind: {result['kind']}")
    print(f"  returncode: {result['returncode']}")
    print(f"  duration_seconds: {result['duration_seconds']:.6f}")
    print(f"  workdir: {result['workdir']}")
    print("  command:")
    print(f"    {shell_join(result['command'])}")
    print("  stdout:")
    stdout = result.get("stdout", "")
    print(stdout if stdout else "<empty>")
    print("  stderr:")
    stderr = result.get("stderr", "")
    print(stderr if stderr else "<empty>")


def default_workdir(value: str | None) -> str:
    return value or os.getcwd()
