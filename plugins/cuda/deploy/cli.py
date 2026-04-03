#!/usr/bin/env python3
"""CLI for managing cuda_exec service on remote GPU hosts.

Usage:
    python cli.py deploy <host>  [--rebuild] [--all]
    python cli.py start  <host>  [--all]
    python cli.py stop   <host>  [--all]
    python cli.py status <host>  [--all]
    python cli.py nuke   <host>  [--data] [--all]

Host names come from conf/hosts/default.yaml.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SERVICE_FILE = _SCRIPT_DIR / "cuda-exec.service"
_REQUIREMENTS_FILE = _SCRIPT_DIR / "requirements.txt"

_SERVICE_KEY = "cuda_exec"  # key in hosts.services map

# Host-level defaults (not in YAML — these are service-specific conventions)
_HOST_DEFAULTS = {
    "service_dir": ".cuda_exec_service",
    "data_dir": ".cuda_exec",
    "key_path": ".keys/cuda_exec.key",
    "service_name": "cuda-exec",
}


def _load_hosts_config() -> dict[str, Any]:
    """Load host configuration from conf/hosts/default.yaml."""
    cfg_path = _REPO_ROOT / "conf" / "hosts" / "default.yaml"
    if not cfg_path.exists():
        print(f"ERROR: hosts config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def _resolve_host(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a host name to a flat config dict for this service."""
    hosts = cfg.get("hosts", {})
    if name not in hosts:
        available = ", ".join(sorted(hosts.keys()))
        print(f"ERROR: unknown host '{name}'. Available: {available}", file=sys.stderr)
        sys.exit(1)

    host = hosts[name]
    svc = (host.get("services") or {}).get(_SERVICE_KEY)
    if svc is None:
        print(f"ERROR: host '{name}' has no {_SERVICE_KEY} service configured.", file=sys.stderr)
        sys.exit(1)

    hw = host.get("hardware") or {}
    return {
        **_HOST_DEFAULTS,
        "ssh_host": host["ssh_host"],
        "description": f"{hw.get('gpu', 'GPU')} ({hw.get('gpu_count', '?')}x)",
        **svc,
    }


def _deployable_host_names(cfg: dict[str, Any]) -> list[str]:
    """Return host names that have cuda_exec configured."""
    return sorted(
        name for name, h in cfg.get("hosts", {}).items()
        if (h.get("services") or {}).get(_SERVICE_KEY)
    )


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------


def _ssh(host: str, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command on a remote host via SSH."""
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, cmd],
        capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        print(f"  SSH command failed (exit {result.returncode}):", file=sys.stderr)
        if result.stderr.strip():
            print(f"  {result.stderr.strip()}", file=sys.stderr)
    return result


def _rsync(src: str, dst: str, excludes: list[str] | None = None, delete: bool = False) -> bool:
    """Rsync files to a remote host."""
    cmd = ["rsync", "-az"]
    if delete:
        cmd.append("--delete")
    for exc in (excludes or []):
        cmd.extend(["--exclude", exc])
    cmd.extend([src, dst])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  rsync failed: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_deploy(host_cfg: dict[str, Any], rebuild: bool = False) -> bool:
    """Deploy cuda_exec to a remote host. Idempotent.

    Syncs code, installs deps, configures systemd. Does NOT start the service.
    """
    ssh_host = host_cfg["ssh_host"]
    svc_dir = host_cfg["service_dir"]
    key_dir = str(Path(host_cfg["key_path"]).parent)
    key_path = host_cfg["key_path"]
    port = host_cfg["port"]
    svc_name = host_cfg["service_name"]

    print(f"=== Deploying cuda_exec to {ssh_host} ===")

    # 1. Install uv
    print("[1/5] Checking uv...")
    _ssh(ssh_host, """
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1
            echo '  uv installed.'
        else
            echo '  uv OK.'
        fi
    """)

    # 2. Sync source code
    print("[2/5] Syncing source code...")
    _ssh(ssh_host, f"mkdir -p ~/{svc_dir}", check=False)
    ok = _rsync(
        f"{_REPO_ROOT}/cuda_exec/",
        f"{ssh_host}:~/{svc_dir}/cuda_exec/",
        excludes=["__pycache__", ".venv", "*.pyc", ".pytest_cache", "tests/"],
        delete=True,
    )
    ok = ok and _rsync(
        f"{_REPO_ROOT}/conf/",
        f"{ssh_host}:~/{svc_dir}/conf/",
        excludes=["__pycache__"],
        delete=True,
    )
    if not ok:
        return False
    print("  Source synced.")

    # 3. Sync requirements file + install deps
    print("[3/5] Installing dependencies...")
    ok = ok and _rsync(
        str(_REQUIREMENTS_FILE),
        f"{ssh_host}:~/{svc_dir}/requirements.txt",
    )
    if not ok:
        return False
    rebuild_flag = "rm -rf .venv &&" if rebuild else ""
    r = _ssh(ssh_host, f"""
        cd ~/{svc_dir}
        export PATH=$HOME/.local/bin:$PATH
        {rebuild_flag}
        if [ ! -d .venv ]; then
            uv venv .venv --python 3.12 2>&1
        fi
        uv pip install --python .venv/bin/python -r requirements.txt 2>&1
    """)
    if r.returncode != 0:
        print(f"  Dependency install failed.", file=sys.stderr)
        return False
    print("  Dependencies OK.")

    # 4. Bearer token
    print("[4/5] Checking bearer token...")
    _ssh(ssh_host, f"""
        mkdir -p ~/{key_dir}
        chmod 700 ~/{key_dir}
        if [ ! -f ~/{key_path} ]; then
            python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/{key_path}
            chmod 600 ~/{key_path}
            echo '  Generated new token.'
        else
            echo '  Token exists.'
        fi
    """)

    # 5. Install systemd unit
    print("[5/5] Installing systemd service...")
    service_content = _SERVICE_FILE.read_text().replace("--port 8000", f"--port {port}")
    cuda_devices = host_cfg.get("cuda_visible_devices")
    if cuda_devices is not None:
        service_content = service_content.replace("__CUDA_VISIBLE_DEVICES__", cuda_devices)
    else:
        service_content = "\n".join(
            line for line in service_content.split("\n")
            if "__CUDA_VISIBLE_DEVICES__" not in line
        )
    _ssh(ssh_host, f"""
        mkdir -p $HOME/.config/systemd/user
        cat > $HOME/.config/systemd/user/{svc_name}.service << 'UNIT_EOF'
{service_content}
UNIT_EOF
        systemctl --user daemon-reload
        systemctl --user enable {svc_name}.service 2>&1
    """)
    print("  Service installed.")

    print(f"\nDeploy complete. Run: python cli.py start {ssh_host}")
    return True


def cmd_start(host_cfg: dict[str, Any]) -> bool:
    """Start the cuda_exec service."""
    ssh_host = host_cfg["ssh_host"]
    svc_name = host_cfg["service_name"]
    port = host_cfg["port"]

    print(f"Starting {svc_name} on {ssh_host}...")
    _ssh(ssh_host, f"systemctl --user restart {svc_name}.service")

    # Wait and verify
    import time
    time.sleep(3)
    r = _ssh(ssh_host, f"systemctl --user is-active {svc_name}.service", check=False)
    if r.stdout.strip() == "active":
        # Health check
        h = _ssh(ssh_host, f"curl -sf http://127.0.0.1:{port}/healthz", check=False)
        if '"ok":true' in h.stdout:
            print(f"  Running. Health: OK")
            return True
        else:
            print(f"  Running but health check failed: {h.stdout.strip()}")
            return False
    else:
        print(f"  Failed to start. Check: python cli.py status {ssh_host}")
        return False


def cmd_stop(host_cfg: dict[str, Any]) -> bool:
    """Stop the cuda_exec service."""
    ssh_host = host_cfg["ssh_host"]
    svc_name = host_cfg["service_name"]

    print(f"Stopping {svc_name} on {ssh_host}...")
    _ssh(ssh_host, f"systemctl --user stop {svc_name}.service")

    r = _ssh(ssh_host, f"systemctl --user is-active {svc_name}.service", check=False)
    state = r.stdout.strip()
    if state in ("inactive", "dead"):
        print("  Stopped.")
        return True
    else:
        print(f"  State: {state}")
        return False


def cmd_status(host_cfg: dict[str, Any]) -> bool:
    """Show full status of the remote service."""
    ssh_host = host_cfg["ssh_host"]
    svc_name = host_cfg["service_name"]
    port = host_cfg["port"]
    desc = host_cfg.get("description", "")

    print(f"=== {ssh_host} ({desc}) ===")

    # Service state
    r = _ssh(ssh_host, f"systemctl --user is-active {svc_name}.service", check=False)
    state = r.stdout.strip()
    print(f"  Service:  {state}")

    # Health check
    if state == "active":
        h = _ssh(ssh_host, f"curl -sf http://127.0.0.1:{port}/healthz", check=False)
        health = h.stdout.strip() if h.returncode == 0 else "unreachable"
        print(f"  Health:   {health}")
    else:
        print(f"  Health:   (service not running)")

    # GPU pin
    cuda_devices = host_cfg.get("cuda_visible_devices")
    if cuda_devices is not None:
        print(f"  GPU pin:  CUDA_VISIBLE_DEVICES={cuda_devices}")

    # GPU
    r = _ssh(ssh_host,
        "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader 2>/dev/null || echo 'N/A'",
        check=False,
    )
    print(f"  GPU:      {r.stdout.strip()}")

    # Disk
    r = _ssh(ssh_host, "df -h ~ | tail -1 | awk '{print $3\"/\"$2\" (\"$5\" used)\"}'", check=False)
    print(f"  Disk:     {r.stdout.strip()}")

    # Data dir size
    data_dir = host_cfg["data_dir"]
    r = _ssh(ssh_host, f"du -sh ~/{data_dir} 2>/dev/null | cut -f1 || echo '0'", check=False)
    print(f"  Data:     {r.stdout.strip()} (~/{data_dir})")

    # Recent logs (last 5 lines)
    r = _ssh(ssh_host,
        f"journalctl --user -u {svc_name} --no-pager -n 5 -o short-iso 2>/dev/null",
        check=False,
    )
    if r.stdout.strip():
        print(f"  Logs (last 5):")
        for line in r.stdout.strip().split("\n"):
            print(f"    {line}")

    return state == "active"


def cmd_health(host_cfg: dict[str, Any]) -> bool:
    """Quick health check: is the API responding?"""
    ssh_host = host_cfg["ssh_host"]
    port = host_cfg["port"]

    h = _ssh(ssh_host, f"curl -sf http://127.0.0.1:{port}/healthz", check=False)
    if h.returncode == 0 and '"ok":true' in h.stdout:
        print(f"{ssh_host}: healthy")
        return True
    else:
        print(f"{ssh_host}: unhealthy ({h.stdout.strip() or 'unreachable'})")
        return False


def cmd_nuke(host_cfg: dict[str, Any], include_data: bool = False) -> bool:
    """Nuclear option: stop service, remove everything."""
    ssh_host = host_cfg["ssh_host"]
    svc_name = host_cfg["service_name"]
    svc_dir = host_cfg["service_dir"]
    data_dir = host_cfg["data_dir"]

    print(f"=== NUKE {ssh_host} ===")

    # Stop
    print("[1/3] Stopping service...")
    _ssh(ssh_host, f"systemctl --user stop {svc_name}.service 2>/dev/null", check=False)
    _ssh(ssh_host, f"systemctl --user disable {svc_name}.service 2>/dev/null", check=False)

    # Remove systemd unit
    print("[2/3] Removing systemd unit...")
    _ssh(ssh_host, f"""
        rm -f $HOME/.config/systemd/user/{svc_name}.service
        systemctl --user daemon-reload 2>/dev/null
    """)

    # Remove service directory
    print("[3/3] Removing service files...")
    _ssh(ssh_host, f"rm -rf ~/{svc_dir}")
    print(f"  Removed ~/{svc_dir}")

    if include_data:
        _ssh(ssh_host, f"rm -rf ~/{data_dir}")
        print(f"  Removed ~/{data_dir}")

    print(f"\nNuke complete. {ssh_host} is clean.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage cuda_exec service on remote GPU hosts",
        prog="python plugins/cuda/deploy/cli.py",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # deploy
    p = sub.add_parser("deploy", help="Sync code + install deps + configure systemd")
    p.add_argument("host", nargs="?", help="Host name from conf/hosts/default.yaml")
    p.add_argument("--rebuild", action="store_true", help="Force recreate venv")
    p.add_argument("--all", action="store_true", help="Deploy to all hosts")

    # start
    p = sub.add_parser("start", help="Start the service")
    p.add_argument("host", nargs="?", help="Host name")
    p.add_argument("--all", action="store_true", help="Start on all hosts")

    # stop
    p = sub.add_parser("stop", help="Stop the service")
    p.add_argument("host", nargs="?", help="Host name")
    p.add_argument("--all", action="store_true", help="Stop on all hosts")

    # status
    p = sub.add_parser("status", help="Show full status report")
    p.add_argument("host", nargs="?", help="Host name")
    p.add_argument("--all", action="store_true", help="Status of all hosts")

    # health
    p = sub.add_parser("health", help="Quick health check: is the API responding?")
    p.add_argument("host", nargs="?", help="Host name")
    p.add_argument("--all", action="store_true", help="Health check all hosts")

    # nuke
    p = sub.add_parser("nuke", help="Nuclear option: stop + remove everything")
    p.add_argument("host", nargs="?", help="Host name")
    p.add_argument("--data", action="store_true", help="Also remove runtime data")
    p.add_argument("--all", action="store_true", help="Nuke all hosts")

    args = parser.parse_args()
    cfg = _load_hosts_config()

    # Resolve target hosts
    use_all = getattr(args, "all", False)
    if use_all:
        targets = _deployable_host_names(cfg)
    elif args.host:
        targets = [args.host]
    else:
        parser.error("provide a host name or --all")

    # Execute
    all_ok = True
    for name in targets:
        host_cfg = _resolve_host(cfg, name)
        if args.command == "deploy":
            ok = cmd_deploy(host_cfg, rebuild=args.rebuild)
        elif args.command == "start":
            ok = cmd_start(host_cfg)
        elif args.command == "stop":
            ok = cmd_stop(host_cfg)
        elif args.command == "status":
            ok = cmd_status(host_cfg)
        elif args.command == "health":
            ok = cmd_health(host_cfg)
        elif args.command == "nuke":
            ok = cmd_nuke(host_cfg, include_data=args.data)
        else:
            ok = False

        if not ok:
            all_ok = False
        if len(targets) > 1:
            print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
