#!/usr/bin/env python3
"""CLI for managing the KB embedding service on remote GPU hosts.

Usage:
    python cli.py deploy <host>  [--rebuild] [--all]
    python cli.py start  <host>  [--all]
    python cli.py stop   <host>  [--all]
    python cli.py status <host>  [--all]
    python cli.py health <host>  [--all]
    python cli.py nuke   <host>  [--data] [--all]

Host names come from conf/kb/default.yaml.
"""

from __future__ import annotations

import argparse
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
_SERVICE_FILE = _SCRIPT_DIR / "kb-embed.service"
_EMBED_SERVER_DIR = _SCRIPT_DIR.parent / "embed_server"


def _load_config() -> dict[str, Any]:
    """Load KB service configuration from conf/kb/default.yaml."""
    cfg_path = _REPO_ROOT / "conf" / "kb" / "default.yaml"
    if not cfg_path.exists():
        print(f"ERROR: KB config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def _resolve_host(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a host name to its config, merged with defaults."""
    hosts = cfg.get("hosts", {})
    if name not in hosts:
        available = ", ".join(sorted(hosts.keys()))
        print(f"ERROR: unknown host '{name}'. Available: {available}", file=sys.stderr)
        sys.exit(1)
    defaults = cfg.get("service_defaults", {})
    return {**defaults, **hosts[name]}


def _all_host_names(cfg: dict[str, Any]) -> list[str]:
    return sorted(cfg.get("hosts", {}).keys())


# ---------------------------------------------------------------------------
# SSH / rsync helpers
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
    """Deploy embedding service to a remote host. Idempotent.

    Syncs code, installs deps, configures systemd. Does NOT start the service.
    """
    ssh_host = host_cfg["ssh_host"]
    svc_dir = host_cfg["service_dir"]
    svc_name = host_cfg["service_name"]
    model_id = host_cfg["model_id"]
    port = host_cfg["port"]

    print(f"=== Deploying {svc_name} to {ssh_host} ===")

    # 1. Install uv
    print("[1/4] Checking uv...")
    _ssh(ssh_host, """
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1
            echo '  uv installed.'
        else
            echo '  uv OK.'
        fi
    """)

    # 2. Sync source code
    print("[2/4] Syncing embed_server code...")
    _ssh(ssh_host, f"mkdir -p ~/{svc_dir}", check=False)
    ok = _rsync(
        f"{_EMBED_SERVER_DIR}/",
        f"{ssh_host}:~/{svc_dir}/embed_server/",
        excludes=["__pycache__", "*.pyc", ".pytest_cache"],
        delete=True,
    )
    if not ok:
        return False
    print("  Source synced.")

    # 3. Venv + deps
    print("[3/4] Installing dependencies...")
    rebuild_flag = "rm -rf .venv &&" if rebuild else ""
    r = _ssh(ssh_host, f"""
        cd ~/{svc_dir}
        export PATH=$HOME/.local/bin:$PATH
        {rebuild_flag}
        if [ ! -d .venv ]; then
            uv venv .venv --python 3.12 2>&1
        fi
        uv pip install --python .venv/bin/python \
            torch \
            transformers \
            'fastapi>=0.116,<1.0' \
            'uvicorn[standard]>=0.35,<1.0' \
            accelerate 2>&1
    """)
    if r.returncode != 0:
        print(f"  Dependency install failed.", file=sys.stderr)
        return False
    print("  Dependencies OK.")

    # 4. Install systemd unit
    print(f"[4/4] Installing systemd service...")
    service_content = _SERVICE_FILE.read_text()
    service_content = service_content.replace("--port 46982", f"--port {port}")
    service_content = service_content.replace(
        "EMBED_MODEL_ID=Qwen/Qwen3-Embedding-4B",
        f"EMBED_MODEL_ID={model_id}",
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

    print(f"\nDeploy complete. Run: python3 cli.py start {ssh_host}")
    return True


def cmd_start(host_cfg: dict[str, Any]) -> bool:
    """Start the embedding service."""
    ssh_host = host_cfg["ssh_host"]
    svc_name = host_cfg["service_name"]
    port = host_cfg["port"]

    print(f"Starting {svc_name} on {ssh_host}...")
    _ssh(ssh_host, f"systemctl --user restart {svc_name}.service")

    # Model loading takes time — poll for readiness.
    # First start also downloads weights (~8GB).
    import time
    max_wait = 300
    interval = 10
    print("  Waiting for model to load (up to 5 min on first start)...")
    for elapsed in range(interval, max_wait + 1, interval):
        time.sleep(interval)
        h = _ssh(ssh_host, f"curl -sf http://127.0.0.1:{port}/health", check=False)
        if h.returncode == 0:
            print(f"  Running. Health: OK ({elapsed}s)")
            return True

        # Check if service is still alive
        r = _ssh(ssh_host, f"systemctl --user is-active {svc_name}.service", check=False)
        state = r.stdout.strip()
        if state not in ("active", "activating"):
            print(f"  Service failed to start (state: {state}).")
            print(f"  Check logs: python3 cli.py status {ssh_host}")
            return False

        print(f"  Still loading... ({elapsed}s)")

    print(f"  Timeout waiting for health check after {max_wait}s.")
    print(f"  Service may still be loading. Check: python3 cli.py status {ssh_host}")
    return False


def cmd_stop(host_cfg: dict[str, Any]) -> bool:
    """Stop the embedding service."""
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
    """Show full status of the remote embedding service."""
    ssh_host = host_cfg["ssh_host"]
    svc_name = host_cfg["service_name"]
    port = host_cfg["port"]
    svc_dir = host_cfg["service_dir"]
    desc = host_cfg.get("description", "")
    model_id = host_cfg["model_id"]

    print(f"=== {ssh_host} ({desc}) ===")

    # Service state
    r = _ssh(ssh_host, f"systemctl --user is-active {svc_name}.service", check=False)
    state = r.stdout.strip()
    print(f"  Service:  {state}")

    # Health check
    if state == "active":
        h = _ssh(ssh_host, f"curl -sf http://127.0.0.1:{port}/health", check=False)
        health = "ready" if h.returncode == 0 else "loading or unreachable"
        print(f"  Health:   {health}")
    else:
        print(f"  Health:   (service not running)")

    # Model
    print(f"  Model:    {model_id}")
    print(f"  Port:     {port}")

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

    # Service dir size
    r = _ssh(ssh_host, f"du -sh ~/{svc_dir} 2>/dev/null | cut -f1 || echo '0'", check=False)
    print(f"  Service:  {r.stdout.strip()} (~/{svc_dir})")

    # Model cache size
    cache_dir = host_cfg.get("model_cache_dir", ".cache/huggingface")
    r = _ssh(ssh_host, f"du -sh ~/{cache_dir} 2>/dev/null | cut -f1 || echo '0'", check=False)
    print(f"  Cache:    {r.stdout.strip()} (~/{cache_dir})")

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
    """Quick health check: is the embedding API responding?"""
    ssh_host = host_cfg["ssh_host"]
    port = host_cfg["port"]

    h = _ssh(ssh_host, f"curl -sf http://127.0.0.1:{port}/health", check=False)
    if h.returncode == 0:
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
    cache_dir = host_cfg.get("model_cache_dir", ".cache/huggingface")

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
        _ssh(ssh_host, f"rm -rf ~/{cache_dir}")
        print(f"  Removed ~/{cache_dir}")

    print(f"\nNuke complete. {ssh_host} is clean.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage KB embedding service on remote GPU hosts",
        prog="python3 plugins/kb/deploy/cli.py",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # deploy
    p = sub.add_parser("deploy", help="Sync code + install deps + configure systemd")
    p.add_argument("host", nargs="?", help="Host name from conf/kb/default.yaml")
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
    p = sub.add_parser("health", help="Quick health check")
    p.add_argument("host", nargs="?", help="Host name")
    p.add_argument("--all", action="store_true", help="Health check all hosts")

    # nuke
    p = sub.add_parser("nuke", help="Nuclear option: stop + remove everything")
    p.add_argument("host", nargs="?", help="Host name")
    p.add_argument("--data", action="store_true", help="Also remove model cache")
    p.add_argument("--all", action="store_true", help="Nuke all hosts")

    args = parser.parse_args()
    cfg = _load_config()

    # Resolve target hosts
    use_all = getattr(args, "all", False)
    if use_all:
        targets = _all_host_names(cfg)
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
