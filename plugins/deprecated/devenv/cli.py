#!/usr/bin/env python3
"""Local development environment manager for kernel_lab.

Auto-detects the current host from conf/hosts/default.yaml, then manages
the project .venv with the correct CUDA toolkit, PyTorch build, driver
workarounds, and network constraints.

Usage:
    python plugins/devenv/cli.py info              # detected host + env status
    python plugins/devenv/cli.py setup [--rebuild]  # create .venv + install deps
    python plugins/devenv/cli.py test               # verify everything works
    python plugins/devenv/cli.py nuke               # remove .venv
    python plugins/devenv/cli.py activate           # print shell exports
    python plugins/devenv/cli.py run -- <cmd...>    # run with correct env vars

Host names come from conf/hosts/default.yaml.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_VENV_DIR = _REPO_ROOT / ".venv"
_HOSTS_YAML = _REPO_ROOT / "conf" / "hosts" / "default.yaml"
_REQUIREMENTS = _SCRIPT_DIR / "requirements.txt"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load YAML. Tries PyYAML first, errors if unavailable."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not available. Install: pip install pyyaml",
              file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Host detection
# ---------------------------------------------------------------------------

def _detect_host(hosts_cfg: dict) -> tuple[str, dict] | None:
    """Match current hostname to a host entry in the config.

    Matches by FQDN or short hostname against the ssh_host field.
    Returns (host_name, host_config) or None.
    """
    fqdn = socket.getfqdn()
    short = socket.gethostname()

    hosts = hosts_cfg.get("hosts", {})
    for name, cfg in hosts.items():
        ssh_host = cfg.get("ssh_host", "")
        if ssh_host == fqdn or fqdn.startswith(ssh_host.rstrip(".")):
            return name, cfg
        if short in ssh_host or ssh_host in short:
            return name, cfg
    return None


def _get_env(host_cfg: dict) -> dict:
    """Extract env section with safe defaults."""
    e = host_cfg.get("env") or {}
    return {
        "cuda_home":       e.get("cuda_home"),
        "cuda_available":  e.get("cuda_available", []),
        "torch_cuda":      e.get("torch_cuda"),
        "torch_cuda_arch": e.get("torch_cuda_arch"),
        "ld_preload":      e.get("ld_preload"),
        "python":          e.get("python", "python3"),
    }


def _get_network(host_cfg: dict) -> dict:
    """Extract network section with safe defaults."""
    n = host_cfg.get("network") or {}
    return {
        "internet":              n.get("internet", True),
        "proxy_bypass_method":   n.get("proxy_bypass_method"),
        "pypi_nvidia_blocked":   n.get("pypi_nvidia_blocked", False),
    }


def _get_hardware(host_cfg: dict) -> dict:
    """Extract hardware section."""
    return host_cfg.get("hardware") or {}


# ---------------------------------------------------------------------------
# pip helpers
# ---------------------------------------------------------------------------

def _build_env_vars(env: dict) -> dict:
    """Build environment variable overrides from host env config."""
    out = {}
    if env.get("cuda_home"):
        out["CUDA_HOME"] = env["cuda_home"]
    if env.get("torch_cuda_arch"):
        out["TORCH_CUDA_ARCH_LIST"] = env["torch_cuda_arch"]
    if env.get("ld_preload"):
        out["LD_PRELOAD"] = env["ld_preload"]
    return out


def _pip_install(args: list[str], env: dict, net: dict,
                 timeout: int = 600) -> bool:
    """Run pip install with correct env and network bypass."""
    pip = str(_VENV_DIR / "bin" / "pip")
    env_vars = _build_env_vars(env)

    if net.get("proxy_bypass_method") == "ssh_localhost":
        # Wrap in SSH to bypass sandbox proxy
        env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())
        # Escape args for shell
        escaped = []
        for a in args:
            if any(c in a for c in " >=<'\""):
                escaped.append(f"'{a}'")
            else:
                escaped.append(a)
        cmd = (f'ssh localhost "cd {_REPO_ROOT} && '
               f'{env_prefix + " " if env_prefix else ""}'
               f'{pip} install {" ".join(escaped)}"')
        r = subprocess.run(cmd, shell=True, timeout=timeout)
    else:
        cmd = [pip, "install"] + args
        run_env = {**os.environ, **env_vars}
        r = subprocess.run(cmd, env=run_env, timeout=timeout)

    return r.returncode == 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_info(host_name: str | None, host_cfg: dict | None) -> int:
    """Show detected host and environment status."""
    fqdn = socket.getfqdn()
    print(f"Hostname:    {fqdn}")

    if host_name is None:
        print(f"Host match:  NONE — not found in {_HOSTS_YAML}")
        print(f"             Add this host to the config file.")
        return 1

    hw = _get_hardware(host_cfg)
    env = _get_env(host_cfg)
    net = _get_network(host_cfg)

    print(f"Host match:  {host_name}")
    print(f"GPU:         {hw.get('gpu', '?')} x{hw.get('gpu_count', '?')}")
    print(f"Driver:      {hw.get('driver_version', '?')} (CUDA {hw.get('driver_cuda_version', '?')})")
    print(f"CUDA home:   {env['cuda_home'] or '?'}")
    print(f"CUDA avail:  {', '.join(str(v) for v in env['cuda_available']) or '?'}")
    print(f"Torch index: {env['torch_cuda'] or '?'}")

    arch = env.get("torch_cuda_arch")
    if arch:
        sm = f"sm_{arch.replace('.', '')}"
        print(f"Arch:        {sm}")
    else:
        print(f"Arch:        ?")

    print(f"LD_PRELOAD:  {env['ld_preload'] or 'not needed'}")

    inet = "yes" if net["internet"] else "NO"
    bypass = f" (bypass: {net['proxy_bypass_method']})" if net["proxy_bypass_method"] else ""
    print(f"Internet:    {inet}{bypass}")

    # --- venv status ---
    print()
    if not _VENV_DIR.exists():
        print(f"Venv:        NOT FOUND")
        print(f"             Run: python plugins/devenv/cli.py setup")
        return 0

    print(f"Venv:        {_VENV_DIR}")
    python = _VENV_DIR / "bin" / "python"
    if not python.exists():
        print(f"             (broken — no bin/python)")
        return 1

    # Python version
    r = subprocess.run([str(python), "--version"], capture_output=True, text=True)
    print(f"Python:      {r.stdout.strip()}")

    # PyTorch
    r = subprocess.run(
        [str(python), "-c",
         "import torch; print(f'{torch.__version__} (CUDA {torch.version.cuda})')"],
        capture_output=True, text=True,
    )
    print(f"PyTorch:     {r.stdout.strip() or 'not installed'}")

    # FA4
    r = subprocess.run(
        [str(python), "-c",
         "from importlib.metadata import version; print(version('flash-attn-4'))"],
        capture_output=True, text=True,
    )
    print(f"FA4:         {r.stdout.strip() or 'not installed'}")

    # triton
    r = subprocess.run(
        [str(python), "-c", "import triton; print(triton.__version__)"],
        capture_output=True, text=True,
    )
    print(f"triton:      {r.stdout.strip() or 'not installed'}")

    return 0


def cmd_setup(host_name: str, host_cfg: dict, rebuild: bool = False) -> int:
    """Create .venv and install all dependencies for this host."""
    env = _get_env(host_cfg)
    net = _get_network(host_cfg)
    hw = _get_hardware(host_cfg)

    if not env["torch_cuda"]:
        print("ERROR: env.torch_cuda not set in host config.", file=sys.stderr)
        return 1
    if not env["cuda_home"]:
        print("ERROR: env.cuda_home not set in host config.", file=sys.stderr)
        return 1

    print(f"=== Setting up .venv for {host_name} ===")
    print(f"    GPU:         {hw.get('gpu')} x{hw.get('gpu_count')}")
    print(f"    Driver:      {hw.get('driver_version')} (CUDA {hw.get('driver_cuda_version')})")
    print(f"    CUDA_HOME:   {env['cuda_home']}")
    print(f"    Torch index: {env['torch_cuda']}")
    if env["ld_preload"]:
        print(f"    LD_PRELOAD:  {env['ld_preload']}")
    if net["proxy_bypass_method"]:
        print(f"    Network:     pip via {net['proxy_bypass_method']}")
    print()

    # ---------------------------------------------------------------
    # Step 1: Create venv
    # ---------------------------------------------------------------
    if rebuild and _VENV_DIR.exists():
        print("[1/4] Removing existing venv...")
        shutil.rmtree(_VENV_DIR)

    python = env.get("python", "python3")
    if not _VENV_DIR.exists():
        print(f"[1/4] Creating venv ({python})...")
        subprocess.run([python, "-m", "venv", str(_VENV_DIR)], check=True)
        subprocess.run(
            [str(_VENV_DIR / "bin" / "pip"), "install", "--upgrade",
             "pip", "setuptools", "wheel"],
            capture_output=True,
        )
    else:
        print("[1/4] Venv exists, reusing.")

    # ---------------------------------------------------------------
    # Step 2: Install PyTorch from CUDA-specific index
    # ---------------------------------------------------------------
    torch_cuda = env["torch_cuda"]
    torch_index = f"https://download.pytorch.org/whl/{torch_cuda}"
    print(f"[2/4] Installing PyTorch (index: {torch_cuda})...")
    ok = _pip_install(
        ["torch>=2.11", f"--index-url={torch_index}"],
        env, net, timeout=600,
    )
    if not ok:
        print("ERROR: PyTorch install failed.", file=sys.stderr)
        return 1
    print("      Done.")

    # ---------------------------------------------------------------
    # Step 3: Install core dependencies (from requirements.txt)
    # ---------------------------------------------------------------
    print("[3/4] Installing core dependencies...")

    # Parse requirements.txt
    deps = []
    for line in _REQUIREMENTS.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        deps.append(line)

    ok = _pip_install(deps, env, net, timeout=600)
    if not ok:
        print("ERROR: Core dependency install failed.", file=sys.stderr)
        return 1
    print("      Done.")

    # ---------------------------------------------------------------
    # Step 4: Update FA4 CuTe DSL (flash_attn.cute from git main)
    # ---------------------------------------------------------------
    print("[4/4] Updating FA4 CuTe DSL (flash_attn.cute from git main)...")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = _VENV_DIR / "lib" / f"python{py_ver}" / "site-packages"
    fa_cute_dir = site_packages / "flash_attn" / "cute"

    clone_cmd = (
        "rm -rf /tmp/_fa4_clone && "
        "git clone --depth 1 --sparse "
        "https://github.com/Dao-AILab/flash-attention.git /tmp/_fa4_clone && "
        "cd /tmp/_fa4_clone && git sparse-checkout set flash_attn/cute && "
        f"rm -rf {fa_cute_dir} && "
        f"cp -r flash_attn/cute {fa_cute_dir} && "
        "rm -rf /tmp/_fa4_clone"
    )
    if net.get("proxy_bypass_method") == "ssh_localhost":
        clone_cmd = f'ssh localhost "{clone_cmd}"'
    r = subprocess.run(clone_cmd, shell=True, timeout=120)
    if r.returncode == 0 and fa_cute_dir.exists():
        print("      Updated from git main.")
    else:
        if fa_cute_dir.exists():
            print("      WARNING: git pull failed, keeping existing version.")
        else:
            print("      WARNING: failed — FA4 CuTe DSL won't work.")

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    print()
    print("=== Setup complete ===")
    print(f"Venv: {_VENV_DIR}")
    print()
    print("Next:")
    print(f"  python plugins/devenv/cli.py test       # verify")
    print(f"  eval $(python plugins/devenv/cli.py activate)  # activate shell")
    return 0


def cmd_test(host_name: str, host_cfg: dict) -> int:
    """Verify the environment works end-to-end."""
    env = _get_env(host_cfg)

    if not _VENV_DIR.exists():
        print("ERROR: .venv not found. Run: python plugins/devenv/cli.py setup",
              file=sys.stderr)
        return 1

    python = str(_VENV_DIR / "bin" / "python")
    run_env = {**os.environ, **_build_env_vars(env)}

    test_script = r'''
import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  PASS  {name}: {result}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False

all_ok = True

# 1. PyTorch + CUDA
all_ok &= check("torch", lambda: __import__("torch").__version__)
all_ok &= check("torch.cuda", lambda: (
    f"CUDA {__import__('torch').version.cuda}, "
    f"{__import__('torch').cuda.device_count()} GPUs, "
    f"{__import__('torch').cuda.get_device_name(0)}"
))

# 2. CUDA matmul smoke test
def matmul_test():
    import torch
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    y = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    z = x @ y
    assert z.shape == (256, 256)
    return f"fp16 matmul OK"
all_ok &= check("cuda matmul", matmul_test)

# 3. FA4 CuTe DSL
def fa4_test():
    import os, torch
    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    from flash_attn.cute import flash_attn_func
    q = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device="cuda")
    out = flash_attn_func(q, k, v, causal=False)
    if isinstance(out, tuple): out = out[0]
    return f"output {out.shape}"
all_ok &= check("FA4 cute", fa4_test)

# 4. Key packages
all_ok &= check("numpy", lambda: __import__("numpy").__version__)
all_ok &= check("triton", lambda: __import__("triton").__version__)
all_ok &= check("cutlass-dsl", lambda: (__import__("nvidia_cutlass_dsl"), "OK")[1])
all_ok &= check("cuda-python", lambda: (__import__("cuda.bindings"), "OK")[1])
all_ok &= check("einops", lambda: __import__("einops").__version__)

print()
if all_ok:
    print("All critical tests PASSED.")
    sys.exit(0)
else:
    print("Some tests FAILED.")
    sys.exit(1)
'''

    print(f"=== Testing environment ({host_name}) ===")
    print()
    r = subprocess.run([python, "-c", test_script], env=run_env)
    return r.returncode


def cmd_nuke() -> int:
    """Remove .venv completely."""
    if not _VENV_DIR.exists():
        print("Nothing to remove — .venv does not exist.")
        return 0

    r = subprocess.run(["du", "-sh", str(_VENV_DIR)],
                       capture_output=True, text=True)
    size = r.stdout.strip().split("\t")[0] if r.returncode == 0 else "?"

    print(f"Removing {_VENV_DIR} ({size})...")
    shutil.rmtree(_VENV_DIR)
    print("Done.")
    print("Recreate with: python plugins/devenv/cli.py setup")
    return 0


def cmd_activate(host_cfg: dict) -> int:
    """Print shell export commands. Use: eval $(python plugins/devenv/cli.py activate)"""
    env = _get_env(host_cfg)

    lines = [f'export PATH="{_VENV_DIR}/bin:$PATH"']
    if env.get("cuda_home"):
        lines.append(f'export CUDA_HOME="{env["cuda_home"]}"')
        lines.append(f'export PATH="{env["cuda_home"]}/bin:$PATH"')
    if env.get("ld_preload"):
        lines.append(f'export LD_PRELOAD="{env["ld_preload"]}"')
    if env.get("torch_cuda_arch"):
        lines.append(f'export TORCH_CUDA_ARCH_LIST="{env["torch_cuda_arch"]}"')
    lines.append('export TVM_FFI_DISABLE_TORCH_C_DLPACK=1')

    for line in lines:
        print(line)
    return 0


def cmd_run(host_cfg: dict, command: list[str]) -> int:
    """Run a command with the correct environment variables set."""
    if not command:
        print("Usage: python plugins/devenv/cli.py run -- <cmd...>",
              file=sys.stderr)
        return 1

    env = _get_env(host_cfg)
    run_env = os.environ.copy()

    run_env["PATH"] = f"{_VENV_DIR}/bin:{run_env.get('PATH', '')}"
    run_env["VIRTUAL_ENV"] = str(_VENV_DIR)
    run_env["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

    if env.get("cuda_home"):
        run_env["CUDA_HOME"] = env["cuda_home"]
        run_env["PATH"] = f"{env['cuda_home']}/bin:{run_env['PATH']}"
    if env.get("ld_preload"):
        run_env["LD_PRELOAD"] = env["ld_preload"]
    if env.get("torch_cuda_arch"):
        run_env["TORCH_CUDA_ARCH_LIST"] = env["torch_cuda_arch"]

    os.execvpe(command[0], command, run_env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local development environment manager",
        prog="python plugins/devenv/cli.py",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info",     help="Show detected host and env status")
    p = sub.add_parser("setup", help="Create .venv and install all deps")
    p.add_argument("--rebuild", action="store_true", help="Remove .venv first")
    sub.add_parser("test",     help="Verify environment works")
    sub.add_parser("nuke",     help="Remove .venv completely")
    sub.add_parser("activate", help="Print shell export commands")
    p = sub.add_parser("run",  help="Run command with correct env vars")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="Command (after --)")

    args = parser.parse_args()

    # Detect host
    cfg = _load_yaml(_HOSTS_YAML)
    match = _detect_host(cfg)
    host_name = match[0] if match else None
    host_cfg  = match[1] if match else None

    # info and nuke don't strictly require a detected host
    if args.command == "info":
        return cmd_info(host_name, host_cfg)
    if args.command == "nuke":
        return cmd_nuke()

    # Everything else requires a detected host
    if host_cfg is None:
        fqdn = socket.getfqdn()
        print(f"ERROR: host {fqdn} not found in {_HOSTS_YAML}", file=sys.stderr)
        print("Run 'info' to debug, or add this host to the config.", file=sys.stderr)
        return 1

    if args.command == "setup":
        return cmd_setup(host_name, host_cfg, rebuild=args.rebuild)
    if args.command == "test":
        return cmd_test(host_name, host_cfg)
    if args.command == "activate":
        return cmd_activate(host_cfg)
    if args.command == "run":
        cmd = args.cmd
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        return cmd_run(host_cfg, cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
