#!/usr/bin/env python3
"""ik:env — Development environment management for kernel_lab.

Auto-detects host from conf/hosts/default.yaml, manages Python venv,
CUDA toolkit, and NVIDIA driver.

IMPORTANT: This script must run WITHOUT .venv (it creates .venv).
Use any available python3 to run it — system, fbcode, or uv standalone.
The script itself installs the standalone Python and creates .venv from it.

Usage:
    python3 plugins/ik/scripts/ik_env.py status
    python3 plugins/ik/scripts/ik_env.py test
    python3 plugins/ik/scripts/ik_env.py install kit=venv
    python3 plugins/ik/scripts/ik_env.py install kit=venv python_version=3.12
    python3 plugins/ik/scripts/ik_env.py nuke kit=venv
    python3 plugins/ik/scripts/ik_env.py reinstall kit=venv
    python3 plugins/ik/scripts/ik_env.py install kit=cuda-toolkit
    python3 plugins/ik/scripts/ik_env.py install kit=cuda-toolkit version=13.0
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # plugins/ik/scripts/ → repo root
VENV_DIR = REPO_ROOT / ".venv"
HOSTS_YAML = REPO_ROOT / "conf" / "hosts" / "default.yaml"
REQUIREMENTS = REPO_ROOT / "plugins" / "ik" / "requirements.txt"

FBCODE_LD_SO = "/usr/local/fbcode/platform010/lib/ld.so"
SYSTEM_LD_SO = "/lib64/ld-linux-x86-64.so.2"


# ---------------------------------------------------------------------------
# YAML loader (no deps — can't use pyyaml before .venv exists)
# ---------------------------------------------------------------------------

def _load_yaml_simple(path: Path) -> dict:
    """Minimal YAML parser for host config. Handles our simple key: value format."""
    try:
        import yaml
        with path.open() as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    # Fallback: crude parser for flat-ish YAML
    # Good enough to extract host matching fields
    import re
    text = path.read_text()
    # Try json-loading after stripping comments
    lines = []
    for line in text.split("\n"):
        stripped = line.split("#")[0].rstrip()
        if stripped:
            lines.append(stripped)
    # This won't work for complex YAML — but we only need it for bootstrapping
    print("WARNING: PyYAML not available, host detection may be limited", file=sys.stderr)
    return {}


# ---------------------------------------------------------------------------
# Host detection
# ---------------------------------------------------------------------------

def detect_host(cfg: dict) -> tuple[str | None, dict]:
    """Match current hostname to a host in the config."""
    fqdn = socket.getfqdn()
    short = socket.gethostname()
    hosts = cfg.get("hosts", {})
    for name, hcfg in hosts.items():
        ssh_host = hcfg.get("ssh_host", "")
        if fqdn.startswith(ssh_host.rstrip(".")) or short in ssh_host:
            return name, hcfg
    return None, {}


def _get_env(host_cfg: dict) -> dict:
    e = host_cfg.get("env") or {}
    return {
        "cuda_home": e.get("cuda_home"),
        "cuda_available": e.get("cuda_available", []),
        "torch_cuda": e.get("torch_cuda"),
        "torch_cuda_arch": e.get("torch_cuda_arch"),
        "ld_preload": e.get("ld_preload"),
        "python": e.get("python", "python3"),
    }


def _get_network(host_cfg: dict) -> dict:
    n = host_cfg.get("network") or {}
    net = {
        "internet": n.get("internet", True),
        "proxy_bypass_method": n.get("proxy_bypass_method"),
    }
    # Auto-detect: if host config unavailable, test if pip can reach pypi.nvidia.com
    # (torch depends on nvidia packages — pypi.org may work but nvidia.com may be blocked)
    if not n and not net.get("proxy_bypass_method"):
        try:
            r = subprocess.run(
                ["python3", "-c",
                 "import urllib.request; urllib.request.urlopen('https://pypi.nvidia.com', timeout=3)"],
                capture_output=True, timeout=8,
            )
            if r.returncode != 0:
                net["internet"] = False
                net["proxy_bypass_method"] = "ssh_localhost"
        except Exception:
            net["internet"] = False
            net["proxy_bypass_method"] = "ssh_localhost"
    return net


def _get_hardware(host_cfg: dict) -> dict:
    return host_cfg.get("hardware") or {}


# ---------------------------------------------------------------------------
# Standalone Python management
# ---------------------------------------------------------------------------

def _find_uv() -> str | None:
    """Find uv binary."""
    for p in [Path.home() / ".local" / "bin" / "uv", Path("/usr/local/bin/uv")]:
        if p.exists():
            return str(p)
    r = shutil.which("uv")
    return r


def _find_standalone_python(version: str = "3.12") -> Path | None:
    """Find uv-installed standalone Python."""
    uv_pythons = Path.home() / ".local" / "share" / "uv" / "python"
    if not uv_pythons.exists():
        return None
    for d in sorted(uv_pythons.iterdir(), reverse=True):
        if f"cpython-{version}" in d.name and "linux" in d.name:
            py = d / "bin" / f"python{version}"
            if py.exists():
                return py
    return None


def _is_standalone(python_path: Path) -> bool:
    """Check if a Python binary uses system ld.so (not fbcode)."""
    try:
        r = subprocess.run(
            ["readelf", "-l", str(python_path)],
            capture_output=True, text=True, timeout=5,
        )
        return SYSTEM_LD_SO in r.stdout and FBCODE_LD_SO not in r.stdout
    except Exception:
        return False


def _install_standalone_python(version: str, net: dict) -> Path:
    """Install standalone Python via uv."""
    uv = _find_uv()
    if not uv:
        raise RuntimeError("uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")

    cmd = f"{uv} python install cpython-{version}"
    if net.get("proxy_bypass_method") == "ssh_localhost":
        cmd = f'ssh localhost "{cmd}"'

    print(f"  Installing Python {version} via uv...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"uv python install failed: {r.stderr}")

    py = _find_standalone_python(version)
    if not py:
        raise RuntimeError(f"Python {version} installed but binary not found")
    if not _is_standalone(py):
        raise RuntimeError(f"Python at {py} uses fbcode ld.so — not standalone")

    print(f"  Installed: {py}")
    return py


# ---------------------------------------------------------------------------
# pip helper
# ---------------------------------------------------------------------------

def _pip_install(args: list[str], net: dict, timeout: int = 600) -> bool:
    """Run pip install, routing through ssh localhost if needed."""
    pip = str(VENV_DIR / "bin" / "pip")
    if net.get("proxy_bypass_method") == "ssh_localhost":
        escaped = " ".join(f"'{a}'" if any(c in a for c in " >=<") else a for a in args)
        cmd = f'ssh localhost "cd {REPO_ROOT} && {pip} install {escaped}"'
        r = subprocess.run(cmd, shell=True, timeout=timeout)
    else:
        r = subprocess.run([pip, "install"] + args, timeout=timeout)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(host_name: str | None, host_cfg: dict) -> int:
    """Show environment status for all kits."""
    fqdn = socket.getfqdn()
    hw = _get_hardware(host_cfg)
    env = _get_env(host_cfg)
    net = _get_network(host_cfg)

    print(f"Host:        {host_name or '?'} ({fqdn})")
    print(f"GPU:         {hw.get('gpu_count', '?')}× {hw.get('gpu', '?')}")
    print(f"Driver:      {hw.get('driver_version', '?')} → CUDA {hw.get('driver_cuda_version', '?')}")
    print()

    # --- Kit: venv ---
    print("Kit: venv")
    standalone_py = _find_standalone_python()
    if standalone_py:
        is_std = _is_standalone(standalone_py)
        print(f"  Python:      {standalone_py.name} (uv standalone)  {'✓' if is_std else '✗ fbcode ld.so!'}")
    else:
        print(f"  Python:      not installed  ✗")

    if VENV_DIR.exists():
        venv_py = VENV_DIR / "bin" / "python"
        if venv_py.exists():
            r = subprocess.run([str(venv_py), "--version"], capture_output=True, text=True, timeout=5)
            ver = r.stdout.strip()
            is_std = _is_standalone(venv_py.resolve())
            print(f"  .venv:       {ver}  {'✓' if is_std else '✗ fbcode!'}")

            # Check torch
            r = subprocess.run(
                [str(venv_py), "-c", "import torch; print(f'{torch.__version__} CUDA {torch.version.cuda}')"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"  PyTorch:     {r.stdout.strip()}  ✓")
            else:
                print(f"  PyTorch:     not installed  ✗")

            # Check libcuda
            r = subprocess.run(
                [str(venv_py), "-c",
                 "import ctypes,os; lib=ctypes.CDLL('libcuda.so.1'); v=ctypes.c_int(); "
                 "lib.cuDriverGetVersion(ctypes.byref(v)); "
                 "pid=os.getpid(); lines=open(f'/proc/{pid}/maps').readlines(); "
                 "cuda=[l.strip().split('/')[-1] for l in lines if 'libcuda.so' in l]; "
                 "print(f'{cuda[0]} CUDA {v.value//1000}.{(v.value%1000)//10}')"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"  libcuda:     {r.stdout.strip()}  ✓")
        else:
            print(f"  .venv:       broken (no bin/python)  ✗")
    else:
        print(f"  .venv:       not found  ✗")

    print()

    # --- Kit: cuda-toolkit ---
    print("Kit: cuda-toolkit")
    cuda_home = env.get("cuda_home") or "/usr/local/cuda"
    avail = env.get("cuda_available", [])
    print(f"  CUDA_HOME:   {cuda_home}")
    print(f"  Available:   {', '.join(str(v) for v in avail) if avail else '?'}")
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if nvcc.exists():
        r = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=5)
        ver_line = [l for l in r.stdout.split("\n") if "release" in l]
        print(f"  nvcc:        {ver_line[0].strip() if ver_line else '?'}  ✓")
    else:
        print(f"  nvcc:        not found at {nvcc}  ✗")

    headers = Path(cuda_home) / "include" / "cuda_bf16.h"
    print(f"  Headers:     {'✓' if headers.exists() else '✗ missing'}")

    print()

    # --- Kit: cuda-driver ---
    print("Kit: cuda-driver")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        driver_ver = r.stdout.strip().split("\n")[0]  # first GPU only
        print(f"  Driver:      {driver_ver}  ✓")
    except Exception:
        print(f"  Driver:      not detected  ✗")

    inet = "yes" if net.get("internet") else "no"
    bypass = f" ({net['proxy_bypass_method']})" if net.get("proxy_bypass_method") else ""
    print(f"\nNetwork:     {inet}{bypass}")

    return 0


def cmd_install_venv(host_cfg: dict, python_version: str = "3.12", force: bool = False) -> int:
    """Install standalone Python + .venv + all deps."""
    env = _get_env(host_cfg)
    net = _get_network(host_cfg)
    hw = _get_hardware(host_cfg)

    torch_cuda = env.get("torch_cuda")
    if not torch_cuda:
        # Auto-detect from nvidia-smi driver version
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            driver_ver = r.stdout.strip().split("\n")[0]
            # Map driver to CUDA version: 580.x → 13.0, 570.x → 12.8, 550.x → 12.4
            major = int(driver_ver.split(".")[0])
            if major >= 580:
                torch_cuda = "cu130"
            elif major >= 570:
                torch_cuda = "cu128"
            elif major >= 560:
                torch_cuda = "cu126"
            elif major >= 550:
                torch_cuda = "cu124"
            else:
                torch_cuda = "cu121"
            print(f"  Auto-detected torch_cuda={torch_cuda} from driver {driver_ver}")
        except Exception:
            print("ERROR: Cannot determine CUDA version. Set env.torch_cuda in host config.", file=sys.stderr)
            return 1

    print(f"=== Installing kit=venv ===")
    print(f"    Python:    {python_version}")
    print(f"    Torch:     {torch_cuda}")
    print(f"    CUDA home: {env.get('cuda_home', '?')}")
    print()

    # [1/4] Standalone Python
    print(f"[1/4] Standalone Python {python_version}")
    py = _find_standalone_python(python_version)
    if py and _is_standalone(py) and not force:
        print(f"  Already installed: {py}")
    else:
        py = _install_standalone_python(python_version, net)

    # [2/4] Create .venv
    print(f"[2/4] Create .venv")
    if VENV_DIR.exists() and not force:
        # Verify existing venv uses standalone Python
        venv_py = VENV_DIR / "bin" / "python"
        if venv_py.exists() and _is_standalone(venv_py.resolve()):
            print(f"  Already exists with standalone Python")
        else:
            print(f"  Exists but uses fbcode Python — recreating")
            shutil.rmtree(VENV_DIR)
            subprocess.run([str(py), "-m", "venv", str(VENV_DIR)], check=True)
    else:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
        subprocess.run([str(py), "-m", "venv", str(VENV_DIR)], check=True)
    # Upgrade pip
    subprocess.run([str(VENV_DIR / "bin" / "pip"), "install", "--upgrade", "pip", "setuptools", "wheel"],
                   capture_output=True, timeout=60)
    print(f"  Created: {VENV_DIR}")

    # [3/4] PyTorch
    print(f"[3/4] PyTorch ({torch_cuda})")
    torch_index = f"https://download.pytorch.org/whl/{torch_cuda}"
    ok = _pip_install(["torch>=2.11", f"--index-url={torch_index}"], net, timeout=600)
    if not ok:
        print("ERROR: PyTorch install failed", file=sys.stderr)
        return 1
    print(f"  Done.")

    # [4/4] Dependencies
    print(f"[4/4] Dependencies")
    if REQUIREMENTS.exists():
        deps = []
        for line in REQUIREMENTS.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                deps.append(line)
        ok = _pip_install(deps, net, timeout=600)
        if not ok:
            print("ERROR: dependency install failed", file=sys.stderr)
            return 1
    else:
        print(f"  WARNING: {REQUIREMENTS} not found, skipping deps")
    print(f"  Done.")

    print()
    print("=== Install complete ===")
    print(f"Venv: {VENV_DIR}")
    print(f"Run: python3 plugins/ik/scripts/ik_env.py test")
    return 0


def cmd_test(host_cfg: dict) -> int:
    """Verify environment with GPU tests."""
    if not VENV_DIR.exists():
        print("ERROR: .venv not found. Run: python3 plugins/ik/scripts/ik_env.py install kit=venv", file=sys.stderr)
        return 1

    python = str(VENV_DIR / "bin" / "python")
    env_vars = _get_env(host_cfg)

    run_env = dict(os.environ)
    if env_vars.get("cuda_home"):
        run_env["CUDA_HOME"] = env_vars["cuda_home"]
    run_env["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

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

# Python standalone check
import os
pid = os.getpid()
with open(f"/proc/{pid}/maps") as f:
    for line in f:
        if "libcuda.so" in line:
            if "fbcode" in line:
                print(f"  FAIL  libcuda: fbcode version loaded!")
                all_ok = False
            else:
                lib = line.strip().split("/")[-1]
                print(f"  PASS  libcuda: {lib} (system)")
            break

all_ok &= check("torch", lambda: __import__("torch").__version__)
all_ok &= check("torch.cuda", lambda: (
    f"CUDA {__import__('torch').version.cuda}, "
    f"{__import__('torch').cuda.device_count()} GPUs, "
    f"{__import__('torch').cuda.get_device_name(0)}"
))

def matmul_test():
    import torch
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    y = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    z = x @ y
    return f"OK ({z.shape})"
all_ok &= check("cuda matmul", matmul_test)

all_ok &= check("numpy", lambda: __import__("numpy").__version__)
all_ok &= check("triton", lambda: __import__("triton").__version__)
all_ok &= check("cutlass-dsl", lambda: (__import__("nvidia_cutlass_dsl"), "OK")[1])
all_ok &= check("cuda-python", lambda: (__import__("cuda.bindings"), "OK")[1])
all_ok &= check("flash-attn-4", lambda: __import__("importlib.metadata").metadata.version("flash-attn-4"))

print()
if all_ok:
    print("All tests PASSED.")
    sys.exit(0)
else:
    print("Some tests FAILED.")
    sys.exit(1)
'''

    print("=== Testing environment ===")
    print()
    r = subprocess.run([python, "-c", test_script], env=run_env)
    return r.returncode


def cmd_nuke_venv() -> int:
    """Remove .venv."""
    if not VENV_DIR.exists():
        print("Nothing to remove — .venv does not exist.")
        return 0
    r = subprocess.run(["du", "-sh", str(VENV_DIR)], capture_output=True, text=True)
    size = r.stdout.strip().split("\t")[0] if r.returncode == 0 else "?"
    print(f"Removing {VENV_DIR} ({size})...")
    shutil.rmtree(VENV_DIR)
    print("Done.")
    return 0


def _resolve_cuda_version(version: str) -> tuple[str, str]:
    """Resolve CUDA version to (toolkit_version, driver_version).

    Uses NVIDIA's official download page to discover the runfile URL.
    Returns e.g. ("13.2.0", "595.45.04").
    """
    # Known mappings — add entries as new versions are tested.
    KNOWN = {
        "12.8": ("12.8.1", "570.86.15"),
        "13.0": ("13.0.1", "580.76.02"),
        "13.2": ("13.2.0", "595.45.04"),
    }
    # Accept "13.2", "13.2.0", or "auto"
    short = ".".join(version.split(".")[:2])
    if short in KNOWN:
        return KNOWN[short]
    full = version
    if full in {v for v, _ in KNOWN.values()}:
        for s, (fv, dv) in KNOWN.items():
            if fv == full:
                return fv, dv
    raise ValueError(
        f"Unknown CUDA version '{version}'. "
        f"Known: {', '.join(sorted(KNOWN.keys()))}. "
        f"Add the mapping to _resolve_cuda_version() or install manually."
    )


def cmd_install_cuda_toolkit(host_cfg: dict, version: str = "auto") -> int:
    """Install CUDA toolkit from NVIDIA's official runfile.

    Downloads from developer.download.nvidia.com and installs toolkit-only
    (no driver) to /usr/local/cuda-<major>.<minor>.
    """
    net = _get_network(host_cfg)

    # --- Resolve version ---
    if version == "auto":
        # Default to latest known
        version = "13.2"

    try:
        toolkit_ver, driver_ver = _resolve_cuda_version(version)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    short_ver = ".".join(toolkit_ver.split(".")[:2])  # "13.2"
    install_dir = Path(f"/usr/local/cuda-{short_ver}")

    # --- Pre-flight ---
    if install_dir.exists():
        nvcc = install_dir / "bin" / "nvcc"
        if nvcc.exists():
            r = subprocess.run(
                [str(nvcc), "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if f"V{toolkit_ver}" in r.stdout:
                print(f"CUDA {toolkit_ver} already installed at {install_dir}")
                return 0
            else:
                print(f"WARNING: {install_dir} exists but nvcc reports different version.")
                print(f"  Remove it first: sudo rm -rf {install_dir}")
                return 1
        else:
            print(f"WARNING: {install_dir} exists but has no nvcc — may be incomplete.")
            print(f"  Remove it first: sudo rm -rf {install_dir}")
            return 1

    # --- Download ---
    runfile = f"cuda_{toolkit_ver}_{driver_ver}_linux.run"
    url = f"https://developer.download.nvidia.com/compute/cuda/{toolkit_ver}/local_installers/{runfile}"
    tmp_dir = Path("/tmp/cuda_install")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    runfile_path = tmp_dir / runfile

    print(f"=== Installing CUDA Toolkit {toolkit_ver} ===")
    print(f"    Source:  {url}")
    print(f"    Target:  {install_dir}")
    print()

    if runfile_path.exists():
        print(f"[1/3] Runfile already downloaded: {runfile_path}")
    else:
        print(f"[1/3] Downloading {runfile} ...")
        wget_cmd = f"wget -q --show-progress -O '{runfile_path}' '{url}'"
        if net.get("proxy_bypass_method") == "ssh_localhost":
            wget_cmd = f'ssh localhost "{wget_cmd}"'
        r = subprocess.run(wget_cmd, shell=True, timeout=1800)
        if r.returncode != 0:
            print(f"ERROR: Download failed.", file=sys.stderr)
            runfile_path.unlink(missing_ok=True)
            return 1
        print(f"  Downloaded: {runfile_path}")

    # --- Make executable ---
    runfile_path.chmod(0o755)

    # --- Install (toolkit only, no driver) ---
    print(f"[2/3] Installing toolkit (no driver) ...")
    install_cmd = (
        f"sudo '{runfile_path}' --silent --toolkit "
        f"--toolkitpath='{install_dir}' --no-opengl-libs --no-drm --no-man-page"
    )
    r = subprocess.run(install_cmd, shell=True, timeout=600)
    if r.returncode != 0:
        print(f"ERROR: Installation failed (exit {r.returncode}).", file=sys.stderr)
        print(f"  Check /var/log/cuda-installer.log for details.", file=sys.stderr)
        return 1

    # --- Verify ---
    print(f"[3/3] Verifying ...")
    nvcc = install_dir / "bin" / "nvcc"
    if not nvcc.exists():
        print(f"ERROR: nvcc not found at {nvcc} after install.", file=sys.stderr)
        return 1
    r = subprocess.run(
        [str(nvcc), "--version"],
        capture_output=True, text=True, timeout=5,
    )
    print(f"  {r.stdout.strip().splitlines()[-1]}")

    headers = install_dir / "include" / "cuda.h"
    print(f"  Headers: {'OK' if headers.exists() else 'MISSING'}")

    nvrtc = install_dir / "lib64" / "libnvrtc.so"
    if nvrtc.exists() or (install_dir / "lib64" / "libnvrtc.so.13").exists():
        print(f"  libnvrtc: OK")
    else:
        print(f"  libnvrtc: not found — cuDNN JIT may not work")

    print()
    print(f"=== CUDA {toolkit_ver} installed at {install_dir} ===")
    print(f"Set CUDA_HOME={install_dir} and update PATH/LD_LIBRARY_PATH.")
    print(f"Cleanup: rm {runfile_path}")
    return 0


def _resolve_driver_version(version: str) -> str:
    """Resolve driver version shorthand to full version string.

    Accepts: "595", "595.45.04", "auto", or a CUDA version like "13.2".
    Returns full driver version e.g. "595.45.04".
    """
    # Known driver versions (major → full)
    KNOWN_DRIVERS = {
        "550": "550.90.07",
        "570": "570.86.15",
        "580": "580.76.02",
        "595": "595.45.04",
    }
    # CUDA version → driver mapping (reuse _resolve_cuda_version)
    CUDA_TO_DRIVER = {
        "12.4": "550.90.07",
        "12.8": "570.86.15",
        "13.0": "580.76.02",
        "13.2": "595.45.04",
    }

    if version == "auto":
        # Default to driver matching latest known CUDA
        return "595.45.04"

    # Try as driver major (e.g. "595")
    if version in KNOWN_DRIVERS:
        return KNOWN_DRIVERS[version]

    # Try as full driver version (e.g. "595.45.04")
    major = version.split(".")[0]
    if major in KNOWN_DRIVERS and version.count(".") >= 1:
        return version  # Trust user-provided full version

    # Try as CUDA version (e.g. "13.2")
    if version in CUDA_TO_DRIVER:
        return CUDA_TO_DRIVER[version]

    raise ValueError(
        f"Unknown driver version '{version}'. "
        f"Known drivers: {', '.join(sorted(KNOWN_DRIVERS.keys()))}. "
        f"Or specify a CUDA version: {', '.join(sorted(CUDA_TO_DRIVER.keys()))}."
    )


def _stop_gpu_services() -> list[str]:
    """Stop services that hold nvidia devices open. Returns list of stopped services."""
    import time
    # Order matters: stop consumers first, then persistence daemon last.
    # Includes Meta infra services that hold GPU devices open.
    SERVICES = [
        "nvidia-dcgm",                    # DCGM monitoring (nv-hostengine)
        "dynologd",                       # Meta performance monitoring daemon
        "fbagentcollectors-workload",     # Meta agent collectors (manages rgpu)
        "nvidia-fabricmanager",           # NVSwitch fabric manager
        "nvidia-persistenced",           # GPU persistence daemon (must be last)
    ]
    stopped = []
    for svc in SERVICES:
        r = subprocess.run(
            ["sudo", "systemctl", "is-active", "--quiet", svc],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:  # service is active
            print(f"  Stopping {svc}...")
            try:
                subprocess.run(["sudo", "systemctl", "stop", svc], timeout=60)
            except subprocess.TimeoutExpired:
                print(f"  WARNING: {svc} stop timed out, force-killing...")
                subprocess.run(
                    ["sudo", "systemctl", "kill", "-s", "SIGKILL", svc],
                    capture_output=True, timeout=10,
                )
            stopped.append(svc)
    time.sleep(3)
    # Kill any remaining processes using nvidia devices, with retry
    nvidia_devs = [f"/dev/nvidia{i}" for i in range(8)] + [
        "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools",
        "/dev/nvidiactl",
    ]
    for attempt in range(3):
        any_procs = False
        for dev in nvidia_devs:
            if not os.path.exists(dev):
                continue
            r = subprocess.run(
                ["sudo", "fuser", dev],
                capture_output=True, text=True, timeout=5,
            )
            pids = r.stdout.strip().split()
            if pids:
                any_procs = True
                if attempt == 0:
                    print(f"  Killing PIDs on {dev}: {' '.join(pids)}")
                subprocess.run(
                    ["sudo", "kill", "-9"] + pids,
                    capture_output=True, timeout=10,
                )
        if not any_procs:
            break
        time.sleep(3)
    return stopped


def _start_gpu_services(services: list[str]) -> None:
    """Restart previously stopped services in reverse order."""
    for svc in reversed(services):
        print(f"  Starting {svc}...")
        try:
            subprocess.run(["sudo", "systemctl", "start", svc], timeout=60)
        except subprocess.TimeoutExpired:
            print(f"  WARNING: {svc} start timed out — may need manual restart")


def _unload_nvidia_modules() -> bool:
    """Try to unload all nvidia kernel modules. Returns True if successful."""
    import time
    # Order: dependents first
    MODULES = ["nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia"]
    for mod in MODULES:
        r = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
        if mod not in r.stdout:
            continue
        print(f"  Unloading {mod}...")
        try:
            r = subprocess.run(
                ["sudo", "rmmod", mod],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                print(f"  WARNING: Failed to unload {mod}: {r.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  WARNING: rmmod {mod} timed out (module likely still in use)")
            return False
        time.sleep(1)
    return True


def cmd_install_cuda_driver(host_cfg: dict, version: str = "auto") -> int:
    """Install NVIDIA driver from official runfile.

    Downloads from download.nvidia.com/XFree86/ and installs with --silent.
    Automatically stops GPU services, unloads modules, installs, then restarts.
    Requires root (uses sudo). A reboot is recommended after installation.
    """
    net = _get_network(host_cfg)

    # --- Resolve version ---
    try:
        driver_ver = _resolve_driver_version(version)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # --- Check current driver ---
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        current_driver = r.stdout.strip().split("\n")[0]
    except Exception:
        current_driver = None

    if current_driver == driver_ver:
        print(f"Driver {driver_ver} is already installed.")
        return 0

    print(f"=== Installing NVIDIA Driver {driver_ver} ===")
    print(f"    Current: {current_driver or 'unknown'}")
    print(f"    Target:  {driver_ver}")
    print()

    # Check root
    if os.geteuid() != 0:
        print("NOTE: This command needs root. Will use sudo.")

    # --- Download ---
    runfile = f"NVIDIA-Linux-x86_64-{driver_ver}.run"
    url = f"https://download.nvidia.com/XFree86/Linux-x86_64/{driver_ver}/{runfile}"
    tmp_dir = Path("/tmp/nvidia_driver_install")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    runfile_path = tmp_dir / runfile

    if runfile_path.exists():
        print(f"[1/5] Runfile already downloaded: {runfile_path}")
    else:
        print(f"[1/5] Downloading {runfile} ...")
        wget_cmd = f"wget -q --show-progress -O '{runfile_path}' '{url}'"
        if net.get("proxy_bypass_method") == "ssh_localhost":
            wget_cmd = f'ssh localhost "{wget_cmd}"'
        r = subprocess.run(wget_cmd, shell=True, timeout=600)
        if r.returncode != 0:
            print(f"ERROR: Download failed.", file=sys.stderr)
            runfile_path.unlink(missing_ok=True)
            return 1
        print(f"  Downloaded: {runfile_path}")

    runfile_path.chmod(0o755)

    # --- Stop services holding GPU devices ---
    print(f"[2/5] Stopping GPU services ...")
    stopped_services = _stop_gpu_services()

    # --- Unload nvidia modules ---
    print(f"[3/5] Unloading nvidia kernel modules ...")
    modules_unloaded = _unload_nvidia_modules()
    if not modules_unloaded:
        print(f"  WARNING: Could not fully unload modules. Installer will attempt anyway.")

    # --- Install ---
    print(f"[4/5] Installing driver {driver_ver} ...")
    # Detect if kernel was built with clang; if so, use full LLVM toolchain.
    # We must pass LLVM=1 so the kernel Makefile uses llvm-nm, llvm-ar, etc.
    env_vars = {}
    try:
        r = subprocess.run(
            ["cat", "/proc/version"],
            capture_output=True, text=True, timeout=5,
        )
        if "clang" in r.stdout.lower():
            cc_path = shutil.which("clang")
            if cc_path:
                # Create symlinks so kernel build scripts find LLVM tools
                # as default 'nm', 'ar', etc. in PATH. Also set CC + env vars.
                compat_dir = Path("/tmp/llvm-compat")
                compat_dir.mkdir(exist_ok=True)
                llvm_tools = {
                    "nm": shutil.which("llvm-nm"),
                    "ar": shutil.which("llvm-ar"),
                    "ld": shutil.which("ld.lld"),
                    "objcopy": shutil.which("llvm-objcopy"),
                    "strip": shutil.which("llvm-strip"),
                }
                for name, target in llvm_tools.items():
                    if target:
                        link = compat_dir / name
                        link.unlink(missing_ok=True)
                        link.symlink_to(target)
                env_vars = {
                    "CC": cc_path,
                    "IGNORE_CC_MISMATCH": "1",
                    "PATH": f"{compat_dir}:{os.environ.get('PATH', '/usr/bin')}",
                }
                print(f"  Kernel was built with clang, using CC=clang + LLVM tools in PATH")
    except Exception:
        pass
    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
    # Use 'sudo env' to properly pass environment variables
    install_cmd = (
        f"sudo env {env_str} '{runfile_path}' --silent --no-questions "
        f"--ui=none --no-backup --no-nouveau-check "
        f"--no-cc-version-check --dkms "
        f"--allow-installation-with-running-driver "
        f"--no-check-for-alternate-installs"
    )
    r = subprocess.run(install_cmd, shell=True, timeout=600)
    install_ok = r.returncode == 0

    if not install_ok:
        print(f"ERROR: Driver installation failed (exit {r.returncode}).", file=sys.stderr)
        print(f"  Check /var/log/nvidia-installer.log for details.", file=sys.stderr)

    # --- Restart services (even if install failed — need GPU back) ---
    print(f"[5/5] Restarting GPU services ...")
    # Load modules first (may take time if kernel module was just built)
    try:
        subprocess.run(["sudo", "modprobe", "nvidia"], capture_output=True, timeout=60)
        subprocess.run(["sudo", "modprobe", "nvidia_uvm"], capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"  WARNING: modprobe timed out — reboot may be needed")
    _start_gpu_services(stopped_services)

    if not install_ok:
        return 1

    # --- Verify ---
    import time
    time.sleep(2)
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    new_driver = r.stdout.strip().split("\n")[0] if r.returncode == 0 else "unknown"

    print()
    if new_driver == driver_ver:
        print(f"  Driver: {new_driver}  OK")
    else:
        print(f"  Driver reports: {new_driver} (expected {driver_ver})")
        print(f"  A reboot is required to load the new driver.")

    print()
    print(f"=== Driver {driver_ver} installation complete ===")
    print(f"RECOMMENDED: Reboot to ensure clean module load.")
    print(f"  sudo reboot")
    print(f"Cleanup: rm {runfile_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    # Parse: first arg is action (positional), rest are key=value
    action = sys.argv[1]
    kwargs = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kwargs[k] = v
        else:
            print(f"ERROR: unexpected argument '{arg}'. Use key=value format.", file=sys.stderr)
            return 1

    # Load host config
    cfg = _load_yaml_simple(HOSTS_YAML) if HOSTS_YAML.exists() else {}
    host_name, host_cfg = detect_host(cfg)

    kit = kwargs.get("kit")

    # Non-destructive actions (no kit required)
    if action == "status":
        return cmd_status(host_name, host_cfg)
    elif action == "test":
        return cmd_test(host_cfg)

    # Destructive actions (kit required)
    if not kit:
        print(f"ERROR: '{action}' requires kit=venv|cuda-toolkit|cuda-driver", file=sys.stderr)
        return 1

    if action == "install":
        if kit == "venv":
            return cmd_install_venv(
                host_cfg,
                python_version=kwargs.get("python_version", "3.12"),
                force=kwargs.get("force", "false").lower() == "true",
            )
        elif kit == "cuda-toolkit":
            return cmd_install_cuda_toolkit(host_cfg, version=kwargs.get("version", "auto"))
        elif kit == "cuda-driver":
            return cmd_install_cuda_driver(host_cfg, version=kwargs.get("version", "auto"))
        else:
            print(f"ERROR: unknown kit '{kit}'. Use: venv, cuda-toolkit, cuda-driver", file=sys.stderr)
            return 1

    elif action == "nuke":
        if kit == "venv":
            return cmd_nuke_venv()
        else:
            print(f"ERROR: nuke only supports kit=venv", file=sys.stderr)
            return 1

    elif action == "reinstall":
        if kit == "venv":
            cmd_nuke_venv()
            return cmd_install_venv(
                host_cfg,
                python_version=kwargs.get("python_version", "3.12"),
                force=True,
            )
        else:
            print(f"ERROR: reinstall only supports kit=venv", file=sys.stderr)
            return 1

    else:
        print(f"ERROR: unknown action '{action}'. Use: status, test, install, nuke, reinstall", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
