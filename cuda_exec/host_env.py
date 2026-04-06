"""Host environment resolution for CUDA compilation.

Resolves CUDA_HOME, LD_PRELOAD, and GPU architecture by matching the
current hostname against conf/hosts/default.yaml.  Falls back to system
defaults when no host entry matches.

Resolution order (per field):
  1. Match hostname against hosts config entries.
  2. Use the matched entry's env.cuda_home / env.ld_preload / env.torch_cuda_arch.
  3. Fall back to system detection (/usr/local/cuda, nvidia-smi).

Results are cached for the process lifetime — the host doesn't change mid-run.
"""

from __future__ import annotations

import functools
import logging
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
_HOSTS_CONFIG = _CONF_DIR / "hosts" / "default.yaml"


@functools.lru_cache(maxsize=1)
def _load_hosts_config() -> dict:
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not available — skipping host config")
        return {}
    if not _HOSTS_CONFIG.exists():
        logger.debug("Host config not found: %s", _HOSTS_CONFIG)
        return {}
    return yaml.safe_load(_HOSTS_CONFIG.read_text(encoding="utf-8")) or {}


@functools.lru_cache(maxsize=1)
def _match_host_entry() -> tuple[Optional[str], Optional[dict]]:
    """Match current hostname against host config entries.

    Returns (host_key, host_entry) or (None, None) if no match.
    """
    config = _load_hosts_config()
    hosts = config.get("hosts", {})
    if not hosts:
        return None, None

    fqdn = socket.getfqdn()
    hostname = socket.gethostname()

    for key, entry in hosts.items():
        ssh_host = (entry.get("ssh_host") or "").rstrip(".")
        if not ssh_host:
            continue
        if fqdn.startswith(ssh_host) or hostname == ssh_host:
            logger.debug("Matched host entry '%s' (fqdn=%s)", key, fqdn)
            return key, entry

    logger.debug("No host entry matched (fqdn=%s, hostname=%s)", fqdn, hostname)
    return None, None


def _detect_gpu_arch() -> Optional[str]:
    """Detect GPU compute capability via nvidia-smi.  Returns e.g. '9.0'."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            cap = result.stdout.strip().split("\n")[0].strip()
            if cap:
                return cap
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def resolve_host_env() -> Dict[str, str]:
    """Resolve CUDA environment variables for the current host.

    Returns a dict suitable for merging into subprocess env.  Keys are
    only included when the value is non-empty.

    Fields resolved:
      CUDA_HOME                          — path to the CUDA toolkit root
      LD_PRELOAD                         — library preload (e.g. fbcode libcuda workaround)
      TVM_FFI_DISABLE_TORCH_C_DLPACK    — CuTe DSL ABI workaround
    """
    _, entry = _match_host_entry()
    env: Dict[str, str] = {}

    if entry:
        host_env = entry.get("env", {})
        cuda_home = host_env.get("cuda_home")
        if cuda_home:
            env["CUDA_HOME"] = str(cuda_home)
        ld_preload = host_env.get("ld_preload")
        if ld_preload:
            env["LD_PRELOAD"] = str(ld_preload)
        if host_env.get("tvm_ffi_disable_torch_c_dlpack"):
            env["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    else:
        # Fallback: resolve /usr/local/cuda symlink
        default_cuda = Path("/usr/local/cuda")
        if default_cuda.exists():
            env["CUDA_HOME"] = str(default_cuda.resolve())

    # CUTLASS include paths for NVCC
    if entry:
        cutlass_include = entry.get("env", {}).get("cutlass_include")
        if cutlass_include:
            if isinstance(cutlass_include, list):
                dirs = [d for d in cutlass_include if Path(d).is_dir()]
            elif Path(cutlass_include).is_dir():
                dirs = [cutlass_include]
            else:
                dirs = []
            if dirs:
                env["NVCC_INCLUDE_DIRS"] = " ".join(str(d) for d in dirs)
                env["NVCC_EXTRA_LIBS"] = "cuda dl"

    key, _ = _match_host_entry()
    logger.info("Host env resolved [%s]: %s", key or "auto-detect", env)
    return env


@functools.lru_cache(maxsize=1)
def resolve_arch() -> str:
    """Resolve the GPU architecture directory name (e.g. 'sm90').

    Resolution order:
      1. Host config env.torch_cuda_arch (e.g. '9.0' → 'sm90')
      2. nvidia-smi auto-detection (e.g. '9.0' → 'sm90')
      3. Raises RuntimeError if neither works.
    """
    _, entry = _match_host_entry()
    arch_str = None

    if entry:
        arch_str = entry.get("env", {}).get("torch_cuda_arch")

    if not arch_str:
        arch_str = _detect_gpu_arch()

    if not arch_str:
        raise RuntimeError(
            "Cannot resolve GPU architecture: no host config match and "
            "nvidia-smi detection failed"
        )

    # '9.0' → 'sm90', '10.0' → 'sm100'
    return "sm" + arch_str.replace(".", "")


def resolve_compile_arch() -> str:
    """Resolve the nvcc/ptxas -arch flag (e.g. 'sm_90a').

    Always uses the arch-specific 'a' suffix to enable features like WGMMA.
    """
    arch_dir = resolve_arch()  # e.g. 'sm90'
    # 'sm90' → 'sm_90a'
    digits = arch_dir[2:]  # '90'
    return f"sm_{digits}a"


@functools.lru_cache(maxsize=1)
def resolve_gpu_peak_tflops() -> float:
    """Resolve GPU peak BF16 Tensor Core TFLOPS from host config.

    Returns the ``hardware.peak_tflops_bf16`` value, or a safe default.
    """
    _, entry = _match_host_entry()
    if entry:
        return float(entry.get("hardware", {}).get("peak_tflops_bf16", 800))
    return 800  # Meta H100 R&R SKU @ 650W default


def resolve_gpu_name() -> str:
    """Resolve GPU display name (e.g. 'NVIDIA H100 SXM5 80GB')."""
    _, entry = _match_host_entry()
    if entry:
        hw = entry.get("hardware", {})
        name = hw.get("gpu", "GPU")
        variant = hw.get("gpu_variant", "")
        return f"{name} {variant}".strip() if variant else name
    return "GPU"


def resolve_benchmark_gpus() -> Optional[str]:
    """Resolve the GPU indices for benchmarking (e.g. '4,5').

    From host config ``benchmark.cuda_visible_devices``.  Returns None if
    no host config match or no benchmark GPU assignment.
    """
    _, entry = _match_host_entry()
    if not entry:
        return None
    return entry.get("benchmark", {}).get("cuda_visible_devices")


def host_env_summary() -> Dict[str, Any]:
    """Return a human-readable summary of the resolved host environment.

    Useful for logging and diagnostics.
    """
    key, entry = _match_host_entry()
    env = resolve_host_env()
    try:
        arch = resolve_arch()
        compile_arch = resolve_compile_arch()
    except RuntimeError:
        arch = None
        compile_arch = None

    summary: Dict[str, Any] = {
        "host_key": key,
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "CUDA_HOME": env.get("CUDA_HOME"),
        "LD_PRELOAD": env.get("LD_PRELOAD"),
        "arch": arch,
        "compile_arch": compile_arch,
    }
    if entry:
        hw = entry.get("hardware", {})
        summary["gpu"] = hw.get("gpu")
        summary["driver_version"] = hw.get("driver_version")
        host_env_cfg = entry.get("env", {})
        summary["cuda_available"] = host_env_cfg.get("cuda_available", [])
    return summary
