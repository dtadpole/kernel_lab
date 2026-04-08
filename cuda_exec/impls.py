"""Implementation slug resolution for cuda_exec.

Shared utility that maps implementation slugs to source files.
Used by compile, trial, bench, and any future tool.

Slug format: {source}-{name}
  - source: "ref" (data/ref/{kernel}/), "gen" (~/kernel_lab_kb/runs/<run_tag>/gen/{arch}/{kernel}/),
            or "peak" (.peak/{arch}/{kernel}/)
  - name: file stem (e.g., "cublas", "cutedsl", "cuda")

Full identifier: {kernel}/{impl_slug}  (e.g., "matmul/ref-pytorch")

File resolution: slug → try .py first, then .cu
  - ref-pytorch  → data/ref/matmul/pytorch/pytorch.py
  - gen-cuda     → ~/kernel_lab_kb/runs/<run_tag>/gen/sm90/matmul/cuda/cuda.cu
  - gen-cutedsl  → ~/kernel_lab_kb/runs/<run_tag>/gen/sm90/matmul/cutedsl/cutedsl.py
  - peak-cuda    → .peak/sm90/fa4/cuda/cuda.cu

Helper files (dependencies of an entry point) are auto-discovered:
  - .py entry points: all other .py files in the same directory are included
  - .cu entry points: all .h/.cuh files in the same directory are included
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data"
_KB_REPO = Path.home() / "kernel_lab_kb"


def _ref_dir(kernel: str, data_root: Path | None = None) -> Path:
    return (data_root or _DEFAULT_DATA_ROOT) / "ref" / kernel


def _peak_dir(kernel: str, arch: str, data_root: Path | None = None) -> Path:
    if data_root:
        # Bench snapshots still use data_root/peak/ (copied during snapshot)
        return data_root / "peak" / arch / kernel
    return _PROJECT_ROOT / ".peak" / arch / kernel


def list_gems(kernel: str, arch: str, impl_name: str = "cuda",
              run_tag: str | None = None,
              kb_repo: Path | None = None) -> list[dict]:
    """List gems for a kernel/arch/impl within the SAME run, newest first.

    Gems are per-run. No cross-run access.
    If run_tag is None, uses the current host run (run_<host_slug>).

    Returns list of dicts with: version, path, gen_path, run, tflops, timestamp.
    """
    repo = kb_repo or _KB_REPO
    gems: list[dict] = []

    run_tag = _resolve_run_tag(run_tag)
    runs_dir = repo / "runs"
    run_dirs = [runs_dir / run_tag] if (runs_dir / run_tag).exists() else []

    # New structure: runs/run_*/gems/<slug>/v*/
    for run_dir in run_dirs:
        gem_base = run_dir / "gems" / f"gen-{impl_name}"
        if not gem_base.exists():
            continue
        for ver_dir in sorted(gem_base.glob("v*"), reverse=True):
            gen_path = ver_dir / "gen" / arch / kernel
            results_file = ver_dir / "results.json"
            entry = {
                "version": ver_dir.name.split("_")[0],  # e.g. "v001"
                "path": ver_dir,
                "gen_path": gen_path if gen_path.exists() else None,
                "run": run_dir.name,
                "timestamp": "_".join(ver_dir.name.split("_")[1:]),
            }
            # Extract TFLOPS from results.json if available
            if results_file.exists():
                try:
                    import json
                    results = json.loads(results_file.read_text())
                    best_tflops = 0.0
                    for cfg_data in results.get("configs", {}).values():
                        ms = cfg_data.get("gen_median_ms") or cfg_data.get("ref_median_ms")
                        if ms and ms > 0:
                            # Approximate TFLOPS from the largest config
                            pass
                    gem_info = results.get("gem", {})
                    entry["improved_configs"] = gem_info.get("improved_configs", [])
                    entry["gem_info"] = gem_info
                except Exception:
                    pass
            gems.append(entry)

    return gems


def reseed_gen(kernel: str, arch: str, *,
               run_tag: str | None = None,
               gem_path: Path | None = None,
               kb_repo: Path | None = None) -> Path:
    """Clear gen/ scratch and reseed from a gem.

    Args:
        kernel: kernel name
        arch: target arch
        run_tag: which run to reseed. None = auto-detect from host.
        gem_path: specific gem's gen/ path to seed from. None = latest gem.
        kb_repo: kernel_lab_kb path.

    Returns:
        Path to the reseeded gen/<arch>/<kernel>/ directory.
    """
    import os
    import shutil
    repo = kb_repo or _KB_REPO

    if not run_tag:
        run_tag = os.environ.get("CUDA_EXEC_RUN_TAG") or f"run_{_detect_host_slug()}"

    run_dir = repo / "runs" / run_tag
    gen_path = run_dir / "gen" / arch / kernel

    # Clear existing
    if gen_path.exists():
        shutil.rmtree(gen_path)

    # Find seed source from THIS run's gems only
    if gem_path is None:
        gem_path = _find_latest_gem(kernel, arch, run_tag=run_tag, kb_repo=repo)

    if gem_path and gem_path.exists():
        shutil.copytree(gem_path, gen_path)
        # Fix flat structure from old gems
        for cu_file in list(gen_path.glob("*.cu")):
            stem = cu_file.stem
            subdir = gen_path / stem
            if not subdir.exists():
                subdir.mkdir()
            (subdir / cu_file.name).replace(cu_file)
            for companion in gen_path.glob(f"{stem}.*"):
                if companion.is_file():
                    (subdir / companion.name).replace(companion)
    else:
        gen_path.mkdir(parents=True, exist_ok=True)

    return gen_path


def _find_latest_gem(kernel: str, arch: str, impl_name: str = "cuda",
                     run_tag: str | None = None,
                     kb_repo: Path | None = None) -> Path | None:
    """Find the latest gem's gen code within the SAME run only.

    Gems are per-run artifacts. No cross-run access.
    A fresh run with no gems returns None — the caller must seed explicitly.
    """
    import os
    repo = kb_repo or _KB_REPO
    runs_dir = repo / "runs"

    if not run_tag:
        run_tag = os.environ.get("CUDA_EXEC_RUN_TAG") or f"run_{_detect_host_slug()}"

    run_dir = runs_dir / run_tag
    if run_dir.exists():
        gem_base = run_dir / "gems" / f"gen-{impl_name}"
        if gem_base.exists():
            versions = sorted(gem_base.glob("v*"), reverse=True)
            for ver in versions:
                candidate = ver / "gen" / arch / kernel
                if candidate.exists():
                    return candidate

    return None


def _resolve_run_home(run_tag: str | None = None,
                      kb_repo: Path | None = None) -> Path:
    """Resolve the active KB run directory.

    Priority:
      1. Explicit run_tag argument
      2. CUDA_EXEC_RUN_TAG env var — set by Supervisor's AgentRunner
      3. IK_RUN_HOME env var — direct path to a run directory
      4. Fallback: kernel_lab_kb/runs/run_{host_slug}
    """
    import os
    repo = kb_repo or _KB_REPO

    if run_tag:
        return repo / "runs" / run_tag

    env_tag = os.environ.get("CUDA_EXEC_RUN_TAG")
    if env_tag:
        return repo / "runs" / env_tag

    env_home = os.environ.get("IK_RUN_HOME")
    if env_home:
        return Path(env_home)

    return repo / "runs" / f"run_{_detect_host_slug()}"


def _ensure_gen_dir(kernel: str, arch: str, *,
                    run_tag: str | None = None,
                    kb_repo: Path | None = None) -> Path:
    """Get or create the gen/ scratch dir in the active KB run.

    If the gen dir doesn't exist yet, seed it from the latest gem.
    No gems → no code. A fresh run starts empty. Does NOT auto-seed.
    The caller (ik:optimize) is responsible for seeding gen/ via reseed_gen().
    """
    run_dir = _resolve_run_home(run_tag, kb_repo)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "gen" / arch / kernel


def _resolve_run_tag(run_tag: str | None = None) -> str:
    """Resolve run_tag — kept for backward compat with list_gems/reseed_gen.

    Priority: explicit > CUDA_EXEC_RUN_TAG > IK_RUN_HOME dirname > run_{host_slug}.
    """
    import os
    if run_tag:
        return run_tag
    env_tag = os.environ.get("CUDA_EXEC_RUN_TAG")
    if env_tag:
        return env_tag
    env_home = os.environ.get("IK_RUN_HOME")
    if env_home:
        return Path(env_home).name
    return f"run_{_detect_host_slug()}"


def _detect_host_slug() -> str:
    """Detect host slug from conf/hosts/default.yaml matching current hostname.

    Uses simple string parsing if PyYAML is not available.
    """
    import socket
    fqdn = socket.getfqdn()
    hostname = socket.gethostname()
    cfg_path = _PROJECT_ROOT / "conf" / "hosts" / "default.yaml"
    if not cfg_path.exists():
        return hostname.split(".")[0]

    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        for slug, host_cfg in cfg.get("hosts", {}).items():
            ssh_host = host_cfg.get("ssh_host", "")
            if fqdn.startswith(ssh_host.rstrip(".")) or hostname in ssh_host:
                return slug
    except ImportError:
        # No PyYAML — parse manually: find lines like "  h8_3:" followed by "ssh_host: ..."
        import re
        text = cfg_path.read_text()
        for m in re.finditer(r"^\s{2}(\w+):\s*$", text, re.MULTILINE):
            slug = m.group(1)
            # Find ssh_host in the next few lines
            block = text[m.end():m.end() + 200]
            ssh_match = re.search(r"ssh_host:\s*['\"]?([^'\"#\n]+)", block)
            if ssh_match:
                ssh_host = ssh_match.group(1).strip()
                if fqdn.startswith(ssh_host.rstrip(".")) or hostname in ssh_host:
                    return slug
    except Exception:
        pass
    return hostname.split(".")[0]


def _gen_dir(kernel: str, arch: str, data_root: Path | None = None) -> Path:
    """Resolve gen directory.

    Priority:
      1. Explicit data_root (from bench run snapshot) — use directly
      2. IK_RUN_HOME env var → $IK_RUN_HOME/gen/{arch}/{kernel}
      3. Fallback: kernel_lab_kb/runs/run_{host_slug}/gen/{arch}/{kernel}
    """
    if data_root:
        return data_root / "gen" / arch / kernel

    return _ensure_gen_dir(kernel, arch)


def resolve_impl(
    kernel: str,
    arch: str,
    impl_slug: str,
    *,
    data_root: Path | None = None,
) -> dict:
    """Resolve an implementation slug to its source files and metadata.

    Each impl lives in its own subdirectory:
        ref-pytorch  → data/ref/{kernel}/pytorch/
        gen-cuda     → ~/kernel_lab_kb/runs/<run_tag>/gen/{arch}/{kernel}/cuda/
        gen-cutedsl  → ~/kernel_lab_kb/runs/<run_tag>/gen/{arch}/{kernel}/cutedsl/

    Entry point: {name}.py or {name}.cu inside the subdir.
    Helpers: all other files in the same subdir.

    Returns:
        {
            "slug": "ref-pytorch",
            "source": "ref",
            "name": "cublas",
            "entry_point": Path(...),
            "file_type": "py",
            "files": {"cublas.py": "..."},
        }
    """
    parts = impl_slug.split("-", 1)
    if len(parts) != 2 or parts[0] not in ("ref", "gen", "peak"):
        raise ValueError(
            f"Invalid impl slug '{impl_slug}'. "
            f"Format: '{{ref|gen|peak}}-{{name}}' (e.g., 'ref-pytorch', 'gen-cuda', 'peak-cuda')"
        )

    source, name = parts

    if source == "ref":
        impl_dir = _ref_dir(kernel, data_root) / name
    elif source == "peak":
        impl_dir = _peak_dir(kernel, arch, data_root) / name
    else:
        impl_dir = _gen_dir(kernel, arch, data_root) / name

    if not impl_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot resolve impl '{impl_slug}' for kernel '{kernel}' arch '{arch}'. "
            f"Directory not found: {impl_dir}"
        )

    # Resolve entry point: try {name}.py first, then {name}.cu
    entry_py = impl_dir / f"{name}.py"
    entry_cu = impl_dir / f"{name}.cu"

    if entry_py.exists():
        entry_point = entry_py
        file_type = "py"
    elif entry_cu.exists():
        entry_point = entry_cu
        file_type = "cu"
    else:
        raise FileNotFoundError(
            f"No entry point found in {impl_dir}. "
            f"Expected {name}.py or {name}.cu"
        )

    # Collect all files in the impl subdir
    files: Dict[str, str] = {}
    for f in impl_dir.iterdir():
        if f.is_file() and f.suffix in (".py", ".cu", ".h", ".cuh"):
            files[f.name] = f.read_text(encoding="utf-8")

    return {
        "slug": impl_slug,
        "source": source,
        "name": name,
        "entry_point": entry_point,
        "file_type": file_type,
        "files": files,
    }


def list_impls(kernel: str, arch: str, *, data_root: Path | None = None) -> List[dict]:
    """List all available implementations for a kernel+arch.

    Scans subdirectories of ref/{kernel}/ and gen/{arch}/{kernel}/.
    Each subdirectory is one impl. Entry point: {subdir_name}.py or .cu.

    Returns list of:
        {"slug": "ref-pytorch", "source": "ref", "name": "cublas",
         "file_type": "py", "path": Path(...)}
    """
    impls = []

    def _scan_dir(base_dir: Path, source: str) -> None:
        if not base_dir.exists():
            return
        for d in sorted(base_dir.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or d.name == "__pycache__":
                continue
            name = d.name
            # Check for entry point: {name}.py or {name}.cu
            entry_py = d / f"{name}.py"
            entry_cu = d / f"{name}.cu"
            if entry_py.exists():
                impls.append({
                    "slug": f"{source}-{name}",
                    "source": source,
                    "name": name,
                    "file_type": "py",
                    "path": entry_py,
                })
            elif entry_cu.exists():
                impls.append({
                    "slug": f"{source}-{name}",
                    "source": source,
                    "name": name,
                    "file_type": "cu",
                    "path": entry_cu,
                })

    _scan_dir(_ref_dir(kernel, data_root), "ref")
    _scan_dir(_peak_dir(kernel, arch, data_root), "peak")
    _scan_dir(_gen_dir(kernel, arch, data_root), "gen")

    # Check for a per-kernel priority config that overrides ref ordering.
    # Useful when a ref impl is broken in subprocess context (e.g., cuDNN segfault)
    # and a different ref should be the golden reference for correctness comparison.
    # File: data/ref/{kernel}/_priority.json  →  {"golden_ref": "ref-pytorch"}
    priority_file = _ref_dir(kernel, data_root) / "_priority.json"
    if priority_file.exists():
        try:
            import json as _json
            priority = _json.loads(priority_file.read_text())
            golden = priority.get("golden_ref")
            if golden:
                golden_items = [i for i in impls if i["slug"] == golden]
                others = [i for i in impls if i["slug"] != golden]
                if golden_items:
                    impls[:] = golden_items + others
        except Exception:
            pass  # Malformed priority file — fall back to alphabetical order

    return impls


def resolve_impls(
    kernel: str,
    arch: str,
    impl_slugs: str | List[str],
    *,
    data_root: Path | None = None,
) -> List[dict]:
    """Resolve one or more implementation slugs.

    Args:
        impl_slugs: "all" to resolve everything, or a list of slug strings

    Returns:
        List of resolved impl dicts (same as resolve_impl output)

    Raises:
        ValueError: if no reference implementation is found
    """
    if impl_slugs == "all":
        available = list_impls(kernel, arch, data_root=data_root)
        slugs = [impl["slug"] for impl in available]
    elif isinstance(impl_slugs, str):
        slugs = [impl_slugs]
    else:
        slugs = list(impl_slugs)

    if not slugs:
        raise ValueError(
            f"No implementations found for kernel '{kernel}' arch '{arch}'"
        )

    resolved = [resolve_impl(kernel, arch, s, data_root=data_root) for s in slugs]

    # Classify by slug prefix: ref- = reference, gen- = generated
    refs = [r for r in resolved if r["source"] == "ref"]
    gens = [r for r in resolved if r["source"] == "gen"]

    if not refs:
        raise ValueError(
            f"At least one reference (ref-*) implementation is required. "
            f"Got only generated (gen-*): {[g['slug'] for g in gens]}"
        )

    return resolved


def load_configs(kernel: str, *, data_root: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load ALL configs for a kernel."""
    root = data_root or _DEFAULT_DATA_ROOT
    configs_path = root / "configs" / f"{kernel}.json"
    if not configs_path.exists():
        raise FileNotFoundError(
            f"Configs not found: {configs_path}. "
            f"Available: {_available_configs_hint(data_root)}"
        )
    import json
    with open(configs_path, encoding="utf-8") as f:
        configs = json.load(f)
    if not configs:
        raise ValueError(f"Empty configs file: {configs_path}")
    return configs


def _available_configs_hint(data_root: Path | None = None) -> list[str]:
    root = data_root or _DEFAULT_DATA_ROOT
    configs_dir = root / "configs"
    if configs_dir.exists():
        return [p.stem for p in configs_dir.glob("*.json")]
    return []
