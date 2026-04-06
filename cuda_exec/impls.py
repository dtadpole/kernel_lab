"""Implementation slug resolution for cuda_exec.

Shared utility that maps implementation slugs to source files.
Used by compile, trial, bench, and any future tool.

Slug format: {source}-{name}
  - source: "ref" (data/ref/{kernel}/) or "gen" (data/gen/{arch}/{kernel}/)
  - name: file stem (e.g., "cublas", "cutedsl", "cuda")

Full identifier: {kernel}/{impl_slug}  (e.g., "matmul/ref-cublas")

File resolution: slug → try .py first, then .cu
  - ref-cublas  → data/ref/matmul/cublas.py
  - gen-cutedsl → data/gen/sm90/matmul/cutedsl.py
  - gen-cuda    → data/gen/sm90/matmul/cuda.cu

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


def list_gems(kernel: str, arch: str, impl_name: str = "cuda",
              run_tag: str | None = None,
              kb_repo: Path | None = None) -> list[dict]:
    """List all available gems for a kernel/arch/impl, newest first.

    If run_tag is specified, only lists gems from that run.
    Otherwise lists gems across all runs.

    Returns list of dicts with: version, path, gen_path, run, tflops, timestamp.
    """
    repo = kb_repo or _KB_REPO
    gems: list[dict] = []

    # Determine which runs to search
    runs_dir = repo / "runs"
    if run_tag:
        run_dirs = [runs_dir / run_tag] if (runs_dir / run_tag).exists() else []
    else:
        run_dirs = sorted(runs_dir.glob("run_*"), reverse=True) if runs_dir.exists() else []

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

    # Old structure: ik_bench/gems/<kernel>/<arch>/<slug>/v*/
    if not run_tag:
        old_gems = repo / "ik_bench" / "gems" / kernel / arch / f"gen-{impl_name}"
        if old_gems.exists():
            for ver_dir in sorted(old_gems.glob("v*"), reverse=True):
                gen_path = ver_dir / "data" / "gen" / arch / kernel
                gems.append({
                    "version": ver_dir.name.split("_")[0],
                    "path": ver_dir,
                    "gen_path": gen_path if gen_path.exists() else None,
                    "run": "ik_bench (legacy)",
                    "timestamp": "_".join(ver_dir.name.split("_")[1:]),
                })

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
    import shutil
    repo = kb_repo or _KB_REPO

    if not run_tag:
        run_tag = f"run_{_detect_host_slug()}"

    run_dir = repo / "runs" / run_tag
    gen_path = run_dir / "gen" / arch / kernel

    # Clear existing
    if gen_path.exists():
        shutil.rmtree(gen_path)

    # Find seed source
    if gem_path is None:
        gem_path = _find_latest_gem(kernel, arch, kb_repo=repo)

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
                     kb_repo: Path | None = None) -> Path | None:
    """Find the latest gem's gen code across all KB runs.

    Searches new structure (runs/run_*/gems/) first, then old
    structure (ik_bench/gems/) for backward compatibility.
    """
    repo = kb_repo or _KB_REPO

    # New structure: runs/run_*/gems/<slug>/v*/gen/<arch>/<kernel>
    runs_dir = repo / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
            gem_base = run_dir / "gems" / f"gen-{impl_name}"
            if not gem_base.exists():
                continue
            versions = sorted(gem_base.glob("v*"), reverse=True)
            for ver in versions:
                candidate = ver / "gen" / arch / kernel
                if candidate.exists():
                    return candidate

    # Old structure: ik_bench/gems/<kernel>/<arch>/<slug>/v*/data/gen/<arch>/<kernel>
    old_gems = repo / "ik_bench" / "gems" / kernel / arch / f"gen-{impl_name}"
    if old_gems.exists():
        versions = sorted(old_gems.glob("v*"), reverse=True)
        for ver in versions:
            candidate = ver / "data" / "gen" / arch / kernel
            if candidate.exists():
                return candidate

    return None


def _ensure_gen_dir(kernel: str, arch: str, *,
                    run_tag: str | None = None,
                    kb_repo: Path | None = None) -> Path:
    """Get or create the gen/ scratch dir in the active KB run.

    If the gen dir doesn't exist yet, seed it from the latest gem.
    If no gem exists, creates an empty directory.

    Args:
        kernel: kernel name (matmul, fa4, etc.)
        arch: target arch (sm90, sm120, etc.)
        run_tag: run folder name (e.g. run_h8_3). If None, auto-detects from host config.
        kb_repo: path to kernel_lab_kb repo.

    Returns:
        Path to gen/<arch>/<kernel>/ in the active run.
    """
    import shutil
    repo = kb_repo or _KB_REPO
    runs_dir = repo / "runs"

    # Find or create the active run
    if run_tag:
        run_dir = runs_dir / run_tag
    else:
        # Auto-detect host slug from conf/hosts/default.yaml
        host_slug = _detect_host_slug()
        run_tag = f"run_{host_slug}"
        run_dir = runs_dir / run_tag

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    gen_path = run_dir / "gen" / arch / kernel
    if gen_path.exists():
        return gen_path

    # Seed from latest gem in kernel_lab_kb
    gem_src = _find_latest_gem(kernel, arch, kb_repo=repo)
    if gem_src:
        shutil.copytree(gem_src, gen_path)
        # Fix flat structure from old gems: if cuda.cu is at top level,
        # move into cuda/ subdirectory for list_impls compatibility
        for cu_file in list(gen_path.glob("*.cu")):
            stem = cu_file.stem  # e.g. "cuda"
            subdir = gen_path / stem
            if not subdir.exists():
                subdir.mkdir()
            (subdir / cu_file.name).replace(cu_file)
            # Also move companion files (.baseline, .md, etc.)
            for companion in gen_path.glob(f"{stem}.*"):
                if companion.is_file():
                    (subdir / companion.name).replace(companion)
        return gen_path

    # Nothing to seed from — return empty dir
    gen_path.mkdir(parents=True, exist_ok=True)
    return gen_path


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

    1. Explicit data_root (from bench run snapshot) — use directly
    2. Default: active KB run's gen/ folder, seeded from latest gem if needed
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
        ref-cublas  → data/ref/{kernel}/cublas/
        gen-cuda    → data/gen/{arch}/{kernel}/cuda/
        gen-cutedsl → data/gen/{arch}/{kernel}/cutedsl/

    Entry point: {name}.py or {name}.cu inside the subdir.
    Helpers: all other files in the same subdir.

    Returns:
        {
            "slug": "ref-cublas",
            "source": "ref",
            "name": "cublas",
            "entry_point": Path(...),
            "file_type": "py",
            "files": {"cublas.py": "..."},
        }
    """
    parts = impl_slug.split("-", 1)
    if len(parts) != 2 or parts[0] not in ("ref", "gen"):
        raise ValueError(
            f"Invalid impl slug '{impl_slug}'. "
            f"Format: '{{ref|gen}}-{{name}}' (e.g., 'ref-cublas', 'gen-cuda')"
        )

    source, name = parts

    if source == "ref":
        impl_dir = _ref_dir(kernel, data_root) / name
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
        {"slug": "ref-cublas", "source": "ref", "name": "cublas",
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
    _scan_dir(_gen_dir(kernel, arch, data_root), "gen")

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
