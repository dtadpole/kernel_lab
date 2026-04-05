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


def _ref_dir(kernel: str) -> Path:
    return _PROJECT_ROOT / "data" / "ref" / kernel


def _gen_dir(kernel: str, arch: str) -> Path:
    return _PROJECT_ROOT / "data" / "gen" / arch / kernel


def resolve_impl(
    kernel: str,
    arch: str,
    impl_slug: str,
) -> dict:
    """Resolve an implementation slug to its source files and metadata.

    Args:
        kernel: Kernel name (e.g., "matmul", "fa4")
        arch: GPU architecture (e.g., "sm90")
        impl_slug: Implementation slug (e.g., "ref-cublas", "gen-cuda")

    Returns:
        {
            "slug": "ref-cublas",
            "source": "ref",          # "ref" or "gen"
            "name": "cublas",          # file stem
            "entry_point": Path(...),  # resolved entry point file
            "file_type": "py",         # "py" or "cu"
            "files": {"cublas.py": "..."},  # all files to pass (entry + helpers)
        }

    Raises:
        FileNotFoundError: if the slug cannot be resolved
        ValueError: if the slug format is invalid
    """
    parts = impl_slug.split("-", 1)
    if len(parts) != 2 or parts[0] not in ("ref", "gen"):
        raise ValueError(
            f"Invalid impl slug '{impl_slug}'. "
            f"Format: '{{ref|gen}}-{{name}}' (e.g., 'ref-cublas', 'gen-cuda')"
        )

    source, name = parts

    if source == "ref":
        base_dir = _ref_dir(kernel)
    else:
        base_dir = _gen_dir(kernel, arch)

    # Resolve entry point: try .py first, then .cu
    entry_py = base_dir / f"{name}.py"
    entry_cu = base_dir / f"{name}.cu"

    if entry_py.exists():
        entry_point = entry_py
        file_type = "py"
    elif entry_cu.exists():
        entry_point = entry_cu
        file_type = "cu"
    else:
        raise FileNotFoundError(
            f"Cannot resolve impl '{impl_slug}' for kernel '{kernel}' arch '{arch}'. "
            f"Tried: {entry_py}, {entry_cu}"
        )

    # Collect files: entry point + helpers from the same directory
    files: Dict[str, str] = {}
    files[entry_point.name] = entry_point.read_text(encoding="utf-8")

    if file_type == "py":
        # Include other .py files as helpers (e.g., cute_gemm_sm90.py)
        for f in base_dir.iterdir():
            if f.is_file() and f.suffix == ".py" and f.name != entry_point.name:
                files[f.name] = f.read_text(encoding="utf-8")
    else:
        # .cu: include .h/.cuh headers as helpers
        for f in base_dir.iterdir():
            if f.is_file() and f.suffix in (".h", ".cuh"):
                files[f.name] = f.read_text(encoding="utf-8")

    return {
        "slug": impl_slug,
        "source": source,
        "name": name,
        "entry_point": entry_point,
        "file_type": file_type,
        "files": files,
    }


def list_impls(kernel: str, arch: str) -> List[dict]:
    """List all available implementations for a kernel+arch.

    Scans ref/{kernel}/ and gen/{arch}/{kernel}/ for entry point files.
    Entry points: .py or .cu files (excludes helper-only files that are
    imported by other .py files but don't define Model themselves).

    Returns list of:
        {"slug": "ref-cublas", "source": "ref", "name": "cublas",
         "file_type": "py", "path": Path(...)}
    """
    impls = []

    # Scan ref/
    ref_dir = _ref_dir(kernel)
    if ref_dir.exists():
        for f in sorted(ref_dir.iterdir()):
            if f.is_file() and f.suffix in (".py", ".cu"):
                impls.append({
                    "slug": f"ref-{f.stem}",
                    "source": "ref",
                    "name": f.stem,
                    "file_type": f.suffix[1:],
                    "path": f,
                })

    # Scan gen/
    gen_dir = _gen_dir(kernel, arch)
    if gen_dir.exists():
        # Only include files that are entry points (have Model class or kernel_run)
        for f in sorted(gen_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix == ".cu":
                impls.append({
                    "slug": f"gen-{f.stem}",
                    "source": "gen",
                    "name": f.stem,
                    "file_type": "cu",
                    "path": f,
                })
            elif f.suffix == ".py":
                # Check if it's an entry point (has Model class)
                content = f.read_text(encoding="utf-8", errors="ignore")
                if "class Model" in content:
                    impls.append({
                        "slug": f"gen-{f.stem}",
                        "source": "gen",
                        "name": f.stem,
                        "file_type": "py",
                        "path": f,
                    })

    return impls


def resolve_impls(
    kernel: str,
    arch: str,
    impl_slugs: str | List[str],
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
        available = list_impls(kernel, arch)
        slugs = [impl["slug"] for impl in available]
    elif isinstance(impl_slugs, str):
        slugs = [impl_slugs]
    else:
        slugs = list(impl_slugs)

    if not slugs:
        raise ValueError(
            f"No implementations found for kernel '{kernel}' arch '{arch}'"
        )

    resolved = [resolve_impl(kernel, arch, s) for s in slugs]

    # Classify by slug prefix: ref- = reference, gen- = generated
    refs = [r for r in resolved if r["source"] == "ref"]
    gens = [r for r in resolved if r["source"] == "gen"]

    if not refs:
        raise ValueError(
            f"At least one reference (ref-*) implementation is required. "
            f"Got only generated (gen-*): {[g['slug'] for g in gens]}"
        )

    return resolved


def load_configs(kernel: str) -> Dict[str, Dict[str, Any]]:
    """Load ALL configs for a kernel."""
    configs_path = _PROJECT_ROOT / "data" / "configs" / f"{kernel}.json"
    if not configs_path.exists():
        raise FileNotFoundError(
            f"Configs not found: {configs_path}. "
            f"Available: {[p.stem for p in (_PROJECT_ROOT / 'data' / 'configs').glob('*.json')]}"
        )
    import json
    with open(configs_path, encoding="utf-8") as f:
        configs = json.load(f)
    if not configs:
        raise ValueError(f"Empty configs file: {configs_path}")
    return configs
