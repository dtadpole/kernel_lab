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


def _ref_dir(kernel: str, data_root: Path | None = None) -> Path:
    return (data_root or _DEFAULT_DATA_ROOT) / "ref" / kernel


def _gen_dir(kernel: str, arch: str, data_root: Path | None = None) -> Path:
    return (data_root or _DEFAULT_DATA_ROOT) / "gen" / arch / kernel


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
