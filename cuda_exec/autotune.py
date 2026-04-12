"""Per-config autotune for CUDA kernels via #define macro parameterization.

Each benchmark config can have its own autotune search space.  Configs with
an ``autotune:`` section get their own best parameters and compiled binary;
configs without one use the kernel's default #define values.

Workflow:
  1. Solver writes kernel.cu with ``#ifndef BM / #define BM 128 / #endif``
  2. Solver writes autotune.yaml next to it with per-config search spaces
  3. formal.py detects autotune.yaml → calls ``run_autotune()``
  4. This module compiles all valid combos (union across configs) in parallel,
     benchmarks each on the relevant configs, and returns per-config winners.

autotune.yaml format::

    configs:
      mat-256x256:
        autotune:
          params:
            BM: [32, 64, 128]
            BN: [32, 64, 128]
            BK: [32, 64]
            STAGES: [2, 3, 4]
          constraints:
            - "(BM * BK + BK * BN) * STAGES * 2 <= 227328"
      mat-512x512:
        autotune:
          params:
            BM: [64, 128]
            BN: [64, 128]
      # mat-8192x8192 not listed → no autotune, use default #define values
"""

from __future__ import annotations

import ast
import itertools
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
COMPILE_SCRIPT = SCRIPTS_DIR / "compile.sh"
EVAL_HARNESS = SCRIPTS_DIR / "eval_harness.cu"


# ---------------------------------------------------------------------------
# Safe constraint evaluator — no eval(), only arithmetic + comparisons
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
    ast.LtE: lambda a, b: a <= b,
    ast.GtE: lambda a, b: a >= b,
    ast.Lt: lambda a, b: a < b,
    ast.Gt: lambda a, b: a > b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.USub: lambda a: -a,
}


def _safe_eval(expr: str, variables: dict[str, int]) -> Any:
    """Evaluate a simple arithmetic + comparison expression safely.

    Only supports: integers, variable names, +, -, *, //, %, **, <=, >=, <, >, ==, !=.
    No function calls, no imports, no attribute access.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid constraint expression: {expr!r}") from e

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable {node.id!r} in constraint. "
                                 f"Available: {list(variables.keys())}")
            return variables[node.id]
        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator {type(node.op).__name__}")
            return op_fn(_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
            return op_fn(_eval_node(node.operand))
        if isinstance(node, ast.Compare):
            left = _eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                op_fn = _SAFE_OPS.get(type(op))
                if op_fn is None:
                    raise ValueError(f"Unsupported comparison {type(op).__name__}")
                right = _eval_node(comparator)
                if not op_fn(left, right):
                    return False
                left = right
            return True
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    return _eval_node(tree)


def check_constraint(expr: str, combo: dict[str, int]) -> bool:
    """Return True if the constraint expression is satisfied by combo."""
    return bool(_safe_eval(expr, combo))


# ---------------------------------------------------------------------------
# Parameter combination generation
# ---------------------------------------------------------------------------

def generate_combos(
    params: dict[str, list[int]],
    constraints: list[str],
) -> list[dict[str, int]]:
    """Generate all valid parameter combinations that satisfy constraints."""
    keys = list(params.keys())
    values = [params[k] for k in keys]

    valid = []
    for combo_values in itertools.product(*values):
        combo = dict(zip(keys, combo_values))
        if all(check_constraint(c, combo) for c in constraints):
            valid.append(combo)
    return valid


def combo_tag(combo: dict[str, int]) -> str:
    """Short tag for a parameter combo, e.g. 'BM128_BN256_BK64_S3'."""
    parts = []
    for k, v in combo.items():
        # Abbreviate common names
        short = k.replace("STAGES", "S").replace("NUM_", "")
        parts.append(f"{short}{v}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    combo: dict[str, int]
    tag: str
    binary_path: str
    ok: bool
    registers: int = 0
    smem_bytes: int = 0
    duration_s: float = 0.0
    error: str = ""


def _compile_variant(
    cu_path: str,
    output_dir: str,
    tag: str,
    defines: dict[str, int],
    arch: str,
    env_base: dict[str, str],
) -> CompileResult:
    """Compile a single variant with -D flags. Runs in a subprocess worker."""
    binary_path = os.path.join(output_dir, tag, f"{tag}.bin")
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)

    # Build -D flags
    define_flags = " ".join(f"-D{k}={v}" for k, v in defines.items())

    env = dict(env_base)
    # Append to any existing NVCC_EXTRA_FLAGS
    existing = env.get("NVCC_EXTRA_FLAGS", "")
    env["NVCC_EXTRA_FLAGS"] = f"{existing} {define_flags}".strip()

    harness_args = []
    try:
        source_text = open(cu_path, "r").read()
        if "kernel_run" in source_text and EVAL_HARNESS.exists():
            harness_args = ["--harness", str(EVAL_HARNESS)]
    except OSError:
        pass

    cmd = [
        "/usr/bin/env", "bash", str(COMPILE_SCRIPT),
        "--source", cu_path,
        "--output", binary_path,
        "--arch", arch,
        *harness_args,
    ]

    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env,
        )
        duration = time.perf_counter() - started

        if result.returncode != 0:
            return CompileResult(
                combo=defines, tag=tag, binary_path=binary_path,
                ok=False, duration_s=duration,
                error=result.stderr[-500:] if result.stderr else f"exit {result.returncode}",
            )

        # Parse ptxas output for register and smem usage
        registers = 0
        smem_bytes = 0
        ptxas_log = os.path.join(output_dir, tag, "logs")
        # ptxas output is in stderr from compile.sh step 2
        stderr = result.stderr or ""
        for line in stderr.split("\n"):
            if "Used" in line and "registers" in line:
                import re
                reg_match = re.search(r"Used (\d+) registers", line)
                if reg_match:
                    registers = int(reg_match.group(1))
                smem_match = re.search(r"(\d+) bytes smem", line)
                if smem_match:
                    smem_bytes = int(smem_match.group(1))

        return CompileResult(
            combo=defines, tag=tag, binary_path=binary_path,
            ok=True, registers=registers, smem_bytes=smem_bytes,
            duration_s=duration,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(
            combo=defines, tag=tag, binary_path=binary_path,
            ok=False, duration_s=120.0, error="compile timeout (120s)",
        )
    except Exception as e:
        return CompileResult(
            combo=defines, tag=tag, binary_path=binary_path,
            ok=False, duration_s=time.perf_counter() - started,
            error=str(e),
        )


def compile_variants(
    cu_path: Path,
    combos: list[dict[str, int]],
    output_dir: Path,
    arch: str,
    env_base: dict[str, str],
    max_workers: int = 16,
    timeout_per_variant: int = 180,
) -> list[CompileResult]:
    """Compile all variants in parallel.

    Each worker compiles independently with its own env copy.
    Worker crashes are caught and recorded as failed CompileResults.
    """
    results = []
    tags = [combo_tag(c) for c in combos]

    with ProcessPoolExecutor(max_workers=min(max_workers, len(combos))) as pool:
        futures = {}
        for combo, tag in zip(combos, tags):
            future = pool.submit(
                _compile_variant,
                str(cu_path), str(output_dir), tag, combo, arch, env_base,
            )
            futures[future] = (combo, tag)

        for future in as_completed(futures, timeout=timeout_per_variant * len(combos)):
            combo, tag = futures[future]
            try:
                results.append(future.result(timeout=timeout_per_variant))
            except Exception as exc:
                logger.warning("compile worker crashed for %s: %s", tag, exc)
                results.append(CompileResult(
                    combo=combo, tag=tag,
                    binary_path=str(output_dir / tag / f"{tag}.bin"),
                    ok=False, error=f"worker crash: {exc}",
                ))

    return results


# ---------------------------------------------------------------------------
# Quick benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    tag: str
    combo: dict[str, int]
    median_ms: float
    all_latencies: dict[str, float] = field(default_factory=dict)  # config_slug → median_ms
    ok: bool = True
    error: str = ""


def _quick_bench_variant(
    binary_path: str,
    configs: dict[str, dict],
    env_base: dict[str, str],
    num_warmups: int = 2,
    num_trials: int = 3,
) -> dict[str, float | None]:
    """Run a compiled binary on each config and return {config_slug: median_ms}.

    Uses the eval_harness env-based config interface.
    """
    results: dict[str, float | None] = {}

    for config_slug, config in configs.items():
        env = dict(env_base)
        # Set config env vars matching eval_harness expectations
        env["CUDA_EXEC_CONFIG_ID"] = config_slug
        env["CUDA_EXEC_CONFIG_JSON"] = json.dumps({"slug": config_slug, "params": config})
        env["CUDA_EXEC_PARAM_INPUT_SIZE"] = str(config.get("input_size", 1048576))
        env["CUDA_EXEC_PARAM_RANK"] = str(config.get("rank", 2))
        env["CUDA_EXEC_PARAM_SHAPE_KIND"] = config.get("shape_kind", "2d")
        env["CUDA_EXEC_PARAM_SHAPE"] = json.dumps(config.get("shape", [1024, 1024]))
        env["CUDA_EXEC_NUM_WARMUPS"] = str(num_warmups)
        env["CUDA_EXEC_NUM_TRIALS"] = str(num_trials)

        try:
            result = subprocess.run(
                [binary_path],
                capture_output=True, text=True, timeout=120, env=env,
            )
            if result.returncode != 0:
                results[config_slug] = None
                continue

            # Parse JSON output for median latency
            stdout = result.stdout.strip()
            if stdout:
                data = json.loads(stdout)
                perf = data.get("performance", data.get("summary", {}))
                lat = perf.get("latency_ms", {})
                median = lat.get("p50") or lat.get("median")
                results[config_slug] = median
            else:
                results[config_slug] = None
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            results[config_slug] = None

    return results


def bench_variants(
    compile_results: list[CompileResult],
    configs: dict[str, dict],
    env_base: dict[str, str],
    num_warmups: int = 2,
    num_trials: int = 3,
) -> list[BenchResult]:
    """Benchmark all successfully compiled variants sequentially.

    Sequential because GPU benchmarks need exclusive access for stable results.
    """
    results = []
    ok_variants = [r for r in compile_results if r.ok]

    for idx, cr in enumerate(ok_variants):
        logger.info("    %d/%d: %s", idx + 1, len(ok_variants), cr.tag)
        latencies = _quick_bench_variant(
            cr.binary_path, configs, env_base,
            num_warmups=num_warmups, num_trials=num_trials,
        )

        # Compute geometric mean across configs for ranking
        valid = [v for v in latencies.values() if v is not None and v > 0]
        if valid:
            from math import exp, log
            geo_mean = exp(sum(log(v) for v in valid) / len(valid))
            results.append(BenchResult(
                tag=cr.tag, combo=cr.combo,
                median_ms=geo_mean, all_latencies=latencies, ok=True,
            ))
        else:
            results.append(BenchResult(
                tag=cr.tag, combo=cr.combo,
                median_ms=float("inf"), all_latencies=latencies,
                ok=False, error="all configs failed",
            ))

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class PerConfigWinner:
    """Best autotune result for a single config."""
    config_slug: str
    best_combo: dict[str, int]
    best_tag: str
    best_median_ms: float
    best_registers: int
    best_smem_bytes: int
    defines_flags: str  # e.g. "-DBM=128 -DBN=256"
    binary_path: str = ""  # path to compiled binary (reusable, no need to recompile)


@dataclass
class AutotuneResult:
    """Result of an autotune run with per-config winners."""
    per_config_results: dict[str, PerConfigWinner]  # config_slug → winner
    total_combos: int       # raw cartesian product size (sum across configs)
    valid_combos: int       # after constraint filtering (union)
    compiled_ok: int
    benchmarked_ok: int
    all_results: list[dict]  # sorted by geo-mean performance
    duration_s: float
    configs_without_autotune: list[str]  # configs that had no autotune section


def load_autotune_yaml(yaml_path: Path) -> dict:
    """Load and validate autotune.yaml.

    Expected format::

        configs:
          mat-256x256:
            autotune:
              params:
                BM: [32, 64, 128]
              constraints:
                - "..."
          mat-8192x8192: {}   # no autotune for this config

    Returns the parsed dict with validated params converted to int lists.
    """
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"autotune.yaml must be a YAML mapping, got {type(data)}")
    if "configs" not in data:
        raise ValueError("autotune.yaml missing 'configs' section")
    configs_section = data["configs"]
    if not isinstance(configs_section, dict) or not configs_section:
        raise ValueError("autotune.yaml 'configs' must be a non-empty mapping")

    for config_slug, config_spec in configs_section.items():
        if config_spec is None:
            configs_section[config_slug] = {}
            continue
        if not isinstance(config_spec, dict):
            raise ValueError(f"configs[{config_slug!r}] must be a mapping")
        if "autotune" not in config_spec:
            continue
        at = config_spec["autotune"]
        if not isinstance(at, dict):
            raise ValueError(f"configs[{config_slug!r}].autotune must be a mapping")
        if "params" not in at:
            raise ValueError(f"configs[{config_slug!r}].autotune missing 'params'")
        params = at["params"]
        if not isinstance(params, dict) or not params:
            raise ValueError(f"configs[{config_slug!r}].autotune.params must be a non-empty mapping")
        for key, values in params.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"configs[{config_slug!r}].autotune.params[{key!r}] must be a non-empty list"
                )
            params[key] = [int(v) for v in values]
        at["constraints"] = [str(c) for c in at.get("constraints", [])]

    return data


# ---------------------------------------------------------------------------
# Per-config helpers
# ---------------------------------------------------------------------------

def _compute_per_config_valid_combos(
    config_autotunes: dict[str, dict],
) -> tuple[list[dict[str, int]], dict[str, set[str]]]:
    """Compute the UNION of all per-config valid combos for shared compilation.

    Args:
        config_autotunes: {config_slug: {"params": {...}, "constraints": [...]}}

    Returns:
        union_combos: deduplicated list of all combos that any config needs
        config_combo_tags: {config_slug: set of combo tags valid for that config}
    """
    all_combos: dict[str, dict[str, int]] = {}  # tag -> combo (dedup)
    config_combo_tags: dict[str, set[str]] = {}

    for config_slug, at_spec in config_autotunes.items():
        params = at_spec["params"]
        constraints = at_spec.get("constraints", [])
        combos = generate_combos(params, constraints)
        tags = set()
        for combo in combos:
            tag = combo_tag(combo)
            all_combos[tag] = combo
            tags.add(tag)
        config_combo_tags[config_slug] = tags

    return list(all_combos.values()), config_combo_tags


def _select_per_config_winners(
    bench_results: list[BenchResult],
    compile_results: list[CompileResult],
    config_combo_tags: dict[str, set[str]],
) -> dict[str, PerConfigWinner]:
    """Select the best combo for each config from its valid search space.

    For each config, filters bench_results to only combos in that config's
    search space, then picks the one with the lowest latency for that config.
    """
    compile_by_tag = {cr.tag: cr for cr in compile_results}

    winners = {}
    for config_slug, valid_tags in config_combo_tags.items():
        best_ms = float("inf")
        best_br: BenchResult | None = None

        for br in bench_results:
            if not br.ok or br.tag not in valid_tags:
                continue
            config_lat = br.all_latencies.get(config_slug)
            if config_lat is not None and config_lat < best_ms:
                best_ms = config_lat
                best_br = br

        if best_br is None:
            # Fallback: first valid & ok result
            for br in bench_results:
                if br.ok and br.tag in valid_tags:
                    best_br = br
                    lat = br.all_latencies.get(config_slug)
                    best_ms = lat if lat is not None else br.median_ms
                    break

        if best_br is None:
            continue

        cr = compile_by_tag.get(best_br.tag)
        winners[config_slug] = PerConfigWinner(
            config_slug=config_slug,
            best_combo=best_br.combo,
            best_tag=best_br.tag,
            best_median_ms=best_ms,
            best_registers=cr.registers if cr else 0,
            best_smem_bytes=cr.smem_bytes if cr else 0,
            defines_flags=" ".join(f"-D{k}={v}" for k, v in best_br.combo.items()),
        )

    return winners


def run_autotune(
    cu_path: Path,
    autotune_yaml: Path,
    configs: dict[str, dict],
    arch: str,
    env_base: dict[str, str],
    *,
    max_compile_workers: int = 16,
    bench_warmups: int = 2,
    bench_trials: int = 3,
    output_dir: Path | None = None,
) -> AutotuneResult:
    """Run per-config autotune pipeline.

    Only configs with an ``autotune:`` section in the YAML are tuned.
    Configs without one are listed in ``configs_without_autotune``.

    Args:
        cu_path: Path to the CUDA source file with #ifndef'd parameters.
        autotune_yaml: Path to autotune.yaml with per-config search spaces.
        configs: All benchmark configs {slug: config_dict}.
        arch: GPU architecture (e.g. 'sm_90a').
        env_base: Base environment for compilation and benchmarking.
        max_compile_workers: Max parallel compilations.
        bench_warmups: Warmup iterations per variant.
        bench_trials: Timed iterations per variant.
        output_dir: Directory for compiled binaries. If None, uses a temp dir.
            When provided, binaries are KEPT (not deleted) so formal.py can
            reuse them without recompiling.

    Returns:
        AutotuneResult with per-config winners (including binary_path).
    """
    started = time.perf_counter()

    # 1. Parse autotune.yaml — extract configs with autotune sections
    spec = load_autotune_yaml(autotune_yaml)
    configs_section = spec["configs"]

    config_autotunes: dict[str, dict] = {}  # config_slug → autotune spec
    configs_without_autotune: list[str] = []

    for config_slug in configs:
        config_spec = configs_section.get(config_slug, {})
        if "autotune" in config_spec:
            config_autotunes[config_slug] = config_spec["autotune"]
        else:
            configs_without_autotune.append(config_slug)

    if not config_autotunes:
        raise ValueError("No configs with autotune settings in autotune.yaml")

    logger.info("Autotune: %d configs to tune, %d without autotune",
                len(config_autotunes), len(configs_without_autotune))

    # 2. Generate combos per config, compute union for shared compilation
    combos, config_combo_tags = _compute_per_config_valid_combos(config_autotunes)

    total_combos = 0
    for at_spec in config_autotunes.values():
        n = 1
        for values in at_spec["params"].values():
            n *= len(values)
        total_combos += n

    logger.info("Autotune: %d unique combos (total raw: %d)", len(combos), total_combos)
    for slug, tags in config_combo_tags.items():
        logger.info("  %s: %d valid combos", slug, len(tags))

    if not combos:
        raise ValueError("No valid parameter combinations after applying constraints")

    # 3. Parallel compile
    if output_dir:
        work_dir = Path(output_dir)
        # Clean previous autotune results to avoid stale binaries
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="autotune_"))
    logger.info("Autotune: compiling %d variants (max %d workers) in %s",
                len(combos), max_compile_workers, work_dir)

    compile_started = time.perf_counter()
    compile_results = compile_variants(
        cu_path, combos, work_dir, arch, env_base,
        max_workers=max_compile_workers,
    )
    compile_duration = time.perf_counter() - compile_started

    ok_compiles = [r for r in compile_results if r.ok]
    logger.info("Autotune: compiled %d ok, %d failed (%.1fs)",
                len(ok_compiles), len(compile_results) - len(ok_compiles),
                compile_duration)

    if not ok_compiles:
        raise RuntimeError(
            f"All {len(compile_results)} variants failed to compile. "
            f"First error: {compile_results[0].error if compile_results else 'unknown'}"
        )

    # 4. Benchmark per config — each config only benchmarks its own combos
    compile_by_tag = {cr.tag: cr for cr in compile_results if cr.ok}
    per_config_results: dict[str, PerConfigWinner] = {}
    all_results: list[dict] = []
    total_benched = 0

    bench_started = time.perf_counter()
    for config_slug, valid_tags in config_combo_tags.items():
        # Filter to only this config's valid compiled combos
        config_compiles = [compile_by_tag[t] for t in valid_tags if t in compile_by_tag]
        if not config_compiles:
            logger.warning("  %s: no compiled combos, skipping", config_slug)
            continue

        single_cfg = {config_slug: configs[config_slug]}
        logger.info("  %s: benchmarking %d combos", config_slug, len(config_compiles))

        # Benchmark this config's combos on this single config
        cfg_bench = bench_variants(
            config_compiles, single_cfg, env_base,
            num_warmups=bench_warmups, num_trials=bench_trials,
        )
        total_benched += len([b for b in cfg_bench if b.ok])

        # Pick best for this config
        cfg_bench.sort(key=lambda r: r.median_ms)
        ok_bench = [b for b in cfg_bench if b.ok]
        if ok_bench:
            best = ok_bench[0]
            lat = best.all_latencies.get(config_slug)
            best_ms = lat if lat is not None else best.median_ms
            cr = compile_by_tag.get(best.tag)
            per_config_results[config_slug] = PerConfigWinner(
                config_slug=config_slug,
                best_combo=best.combo,
                best_tag=best.tag,
                best_median_ms=best_ms,
                best_registers=cr.registers if cr else 0,
                best_smem_bytes=cr.smem_bytes if cr else 0,
                defines_flags=" ".join(f"-D{k}={v}" for k, v in best.combo.items()),
                binary_path=cr.binary_path if cr else "",
            )
            logger.info("  %s: best=%s (%.4f ms)", config_slug, best.tag, best_ms)

        # Collect for reporting
        for br in cfg_bench:
            cr = compile_by_tag.get(br.tag)
            all_results.append({
                "config": config_slug,
                "tag": br.tag,
                "combo": br.combo,
                "median_ms": round(br.median_ms, 4) if br.ok else None,
                "registers": cr.registers if cr else 0,
                "smem_bytes": cr.smem_bytes if cr else 0,
                "ok": br.ok,
            })

    bench_duration = time.perf_counter() - bench_started
    logger.info("Autotune: benchmarked %d combos across %d configs in %.1fs",
                total_benched, len(config_combo_tags), bench_duration)

    if not per_config_results:
        raise RuntimeError("All variants failed benchmarking on all configs")

    distinct_tags = {pcw.best_tag for pcw in per_config_results.values()}
    logger.info("Autotune: %d distinct winners for %d configs",
                len(distinct_tags), len(per_config_results))

    total_duration = time.perf_counter() - started

    # Clean up work dir only if we created a temp dir (no output_dir provided)
    if not output_dir:
        try:
            shutil.rmtree(work_dir)
        except OSError:
            pass

    result = AutotuneResult(
        per_config_results=per_config_results,
        total_combos=total_combos,
        valid_combos=len(combos),
        compiled_ok=len(ok_compiles),
        benchmarked_ok=total_benched,
        all_results=all_results,
        duration_s=total_duration,
        configs_without_autotune=configs_without_autotune,
    )

    logger.info("Autotune: %.1fs total", total_duration)
    return result


def format_autotune_report(result: AutotuneResult) -> str:
    """Format autotune results as a human-readable report."""
    lines = [
        f"=== Autotune: {result.total_combos} combos → "
        f"{result.valid_combos} valid → "
        f"{result.compiled_ok} compiled → "
        f"{result.benchmarked_ok} benchmarked "
        f"({result.duration_s:.1f}s) ===",
    ]

    # Per-config winners
    lines.append("")
    for config_slug, pcw in result.per_config_results.items():
        combo_str = ", ".join(f"{k}={v}" for k, v in pcw.best_combo.items())
        lines.append(f"  {config_slug}: {combo_str} → {pcw.best_median_ms:.4f} ms")

    # Grouping summary
    groups: dict[str, list[str]] = defaultdict(list)
    for config_slug, pcw in result.per_config_results.items():
        groups[pcw.defines_flags].append(config_slug)
    lines.append("")
    lines.append(f"Distinct combos: {len(groups)}")
    for defines, config_slugs in groups.items():
        lines.append(f"  {defines} → {', '.join(config_slugs)}")

    if result.configs_without_autotune:
        lines.append("")
        lines.append(f"No autotune: {', '.join(result.configs_without_autotune)}")

    return "\n".join(lines)
