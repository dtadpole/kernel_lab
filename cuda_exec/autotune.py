"""Lightweight autotune for CUDA kernels via #define macro parameterization.

Workflow:
  1. Solver writes kernel.cu with ``#ifndef BM / #define BM 128 / #endif``
  2. Solver writes autotune.yaml next to it with search space + constraints
  3. formal.py detects autotune.yaml → calls ``run_autotune()``
  4. This module compiles all valid parameter combos in parallel,
     benchmarks each variant, and returns the best config.

autotune.yaml format::

    params:
      BM: [64, 128, 256]
      BN: [64, 128, 256]
      BK: [32, 64]
      STAGES: [2, 3, 4]
    constraints:
      - "(BM * BK + BK * BN) * STAGES * 2 <= 227328"
    # Optional: benchmark configs to tune on (default: all)
    bench_configs:
      - mat-4096x4096
      - mat-8192x8192
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
    max_workers: int = 8,
) -> list[CompileResult]:
    """Compile all variants in parallel."""
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

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

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
                capture_output=True, text=True, timeout=60, env=env,
            )
            if result.returncode != 0:
                results[config_slug] = None
                continue

            # Parse JSON output for median latency
            stdout = result.stdout.strip()
            if stdout:
                data = json.loads(stdout)
                perf = data.get("performance", data.get("summary", {}))
                median = perf.get("latency_ms", {}).get("median")
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

    for cr in ok_variants:
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
class AutotuneResult:
    """Result of an autotune run."""
    best_combo: dict[str, int]
    best_tag: str
    best_median_ms: float
    best_latencies: dict[str, float]  # per-config latencies for the winner
    best_registers: int
    best_smem_bytes: int
    total_combos: int
    valid_combos: int
    compiled_ok: int
    benchmarked_ok: int
    all_results: list[dict]  # sorted by performance
    duration_s: float
    defines_flags: str  # e.g. "-DBM=128 -DBN=256"


def load_autotune_yaml(yaml_path: Path) -> dict:
    """Load and validate autotune.yaml."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"autotune.yaml must be a YAML mapping, got {type(data)}")
    if "params" not in data:
        raise ValueError("autotune.yaml missing 'params' section")
    params = data["params"]
    if not isinstance(params, dict) or not params:
        raise ValueError("autotune.yaml 'params' must be a non-empty mapping")
    for key, values in params.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"autotune.yaml params[{key!r}] must be a non-empty list")
        params[key] = [int(v) for v in values]
    data["constraints"] = [str(c) for c in data.get("constraints", [])]
    return data


def run_autotune(
    cu_path: Path,
    autotune_yaml: Path,
    configs: dict[str, dict],
    arch: str,
    env_base: dict[str, str],
    *,
    max_compile_workers: int = 8,
    bench_warmups: int = 2,
    bench_trials: int = 3,
    bench_configs: list[str] | None = None,
) -> AutotuneResult:
    """Run the full autotune pipeline.

    Args:
        cu_path: Path to the CUDA source file with #ifndef'd parameters.
        autotune_yaml: Path to autotune.yaml with search space.
        configs: All benchmark configs {slug: config_dict}.
        arch: GPU architecture (e.g. 'sm_90a').
        env_base: Base environment for compilation and benchmarking.
        max_compile_workers: Max parallel compilations.
        bench_warmups: Warmup iterations per variant.
        bench_trials: Timed iterations per variant.
        bench_configs: Subset of config slugs to tune on (None = all).

    Returns:
        AutotuneResult with the best configuration found.
    """
    started = time.perf_counter()

    # 1. Parse autotune.yaml
    spec = load_autotune_yaml(autotune_yaml)
    params = spec["params"]
    constraints = spec["constraints"]

    # Total combos before filtering
    total_combos = 1
    for values in params.values():
        total_combos *= len(values)

    # 2. Generate valid combos
    combos = generate_combos(params, constraints)
    logger.info("Autotune: %d total combos → %d valid (after constraints)",
                total_combos, len(combos))

    if not combos:
        raise ValueError("No valid parameter combinations after applying constraints")

    # 3. Parallel compile
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
    failed_compiles = [r for r in compile_results if not r.ok]
    logger.info("Autotune: compiled %d ok, %d failed (%.1fs)",
                len(ok_compiles), len(failed_compiles), compile_duration)

    if not ok_compiles:
        raise RuntimeError(
            f"All {len(compile_results)} variants failed to compile. "
            f"First error: {compile_results[0].error if compile_results else 'unknown'}"
        )

    # 4. Filter benchmark configs
    if bench_configs:
        bench_cfg = {k: v for k, v in configs.items() if k in bench_configs}
    else:
        bench_cfg = configs

    if not bench_cfg:
        raise ValueError(f"No matching bench configs. Available: {list(configs.keys())}")

    # 5. Sequential benchmark
    logger.info("Autotune: benchmarking %d variants on %d configs",
                len(ok_compiles), len(bench_cfg))

    bench_started = time.perf_counter()
    bench_results = bench_variants(
        compile_results, bench_cfg, env_base,
        num_warmups=bench_warmups, num_trials=bench_trials,
    )
    bench_duration = time.perf_counter() - bench_started
    logger.info("Autotune: benchmarked in %.1fs", bench_duration)

    # 6. Rank by geometric mean latency
    bench_results.sort(key=lambda r: r.median_ms)
    ok_bench = [r for r in bench_results if r.ok]

    if not ok_bench:
        raise RuntimeError("All variants failed benchmarking")

    best = ok_bench[0]

    # Find compile result for best combo to get register/smem info
    best_compile = next(
        (cr for cr in compile_results if cr.tag == best.tag), None
    )

    # Build sorted results list for reporting
    all_results = []
    for br in bench_results:
        cr = next((c for c in compile_results if c.tag == br.tag), None)
        all_results.append({
            "tag": br.tag,
            "combo": br.combo,
            "median_ms": round(br.median_ms, 4) if br.ok else None,
            "latencies": {k: round(v, 4) if v else None for k, v in br.all_latencies.items()},
            "registers": cr.registers if cr else 0,
            "smem_bytes": cr.smem_bytes if cr else 0,
            "ok": br.ok,
        })

    # Build defines flags string
    defines_flags = " ".join(f"-D{k}={v}" for k, v in best.combo.items())

    total_duration = time.perf_counter() - started

    # Clean up work dir
    try:
        shutil.rmtree(work_dir)
    except OSError:
        pass

    result = AutotuneResult(
        best_combo=best.combo,
        best_tag=best.tag,
        best_median_ms=best.median_ms,
        best_latencies=best.all_latencies,
        best_registers=best_compile.registers if best_compile else 0,
        best_smem_bytes=best_compile.smem_bytes if best_compile else 0,
        total_combos=total_combos,
        valid_combos=len(combos),
        compiled_ok=len(ok_compiles),
        benchmarked_ok=len(ok_bench),
        all_results=all_results,
        duration_s=total_duration,
        defines_flags=defines_flags,
    )

    logger.info(
        "Autotune: best=%s (%.4f ms geo-mean, regs=%d, smem=%dB) — "
        "%d/%d combos, %.1fs total",
        best.tag, best.median_ms,
        result.best_registers, result.best_smem_bytes,
        len(ok_bench), total_combos, total_duration,
    )

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

    # Show top 5 results
    top_n = min(5, len(result.all_results))
    for i, r in enumerate(result.all_results[:top_n]):
        marker = "★" if i == 0 else " "
        ms_str = f"{r['median_ms']:.4f}" if r['median_ms'] else "FAIL"
        reg_str = f"regs={r['registers']}" if r['registers'] else ""
        smem_str = f"smem={r['smem_bytes']}" if r['smem_bytes'] else ""
        hw_info = f"({reg_str}, {smem_str})" if reg_str or smem_str else ""
        combo_str = ", ".join(f"{k}={v}" for k, v in r["combo"].items())
        lines.append(f"  {marker} #{i+1}: {combo_str} → {ms_str} ms {hw_info}")

        # Show per-config latencies for the winner
        if i == 0 and r.get("latencies"):
            for cfg, lat in r["latencies"].items():
                lat_str = f"{lat:.4f} ms" if lat else "FAIL"
                lines.append(f"       {cfg}: {lat_str}")

    lines.append("")
    lines.append(f"Winner: {result.defines_flags}")
    return "\n".join(lines)
