#!/usr/bin/env python3
"""Impl-keyed trial runner for cuda_exec.

Scans inputs/{slug}/ directories, runs each impl, compares all vs golden
(first ref-*). No hardcoded reference/generated/cudnn slots.

Each impl directory contains:
  - .py impl: {name}.py entry point (has class Model) + optional helpers
  - .cu impl: compiled binary at artifacts/compile.attempt_001.{name}.bin

Output JSON:
  {
    "config_slug": "...",
    "status": "ok",
    "golden_slug": "ref-pytorch",
    "impls": {
      "ref-pytorch": {"performance": {...}, "correctness": null},
      "gen-cutedsl": {"performance": {...}, "correctness": {"passed": true, ...}},
      "gen-cuda": {"performance": {...}, "correctness": {"passed": true, ...}}
    }
  }
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from _cli_common import add_metadata_args, ensure_repo_root_on_path

ensure_repo_root_on_path()

from cuda_exec.models import Metadata  # noqa: E402
from cuda_exec.runner import resolve_workspace_bundle  # noqa: E402
from cuda_exec.tasks import _config_env, _primary_artifact_from_manifest, _slugify  # noqa: E402
from cuda_exec.scripts.eval_support import (  # noqa: E402
    DEFAULT_SEED,
    NUM_CORRECTNESS_TRIALS,
    NUM_WARMUP_RUNS,
    NUM_PERF_TRIALS,
    set_seed,
    acquire_device_lock,
    release_device_lock,
    cleanup_lockfile,
    gpu_cleanup,
    watchdog_handler,
    load_reference_module,
    extract_config_payload,
    generate_inputs,
    measure_reference,
)


# ---------------------------------------------------------------------------
# Run a .cu impl via compiled binary
# ---------------------------------------------------------------------------

def _run_cu_impl(
    workspace: dict,
    impl: dict,
    env: dict[str, str],
    workspace_path: str,
    timeout_seconds: int,
    binary_path: str | None = None,
) -> dict:
    """Run a compiled .cu impl binary and return performance payload."""
    if binary_path:
        target_path = Path(binary_path)
    else:
        target_path, _ = _primary_artifact_from_manifest(workspace)

    output_dir = Path(workspace_path) / f"_output_{impl['slug']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_env = {**os.environ, **env, "CUDA_EXEC_OUTPUT_DIR": str(output_dir)}

    completed = subprocess.run(
        [str(target_path)],
        cwd=workspace_path,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return {"error": "empty stdout", "returncode": completed.returncode,
                "stderr": completed.stderr}

    payload = json.loads(stdout)

    # Read binary output for correctness comparison
    output_bin = output_dir / "output_0.bin"
    output_tensor = None
    if output_bin.exists():
        raw = output_bin.read_bytes()
        output_tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()

    return {
        "performance": payload.get("performance", {}),
        "output_tensor": output_tensor,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


# ---------------------------------------------------------------------------
# Correctness comparison
# ---------------------------------------------------------------------------

def _harness_fill_random_bf16(count: int, seed: int) -> torch.Tensor:
    """Reproduce eval_harness.cu fill_random_bf16 PRNG in Python.

    Must match the C code exactly:
        h = idx ^ seed; h = (h^61)^(h>>16); h += h<<3;
        h ^= h>>4; h *= 0x27d4eb2d; h ^= h>>15;
        val = (float)(h & 0xFFFF) / 65536.0f - 0.5f;
    """
    import numpy as np
    idx = np.arange(count, dtype=np.uint32)
    h = idx ^ np.uint32(seed)
    h = (h ^ np.uint32(61)) ^ (h >> np.uint32(16))
    h = (h + (h << np.uint32(3))) & np.uint32(0xFFFFFFFF)
    h = h ^ (h >> np.uint32(4))
    h = (h * np.uint32(0x27d4eb2d)) & np.uint32(0xFFFFFFFF)
    h = h ^ (h >> np.uint32(15))
    f = (h & np.uint32(0xFFFF)).astype(np.float32) / 65536.0 - 0.5
    return torch.from_numpy(f).to(torch.bfloat16)


def _check_correctness(golden_tensor, impl_tensor, atol=1e-2, rtol=1e-2) -> dict:
    """Compare impl output against golden. Returns correctness summary."""
    if golden_tensor is None or impl_tensor is None:
        return {"passed": None, "reason": "missing output"}
    try:
        g = golden_tensor.float().cpu().flatten()
        t = impl_tensor.float().cpu().flatten()
        if g.shape != t.shape:
            return {"passed": False,
                    "reason": f"shape mismatch: {g.shape} vs {t.shape}"}
        diff = (g - t).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        passed = bool(torch.allclose(g, t, atol=atol, rtol=rtol))
        return {"passed": passed, "max_abs_error": max_diff,
                "mean_abs_error": mean_diff}
    except Exception as exc:
        return {"passed": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Impl-keyed trial runner. Scans inputs/*/, runs each impl, "
                    "compares all vs golden (first ref-*).",
    )
    add_metadata_args(parser)
    parser.add_argument("--config-slug", required=True)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--impls", required=True,
                        help="Comma-separated impl slugs in order. First ref-* is golden. "
                             "Example: ref-pytorch,gen-cutedsl,gen-cuda")
    parser.add_argument("--binary-map", default="",
                        help="Comma-separated slug=path pairs for .cu impl binaries. "
                             "Example: ref-cublas=/path/to/cublas.bin,gen-cuda=/path/to/gen.bin")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-warmups", type=int, default=NUM_WARMUP_RUNS)
    parser.add_argument("--num-perf-trials", type=int, default=NUM_PERF_TRIALS)
    parser.add_argument("--num-correctness-trials", type=int, default=NUM_CORRECTNESS_TRIALS)
    args = parser.parse_args()

    metadata = Metadata(
        run_tag=args.run_tag,
        version=args.version,
        direction_id=args.direction_id,
        direction_slug=args.direction_slug,
        revision=args.revision,
    )
    config = json.loads(args.config_json)
    if not isinstance(config, dict):
        raise SystemExit("--config-json must decode to a JSON object")

    device = torch.device("cuda")
    lock_fd: int | None = None

    # Parse binary map: slug=/path/to/bin,...
    binary_map: dict[str, str] = {}
    if args.binary_map:
        for pair in args.binary_map.split(","):
            if "=" in pair:
                slug, path = pair.split("=", 1)
                binary_map[slug.strip()] = path.strip()

    old_handler = signal.signal(signal.SIGALRM, watchdog_handler)
    signal.alarm(args.timeout)

    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    try:
        lock_fd = acquire_device_lock(device)

        workspace = resolve_workspace_bundle(**metadata.model_dump())
        workspace_path = workspace["workspace_path"]
        config_rel = f"state/trial.inline.{_slugify(args.config_slug)}.json"
        env = _config_env(workspace, "trial", 1, args.config_slug, config, config_rel)
        trial_config = extract_config_payload(env["CUDA_EXEC_CONFIG_JSON"])

        # --- Resolve impls from --impls parameter ---
        impl_slugs = [s.strip() for s in args.impls.split(",") if s.strip()]
        if not impl_slugs:
            raise RuntimeError("--impls must specify at least one impl slug")

        inputs_dir = Path(workspace_path) / "inputs"
        impls = []
        for slug in impl_slugs:
            source = slug.split("-", 1)[0]
            name = slug.split("-", 1)[1] if "-" in slug else slug
            impl_dir = inputs_dir / slug

            # If binary-map provides a path for this slug, it's a .cu impl
            if slug in binary_map:
                impls.append({"slug": slug, "source": source, "name": name,
                              "type": "cu", "dir": impl_dir})
                continue

            # Otherwise, resolve from staged inputs/ directory
            if not impl_dir.is_dir():
                raise RuntimeError(f"Impl directory not found: {impl_dir}")

            cu_files = list(impl_dir.glob("*.cu"))
            py_files = [f for f in impl_dir.glob("*.py")
                        if "class Model" in f.read_text(errors="ignore")]
            if cu_files:
                impls.append({"slug": slug, "source": source, "name": name,
                              "type": "cu", "dir": impl_dir, "entry": cu_files[0]})
            elif py_files:
                impls.append({"slug": slug, "source": source, "name": name,
                              "type": "py", "dir": impl_dir, "entry": py_files[0]})
            else:
                raise RuntimeError(f"No .cu or .py entry point in {impl_dir}")

        # First ref-* is golden
        golden_slug = ""
        for impl in impls:
            if impl["source"] == "ref":
                golden_slug = impl["slug"]
                break
        if not golden_slug:
            golden_slug = impls[0]["slug"]

        print(f"[{_ts()}] discovered {len(impls)} impls, golden={golden_slug}",
              file=sys.stderr)

        # --- Run each impl ---
        impl_results: dict[str, dict] = {}
        golden_tensor = None

        for impl in impls:
            slug = impl["slug"]
            t0 = time.perf_counter()
            print(f"[{_ts()}] {slug} start ({impl['type']})", file=sys.stderr)

            try:
                if impl["type"] == "py":
                    # Load and run .py impl
                    old_path = list(sys.path)
                    sys.path.insert(0, str(impl["dir"]))
                    try:
                        mod = load_reference_module(impl["entry"])
                        result = measure_reference(
                            mod, trial_config, device=device,
                            seed=args.seed,
                            num_warmups=args.num_warmups,
                            num_trials=args.num_perf_trials,
                        )
                        output_tensor = result.get("output_tensor")
                        impl_results[slug] = {
                            "performance": result["performance"],
                            "output_tensor": output_tensor,
                        }
                    finally:
                        sys.path[:] = old_path

                elif impl["type"] == "cu":
                    # Run compiled binary (use binary-map if available)
                    bp = binary_map.get(impl["slug"])
                    r = _run_cu_impl(workspace, impl, env, workspace_path,
                                     args.timeout, binary_path=bp)
                    if "error" in r and "performance" not in r:
                        impl_results[slug] = {"error": r["error"]}
                    else:
                        impl_results[slug] = {
                            "performance": r["performance"],
                            "output_tensor": r.get("output_tensor"),
                        }

            except Exception as exc:
                impl_results[slug] = {"error": str(exc)}

            dt = time.perf_counter() - t0
            print(f"[{_ts()}] {slug} done ({dt:.1f}s)", file=sys.stderr)

            # Capture golden output
            if slug == golden_slug and "output_tensor" in impl_results.get(slug, {}):
                golden_tensor = impl_results[slug]["output_tensor"]

        # --- Correctness: compare all non-golden vs golden ---
        # For .py impls: both used generate_inputs() with same seed → direct compare.
        # For .cu impls: C harness used fill_random_bf16(seed=1,2,...) as inputs.
        #   We must run the golden model with those SAME inputs to get a fair reference.
        golden_impl = next((im for im in impls if im["slug"] == golden_slug), None)
        golden_is_cu = golden_impl and golden_impl["type"] == "cu"

        # Load golden .py module (needed when golden is .py and impl is .cu,
        # OR when golden is .cu and impl is .py — to re-run .py with C inputs)
        golden_module = None
        if golden_impl and golden_impl["type"] == "py":
            old_path = list(sys.path)
            sys.path.insert(0, str(golden_impl["dir"]))
            try:
                golden_module = load_reference_module(golden_impl["entry"])
            finally:
                sys.path[:] = old_path

        # Helper: generate C harness inputs (matching eval_harness.cu's fill_random_bf16)
        def _make_harness_inputs():
            input_size = int(trial_config.get("input_size", 0))
            shape = [int(v) for v in trial_config.get("shape", [])]
            ref_inputs = generate_inputs(trial_config, device)
            harness_inputs = []
            tensor_idx = 0
            for item in ref_inputs:
                if isinstance(item, torch.Tensor):
                    t = _harness_fill_random_bf16(input_size, tensor_idx + 1)
                    harness_inputs.append(t.to(device).reshape(shape))
                    tensor_idx += 1
                else:
                    harness_inputs.append(item)
            return harness_inputs

        # Helper: run a .py module with given inputs, return output tensor
        def _run_py_model(module, inputs):
            model_cls = getattr(module, "Model")
            get_init = getattr(module, "get_init_inputs", None)
            init_args = list(get_init()) if get_init else []
            model = model_cls(*init_args).cuda(device)
            out = model(*inputs)
            torch.cuda.synchronize(device)
            return out.float().cpu().flatten() if hasattr(out, 'float') else None

        for slug, r in impl_results.items():
            if slug == golden_slug:
                r["correctness"] = None  # golden = no comparison
            elif "error" in r:
                r["correctness"] = {"passed": False, "reason": r["error"]}
            else:
                impl_info = next((im for im in impls if im["slug"] == slug), None)
                impl_tensor = r.get("output_tensor")

                if impl_info and impl_info["type"] == "cu" and golden_module is not None:
                    # .cu impl vs .py golden: re-run golden with C harness inputs
                    try:
                        with torch.no_grad(), torch.cuda.device(device):
                            ref_tensor = _run_py_model(golden_module, _make_harness_inputs())
                        r["correctness"] = _check_correctness(ref_tensor, impl_tensor)
                    except Exception as exc:
                        r["correctness"] = {"passed": None, "reason": f"cu correctness error: {exc}"}

                elif impl_info and impl_info["type"] == "py" and golden_is_cu:
                    # .py impl vs .cu golden: re-run .py impl with C harness inputs,
                    # compare against golden's output (which also used C harness inputs)
                    try:
                        old_path = list(sys.path)
                        sys.path.insert(0, str(impl_info["dir"]))
                        try:
                            impl_module = load_reference_module(impl_info["entry"])
                        finally:
                            sys.path[:] = old_path
                        with torch.no_grad(), torch.cuda.device(device):
                            impl_out = _run_py_model(impl_module, _make_harness_inputs())
                        r["correctness"] = _check_correctness(golden_tensor, impl_out)
                    except Exception as exc:
                        r["correctness"] = {"passed": None, "reason": f"py vs cu correctness error: {exc}"}

                else:
                    # .py impl vs .py golden: same inputs → direct compare
                    r["correctness"] = _check_correctness(golden_tensor, impl_tensor)

        # --- Build output (strip output_tensor, not JSON-serializable) ---
        output_impls = {}
        for slug, r in impl_results.items():
            entry = {}
            if "performance" in r:
                entry["performance"] = r["performance"]
            if "correctness" in r:
                entry["correctness"] = r["correctness"]
            if "error" in r:
                entry["error"] = r["error"]
            output_impls[slug] = entry

        result = {
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "status": "ok",
            "golden_slug": golden_slug,
            "impls": output_impls,
        }
        print(json.dumps(result, indent=2))
        return 0

    except TimeoutError:
        print(json.dumps({
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "status": "timeout",
            "error": "trial watchdog timeout expired",
        }, indent=2))
        return 1

    except Exception as exc:
        print(json.dumps({
            "metadata": metadata.model_dump(),
            "config_slug": args.config_slug,
            "status": "error",
            "error": str(exc),
        }, indent=2))
        return 1

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        try:
            gpu_cleanup(device)
        except Exception:
            pass
        release_device_lock(lock_fd)
        device_index = device.index if device.index is not None else 0
        lock_path = Path.home() / ".cuda_exec" / f".lock_cuda_{device_index}"
        cleanup_lockfile(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
