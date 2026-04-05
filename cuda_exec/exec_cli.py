"""Hydra-based CLI for ik:exec (compile, trial, profile).

Usage:
    .venv/bin/python -m cuda_exec.exec_cli exec.action=compile exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda
    .venv/bin/python -m cuda_exec.exec_cli exec.action=trial exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda
    .venv/bin/python -m cuda_exec.exec_cli exec.action=profile exec.kernel=matmul exec.arch=sm90 exec.impl=gen-cuda exec.side=generated
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from cuda_exec.impls import load_configs, resolve_impl, resolve_impls
from cuda_exec.models import (
    CompileRequest,
    Metadata,
    ProfileRequest,
    TrialRequest,
)
from cuda_exec.tasks import compile_endpoint, trial_endpoint, profile_endpoint

logger = logging.getLogger(__name__)


def _build_metadata(kernel: str, impl_slug: str, run_tag: str, turn: int) -> Metadata:
    """Build metadata from exec config."""
    tag = run_tag
    return Metadata(
        run_tag=tag,
        version="v1",
        direction_id=0,
        direction_slug=f"{kernel}-{impl_slug}",
        turn=turn,
    )


def do_compile(
    kernel: str,
    arch: str,
    impl_slug: str,
    *,
    run_tag: str = "auto",
    turn: int = 1,
    timeout: int = 300,
    data_root: Path | None = None,
) -> dict:
    """Compile an implementation against its reference."""
    impl = resolve_impl(kernel, arch, impl_slug, data_root=data_root)
    metadata = _build_metadata(kernel, impl_slug, run_tag, turn)

    # Find the primary reference
    all_impls = resolve_impls(kernel, arch, "all", data_root=data_root)
    refs = [r for r in all_impls if r["source"] == "ref"]
    if not refs:
        raise ValueError(f"No reference implementations found for {kernel}/{arch}")
    primary_ref = refs[0]

    # Build file maps
    ref_files = dict(primary_ref["files"])
    # Include .py gen impls as additional reference files
    py_gens = [r for r in all_impls if r["source"] == "gen" and r["file_type"] == "py"]
    for pg in py_gens:
        ref_files.update(pg["files"])
    # cudnn = second ref if available
    cudnn_files = refs[1]["files"] if len(refs) > 1 else {}

    compile_req = CompileRequest(
        metadata=metadata,
        timeout_seconds=timeout,
        reference_files=ref_files,
        generated_files=impl["files"],
        cudnn_files=cudnn_files,
    )
    resp = compile_endpoint(compile_req)
    return resp.model_dump(mode="json")


def do_trial(
    kernel: str,
    arch: str,
    impl_slug: str,
    *,
    configs: str | list[str] = "all",
    run_tag: str = "auto",
    turn: int = 1,
    timeout: int = 300,
    data_root: Path | None = None,
) -> dict:
    """Trial an implementation across configs."""
    metadata = _build_metadata(kernel, impl_slug, run_tag, turn)

    all_configs = load_configs(kernel, data_root=data_root)
    if configs == "all":
        trial_configs = all_configs
    else:
        trial_configs = {k: v for k, v in all_configs.items() if k in configs}
        if not trial_configs:
            raise ValueError(f"No matching configs found. Available: {list(all_configs.keys())}")

    trial_req = TrialRequest(
        metadata=metadata,
        timeout_seconds=timeout,
        configs=trial_configs,
    )
    resp = trial_endpoint(trial_req)
    return resp.model_dump(mode="json")


def do_profile(
    kernel: str,
    arch: str,
    impl_slug: str,
    *,
    configs: str | list[str] = "all",
    side: str = "generated",
    run_tag: str = "auto",
    turn: int = 1,
    timeout: int = 300,
    data_root: Path | None = None,
) -> dict:
    """Profile an implementation with NCU."""
    metadata = _build_metadata(kernel, impl_slug, run_tag, turn)

    all_configs = load_configs(kernel, data_root=data_root)
    if configs == "all":
        profile_configs = all_configs
    else:
        profile_configs = {k: v for k, v in all_configs.items() if k in configs}
        if not profile_configs:
            raise ValueError(f"No matching configs found. Available: {list(all_configs.keys())}")

    profile_req = ProfileRequest(
        metadata=metadata,
        timeout_seconds=timeout,
        configs=profile_configs,
        side=side,
    )
    resp = profile_endpoint(profile_req)
    return resp.model_dump(mode="json")


def cli_main() -> None:
    """CLI entry point for ik:exec (Hydra-based)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    _CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")

    overrides = [arg for arg in sys.argv[1:] if "=" in arg]

    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    exec_cfg = cfg.exec

    # Set CUDA_VISIBLE_DEVICES from config if specified
    gpu = exec_cfg.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    action = exec_cfg.action
    kernel = exec_cfg.kernel
    arch = exec_cfg.arch
    impl_slug = exec_cfg.impl
    run_tag = exec_cfg.get("run_tag", "auto")
    turn = exec_cfg.get("turn", 1)
    timeout = exec_cfg.get("timeout", 300)
    data_root_str = exec_cfg.get("data_root")
    data_root = Path(data_root_str).expanduser() if data_root_str else None

    # Parse configs
    configs_val = exec_cfg.get("configs", "all")
    if isinstance(configs_val, str) and configs_val != "all":
        configs_val = [configs_val]
    elif hasattr(configs_val, "__iter__") and not isinstance(configs_val, str):
        configs_val = list(configs_val)

    if action == "compile":
        result = do_compile(
            kernel, arch, impl_slug,
            run_tag=run_tag, turn=turn, timeout=timeout, data_root=data_root,
        )
    elif action == "trial":
        result = do_trial(
            kernel, arch, impl_slug,
            configs=configs_val, run_tag=run_tag, turn=turn, timeout=timeout, data_root=data_root,
        )
    elif action == "profile":
        side = exec_cfg.get("side", "generated")
        result = do_profile(
            kernel, arch, impl_slug,
            configs=configs_val, side=side, run_tag=run_tag, turn=turn, timeout=timeout, data_root=data_root,
        )
    else:
        print(f"Unknown action: {action}. Must be compile, trial, or profile.", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    cli_main()
