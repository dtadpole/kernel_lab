"""Optimization task specification for the CUDA agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationTask:
    """Everything the agent needs to start an iterative optimization run.

    The agent receives this as its initial context and uses it to drive
    compile -> evaluate -> modify -> repeat cycles via cuda_exec.
    """

    run_tag: str
    version: str
    direction_id: int
    direction_slug: str
    reference_files: dict[str, str]
    initial_generated_files: dict[str, str]
    configs: dict[str, dict[str, Any]]
    max_iterations: int = 10
    speedup_target: float | None = None
    cuda_exec_url: str = "http://127.0.0.1:8000"
