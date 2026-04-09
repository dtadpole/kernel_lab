"""GPU clock locking for benchmark noise reduction.

Dynamically queries max GPU clocks via nvidia-smi and locks them for the
duration of a benchmark.  Gracefully degrades if sudo is unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


def _resolve_gpu_id() -> str:
    """Resolve physical GPU index from CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return cvd.split(",")[0].strip()


def query_gpu_clocks(gpu_id: Optional[str] = None) -> Dict[str, Any]:
    """Query current and max GPU clocks via nvidia-smi (no sudo needed).

    Returns dict with max_sm_mhz, max_mem_mhz, current_sm_mhz, current_mem_mhz.
    """
    if gpu_id is None:
        gpu_id = _resolve_gpu_id()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.max.graphics,clocks.max.memory,"
                "clocks.current.sm,clocks.current.memory",
                "--format=csv,noheader,nounits",
                "-i", gpu_id,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"error": f"nvidia-smi failed: {result.stderr.strip()}"}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 4:
            return {"error": f"unexpected nvidia-smi output: {result.stdout.strip()}"}
        return {
            "max_sm_mhz": int(parts[0]),
            "max_mem_mhz": int(parts[1]),
            "current_sm_mhz": int(parts[2]),
            "current_mem_mhz": int(parts[3]),
        }
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out"}
    except Exception as exc:
        return {"error": str(exc)}


def lock_gpu_clocks(gpu_id: str, sm_mhz: int) -> Dict[str, Any]:
    """Lock GPU SM clock to a fixed frequency. Requires sudo."""
    try:
        # Enable persistence mode
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", gpu_id, "-pm", "1"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        # Lock SM clock
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", gpu_id, "-lgc", str(sm_mhz)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return {"status": "ok", "locked_sm_mhz": sm_mhz}
    except subprocess.CalledProcessError as exc:
        return {"status": "error", "error": f"nvidia-smi failed (exit {exc.returncode}): {exc.stderr.strip()}"}
    except FileNotFoundError:
        return {"status": "error", "error": "nvidia-smi not found"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def unlock_gpu_clocks(gpu_id: str) -> Dict[str, Any]:
    """Reset GPU clocks to default. Requires sudo."""
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", gpu_id, "-rgc"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return {"status": "ok"}
    except subprocess.CalledProcessError as exc:
        return {"status": "error", "error": f"nvidia-smi rgc failed: {exc.stderr.strip()}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@contextmanager
def gpu_clock_context(
    gpu_id: Optional[str] = None,
    enabled: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager that locks GPU clocks for the duration of a benchmark.

    Yields a status dict:
      - {"status": "ok", "locked_sm_mhz": ..., "locked_mem_mhz": ...}
      - {"status": "error", "error": "..."}
      - {"status": "skipped"}

    Always unlocks in finally, even on error.
    """
    if not enabled:
        yield {"status": "skipped"}
        return

    if gpu_id is None:
        gpu_id = _resolve_gpu_id()

    info: Dict[str, Any] = {"status": "skipped"}
    locked = False

    try:
        # Query max clocks dynamically
        clocks = query_gpu_clocks(gpu_id)
        if "error" in clocks:
            logger.warning("Cannot query GPU clocks: %s", clocks["error"])
            info = {"status": "error", "error": clocks["error"]}
            yield info
            return

        max_sm = clocks["max_sm_mhz"]
        max_mem = clocks["max_mem_mhz"]
        logger.info("GPU %s max clocks: SM %d MHz, Mem %d MHz", gpu_id, max_sm, max_mem)

        # Attempt lock
        lock_result = lock_gpu_clocks(gpu_id, max_sm)
        if lock_result["status"] != "ok":
            logger.warning("GPU clock lock failed: %s — continuing without lock", lock_result["error"])
            info = {"status": "error", "error": lock_result["error"],
                    "max_sm_mhz": max_sm, "max_mem_mhz": max_mem}
            yield info
            return

        locked = True

        # Verify lock took effect (idle GPU may not immediately show max clock,
        # but nvidia-smi -lgc ensures it will reach max under load)
        verify = query_gpu_clocks(gpu_id)
        if "error" not in verify:
            actual = verify["current_sm_mhz"]
            logger.info("Post-lock SM clock: %d MHz (max: %d MHz)", actual, max_sm)

        info = {"status": "ok", "locked_sm_mhz": max_sm, "locked_mem_mhz": max_mem}
        logger.info("GPU %s clocks locked: SM %d MHz", gpu_id, max_sm)
        yield info

    finally:
        if locked:
            unlock_result = unlock_gpu_clocks(gpu_id)
            if unlock_result["status"] == "ok":
                logger.info("GPU %s clocks unlocked", gpu_id)
            else:
                logger.warning("GPU %s clock unlock failed: %s", gpu_id, unlock_result.get("error"))
