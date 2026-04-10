"""Python wrapper for FA3 kernels.

Loads the compiled fa3_kernels.so and re-exports FA3's Python interface
(flash_attn_interface.py) from fbcode. No FA3 code is duplicated — we
reference fbcode's original files and only provide the compiled .so.

Usage:
    from fa3.wrapper import flash_attn_func

    out, lse = flash_attn_func(q, k, v, causal=True)
"""

import os
import sys

from fa3.config import BUILD_DIR, FA3_HOPPER_DIR


def _setup_import_paths():
    """Add the compiled .so and torch libs to the search paths.

    The fa3_kernels.so links against libc10.so, libtorch.so, etc. from
    PyTorch. We must ensure these are findable via LD_LIBRARY_PATH before
    the .so is loaded.
    """
    import torch

    # Add torch's lib directory to LD_LIBRARY_PATH so fa3_kernels.so
    # can find libc10.so, libtorch_cuda.so, etc. at load time.
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if torch_lib not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{ld_path}" if ld_path else torch_lib

    # Pre-load libcuda.so (CUDA driver API) as RTLD_GLOBAL so that
    # fa3_kernels.so can resolve cuDriverGetVersion and other driver symbols.
    import ctypes
    try:
        ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # Will fail later with a clearer error if truly missing

    # Pre-load torch to ensure its shared libraries are in memory.
    import torch.cuda  # noqa: F401 — ensures CUDA runtime is initialized

    # Ensure our compiled .so is findable on sys.path
    if BUILD_DIR not in sys.path:
        sys.path.insert(0, BUILD_DIR)

    # Verify .so exists
    so_path = os.path.join(BUILD_DIR, "fa3_kernels.so")
    if not os.path.exists(so_path):
        raise ImportError(
            f"fa3_kernels.so not found at {so_path}.\n"
            f"Run: cd ~/kernel_lab && .venv/bin/python -m fa3.build"
        )


def _import_interface():
    """Import flash_attn_interface.py from fbcode using importlib.

    We can't simply do `from fa3.hopper.flash_attn_interface import ...`
    because kernel_lab's own `fa3/` package shadows fbcode's `fa3/` on
    sys.path. Instead we load the module directly by file path.
    """
    import importlib.util

    interface_path = os.path.join(FA3_HOPPER_DIR, "flash_attn_interface.py")
    if not os.path.exists(interface_path):
        raise ImportError(
            f"flash_attn_interface.py not found at {interface_path}.\n"
            f"Check fa3/config.py — FA3_HOPPER_DIR={FA3_HOPPER_DIR}"
        )

    spec = importlib.util.spec_from_file_location(
        "fa3_flash_attn_interface", interface_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Module initialization ---

_setup_import_paths()
_interface = _import_interface()

# Re-export the key functions
flash_attn_func = _interface.flash_attn_func
flash_attn_varlen_func = _interface.flash_attn_varlen_func
flash_attn_with_kvcache = _interface.flash_attn_with_kvcache
flash_attn_combine = _interface.flash_attn_combine
