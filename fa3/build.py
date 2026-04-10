"""Build FA3 kernels from fbcode source using kernel_lab's torch + nvcc.

Compiles the FA3 CUDA/C++ source files from fbcode/fa3/hopper/ against
kernel_lab's pip-installed PyTorch headers, producing an ABI-compatible
fa3_kernels.so in fa3/lib/.

Usage:
    cd ~/kernel_lab
    .venv/bin/python -m fa3.build          # full build (~270 files, ~10-15 min)
    .venv/bin/python -m fa3.build --check  # verify .so exists and loads

The compiled .so is cached in fa3/lib/ and only needs to be rebuilt when:
  - PyTorch is upgraded (ABI change)
  - FA3 source code changes in fbcode
  - CUTLASS headers change
"""

import glob
import os
import sys
import time


def get_sources():
    """Collect all FA3 source files from fbcode (no copying)."""
    from fa3.config import FA3_HOPPER_DIR

    sources = [
        os.path.join(FA3_HOPPER_DIR, "flash_api.cpp"),
        os.path.join(FA3_HOPPER_DIR, "flash_fwd_combine.cu"),
    ]
    instantiations = sorted(glob.glob(
        os.path.join(FA3_HOPPER_DIR, "instantiations", "*.cu")
    ))
    sources.extend(instantiations)

    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(
            f"FA3 source files not found. Missing {len(missing)} files.\n"
            f"First missing: {missing[0]}\n"
            f"Is FBSOURCE set correctly? Current FA3_HOPPER_DIR: {FA3_HOPPER_DIR}"
        )

    return sources


def get_include_paths():
    """Collect all include paths needed for compilation."""
    from fa3.config import (
        CUTLASS_INCLUDE,
        CUTLASS_TOOLS_LIBRARY_INCLUDE,
        CUTLASS_TOOLS_LIBRARY_SRC,
        CUTLASS_TOOLS_UTIL_INCLUDE,
        FA3_HOPPER_DIR,
    )

    paths = [
        FA3_HOPPER_DIR,
        CUTLASS_INCLUDE,
        CUTLASS_TOOLS_UTIL_INCLUDE,
        CUTLASS_TOOLS_LIBRARY_INCLUDE,
        CUTLASS_TOOLS_LIBRARY_SRC,
    ]

    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            f"Include directories not found: {missing}\n"
            f"Check fa3/config.py paths."
        )

    return paths


# nvcc flags matching fbcode's BUCK configuration
EXTRA_CUDA_CFLAGS = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--ftemplate-backtrace-limit=0",
    "--resource-usage",
    "-lineinfo",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DNDEBUG",
    # SM90a: the 'a' suffix enables accelerated features (WGMMA, setmaxnreg)
    "-gencode=arch=compute_90a,code=sm_90a",
    # SM80 for Ampere compatibility
    "-gencode=arch=compute_80,code=sm_80",
]

EXTRA_CFLAGS = [
    "-Wno-strict-aliasing",
]


def build(verbose: bool = True):
    """Compile FA3 and return the loaded module."""
    from fa3.config import BUILD_DIR, CUDA_HOME

    os.makedirs(BUILD_DIR, exist_ok=True)

    # Set CUDA_HOME so torch.utils.cpp_extension finds the right nvcc
    os.environ["CUDA_HOME"] = CUDA_HOME

    from torch.utils.cpp_extension import load

    sources = get_sources()
    include_paths = get_include_paths()

    if verbose:
        print(f"FA3 build: {len(sources)} source files")
        print(f"FA3 build: include paths: {len(include_paths)}")
        print(f"FA3 build: output directory: {BUILD_DIR}")
        print(f"FA3 build: CUDA_HOME: {CUDA_HOME}")
        print(f"FA3 build: starting compilation (this may take 10-15 minutes)...")

    t0 = time.time()

    module = load(
        name="fa3_kernels",
        sources=sources,
        extra_include_paths=include_paths,
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        extra_cflags=EXTRA_CFLAGS,
        build_directory=BUILD_DIR,
        verbose=verbose,
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"FA3 build: done in {elapsed:.1f}s")

    return module


def check():
    """Verify that the compiled .so exists and can be loaded."""
    from fa3.config import BUILD_DIR

    so_path = os.path.join(BUILD_DIR, "fa3_kernels.so")
    if not os.path.exists(so_path):
        print(f"FAIL: {so_path} does not exist. Run: python -m fa3.build")
        return False

    sys.path.insert(0, BUILD_DIR)
    try:
        import fa3_kernels
        funcs = [x for x in dir(fa3_kernels) if not x.startswith("_")]
        print(f"OK: fa3_kernels loaded from {so_path}")
        print(f"    exports: {funcs}")
        return True
    except ImportError as e:
        print(f"FAIL: could not import fa3_kernels: {e}")
        return False


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = check()
        sys.exit(0 if ok else 1)
    else:
        build(verbose=True)
        check()
