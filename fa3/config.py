"""Path configuration for FA3 compilation.

All fbcode paths are centralized here so they can be adjusted via environment
variables when the fbsource checkout location differs.
"""

import os

# Root of the fbsource checkout.  Override with FBSOURCE env var if needed.
FBSOURCE = os.environ.get("FBSOURCE", "/data/users/zhenc/fbsource")

# FA3 source directory (CUDA kernels + C++ API + Python interface)
FA3_HOPPER_DIR = os.path.join(FBSOURCE, "fbcode", "fa3", "hopper")

# CUTLASS 4.3.5 header directories (header-only library)
CUTLASS_ROOT = os.path.join(FBSOURCE, "third-party", "cutlass", "4.3.5")
CUTLASS_INCLUDE = os.path.join(CUTLASS_ROOT, "include")
CUTLASS_TOOLS_UTIL_INCLUDE = os.path.join(CUTLASS_ROOT, "tools", "util", "include")
CUTLASS_TOOLS_LIBRARY_INCLUDE = os.path.join(CUTLASS_ROOT, "tools", "library", "include")
CUTLASS_TOOLS_LIBRARY_SRC = os.path.join(CUTLASS_ROOT, "tools", "library", "src")

# Build output directory for compiled .so
BUILD_DIR = os.path.join(os.path.dirname(__file__), "lib")

# CUDA toolkit — must match kernel_lab's torch (compiled with CUDA 13.0).
# Uses FA3_CUDA_HOME env var (not CUDA_HOME, which the system may override).
CUDA_HOME = os.environ.get("FA3_CUDA_HOME", "/usr/local/cuda-13.0")
