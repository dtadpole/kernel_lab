---
name: Always use compile.sh for kernel compilation
description: Never hand-type nvcc commands. Use cuda_exec/scripts/compile.sh which handles arch detection, CUDA_HOME, flags consistently.
type: feedback
---

Always use `cuda_exec/scripts/compile.sh` for kernel compilation, never hand-type nvcc commands.

**Why:** Hand-typed nvcc commands lead to inconsistent flags (wrong -arch, missing -lcuda, wrong CUDA_HOME path). The compile.sh script auto-detects GPU architecture, generates PTX/CUBIN/SASS artifacts, logs resource usage, and validates harness symbols.

**How to apply:**
```bash
# For FA4 kernel with eval_harness:
CUDA_HOME=/usr/local/cuda-13.0 bash cuda_exec/scripts/compile.sh \
  --source data/generated/sm90/fa4/generated.cu \
  --output /tmp/fa4_bench \
  --harness cuda_exec/scripts/eval_harness.cu

# For shared library (benchmark script):
CUDA_HOME=/usr/local/cuda-13.0 bash cuda_exec/scripts/compile.sh \
  --source data/generated/sm90/fa4/generated.cu \
  --output /tmp/fa4_sm90_bench.so
```

If compile.sh doesn't support .so output, create a thin wrapper `scripts/compile_fa4.sh`.
Don't re-invent compilation in each session.
