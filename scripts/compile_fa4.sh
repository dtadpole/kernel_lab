#!/usr/bin/env bash
# Compile FA4 kernel as shared library for benchmark harness.
# Usage: bash scripts/compile_fa4.sh [source.cu] [output.so]
set -euo pipefail

SOURCE="${1:-data/generated/sm90/fa4/generated.cu}"
OUTPUT="${2:-/tmp/fa4_sm90_bench.so}"
# Default to cuda-13.0; fallback to 12.9 if 13.0 missing headers
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
if [[ ! -f "${CUDA_HOME}/include/cuda_runtime.h" ]]; then
    for try in /usr/local/cuda-13.0 /usr/local/cuda-12.9; do
        [[ -f "${try}/include/cuda_runtime.h" ]] && { CUDA_HOME="$try"; break; }
    done
fi
NVCC="${CUDA_HOME}/bin/nvcc"

# Auto-detect GPU arch
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
ARCH="sm_${CC/./}a"
COMPUTE="compute_${CC/./}a"

echo "compile_fa4: ${SOURCE} → ${OUTPUT}"
echo "  CUDA_HOME=${CUDA_HOME}"
echo "  arch=${ARCH}"

"${NVCC}" -gencode "arch=${COMPUTE},code=${ARCH}" \
    -O2 -Xcompiler -fPIC --shared \
    -I"${CUDA_HOME}/include" -lcuda \
    -Xptxas -v \
    -o "${OUTPUT}" "${SOURCE}" 2>&1

echo "compile_fa4: done"
