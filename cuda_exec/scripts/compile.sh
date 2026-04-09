#!/usr/bin/env bash
set -euo pipefail

# ── Interface ──────────────────────────────────────────────────────────
#
# CLI args (the only interface):
#   --source FILE     Source .cu file (required)
#   --output FILE     Output binary path (required)
#   --arch ARCH       GPU arch: "native" (auto-detect) or e.g. "sm_90a" (default: native)
#   --harness FILE    Link with eval harness .cu (optional)
#   --std STD         C++ standard (default: c++17)
#   --opt LEVEL       Optimization level (default: 3)
#   --no-lineinfo     Disable -lineinfo
#
# Environment (only two):
#   CUDA_HOME              CUDA toolkit path (auto-detected if unset)
#   CUDA_VISIBLE_DEVICES   GPU selection (not used by this script, but logged)
#
# Everything else is hardcoded with sensible defaults.
# ───────────────────────────────────────────────────────────────────────

SOURCE=""
OUTPUT=""
ARCH="native"
CPP_STD="c++17"
OPT_LEVEL="3"
LINEINFO="1"
HARNESS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --std)
      CPP_STD="$2"
      shift 2
      ;;
    --opt)
      OPT_LEVEL="$2"
      shift 2
      ;;
    --no-lineinfo)
      LINEINFO="0"
      shift
      ;;
    --harness)
      HARNESS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE" ]]; then
  echo "Missing required --source" >&2
  exit 2
fi
if [[ -z "$OUTPUT" ]]; then
  echo "Missing required --output" >&2
  exit 2
fi

# ── Auto-detect CUDA_HOME ─────────────────────────────────────────────
# Priority: $CUDA_HOME > /usr/local/cuda > newest /usr/local/cuda-*
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME="/usr/local/cuda"
  else
    # Find newest cuda-* with nvcc
    for d in $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V -r); do
      if [[ -x "$d/bin/nvcc" ]]; then
        CUDA_HOME="$d"
        break
      fi
    done
  fi
  if [[ -z "${CUDA_HOME:-}" ]]; then
    echo "ERROR: cannot find CUDA toolkit. Set CUDA_HOME." >&2
    exit 2
  fi
fi

# ── Auto-detect GPU architecture ──────────────────────────────────────
if [[ "$ARCH" == "native" ]]; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -n "$CC" ]]; then
    ARCH="sm_${CC/./}a"  # e.g., 9.0 -> sm_90a (always use arch-specific features)
  else
    echo "ERROR: --arch=native but cannot detect GPU compute capability" >&2
    exit 2
  fi
fi

# ── Toolkit binaries ──────────────────────────────────────────────────
NVCC="${CUDA_HOME}/bin/nvcc"
PTXAS="${CUDA_HOME}/bin/ptxas"
CUOBJDUMP="${CUDA_HOME}/bin/cuobjdump"
NVDISASM="${CUDA_HOME}/bin/nvdisasm"

# ptxas arch always matches nvcc arch (including the 'a' suffix)
PTXAS_ARCH="${ARCH}"

# ── Output paths ──────────────────────────────────────────────────────
OUTPUT_DIR="$(dirname "$OUTPUT")"
OUTPUT_NAME="$(basename "$OUTPUT")"
OUTPUT_STEM="${OUTPUT_NAME%.*}"
ARTIFACT_DIR="$OUTPUT_DIR"
TURN_ROOT="$(dirname "$ARTIFACT_DIR")"
LOG_DIR="${TURN_ROOT}/logs"
PTX_PATH="${ARTIFACT_DIR}/${OUTPUT_STEM}.ptx"
CUBIN_PATH="${ARTIFACT_DIR}/${OUTPUT_STEM}.cubin"
RESOURCE_USAGE_PATH="${ARTIFACT_DIR}/${OUTPUT_STEM}.resource-usage.txt"
SASS_NVDISASM_PATH="${ARTIFACT_DIR}/${OUTPUT_STEM}.nvdisasm.sass"
AGGREGATE_LOG_BASENAME="compile.attempt_001"
NVCC_PTX_STDOUT_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.nvcc-ptx.stdout"
NVCC_PTX_STDERR_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.nvcc-ptx.stderr"
PTXAS_STDOUT_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.ptxas.stdout"
PTXAS_STDERR_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.ptxas.stderr"
RESOURCE_USAGE_STDOUT_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.resource-usage.stdout"
RESOURCE_USAGE_STDERR_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.resource-usage.stderr"
NVDISASM_STDOUT_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.nvdisasm.stdout"
NVDISASM_STDERR_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.nvdisasm.stderr"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

# ── Build nvcc flags ──────────────────────────────────────────────────
# Use -gencode to ensure both PTX and SASS targets carry the 'a' suffix
# when needed (e.g., sm_90a for WGMMA). Plain -arch=sm_90a generates PTX
# with .target sm_90 (no 'a') which ptxas rejects for WGMMA instructions.
_COMPUTE_ARCH="${ARCH/sm_/compute_}"  # sm_90a -> compute_90a
COMMON_NVCC_ARGS=("-gencode" "arch=${_COMPUTE_ARCH},code=${ARCH}" "-std=${CPP_STD}" "-O${OPT_LEVEL}")
if [[ "$LINEINFO" == "1" ]]; then
  COMMON_NVCC_ARGS+=("-lineinfo")
fi

# Always link libcuda (needed for TMA / cuTensorMap APIs)
COMMON_NVCC_ARGS+=("-lcuda")

# If harness is provided, add its directory as an include path and link NVML
# (eval_harness.cu uses nvmlDeviceGetClockInfo / nvmlDeviceGetTemperature)
HARNESS_INCLUDE_ARGS=()
if [[ -n "$HARNESS" ]]; then
  HARNESS_DIR="$(dirname "$HARNESS")"
  HARNESS_INCLUDE_ARGS=("-I${HARNESS_DIR}")
  COMMON_NVCC_ARGS+=("-L/usr/lib" "-lnvidia-ml")
fi

# Extra include directories from NVCC_INCLUDE_DIRS (space-separated)
NVCC_INCLUDE_DIRS="${NVCC_INCLUDE_DIRS:-}"
if [[ -n "$NVCC_INCLUDE_DIRS" ]]; then
  for dir in $NVCC_INCLUDE_DIRS; do
    COMMON_NVCC_ARGS+=("-I${dir}")
  done
fi

# Extra library directories from NVCC_LIB_DIRS (space-separated)
NVCC_LIB_DIRS="${NVCC_LIB_DIRS:-}"
if [[ -n "$NVCC_LIB_DIRS" ]]; then
  for dir in $NVCC_LIB_DIRS; do
    COMMON_NVCC_ARGS+=("-L${dir}" "-Xlinker" "-rpath=${dir}")
  done
fi

# Extra linker libraries from NVCC_EXTRA_LIBS (space-separated, e.g. "cuda cudnn")
NVCC_EXTRA_LIBS="${NVCC_EXTRA_LIBS:-}"
if [[ -n "$NVCC_EXTRA_LIBS" ]]; then
  for lib in $NVCC_EXTRA_LIBS; do
    COMMON_NVCC_ARGS+=("-l${lib}")
  done
fi

# Extra nvcc flags from NVCC_EXTRA_FLAGS (space-separated)
NVCC_EXTRA_FLAGS="${NVCC_EXTRA_FLAGS:-}"
if [[ -n "$NVCC_EXTRA_FLAGS" ]]; then
  # shellcheck disable=SC2206
  COMMON_NVCC_ARGS+=( $NVCC_EXTRA_FLAGS )
fi

# ── Assemble commands ─────────────────────────────────────────────────
PTX_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "${HARNESS_INCLUDE_ARGS[@]}" -ptx "$SOURCE" -o "$PTX_PATH")
PTXAS_CMD=("$PTXAS" "-arch=${PTXAS_ARCH}" -v "$PTX_PATH" -o "$CUBIN_PATH")
RESOURCE_USAGE_CMD=("$CUOBJDUMP" --dump-resource-usage "$CUBIN_PATH")
NVDISASM_CMD=("$NVDISASM" --print-code --print-instruction-encoding --print-life-ranges "$CUBIN_PATH")

if [[ -n "$HARNESS" ]]; then
  BINARY_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "${HARNESS_INCLUDE_ARGS[@]}" "$HARNESS" "$SOURCE" -o "$OUTPUT")
else
  BINARY_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "$SOURCE" -o "$OUTPUT")
fi

# ── Dump environment info ─────────────────────────────────────────────
ENV_INFO_PATH="${LOG_DIR}/${AGGREGATE_LOG_BASENAME}.env-info.txt"
{
  echo "=== Environment ==="
  echo "hostname: $(hostname)"
  echo "date: $(date -Iseconds)"
  echo "user: $(whoami)"
  echo ""
  echo "=== CUDA ==="
  echo "CUDA_HOME: ${CUDA_HOME}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
  echo "nvcc: $("$NVCC" --version 2>&1 | tail -1)"
  echo "ptxas: $("$PTXAS" --version 2>&1 | tail -1)"
  echo "nvdisasm: $("$NVDISASM" --version 2>&1 | tail -1)"
  echo "cuobjdump: $("$CUOBJDUMP" --version 2>&1 | tail -1)"
  echo "ARCH: $ARCH"
  echo ""
  echo "=== GPU ==="
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "<nvidia-smi not available>"
  echo ""
  echo "=== Compile flags ==="
  echo "CPP_STD: $CPP_STD"
  echo "OPT_LEVEL: $OPT_LEVEL"
  echo "LINEINFO: $LINEINFO"
  printf 'COMMON_NVCC_ARGS:'
  printf ' %q' "${COMMON_NVCC_ARGS[@]}"
  printf '\n'
  echo ""
  echo "=== Python ==="
  python3 --version 2>/dev/null || echo "<python3 not available>"
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<not set>}"
} > "$ENV_INFO_PATH" 2>&1

# ── Execute pipeline ──────────────────────────────────────────────────
echo "toolkit_pipeline:"
echo "  cwd: $(pwd)"
echo "  source: $SOURCE"
if [[ -n "$HARNESS" ]]; then
  echo "  harness: $HARNESS"
fi
echo "  ptx: $PTX_PATH"
echo "  cubin: $CUBIN_PATH"
echo "  resource_usage: $RESOURCE_USAGE_PATH"
printf '  nvcc_ptx_cmd:'
printf ' %q' "${PTX_CMD[@]}"
printf '\n'
printf '  ptxas_cmd:'
printf ' %q' "${PTXAS_CMD[@]}"
printf '\n'
printf '  resource_usage_cmd:'
printf ' %q' "${RESOURCE_USAGE_CMD[@]}"
printf '\n'
printf '  nvdisasm_cmd:'
printf ' %q' "${NVDISASM_CMD[@]}"
printf '\n'
printf '  binary_cmd:'
printf ' %q' "${BINARY_CMD[@]}"
printf '\n'

echo "[compile 1/5] Generate PTX -> $PTX_PATH"
"${PTX_CMD[@]}" > >(tee "$NVCC_PTX_STDOUT_PATH") 2> >(tee "$NVCC_PTX_STDERR_PATH" >&2)

echo "[compile 2/5] Assemble PTX -> CUBIN ($PTXAS_ARCH) -> $CUBIN_PATH"
"${PTXAS_CMD[@]}" > >(tee "$PTXAS_STDOUT_PATH") 2> >(tee "$PTXAS_STDERR_PATH" >&2)

echo "[compile 3/5] Dump resource usage -> $RESOURCE_USAGE_PATH"
"${RESOURCE_USAGE_CMD[@]}" > >(tee "$RESOURCE_USAGE_STDOUT_PATH" > "$RESOURCE_USAGE_PATH") 2> >(tee "$RESOURCE_USAGE_STDERR_PATH" >&2)

echo "[compile 4/5] Dump SASS with nvdisasm -> $SASS_NVDISASM_PATH"
if ! "${NVDISASM_CMD[@]}" > >(tee "$NVDISASM_STDOUT_PATH" > "$SASS_NVDISASM_PATH") 2> >(tee "$NVDISASM_STDERR_PATH" >&2); then
  echo "[compile 4/5] WARNING: nvdisasm failed (non-fatal, continuing)"
fi

echo "[compile 5/5] Build runnable binary -> $OUTPUT"
"${BINARY_CMD[@]}"

# Harness mode: validate required symbols
if [[ -n "$HARNESS" ]]; then
  echo "[compile 5/5+] Validating harness symbols in $OUTPUT"
  NM_OUT="$(nm "$OUTPUT" 2>/dev/null || true)"
  MISSING=""
  if ! grep -q " T kernel_run" <<< "$NM_OUT"; then
    MISSING=" kernel_run"
  fi
  if [[ -n "$MISSING" ]]; then
    echo "ERROR: generated.cu must export extern \"C\" symbol:${MISSING}" >&2
    echo "Implement: extern \"C\" int kernel_run(__nv_bfloat16**, int, __nv_bfloat16**, int, int, cudaStream_t)" >&2
    exit 1
  fi
fi
