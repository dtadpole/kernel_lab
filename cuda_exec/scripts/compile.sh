#!/usr/bin/env bash
set -euo pipefail

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

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${CUDA_HOME}/bin/nvcc"
PTXAS="${CUDA_HOME}/bin/ptxas"
CUOBJDUMP="${CUDA_HOME}/bin/cuobjdump"
NVDISASM="${CUDA_HOME}/bin/nvdisasm"
PTXAS_ARCH="${PTXAS_ARCH:-sm_120}"
PTXAS_FLAGS="${PTXAS_FLAGS:--v}"
NVDISASM_FLAGS="${NVDISASM_FLAGS:---print-code --print-instruction-encoding --print-life-ranges}"

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

COMMON_NVCC_ARGS=("-arch=${ARCH}" "-std=${CPP_STD}" "-O${OPT_LEVEL}")
if [[ "$LINEINFO" == "1" ]]; then
  COMMON_NVCC_ARGS+=("-lineinfo")
fi

# If harness is provided, add its directory as an include path
HARNESS_INCLUDE_ARGS=()
if [[ -n "$HARNESS" ]]; then
  HARNESS_DIR="$(dirname "$HARNESS")"
  HARNESS_INCLUDE_ARGS=("-I${HARNESS_DIR}")
fi

PTX_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "${HARNESS_INCLUDE_ARGS[@]}" -ptx "$SOURCE" -o "$PTX_PATH")
PTXAS_CMD=("$PTXAS" "-arch=${PTXAS_ARCH}")
if [[ -n "$PTXAS_FLAGS" ]]; then
  # shellcheck disable=SC2206
  PTXAS_EXTRA=( $PTXAS_FLAGS )
  PTXAS_CMD+=("${PTXAS_EXTRA[@]}")
fi
PTXAS_CMD+=("$PTX_PATH" -o "$CUBIN_PATH")
RESOURCE_USAGE_CMD=("$CUOBJDUMP" --dump-resource-usage "$CUBIN_PATH")
# shellcheck disable=SC2206
NVDISASM_EXTRA=( $NVDISASM_FLAGS )
NVDISASM_CMD=("$NVDISASM" "${NVDISASM_EXTRA[@]}" "$CUBIN_PATH")

# Binary command: with or without harness
# -lcuda needed for cuTensorMapEncodeTiled (CUDA driver API)
if [[ -n "$HARNESS" ]]; then
  BINARY_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "${HARNESS_INCLUDE_ARGS[@]}" "$HARNESS" "$SOURCE" -lcuda -o "$OUTPUT")
else
  BINARY_CMD=("$NVCC" "${COMMON_NVCC_ARGS[@]}" "$SOURCE" -lcuda -o "$OUTPUT")
fi

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
