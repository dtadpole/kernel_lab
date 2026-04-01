#!/usr/bin/env bash
set -euo pipefail

# NCU capture wrapper for both generated (binary) and reference (Python) sides.
#
# Generated:  profile.sh --target ./binary --export-prefix PREFIX --set detailed
# Reference:  profile.sh --target python ref.py --export-prefix PREFIX --set detailed \
#               --kernel-name 'regex:"cutlass|vector_add"'
#
# Everything after --target (up to the next --flag) becomes the profiled command.

TARGET=()
EXPORT_PREFIX=""
SET_NAME="default"
TARGET_PROCESSES="all"
FORCE_OVERWRITE="1"
KERNEL_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      shift
      # Consume all non-flag arguments as the target command.
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        TARGET+=("$1")
        shift
      done
      ;;
    --export-prefix)
      EXPORT_PREFIX="$2"
      shift 2
      ;;
    --set)
      SET_NAME="$2"
      shift 2
      ;;
    --target-processes)
      TARGET_PROCESSES="$2"
      shift 2
      ;;
    --kernel-name)
      KERNEL_NAME="$2"
      shift 2
      ;;
    --no-force-overwrite)
      FORCE_OVERWRITE="0"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ${#TARGET[@]} -eq 0 ]]; then
  echo "Missing required --target" >&2
  exit 2
fi
if [[ -z "$EXPORT_PREFIX" ]]; then
  echo "Missing required --export-prefix" >&2
  exit 2
fi

mkdir -p "$(dirname "$EXPORT_PREFIX")"

CMD=(sudo --preserve-env /usr/local/cuda/bin/ncu --set "$SET_NAME" --target-processes "$TARGET_PROCESSES")
if [[ "$FORCE_OVERWRITE" == "1" ]]; then
  CMD+=(--force-overwrite)
fi
if [[ -n "$KERNEL_NAME" ]]; then
  CMD+=(--kernel-name "$KERNEL_NAME")
fi
CMD+=(--export "$EXPORT_PREFIX" "${TARGET[@]}")

echo "toolkit_command:"
echo "  cwd: $(pwd)"
printf '  cmd:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
