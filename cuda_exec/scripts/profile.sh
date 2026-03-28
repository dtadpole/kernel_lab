#!/usr/bin/env bash
set -euo pipefail

TARGET=""
EXPORT_PREFIX=""
SET_NAME="default"
TARGET_PROCESSES="all"
FORCE_OVERWRITE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
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

if [[ -z "$TARGET" ]]; then
  echo "Missing required --target" >&2
  exit 2
fi
if [[ -z "$EXPORT_PREFIX" ]]; then
  echo "Missing required --export-prefix" >&2
  exit 2
fi

mkdir -p "$(dirname "$EXPORT_PREFIX")"

CMD=(/usr/local/cuda/bin/ncu --set "$SET_NAME" --target-processes "$TARGET_PROCESSES")
if [[ "$FORCE_OVERWRITE" == "1" ]]; then
  CMD+=(--force-overwrite)
fi
CMD+=(--export "$EXPORT_PREFIX" "$TARGET")

echo "toolkit_command:"
echo "  cwd: $(pwd)"
printf '  cmd:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
