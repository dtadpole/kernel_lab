#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
OUTPUT=""
ARCH="native"
CPP_STD="c++17"
OPT_LEVEL="3"
LINEINFO="1"

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

mkdir -p "$(dirname "$OUTPUT")"

CMD=(/usr/local/cuda/bin/nvcc "-arch=${ARCH}" "-std=${CPP_STD}" "-O${OPT_LEVEL}")
if [[ "$LINEINFO" == "1" ]]; then
  CMD+=("-lineinfo")
fi
CMD+=("$SOURCE" -o "$OUTPUT")

echo "toolkit_command:"
echo "  cwd: $(pwd)"
printf '  cmd:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
