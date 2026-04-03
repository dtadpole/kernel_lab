#!/usr/bin/env bash
# Setup FA4 CuTe DSL venv for H100 hosts with CUDA 12.x drivers.
#
# The standard service venv has cuda-python==13.2.0 which requires CUDA 13.x
# driver. H100 hosts (h8_3, h8_4) have driver 550.90.07 (CUDA 12.4), so a
# dedicated venv with cuda-bindings==12.8.0 is needed.
#
# Usage:
#   bash scripts/setup_fa4_venv.sh           # creates ~/.fa4_venv
#   bash scripts/setup_fa4_venv.sh /path     # creates venv at /path
#
# After setup, run FA4 CuTe DSL with:
#   CUDA_VISIBLE_DEVICES=4 CUTE_DSL_ARCH=sm_90a ~/.fa4_venv/bin/python cutedsl.py
#
# Requirements:
#   - Python 3.12
#   - Internet access (pip install from PyPI + PyTorch index)
#   - NVIDIA driver >= 550 (CUDA 12.x)

set -euo pipefail

VENV_DIR="${1:-$HOME/.fa4_venv}"

echo "=== FA4 CuTe DSL venv setup ==="
echo "Target: $VENV_DIR"
echo ""

# Check driver
if command -v nvidia-smi &>/dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "NVIDIA driver: $DRIVER"
else
    echo "WARNING: nvidia-smi not found"
fi

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR"
    read -p "Recreate? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$VENV_DIR"
fi

echo "[1/5] Creating venv..."
python3.12 -m venv "$VENV_DIR"

echo "[2/5] Installing PyTorch 2.6.0+cu124..."
"$VENV_DIR/bin/pip" install --quiet \
    torch==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Installing cuda-bindings==12.8.0..."
"$VENV_DIR/bin/pip" install --quiet cuda-bindings==12.8.0

echo "[4/5] Installing nvidia-cutlass-dsl==4.4.2 (--no-deps)..."
"$VENV_DIR/bin/pip" install --quiet nvidia-cutlass-dsl==4.4.2 --no-deps

echo "[5/5] Installing flash-attn-4..."
"$VENV_DIR/bin/pip" install --quiet "flash-attn-4>=4.0.0b5"

echo ""
echo "=== Verifying installation ==="

# Smoke test
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
CUTE_DSL_ARCH=sm_90a \
"$VENV_DIR/bin/python" -c "
import torch
print(f'  torch:           {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:             {torch.cuda.get_device_name()}')

import flash_attn
print(f'  flash-attn:      {flash_attn.__version__}')

try:
    import nvidia.cutlass.dsl
    print(f'  cutlass-dsl:     OK')
except ImportError as e:
    print(f'  cutlass-dsl:     FAIL ({e})')

try:
    import cuda.bindings
    print(f'  cuda-bindings:   OK')
except ImportError as e:
    print(f'  cuda-bindings:   FAIL ({e})')

# FA4 CuTe DSL smoke test
from flash_attn.cute import flash_attn_func
q = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device='cuda')
out = flash_attn_func(q, k, v, causal=False)
if isinstance(out, tuple): out = out[0]
assert out.shape == (1, 32, 1, 64), f'Bad shape: {out.shape}'
print(f'  FA4 smoke test:  PASSED')
" 2>&1 | grep -v DEBUG

echo ""
echo "=== Setup complete ==="
echo "Venv: $VENV_DIR"
echo ""
echo "Usage:"
echo "  CUDA_VISIBLE_DEVICES=4 CUTE_DSL_ARCH=sm_90a $VENV_DIR/bin/python cutedsl.py"
