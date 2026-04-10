# FA3 (Flash Attention 3) integration module.
#
# Compiles Meta's internal FA3 CUDA kernels (fbcode/fa3/) using kernel_lab's
# own torch + nvcc, producing an ABI-compatible .so that can be imported
# directly without Buck.
#
# Usage:
#   # First-time build (~10-15 min):
#   python -m fa3.build
#
#   # Then import:
#   from fa3.wrapper import flash_attn_func
