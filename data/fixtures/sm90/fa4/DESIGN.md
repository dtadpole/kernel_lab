# FA4 Fixture Design — SM90 (Hopper)

## Overview

Flash Attention 4 forward pass benchmark fixture for SM90 (H100 SXM5 / H100 PCIe Hopper).

- **Reference:** FA4 CuTe DSL (`flash_attn.cute.flash_attn_func`) — SM80 base class path on SM90
- **Configs:** 6 MHA configs from [AVO paper (arXiv 2603.24517)](https://arxiv.org/abs/2603.24517) — 3 causal + 3 non-causal, total tokens fixed at 32768

## Target Hardware

| Variant | GPU | SMs | BF16 TFLOPS (dense) | HBM3 BW | SMEM/SM |
|---------|-----|-----|---------------------|---------|---------|
| H100 SXM5 | GH100 | 132 | 989.5 | 3352 GB/s | 228 KB |
| H100 PCIe | GH100 | 114 | 756.0 | 2039 GB/s | 228 KB |

## Configs

All configs: `num_heads=16`, `head_dim=128`, `dtype=BF16`, `batch_size * seq_len = 32768`.

| Slug | Batch | SeqLen | Causal |
|------|-------|--------|--------|
| mha-causal-b8-s4096 | 8 | 4096 | yes |
| mha-causal-b4-s8192 | 4 | 8192 | yes |
| mha-causal-b2-s16384 | 2 | 16384 | yes |
| mha-noncausal-b8-s4096 | 8 | 4096 | no |
| mha-noncausal-b4-s8192 | 4 | 8192 | no |
| mha-noncausal-b2-s16384 | 2 | 16384 | no |

## Reference Implementations

### cuDNN (`cudnn.py`)

Uses `torch.nn.functional.scaled_dot_product_attention` with `SDPBackend.CUDNN_ATTENTION` forced. This dispatches to cuDNN's fused flash attention kernel on H100. Serves as the vendor-optimized baseline.

### FA4 CuTe DSL (`cutedsl.py`)

Uses `flash_attn.cute.flash_attn_func` (FlashAttention-4 CuTe DSL) as the primary backend. FA4 CuTe DSL uses the SM80 base class path on SM90, which compiles and runs correctly on Hopper. Falls back to PyTorch SDPA (cuDNN) if flash-attn-4 or nvidia-cutlass-dsl are not installed.

Requires: `flash-attn-4 >= 4.0.0b5`, `nvidia-cutlass-dsl >= 4.4`.

## Generated Kernel (`generated.cu`)

Warp-specialized flash attention kernel targeting SM90:

- **Architecture:** 5 warps per thread-block (1 DMA + 4 MMA)
- **Tile sizes:** BLOCK_Q=128, BLOCK_KV=64, DIM=128
- **MMA:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
- **Data movement:** `cp.async.cg.shared.global` with commit/wait groups
- **Synchronization:** Named barriers (`bar.sync` / `bar.arrive`) for DMA/MMA decoupling
- **SMEM layout:** Q (32KB persistent) + K (32KB double-buffered) + V (16KB single-buffered) = 80KB total
- **Register strategy:** On-the-fly SMEM→register loading with ldmatrix pipelining
- **Layout:** Works directly on [B, S, H, D] with strided access, no transpose needed

### SM90-specific considerations

- 228 KB SMEM per SM (vs 100 KB on SM120) — room for larger tiles or higher occupancy
- Current kernel uses 80 KB, could potentially run 2 blocks/SM if register pressure allows
- Future optimization path: migrate to wgmma (warp group MMA) + TMA for SM90-native performance
  - wgmma operates on 128-thread warp groups with async MMA
  - TMA provides hardware-accelerated tensor loads with built-in tiling/swizzling
  - Both require significant architectural rewrite from the current mma.sync approach

### Limitations

- DIM=128 only (head_dim must be 128)
- MHA only (num_kv_heads must equal num_heads)
- Forward pass only (no backward/gradient support)

## Benchmark Results

See `results/sm90/h100_sxm/fa4/` for full benchmark reports.
