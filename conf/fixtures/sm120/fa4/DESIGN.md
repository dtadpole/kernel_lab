# FA4 Fixture Design

## Overview

Flash Attention 4 forward pass benchmark fixture for SM120 (RTX 5090 / RTX PRO 6000 Blackwell).

- **Reference:** FA4 CuTe DSL (`flash_attn.cute.flash_attn_func`) — SM80 base class path on SM120
- **Configs:** 8 MHA configs from [AVO paper (arXiv 2603.24517)](https://arxiv.org/abs/2603.24517) — 4 causal + 4 non-causal, total tokens fixed at 32768

## Configs

All configs: `num_heads=16`, `head_dim=128`, `dtype=BF16`, `batch_size × seq_len = 32768`.

| Slug | Batch | SeqLen | Causal |
|------|-------|--------|--------|
| mha-causal-b8-s4096 | 8 | 4096 | yes |
| mha-causal-b4-s8192 | 4 | 8192 | yes |
| mha-causal-b2-s16384 | 2 | 16384 | yes |
| mha-causal-b1-s32768 | 1 | 32768 | yes |
| mha-noncausal-b8-s4096 | 8 | 4096 | no |
| mha-noncausal-b4-s8192 | 4 | 8192 | no |
| mha-noncausal-b2-s16384 | 2 | 16384 | no |
| mha-noncausal-b1-s32768 | 1 | 32768 | no |

## FA4 SM120 Patch

`flash-attn-4` (tested on 4.0.0b5 and 4.0.0b7) has 3 bugs on SM120 (affects both RTX 5090 and RTX PRO 6000 Blackwell). Patches from [Dao-AILab/flash-attention#2386](https://github.com/Dao-AILab/flash-attention/issues/2386) / [PR #2404](https://github.com/Dao-AILab/flash-attention/pull/2404) (open, not yet merged as of 2026-04-01).

Patched files in `cuda_exec/.venv/lib/python3.12/site-packages/flash_attn/cute/`:

### Bug 1: Forward TMA epilogue mis-gating (`flash_fwd.py:652`)

**Symptom:** `AttributeError: 'NoneType' object has no attribute '_trait'` in `cpasync.tma_partition()`
**Root cause:** `self.use_tma_O = self.arch >= Arch.sm_90` is True on SM120 (120 >= 90), but SM80 base class never provides a TMA output atom (`tma_atom_O=None`). CuTe DSL JIT traces both branches during compilation.
**Fix:** `self.use_tma_O = False`

### Bug 2: Backward `dQ_single_wg` unbound (`interface.py`, SM120 config block)

**Symptom:** `UnboundLocalError: cannot access local variable 'dQ_single_wg'`
**Root cause:** Only assigned in `arch // 10 == 9` (Hopper) path; compile-key covers `arch // 10 in [8, 9, 12]`.
**Fix:** Add to the `arch // 10 == 12` block:
```python
dQ_single_wg = False
num_stages_PdS = 2 if head_dim <= 64 else 1
```

### Bug 3: `nvvm.atomicrmw()` API incompatibility (`utils.py:394`)

**Symptom:** `TypeError: atomicrmw() got an unexpected keyword argument 'res'`
**Root cause:** `nvidia-cutlass-dsl==4.4.2` changed NVVM binding API.
**Fix:** Replace `nvvm.atomicrmw(res=...)` with PTX inline asm `red.global.add.f32 [$0], $1;` via `llvm.inline_asm`.

### Verification

After patches, FA4 forward on RTX 5090 achieves 207–239 TFLOPS across all 8 configs:

| Config | Latency | TFLOPS |
|--------|---------|--------|
| causal b8-s4096 | 2.65ms | 207.8 |
| causal b4-s8192 | 5.12ms | 214.7 |
| causal b2-s16384 | 10.06ms | 218.7 |
| causal b1-s32768 | 20.09ms | 218.9 |
| noncausal b8-s4096 | 4.60ms | 239.0 |
| noncausal b4-s8192 | 9.22ms | 238.4 |
| noncausal b2-s16384 | 18.36ms | 239.5 |
| noncausal b1-s32768 | 36.82ms | 238.9 |

NCU profile (causal b2-s16384): 79.5% SM throughput, 8.33% occupancy (255 regs/thread), L2 hit 97%.

### When to remove patches

Remove when `flash-attn-4` merges PR #2404 or `nvidia-cutlass-dsl` fixes `tma_partition` None-atom handling.
