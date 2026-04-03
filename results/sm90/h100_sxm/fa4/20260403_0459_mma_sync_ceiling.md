# FA4 Optimization Findings — mma.sync Ceiling Analysis

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 |
| Architecture | SM 9.0 (Hopper) |
| SMs | 132 |
| Peak BF16 Tensor Core | 989.5 TFLOPS |
| DRAM Bandwidth | 3,352 GB/s |
| Registers/SM | 65,536 |
| SMEM/SM | 228 KB |

## Baseline Performance (no change committed)

| Config | Generated TFLOPS | cuDNN TFLOPS | Gen/cuDNN |
|--------|-----------------|-------------|-----------|
| causal-b8-s4096 | 202 | 579 | 35% |
| causal-b4-s8192 | 141 | 615 | 23% |
| causal-b2-s16384 | 222 | 615 | 36% |
| noncausal-b8-s4096 | 210 | 646 | 32% |
| noncausal-b4-s8192 | 228 | 607 | 38% |
| noncausal-b2-s16384 | 233 | 593 | 39% |

Baseline: 168 registers, 0 spills, 5 barriers, 2 blocks/SM.

## Optimization Attempts (all reverted)

### 1. Fused QK-Softmax-PV (0.63-0.67x regression)

**Hypothesis**: Process 16 KV rows at a time (QK chunk → softmax → PV chunk)
instead of separate full QK → softmax → PV passes. Eliminates P_rmem storage
(saves 36 registers), reduces barriers from 4 to 2.

**Result**: 126 registers, 0 spills, 3 barriers. Correctness OK after fixing
P_frag ordering. Performance regressed to 137-147 TFLOPS (0.63-0.67x).

**Root cause**: 4x more Q ldmatrix reloads (Q re-loaded per kv_chunk instead
of once). Lost K/V load overlap (both K and V must be loaded before compute
starts, removing the pipelining benefit where V loads during QK computation).

**Learning**: Register savings don't help if instruction count increases. The
ldmatrix overhead from Q reloading outweighed the register pressure improvement.
Combined K+V barriers eliminate the DMA/compute overlap that the separate
K_FULL/V_FULL barriers enabled.

### 2. Force 3 blocks/SM via launch_bounds(160, 3) (0.73-0.76x regression)

**Hypothesis**: Higher occupancy (12 MMA warps vs 8) improves tensor core
pipeline utilization from 50% to 75%.

**Result**: 128 registers, 100B spill stores, 104B spill loads, 88B stack.
Performance regressed to 155-175 TFLOPS (0.73-0.76x).

**Root cause**: Compiler introduced register spills to meet the 136-register
budget. Each spill access adds 30-100 cycles of local memory latency. The
spill overhead far exceeded the occupancy benefit.

**Learning**: Forcing occupancy through launch_bounds only works if the code
can naturally fit in the register budget. Spilling to achieve higher occupancy
is counterproductive for compute-bound kernels.

### 3. Inline MMA function (__forceinline__) (0.95-1.00x, neutral)

**Hypothesis**: Eliminating the __noinline__ function call overhead and giving
the compiler full visibility into both DMA and MMA code paths might improve
register allocation.

**Result**: 160 registers (vs 168), 0 spills. Performance neutral (0.95-1.00x).

**Root cause**: The compiler saved 8 registers but didn't improve scheduling.
The `__noinline__` attribute is actually beneficial: it gives ptxas independent
register allocation for each function, preventing the MMA path's register
pressure from affecting the DMA path.

**Learning**: `__noinline__` on warp-specialized functions is a good practice.
It lets the compiler optimize each execution path independently.

## Root Cause Analysis

### SASS Instruction Mix (MMA warp function)

| Instruction | Count | % of Total |
|-------------|------:|----------:|
| HMMA (tensor core) | 128 | 8.0% |
| LDSM (ldmatrix) | 136 | 8.5% |
| FMUL (float multiply) | 160 | 10.0% |
| IMAD (int mul-add) | 177 | 11.1% |
| LOP3 (3-input logic) | 164 | 10.3% |
| FADD (float add) | 68 | 4.3% |
| F2FP (float→bf16) | 48 | 3.0% |
| ISETP (int predicate) | 41 | 2.6% |
| MUFU (exp2f, rcp) | 36 | 2.3% |
| FMNMX (float minmax) | 36 | 2.3% |
| STG (global store) | 32 | 2.0% |
| BAR (barriers) | 14 | 0.9% |
| Other | 560 | 35.0% |
| **Total** | **1600** | **100%** |

### Why mma.sync Hits a Ceiling

1. **Synchronous execution**: `mma.sync.m16n8k16` stalls the warp during
   tensor core execution. Between each HMMA instruction, ~12 non-HMMA
   instructions must execute (softmax, address computation, ldmatrix loads).

2. **Low HMMA ratio**: Only 8% of instructions are tensor core operations.
   With 4 warp schedulers issuing from up to 8 warps (2 blocks/SM), the
   theoretical maximum is 4 HMMA per ~12.5 cycles = 32% tensor core
   utilization. Actual: ~23%.

3. **Address computation overhead**: IMAD (177) + LOP3 (164) = 341
   instructions (21%) dedicated to SMEM address calculation with swizzle
   XOR patterns. This is inherent to the mma.sync + cp.async approach.

4. **O_rmem rescaling**: FMUL (160) includes 64 rescale multiplies per KV
   iteration (online softmax) + 64 output normalization multiplies.
   Rescaling is mathematically necessary for numerical stability.

5. **Arithmetic intensity**: 67 FLOPs/byte — extremely compute-bound.
   Memory bandwidth is NOT a bottleneck. Changing data layout or improving
   coalescing would not help.

### The Path Forward: SM90-Native Instructions

Closing the remaining 3x gap to FA4/cuDNN (~600 TFLOPS) requires:

| Feature | Current | Target | Benefit |
|---------|---------|--------|---------|
| MMA | mma.sync.m16n8k16 | wgmma.mma_async.m64nNk16 | Async execution: tensor cores run while scalar pipeline does softmax |
| Data movement | cp.async + manual swizzle | TMA (Tensor Memory Accelerator) | Hardware-managed tiling/swizzling, eliminates ~341 IMAD+LOP3 per iteration |
| Barriers | bar.sync / bar.arrive | mbarrier | SM90-native async barriers with integrated TMA completion tracking |
| Warp model | 4 × 32-thread warps | 1 × 128-thread warp group | Matches wgmma's 128-thread warp group requirement |

This is an architectural rewrite, not an incremental optimization.
