# FA4 All-Compute Kernel — NCU-Guided Optimization Analysis

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Architecture | SM 12.0 (Blackwell) |
| SMs | 188 |
| Peak BF16 Tensor Core | 503.8 TFLOPS (FP32 accum, dense) |
| SM Frequency | 2.52 GHz (observed in NCU) |
| DRAM Bandwidth | 1,792 GB/s (GDDR7) |
| L2 Cache | 96 MB |
| CUDA | 13.0 |

## Kernel Architecture

**All-compute, no warp specialization:**
- BLOCK_Q=128, BLOCK_KV=128, DIM=128
- 8 warps (256 threads), all participate in cp.async loads AND mma.sync compute
- mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
- SMEM: Q(32KB persistent) + K(32KB) + V(32KB) = 96KB
- Pipeline: V loads overlap QK compute, K_next loads overlap PV compute
- 255 registers, 0 spills, 1 barrier

## Performance (5 warmup + 10 timed trials, L2 flush)

| Config | FLOPs | CuTe DSL (ref) | Generated | cuDNN | gen/ref | gen/cuDNN | Gen TFLOPS | % Peak |
|--------|-------|------|------|------|---------|-----------|-----------|--------|
| causal-b8-s4096 | 0.550T | 1.622ms | 1.649ms | 1.545ms | 0.98x | 0.94x | 333.5 | 66.2% |
| causal-b4-s8192 | 1.100T | 3.044ms | 3.136ms | 2.900ms | 0.97x | 0.92x | 350.7 | 69.6% |
| noncausal-b8-s4096 | 1.100T | 2.622ms | 2.992ms | 2.756ms | 0.88x | 0.92x | 367.6 | 73.0% |
| noncausal-b4-s8192 | 2.199T | 5.138ms | 5.864ms | 5.396ms | 0.88x | 0.92x | 375.0 | 74.5% |

### Summary

| | vs CuTe DSL | vs cuDNN | Peak Utilization |
|---|---|---|---|
| **Causal** | 0.97–0.98x | 0.92–0.94x | 66–70% |
| **Non-causal** | 0.88x | 0.92x | 73–75% |

## NCU Profile (causal-b8-s4096)

### Speed of Light

| Metric | Value |
|--------|-------|
| Duration | 1.72ms |
| Compute (SM) Throughput | 67.85% |
| **Pipe Tensor Cycles Active** | **67.85%** |
| Memory Throughput | 39.86% |
| DRAM Throughput | 17.31% |
| L1/TEX Throughput | 42.19% |
| L2 Throughput | 30.34% |
| DRAM Bandwidth | 295.9 GB/s |

### Occupancy

| Metric | Value |
|--------|-------|
| Theoretical Occupancy | 16.67% |
| Achieved Occupancy | 16.65% |
| Active Warps Per Scheduler | 2.00 |
| Eligible Warps Per Scheduler | 0.35 |
| Issued Warp Per Scheduler | 0.25 |

### Scheduler Statistics

| Metric | Value |
|--------|-------|
| One or More Eligible | 24.92% |
| **No Eligible** | **75.08%** |
| Warp Cycles Per Issued Instruction | 8.02 |
| Executed IPC (active) | 1.00 |
| Issue Slots Busy | 23.54% |

### Warp Stall Breakdown

| Stall Reason | Cycles/Inst | % of Total | Analysis |
|---|---|---|---|
| **Stall Math Pipe Throttle** | **3.05** | **37.6%** | Tensor core pipeline is saturated — THE bottleneck |
| **Stall Wait** | **2.15** | **26.5%** | __syncthreads + cp.async.wait overhead |
| Selected (useful work) | 1.00 | 12.3% | Only 12% of cycles issue useful instructions |
| Stall MIO Throttle | 0.63 | 7.8% | ldmatrix pipeline throttle |
| Stall Not Selected | 0.41 | 5.0% | Warp eligible but not selected |
| Stall Barrier | 0.23 | 2.8% | barrier sync |
| Stall Long Scoreboard | 0.19 | 2.3% | Global memory dependency |
| Stall Short Scoreboard | 0.12 | 1.5% | Shared memory dependency |
| Stall LG Throttle | 0.09 | 1.1% | Load/global throttle |

### Instruction Statistics

| Metric | Value |
|--------|-------|
| Executed Instructions | 768,360,670 |
| Local Memory Spilling | 0 bytes |
| Shared Memory Spilling | 0 bytes |
| Avg. Not Predicated Off Threads | 28.85 of 32 |

### Compute Throughput Breakdown

| Pipeline | % of Peak |
|----------|-----------|
| SM: Pipe Tensor Cycles Active | 67.85% |
| SM: Inst Executed Pipe Lsu | 39.86% |
| SM: Inst Executed | 23.54% |
| SM: Pipe Fmaheavy Cycles Active | 8.77% |
| SM: Pipe Aluheavy Cycles Active | 7.93% |
| SM: Pipe Fma Cycles Active | 7.17% |
| **SM: Pipe Tma Cycles Active** | **0%** — TMA not used |

## Comparison: All-Compute vs Old Warp-Specialized

| Metric | Warp-Specialized (old) | All-Compute (new) |
|--------|----------------------|-------------------|
| Tensor Pipe Active | 60.18% | **67.85%** (+7.7pp) |
| #1 Stall | stall_wait 47% | **math_pipe 37.6%** |
| #2 Stall | stall_barrier 19% | stall_wait 26.5% |
| Registers | 255 (28B spill) | **255 (0 spill)** |
| Barriers | 5 named | **1 (syncthreads)** |
| Occupancy | 10.4% | **16.65%** |
| causal-b8 | 1.686ms | **1.649ms** (+2.2%) |
| noncausal-b8 | 3.020ms | **2.992ms** (+0.9%) |

**Key insight**: The all-compute architecture shifted the bottleneck from barrier stalls (66% in warp-specialized) to tensor core saturation (37.6%). This is a **qualitatively better** stall profile — the tensor core IS the limiting resource, not synchronization overhead.

## Optimization Attempts

### Attempt: BLOCK_KV=64 with K Double-Buffer

**Hypothesis**: Smaller KV tiles + K double-buffering = better load/compute overlap via 2-stage async pipeline.

**Changes**:
- BLOCK_KV: 128→64
- K double-buffer: K0(16KB) + K1(16KB) in SMEM
- `cp.async.wait_group<1>` to allow one group in flight
- SMEM: Q(32KB) + K0(16KB) + K1(16KB) + V(16KB) = 80KB
- 239 registers, 0 spills

**Result**: **-8% regression** (causal-b8: 1.782ms vs 1.649ms)

**Root cause**: BLOCK_KV=64 requires 2x more KV iterations → 2x more __syncthreads → doubled sync overhead. The K double-buffering saved nothing because K loads (16KB from L2) were already fully hidden behind PV compute (~2000 clocks). The cure was worse than the disease.

### Attempted: wgmma on SM120

**Result**: `wgmma.mma_async with floating point types` **NOT supported on SM120**. Confirmed by ptxas error: "Instruction 'wgmma.mma_async' cannot be compiled for architecture 'compute_120'". wgmma is SM90 (Hopper) only.

### Verified: tcgen05.mma on SM120

**Result**: `tcgen05.mma.cta_group::1.kind::f16` **compiles successfully** on SM120. This is the SM120-native 5th-gen tensor core instruction. Requires descriptor-based operands, SMEM-resident accumulators, and async fence/commit/wait semantics.

## Architecture Ceiling Analysis

### Why mma.sync.m16n8k16 tops out at ~68-75%

1. **Instruction width**: Each mma.sync does 16×8×16×2 = 4,096 FLOPs. CuTe DSL's tcgen05 does tiles up to 128×256×16 = 1,048,576 FLOPs per instruction — **256x more**.

2. **Instruction count**: Our QK inner loop needs 128 mma.sync + 128 ldmatrix per warp per KV iteration. CuTe DSL needs ~8 tcgen05.mma for the same work — **16x fewer instructions**.

3. **Scheduler utilization**: With 768M instructions across the kernel, at 1.00 IPC and 23.54% issue slots busy, the scheduler is massively underutilized. The warps spend 75% of time with no eligible warp.

4. **TMA absence**: We use thread-based cp.async (all 256 threads issue loads). CuTe DSL uses TMA (hardware DMA, zero thread overhead). NCU confirms: `Pipe Tma Cycles Active = 0%` — we're not using TMA at all.

### What would tcgen05.mma + TMA enable

| Resource | mma.sync (current) | tcgen05 + TMA (potential) |
|----------|--------------------|----|
| FLOPs/instruction | 4,096 | 262,144+ (64x–256x more) |
| Instructions per QK (128×128×128) | 128 per warp | ~8 per warpgroup |
| Load mechanism | cp.async (thread-based) | TMA (hardware DMA) |
| Operand source | Registers (ldmatrix from SMEM) | SMEM descriptors (no ldmatrix) |
| Accumulator | Registers | SMEM |
| Expected utilization | 68–75% | 83–88% (matching CuTe DSL) |

### Recommendation

The mma.sync.m16n8k16 kernel has reached its architectural ceiling at **68–75% tensor utilization**. Further meaningful improvement requires:

1. **tcgen05.mma** for SM120-native wider tensor core operations
2. **TMA** for zero-thread hardware DMA loads
3. **SMEM-based accumulator** pattern (tcgen05 writes results to SMEM, not registers)

This is a ground-up kernel rewrite using inline PTX for SM120 tcgen05 instructions.

## Commit

- **Branch**: `worktree-fa4`
- **Kernel**: `data/generated/sm120/fa4/generated.cu` (BLOCK_KV=128, all-compute)
