# Plan: SM90 Matmul WGMMA Optimization (69 → 700+ TFLOPS)

**Goal**: Replace WMMA-based matmul with native SM90 WGMMA+TMA kernel reaching cuBLAS-level performance (~740 TFLOPS at 8192×8192), maintaining correctness at every step.

**Architecture**: Incremental build — each step adds one capability, verified by `ik:bench` before proceeding. Correctness failure = revert immediately.

**Tech Stack**: CUDA C++ with inline PTX, SM90a (H100), BF16 I/O, FP32 accumulation

## Current State

- Kernel: `data/gen/sm90/matmul/cuda.cu` (WMMA 128×128, 69 TFLOPS, correct)
- Reference: cuBLAS via `data/ref/matmul/cublas.py` (744 TFLOPS)
- Configs: `data/configs/matmul.json` (6 configs: 256×256 to 8192×8192)
- Bench: `ik:bench matmul` (all 6 must pass ✓)

## Key Learnings from Old WGMMA Kernel (752 TFLOPS, broken correctness)

From `results/sm90/h100_sxm/matmul/` and git history (commit 30810af):

1. **SMEM layout**: A(128×64) and B(128×64) stored with 128B swizzle. WGMMA reads 8-row × K-col chunks.
2. **Descriptor**: 64-bit with `layout_type=1` (128B swizzle), `stride = 8*TILE_K*2 >> 4`
3. **Tile**: 128×256×64, 4-stage pipeline, 3 warpgroups (1 producer + 2 consumers)
4. **The bug**: likely in TMA descriptor setup or SMEM swizzle — output was garbage for small matrices

## Steps

### Step 1: WGMMA m64n128k16 — single K-step, no pipeline (target: ~100 TFLOPS)

**File**: `data/gen/sm90/matmul/cuda.cu`

**What**: Replace WMMA with a single WGMMA m64n128k16 instruction via inline PTX.
Keep cp.async loads (no TMA). Single-buffered. 1 warpgroup (128 threads).
CTA tile: 64×128, iterate over K in steps of 16.

**Key details**:
- A in SMEM: 64×16 BF16, row-major, stride = 16*2 = 32 bytes
- B in SMEM: 128×16 BF16, stored **K-major** (transposed): 16×128, stride = 128*2 = 256 bytes
  - WGMMA SS mode reads B as K×N, so B must be laid out with K as the leading dimension
  - Load B transposed: `sB[k][n] = B[k*N + n]` (K-major = row-major of K×N)
- WGMMA descriptor: no swizzle first (layout_type=0), simple stride
- Accumulator store: use per-thread (row,col) mapping from CUTLASS reference
- `setmaxnreg.inc.sync.aligned.u32 232` at kernel start

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul WGMMA m64n128k16 baseline — XX TFLOPS`

### Step 2: 128B swizzle for SMEM (target: ~150 TFLOPS)

**File**: `data/gen/sm90/matmul/cuda.cu`

**What**: Add 128B swizzle to SMEM layout to eliminate bank conflicts.
- Swizzle function: `addr ^= ((addr >> 7) & 0x7) << 4` (128B XOR swizzle)
- Update descriptor: `layout_type = 1` (B128 swizzle)
- Update stride in descriptor to match swizzled layout

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul 128B SMEM swizzle — XX TFLOPS`

### Step 3: Multi-stage pipeline with mbarrier (target: ~250 TFLOPS)

**File**: `data/gen/sm90/matmul/cuda.cu`

**What**: Add 4-stage pipeline with mbarrier synchronization.
- 4 SMEM buffers for A and B
- mbarrier_init for full[4] + empty[4] barriers
- K-loop: producer loads → mbarrier_arrive_expect_tx → consumer waits → WGMMA → release
- cp.async.bulk or cp.async for loads

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul 4-stage mbarrier pipeline — XX TFLOPS`

### Step 4: Larger tile 128×256 with 2 WGMMA calls per K-step (target: ~400 TFLOPS)

**File**: `data/gen/sm90/matmul/cuda.cu`

**What**: Expand CTA tile to 128×256. Each K-step does 2× m64n128k16 WGMMA
(or 1× m64n256k16). Use wgmma.fence + 2×wgmma + commit_group pattern.

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul 128×256 tile — XX TFLOPS`

### Step 5: TMA loads replacing cp.async (target: ~500 TFLOPS)

**File**: `data/gen/sm90/matmul/cuda.cu`

**What**: Replace cp.async with TMA (cp.async.bulk.tensor).
- Create TMA descriptors via CUDA driver API (`cuTensorMapEncode`)
- TMA loads A and B tiles with hardware address generation
- mbarrier.arrive.expect_tx for TMA completion tracking

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul TMA loads — XX TFLOPS`

### Step 6: Warp specialization — 1 producer + 2 consumers (target: ~650 TFLOPS)

**File**: `data/gen/sm90/matmul/cuda.cu`

**What**: 3 warpgroups (384 threads). WG0 = producer (TMA loads only),
WG1+WG2 = consumers (WGMMA compute only). Producer uses setmaxnreg.dec
to release registers. Consumers use setmaxnreg.inc for more accumulators.

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul warp specialization — XX TFLOPS`

### Step 7: Persistent scheduling + L2 tiling (target: ~700+ TFLOPS)

**File**: `data/gen/sm90/matmul/cuda.cu`

**What**: Grid-stride persistent tile loop. CTA swizzle group for L2 locality.
Launch 132 CTAs (= H100 SM count).

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul persistent scheduling — XX TFLOPS`

### Step 8: Final bench + results

Run `/ik:bench matmul` as final authoritative measurement.
Write results to `results/sm90/h100_sxm/matmul/`.

## Task Dependencies

| Group | Steps | Can Parallelize |
|-------|-------|-----------------|
| 1 | Step 1 | No — foundation |
| 2 | Step 2 | No — depends on 1 |
| 3 | Step 3 | No — depends on 2 |
| 4 | Step 4 | No — depends on 3 |
| 5 | Step 5 | No — depends on 4 |
| 6 | Step 6 | No — depends on 5 |
| 7 | Step 7 | No — depends on 6 |
| 8 | Step 8 | No — depends on 7 |

Strictly sequential — each step builds on the previous. No parallelism.

## Critical Rules

1. **Correctness gate**: `ik:bench matmul` must show all ✓ before committing
2. **Incremental**: one capability per step, never two
3. **Revert on failure**: if a step can't pass correctness after 3 attempts, revert and rethink
4. **Small tests first**: verify with 256×256 before running full bench
5. **Keep baseline**: always backup working kernel before modifying
