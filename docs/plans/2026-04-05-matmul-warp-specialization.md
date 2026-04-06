# Plan: SM90 Matmul — Warp Specialization + No-Transpose B

**Goal**: Transform the current 1-warpgroup TMA+WGMMA matmul into a 3-warpgroup warp-specialized kernel (1 producer + 2 consumers) with MN-major B (no transpose), reaching ~650 TFLOPS on H100.

**Architecture**: 3 warpgroups (384 threads). WG0 = producer (TMA loads only). WG1 = consumer0 (WGMMA rows 0-63). WG2 = consumer1 (WGMMA rows 64-127). B is loaded directly from PyTorch's K×N row-major layout (MN-major) — zero transpose. Dual mbarrier sets: mbar_full (producer→consumers) + mbar_empty (consumers→producer).

**Tech Stack**: Inline PTX (explicit register allocation, no compiler dependence), SM90a (H100), BF16 I/O, FP32 accumulation. C++ is only for the `kernel_run` host function and kernel launch — the kernel body is 100% inline PTX asm.

**File**: `data/gen/sm90/matmul/cuda/cuda.cu`

## Current State

- Kernel: 1 warpgroup (128 threads), 3-stage TMA pipeline, K-major B (requires transpose)
- `transpose_bf16` kernel pre-transposes B: K×N → N×K before WGMMA
- WGMMA: `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16` with trans-b=0
- Performance: ~500 TFLOPS
- Verify: `ik:bench matmul` (all 6 configs ✓)

## Architecture Decision: Why This Design

1. **No transpose**: B comes from PyTorch as K×N row-major (N is contiguous = MN-major). WGMMA supports trans-b=1 for MN-major B. Eliminating the transpose kernel saves GMEM bandwidth and launch overhead.

2. **1 producer + 2 consumers**: Decouples TMA loads from WGMMA compute. While consumers run WGMMA on stage N, producer loads stage N+1. Two consumers double the compute throughput — each handles a 64×128 output sub-tile.

3. **No compiler dependence**: The entire kernel body is inline PTX. All registers are explicitly declared with `.reg` directives — no `float acc[64]` C++ arrays that the compiler maps to registers. The compiler cannot spill, reorder, or interfere. `setmaxnreg.dec 40` on producer frees registers for consumers. `setmaxnreg.inc 232` on consumers claims them. We know EXACTLY how many registers every thread uses.

4. **Matmul is simple**: The core loop is just 5 operations per K-step: wait barrier, build descriptor, fence, WGMMA ×4, commit. The epilogue is just convert+store. This doesn't need compiler magic — inline PTX handles it directly.

## Key Technical Details

### MN-Major B Descriptor (from WGMMA_DESCRIPTOR.md)

```
Bits 0-13:  SMEM address >> 4
Bits 16-29: LBO >> 4 = 32  (LBO = 512 bytes = 256 bf16)
Bits 32-45: SBO >> 4 = 64  (SBO = 1024 bytes = 512 bf16)
Bits 62-63: Swizzle = 1 (128B)
```

### K-substep advance for MN-major B

B in SMEM: TILE_K(64) rows × TILE_N(128) cols, N contiguous.
Each k16 substep = 16 rows × 128 cols × 2 bytes = 4096 bytes = 256 × 16B.
`desc_advance(db, ks * 256)` (was `ks * 2` for K-major).

### TMA descriptor for MN-major B

```cpp
// OLD (transposed B = N×K, K contiguous):
dims = [K, N], stride = K*2, box = [TILE_K, TILE_N], ptr = Bt

// NEW (original B = K×N, N contiguous):
dims = [N, K], stride = N*2, box = [TILE_N, TILE_K], ptr = B
```

TMA load coord order flips: `(ctaCol, kt*TILE_K)` (was `(kt*TILE_K, ctaCol)`).

### Synchronization

```
mbar_full[STAGES]:  init count=1 (1 TMA arrive)
                    Producer arrive_expect_tx → TMA hardware completes → phase flips
                    Both consumers wait_parity → proceed to WGMMA

mbar_empty[STAGES]: init count=2 (2 consumer arrives)
                    Each consumer arrives after wgmma.wait_group 1 (SMEM safe)
                    Producer waits → stage free for TMA reuse
```

Phase tracking:
- Consumer wait on mbar_full[s]:  parity = (kt / STAGES) & 1
- Producer wait on mbar_empty[s]: parity = ((kt / STAGES) - 1) & 1

### SMEM Layout

```
Stage 0: sA[0] (16384 B) | sB[0] (16384 B)
Stage 1: sA[1] (16384 B) | sB[1] (16384 B)
Stage 2: sA[2] (16384 B) | sB[2] (16384 B)
mbar_full[3]  (128 B aligned)
mbar_empty[3] (128 B aligned)
Total: 98560 bytes (~96 KB, well within H100's 228 KB)
```

## Steps

### Step 1: Delete transpose kernel and host-side transpose logic

**What**: Remove `transpose_bf16` kernel. Remove `s_Bt` allocation, transpose launch, and `Bt` pointer from `kernel_run`. Change TMA descriptor for B to use original `B` pointer with MN-major layout.

**Changes**:
1. Delete lines 296-316 (`transpose_bf16` kernel)
2. In `kernel_run`: delete `s_Bt` static buffer, `bt_bytes`, `cudaMalloc`, transpose launch (lines 363-377)
3. Change TMA B descriptor:
   - `dims = {N, K}` (was `{K, N}`)
   - `str = {N * 2}` (was `{K * 2}`)
   - `box = {TILE_N, TILE_K}` (was `{TILE_K, TILE_N}`)
   - `ptr = (void*)B` (was `(void*)Bt`)
4. Update kernel TMA load coords: `(ctaCol, kt*TILE_K)` (was `(kt*TILE_K, ctaCol)`)
5. Delete "Bt buffer is cached" comment

**Note**: Kernel will NOT produce correct results after this step alone — the WGMMA descriptor and trans-b flag haven't changed yet. This step only cleans the host side. Correctness comes after Step 2.

### Step 2: Add MN-major descriptor + change WGMMA trans-b to 1

**What**: Add `make_desc_mn_major` function. Change `wgmma_m64n128k16` to use trans-b=1. Update descriptor creation and k-substep advance for B.

**Changes**:
1. Add `make_desc_mn_major`:
```cpp
__device__ __forceinline__
uint64_t make_desc_mn_major(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)(32 & 0x3FFF) << 16;           /* LBO=512 bytes */
    desc |= (uint64_t)(64 & 0x3FFF) << 32;           /* SBO=1024 bytes */
    desc |= (uint64_t)(1) << 62;                      /* swizzle=128B */
    return desc;
}
```
2. In `wgmma_m64n128k16`, change PTX: `p,1,1,0,0` → `p,1,1,0,1` (trans-b=1)
3. In K-loop: `make_desc_mn_major(sB[cur])` for B descriptor (was `make_desc(sB[cur], TILE_K)`)
4. K-substep advance: `desc_advance(db, ks * 256)` (was `ks * 2`)

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓. If correctness fails, debug the MN-major descriptor values before proceeding. Do NOT revert to transpose. Iterate on LBO/SBO values.

**Commit**: `perf: matmul MN-major B — eliminate transpose kernel`

### Step 3: Rewrite kernel body in full inline PTX — 3 warpgroups

**What**: This is the core step. Rewrite the entire kernel body as inline PTX asm blocks with explicit `.reg` declarations. No C++ variables for accumulators, descriptors, or loop counters. Three branches: producer (WG0), consumer0 (WG1), consumer1 (WG2).

**Why inline PTX?** Matmul is simple — the core loop is 5 operations: wait barrier, build descriptor, fence, WGMMA×4, commit. The compiler can't add value here; it can only add risk (spills, reordering). By writing PTX directly, we know EXACTLY what registers are used.

**Constants** (C++ defines, used to compute asm immediates):
```cpp
#define WG_SIZE   128
#define THREADS   (3 * WG_SIZE)   /* 384 */
#define MBAR_FULL_OFFSET  (STAGES * STAGE_BYTES)           /* 98304 */
#define MBAR_EMPTY_OFFSET (MBAR_FULL_OFFSET + 128)         /* 98432 */
#define SMEM_TOTAL        (MBAR_EMPTY_OFFSET + 128)        /* 98560 */
```

**Kernel signature** (thin C++ wrapper, body is 100% asm):
```cpp
__global__ void __launch_bounds__(384, 1)
matmul_wgmma_tma(
    __nv_bfloat16* __restrict__ C, int M, int N, int K,
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B)
{
    extern __shared__ char smem[];
    /* Pass params into asm; everything else is PTX */
    asm volatile( ... );
}
```

**Producer branch (WG0) — inline PTX register budget: ~20 regs**:
```ptx
.reg .u32  tid, wg_id, numK, stage, kt, prefill;
.reg .u32  smem_base, sA_addr, sB_addr, mbar_f_addr, mbar_e_addr;
.reg .u32  tx_bytes, tile_k_offset, cta_row, cta_col;
.reg .u32  empty_parity, tmp;
.reg .pred p_tid0, p_producer, p_need_load, p_kt_ge1;

// setmaxnreg.dec frees registers for consumers
setmaxnreg.dec.sync.aligned.u32 40;

// Prefill loop: thread 0 issues TMA for stages 0..min(STAGES,numK)-1
// Main loop: wait mbar_empty[stage] → TMA load → arrive mbar_full[stage]
```

**Consumer branch (WG1/WG2) — inline PTX register budget: ~140 regs**:
```ptx
// Accumulators: 64 × f32 = 64 regs
.reg .f32  acc<64>;

// Descriptors: 4 × b64 = 8 regs
.reg .b64  da, db, dak, dbk;

// Loop control + addresses: ~10 regs
.reg .u32  kt, ks, numK, stage, parity, sd, prev_stage;
.reg .u32  sA_addr, sB_addr, mbar_f_addr, mbar_e_addr;
.reg .u32  local_tid, a_row_offset;
.reg .pred p_first, p_done, p_release, p_tid0;

// setmaxnreg.inc claims freed registers
setmaxnreg.inc.sync.aligned.u32 232;

// Zero accumulators
mov.f32 acc0,  0f00000000;
mov.f32 acc1,  0f00000000;
// ... (all 64)

// K-loop
CONSUMER_KLOOP:
  // 1. Compute stage = kt % 3, parity = (kt / 3) & 1
  // 2. Wait mbar_full[stage] — PTX mbarrier.try_wait.parity
  // 3. Build A descriptor (K-major): addr + LBO=1 + SBO=64 + swizzle=128B
  //    A offset: sA[stage] + a_row_offset (0 for WG1, 8192 for WG2)
  // 4. Build B descriptor (MN-major): addr + LBO=32 + SBO=64 + swizzle=128B
  // 5. wgmma.fence
  // 6. 4× wgmma.mma_async with desc_advance:
  //    A: +2 per substep (k16 = 32 bytes)
  //    B: +256 per substep (k16 = 4096 bytes)
  // 7. wgmma.commit_group
  // 8. If kt >= 1: wgmma.wait_group 1 → mbarrier.arrive on mbar_empty[prev]
  bra CONSUMER_KLOOP;

// Drain: wgmma.wait_group 0 → release last stage
// Epilogue: cvt.rn.bf16.f32 + st.global per accumulator element
```

**Descriptor construction in PTX** (no C++ helper function):
```ptx
// K-major A descriptor:
shr.u32   addr_16B, sA_addr, 4;
and.b32   addr_16B, addr_16B, 0x3FFF;
mov.b64   da, 0;
or.b64    da, da, addr_16B;          // bits 0-13: address
or.b64    da, da, 0x10000;           // bits 16: LBO=1
mov.u32   sbo, 64;                   // SBO=64
shl.b64   sbo64, sbo, 32;
or.b64    da, da, sbo64;             // bits 32-45: SBO
or.b64    da, da, 0x4000000000000000; // bit 62: swizzle=128B

// MN-major B descriptor:
shr.u32   addr_16B, sB_addr, 4;
and.b32   addr_16B, addr_16B, 0x3FFF;
mov.b64   db, 0;
or.b64    db, db, addr_16B;
mov.u64   lbo_shifted, 0x200000;     // 32 << 16
or.b64    db, db, lbo_shifted;       // bits 16-29: LBO=32
mov.u64   sbo_shifted, 0x4000000000; // 64 << 32
or.b64    db, db, sbo_shifted;       // bits 32-45: SBO=64
or.b64    db, db, 0x4000000000000000; // bit 62: swizzle=128B
```

**Desc advance in PTX** (add offset to low 32 bits):
```ptx
// A advance: +2 per k16 substep
mov.b64  {lo, hi}, da;
add.u32  lo, lo, 2;    // ks * 2 — unrolled
mov.b64  dak, {lo, hi};

// B advance: +256 per k16 substep
mov.b64  {lo, hi}, db;
add.u32  lo, lo, 256;  // ks * 256 — unrolled
mov.b64  dbk, {lo, hi};
```

**Epilogue store in PTX** (no C++ store_acc):
```ptx
// WGMMA m64n128 register mapping:
// Thread (warp, lane) → element (row, col)
// warp = local_tid / 32, lane = local_tid % 32
// row_base = ctaRow + warp*16 + lane/4
// col_base = ctaCol + (lane%4)*2
// acc[4*p+0] → (row_base, col_base + p*8)
// acc[4*p+1] → (row_base, col_base + p*8 + 1)
// acc[4*p+2] → (row_base+8, col_base + p*8)
// acc[4*p+3] → (row_base+8, col_base + p*8 + 1)

.reg .b16 bf_val;
.reg .u64 C_addr;
cvt.rn.bf16.f32 bf_val, acc0;
// compute global address: C + (row * N + col) * 2
st.global.b16 [C_addr], bf_val;
// ... repeat for all 64 accumulators
```

**Modular arithmetic in PTX** (stage = kt % 3):
```ptx
// Use a lookup table approach: kt % 3 cycles through 0,1,2,0,1,2,...
// Track stage as a separate counter, reset when stage == 3
add.u32   stage, stage, 1;
setp.eq.u32 p_reset, stage, 3;
@p_reset mov.u32 stage, 0;
```

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓. If correctness fails:
- First: verify descriptor bit patterns with printf from thread 0
- Second: verify mbar phase parity with single-tile test (256×256)
- Third: verify epilogue store mapping with known input pattern
Do NOT revert to single warpgroup. Debug within this architecture.

**Commit**: `perf: matmul 3-WG warp specialization — full PTX, no compiler dependence`

### Step 4: Performance tuning + final bench

**What**: Run full benchmark, tune if needed, record results.

**Possible tunings** (only if below 600 TFLOPS):
- Adjust STAGES (3 → 4) — just change the constants and unroll
- Try wgmma.wait_group 1 vs wait_group 0 in consumer loop
- Adjust setmaxnreg values (producer 24-40, consumer 224-240)
- Prefetch next mbar_full in consumer loop
- All tuning is PTX-level: change instruction ordering, barrier timing

**Verify**: `ik:bench matmul` — all 6 configs must pass ✓

**Commit**: `perf: matmul warp-specialized — XX TFLOPS`

**Write results**: `results/sm90/h100_sxm/matmul/YYYYMMDD_HHMM_<hash>_warp-specialization.md`

## Task Dependencies

| Group | Steps | Can Parallelize |
|-------|-------|-----------------|
| 1 | Step 1 (delete transpose + MN-major TMA) | No — foundation |
| 2 | Step 2 (MN-major WGMMA descriptor + trans-b) | No — depends on 1 |
| 3 | Step 3 (full PTX rewrite: 3 WGs, producer/consumer) | No — depends on 2 |
| 4 | Step 4 (tune + bench) | No — depends on 3 |

Strictly sequential — each step builds on the previous.

## Critical Rules

1. **Architecture is fixed**: 1 producer + 2 consumers, 3 warpgroups, no transpose. Do NOT deviate.
2. **No transpose**: All transpose code must be deleted. If MN-major B doesn't work, debug the descriptor — never revert to transpose.
3. **No compiler dependence**: The kernel body is 100% inline PTX. Accumulators, descriptors, loop counters — all `.reg` declarations. No C++ `float` arrays, no compiler register allocation. The compiler compiles the asm block verbatim — it has zero discretion over register usage.
4. **Correctness gate**: `ik:bench matmul` must show all ✓ before committing.
5. **Debug forward**: On errors, iterate within the architecture. Try different LBO/SBO values, different phase parity formulas, different barrier counts. Never fall back to a simpler architecture.
6. **Commit to the architecture**: Explore this design fully. Don't abandon it after one error. Matmul is simple — wait, load, multiply, store. Debug each piece independently.
