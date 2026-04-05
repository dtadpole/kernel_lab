# SM90 Matmul — Warp Specialization via __noinline__ Functions

## Hardware
- NVIDIA H100 SXM (96GB HBM3, 650W TDP), GPU7
- CUDA 13.0 (nvcc V13.0.88), Driver 580.65.06

## Baseline
- 128×128 tile, 1 WG, 2-stage TMA pipeline: **454 TFLOPS** (56.8% of 800TF peak)

## Idea
Previous optimization ceiling report identified `setmaxnreg` C7507 as the
blocker. The root cause: ptxas ignores `setmaxnreg` inside conditional
branches (`if (wg_id == 0)`).

**Discovery**: Putting `setmaxnreg` in `__noinline__` device functions avoids
C7507 — ptxas can determine register count per function. Confirmed with both
CUDA 12.9 and 13.0:
```cpp
__device__ __noinline__ void producer_path() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
}
__device__ __noinline__ void consumer_path() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
}
```
No C7507/C7508 warning with this pattern.

## Result: BLOCKED by ptxas compile time

When the `__noinline__` functions contain WGMMA inline asm (64 accumulator
operands), ptxas compilation becomes extremely expensive:
- **21+ minutes** at 100% CPU
- **56 GB RAM** consumed
- Exceeds 300s timeout → compile fails

The complex inline asm in separate functions causes ptxas to explore an
exponentially larger register allocation space. With `__forceinline__`
(baseline), ptxas can see the full function and optimize in ~10 seconds.

## What This Means

1. `setmaxnreg` in `__noinline__` functions: **ptxas accepts it** (no C7507)
2. But `__noinline__` + WGMMA asm: **ptxas takes forever** (>21 min, 56GB)
3. `setmaxnreg` in branches: **ptxas rejects it** (C7507)
4. Single WG without setmaxnreg: **works** but limited to ~454 TFLOPS

## Remaining Options

1. **Compile with `--maxrregcount`** to globally limit registers (may help
   ptxas converge faster with `__noinline__` pattern)
2. **Use `.maxnreg` PTX directive** in the generated PTX before ptxas
   (post-process the PTX file)
3. **CUTLASS 3.x CollectiveBuilder** which generates warp-specialized code
   through C++ templates (avoids inline asm entirely)
4. **Two separate kernels** — producer-only and consumer-only, launched as
   cooperative groups
