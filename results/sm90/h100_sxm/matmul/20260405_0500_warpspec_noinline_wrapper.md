# SM90 Matmul — __noinline__ setmaxnreg Wrapper Attempt

## Hardware
- H100 SXM (96GB, 650W), GPU4, CUDA 13.0, Driver 580.65.06

## Baseline
- 1 WG, 128×128, 3-stage TMA pipeline: **469 TFLOPS** at 8192

## Idea
Put ONLY `setmaxnreg` in `__noinline__` wrapper functions (trivial, 1 asm
instruction each), keep WGMMA `__forceinline__` in the main kernel body.
This avoids both C7507 (setmaxnreg in branch) and ptxas explosion
(WGMMA in __noinline__).

## Test Result
- Trivial test kernel: `__noinline__` setmaxnreg wrappers **compile instantly,
  no C7507** ✓
- Real kernel: `__noinline__` wrappers still produce **C7507** ✗

## Why It Failed
ptxas analyzes the CALLING context, not just the callee. When
`_consumer_set_regs()` is called from inside `if (wg_id != 0)`, ptxas
still sees a conditional `setmaxnreg` and ignores it. The `__noinline__`
boundary doesn't isolate register analysis.

## Performance With C7507 (setmaxnreg ignored)
- 3 WG warp-specialized: **318 TFLOPS** (regression from 469)
- The 128 idle producer threads waste 90 regs × 128 = 11520 registers
- With setmaxnreg properly honored (40 regs producer), savings would be
  (90-40) × 128 = 6400 regs → more for consumers

## CuTe DSL Architecture (for comparison)
- 128×256 tile, 2 cooperative warpgroups, 4-stage pipeline
- TMA G2S + TMA S2G (store via TMA too)
- Warp specialization via CuTe DSL pipeline framework (bypasses ptxas)
- Result: **730 TFLOPS** at 8192 (91% of peak)

## Conclusion
All hand-written CUDA approaches to `setmaxnreg` are blocked by CUDA 13.0
ptxas. The CuTe DSL framework uses a different compilation path that
bypasses ptxas's register analysis limitations. To achieve >500 TF in
hand-written CUDA on CUDA 13.0, need an approach that doesn't rely on
`setmaxnreg` at all.
