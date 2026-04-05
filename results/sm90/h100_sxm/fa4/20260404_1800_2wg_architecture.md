# FA4 2-Consumer-WG Architecture — 427–577 TFLOPS (+5–8%)

## Hardware

- **GPU:** NVIDIA H100 SXM5 96GB HBM3
- **Architecture:** SM 9.0a (Hopper)
- **Peak BF16:** 800 TFLOPS (650W TDP)
- **Host:** devvm8490 (h8_3), GPU 4, CUDA 12.8

## Objective

Replace single-WG kernel with 2-consumer-WG architecture matching FA4 CuTe
DSL's approach. 384 threads = 1 producer WG + 2 consumer WGs. Each consumer
processes m=64 rows of Q, doubling MMA parallelism. Dedicated producer feeds
K/V tiles via TMA pipeline.

## Architecture

| Parameter | 1-WG (previous) | 2-WG (new) |
|-----------|-----------------|------------|
| Threads | 128 (1 WG) | 384 (3 WGs) |
| Warps | 4 | 12 |
| BLOCK_Q | 64 | 128 (2 × m64) |
| BLOCK_KV | 128 | 128 |
| Producer | Thread 0 (inline) | Dedicated WG0 (warps 0-3) |
| Consumers | All 128 threads | WG1 (warps 4-7) + WG2 (warps 8-11) |
| Registers | 168/thread | Producer: 24, Consumer: 240 |
| SMEM | 112KB | 160KB |
| K pipeline | Double-buffered, simple mbar | Double-buffered, full+empty mbar |
| V pipeline | Single buffer | Double-buffered, full+empty mbar |
| Scheduler barriers | None | bar 2 (WG1), bar 3 (WG2), 256 threads |

### Key Implementation Details

- **setmaxnreg**: Producer decreases to 24, consumers increase to 240
- **Pipeline protocol**: K and V use full+empty mbarrier pairs. Empty barriers
  initialized with arrive_count=2 (one per consumer WG), pre-arrived twice
  during init to mark stages as initially free
- **Scheduler barriers**: FA4-style named barriers. WG1 primes bar 2 at init
  (mma_init pattern). Each WG syncs on its own barrier and arrives on the
  other WG's barrier after QK GEMM
- **Consumer WG mapping**: WG1 uses Q groups 0-7 (rows 0-63), WG2 uses
  Q groups 8-15 (rows 64-127). Both share K/V tiles

## Performance Comparison

```
┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────┬──────────┐
│ H100 SXM (h8_3)        │      cuDNN       │  1-WG BLK128     │  2-WG (new)      │ Speedup  │ vs cuDNN │
│ GPU4, CUDA 12.8        │  TFLOPS   (ms)   │  TFLOPS   (ms)   │  TFLOPS   (ms)   │ 1→2 WG   │ 2-WG     │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ causal-b8-s4096        │  545.5  (1.008)  │  407.4  (1.350)  │  427.4  (1.286)  │  +4.9%   │  0.78×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ causal-b4-s8192        │  595.5  (1.846)  │  449.8  (2.445)  │  480.4  (2.289)  │  +6.8%   │  0.81×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ causal-b2-s16384       │  621.3  (3.539)  │  471.6  (4.663)  │  510.9  (4.304)  │  +8.3%   │  0.82×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ nc-b8-s4096            │  618.5  (1.778)  │  499.4  (2.202)  │  526.8  (2.087)  │  +5.5%   │  0.85×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ nc-b4-s8192            │  639.7  (3.438)  │  527.2  (4.171)  │  559.2  (3.932)  │  +6.1%   │  0.87×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ nc-b2-s16384           │  632.7  (6.952)  │  536.5  (8.197)  │  576.7  (7.626)  │  +7.5%   │  0.91×   │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────┼──────────┤
│ % of peak (best cfg)   │      80.0%       │      67.1%       │      72.1%       │          │  800 TF  │
└────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────┴──────────┘
```

## What Worked

1. **Dedicated producer WG**: Continuous TMA feeding without interfering with MMA
2. **Register redistribution**: 240 regs/consumer gives compiler scheduling freedom
3. **Double-buffered K AND V**: Both pipelines run independently via full+empty mbar
4. **Scheduler barriers**: FA4-style bar 2/3 prevent warp scheduler thrashing
5. **mma_init pattern**: WG1 pre-arrives on bar 2 to bootstrap the round-robin

## Previous 2-WG Failure (Fixed)

The earlier 2-WG attempt (20260403) deadlocked because:
- Used simple bar.sync instead of arrive/sync protocol
- Single K/V buffer caused race conditions
- No empty-barrier pipeline for producer backpressure

This implementation fixes all three issues with the FA4 pipeline protocol.

## Remaining Gap to cuDNN (~1.1–1.2×)

| Source | Impact | Fix |
|--------|--------|-----|
| Intra-WG QK/PV overlap | ~1.1× | Need to overlap QK[n+1]/PV[n] between WGs |
| TMA O store | ~1.02× | Replace STG with TMA S2G bulk store |
| Producer scheduling | ~1.05× | Interleave K[n]/V[n-1] loads (K leads V) |
