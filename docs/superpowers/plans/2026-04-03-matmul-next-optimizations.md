# Matmul Next Optimizations Plan

## Current State (V10)

- **752 TFLOPS peak at 8192** (102% cuBLAS), median ~720-740
- Architecture: 3 WGs (1 producer + 2 consumers), persistent, SMEM coalesced epilogue
- 154 regs, 0 spills, no C7515

## Failed Attempts This Session

| Attempt | Result | Learning |
|---------|--------|----------|
| st.global.cs stores | -5% | L2 write-coalescing valuable |
| group_m=8 vs 16 | 0% | Identical peak with persistent |
| wait_group_1 pipeline | -7.5% | Starves producer in 3-WG model |
| 128 SM grid | -3% | 4 idle SMs = wasted capacity |
| Continuous phase cycling | 0% | Barrier re-init overhead negligible |
| Cluster multicast (2×1) | -5% | Cluster sync overhead > B bandwidth savings |

## Next: Ping-Pong Scheduling

The single most impactful remaining optimization. Expected gain: 5-10%.

### Architecture

Current (cooperative):
```
Consumer0: [MMA tile 0] [epilogue 0] [MMA tile 1] [epilogue 1] ...
Consumer1: [MMA tile 0] [epilogue 0] [MMA tile 1] [epilogue 1] ...
           ^^^^^^^^^^^^              ^^^^^^^^^^^^
           tensor cores busy         tensor cores busy
                        ^^^^^^^^^^^^              ^^^^^^^^^^^^
                        tensor cores IDLE          tensor cores IDLE
```

Ping-pong:
```
Consumer0: [MMA tile 0] [epilogue 0] [MMA tile 2] [epilogue 2] ...
Consumer1:              [MMA tile 1] [epilogue 1] [MMA tile 3] [epilogue 3] ...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           tensor cores ALWAYS busy (one consumer's epilogue overlaps other's MMA)
```

### Implementation Requirements

1. **Ordered sequence barriers**: Consumer0 and Consumer1 alternate who does MMA
   vs epilogue. Use `bar.sync N, 128` with different barrier IDs per consumer.

2. **Per-consumer tile assignment**: Consumer0 gets even tiles, Consumer1 gets
   odd tiles. The producer loads for both, alternating.

3. **Separate SMEM for each consumer's epilogue**: Each consumer writes its
   64-row output to a different SMEM region. Need 2 × 64 × 264 × 2 = 67KB
   total for epilogue buffers.

4. **Double the producer work**: Producer must load A/B for BOTH consumers'
   tiles. With 4 stages, this means 8 stage buffers or time-sharing 4 stages
   between two consumers.

### Reference

- CUTLASS `sm90_gemm_tma_warpspecialized_pingpong.hpp`
- Pranjal's `matmul_12.cuh` (stmatrix + padded TMA S2G + ping-pong)
- PyTorch blog: "CUTLASS Ping-Pong GEMM Kernel"

### PTX for Ordered Sequence Barrier

```ptx
bar.sync barID, threadCount;  -- sync within consumer warpgroup
```

Each consumer uses its own barrier ID (e.g., Consumer0 uses bar 2, Consumer1
uses bar 3). The ordering ensures:
- Consumer0 does MMA for tile T → commits → Consumer1 can start tile T's epilogue
- Consumer1 does MMA for tile T+1 → commits → Consumer0 can start tile T+1's epilogue

### Key Constraint

Ping-pong requires the SMEM layout to support TWO simultaneous output tiles
(one being written by MMA, one being read for epilogue). With 197KB SMEM for
mainloop and 67KB for epilogue, total = 264KB > 228KB. Need to either:
- Reduce mainloop stages from 4 to 3 (saves 49KB)
- Use smaller epilogue buffer (no padding)
- Reuse mainloop SMEM for one consumer's epilogue

### Estimated Complexity

High — this is essentially a rewrite of the tile loop and consumer logic.
Estimate 300-500 lines of changes. Should be done in a dedicated session.
<<<<<<< HEAD

## Attempted in this session

### Cluster Multicast (2×1)
- **Implemented** — correct output achieved
- **Result**: -5% regression at 8192
- **Root cause**: `cluster_arrive/cluster_wait` at tile boundaries + cross-CTA
  `mbarrier_arrive_remote` latency exceeds B bandwidth savings
- **Key learnings**:
  - Must use `shared::cluster` state space for ALL TMA in cluster kernels (not `shared::cta`)
  - Need `cluster_arrive/wait` between barrier re-init and cross-CTA pre-signaling
  - Correct PTX: `%cluster_ctarank` for CTA rank, `mapa.shared::cluster` for remote arrive
  - Cluster-coordinated tile assignment needed for B sharing (same tile_n in cluster)

### stmatrix Epilogue
- **Attempted** — compilation succeeded but correctness failed
- **Root cause**: stmatrix address computation for row-major output differs from
  Pranjal's column-major version. The `.trans` flag transposes relative to the
  write layout, so the address offsets need to be recalculated.
- **Next step**: Study the stmatrix m8n8.x4.trans register-to-SMEM mapping using
  the PTX ISA documentation and write a small test kernel to verify the mapping
  before integrating into the matmul kernel.
=======
>>>>>>> 232fb28 (docs: matmul ping-pong scheduling plan + cluster multicast findings)
