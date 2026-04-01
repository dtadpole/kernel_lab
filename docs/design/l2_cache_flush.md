# L2 Cache Flush — Benchmark Harness Design

## Problem

GPU kernels access global memory through the L2 cache (96 MB on RTX PRO 6000 Blackwell).
When benchmarking a kernel over multiple trials, L2 retains data from previous runs.
This makes trials after the first artificially fast — they hit L2 instead of DRAM,
producing latency numbers that don't reflect real-world (cold-cache) performance.

Without L2 flush, a kernel might appear 10-30% faster than its true sustained throughput,
because real workloads rarely enjoy a fully warm L2 for the exact data they need.

## Technique: Zero-Fill Eviction

The standard approach (used by Triton `do_bench`, NVBench, and our harness):

1. **Allocate** a device buffer equal to the L2 cache size
2. **Zero-fill** it before each timed trial — this walks the entire L2, evicting all prior data
3. **Synchronize** to ensure the flush completes before timing starts

### Why this works

The L2 cache is physically indexed. Writing `L2_size` bytes of contiguous data
forces every cache line to be replaced. After the flush, the kernel under test
starts from a cold L2 — every global memory access goes to DRAM first.

### Why zero-fill (not random-fill)

`memset` / `zero_()` generates the simplest possible memory access pattern
(sequential writes, no read-modify-write). It completes quickly and predictably.
The goal is eviction, not exercising the cache — simpler is better.

## Implementation

### C++ Harness (`eval_harness.cu`)

```cpp
// Setup (once)
int l2_attr = 0;
cudaDeviceGetAttribute(&l2_attr, cudaDevAttrL2CacheSize, 0);
size_t l2_size = (size_t)(l2_attr > 0 ? l2_attr : 0);
void* l2_flush_buf = nullptr;
if (l2_size > 0)
    cudaMalloc(&l2_flush_buf, l2_size);

// Per-trial (before timing)
if (l2_flush_buf)
    cudaMemsetAsync(l2_flush_buf, 0, l2_size, stream);
cudaStreamSynchronize(stream);   // ensure flush completes

// Timed region
cudaEventRecord(start_ev, stream);
kernel_run(...);
cudaEventRecord(end_ev, stream);
```

### Python Harness (`eval_support.py`)

```python
# Setup (once)
l2_size = torch.cuda.get_device_properties(device).L2_cache_size
l2_flush = torch.empty(l2_size, dtype=torch.uint8, device=device)

# Per-trial (before timing)
l2_flush.zero_()
torch.cuda.synchronize(device=device)

# Timed region
start_ev.record()
output = model(*inputs)
end_ev.record()
```

## Full Per-Trial Measurement Protocol

Both harnesses follow the same sequence before each timed trial:

```
┌─────────────────────────────────┐
│  1. L2 cache flush              │  Evict all cached data
│     (zero-fill L2-sized buffer) │
├─────────────────────────────────┤
│  2. Input re-randomization      │  Prevent content-based caching;
│     (normal_ / fill_random)     │  ensures kernel processes fresh data
├─────────────────────────────────┤
│  3. GPU synchronize             │  Barrier: steps 1-2 must complete
│     (cudaStreamSynchronize /    │  before timing starts
│      torch.cuda.synchronize)    │
├─────────────────────────────────┤
│  4. Record start event          │  ← Timing begins
│  5. Execute kernel              │
│  6. Record end event            │  ← Timing ends
│  7. Synchronize end event       │  Wait for kernel completion
└─────────────────────────────────┘
```

Steps 1-3 run **outside** the timed region. They add wall-clock time between
trials but do not affect the measured kernel latency.

## Why Each Step Matters

| Step | Without it | Impact |
|------|-----------|--------|
| L2 flush | Trials 2+ see warm L2 | 10-30% faster than real-world |
| Input re-randomization | Same data every trial | Kernel may cache transposed/preprocessed copies |
| GPU synchronize | Flush overlaps with timed kernel | Measured latency includes partial flush overhead |

## Hardware Details: RTX PRO 6000 Blackwell

| Property | Value |
|----------|-------|
| L2 cache size | 96 MB |
| L2 cache line | 128 bytes |
| DRAM bandwidth | 1,536 GB/s (HBM3e-equivalent) |
| L2 flush time (96 MB @ ~1.5 TB/s) | ~0.06 ms |

The flush itself costs ~0.06 ms — negligible compared to kernel execution at
large matrix sizes, but non-trivial relative to small-size kernels (0.01 ms).
This is acceptable: we measure the kernel's cold-cache behavior, which is the
realistic deployment scenario.

## Reference Implementations

- **Triton `do_bench`**: [`triton/testing.py`](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py) — `cache.zero_()`
- **NVBench**: [`nvbench/detail/l2flush.cuh`](https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh) — `cudaMemsetAsync`
- **cutlass profiler**: Same pattern, allocate + memset L2-sized buffer
