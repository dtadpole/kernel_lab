# Profiling Guide

## When to Profile

Profile **selectively** — only after you have bench results AND have read the
code. NCU profiling is expensive (~30s per config per impl). Pick 1-2 configs:
- The config with the **largest gap** vs the best `ref-*` baseline
- One **representative** config (e.g., the most common batch/seq_len combo)

## How to Profile

```bash
# Profile the target impl on a specific config
.venv/bin/python -m cuda_exec.exec_cli exec.action=profile exec.kernel={kernel} exec.arch={arch} exec.impl={slug} exec.gpu={gpu} exec.run_tag={run_tag} exec.turn={turn} 'exec.configs=[{config_slug}]' exec.side=generated

# Profile the best ref-* impl for comparison
.venv/bin/python -m cuda_exec.exec_cli exec.action=profile exec.kernel={kernel} exec.arch={arch} exec.impl={slug} exec.gpu={gpu} exec.run_tag={run_tag} exec.turn={turn} 'exec.configs=[{config_slug}]' exec.side=reference
```

## Key NCU Metrics to Collect

- **Compute throughput** (SM %) — how busy are the SMs?
- **Memory throughput** (DRAM %) — how busy is memory?
- **Achieved occupancy** — active warps vs theoretical max
- **Warp stall reasons** — what are warps waiting on?
- **L1/L2 hit rates** — cache effectiveness
- **Instructions per cycle** — execution efficiency
- **Register usage** — registers per thread, spill loads/stores

## Bottleneck Classification

- **Compute-bound**: SM % is high, memory % is low
- **Memory-bound**: DRAM % is high, SM % is low
- **Latency-bound**: Both are low — look at warp stalls

## Assembly Examination

Review from the compile output:
- **SASS** — actual GPU instructions, look for inefficiencies
- **PTX** — compiler input, check for unnecessary barriers or redundant ops
- **Resource usage** — registers per thread, shared memory, spill bytes

Red flags:
- Register spills (spill stores/loads > 0)
- Poor instruction mix (too much control vs compute)
- Excessive barriers (`bar.sync`, `membar`)
- Shared memory bank conflicts
