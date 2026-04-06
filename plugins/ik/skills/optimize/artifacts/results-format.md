# Results File Format

## File Location

```
results/{arch}/{gpu_name}/{kernel}/YYYYMMDD_HHMM_{description}.md
```

## Required Sections

### For Successful Optimizations

- Hardware specs
- Objective and approach
- Before/after performance table (all configs)
- Key changes made
- NCU metrics comparison (before vs after)
- What worked and why

### For Failed Attempts

- What was tried and why
- NCU profiling data before/after
- What didn't work and root cause analysis
- Architectural insights discovered
- Constraints/dead ends that should not be revisited

## Optimization Targets

Both must be met for a kernel to be considered "done":

1. **+10% above native baseline** — gen kernel's first gem (v001) must improve
   by ≥10% TFLOPS at the largest config by end of session.
2. **+10% above reference** — gen kernel must BEAT the best ref-* (e.g., cuBLAS)
   by ≥10% TFLOPS at the largest config.

If target 2 is unreachable (e.g., cuBLAS at 92% peak), aim to close within
5% first, then iterate. Document why the gap exists.
