# kernel_lab_kb Design

## Purpose

Separate repo for storing important benchmark trajectories — source files, compile info, and per-config results for each benchmark run.

## Repo Structure

```
kernel_lab_kb/
  trajectories/
    <kernel>/<arch>/<impl_slug>/<YYYYMMDD_HHMMSS>/
      sources/
        reference/        # copied ref source files
        generated/        # copied gen source files
      compile/
        ptxas.txt         # register count, spills, barriers
        resource_usage.txt # REG/SMEM/STACK/CONSTANT
      results.json        # structured benchmark results (compact)
      report.md           # human-readable summary table
```

## results.json Schema

```json
{
  "kernel": "matmul",
  "arch": "sm90",
  "impl": "gen-cuda",
  "timestamp": "2026-04-04T21:00:35",
  "git_commit": "a51b864",
  "git_branch": "main",
  "device": "NVIDIA H100 SXM5",
  "gpu_index": 4,
  "compile": {
    "ok": true,
    "registers": 14,
    "spill_stores": 0,
    "spill_loads": 0,
    "shared_mem": 0,
    "barriers": 0
  },
  "configs": {
    "mat-256x256": {
      "correctness": true,
      "ref_median_ms": 0.033,
      "gen_median_ms": 0.011,
      "speedup": 3.07,
      "ref_latency": {"min": 0.030, "median": 0.033, "max": 0.038, "mean": 0.034, "std": 0.002},
      "gen_latency": {"min": 0.010, "median": 0.011, "max": 0.013, "mean": 0.011, "std": 0.001}
    }
  }
}
```

## report.md Format

```markdown
# matmul/sm90/gen-cuda — 2026-04-04 21:00:35

**Device:** NVIDIA H100 SXM5 (GPU 4)
**Commit:** a51b864
**Compile:** 14 regs, 0 spills, 0B shared

| Config | Correct | Ref (ms) | Gen (ms) | Speedup |
|--------|---------|----------|----------|---------|
| mat-256x256 | ✓ | 0.033 | 0.011 | 3.07× |
| mat-8192x8192 | ✓ | 1.491 | 1.464 | 1.02× |
```

## ik:bench Integration

After each benchmark run, ik:bench will:
1. Create the trajectory directory in kernel_lab_kb
2. Copy source files from data/ref/ and data/gen/
3. Extract ptxas + resource_usage from compile output
4. Generate results.json (compact, ~5KB)
5. Generate report.md (human-readable table)
6. Auto-commit + push to kernel_lab_kb

## Compare Mode

When a previous trajectory exists for the same kernel/arch/impl:
- Load the latest previous results.json
- Compare per-config latencies
- Report improvements/regressions in the table
- Add a "vs_previous" section to results.json with deltas
