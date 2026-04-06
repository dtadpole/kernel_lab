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
