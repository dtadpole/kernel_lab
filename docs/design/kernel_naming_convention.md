# Kernel Naming Convention — NCU Profile Filter Design

## Problem

NCU `--kernel-name` filter requires knowing the GPU kernel function name at
profile time. When kernel names change (e.g. `matmul_bf16_v2` → `mma_matmul_tma_big`),
the filter silently matches nothing — NCU reports "No kernels were profiled" and
exits with code 9. This failure is silent and wastes minutes of profiling time.

## Root Cause

The original filter was hardcoded: `regex:"matmul_bf16|cutlass"`. When kernel
names evolved, the filter was never updated. No mechanism enforced consistency
between kernel source and profile filter.

## Rejected Alternatives

### Extract kernel name from compile artifacts (cross-stage state)

Compile produces `resource-usage.txt` which contains mangled kernel names.
Profile stage could parse it and build the filter dynamically.

**Why rejected:** Introduces a fragile cross-stage dependency. The compile
artifact path, format, or cleanup policy could change. Any break in this
chain produces the same silent failure. State that flows between stages is
a persistent source of bugs — especially when stages can be re-run independently.

### Launch-count based profiling (skip warmup launches)

Use `--launch-skip N --launch-count 1` to profile by ordinal position
instead of name.

**Why rejected:** Brittle when kernels have multiple launches internally,
or when warmup count changes. Position-based addressing breaks silently.

## Solution: Kernel Family Name Convention

### Rule

> **Every user-authored GPU kernel function name MUST contain the kernel family
> name as a substring.**

The kernel family name is the value of `KERNEL=` passed to Make — the same
string used across compile, evaluate, and profile stages.

### Examples

| Family (`KERNEL=`) | Valid kernel names | Invalid |
|--------------------|--------------------|---------|
| `matmul` | `mma_matmul_tma_big`, `matmul_naive`, `matmul_v3` | `gemm_tma_big` |
| `vecadd` | `vecadd_bf16`, `vecadd_tiled` | `vector_add_bf16` |
| `fa4` | `fa4_fwd_kernel`, `fa4_bwd_dq` | `flash_attn_fwd` |

### NCU Filter

The Makefile profile targets use:

```makefile
--kernel-name 'regex:"$(KERNEL)"'
```

This matches any kernel whose demangled name contains the family string.
No cross-stage state. No hardcoded names. The filter is derived directly
from the Make variable that already flows through the entire pipeline.

### Compile-Time Verification

As a safety net, the compile stage should verify that at least one kernel
in the compiled binary matches the family name:

```makefile
# After compile step 3 (resource-usage dump), verify kernel naming convention
@if ! grep -q '$(KERNEL)' "$(RESOURCE_USAGE_FILE)"; then \
    echo "ERROR: no kernel function contains '$(KERNEL)' — violates naming convention" >&2; \
    echo "Found kernels:" >&2; \
    grep '^[[:space:]]*Function' "$(RESOURCE_USAGE_FILE)" >&2; \
    exit 1; \
fi
```

This catches violations at compile time, not minutes later during profiling.

### Helper Kernels

Helper kernels (e.g. `fill_random_bf16` in eval_harness.cu) are NOT required
to follow this convention. The NCU filter intentionally excludes them — only
kernels matching the family name are profiled.

## Integration Checklist

1. [ ] Update `profile-ncu-generated` to use `--kernel-name 'regex:"$(KERNEL)"'`
2. [ ] Update `profile-ncu-reference` similarly
3. [ ] Add compile-time kernel name verification after resource-usage dump
4. [ ] Remove any hardcoded kernel name filters
5. [ ] Document convention in AGENTS.md or CLAUDE.md for kernel authors
