# Analysis Guide

## Compare Profiles (gen vs ref)

Side-by-side the NCU metrics between the target impl and the best `ref-*`
baseline. Identify which hardware resource is the bottleneck (compute, memory,
or latency bound — see profiling-guide.md for classification).

## Consult NVIDIA Documentation (Source of Truth)

**Documentation is the ground truth. Always consult docs before forming
hypotheses.** Use two complementary channels:

**Local indexed docs** — `/ik:docs`:
- CUDA C++ Programming Guide (memory model, warp-level primitives, async copy)
- PTX ISA Reference (instruction semantics, latency, constraints)
- CUDA Best Practices Guide / Tuning Guide
- Architecture-specific features (e.g., SM90 WGMMA, SM120 tcgen05)

**Online NVIDIA docs** — web search:
- Search `site:docs.nvidia.com` or `site:developer.nvidia.com` for specific
  instructions, intrinsics, or hardware features
- PTX ISA changelog for new instructions on the target arch
- CUTLASS/CuTe source-level documentation on GitHub
- Nsight Compute metrics interpretation guides

**What to look for:**
- Exact instruction latencies and throughput for the target SM
- Memory hierarchy behavior (L1/L2 sector size, cache policies)
- Async copy / TMA programming constraints
- Warp scheduling and instruction interleaving rules
- Architecture-specific limitations or opportunities

When profiling data contradicts your assumptions, go back to docs to understand
why. The docs are authoritative — NCU data confirms or reveals, docs explain.

## Check Roofline

Read `docs/roofline/` specs for the target GPU. Calculate:
- Arithmetic intensity of the kernel (FLOPs / bytes transferred)
- Theoretical peak for this workload given compute vs memory bound
- Current achieved % of roofline ceiling
- Whether optimization should target compute, memory, or latency

## Search for External Insights

Use web search for **grounded, verifiable** information:
- NVIDIA GTC talks and whitepapers with published benchmarks
- Peer-reviewed papers with reproducible results
- Official NVIDIA blog posts with performance data
- CUTLASS / FlashAttention GitHub issues and commit messages

**Priority**: NVIDIA official docs > published papers > reproducible benchmarks
> blog posts > forum posts. Discard advice lacking measurable evidence.

## Study Previous Implementations and Results

**Previous implementations** — compare different implementations:
- Use `list_impls(kernel, arch)` to discover all available slugs
- Compare `ref-*` vs `gen-*` — what techniques does each use?
- Use `git log --oneline --all -- data/gen/{arch}/{kernel}/` for history
- Read previous versions: `git show <commit>:data/gen/{arch}/{kernel}/{name}.cu`

**Previous results** — `results/{arch}/{gpu_name}/{kernel}/` documents past
optimization attempts: what was tried, NCU data, what worked/didn't, and
architectural insights. Always read ALL results files before brainstorming.
Failed experiments are especially valuable — they document dead ends.
