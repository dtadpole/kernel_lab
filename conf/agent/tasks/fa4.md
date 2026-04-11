Optimize the Flash Attention 4 kernel for SM90.
Write your kernel to ~/kernel_lab_kb/runs/<run_tag>/gen/sm90/fa4/cuda/cuda.cu.
Use ik:exec to compile, trial, and profile.
Target: beat the current best gem.

IMPORTANT: Write raw CUDA/PTX code only. Do NOT use CUTLASS, cuDNN, cuBLAS,
or any high-level GPU library. Implement WGMMA, TMA, mbarrier, and all
optimization logic yourself.

Follow THE OPTIMIZATION LOOP in your system prompt (Phases 1-7).
You MUST complete Phase 1 (Understand), Phase 2 (Analyze), and Phase 3
(Brainstorm) BEFORE writing any kernel code. This means:

1. Read the reference implementations (.peak/ and data/ref/) to understand
   what 85-90% peak performance looks like — their architecture, tile sizes,
   WGMMA usage, TMA patterns, pipeline depth.
2. NCU profile the best reference (peak-cuda or ref-cutedsl) to see what
   the hardware is actually doing. This is not optional.
3. Search NVIDIA docs (ik:docs) and the web (WebSearch) for the target
   architecture's capabilities — what instructions are available, what
   scheduling strategies work, how CUTLASS/FlashAttention solve this.
4. Brainstorm optimization ideas grounded in profile data and research.
5. THEN write a plan and implement.

Call request_formal_bench after your code compiles and passes correctness.
Benchmark early and often — but research first.
