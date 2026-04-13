Optimize the CUDA matmul kernel.
Write your kernel to ~/kernel_lab_kb/runs/<run_tag>/gen/<arch>/matmul/cuda/cuda.cu
where <arch> is auto-detected by exec.arch=auto (e.g. sm90, sm100).
Use ik:exec (with exec.arch=auto) to compile, trial, and profile.
Target: beat the current best gem in kernel_lab_kb.

IMPORTANT: Write raw CUDA/PTX code only. Do NOT use CUTLASS, cuDNN, cuBLAS,
or any high-level GPU library. Use the native tensor core instructions for
your GPU architecture (detect with exec.arch=auto).

A high-quality reference implementation may be available at:
  ~/kernel_lab_kb/runs/<run_tag>/pick/<arch>/matmul/cuda/cuda.cu
If it exists, read this code first — it represents the current peak performance.
Study its architecture (tile sizes, warp specialization, TMA, pipeline
strategy) and use it as a starting point for your optimization.

Follow THE OPTIMIZATION LOOP in your system prompt (Phases 1-7).
You MUST complete Phase 1 (Understand), Phase 2 (Analyze), and Phase 3
(Brainstorm) BEFORE writing any kernel code. This means:

1. Read the reference/pick code and NCU profile it to understand what
   the hardware is actually doing at 85-90% peak.
2. Search NVIDIA docs (ik:docs) and the web (WebSearch) for the target
   architecture's capabilities and scheduling strategies.
3. Brainstorm optimization ideas grounded in profile data and research.
4. THEN write a plan and implement.

Call request_formal_bench after your code compiles and passes correctness.
Benchmark early and often — but research first.
