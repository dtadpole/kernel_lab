Optimize the Flash Attention 4 kernel for SM90.
Write your kernel to ~/kernel_lab_kb/runs/<run_tag>/gen/sm90/fa4/cuda/cuda.cu.
Use ik:exec to compile, trial, and profile.
Call request_formal_bench as soon as your code compiles and passes
correctness — do not wait for perfection.
Target: beat the current best gem.

IMPORTANT: Write raw CUDA/PTX code only. Do NOT use CUTLASS, cuDNN, cuBLAS,
or any high-level GPU library. Implement WGMMA, TMA, mbarrier, and all
optimization logic yourself.

Start by writing a plan (architecture, tile sizes, scheduling strategy)
BEFORE writing kernel code. Then implement step by step — compile after
each change.
