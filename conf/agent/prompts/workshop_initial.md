Run tag for this session: {run_tag}
Working directory: ~/kernel_lab
Python: .venv/bin/python (ALWAYS use this, never search for other venvs)
Kernel: {kernel}
GPU: {gpu}

Use these for ALL exec commands:
  exec.run_tag={run_tag}
  exec.gpu={gpu}

Example:
  .venv/bin/python -m cuda_exec.exec_cli exec.action=compile exec.kernel={kernel} exec.arch={arch} exec.impl=gen-cuda exec.gpu={gpu} exec.run_tag={run_tag}

---

{task}

---

IMPORTANT: Your ik:exec trial results are preliminary — only the formal
benchmark (request_formal_bench) produces official results. Call
request_formal_bench after your code compiles and passes correctness.
Benchmark early and often. If it shows no improvement, iterate. If it
improves, a new gem is recorded.

Keep optimizing until the formal benchmark shows improvement or you
exhaust your ideas. Do not stop after a single attempt.

REMINDER: Follow THE OPTIMIZATION LOOP (Phases 1-7) in your system prompt.
You MUST complete Phase 1 (Understand — read code, profile reference),
Phase 2 (Analyze — classify bottleneck), and Phase 3 (Brainstorm — 
data-backed ideas) BEFORE writing any kernel code.
Do NOT skip straight to coding. Research first, code second.
