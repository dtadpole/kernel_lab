Run tag for this session: {run_tag}
Working directory: /home/zhenc/kernel_lab
Python: .venv/bin/python (ALWAYS use this, never search for other venvs)
Kernel: {kernel}
GPU: {gpu}

Use these for ALL exec commands:
  exec.run_tag={run_tag}
  exec.gpu={gpu}

Example:
  .venv/bin/python -m cuda_exec.exec_cli exec.action=compile exec.kernel={kernel} exec.arch=sm90 exec.impl=gen-cuda exec.gpu={gpu} exec.run_tag={run_tag}

---

{task}

---

IMPORTANT: Your ik:exec trial results are preliminary — only the formal
benchmark (request_formal_bench) produces official results. Call
request_formal_bench(kernel="{kernel}", reason="...") as soon as your code
compiles and passes correctness. Do not wait for perfection — benchmark
early and often. If it shows no improvement, iterate. If it improves,
a new gem is recorded.

Keep optimizing until the formal benchmark shows improvement or you
exhaust your ideas. Do not stop after a single attempt.

REMINDER: You MUST output a written plan BEFORE writing any kernel code.
Do not think silently for a long time — output your plan as text first,
then implement step by step.
