Run tag for this session: {run_tag}
Use this run_tag for ALL ik:exec commands (exec.run_tag={run_tag}).
Scratch directory: ~/.cuda_exec/{run_tag}/
Kernel: {kernel}

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
