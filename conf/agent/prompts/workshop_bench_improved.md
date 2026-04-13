BENCHMARK RESULT: IMPROVED

{bench_result_text}

★ NEW GEM: {gem_id} ({n_improved} configs improved)

### References
- Source snapshot: {impls_dir}/gen/{arch}/{kernel}/cuda/cuda.cu
- Results: {impls_dir}/gen-cuda/results.json
- Compile logs: {impls_dir}/gen-cuda/compile/
- Previous gem: {prev_gem_path}

### Action Required

Call submit_bench_reflection with three arguments:

submit_bench_reflection(
    gem_id="{gem_id}",
    gem_notes_md="...",
    reflection_md="..."
)

**gem_notes_md** — What you implemented (Markdown):
- What you changed in this iteration
- What you generated (architecture, key parameters)
- Core technical points (why this works)

**reflection_md** — Write in Markdown. At most 3 points.

What do you know now that you didn't know before this iteration?

Each point must be something you discovered by doing — not something
you could have read in a manual. Think about the entire process of
this iteration, not just the final result:

- What surprised you? Results you didn't predict, bugs with unexpected
  root causes, optimizations that helped or hurt more than expected,
  compiler behavior you didn't anticipate.
- What would you do differently? Looking back at your compilation
  attempts, debugging steps, and decision points — where did you
  spend time that could have been avoided?
- What should future work remember? If someone encounters a similar
  situation, what knowledge would save them time or prevent mistakes?

Be specific: name the NCU metric, the ptxas warning, the per-config
performance delta, the line of code. These notes become part of a
permanent knowledge base for future kernel optimization work.
