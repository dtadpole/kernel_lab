# cuda_exec evaluate/profile status

## Completed

1. Environment readiness
   - `cuda_exec/.venv` is the working runtime environment for `torch`, `cutlass`, and `cutlass.cute`
   - reference fixture contract test now skips cleanly when `torch` / `cutlass.cute` / CUDA are unavailable

2. Reference contract
   - `cuda_exec/tests/fixtures/reference/vector_add_cutedsl.py` now exports:
     - `Model`
     - `get_inputs(config)`
     - `get_init_inputs()`
   - the vector-add reference fixture now genuinely launches a CuTe DSL kernel through a `@cute.jit` host launcher

3. Evaluate runtime
   - `cuda_exec/scripts/evaluate.py` runs reference + generated sides
   - evaluate persists structured comparison artifacts per config
   - `/evaluate` returns per-config correctness/performance/artifacts/logs

4. Profile runtime
   - `cuda_exec/scripts/profile.py` now supports:
     - `reference_only`
     - `generated_only`
     - `dual`
   - profile persists structured summary artifacts per config
   - `/profile` returns per-config summary/reference/generated/artifacts/logs

5. Integration coverage
   - end-to-end suite currently passes with the updated evaluate/profile behavior

## Remaining real follow-ups

- [ ] Decide whether `profile` should keep the current behavior-first comparison runtime long-term,
      or also expose a separate NCU-backed profiler path in parallel.

- [ ] Decide whether profile summaries should return richer structured side-by-side fields
      for `reference` and `generated` directly in the public response model, instead of only
      the top-level `summary` plus kept artifact payload.

- [x] Dedicated tests for `reference_only` and `generated_only` profile modes are now in place,
      including side-specific artifact/log assertions in the public response payloads.
