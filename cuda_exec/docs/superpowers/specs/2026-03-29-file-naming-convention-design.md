# File Naming Convention for Reference and Generated Inputs

## Context

`cuda_exec` currently accepts any filename for reference (`.py`) and generated (`.cu`) entry files. This flexibility creates unpredictable artifact names (since the `.cu` stem becomes the artifact prefix) and makes it harder for callers to reason about the contract. Establishing a fixed naming convention simplifies the interface and stabilizes artifact paths.

## Convention

### Reference side

- `reference_files` **must** include a file with key exactly `reference.py` (no subdirectory prefix). This is the entry point.
- Additional helper files (any name/extension, subdirectories allowed) are permitted alongside it.
- The service validates this at the `/compile` endpoint and returns HTTP 400 if missing.

### Generated side

- `generated_files` **must** include exactly one `.cu` file, and its key **must** be exactly `generated.cu` (no subdirectory prefix).
- Additional helper files (`.h`, `.inc`, etc., subdirectories allowed) are permitted alongside it.
- The existing "exactly one `.cu` file" rule remains; the new rule adds a name constraint on top.

### Artifact naming impact

With the stem fixed to `generated`, compile artifacts become predictable:

| Before (variable stem) | After (fixed stem) |
|------------------------|--------------------|
| `artifacts/compile.attempt_001.vector_add_inline_ptx.bin` | `artifacts/compile.attempt_001.generated.bin` |
| `artifacts/compile.attempt_001.vector_add_inline_ptx.ptx` | `artifacts/compile.attempt_001.generated.ptx` |
| `artifacts/compile.attempt_001.vector_add_inline_ptx.cubin` | `artifacts/compile.attempt_001.generated.cubin` |
| `artifacts/compile.attempt_001.vector_add_inline_ptx.resource-usage.txt` | `artifacts/compile.attempt_001.generated.resource-usage.txt` |
| `artifacts/compile.attempt_001.vector_add_inline_ptx.nvdisasm.sass` | `artifacts/compile.attempt_001.generated.nvdisasm.sass` |

### Error messages

Missing `reference.py`:
```json
HTTP 400
{"detail": "reference_files must include a file named reference.py as the entry point. ..."}
```

`.cu` file not named `generated.cu`:
```json
HTTP 400
{"detail": "the .cu entry file in generated_files must be named generated.cu. ..."}
```

## Code changes

### Validation (tasks.py)

1. **`_pick_single_cuda_source`**: After finding exactly one `.cu` file, add check that `path.name == "generated.cu"`. Return 400 with guidance if not.
2. **`run_compile_task`**: After writing reference files, validate that at least one key in `reference_files` has basename `reference.py`. Return 400 with guidance if not.

### Reference discovery (scripts/evaluate.py)

- **`_load_reference_entry`**: Change from `rglob("*.py")` finding exactly one `.py` to `rglob("reference.py")` finding exactly one match. Error message updated accordingly.
- **`scripts/profile.py`**: (removed — profile is now NCU-only via `profile.sh` and `tasks.py`).

### Test fixtures

| Current name | New name |
|-------------|----------|
| `fixtures/reference/vector_add_cutedsl.py` | `fixtures/reference/reference.py` |
| `fixtures/generated/vector_add_inline_ptx.cu` | `fixtures/generated/generated.cu` |
| `fixtures/generated/vector_add_runtime_launch.cu` | `fixtures/generated/generated_runtime_launch.cu` |

- The inline-PTX fixture becomes the default `generated.cu`.
- The runtime-launch fixture keeps a distinct name since it is swapped in by `_compile_payload_runtime_launch()` — the test helper overrides the `generated_files` key to `"generated.cu"` pointing to this file's content.

### Test assertions

- Assertions checking for `.visible .entry vector_add_inline_ptx` in PTX content change to `.visible .entry generated`.
- Assertions checking artifact paths containing `vector_add_inline_ptx` change to `generated`.

### Documentation

- `DESIGN.md`: Add convention rules to the compile input section.
- `CLAUDE.md`: Add bullet about entry file naming convention.
- `models.py`: Update CompileRequest docstring.

## Files to modify

| File | Action |
|------|--------|
| `cuda_exec/tasks.py` | Add `reference.py` and `generated.cu` name validation |
| `cuda_exec/scripts/evaluate.py` | Change reference discovery to find `reference.py` by name |
| `cuda_exec/main.py` | No code changes needed |
| `cuda_exec/models.py` | Update docstring |
| `cuda_exec/DESIGN.md` | Document convention |
| `cuda_exec/CLAUDE.md` | Add convention bullet |
| `tests/fixtures/reference/vector_add_cutedsl.py` | Rename to `reference.py` |
| `tests/fixtures/generated/vector_add_inline_ptx.cu` | Rename to `generated.cu` |
| `tests/fixtures/generated/vector_add_runtime_launch.cu` | Rename to `generated_runtime_launch.cu` |
| `tests/test_e2e_service.py` | Update fixture refs, payload keys, assertions |
