# File Naming Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce that reference entry files must be named `reference.py` and generated entry files must be named `generated.cu` (both at root level, no subdirectory prefix), and update all artifacts/tests/docs accordingly.

**Architecture:** Add name validation to the existing `_pick_single_cuda_source` function in tasks.py and a new `_validate_reference_entry` check. Update `evaluate.py` to find `reference.py` by name instead of globbing for any `.py`. Rename test fixtures to match the convention.

**Tech Stack:** Python, FastAPI, bash (compile.sh unchanged — it receives paths, doesn't discover files)

---

### Task 1: Rename test fixtures

**Files:**
- Rename: `tests/fixtures/reference/vector_add_cutedsl.py` → `tests/fixtures/reference/reference.py`
- Rename: `tests/fixtures/generated/vector_add_inline_ptx.cu` → `tests/fixtures/generated/generated.cu`
- Rename: `tests/fixtures/generated/vector_add_runtime_launch.cu` → `tests/fixtures/generated/generated_runtime_launch.cu`

- [ ] **Step 1: Rename the three fixture files**

```bash
cd /home/centos/kernel_lab/cuda_exec
git mv tests/fixtures/reference/vector_add_cutedsl.py tests/fixtures/reference/reference.py
git mv tests/fixtures/generated/vector_add_inline_ptx.cu tests/fixtures/generated/generated.cu
git mv tests/fixtures/generated/vector_add_runtime_launch.cu tests/fixtures/generated/generated_runtime_launch.cu
```

- [ ] **Step 2: Commit fixture renames**

```bash
git commit -m "chore: rename test fixtures to match reference.py / generated.cu convention"
```

---

### Task 2: Add generated.cu name validation in tasks.py

**Files:**
- Modify: `cuda_exec/tasks.py:57-93` (function `_pick_single_cuda_source`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e2e_service.py` in `CudaExecE2ETest`, after `test_compile_rejects_generated_files_without_a_cu_file`:

```python
def test_compile_rejects_cu_file_not_named_generated_cu(self) -> None:
    payload = self._compile_payload(turn=152)
    content = payload["generated_files"]["generated.cu"]
    payload["generated_files"] = {"my_kernel.cu": content}
    status, body = self.service.post_json("/compile", payload)
    self.assertEqual(status, 400)
    self.assertIn("detail", body)
    self.assertIn("must be named generated.cu", body["detail"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/centos/kernel_lab
uv run --with pytest -- python -m pytest cuda_exec/tests/test_e2e_service.py -v -k test_compile_rejects_cu_file_not_named_generated_cu
```

Expected: FAIL — currently `my_kernel.cu` is accepted (status 200, not 400).

- [ ] **Step 3: Add name validation to `_pick_single_cuda_source`**

In `cuda_exec/tasks.py`, after the existing check at line 77 (`if len(generated_cu) == 1: return generated_cu[0]`), add a name check before the return:

```python
if len(generated_cu) == 1:
    if generated_cu[0].name != "generated.cu":
        raise HTTPException(
            status_code=400,
            detail=(
                "the .cu entry file in generated_files must be named generated.cu. "
                "Rename your CUDA source to generated.cu and resubmit. "
                "Additional header or helper files may use any name."
            ),
        )
    return generated_cu[0]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/centos/kernel_lab
uv run --with pytest -- python -m pytest cuda_exec/tests/test_e2e_service.py -v -k test_compile_rejects_cu_file_not_named_generated_cu
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cuda_exec/tasks.py cuda_exec/tests/test_e2e_service.py
git commit -m "feat: enforce generated.cu naming convention for .cu entry file"
```

---

### Task 3: Add reference.py name validation in tasks.py

**Files:**
- Modify: `cuda_exec/tasks.py:518-522` (in `run_compile_task`, after `_write_input_files`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e2e_service.py` in `CudaExecE2ETest`:

```python
def test_compile_rejects_reference_files_without_reference_py(self) -> None:
    payload = self._compile_payload(turn=153)
    content = payload["reference_files"]["reference.py"]
    payload["reference_files"] = {"my_model.py": content}
    status, body = self.service.post_json("/compile", payload)
    self.assertEqual(status, 400)
    self.assertIn("detail", body)
    self.assertIn("must include a file named reference.py", body["detail"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/centos/kernel_lab
uv run --with pytest -- python -m pytest cuda_exec/tests/test_e2e_service.py -v -k test_compile_rejects_reference_files_without_reference_py
```

Expected: FAIL — currently any `.py` name is accepted.

- [ ] **Step 3: Add reference.py validation in `run_compile_task`**

In `cuda_exec/tasks.py`, inside `run_compile_task`, right after the line that writes reference files and before `_pick_single_cuda_source`:

```python
copied_reference = _write_input_files(reference_files, workspace_path / "inputs" / "reference")
copied_generated = _write_input_files(generated_files, workspace_path / "inputs" / "generated")

# --- NEW: validate reference entry name ---
if not any(path.name == "reference.py" for path in copied_reference):
    raise HTTPException(
        status_code=400,
        detail=(
            "reference_files must include a file named reference.py as the entry point. "
            "Rename your reference module to reference.py and resubmit. "
            "Additional helper files may use any name."
        ),
    )

source = _pick_single_cuda_source(copied_generated, copied_reference)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/centos/kernel_lab
uv run --with pytest -- python -m pytest cuda_exec/tests/test_e2e_service.py -v -k test_compile_rejects_reference_files_without_reference_py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cuda_exec/tasks.py cuda_exec/tests/test_e2e_service.py
git commit -m "feat: enforce reference.py naming convention for reference entry file"
```

---

### Task 4: Update evaluate.py reference discovery

**Files:**
- Modify: `cuda_exec/scripts/evaluate.py:26-32` (function `_load_reference_entry`)

- [ ] **Step 1: Update `_load_reference_entry` to find by name**

In `cuda_exec/scripts/evaluate.py`, replace the function:

```python
def _load_reference_entry(reference_root: Path) -> Path:
    candidates = sorted(reference_root.rglob("reference.py"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"reference execution requires exactly one reference.py under {reference_root}; found {len(candidates)}"
        )
    return candidates[0]
```

- [ ] **Step 2: Run evaluate-related tests**

```bash
cd /home/centos/kernel_lab
uv run --with pytest -- python -m pytest cuda_exec/tests/test_e2e_service.py -v -k "test_evaluate"
```

Expected: PASS (both `test_evaluate_endpoint_accepts_slug_keyed_configs` and `test_evaluate_cli_runs_standalone_after_compile`)

- [ ] **Step 3: Commit**

```bash
git add cuda_exec/scripts/evaluate.py
git commit -m "feat: change reference discovery to find reference.py by name"
```

---

### Task 5: Update test payloads and assertions

**Files:**
- Modify: `cuda_exec/tests/test_e2e_service.py`

This task updates all test code that references the old fixture filenames or the old `.cu` stem in assertions.

- [ ] **Step 1: Update `_compile_payload` helper**

Change the file keys from subdirectory-prefixed old names to root-level convention names:

```python
def _compile_payload(self, turn: int) -> dict:
    return {
        "metadata": self._metadata(turn),
        "timeout_seconds": 20,
        "reference_files": {
            "reference.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8")
        },
        "generated_files": {
            "generated.cu": (FIXTURES / "generated" / "generated.cu").read_text(encoding="utf-8")
        },
    }
```

- [ ] **Step 2: Update `_compile_payload_runtime_launch` helper**

```python
def _compile_payload_runtime_launch(self, turn: int) -> dict:
    payload = self._compile_payload(turn)
    payload["generated_files"] = {
        "generated.cu": (FIXTURES / "generated" / "generated_runtime_launch.cu").read_text(encoding="utf-8")
    }
    return payload
```

- [ ] **Step 3: Update individual test payloads that inline fixture references**

These tests construct their own payloads instead of using `_compile_payload`. Update each one:

**`test_compile_requires_both_reference_and_generated_file_groups`** (two sub-cases):
- Only-generated case: change key from `"cuda/vector_add_inline_ptx.cu"` to `"generated.cu"`, change fixture path from `"vector_add_inline_ptx.cu"` to `"generated.cu"`
- Only-reference case: change key from `"dsl/vector_add_cutedsl.py"` to `"reference.py"`, change fixture path from `"vector_add_cutedsl.py"` to `"reference.py"`

**`test_compile_requires_exactly_one_generated_cu_file`**:
- The test adds a second `.cu` file. Change the base payload key reference: `payload["generated_files"]["alt.cu"] = payload["generated_files"]["generated.cu"]`

**`test_compile_accepts_single_generated_cu_with_helper_files`**:
- No key changes needed (adds `.h` and `.inc` alongside the default payload).

**`test_compile_accepts_reference_files_without_any_reference_cu_file`**:
- Change key from `"dsl/vector_add_cutedsl.py"` to `"reference.py"`, change fixture path from `"vector_add_cutedsl.py"` to `"reference.py"`

**`test_compile_rejects_invalid_relative_paths`**:
- Absolute path case: change generated key from `"/tmp/vector_add_inline_ptx.cu"` to `"/tmp/generated.cu"`, reference key from `"dsl/vector_add_cutedsl.py"` to `"reference.py"`, fixture paths accordingly
- Traversal case: change reference key from `"../dsl/vector_add_cutedsl.py"` to `"../reference.py"`, generated key from `"cuda/vector_add_inline_ptx.cu"` to `"generated.cu"`, fixture paths accordingly

**`test_files_read_rejects_paths_outside_public_turn_dirs`**:
- Change path from `"workspace/inputs/generated/vector_add_inline_ptx.cu"` to `"workspace/inputs/generated/generated.cu"`

- [ ] **Step 4: Update PTX content assertion**

In `test_files_read_returns_inline_artifact_content_by_relative_path`:

Change:
```python
self.assertIn(".visible .entry vector_add_inline_ptx", body["file"]["content"])
```
To:
```python
self.assertIn(".visible .entry generated", body["file"]["content"])
```

Note: The PTX entry point name comes from the CUDA kernel function name inside the `.cu` source, NOT from the filename. Read `tests/fixtures/generated/generated.cu` to find the actual kernel function name and use that in the assertion. If the kernel function is still named `vector_add_inline_ptx`, the assertion stays as-is. Only change if the PTX entry name derives from the filename.

- [ ] **Step 5: Update `ReferenceFixtureContractTest` fixture paths**

Change both test methods in `ReferenceFixtureContractTest`:

```python
def test_reference_fixture_declares_explicit_module_contract(self) -> None:
    fixture_path = FIXTURES / "reference" / "reference.py"
    # ... rest unchanged

def test_reference_fixture_runs_from_config_env(self) -> None:
    # ...
    fixture_path = FIXTURES / "reference" / "reference.py"
    # ... rest unchanged
```

- [ ] **Step 6: Run full test suite**

```bash
cd /home/centos/kernel_lab
uv run --with pytest -- python -m pytest cuda_exec/tests/test_e2e_service.py -v
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add cuda_exec/tests/test_e2e_service.py
git commit -m "test: update all test payloads and assertions for reference.py / generated.cu convention"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `cuda_exec/DESIGN.md:134-160`
- Modify: `cuda_exec/CLAUDE.md`
- Modify: `cuda_exec/models.py:42-69`

- [ ] **Step 1: Update DESIGN.md compile input rules**

In `cuda_exec/DESIGN.md`, update the rules section (around line 134) to add these two new rules after the existing ones:

```
- `reference_files` must include a file keyed as `reference.py` (the entry point)
- `generated_files` must have its single `.cu` file keyed as `generated.cu` (the entry point)
```

Also update the conceptual JSON example (around line 151) to use the new names:

```json
{
  "metadata": { "...": "..." },
  "generated_files": {
    "generated.cu": "extern \"C\" __global__ void ..."
  },
  "reference_files": {
    "reference.py": "import torch ..."
  }
}
```

- [ ] **Step 2: Update CLAUDE.md**

Add a bullet under "Key design decisions":

```
- **Fixed entry file names** — reference entry must be `reference.py`, generated entry must be `generated.cu`. Additional helper files may use any name.
```

- [ ] **Step 3: Update models.py CompileRequest docstring**

Update the docstring of `CompileRequest` (around line 42) to include:

```python
class CompileRequest(RequestBase):
    """Compile request using inline file maps.

    Both `reference_files` and `generated_files` are maps of:
        relative_path -> file_content

    Compile request contract:
    - `reference_files` must be non-empty
    - `reference_files` must include a file keyed as `reference.py` (the entry point)
    - `generated_files` must be non-empty
    - `generated_files` must contain exactly one `.cu` file, keyed as `generated.cu`
    - `generated_files` may include additional headers or inline helper files
    - `reference_files` may include additional helper files of any type
    - compile may run only once per turn; use a new turn for a different upload set

    Why request-side files stay this simple:
    - compile inputs are expected to be normal text source files
    - the caller already knows the intended relative path
    - request-side inputs do not need response-only metadata like encoding or truncation
    """
```

Also update the Field descriptions:

```python
reference_files: Dict[str, str] = Field(
    default_factory=dict,
    description="Non-empty map of relative path to file content; must include reference.py as the entry point",
)
generated_files: Dict[str, str] = Field(
    default_factory=dict,
    description="Non-empty map of generated source inputs; must include generated.cu as the single .cu entry file; headers and inline helper files also allowed",
)
```

- [ ] **Step 4: Commit**

```bash
git add cuda_exec/DESIGN.md cuda_exec/CLAUDE.md cuda_exec/models.py
git commit -m "docs: document reference.py / generated.cu naming convention"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run the full e2e test suite**

```bash
cd /home/centos/kernel_lab
uv run --with pytest -- python -m pytest cuda_exec/tests/test_e2e_service.py -v
```

Expected: All tests PASS (including the 2 new validation tests from Tasks 2-3).

- [ ] **Step 2: Verify error messages manually**

```bash
TOKEN=$(cat ~/.keys/cuda_exec.key)

# Test: wrong .cu name → 400
curl -s -X POST http://127.0.0.1:8199/compile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"metadata":{"run_tag":"t","version":"v","direction_id":0,"direction_slug":"s","turn":0},"reference_files":{"reference.py":"x"},"generated_files":{"my_kernel.cu":"x"}}' | python3 -m json.tool

# Test: missing reference.py → 400
curl -s -X POST http://127.0.0.1:8199/compile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"metadata":{"run_tag":"t","version":"v","direction_id":0,"direction_slug":"s","turn":1},"reference_files":{"model.py":"x"},"generated_files":{"generated.cu":"x"}}' | python3 -m json.tool
```

Expected: Both return HTTP 400 with appropriate error messages.
