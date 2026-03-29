# cuda_exec Service Proof

This note records concrete evidence that `/home/centos/kernel_lab/cuda_exec` is working as a service.

## Contract evidence

- Public service contract audited in `/home/centos/kernel_lab/cuda_exec/CONTRACT_AUDIT.md`.
- Contract surface is represented by `/home/centos/kernel_lab/cuda_exec/models.py` and wired by `/home/centos/kernel_lab/cuda_exec/main.py`.

## Test evidence

- Full integration suite output captured in `/home/centos/.openclaw/workspace/.cuda-exec-test-proof.txt`.
- The current suite covers compile, file-read, evaluate, profile, execute, reference fixture contract, NCU success/fallback behavior, and profile retention behavior.

## Runnable evidence

- Live service proof captured in `/home/centos/.openclaw/workspace/.cuda-exec-live-proof.json`.
- The service was started via `uvicorn cuda_exec.main:app` and served `GET /healthz` successfully.
- A representative `POST /compile` request succeeded.
- A representative `POST /evaluate` request succeeded and returned `correctness.passed = true` for the tested config.

## Conclusion

Taken together, the contract audit, passing test suite, and live service run provide concrete evidence that the current `cuda_exec` service is working.
