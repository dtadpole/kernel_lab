# kernel_lab

A repository for kernel optimization experiments and related tooling.

## Current component

- `cuda_exec/` — FastAPI-based remote CUDA execution service

## Repo layout

```text
kernel_lab/
  cuda_exec/
    __init__.py
    main.py
    models.py
    runner.py
    requirements.txt
    README.md
  README.md
  LICENSE
  .gitignore
```

## Notes

- `cuda_exec` is intended to manage its own Python dependencies and its own `uv` environment.
- the repo root does **not** define a shared `uv` / `venv` environment for all future components
- future agent-side environment management can live in a separate directory and be managed independently

## Owner

- d.t.p

## License

MIT — see `LICENSE`.
