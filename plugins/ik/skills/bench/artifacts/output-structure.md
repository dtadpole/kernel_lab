# Output Structure & Gem Rules

## kernel_lab_kb (git repo, text only)

```
runs/
  run_<host>/
    ref/<kernel>/              # snapshot of source files
    peak/<arch>/<kernel>/      # peak impls snapshot
    gen/<arch>/<kernel>/       # gen impls snapshot
    configs/<kernel>.json
    impls/<bench_ts>/
      <impl_slug>/
        results.json           # structured results (compact)
    gems/<impl_slug>/
      v001_<YYYYMMDD_HHMMSS>/  # first gem = first run
      v002_<YYYYMMDD_HHMMSS>/  # only created when a config beats v001
```

## Local runtime (not in git)

```
~/.cuda_exec/<kernel>/<arch>/<run_tag>/bench-<kernel>-<ts>/
  workspace/, artifacts/, logs/, state/
```

## Gem Rules

A new gem is created when at least one config is faster than the latest gem:

- Absolute improvement > 0.002 ms
- Relative improvement > 0.2%
- Both must be exceeded (AND)
- First run is always a gem
- Gems are never deleted — they form a progression of improvements

## JSON Response Format

```json
{
  "kernel": "<kernel>",
  "arch": "<arch>",
  "num_configs": 6,
  "impls_requested": ["ref-<name>", "gen-<name>"],
  "refs": ["ref-<name>"],
  "gens": ["gen-<name>"],
  "improved": true,
  "gems": {
    "gen-<name>": { "version": 2, "improved_configs": ["<config>"] }
  },
  "results": {
    "gen-<name>": {
      "compile_ok": true,
      "trial_ok": true,
      "trial_result": {
        "all_ok": true,
        "configs": {
          "<config>": {
            "status": "ok",
            "correctness": { "passed": true },
            "performance": { "latency_ms": { "median": 0.011 } }
          }
        }
      }
    }
  }
}
```

Key fields:
- `improved` (bool): `true` if any impl set a new gem
- `gems` (dict): per-impl gem info, only for impls that improved
