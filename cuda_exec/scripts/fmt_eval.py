#!/usr/bin/env python3
"""Format evaluate.py JSON output as a compact summary for terminal display."""
import json
import sys


def fmt(val, suffix=""):
    if val is None or val == "?":
        return "n/a"
    if isinstance(val, float):
        return f"{val:.4f}{suffix}"
    return str(val)


def main():
    d = json.load(sys.stdin)
    status = d.get("status", "unknown")
    config = d.get("config_slug", "")

    print(f"evaluate  config={config}  status={status}")

    if status != "ok":
        print(f"  error: {d.get('error', '')}")
        return 1

    c = d.get("comparison", {}).get("correctness", {})
    p = d.get("comparison", {}).get("performance", {})
    rl = d.get("reference", {}).get("performance", {}).get("latency_ms", {})
    gl = d.get("generated", {}).get("performance", {}).get("latency_ms", {})
    rr = d.get("reference", {}).get("performance", {}).get("runs", "?")
    gr = d.get("generated", {}).get("performance", {}).get("runs", "?")

    print(f"  correctness: passed={c.get('passed')}  trials={c.get('trials')}"
          f"  max_diff={fmt(c.get('max_abs_error'))}  avg_diff={fmt(c.get('mean_abs_error'))}"
          f"  shape={c.get('output_shape', '?')}")

    print(f"  reference:   median={fmt(rl.get('median'), 'ms')}"
          f"  mean={fmt(rl.get('mean'), 'ms')}"
          f"  min={fmt(rl.get('min'), 'ms')}"
          f"  max={fmt(rl.get('max'), 'ms')}"
          f"  std={fmt(rl.get('std'), 'ms')}"
          f"  runs={rr}")

    print(f"  generated:   median={fmt(gl.get('median'), 'ms')}"
          f"  mean={fmt(gl.get('mean'), 'ms')}"
          f"  min={fmt(gl.get('min'), 'ms')}"
          f"  max={fmt(gl.get('max'), 'ms')}"
          f"  runs={gr}")

    su = p.get("speedup")
    su_str = f"{su:.2f}x" if su else "n/a"
    print(f"  speedup:     {su_str}"
          f"  (ref={fmt(p.get('reference_median_ms'), 'ms')}"
          f"  gen={fmt(p.get('generated_median_ms'), 'ms')})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
