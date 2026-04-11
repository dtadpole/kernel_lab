"""Tests for per-config autotune: YAML parsing, combo generation, winner selection."""

import tempfile
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

from cuda_exec.autotune import (
    AutotuneResult,
    BenchResult,
    CompileResult,
    PerConfigWinner,
    _compute_per_config_valid_combos,
    _select_per_config_winners,
    combo_tag,
    format_autotune_report,
    generate_combos,
    load_autotune_yaml,
)


# ---------------------------------------------------------------------------
# load_autotune_yaml
# ---------------------------------------------------------------------------


class TestLoadAutotuneYaml:
    def _write(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(textwrap.dedent(content))
        f.close()
        return Path(f.name)

    def test_basic_per_config(self):
        p = self._write("""
            configs:
              mat-256x256:
                autotune:
                  params:
                    BM: [32, 64]
                    BN: [32, 64]
                  constraints:
                    - "BM * BN <= 4096"
              mat-1024x1024:
                autotune:
                  params:
                    BM: [64, 128]
                    BN: [64, 128]
        """)
        data = load_autotune_yaml(p)
        assert "configs" in data
        assert "mat-256x256" in data["configs"]
        at = data["configs"]["mat-256x256"]["autotune"]
        assert at["params"]["BM"] == [32, 64]
        assert at["constraints"] == ["BM * BN <= 4096"]

    def test_config_without_autotune(self):
        p = self._write("""
            configs:
              mat-256x256:
                autotune:
                  params:
                    BM: [32, 64]
              mat-8192x8192: {}
        """)
        data = load_autotune_yaml(p)
        assert "autotune" in data["configs"]["mat-256x256"]
        assert "autotune" not in data["configs"]["mat-8192x8192"]

    def test_config_null_value(self):
        """Config with null value (e.g. `mat-8192x8192:` alone in YAML)."""
        p = self._write("""
            configs:
              mat-256x256:
                autotune:
                  params:
                    BM: [32, 64]
              mat-8192x8192:
        """)
        data = load_autotune_yaml(p)
        assert data["configs"]["mat-8192x8192"] == {}

    def test_missing_configs_section(self):
        p = self._write("""
            params:
              BM: [32, 64]
        """)
        with pytest.raises(ValueError, match="missing 'configs'"):
            load_autotune_yaml(p)

    def test_missing_params_in_autotune(self):
        p = self._write("""
            configs:
              mat-256x256:
                autotune:
                  constraints:
                    - "BM <= 64"
        """)
        with pytest.raises(ValueError, match="missing 'params'"):
            load_autotune_yaml(p)

    def test_empty_params_list(self):
        p = self._write("""
            configs:
              mat-256x256:
                autotune:
                  params:
                    BM: []
        """)
        with pytest.raises(ValueError, match="non-empty list"):
            load_autotune_yaml(p)

    def test_params_converted_to_int(self):
        p = self._write("""
            configs:
              mat-256x256:
                autotune:
                  params:
                    BM: [32.0, 64.0]
        """)
        data = load_autotune_yaml(p)
        assert data["configs"]["mat-256x256"]["autotune"]["params"]["BM"] == [32, 64]

    def test_constraints_default_empty(self):
        p = self._write("""
            configs:
              mat-256x256:
                autotune:
                  params:
                    BM: [32, 64]
        """)
        data = load_autotune_yaml(p)
        assert data["configs"]["mat-256x256"]["autotune"]["constraints"] == []


# ---------------------------------------------------------------------------
# _compute_per_config_valid_combos
# ---------------------------------------------------------------------------


class TestComputePerConfigValidCombos:
    def test_disjoint_search_spaces(self):
        config_autotunes = {
            "small": {"params": {"BM": [32, 64]}, "constraints": []},
            "large": {"params": {"BM": [128, 256]}, "constraints": []},
        }
        union_combos, tags = _compute_per_config_valid_combos(config_autotunes)
        assert len(union_combos) == 4  # 32, 64, 128, 256
        assert len(tags["small"]) == 2
        assert len(tags["large"]) == 2
        # No overlap
        assert tags["small"].isdisjoint(tags["large"])

    def test_overlapping_search_spaces(self):
        config_autotunes = {
            "a": {"params": {"BM": [64, 128]}, "constraints": []},
            "b": {"params": {"BM": [128, 256]}, "constraints": []},
        }
        union_combos, tags = _compute_per_config_valid_combos(config_autotunes)
        assert len(union_combos) == 3  # 64, 128, 256 (128 deduped)
        assert len(tags["a"]) == 2
        assert len(tags["b"]) == 2

    def test_constraints_filter_combos(self):
        config_autotunes = {
            "x": {
                "params": {"BM": [32, 64, 128], "BN": [32, 64, 128]},
                "constraints": ["BM * BN <= 4096"],
            },
        }
        union_combos, tags = _compute_per_config_valid_combos(config_autotunes)
        # Valid: 32*32, 32*64, 64*32, 32*128, 128*32, 64*64 = 6
        assert len(union_combos) == 6
        assert len(tags["x"]) == 6

    def test_empty_after_constraints(self):
        config_autotunes = {
            "x": {
                "params": {"BM": [256]},
                "constraints": ["BM <= 64"],
            },
        }
        union_combos, tags = _compute_per_config_valid_combos(config_autotunes)
        assert len(union_combos) == 0
        assert len(tags["x"]) == 0


# ---------------------------------------------------------------------------
# _select_per_config_winners
# ---------------------------------------------------------------------------


class TestSelectPerConfigWinners:
    def _make_bench_result(self, combo, latencies, ok=True):
        tag = combo_tag(combo)
        valid = [v for v in latencies.values() if v is not None and v > 0]
        if valid:
            from math import exp, log
            geo_mean = exp(sum(log(v) for v in valid) / len(valid))
        else:
            geo_mean = float("inf")
        return BenchResult(
            tag=tag, combo=combo, median_ms=geo_mean,
            all_latencies=latencies, ok=ok,
        )

    def _make_compile_result(self, combo, regs=128, smem=0):
        tag = combo_tag(combo)
        return CompileResult(
            combo=combo, tag=tag, binary_path=f"/tmp/{tag}.bin",
            ok=True, registers=regs, smem_bytes=smem,
        )

    def test_different_winners_per_config(self):
        """Small tiles win for small matrix, large tiles win for large matrix."""
        combo_small = {"BM": 32, "BN": 32}
        combo_large = {"BM": 128, "BN": 128}

        bench_results = [
            self._make_bench_result(combo_small, {
                "mat-256": 0.01, "mat-4096": 5.0,
            }),
            self._make_bench_result(combo_large, {
                "mat-256": 0.05, "mat-4096": 1.0,
            }),
        ]
        compile_results = [
            self._make_compile_result(combo_small),
            self._make_compile_result(combo_large),
        ]
        config_combo_tags = {
            "mat-256": {combo_tag(combo_small), combo_tag(combo_large)},
            "mat-4096": {combo_tag(combo_small), combo_tag(combo_large)},
        }

        winners = _select_per_config_winners(
            bench_results, compile_results, config_combo_tags,
        )

        assert winners["mat-256"].best_combo == combo_small  # 0.01 < 0.05
        assert winners["mat-4096"].best_combo == combo_large  # 1.0 < 5.0

    def test_restricted_search_space(self):
        """Each config only considers combos in its search space."""
        combo_a = {"BM": 32}
        combo_b = {"BM": 128}

        bench_results = [
            self._make_bench_result(combo_a, {"cfg1": 0.5, "cfg2": 0.5}),
            self._make_bench_result(combo_b, {"cfg1": 0.3, "cfg2": 0.3}),
        ]
        compile_results = [
            self._make_compile_result(combo_a),
            self._make_compile_result(combo_b),
        ]
        # cfg1 can only pick combo_a; cfg2 can only pick combo_b
        config_combo_tags = {
            "cfg1": {combo_tag(combo_a)},
            "cfg2": {combo_tag(combo_b)},
        }

        winners = _select_per_config_winners(
            bench_results, compile_results, config_combo_tags,
        )

        assert winners["cfg1"].best_combo == combo_a
        assert winners["cfg2"].best_combo == combo_b

    def test_all_bench_failed_for_config(self):
        """Config with no successful benchmarks is omitted from results."""
        combo = {"BM": 64}
        bench_results = [
            self._make_bench_result(combo, {"cfg1": None}, ok=False),
        ]
        compile_results = [self._make_compile_result(combo)]
        config_combo_tags = {"cfg1": {combo_tag(combo)}}

        winners = _select_per_config_winners(
            bench_results, compile_results, config_combo_tags,
        )

        assert "cfg1" not in winners

    def test_none_latency_fallback(self):
        """When specific config latency is None, use geo-mean as fallback."""
        combo = {"BM": 64}
        bench_results = [
            self._make_bench_result(combo, {"cfg1": None, "cfg2": 1.0}, ok=True),
        ]
        compile_results = [self._make_compile_result(combo)]
        config_combo_tags = {"cfg1": {combo_tag(combo)}}

        winners = _select_per_config_winners(
            bench_results, compile_results, config_combo_tags,
        )

        # cfg1 latency is None, falls back to geo-mean (1.0)
        assert "cfg1" in winners
        assert winners["cfg1"].best_median_ms == 1.0


# ---------------------------------------------------------------------------
# format_autotune_report
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration: sample matmul autotune.yaml
# ---------------------------------------------------------------------------


class TestSampleMatmul:
    SAMPLE_YAML = Path(__file__).resolve().parents[1] / "data" / "sample" / "matmul" / "cuda" / "autotune.yaml"

    def test_load_sample_yaml(self):
        data = load_autotune_yaml(self.SAMPLE_YAML)
        configs = data["configs"]
        assert "mat-256x256" in configs
        assert "mat-512x512" in configs
        assert "mat-1024x1024" in configs
        # mat-256x256 has autotune
        assert "autotune" in configs["mat-256x256"]
        assert "BM" in configs["mat-256x256"]["autotune"]["params"]

    def test_sample_combo_generation(self):
        data = load_autotune_yaml(self.SAMPLE_YAML)
        config_autotunes = {
            slug: spec["autotune"]
            for slug, spec in data["configs"].items()
            if "autotune" in spec
        }
        union_combos, tags = _compute_per_config_valid_combos(config_autotunes)

        # Should have combos for all 3 configs
        assert len(tags) == 3
        assert "mat-256x256" in tags
        assert "mat-512x512" in tags
        assert "mat-1024x1024" in tags

        # Each config should have valid combos
        for slug, tag_set in tags.items():
            assert len(tag_set) > 0, f"{slug} has no valid combos"

        # Union should be non-empty and <= sum of per-config combos
        total_per_config = sum(len(t) for t in tags.values())
        assert 0 < len(union_combos) <= total_per_config

    def test_sample_smem_constraint(self):
        """Verify shared memory constraints actually filter combos."""
        # Use a tight constraint that filters some combos
        params = {"BM": [32, 64, 128], "BN": [32, 64, 128], "BK": [16, 32]}
        all_combos = generate_combos(params, [])
        valid_combos = generate_combos(params, ["BM * BK * 2 + BK * BN * 2 <= 8192"])
        # BM=128,BN=128,BK=32 → 128*32*2 + 32*128*2 = 16384 > 8192 → filtered
        assert len(valid_combos) < len(all_combos)


# ---------------------------------------------------------------------------
# format_autotune_report
# ---------------------------------------------------------------------------


class TestFormatAutotuneReport:
    def test_basic_report(self):
        result = AutotuneResult(
            per_config_results={
                "mat-256": PerConfigWinner(
                    config_slug="mat-256", best_combo={"BM": 32},
                    best_tag="BM32", best_median_ms=0.01,
                    best_registers=128, best_smem_bytes=0,
                    defines_flags="-DBM=32",
                ),
                "mat-4096": PerConfigWinner(
                    config_slug="mat-4096", best_combo={"BM": 128},
                    best_tag="BM128", best_median_ms=1.0,
                    best_registers=128, best_smem_bytes=0,
                    defines_flags="-DBM=128",
                ),
            },
            total_combos=4,
            valid_combos=4,
            compiled_ok=4,
            benchmarked_ok=4,
            all_results=[],
            duration_s=10.0,
            configs_without_autotune=["mat-8192"],
        )
        report = format_autotune_report(result)
        assert "mat-256" in report
        assert "mat-4096" in report
        assert "-DBM=32" in report
        assert "-DBM=128" in report
        assert "Distinct combos: 2" in report
        assert "No autotune: mat-8192" in report
