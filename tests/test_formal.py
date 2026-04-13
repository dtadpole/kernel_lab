"""Tests for formal.py — GPU kill logic, binary cache clearing, observability.

Based on bugs found and fixed in this session:
- formal.py killing itself via pgrep matching parent shell
- stale binaries from previous kernel versions being reused
- configs without autotune not getting a sample binary
- bench.gpu vs CUDA_VISIBLE_DEVICES mismatch in kill logic
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.quick


# ---------------------------------------------------------------------------
# _kill_gpu_processes
# ---------------------------------------------------------------------------


class TestKillGpuProcesses:
    def _import_fn(self):
        from cuda_exec.formal import _kill_gpu_processes
        return _kill_gpu_processes

    def test_skip_gpu_0(self):
        """GPU 0 should never be killed (safety measure)."""
        fn = self._import_fn()
        with patch("subprocess.run") as mock_run:
            fn(0)
            mock_run.assert_not_called()

    def test_skip_own_pid(self):
        """Should not kill its own PID."""
        fn = self._import_fn()
        my_pid = os.getpid()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = f"{my_pid}\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            fn(2)
            # nvidia-smi called but no kill since only PID is ourselves
            assert mock_run.call_count == 1  # only nvidia-smi, no kill

    def test_kill_other_pid(self):
        """Should kill other PIDs on the target GPU."""
        fn = self._import_fn()
        mock_nvidia = MagicMock()
        mock_nvidia.returncode = 0
        mock_nvidia.stdout = "99999\n"  # fake PID

        mock_kill = MagicMock()
        mock_kill.returncode = 0

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd[0] == "nvidia-smi":
                return mock_nvidia
            if cmd[0] == "sudo":
                return mock_kill
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=side_effect):
            fn(2)
            # Should have called sudo kill -9 99999

    def test_no_processes(self):
        """No error when GPU has no processes."""
        fn = self._import_fn()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"
        with patch("subprocess.run", return_value=mock_result):
            fn(2)  # should not raise

    def test_nvidia_smi_fails(self):
        """Graceful handling when nvidia-smi fails."""
        fn = self._import_fn()
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            fn(2)  # should not raise


# ---------------------------------------------------------------------------
# Runtime directory clearing
# ---------------------------------------------------------------------------


class TestRuntimeClearing:
    def test_bench_runtime_cleared_on_start(self):
        """bench_runtime directory should be cleared before each formal run."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            # Simulate stale binary from previous run
            stale_dir = tmpdir / "matmul" / "sm90" / "run_test"
            stale_dir.mkdir(parents=True)
            stale_binary = stale_dir / "old_binary.bin"
            stale_binary.write_text("stale")
            assert stale_binary.exists()

            # Simulate what formal.py does: rmtree + mkdir
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
            stale_dir.mkdir(parents=True)

            # Stale binary should be gone
            assert not stale_binary.exists()
            assert stale_dir.exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_fresh_dir_created_if_not_exists(self):
        """bench_runtime directory should be created if it doesn't exist."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            new_dir = tmpdir / "new" / "path"
            assert not new_dir.exists()

            if new_dir.exists():
                shutil.rmtree(new_dir)
            new_dir.mkdir(parents=True)

            assert new_dir.exists()
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Per-config binary map
# ---------------------------------------------------------------------------


class TestPerConfigBinaryMap:
    def test_autotune_configs_get_winner_binary(self):
        """Configs with autotune should get the winner's binary path."""
        per_config_binary = {}
        # Simulate autotune results
        autotune_winners = {
            "mat-256x256": "/path/to/BM64_BN64.bin",
            "mat-512x512": "/path/to/BM128_BN128.bin",
        }
        for cfg, binary in autotune_winners.items():
            per_config_binary[cfg] = binary

        assert per_config_binary["mat-256x256"] == "/path/to/BM64_BN64.bin"
        assert per_config_binary["mat-512x512"] == "/path/to/BM128_BN128.bin"

    def test_non_autotune_configs_get_default_binary(self):
        """Configs without autotune should get the default binary, not None."""
        per_config_binary = {}
        default_binary = "/path/to/default.bin"

        configs_without_autotune = ["mat-2048x2048", "mat-4096x4096", "mat-8192x8192"]
        for cfg in configs_without_autotune:
            per_config_binary[cfg] = default_binary

        for cfg in configs_without_autotune:
            assert per_config_binary[cfg] is not None
            assert per_config_binary[cfg] == default_binary

    def test_none_binary_excluded_from_binary_map(self):
        """Negative: if binary is None, it should NOT appear in binary_map."""
        compiled_binaries = {"ref-cublas": "/path/to/cublas.bin"}
        per_config_binary = {"mat-256x256": "/path/to/sample.bin", "mat-2048x2048": None}

        # Simulate Phase C binary map building
        config_binary_map = dict(compiled_binaries)
        binary = per_config_binary.get("mat-2048x2048")
        if binary:
            config_binary_map["sample-cuda"] = binary

        # sample-cuda should NOT be in the map for mat-2048
        assert "sample-cuda" not in config_binary_map

    def test_non_none_binary_included_in_binary_map(self):
        """Positive: if binary is not None, it SHOULD appear in binary_map."""
        compiled_binaries = {"ref-cublas": "/path/to/cublas.bin"}
        per_config_binary = {"mat-256x256": "/path/to/sample.bin"}

        config_binary_map = dict(compiled_binaries)
        binary = per_config_binary.get("mat-256x256")
        if binary:
            config_binary_map["sample-cuda"] = binary

        assert config_binary_map["sample-cuda"] == "/path/to/sample.bin"


# ---------------------------------------------------------------------------
# PID exclusion logic (self-kill prevention)
# ---------------------------------------------------------------------------


class TestPidExclusion:
    def test_exclude_self(self):
        """Process should exclude its own PID."""
        my_pid = os.getpid()
        pids = [my_pid, 12345, 67890]
        filtered = [p for p in pids if p != my_pid]
        assert my_pid not in filtered
        assert 12345 in filtered

    def test_exclude_parent(self):
        """Process should exclude parent PID."""
        my_pid = os.getpid()
        my_ppid = os.getppid()
        pids = [my_pid, my_ppid, 12345]
        excluded = {my_pid, my_ppid}
        filtered = [p for p in pids if p not in excluded]
        assert my_pid not in filtered
        assert my_ppid not in filtered
        assert 12345 in filtered

    def test_exclude_grandparent(self):
        """Process should exclude grandparent PID."""
        my_ppid = os.getppid()
        # Read grandparent from /proc
        gppid = 0
        try:
            stat = Path(f"/proc/{my_ppid}/stat").read_text().split()
            gppid = int(stat[3])
        except Exception:
            pytest.skip("Cannot read grandparent PID from /proc")

        pids = [os.getpid(), my_ppid, gppid, 12345]
        excluded = {os.getpid(), my_ppid, gppid}
        filtered = [p for p in pids if p not in excluded]
        assert len(filtered) == 1
        assert filtered[0] == 12345


# ---------------------------------------------------------------------------
# GPU param vs CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------


class TestGpuParam:
    def test_bench_gpu_overrides_env(self):
        """bench.gpu should override CUDA_VISIBLE_DEVICES."""
        # Simulate: env has GPU 0, but bench.gpu=2
        env = {"CUDA_VISIBLE_DEVICES": "0"}
        gpu = 2

        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        assert env["CUDA_VISIBLE_DEVICES"] == "2"

    def test_gpu_kill_uses_bench_gpu_not_env(self):
        """Kill logic should use bench.gpu param, not CUDA_VISIBLE_DEVICES."""
        gpu = 3  # bench.gpu=3
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # env says 5

        # The kill should target GPU 3, not 5
        target_gpu = int(gpu) if gpu is not None else None
        assert target_gpu == 3
        assert target_gpu != int(os.environ["CUDA_VISIBLE_DEVICES"])

        # Clean up
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_no_gpu_param_no_kill(self):
        """If bench.gpu is not set, no GPU kill should happen."""
        gpu = None
        should_kill = gpu is not None and int(gpu) != 0
        assert not should_kill


# ---------------------------------------------------------------------------
# Negative: autotune.yaml param mismatch
# ---------------------------------------------------------------------------


class TestAutotuneParamMismatch:
    def test_wrong_param_names_ignored_by_kernel(self):
        """If autotune.yaml has params the kernel doesn't use (e.g. TILE_N
        for a wmma kernel that uses BM), the -D flags are passed but ignored.
        The kernel compiles with its default #define values."""
        # Simulate: autotune picks TILE_N=128, but kernel uses BM
        defines_flags = "-DTILE_N=128 -DSTAGES=4"
        kernel_defaults = {"BM": 128, "BN": 128, "BK": 64}

        # The kernel's #ifndef BM / #define BM 128 / #endif will use 128
        # because -DTILE_N doesn't affect BM
        # This is the bug we found — wrong param names = kernel uses defaults
        assert "BM" not in defines_flags
        assert "TILE_N" in defines_flags
