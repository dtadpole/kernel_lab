"""Tests for GPU environment: driver, CUDA toolkit, fabric manager, and PyTorch CUDA."""

import ctypes
import re
import subprocess

import pytest


@pytest.mark.gpu
class TestGPUEnvironment:
    """Verify the GPU stack is healthy: driver → fabric manager → CUDA runtime → PyTorch."""

    def test_nvidia_driver_loaded(self):
        """libcuda.so.1 must be loadable."""
        lib = ctypes.CDLL("libcuda.so.1")
        assert lib is not None

    def test_cuda_init(self):
        """cuInit(0) must return 0 (success). Non-zero typically means
        fabric manager is down or driver mismatch."""
        lib = ctypes.CDLL("libcuda.so.1")
        rc = lib.cuInit(0)
        assert rc == 0, (
            f"cuInit failed with error {rc}. "
            "If rc=802, check that nvidia-fabricmanager is running and "
            "its version matches the driver."
        )

    def test_fabric_manager_running(self):
        """nvidia-fabricmanager service must be active on multi-GPU SXM systems."""
        result = subprocess.run(
            ["systemctl", "is-active", "nvidia-fabricmanager"],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "active", (
            f"nvidia-fabricmanager is '{result.stdout.strip()}'. "
            "Start it with: sudo systemctl start nvidia-fabricmanager"
        )

    def test_fabric_manager_version_matches_driver(self):
        """Fabric manager version must match the installed driver version."""
        # Get driver version from nvidia-smi
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        driver_ver = smi.stdout.strip().split("\n")[0]
        assert driver_ver, "Could not determine driver version from nvidia-smi"

        # Get fabric manager binary version
        fm = subprocess.run(
            ["nv-fabricmanager", "--version"],
            capture_output=True, text=True,
        )
        fm_output = fm.stdout + fm.stderr
        match = re.search(r"Fabric Manager version is\s*:\s*(\S+)", fm_output)
        if match:
            fm_ver = match.group(1)
        else:
            # Fall back to rpm query
            rpm = subprocess.run(
                ["rpm", "-q", "--queryformat", "%{VERSION}",
                 "nvidia-fabricmanager"],
                capture_output=True, text=True,
            )
            fm_ver = rpm.stdout.strip()

        assert fm_ver == driver_ver, (
            f"Fabric manager version ({fm_ver}) != driver version ({driver_ver}). "
            f"Install matching package: sudo dnf install nvidia-fabricmanager-{driver_ver}"
        )

    def test_torch_cuda_available(self):
        """torch.cuda.is_available() must return True."""
        import torch
        assert torch.cuda.is_available(), (
            "torch.cuda.is_available() is False. "
            "Check driver, fabric manager, and CUDA toolkit compatibility."
        )

    def test_torch_cuda_device_count(self):
        """At least one CUDA device must be visible."""
        import torch
        count = torch.cuda.device_count()
        assert count > 0, "No CUDA devices found"

    def test_torch_cuda_device_name(self):
        """First visible device should report a valid GPU name."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        name = torch.cuda.get_device_name(0)
        assert name, "GPU device name is empty"
        assert "NVIDIA" in name or "H100" in name or "A100" in name, (
            f"Unexpected GPU name: {name}"
        )

    def test_torch_cuda_simple_op(self):
        """A basic tensor operation on GPU must succeed."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        c = a @ b
        assert c.shape == (64, 64)
        assert c.device.type == "cuda"
