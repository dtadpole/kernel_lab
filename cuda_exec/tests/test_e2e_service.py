import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, UTC
from pathlib import Path
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_EXEC_DIR = REPO_ROOT / "cuda_exec"
VENV_PYTHON = CUDA_EXEC_DIR / ".venv" / "bin" / "python"
PRUNE_SCRIPT = CUDA_EXEC_DIR / "scripts" / "prune_temp_runs.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG_FIXTURE = FIXTURES / "configs" / "vector_add_shapes.json"
SUITE_RUN_DIR: Path | None = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _prune_old_runs() -> None:
    subprocess.run(
        [str(VENV_PYTHON), str(PRUNE_SCRIPT)],
        cwd=str(REPO_ROOT),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _create_suite_run_dir() -> Path:
    temp_parent = Path.home() / "temp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    timestamp_prefix = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M")
    return Path(
        tempfile.mkdtemp(
            dir=str(temp_parent),
            prefix=f"{timestamp_prefix}-cuda-exec-integration-{os.getpid()}-",
        )
    )


def _suite_run_dir() -> Path:
    if SUITE_RUN_DIR is None:
        raise RuntimeError("suite temp directory is not initialized")
    return SUITE_RUN_DIR


def setUpModule() -> None:
    global SUITE_RUN_DIR
    _prune_old_runs()
    SUITE_RUN_DIR = _create_suite_run_dir()


class ServiceProcess:
    _instance_counter = 0

    def __init__(self, run_dir: Path) -> None:
        ServiceProcess._instance_counter += 1
        self.instance_id = ServiceProcess._instance_counter
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.run_dir = run_dir
        self.runtime_root = self.run_dir / f"runtime-root-{self.instance_id:02d}"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / f"uvicorn-{self.instance_id:02d}.log"
        self._log_file = self.log_path.open("w", encoding="utf-8")
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_EXEC_ROOT"] = str(self.runtime_root)
        self.process = subprocess.Popen(
            [
                str(VENV_PYTHON),
                "-m",
                "uvicorn",
                "cuda_exec.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=str(REPO_ROOT),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"uvicorn exited early; log at {self.log_path}")
            try:
                with request.urlopen(f"{self.base_url}/healthz", timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError(f"timed out waiting for service; log at {self.log_path}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self._log_file.close()

    def get_json(self, path: str) -> tuple[int, dict]:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            return exc.code, json.loads(body)

    def post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            return exc.code, json.loads(body)


class CudaExecE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ServiceProcess(_suite_run_dir())
        cls.service.start()

    @classmethod
    def tearDownClass(cls) -> None:
        runtime_root = cls.service.runtime_root
        cls.service.stop()
        cls.runtime_root_after_stop = runtime_root

    def _metadata(self, turn: int) -> dict:
        return {
            "run_tag": "integration-tests",
            "version": "v1",
            "direction_id": 1,
            "direction_slug": "vector-add",
            "turn": turn,
        }

    def _compile_payload(self, turn: int) -> dict:
        return {
            "metadata": self._metadata(turn),
            "timeout_seconds": 20,
            "reference_files": {
                "dsl/vector_add_cutedsl.py": (FIXTURES / "reference" / "vector_add_cutedsl.py").read_text(encoding="utf-8")
            },
            "generated_files": {
                "cuda/vector_add_inline_ptx.cu": (FIXTURES / "generated" / "vector_add_inline_ptx.cu").read_text(encoding="utf-8")
            },
        }

    def _config_map(self) -> dict:
        return json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))

    def test_healthz(self) -> None:
        status, body = self.service.get_json("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("ok"), True)
        self.assertEqual(body.get("service"), "cuda_exec")

    def test_compile_endpoint_accepts_inline_file_maps(self) -> None:
        status, body = self.service.post_json("/compile", self._compile_payload(turn=102))
        self.assertEqual(status, 200)
        self.assertIn("metadata", body)
        self.assertIn("all_ok", body)
        self.assertIn("attempt", body)
        self.assertIn("artifacts", body)
        self.assertIn("tool_outputs", body)

        artifacts = body["artifacts"]
        self.assertTrue(artifacts["binary"]["path"].endswith(".bin"))
        self.assertTrue(artifacts["ptx"]["path"].endswith(".ptx"))
        self.assertTrue(artifacts["cubin"]["path"].endswith(".cubin"))
        self.assertTrue(artifacts["resource_usage"]["path"].endswith(".resource-usage.txt"))
        self.assertTrue(artifacts["sass"]["nvdisasm"]["path"].endswith(".nvdisasm.sass"))

        for payload in [
            artifacts["binary"],
            artifacts["ptx"],
            artifacts["cubin"],
            artifacts["resource_usage"],
            artifacts["sass"]["nvdisasm"],
        ]:
            self.assertEqual(payload["inline"], False)
            self.assertIsNone(payload.get("content"))
            self.assertIsNone(payload.get("encoding"))

        tool_outputs = body["tool_outputs"]
        for pair in [
            tool_outputs["nvcc_ptx"],
            tool_outputs["ptxas"],
            tool_outputs["resource_usage"],
            tool_outputs["nvdisasm"],
        ]:
            self.assertIn("stdout", pair)
            self.assertIn("stderr", pair)
            self.assertEqual(pair["stdout"]["inline"], True)
            self.assertEqual(pair["stderr"]["inline"], True)
            self.assertIn("content", pair["stdout"])
            self.assertIn("content", pair["stderr"])
            self.assertEqual(pair["stdout"]["encoding"], "utf8")
            self.assertEqual(pair["stderr"]["encoding"], "utf8")

    def test_compile_requires_both_reference_and_generated_file_groups(self) -> None:
        only_generated_status, only_generated_body = self.service.post_json(
            "/compile",
            {
                "metadata": self._metadata(106),
                "timeout_seconds": 20,
                "reference_files": {},
                "generated_files": {
                    "cuda/vector_add_inline_ptx.cu": (FIXTURES / "generated" / "vector_add_inline_ptx.cu").read_text(encoding="utf-8")
                },
            },
        )
        self.assertEqual(only_generated_status, 400)
        self.assertIn("detail", only_generated_body)
        self.assertIn("requires non-empty reference_files and generated_files", only_generated_body["detail"])
        self.assertIn("Do not compile with only generated files", only_generated_body["detail"])

        only_reference_status, only_reference_body = self.service.post_json(
            "/compile",
            {
                "metadata": self._metadata(107),
                "timeout_seconds": 20,
                "reference_files": {
                    "dsl/vector_add_cutedsl.py": (FIXTURES / "reference" / "vector_add_cutedsl.py").read_text(encoding="utf-8")
                },
                "generated_files": {},
            },
        )
        self.assertEqual(only_reference_status, 400)
        self.assertIn("detail", only_reference_body)
        self.assertIn("requires non-empty reference_files and generated_files", only_reference_body["detail"])
        self.assertIn("Do not compile with only reference files", only_reference_body["detail"])

    def test_compile_requires_exactly_one_generated_cu_file(self) -> None:
        payload = self._compile_payload(turn=108)
        payload["generated_files"]["cuda/vector_add_alt.cu"] = payload["generated_files"]["cuda/vector_add_inline_ptx.cu"]
        status, body = self.service.post_json("/compile", payload)
        self.assertEqual(status, 400)
        self.assertIn("detail", body)
        self.assertIn("generated_files must contain exactly one .cu file", body["detail"])
        self.assertIn("We recommend a generator", body["detail"])
        self.assertIn("headers or inline helper files", body["detail"])

    def test_compile_rejects_second_attempt_for_same_turn(self) -> None:
        first_status, _ = self.service.post_json("/compile", self._compile_payload(turn=109))
        self.assertEqual(first_status, 200)

        second_status, second_body = self.service.post_json("/compile", self._compile_payload(turn=109))
        self.assertEqual(second_status, 409)
        self.assertIn("detail", second_body)
        self.assertIn("compile may run only once per turn", second_body["detail"])
        self.assertIn("Do not reuse the same turn", second_body["detail"])
        self.assertIn("start a new turn", second_body["detail"])

    def test_compile_rejects_generated_files_without_a_cu_file(self) -> None:
        payload = self._compile_payload(turn=110)
        payload["generated_files"] = {
            "include/vector_add_device.h": "#pragma once\n__device__ inline float addf(float a, float b) { return a + b; }\n",
            "include/vector_add_inline.inc": "#define VECTOR_ADD_INLINE 1\n",
        }
        status, body = self.service.post_json("/compile", payload)
        self.assertEqual(status, 400)
        self.assertIn("detail", body)
        self.assertIn("generated_files must contain exactly one .cu file", body["detail"])
        self.assertIn("We recommend a generator", body["detail"])

    def test_compile_accepts_single_generated_cu_with_helper_files(self) -> None:
        payload = self._compile_payload(turn=111)
        payload["generated_files"]["include/vector_add_device.h"] = (
            "#pragma once\n"
            "__device__ inline float addf(float a, float b) { return a + b; }\n"
        )
        payload["generated_files"]["include/vector_add_inline.inc"] = "#define VECTOR_ADD_INLINE 1\n"
        status, body = self.service.post_json("/compile", payload)
        self.assertEqual(status, 200)
        self.assertIn("artifacts", body)
        self.assertIn("tool_outputs", body)

    def test_compile_accepts_reference_files_without_any_reference_cu_file(self) -> None:
        payload = self._compile_payload(turn=112)
        payload["reference_files"] = {
            "dsl/vector_add_cutedsl.py": (FIXTURES / "reference" / "vector_add_cutedsl.py").read_text(encoding="utf-8"),
            "notes/reference.txt": "vector-add reference notes\n",
        }
        status, body = self.service.post_json("/compile", payload)
        self.assertEqual(status, 200)
        self.assertIn("artifacts", body)
        self.assertIn("tool_outputs", body)

    def test_compile_rejects_invalid_relative_paths(self) -> None:
        absolute_status, absolute_body = self.service.post_json(
            "/compile",
            {
                "metadata": self._metadata(113),
                "timeout_seconds": 20,
                "reference_files": {
                    "dsl/vector_add_cutedsl.py": (FIXTURES / "reference" / "vector_add_cutedsl.py").read_text(encoding="utf-8")
                },
                "generated_files": {
                    "/tmp/vector_add_inline_ptx.cu": (FIXTURES / "generated" / "vector_add_inline_ptx.cu").read_text(encoding="utf-8")
                },
            },
        )
        self.assertEqual(absolute_status, 400)
        self.assertIn("detail", absolute_body)
        self.assertIn("path must be relative", absolute_body["detail"])

        traversal_status, traversal_body = self.service.post_json(
            "/compile",
            {
                "metadata": self._metadata(114),
                "timeout_seconds": 20,
                "reference_files": {
                    "../dsl/vector_add_cutedsl.py": (FIXTURES / "reference" / "vector_add_cutedsl.py").read_text(encoding="utf-8")
                },
                "generated_files": {
                    "cuda/vector_add_inline_ptx.cu": (FIXTURES / "generated" / "vector_add_inline_ptx.cu").read_text(encoding="utf-8")
                },
            },
        )
        self.assertEqual(traversal_status, 400)
        self.assertIn("detail", traversal_body)
        self.assertIn("path contains invalid relative segments", traversal_body["detail"])

    def test_files_read_returns_inline_artifact_content_by_relative_path(self) -> None:
        compile_status, compile_body = self.service.post_json("/compile", self._compile_payload(turn=115))
        self.assertEqual(compile_status, 200)
        ptx_path = compile_body["artifacts"]["ptx"]["path"]

        status, body = self.service.post_json(
            "/files/read",
            {
                "metadata": self._metadata(115),
                "path": ptx_path,
            },
        )
        self.assertEqual(status, 200)
        self.assertIn("metadata", body)
        self.assertIn("file", body)
        self.assertEqual(body["file"]["path"], ptx_path)
        self.assertEqual(body["file"]["inline"], True)
        self.assertEqual(body["file"]["encoding"], "utf8")
        self.assertIn(".visible .entry vector_add_inline_ptx", body["file"]["content"])

    def test_files_read_rejects_paths_outside_public_turn_dirs(self) -> None:
        status, body = self.service.post_json(
            "/files/read",
            {
                "metadata": self._metadata(116),
                "path": "workspace/inputs/generated/vector_add_inline_ptx.cu",
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("detail", body)
        self.assertIn("file reads are limited to artifacts/, logs/, and state/", body["detail"])

    def test_files_read_returns_404_for_missing_turn_file(self) -> None:
        status, body = self.service.post_json(
            "/files/read",
            {
                "metadata": self._metadata(117),
                "path": "artifacts/compile.attempt_001.missing.ptx",
            },
        )
        self.assertEqual(status, 404)
        self.assertIn("detail", body)
        self.assertIn("file not found for this turn/path", body["detail"])

    def test_files_read_respects_max_bytes(self) -> None:
        compile_status, compile_body = self.service.post_json("/compile", self._compile_payload(turn=118))
        self.assertEqual(compile_status, 200)
        path = compile_body["artifacts"]["sass"]["nvdisasm"]["path"]

        status, body = self.service.post_json(
            "/files/read",
            {
                "metadata": self._metadata(118),
                "path": path,
                "max_bytes": 128,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["file"]["path"], path)
        self.assertEqual(body["file"]["inline"], True)
        self.assertEqual(body["file"]["encoding"], "utf8")
        self.assertEqual(body["file"]["truncated"], True)
        self.assertLessEqual(len(body["file"]["content"].encode("utf-8")), 128)

    def test_evaluate_endpoint_accepts_slug_keyed_configs(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=103))
        self.assertEqual(compile_status, 200)

        status, body = self.service.post_json(
            "/evaluate",
            {
                "metadata": self._metadata(103),
                "timeout_seconds": 5,
                "configs": self._config_map(),
            },
        )
        self.assertIn(status, {200, 400, 408})
        if status == 200:
            self.assertIn("all_ok", body)
            self.assertIn("configs", body)
            self.assertIsInstance(body["configs"], dict)
            if body["configs"]:
                first_slug, first = next(iter(body["configs"].items()))
                self.assertIn("status", first)
                self.assertIn("correctness", first)
                self.assertIn("performance", first)
                self.assertIn("artifacts", first)
                self.assertIn("logs", first)
                correctness_meta = first["correctness"].get("metadata", {})
                perf_meta = first["performance"].get("metadata", {})
                self.assertEqual(correctness_meta.get("shape_kind"), self._config_map()[first_slug]["shape_kind"])
                self.assertEqual(perf_meta.get("input_size"), self._config_map()[first_slug]["input_size"])
                artifact_paths = list(first["artifacts"].keys())
                self.assertTrue(any(path.endswith("comparison.json") for path in artifact_paths))
        else:
            self.assertIn("detail", body)

    def test_profile_endpoint_accepts_slug_keyed_configs(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=104))
        self.assertEqual(compile_status, 200)

        status, body = self.service.post_json(
            "/profile",
            {
                "metadata": self._metadata(104),
                "timeout_seconds": 5,
                "configs": self._config_map(),
            },
        )
        self.assertIn(status, {200, 400, 408})
        if status == 200:
            self.assertIn("all_ok", body)
            self.assertIn("configs", body)
            self.assertIsInstance(body["configs"], dict)
            if body["configs"]:
                first_slug, first = next(iter(body["configs"].items()))
                self.assertIn("status", first)
                self.assertIn("summary", first)
                self.assertIn("artifacts", first)
                self.assertIn("logs", first)
                summary_meta = first["summary"].get("metadata", {})
                self.assertEqual(summary_meta.get("rank"), self._config_map()[first_slug]["rank"])
                self.assertEqual(summary_meta.get("shape_kind"), self._config_map()[first_slug]["shape_kind"])
        else:
            self.assertIn("detail", body)

    def test_execute_endpoint_calls_public_interface(self) -> None:
        status, body = self.service.post_json(
            "/execute",
            {
                "metadata": self._metadata(105),
                "timeout_seconds": 5,
                "command": ["/usr/local/cuda/bin/nvcc", "--version"],
                "env": {},
            },
        )
        self.assertIn(status, {200, 400, 408})
        if status == 200:
            self.assertIn("all_ok", body)
            self.assertIn("logs", body)
            self.assertIsInstance(body["logs"], dict)
        else:
            self.assertIn("detail", body)


class ReferenceFixtureContractTest(unittest.TestCase):
    def test_reference_fixture_runs_from_config_env(self) -> None:
        try:
            completed_probe = subprocess.run(
                [str(VENV_PYTHON), "-c", "import torch; print(torch.__version__)"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(bool(completed_probe.stdout.strip()))
        except Exception:
            self.skipTest("torch is unavailable in cuda_exec runtime environment")

        fixture_path = FIXTURES / "reference" / "vector_add_cutedsl.py"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_EXEC_PARAM_SHAPE": "[1024, 1024]",
                "CUDA_EXEC_PARAM_INPUT_SIZE": "1048576",
                "CUDA_EXEC_PARAM_RANK": "2",
                "CUDA_EXEC_PARAM_SHAPE_KIND": "2d",
            }
        )
        completed = subprocess.run(
            [str(VENV_PYTHON), str(fixture_path)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = json.loads(completed.stdout)
        self.assertIn("output", payload)
        self.assertIn("correctness", payload)
        self.assertIn("performance", payload)
        self.assertIn("summary", payload)
        self.assertEqual(payload["output"]["metadata"]["shape"], [1024, 1024])
        self.assertEqual(payload["correctness"]["metadata"]["rank"], 2)
        self.assertEqual(payload["correctness"]["metadata"]["shape_kind"], "2d")
        self.assertEqual(payload["correctness"]["metadata"]["input_size"], 1048576)
        self.assertEqual(payload["correctness"]["passed"], True)
        self.assertIn("latency_ms", payload["performance"])
        self.assertIn("latency_ms", payload["summary"])


class CudaExecIsolationTest(unittest.TestCase):
    def test_temporary_runtime_root_is_preserved_for_inspection(self) -> None:
        service = ServiceProcess(_suite_run_dir())
        runtime_root = service.runtime_root
        run_dir = service.run_dir
        service.start()
        status, body = service.get_json("/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(runtime_root.exists())
        service.stop()
        self.assertTrue(run_dir.exists())
        self.assertTrue(runtime_root.exists())


if __name__ == "__main__":
    unittest.main()
