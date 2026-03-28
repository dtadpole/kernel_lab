import json
import os
import signal
import socket
import subprocess
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


def setUpModule() -> None:
    _prune_old_runs()


class ServiceProcess:
    def __init__(self) -> None:
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        temp_parent = Path.home() / "temp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        timestamp_prefix = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M")
        self._temp_dir = tempfile.TemporaryDirectory(
            dir=str(temp_parent),
            prefix=f"{timestamp_prefix}-cuda-exec-integration-{os.getpid()}-",
        )
        self.runtime_root = Path(self._temp_dir.name) / "runtime-root"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(self._temp_dir.name) / "uvicorn.log"
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
        self._temp_dir.cleanup()

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
        cls.service = ServiceProcess()
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
            "original_files": {
                "dsl/vector_add_cutedsl.py": (FIXTURES / "original" / "vector_add_cutedsl.py").read_text(encoding="utf-8")
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
        self.assertIsInstance(body["artifacts"], dict)
        self.assertIsInstance(body["logs"], dict)

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
                self.assertIn("logs", first)
                correctness_meta = first["correctness"].get("metadata", {})
                perf_meta = first["performance"].get("metadata", {})
                self.assertEqual(correctness_meta.get("shape_kind"), self._config_map()[first_slug]["extra"]["shape_kind"])
                self.assertEqual(perf_meta.get("input_size"), self._config_map()[first_slug]["extra"]["input_size"])
        else:
            self.assertIn("detail", body)
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
                self.assertEqual(summary_meta.get("rank"), self._config_map()[first_slug]["extra"]["rank"])
                self.assertEqual(summary_meta.get("shape_kind"), self._config_map()[first_slug]["extra"]["shape_kind"])
        else:
            self.assertIn("detail", body)
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


class CudaExecIsolationTest(unittest.TestCase):
    def test_temporary_runtime_root_is_cleaned_up(self) -> None:
        service = ServiceProcess()
        runtime_root = service.runtime_root
        service.start()
        status, body = service.get_json("/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(runtime_root.exists())
        service.stop()
        self.assertFalse(runtime_root.exists())


if __name__ == "__main__":
    unittest.main()
