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
PRUNE_SCRIPT = CUDA_EXEC_DIR / "scripts" / "prune_temp_runs.py"
FIXTURES = REPO_ROOT / "conf" / "fixtures"
CONFIG_FIXTURE = FIXTURES / "configs" / "vector_add_shapes.json"
SUITE_RUN_DIR: Path | None = None
SUITE_VENV_DIR: Path | None = None
REFERENCE_STACK_SITE_PACKAGES = CUDA_EXEC_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
REFERENCE_STACK_EXTRA_PATH = REFERENCE_STACK_SITE_PACKAGES / "nvidia_cutlass_dsl" / "python_packages"
TEST_BEARER_TOKEN = "test-cuda-exec-token-for-e2e"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _suite_venv_dir() -> Path:
    if SUITE_VENV_DIR is None:
        raise RuntimeError("suite temp venv is not initialized")
    return SUITE_VENV_DIR


def _suite_python() -> Path:
    return _suite_venv_dir() / "bin" / "python"


def _reference_pythonpath_entries(existing_pythonpath: str | None = None) -> list[str]:
    entries = [str(REPO_ROOT)]
    for candidate in [REFERENCE_STACK_SITE_PACKAGES, REFERENCE_STACK_EXTRA_PATH]:
        if candidate.exists():
            entries.append(str(candidate))
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    return entries


def _provision_suite_venv(run_dir: Path) -> Path:
    venv_dir = run_dir / ".venv"
    subprocess.run(["uv", "venv", str(venv_dir)], cwd=str(REPO_ROOT), check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_dir / "bin" / "python"),
            "-r",
            str(CUDA_EXEC_DIR / "requirements.txt"),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return venv_dir


def _prune_old_runs() -> None:
    subprocess.run(
        [str(_suite_python()), str(PRUNE_SCRIPT)],
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
    global SUITE_RUN_DIR, SUITE_VENV_DIR
    SUITE_RUN_DIR = _create_suite_run_dir()
    SUITE_VENV_DIR = _provision_suite_venv(SUITE_RUN_DIR)
    _prune_old_runs()


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
        key_path = self.run_dir / f"cuda_exec-{self.instance_id:02d}.key"
        key_path.write_text(TEST_BEARER_TOKEN, encoding="utf-8")
        env["CUDA_EXEC_KEY_PATH"] = str(key_path)
        env["PYTHONPATH"] = os.pathsep.join(_reference_pythonpath_entries(env.get("PYTHONPATH")))
        self.process = subprocess.Popen(
            [
                str(_suite_python()),
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
        req.add_header("Authorization", f"Bearer {TEST_BEARER_TOKEN}")
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
        req.add_header("Authorization", f"Bearer {TEST_BEARER_TOKEN}")
        try:
            with request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            return exc.code, json.loads(body)

    def raw_post_json(self, path: str, payload: dict, *, bearer: str | None = None) -> tuple[int, dict]:
        """POST without the default auth header.  Pass *bearer* to send a custom token."""
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if bearer is not None:
            req.add_header("Authorization", f"Bearer {bearer}")
        try:
            with request.urlopen(req, timeout=10) as resp:
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
                "reference.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8")
            },
            "generated_files": {
                "generated.cu": (FIXTURES / "generated" / "generated.cu").read_text(encoding="utf-8")
            },
        }

    def _config_map(self) -> dict:
        return json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))

    def test_healthz(self) -> None:
        status, body = self.service.get_json("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("ok"), True)
        self.assertEqual(body.get("service"), "cuda_exec")

    def test_healthz_requires_no_auth(self) -> None:
        req = request.Request(f"{self.service.base_url}/healthz", method="GET")
        with request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)

    def test_protected_endpoint_rejects_missing_token(self) -> None:
        status, body = self.service.raw_post_json("/compile", self._compile_payload(turn=150))
        self.assertIn(status, {401, 403})
        self.assertIn("detail", body)

    def test_protected_endpoint_rejects_wrong_token(self) -> None:
        status, body = self.service.raw_post_json(
            "/compile", self._compile_payload(turn=151), bearer="wrong-token"
        )
        self.assertEqual(status, 401)
        self.assertIn("detail", body)
        self.assertIn("invalid bearer token", body["detail"])

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
                    "generated.cu": (FIXTURES / "generated" / "generated.cu").read_text(encoding="utf-8")
                },
            },
        )
        self.assertEqual(only_generated_status, 400)
        self.assertIn("detail", only_generated_body)
        self.assertIn("reference_files must include a file named reference.py", only_generated_body["detail"])

        only_reference_status, only_reference_body = self.service.post_json(
            "/compile",
            {
                "metadata": self._metadata(107),
                "timeout_seconds": 20,
                "reference_files": {
                    "reference.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8")
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
        payload["generated_files"]["alt.cu"] = payload["generated_files"]["generated.cu"]
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

    def test_compile_rejects_cu_file_not_named_generated_cu(self) -> None:
        cu_content = (FIXTURES / "generated" / "generated.cu").read_text(encoding="utf-8")
        payload = {
            "metadata": self._metadata(turn=152),
            "timeout_seconds": 20,
            "reference_files": {
                "reference.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8"),
            },
            "generated_files": {"my_kernel.cu": cu_content},
        }
        status, body = self.service.post_json("/compile", payload)
        self.assertEqual(status, 400)
        self.assertIn("detail", body)
        self.assertIn("must be named generated.cu", body["detail"])

    def test_compile_rejects_reference_files_without_reference_py(self) -> None:
        status, body = self.service.post_json(
            "/compile",
            {
                "metadata": self._metadata(153),
                "timeout_seconds": 20,
                "reference_files": {
                    "my_model.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8")
                },
                "generated_files": {
                    "generated.cu": (FIXTURES / "generated" / "generated.cu").read_text(encoding="utf-8")
                },
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("detail", body)
        self.assertIn("must include a file named reference.py", body["detail"])

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
            "reference.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8"),
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
                    "reference.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8")
                },
                "generated_files": {
                    "/tmp/generated.cu": (FIXTURES / "generated" / "generated.cu").read_text(encoding="utf-8")
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
                    "../reference.py": (FIXTURES / "reference" / "reference.py").read_text(encoding="utf-8")
                },
                "generated_files": {
                    "generated.cu": (FIXTURES / "generated" / "generated.cu").read_text(encoding="utf-8")
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
        self.assertIn(".visible .entry", body["file"]["content"])

    def test_files_read_rejects_paths_outside_public_turn_dirs(self) -> None:
        status, body = self.service.post_json(
            "/files/read",
            {
                "metadata": self._metadata(116),
                "path": "workspace/inputs/generated/generated.cu",
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
                self.assertIn("reference", first)
                self.assertIn("generated", first)
                self.assertIn("correctness", first)
                self.assertIn("performance", first)
                self.assertIn("artifacts", first)
                self.assertIn("logs", first)
                correctness_meta = first["correctness"].get("metadata", {})
                perf_meta = first["performance"].get("metadata", {})
                self.assertEqual(correctness_meta.get("shape_kind"), self._config_map()[first_slug]["shape_kind"])
                self.assertEqual(perf_meta.get("input_size"), self._config_map()[first_slug]["input_size"])
                self.assertIn("output", first["reference"])
                self.assertIn("performance", first["reference"])
                self.assertIn("output", first["generated"])
                self.assertIn("performance", first["generated"])
                self.assertEqual(first["reference"]["output"]["metadata"]["rank"], self._config_map()[first_slug]["rank"])
                self.assertEqual(first["generated"]["performance"]["metadata"]["shape_kind"], self._config_map()[first_slug]["shape_kind"])
                artifact_paths = list(first["artifacts"].keys())
                self.assertTrue(any(path.endswith("comparison.json") for path in artifact_paths))
        else:
            self.assertIn("detail", body)

    def test_evaluate_cli_runs_standalone_after_compile(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=123))
        self.assertEqual(compile_status, 200)

        first_slug, first_config = next(iter(self._config_map().items()))
        env = os.environ.copy()
        env["CUDA_EXEC_ROOT"] = str(self.service.runtime_root)
        env["PYTHONPATH"] = os.pathsep.join(_reference_pythonpath_entries(env.get("PYTHONPATH")))
        completed = subprocess.run(
            [
                str(_suite_python()),
                str(CUDA_EXEC_DIR / "scripts" / "evaluate.py"),
                "--run-tag",
                "integration-tests",
                "--version",
                "v1",
                "--direction-id",
                "1",
                "--direction-slug",
                "vector-add",
                "--turn",
                "123",
                "--config-slug",
                first_slug,
                "--config-json",
                json.dumps(first_config),
                "--timeout",
                "5",
            ],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["config_slug"], first_slug)
        self.assertIn("reference", payload)
        self.assertIn("generated", payload)
        self.assertIn("comparison", payload)
        self.assertEqual(payload["reference"]["output"]["metadata"]["rank"], first_config["rank"])
        self.assertEqual(payload["generated"]["performance"]["metadata"]["shape_kind"], first_config["shape_kind"])
        self.assertIn("speedup", payload["comparison"]["performance"])

    def test_profile_endpoint_accepts_slug_keyed_configs(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=104))
        self.assertEqual(compile_status, 200)

        status, body = self.service.post_json(
            "/profile",
            {
                "metadata": self._metadata(104),
                "timeout_seconds": 5,
                "mode": "dual",
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
                self.assertIn("reference", first)
                self.assertIn("generated", first)
                self.assertIn("reference_summary", first)
                self.assertIn("generated_summary", first)
                self.assertIn("artifacts", first)
                self.assertIn("logs", first)
                summary_meta = first["summary"].get("metadata", {})
                self.assertEqual(summary_meta.get("rank"), self._config_map()[first_slug]["rank"])
                self.assertEqual(summary_meta.get("shape_kind"), self._config_map()[first_slug]["shape_kind"])
                self.assertIn("summary", first["reference"])
                self.assertIn("summary", first["generated"])
                self.assertEqual(first["reference_summary"].get("metadata"), first["reference"]["summary"].get("metadata"))
                self.assertEqual(first["reference_summary"].get("latency_ms"), first["reference"]["summary"].get("latency_ms"))
                self.assertEqual(first["reference_summary"].get("runs"), first["reference"]["summary"].get("runs"))
                self.assertEqual(first["generated_summary"].get("metadata"), first["generated"]["summary"].get("metadata"))
                self.assertEqual(first["generated_summary"].get("latency_ms"), first["generated"]["summary"].get("latency_ms"))
                self.assertEqual(first["generated_summary"].get("runs"), first["generated"]["summary"].get("runs"))
                artifact_paths = list(first["artifacts"].keys())
                self.assertTrue(any(path.endswith("summary.json") for path in artifact_paths))
                comparison = first["summary"].get("comparison", {})
                self.assertIn("reference_median_ms", comparison)
                self.assertIn("generated_median_ms", comparison)
        else:
            self.assertIn("detail", body)

    def test_profile_endpoint_ncu_generated(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=119))
        self.assertEqual(compile_status, 200)

        status, body = self.service.post_json(
            "/profile",
            {
                "metadata": self._metadata(119),
                "timeout_seconds": 20,
                "side": "generated",
                "configs": self._config_map(),
            },
        )
        self.assertIn(status, {200, 400, 408})
        if status == 200:
            self.assertTrue(body["configs"])
            first_slug, first = next(iter(body["configs"].items()))
            self.assertIn("summary", first)
            self.assertIn("artifacts", first)
            self.assertIn("logs", first)
            summary = first["summary"]
            self.assertEqual(summary.get("side"), "generated")
            self.assertIn("ncu_profiled", summary)
            self.assertIn("ncu_report_exists", summary)
            artifact_paths = list(first["artifacts"].keys())
            self.assertTrue(any(path.endswith("summary.json") for path in artifact_paths))
            has_report = any(path.endswith(".ncu-rep") for path in artifact_paths)
            self.assertEqual(has_report, bool(summary.get("ncu_report_exists")))
        else:
            self.assertIn("detail", body)

    def test_profile_endpoint_ncu_reference(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=120))
        self.assertEqual(compile_status, 200)

        status, body = self.service.post_json(
            "/profile",
            {
                "metadata": self._metadata(120),
                "timeout_seconds": 60,
                "side": "reference",
                "configs": self._config_map(),
            },
        )
        self.assertIn(status, {200, 400, 408})
        if status == 200:
            self.assertTrue(body["configs"])
            first_slug, first = next(iter(body["configs"].items()))
            self.assertIn("summary", first)
            self.assertIn("artifacts", first)
            self.assertIn("logs", first)
            summary = first["summary"]
            self.assertEqual(summary.get("side"), "reference")
            self.assertIn("ncu_profiled", summary)
            self.assertIn("ncu_report_exists", summary)
        else:
            self.assertIn("detail", body)

    def test_profile_endpoint_rejects_invalid_side(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=121))
        self.assertEqual(compile_status, 200)

        status, body = self.service.post_json(
            "/profile",
            {
                "metadata": self._metadata(121),
                "timeout_seconds": 5,
                "side": "dual",
                "configs": self._config_map(),
            },
        )
        self.assertEqual(status, 422)

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

    def test_execute_endpoint_rejects_non_toolkit_command(self) -> None:
        status, body = self.service.post_json(
            "/execute",
            {
                "metadata": self._metadata(126),
                "timeout_seconds": 5,
                "command": [str(_suite_python()), "-c", "import sys; sys.exit(7)"],
                "env": {},
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("detail", body)
        self.assertIn("must point to a CUDA Toolkit binary", body["detail"])


    def test_profile_attempts_retain_distinct_attempt_tagged_files(self) -> None:
        compile_status, _ = self.service.post_json("/compile", self._compile_payload(turn=127))
        self.assertEqual(compile_status, 200)

        first_status, first_body = self.service.post_json(
            "/profile",
            {
                "metadata": self._metadata(127),
                "timeout_seconds": 5,
                "mode": "generated_only",
                "configs": self._config_map(),
            },
        )
        second_status, second_body = self.service.post_json(
            "/profile",
            {
                "metadata": self._metadata(127),
                "timeout_seconds": 5,
                "mode": "generated_only",
                "configs": self._config_map(),
            },
        )
        self.assertEqual(first_status, 200)
        self.assertEqual(second_status, 200)
        self.assertNotEqual(first_body["attempt"], second_body["attempt"])
        first_slug, first_item = next(iter(first_body["configs"].items()))
        second_slug, second_item = next(iter(second_body["configs"].items()))
        self.assertEqual(first_slug, second_slug)
        first_artifacts = list(first_item["artifacts"].keys())
        second_artifacts = list(second_item["artifacts"].keys())
        self.assertTrue(any(f"attempt_{first_body['attempt']:03d}" in path for path in first_artifacts))
        self.assertTrue(any(f"attempt_{second_body['attempt']:03d}" in path for path in second_artifacts))
        self.assertNotEqual(set(first_artifacts), set(second_artifacts))
        first_logs = list(first_item["logs"].keys())
        second_logs = list(second_item["logs"].keys())
        self.assertTrue(any(f"attempt_{first_body['attempt']:03d}" in path for path in first_logs))
        self.assertTrue(any(f"attempt_{second_body['attempt']:03d}" in path for path in second_logs))
        self.assertNotEqual(set(first_logs), set(second_logs))

class ReferenceFixtureContractTest(unittest.TestCase):
    def test_reference_fixture_declares_explicit_module_contract(self) -> None:
        fixture_path = FIXTURES / "reference" / "reference.py"
        source = fixture_path.read_text(encoding="utf-8")
        self.assertIn("class Model(nn.Module)", source)
        self.assertIn("def get_init_inputs()", source)
        self.assertIn("def get_inputs(config", source)

    def test_reference_fixture_runs_from_config_env(self) -> None:
        try:
            completed_probe = subprocess.run(
                [
                    str(_suite_python()),
                    "-c",
                    "import os, sys, importlib.util; "
                    f"sys.path.insert(0, '{REFERENCE_STACK_SITE_PACKAGES}'); "
                    f"sys.path.insert(0, '{REFERENCE_STACK_EXTRA_PATH}'); "
                    "import torch; "
                    "assert importlib.util.find_spec('cutlass.cute') is not None; "
                    "assert torch.cuda.is_available(); "
                    "print(torch.__version__)"
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(bool(completed_probe.stdout.strip()))
        except Exception:
            self.skipTest("torch/cutlass.cute/CUDA is unavailable in cuda_exec runtime environment")

        fixture_path = FIXTURES / "reference" / "reference.py"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_EXEC_PARAM_SHAPE": "[1024, 1024]",
                "CUDA_EXEC_PARAM_INPUT_SIZE": "1048576",
                "CUDA_EXEC_PARAM_RANK": "2",
                "CUDA_EXEC_PARAM_SHAPE_KIND": "2d",
            }
        )
        env["PYTHONPATH"] = os.pathsep.join(_reference_pythonpath_entries(env.get("PYTHONPATH")))
        completed = subprocess.run(
            [str(_suite_python()), str(fixture_path)],
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
