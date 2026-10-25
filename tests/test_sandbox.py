"""Smoke tests for the Docker isolation contract (spec §6).

These tests exercise a real daemon: build, deadline SIGKILL, read-only rootfs,
network namespace, pids cgroup.  They skip only when no Docker daemon answers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "sandbox_run.py"
sys.path.insert(0, str(ROOT / "scripts"))

import sandbox_run  # noqa: E402

pytestmark = pytest.mark.skipif(
    not sandbox_run.docker_available(), reason="Docker daemon is not reachable"
)

IMAGE = "dailybench/runner-python:sandbox-test"

_FAKE_DOCKER = """#!/bin/sh
# Minimal `docker` stand-in: only the verbs the supervisor calls.
case "$1" in
  info) printf '{{"runc": {{"path": "runc"}}}}\\n' ;;
  image) printf 'sha256:%064d|sha256:%064d\\n' 0 1 ;;
  run) exit "${FAKE_RUN_EXIT:-0}" ;;
  inspect)
    if [ "${FAKE_INSPECT_EXIT:-0}" != "0" ]; then exit "$FAKE_INSPECT_EXIT"; fi
    printf '{"Status":"exited","Running":%s,"ExitCode":1,"OOMKilled":false}\\n' \\
      "${FAKE_RUNNING:-false}"
    ;;
  *) exit 0 ;;
esac
"""


@pytest.fixture
def fake_docker(tmp_path: Path) -> Path:
    """A `docker` directory to prepend to PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "docker"
    script.write_text(_FAKE_DOCKER, encoding="utf-8")
    script.chmod(0o755)
    return bin_dir


def _fake_env(bin_dir: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


def _fake_run(
    bin_dir: Path, task_dir: Path, **extra: str
) -> subprocess.CompletedProcess[str]:
    return _cli(
        [
            "run",
            "--allow-latest",
            "--image",
            "dailybench/runner-python:fake",
            "--task-dir",
            str(task_dir),
            "--",
            "true",
        ],
        env=_fake_env(bin_dir, **extra),
    )



def _cli(
    args: list[str], *, timeout: float = 120, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.fixture(scope="module")
def image() -> str:
    proc = _cli(["build", "--tag", IMAGE], timeout=1800)
    if proc.returncode != 0:
        pytest.fail(f"docker build failed: {proc.stderr[-2000:]}")
    yield IMAGE
    subprocess.run(
        ["docker", "image", "rm", "-f", IMAGE],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def task_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix=".sandbox-test-", dir=ROOT))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _run(
    command: list[str],
    *,
    task_dir: Path,
    deadline: float = 60,
    extra: list[str] | None = None,
    timeout: float = 180,
) -> dict[str, object]:
    args = [
        "run",
        "--allow-latest",
        "--image",
        IMAGE,
        "--task-dir",
        str(task_dir),
        "--deadline",
        str(deadline),
        *(extra or []),
        "--",
        *command,
    ]
    proc = _cli(args, timeout=timeout)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:  # pragma: no cover - diagnostic aid
        pytest.fail(f"no report (rc={proc.returncode}): {proc.stderr[-2000:]}")
    return report


def test_build_produces_rootless_image(image: str) -> None:
    user = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Config.User}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert user == "1000:1000"

    proc = subprocess.run(
        ["docker", "run", "--rm", "--read-only", image, "id", "-u"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1000"


def test_run_report_records_the_executor(image: str, task_dir: Path) -> None:
    """The report must state digest, runtime, epoch and package list."""
    report = _run(["python", "-c", "print('ok')"], task_dir=task_dir)
    assert report["status"] == "ok"
    assert report["runtime"] == "runc"
    assert report["runtime_verified"] is True
    assert str(report["image_digest"]).startswith("sha256:") or str(
        report["image_id"]
    ).startswith("sha256:")
    assert report["source_date_epoch"] == 1700000000
    packages = [str(line) for line in report["sbom"]]
    assert len(packages) > 10, "SBOM must list the image packages"
    assert all("\t" in line for line in packages)


def test_deadline_kills_infinite_loop(image: str, task_dir: Path) -> None:
    report = _run(
        ["python", "-c", "while True: pass"],
        task_dir=task_dir,
        deadline=3,
    )
    assert report["status"] == "timeout"
    assert report["exit_code"] == 3
    assert report["timed_out"] is True
    assert float(report["duration_s"]) < 30
    # SIGKILL really happened, not a graceful stop.
    assert report["container_exit_code"] != 0


def test_write_outside_tmpfs_is_read_only(image: str, task_dir: Path) -> None:
    write_dir = task_dir / "out"
    script = (
        "from pathlib import Path\n"
        "import os\n"
        "res = []\n"
        "for p in ('/etc/sandbox_probe', '/workspace/sandbox_probe'):\n"
        "    try:\n"
        "        Path(p).write_text('x'); res.append(p + ':ok')\n"
        "    except OSError:\n"
        "        res.append(p + ':ro')\n"
        "Path('/tmp/probe').write_text('x')\n"
        "Path('/dev/shm/probe').write_text('x')\n"
        "Path('/out/probe').write_text('x')\n"
        "st = os.statvfs('/dev/shm')\n"
        "print(' '.join(res))\n"
        "print('shm_mb', st.f_blocks * st.f_frsize // (1024 * 1024))\n"
    )
    report = _run(
        ["python", "-c", script],
        task_dir=task_dir,
        extra=["--write-dir", str(write_dir)],
    )
    assert report["status"] == "ok", report["output"]
    assert "/etc/sandbox_probe:ro" in str(report["output"])
    assert "/workspace/sandbox_probe:ro" in str(report["output"])
    # /dev/shm is our explicit 64 MiB tmpfs, not a daemon default.
    assert "shm_mb 64" in str(report["output"])
    assert (write_dir / "probe").read_text() == "x"
    assert report["write_dir"] == str(write_dir.resolve())


def test_network_is_unavailable(image: str, task_dir: Path) -> None:
    report = _run(
        [
            "python",
            "-c",
            "import socket; s = socket.socket(); s.settimeout(3); "
            "s.connect(('1.1.1.1', 53)); print('connected')",
        ],
        task_dir=task_dir,
        deadline=20,
    )
    assert report["network"] == "none"
    assert report["status"] == "failed"
    assert "connected" not in str(report["output"])


def test_pids_limit_is_enforced(image: str, task_dir: Path) -> None:
    (task_dir / "forker.py").write_text(
        "import os, time\n"
        "pids = []\n"
        "for _ in range(200):\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError:\n"
        "        break\n"
        "    if pid == 0:\n"
        "        time.sleep(5)\n"
        "        os._exit(0)\n"
        "    pids.append(pid)\n"
        "print('forks', len(pids))\n"
    )
    report = _run(
        ["python", "forker.py"],
        task_dir=task_dir,
        deadline=30,
        extra=["--pids-limit", "16"],
    )
    assert report["status"] == "ok", report["output"]
    assert report["pids_limit"] == 16
    forks = int(str(report["output"]).split("forks", 1)[1].split()[0])
    assert 0 < forks < 200


def test_oom_is_a_distinct_status(image: str, task_dir: Path) -> None:
    report = _run(
        ["python", "-c", "x = bytearray(400 * 1024 * 1024); print('done')"],
        task_dir=task_dir,
        deadline=30,
        extra=["--memory", "64m"],
    )
    assert report["status"] == "oom"
    assert report["exit_code"] == 4
    assert report["oom_killed"] is True


def test_unknown_runtime_is_an_infra_error(task_dir: Path) -> None:
    proc = _cli(
        [
            "run",
            "--allow-latest",
            "--image",
            IMAGE,
            "--task-dir",
            str(task_dir),
            "--runtime",
            "no-such-runtime",
            "--",
            "true",
        ]
    )
    assert proc.returncode == 2
    assert "not registered" in proc.stderr


def test_compose_config_is_valid() -> None:
    proc = subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "docker" / "compose.yaml"), "config", "-q"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_docker_cli_failure_is_infrastructure(
    fake_docker: Path, tmp_path: Path
) -> None:
    """Exit 125 came from the CLI/daemon, not from the candidate."""
    proc = _fake_run(fake_docker, tmp_path, FAKE_RUN_EXIT="125", FAKE_INSPECT_EXIT="1")
    assert proc.returncode == 2, proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "infra_error"
    assert report["exit_code"] == 2
    assert report["container_exit_code"] == 125
    assert "125" in str(report["error"])


def test_exec_failure_stays_a_candidate_failure(
    fake_docker: Path, tmp_path: Path
) -> None:
    """126/127 happen inside the container and are the candidate's problem."""
    for code in ("126", "127"):
        proc = _fake_run(fake_docker, tmp_path, FAKE_RUN_EXIT=code)
        assert proc.returncode == 1, proc.stderr
        report = json.loads(proc.stdout)
        assert report["status"] == "failed"
        assert report["container_exit_code"] == int(code)


def test_unreadable_state_is_infrastructure(
    fake_docker: Path, tmp_path: Path
) -> None:
    """A dead daemon after attach must not be scored as a candidate failure."""
    proc = _fake_run(fake_docker, tmp_path, FAKE_INSPECT_EXIT="1")
    assert proc.returncode == 2, proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "infra_error"
    assert "inspect" in str(report["error"])


def test_missing_terminal_state_is_infrastructure(
    fake_docker: Path, tmp_path: Path
) -> None:
    proc = _fake_run(fake_docker, tmp_path, FAKE_RUNNING="true")
    assert proc.returncode == 2, proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "infra_error"
    assert "terminal" in str(report["error"])


def test_colon_in_mount_path(image: str, tmp_path: Path) -> None:
    """`--mount` keeps a POSIX `:` in the host path from splitting the spec."""
    task_dir = tmp_path / "run:a"
    task_dir.mkdir()
    report = _run(["python", "-c", "print('ok')"], task_dir=task_dir)
    assert report["status"] == "ok", report["output"]
    assert report["task_dir"] == str(task_dir.resolve())


def test_comma_in_mount_path_is_refused(tmp_path: Path) -> None:
    task_dir = tmp_path / "run,a"
    task_dir.mkdir()
    proc = _cli(
        [
            "run",
            "--allow-latest",
            "--image",
            IMAGE,
            "--task-dir",
            str(task_dir),
            "--",
            "true",
        ]
    )
    assert proc.returncode == 2
    assert "','" in proc.stderr


def test_affinity_and_threads_pass_the_seccomp_profile(
    image: str, task_dir: Path
) -> None:
    """sched_setaffinity/set_robust_list are part of the allow-list."""
    script = (
        "import os, threading\n"
        "os.sched_setaffinity(0, {0})\n"
        "seen = []\n"
        "def work():\n"
        "    seen.append(threading.get_ident())\n"
        "threads = [threading.Thread(target=work) for _ in range(8)]\n"
        "[t.start() for t in threads]\n"
        "[t.join() for t in threads]\n"
        "print('affinity', sorted(os.sched_getaffinity(0)))\n"
        "print('threads', len(seen))\n"
    )
    report = _run(["python", "-c", script], task_dir=task_dir)
    assert report["status"] == "ok", report["output"]
    assert "affinity [0]" in str(report["output"])
    assert "threads 8" in str(report["output"])


def test_argv_pins_init_and_shm() -> None:
    argv = sandbox_run._run_argv(
        docker="docker",
        container="c",
        image="i",
        command=["true"],
        runtime="runc",
        network="none",
        task_dir=Path("/tmp/task"),
        write_dir=None,
        cpus="1",
        memory="1g",
        pids_limit=8,
        seccomp=Path("/tmp/seccomp.json"),
    )
    assert "--init" in argv
    assert "/dev/shm:rw,nosuid,nodev,noexec,size=64m,mode=1777" in argv
    assert "type=bind,src=/tmp/task,dst=/workspace,readonly" in argv


# --------------------------------------------------------------------------- #
# The task boundary (H01): fixture only, allowlisted env, no leak, cleanup
# --------------------------------------------------------------------------- #


def test_the_fixture_is_the_only_visible_path(image: str, task_dir: Path, tmp_path: Path):
    """A neighbouring hidden test must not be reachable from inside.

    The oracle and the neighbours live next to the fixture on the host; only the
    fixture is mounted, so a ``find /`` cannot reach the rest of the machine.
    """
    marker = f"neighbour-oracle-{os.getpid()}.py"
    neighbour = task_dir.parent / marker
    neighbour.write_text("# hidden\n", encoding="utf-8")
    (task_dir / "own.txt").write_text("фикстура\n", encoding="utf-8")
    try:
        report = _run(
            ["sh", "-c", f"cat own.txt; find / -name {marker} 2>/dev/null || true"],
            task_dir=task_dir,
        )
    finally:
        neighbour.unlink()
    output = str(report["output"])
    assert report["status"] == "ok", output
    assert "фикстура" in output
    assert marker not in output, "соседний oracle виден из контейнера"
    assert [entry["dst"] for entry in report["mounts"]] == ["/workspace"]
    assert report["mounts"][0]["readonly"] is True


def test_the_host_secret_does_not_reach_the_container(
    image: str, task_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Docker inherits nothing, so the container env is exactly the allowlist."""
    monkeypatch.setenv("DAILYBENCH_API_KEY", "task-secret-9f3c")
    monkeypatch.setenv("HOST_ONLY", "unrelated")
    report = _run(
        ["sh", "-c", "env"],
        task_dir=task_dir,
        extra=["--env", "SAFE_NAME=safe-value"],
    )
    output = str(report["output"])
    assert report["status"] == "ok", output
    assert "SAFE_NAME=safe-value" in output
    assert "task-secret-9f3c" not in output
    assert "HOST_ONLY" not in output
    assert report["env"] == ["SAFE_NAME"]


def test_the_workspace_becomes_writable_on_request(image: str, task_dir: Path):
    """The agent must write the answer; the write surface is in the argv."""
    report = _run(
        ["sh", "-c", "echo answer > answer.txt && cat answer.txt"],
        task_dir=task_dir,
        extra=["--workspace-rw"],
    )
    assert report["status"] == "ok", str(report["output"])
    assert "answer" in str(report["output"])
    assert (task_dir / "answer.txt").read_text(encoding="utf-8").strip() == "answer"
    assert report["mounts"][0]["readonly"] is False


def test_the_container_is_removed_after_a_timeout(image: str, task_dir: Path):
    """Cleanup runs in ``finally``: a killed run leaves nothing behind."""
    report = _run(
        ["python", "-c", "while True: pass"], task_dir=task_dir, deadline=5
    )
    assert report["status"] == "timeout", report["output"]
    assert report["timed_out"] is True
    name = str(report["container"])
    inspect = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True
    )
    assert inspect.returncode != 0, f"контейнер {name} остался после таймаута"


def test_the_isolation_module_mounts_the_fixture_and_the_config_only(
    image: str, task_dir: Path, tmp_path: Path
):
    """The module the runner uses: workspace rw, ``/agent`` ro, nothing else."""
    from dailybench import isolation

    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    box = isolation.Sandbox(work_dir=task_dir, agent_dir=agent_dir, image=image)
    report = box.run_command(["python", "-c", "print('ok')"], host_env={})

    assert report["status"] == "ok", report.get("output")
    assert [(m["dst"], m["readonly"]) for m in report["mounts"]] == [
        ("/workspace", False),
        ("/agent", False),
    ]
    recorded = box.record()
    assert recorded["executor_status"] == "container_run"
    assert recorded["cgroups"]["pids_limit"] == 256
    assert recorded["rootfs"] == "read_only"
