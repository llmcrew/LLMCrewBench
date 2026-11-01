"""Sandbox egress, image pin, and supervisor — no Docker daemon required.

Docker-backed checks live in ``tests/test_sandbox.py`` and skip when the
daemon is down. These tests inspect argv, run the stdlib forwarder against a
local stub, and drive the supervisor with a fake ``docker``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import sandbox_run  # noqa: E402

from dailybench import isolation  # noqa: E402

_IMAGE_ID = "sha256:" + "11" * 32
_DIGEST_REF = "dailybench/runner-agent@" + _IMAGE_ID


def test_endpoint_network_is_internal_with_a_forwarder() -> None:
    """``network=endpoint`` is an internal network plus a single-target forwarder.

    The agent joins only that network. HTTP is rewritten to the forwarder's
    alias; HTTPS keeps the original hostname (SNI) and maps it to the forwarder.
    """
    script = Path("/opt/egress_forward.py")
    plan = sandbox_run.endpoint_plan(
        "http://127.0.0.1:1234/v1",
        run_id="abc123",
        docker="docker",
        image=_IMAGE_ID,
        script=script,
    )
    assert plan.network_create == [
        "docker",
        "network",
        "create",
        "--internal",
        "dailybench-abc123",
    ]
    assert plan.network == "dailybench-abc123"
    forwarder = plan.forwarder_argv
    assert "--listen" in forwarder
    assert forwarder[forwarder.index("--listen") + 1] == "0.0.0.0:8080"
    assert "--target" in forwarder
    # 127.0.0.1 inside the forwarder is the forwarder itself; the host stub is
    # reached as host.docker.internal.
    assert forwarder[forwarder.index("--target") + 1] == "host.docker.internal:1234"
    assert "egress_forward.py" in " ".join(forwarder)
    assert "--cpus" in forwarder and "--memory" in forwarder
    assert any(item.startswith("seccomp=") for item in forwarder)
    assert "dailybench-abc123" in forwarder
    assert forwarder[forwarder.index("--network-alias") + 1] == "dailybench-egress"
    assert plan.bridge_connect == [
        "docker",
        "network",
        "connect",
        "bridge",
        plan.forwarder_name,
    ]
    assert "bridge" not in plan.agent_network_argv
    assert "host-gateway" not in " ".join(plan.agent_network_argv)
    assert plan.rewritten_url == "http://dailybench-egress:8080/v1"
    assert isolation.endpoint_url("http://127.0.0.1:1234/v1") == (
        "http://dailybench-egress:8080/v1",
        True,
    )

    https = sandbox_run.endpoint_plan(
        "https://api.example.com/v1",
        run_id="abc123",
        docker="docker",
        image=_IMAGE_ID,
        script=script,
    )
    assert https.forwarder_argv[https.forwarder_argv.index("--target") + 1] == (
        "api.example.com:443"
    )
    assert https.rewritten_url == "https://api.example.com:8080/v1"
    agent = sandbox_run.agent_endpoint_argv(https, "172.28.0.2")
    assert agent == [
        "--network",
        "dailybench-abc123",
        "--add-host",
        "api.example.com:172.28.0.2",
    ]
    assert "host-gateway" not in " ".join(agent)
    assert "bridge" not in agent
    assert isolation.endpoint_url("https://api.example.com/v1") == (
        "https://api.example.com:8080/v1",
        True,
    )
    # Loopback HTTPS cannot be add-host'd as an IP; SNI uses the host alias.
    assert isolation.endpoint_url("https://127.0.0.1:8443/v1") == (
        "https://host.docker.internal:8080/v1",
        True,
    )


def test_forwarder_refuses_other_targets() -> None:
    """The forwarder connects only to its configured target, never to a peer."""
    import socket

    def _stub(payload: bytes) -> tuple[socket.socket, int, list[bytes]]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(8)
        sock.settimeout(5)
        hits: list[bytes] = []

        def loop() -> None:
            while True:
                try:
                    conn, _addr = sock.accept()
                except OSError:
                    return
                try:
                    hits.append(conn.recv(128))
                    conn.sendall(payload)
                except OSError:
                    pass
                finally:
                    conn.close()

        threading.Thread(target=loop, daemon=True).start()
        return sock, sock.getsockname()[1], hits

    target_sock, target_port, target_hits = _stub(b"TARGET")
    other_sock, other_port, other_hits = _stub(b"OTHER")
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    listen_port = probe.getsockname()[1]
    probe.close()
    script = ROOT / "docker" / "egress_forward.py"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--listen",
            f"127.0.0.1:{listen_port}",
            "--target",
            f"127.0.0.1:{target_port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = __import__("time").monotonic() + 5
        last_error = ""
        while __import__("time").monotonic() < deadline:
            try:
                client = socket.create_connection(("127.0.0.1", listen_port), timeout=0.2)
                break
            except OSError as exc:
                last_error = str(exc)
                if proc.poll() is not None:
                    err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
                    pytest.fail(f"forwarder exited {proc.returncode}: {err}")
        else:
            pytest.fail(f"forwarder did not listen: {last_error}")
        client.sendall(b"CONNECT 127.0.0.1:%d HTTP/1.1\r\n\r\n" % other_port)
        client.settimeout(5)
        body = client.recv(64)
        client.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        target_sock.close()
        other_sock.close()
    assert body == b"TARGET"
    assert target_hits, "configured target was never contacted"
    assert other_hits == [], "forwarder reached a host other than --target"


def test_tag_without_digest_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unresolvable tag is refused; ``latest`` stays refused.

    A ``name@sha256:<64 hex>`` reference is used as is. A tag that inspects is
    replaced by that image id once per process, and the container is started
    by the id so a rebuild mid-run cannot change it.
    """
    sandbox_run.forget_image_pins()
    with pytest.raises(sandbox_run.InfraError, match="latest"):
        sandbox_run.resolve_image("dailybench/runner-agent:latest")
    with pytest.raises(sandbox_run.InfraError, match="latest"):
        sandbox_run.resolve_image("dailybench/runner-agent:latest", allow_latest=True)
    with pytest.raises(sandbox_run.InfraError):
        sandbox_run.resolve_image("ubuntu")

    def missing(image: str) -> tuple[str, str]:
        raise sandbox_run.InfraError(f"image {image!r} not available")

    monkeypatch.setattr(sandbox_run, "image_digest", missing)
    with pytest.raises(sandbox_run.InfraError):
        sandbox_run.resolve_image("dailybench/runner-agent:local")

    assert sandbox_run.resolve_image(_DIGEST_REF) == _DIGEST_REF

    monkeypatch.setattr(sandbox_run, "image_digest", lambda image: ("", _IMAGE_ID))
    sandbox_run.forget_image_pins()
    assert sandbox_run.resolve_image("dailybench/runner-agent:local") == _IMAGE_ID

    def later(image: str) -> tuple[str, str]:
        raise AssertionError("image id was resolved twice")

    monkeypatch.setattr(sandbox_run, "image_digest", later)
    assert sandbox_run.resolve_image("dailybench/runner-agent:local") == _IMAGE_ID

    started: list[list[str]] = []

    def popen(argv: list[str], **_kwargs: object) -> None:
        started.append(list(argv))
        raise RuntimeError("stop-after-argv")

    monkeypatch.setattr(sandbox_run, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox_run, "_docker", lambda: "docker")
    monkeypatch.setattr(sandbox_run, "image_sbom", lambda *_a, **_k: [])
    monkeypatch.setattr(
        sandbox_run,
        "_call",
        lambda argv, timeout=None: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    monkeypatch.setattr(sandbox_run.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="stop-after-argv"):
        sandbox_run.run_sandbox(
            ["true"],
            image="dailybench/runner-agent:local",
            task_dir=tmp_path,
            network="none",
        )
    assert started, "container was not started"
    assert _IMAGE_ID in started[0]
    assert "dailybench/runner-agent:local" not in started[0]


def test_cleanup_runs_rm_after_kill_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``docker rm`` still runs when ``docker kill`` fails; the primary error stays."""
    calls: list[list[str]] = []

    def fake_call(argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if "kill" in argv:
            raise OSError("docker kill failed")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def popen(_argv: list[str], **_kwargs: object) -> None:
        raise RuntimeError("primary failure")

    monkeypatch.setattr(sandbox_run, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox_run, "_docker", lambda: "docker")
    monkeypatch.setattr(sandbox_run, "image_digest", lambda _image: ("", _IMAGE_ID))
    monkeypatch.setattr(sandbox_run, "image_sbom", lambda *_a, **_k: [])
    monkeypatch.setattr(sandbox_run, "resolve_image", lambda image, **_k: image)
    monkeypatch.setattr(sandbox_run, "_call", fake_call)
    monkeypatch.setattr(sandbox_run.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="primary failure"):
        sandbox_run.run_sandbox(
            ["true"],
            image=_DIGEST_REF,
            task_dir=tmp_path,
            network="none",
        )
    assert any("kill" in argv for argv in calls)
    assert any("rm" in argv for argv in calls)
    kill_at = next(i for i, argv in enumerate(calls) if "kill" in argv)
    rm_at = next(i for i, argv in enumerate(calls) if "rm" in argv)
    assert rm_at > kill_at


def test_timeout_recovery_on_windows_uses_kill_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A timed-out CLI is killed with taskkill, not os.killpg, on Windows."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "platform", "win32")
    taskkill: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        taskkill.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fail_killpg(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("os.killpg must not run on Windows")

    class Proc:
        pid = 5150
        returncode = None
        stdout = open(os.devnull, "rb")
        stderr = open(os.devnull, "rb")

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired(cmd="docker", timeout=timeout or 0)

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(os, "killpg", fail_killpg, raising=False)
    monkeypatch.setattr(sandbox_run.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox_run.subprocess, "Popen", lambda *_a, **_k: Proc())
    monkeypatch.setattr(sandbox_run, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox_run, "_docker", lambda: "docker")
    monkeypatch.setattr(sandbox_run, "image_digest", lambda _image: ("", _IMAGE_ID))
    monkeypatch.setattr(sandbox_run, "image_sbom", lambda *_a, **_k: [])
    monkeypatch.setattr(sandbox_run, "resolve_image", lambda image, **_k: image)
    monkeypatch.setattr(
        sandbox_run,
        "_call",
        lambda argv, timeout=None: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    report = sandbox_run.run_sandbox(
        ["true"],
        image=_DIGEST_REF,
        task_dir=tmp_path,
        network="none",
        deadline=30,
    )
    assert report["status"] == "timeout"
    assert report["timed_out"] is True
    assert any(
        argv[:1] == ["taskkill"] and "/T" in argv and "/F" in argv and "5150" in argv
        for argv in taskkill
    )


def test_output_is_streamed_not_buffered(tmp_path: Path) -> None:
    """10 MiB of output is written through; the in-memory tail stays ≤ 64 KiB."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'x' * (10 * 1024 * 1024)); sys.stdout.buffer.flush()",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = tmp_path / "stdout.txt"
    err = tmp_path / "stderr.txt"
    tail_out, tail_err = sandbox_run.capture_streams(proc, out, err)
    proc.wait(timeout=30)
    assert tail_out.peak <= 64 * 1024
    assert tail_err.peak <= 64 * 1024
    assert out.stat().st_size == 10 * 1024 * 1024
    assert len(tail_out.text().encode("utf-8")) <= 64 * 1024
    # Invalid UTF-8 is replaced, not raised (S1).
    broken = sandbox_run._BoundedTail()
    broken.extend(b"ok\xff")
    assert broken.text() == "ok\ufffd"


def test_sbom_is_computed_once_per_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dpkg-query`` runs once per image id for the life of the process."""
    sandbox_run.forget_sbom_cache()
    runs: list[list[str]] = []

    def fake_call(argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        runs.append(list(argv))
        if "dpkg-query" in argv:
            return subprocess.CompletedProcess(argv, 0, "libc6\t2.36-9\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sandbox_run, "_call", fake_call)
    first = sandbox_run.image_sbom(_IMAGE_ID)
    second = sandbox_run.image_sbom(_IMAGE_ID)
    assert first == second == ["libc6\t2.36-9"]
    assert sum("dpkg-query" in argv for argv in runs) == 1


def test_endpoint_setup_is_removed_when_argv_build_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure after the forwarder starts still removes the network and container."""
    calls: list[list[str]] = []

    def fake_call(argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def boom(**_kwargs: object) -> list[str]:
        raise RuntimeError("argv failed")

    monkeypatch.setattr(sandbox_run, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox_run, "_docker", lambda: "docker")
    monkeypatch.setattr(sandbox_run, "resolve_image", lambda image, **_k: _IMAGE_ID)
    monkeypatch.setattr(sandbox_run, "image_sbom", lambda *_a, **_k: [])
    monkeypatch.setattr(sandbox_run, "_call", fake_call)
    monkeypatch.setattr(sandbox_run, "_run_argv", boom)
    with pytest.raises(RuntimeError, match="argv failed"):
        sandbox_run.run_sandbox(
            ["true"],
            image=_IMAGE_ID,
            task_dir=tmp_path,
            network="endpoint",
            endpoint_url="http://127.0.0.1:9/v1",
        )
    assert any(argv[1:3] == ["network", "rm"] for argv in calls)
    assert any(
        argv[1:2] == ["rm"] and any(part.startswith("dailybench-egress-") for part in argv)
        for argv in calls
    )


def test_forwarder_survives_a_pause_past_the_connect_timeout() -> None:
    """The handshake timeout must not stay on the connected socket."""
    import importlib.util
    import socket
    import time

    spec = importlib.util.spec_from_file_location(
        "egress_forward", ROOT / "docker" / "egress_forward.py"
    )
    assert spec is not None and spec.loader is not None
    forwarder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forwarder)

    stub = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stub.bind(("127.0.0.1", 0))
    stub.listen(1)
    stub_port = stub.getsockname()[1]
    pause = 0.4

    def accept_late() -> None:
        try:
            conn, _addr = stub.accept()
        except OSError:
            return
        try:
            time.sleep(pause)
            conn.sendall(b"LATE")
        except OSError:
            pass
        finally:
            conn.close()

    threading.Thread(target=accept_late, daemon=True).start()
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    listen_port = probe.getsockname()[1]
    probe.close()
    threading.Thread(
        target=forwarder.serve,
        args=(f"127.0.0.1:{listen_port}", ("127.0.0.1", stub_port)),
        kwargs={"connect_timeout": pause / 2, "idle_timeout": 2.0, "max_connections": 4},
        daemon=True,
    ).start()
    client = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            client = socket.create_connection(("127.0.0.1", listen_port), timeout=0.2)
            break
        except OSError:
            time.sleep(0.05)
    assert client is not None, "forwarder did not listen"
    try:
        client.settimeout(2)
        body = client.recv(16)
    finally:
        client.close()
        stub.close()
    assert body == b"LATE"

