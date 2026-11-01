"""Ceiling taxonomy: a declared time limit is a model outcome, not an outage."""

from __future__ import annotations

import json
import random
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dailybench import core
from dailybench.config import RunConfig
from dailybench.providers import OpenAICompatibleProvider, ScriptedProvider
from dailybench.report import summarise
from dailybench.runner import Runner
from dailybench.tasks import tools

_SUITE = {"per_task": [{"id": "T-1", "category": "D"}]}


def _task():
    for task in tools.generate(random.Random(3), RunConfig()):
        if task.id == "T-1":
            return task
    raise AssertionError("T-1")


def _summary(run) -> dict:
    return summarise([run.score.as_dict()], _SUITE)


class _QuietHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        return

    def _read_body(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _SilentEndpoint(_QuietHandler):
    """Accepts the chat request and never answers."""

    def do_GET(self) -> None:
        self._json({"error": "not here"}, status=404)

    def do_POST(self) -> None:
        self._read_body()
        if not self.path.endswith("/chat/completions"):
            self._json({"error": "not here"}, status=404)
            return
        self.server.hold.wait(30)  # type: ignore[attr-defined]


class _ResetEndpoint(_QuietHandler):
    """Closes the chat socket without a response."""

    def do_GET(self) -> None:
        self._json({"error": "not here"}, status=404)

    def do_POST(self) -> None:
        self._read_body()
        if not self.path.endswith("/chat/completions"):
            self._json({"error": "not here"}, status=404)
            return
        self.connection.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        self.connection.close()


class _NoUsageEndpoint(_QuietHandler):
    """A finished answer with no ``usage`` object."""

    def do_GET(self) -> None:
        self._json({"error": "not here"}, status=404)

    def do_POST(self) -> None:
        self._read_body()
        if not self.path.endswith("/chat/completions"):
            self._json({"error": "not here"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        first = {"choices": [{"delta": {"content": "нет"}, "finish_reason": None}]}
        last = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
        self.wfile.write(f"data: {json.dumps(last)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.hold = threading.Event()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _stop(server: ThreadingHTTPServer) -> None:
    server.hold.set()  # type: ignore[attr-defined]
    server.shutdown()
    server.server_close()


def _run(server: ThreadingHTTPServer, **cfg_kwargs):
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    cfg = RunConfig(base_url=base_url, model="test-model", agent=False, **cfg_kwargs)
    provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=30)
    return Runner(cfg, provider).run_task(_task())


def test_socket_timeout_at_the_declared_ceiling_is_resource_exhausted():
    """A socket that waits out the declared request ceiling is not an outage.

    The break this catches: mapping that timeout to ``infra_failure`` drops the
    run out of the ranking. The model spent the budget it was given.
    """
    server = _serve(_SilentEndpoint)
    try:
        started = time.monotonic()
        run = _run(server, request_wall_clock=1, task_wall_clock=30)
        elapsed = time.monotonic() - started
    finally:
        _stop(server)

    assert elapsed < 8, f"ждал не объявленный предел: {elapsed:.1f} с"
    assert run.score.failure == "resource_exhausted"
    assert run.score.earned == 0
    assert run.score.possible == _task().max_points
    assert not run.transcript.extra.get("infra_error")
    assert _summary(run)["eligible"] is True


def test_task_deadline_between_requests_is_resource_exhausted():
    """The stop before the next request is the same outcome as the socket path."""
    task = _task()

    class _Slow(ScriptedProvider):
        def chat(self, messages, tools=None, **kwargs):
            time.sleep(float(kwargs.get("wall_clock") or 0) * 1.4)
            return super().chat(messages, tools, **kwargs)

    hung = [
        core.ModelReply(
            text="",
            tool_calls=[
                {"id": "c1", "name": "read_file", "arguments": '{"path": "README.md"}'}
            ],
            completion_tokens=4,
        )
        for _ in range(6)
    ]
    cfg = RunConfig(agent=False, request_wall_clock=0.15, task_wall_clock=0.3)
    run = Runner(cfg, _Slow(hung)).run_task(task)

    assert run.score.failure == "resource_exhausted"
    assert "предел времени на задание" in (run.transcript.error or "")
    assert run.score.earned == 0
    assert run.score.possible == task.max_points
    assert not run.transcript.extra.get("infra_error")
    assert _summary(run)["eligible"] is True


def test_connection_reset_stays_infrastructure():
    """A socket the server closes is an outage, not a spent ceiling."""
    server = _serve(_ResetEndpoint)
    try:
        run = _run(server, request_wall_clock=5, task_wall_clock=30)
    finally:
        _stop(server)

    assert run.score.failure == "infra_failure"
    assert _summary(run)["eligible"] is False


def test_missing_usage_is_not_infrastructure():
    """An answer without ``usage`` is not an endpoint outage.

    The break this catches: the missing-usage note used to fall through to
    ``infra_failure``, so a stand that never reports usage was retried forever
    and the run could not be ranked.
    """
    server = _serve(_NoUsageEndpoint)
    try:
        run = _run(server, request_wall_clock=5, task_wall_clock=30)
    finally:
        _stop(server)

    assert run.score.failure != "infra_failure"
    assert run.score.failure in (None, "model_failure")
    assert "infra_failure" not in _summary(run)["infra_failures"]
    assert _summary(run)["eligible"] is True
