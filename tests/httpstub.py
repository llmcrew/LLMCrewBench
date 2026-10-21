"""A minimal OpenAI-compatible endpoint for the CLI, measure and reference tests.

Deliberately a plain factory rather than a pytest fixture: the three test
modules need slightly different shapes (a context manager, a long-lived
fixture, a throwaway probe), and duplicating the handler in each file would
let the three copies drift apart. The factory starts a real HTTP server on an
ephemeral port and records every request body, so a test can assert on what the
harness actually sent instead of on internal state.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_MODEL = "test-model"


def _usage(content: str, completion_tokens: int, reasoning_tokens: int) -> dict[str, Any]:
    """Report a prompt size derived from the request, like a real tokenizer.

    A constant would be worse than useless here: the full-window probe compares
    the reported usage against its target and rescales the prompt when the two
    disagree, so a constant would make the test measure the stub instead of the
    sizing code (and, worse, could build a gigabyte prompt).
    """
    return {
        "prompt_tokens": max(1, len(content) // 4),
        "completion_tokens": completion_tokens,
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def start(
    *,
    models: list[str] | None = None,
    props: dict[str, Any] | None = None,
    tool_call: dict[str, Any] | None = None,
    reply_text: str = "ответ",
    reasoning: str | None = "думаю ",
    tokenize: bool = True,
    completion_tokens: int = 7,
    reasoning_tokens: int = 3,
    drip: float = 0.0,
    stall: float = 0.0,
    finish: bool = True,
    stream_error: dict[str, Any] | None = None,
    status: int = 200,
) -> tuple[str, ThreadingHTTPServer]:
    """Start the stub and return ``(base_url, server)``.

    ``base_url`` already carries the ``/v1`` suffix, so it can be handed to
    ``OpenAICompatibleProvider`` unchanged. ``server.payloads`` is the list of
    every JSON body received (GET requests are not recorded); ``props`` makes
    ``GET /props`` answer instead of 404, and ``tool_call`` makes every
    completion report one tool call.

    Deadline shapes: ``drip`` writes the blocking body one byte at a time with
    that delay, ``stall`` pauses mid-stream, ``finish=False`` drops the terminal
    marker and ``[DONE]``, ``stream_error`` emits an SSE ``error`` event and
    ``status`` answers with that HTTP code instead of a completion.
    """
    model_list = [DEFAULT_MODEL] if models is None else models

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # silence the test server
            return

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _request_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}

        def do_GET(self) -> None:
            if self.path.endswith("/models"):
                self._send_json({"data": [{"id": name} for name in model_list]})
                return
            if self.path.endswith("/props") and props is not None:
                self._send_json(props)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            payload = self._request_body()
            self.server.payloads.append(payload)  # type: ignore[attr-defined]
            if self.path.endswith("/tokenize"):
                if not tokenize:
                    self.send_error(404)
                    return
                content = payload.get("prompt") or payload.get("content") or ""
                self._send_json({"tokens": list(range(max(1, len(content) // 4)))})
                return
            if self.path.endswith("/chat/completions"):
                messages = payload.get("messages") or [{}]
                content = str(messages[-1].get("content") or "")
                if status != 200:
                    self._send_json({"error": {"message": f"stub failure {status}"}}, status)
                    return
                if payload.get("stream"):
                    self._send_stream(content)
                else:
                    self._send_blocking(content)
                return
            self.send_error(404)

        def _send_stream(self, content: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            if stream_error is not None:
                self.wfile.write(f"data: {json.dumps({'error': stream_error})}\n\n".encode())
                self.wfile.flush()
                self.close_connection = True
                return
            chunks: list[dict[str, Any]] = []
            if reasoning:
                chunks.append({"choices": [{"delta": {"reasoning_content": reasoning}}]})
            if reply_text:
                chunks.append({"choices": [{"delta": {"content": reply_text}}]})
            if tool_call:
                name = str(tool_call.get("name", "read_file"))
                arguments = json.dumps(tool_call.get("arguments") or {}, ensure_ascii=False)
                chunks.append(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": str(tool_call.get("id", "c1")),
                                            "function": {"name": name, "arguments": arguments},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            reason = "tool_calls" if tool_call else "stop"
            usage = _usage(content, completion_tokens, reasoning_tokens)
            if finish:
                chunks.append({"choices": [{"delta": {}, "finish_reason": reason}], "usage": usage})
            for index, chunk in enumerate(chunks):
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
                if stall and index == 0:
                    time.sleep(stall)
            if finish:
                self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True

        def _send_blocking(self, content: str) -> None:
            message: dict[str, Any] = {"content": reply_text}
            if reasoning:
                message["reasoning_content"] = reasoning
            if tool_call:
                message["tool_calls"] = [
                    {
                        "id": str(tool_call.get("id", "c9")),
                        "type": "function",
                        "function": {
                            "name": str(tool_call.get("name", "read_file")),
                            "arguments": json.dumps(tool_call.get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                ]
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls" if tool_call else "stop",
                            "message": message,
                        }
                    ],
                    "usage": _usage(content, completion_tokens, reasoning_tokens),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if drip:
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(drip)
            else:
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.payloads = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return f"http://127.0.0.1:{port}/v1", server


def stop(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


def payloads(server: ThreadingHTTPServer) -> list[dict[str, Any]]:
    """Every POST body the stub received, in order."""
    return server.payloads  # type: ignore[attr-defined]


def chat_payloads(server: ThreadingHTTPServer) -> list[dict[str, Any]]:
    """Only the completion requests, so a test can count model calls."""
    return [body for body in payloads(server) if "messages" in body]
