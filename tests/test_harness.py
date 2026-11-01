"""Tests for the harness itself: transport, text rules, tool loop, reporting.

These cover the parts of DailyBench-16 that no category module owns: SSE
streaming and tool-call assembly, the answer-normalisation rules, the runner's
multi-turn tool loop, the admission rule in the totals, and report rendering.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import random
import shutil
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import httpstub

from dailybench import core, text
from dailybench.config import RunConfig
from dailybench.providers import OpenAICompatibleProvider, ProviderError, ScriptedProvider
from dailybench.report import compare_markdown, render_markdown, summarise
from dailybench.runner import Runner
from dailybench.suite import apply_profile_weights, build_suite
from dailybench.tasks import load_all, tools

load_all()


# --------------------------------------------------------------------------- #
# A real HTTP endpoint speaking SSE
# --------------------------------------------------------------------------- #

SSE_STREAM = [
    {"choices": [{"delta": {"reasoning_content": "думаю "}}]},
    {"choices": [{"delta": {"content": "ответ"}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "search_code", "arguments": '{"query"'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "x"}'}}]}}]},
    {
        "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    },
]


def _prompt_tokens(content: str) -> int:
    """A size derived from the request, like a real tokenizer.

    A constant would make the full-window probe rescale its prompt against an
    unrelated number (and build a half-gigabyte string when the ratio explodes).
    """
    return max(1, len(content) // 4)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the test server
        return

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.last_payload = payload  # type: ignore[attr-defined]

        if self.path.endswith("/tokenize"):
            content = payload.get("prompt") or payload.get("content") or ""
            body = json.dumps({"tokens": list(range(max(1, len(content) // 4)))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.endswith("/chat/completions") and payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            content = str((payload.get("messages") or [{}])[-1].get("content") or "")
            chunks = [dict(chunk) for chunk in SSE_STREAM]
            chunks[-1]["usage"] = {**chunks[-1]["usage"], "prompt_tokens": _prompt_tokens(content)}
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return

        if self.path.endswith("/chat/completions"):
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "42",
                                "tool_calls": [
                                    {
                                        "id": "c9",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": _prompt_tokens(str((payload.get("messages") or [{}])[-1].get("content") or "")),
                        "completion_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)


class _CountingTokenizer(BaseHTTPRequestHandler):
    """A tokenizer whose count depends on the text, not only on its length.

    ``aaaa`` and ``bbbb`` are the same length and must not share a cache entry.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the test server
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        content = str(payload.get("prompt") or payload.get("content") or "")
        self.server.tokenize_calls += 1  # type: ignore[attr-defined]
        body = json.dumps({"tokens": list(range(max(1, len(content) + 4 * content.count("b"))))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def _serve(handler_cls=_Handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    server.tokenize_calls = 0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def endpoint():
    with _serve() as served:
        yield served


@pytest.fixture()
def tokenizer_endpoint():
    with _serve(_CountingTokenizer) as served:
        yield served


def test_streaming_reply_assembly(endpoint):
    base_url, server = endpoint
    provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
    assert provider.list_models() == ["test-model"]

    reply = provider.chat([{"role": "user", "content": "hi"}], tools=[], max_new_tokens=64)
    assert reply.text == "ответ"
    assert reply.reasoning == "думаю "
    assert reply.tool_calls == [{"id": "c1", "name": "search_code", "arguments": '{"query": "x"}'}]
    assert reply.prompt_tokens == 1, "размер входа считается от запроса"
    assert reply.completion_tokens == 7
    assert reply.reasoning_tokens == 3
    # OpenAI semantics: completion_tokens already includes reasoning_tokens (7, not 10).
    assert reply.generated_tokens == 7
    assert reply.ttft_s is not None and reply.ttft_s >= 0
    assert reply.finish_reason == "tool_calls"
    assert server.last_payload["temperature"] == 0.0 and server.last_payload["top_p"] == 1.0
    assert server.last_payload["stream_options"] == {"include_usage": True}, "usage нужен для контроля контекста"


def test_blocking_reply_and_tokenizer(endpoint):
    base_url, _ = endpoint
    provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
    reply = provider.chat([{"role": "user", "content": "hi"}], stream=False)
    assert reply.text == "42"
    assert reply.tool_calls[0]["name"] == "read_file"
    assert reply.prompt_tokens == 1

    count = provider.tokenize("abcdefgh")
    assert count == 2  # the stub reports len/4
    assert provider.tokenize("") == 0


def test_the_tokenizer_cache_is_keyed_on_the_whole_text(tokenizer_endpoint):
    """P3: the key was ``len`` plus 32 random bits of ``hash()``.

    Two inputs of the same length could collide and the second one then got the
    first one's count — an exact-looking number that silently moves the
    admission decision.
    """
    base_url, server = tokenizer_endpoint
    provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
    assert provider.tokenize("aaaa") == 4
    assert provider.tokenize("bbbb") == 20, "тот же размер, другой текст — свой счёт"
    assert provider.tokenize("aaaa") == 4
    assert server.tokenize_calls == 2, "повтор текста берётся из кэша, а не по сети"


def test_unreachable_endpoint_raises(refusing_endpoint):
    provider = OpenAICompatibleProvider(base_url=refusing_endpoint, model="x", timeout=2)
    with pytest.raises(ProviderError):
        provider.list_models()


def test_provider_error_is_an_infrastructure_failure_by_default():
    assert ProviderError("boom").infra is True
    assert ProviderError("bad input", infra=False).infra is False


def test_a_trickling_blocking_reply_is_cut_by_the_wall_clock():
    """The ceiling covers the whole request, not one socket operation (R10).

    ``urlopen(timeout=...)`` alone would let a byte-at-a-time body run for as
    long as it likes: every single read stays inside the timeout.
    """
    base_url, server = httpstub.start(drip=0.005, reply_text="x" * 120, reasoning=None)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=30)
        started = time.monotonic()
        with pytest.raises(ProviderError, match="предел времени"):
            provider.chat([{"role": "user", "content": "hi"}], stream=False, wall_clock=0.05)
        assert time.monotonic() - started < 0.15
    finally:
        httpstub.stop(server)


def test_a_stalled_stream_is_cut_by_the_wall_clock():
    """A stream that stops mid-flight must not wait out the socket timeout."""
    base_url, server = httpstub.start(stall=5.0)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=30)
        started = time.monotonic()
        with pytest.raises(ProviderError, match="предел времени"):
            provider.chat([{"role": "user", "content": "hi"}], wall_clock=0.05)
        assert time.monotonic() - started < 0.2
    finally:
        httpstub.stop(server)


def test_the_request_deadline_uses_the_clock_it_is_compared_with(monkeypatch):
    """M01: the deadline used to be built from ``perf_counter`` but checked
    against ``monotonic``.

    The two domains share an origin on Linux and on this host, but not
    everywhere (Windows, and whatever the standing review measured gave a 1.11 s
    offset). A huge offset turns a 50 ms ceiling into an effectively unlimited
    wait: the offset below is added to ``perf_counter`` only, so a deadline
    mixed across domains would never be reached.
    """
    real_perf_counter = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real_perf_counter() + 1000.0)
    base_url, server = httpstub.start(stall=5.0)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=30)
        started = time.monotonic()
        with pytest.raises(ProviderError, match="предел времени"):
            provider.chat([{"role": "user", "content": "hi"}], wall_clock=0.05)
        assert time.monotonic() - started < 0.15
    finally:
        httpstub.stop(server)


def test_eof_without_a_finish_marker_is_a_provider_error():
    """A truncated stream used to become a successful reply (R14)."""
    base_url, server = httpstub.start(finish=False)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        with pytest.raises(ProviderError, match="finish_reason"):
            provider.chat([{"role": "user", "content": "hi"}])
    finally:
        httpstub.stop(server)


def test_a_cut_stream_keeps_the_fragments_it_already_produced():
    """P3: the error used to throw away text, reasoning and usage.

    They were generated (and paid for) before the stream died; only the missing
    terminal marker makes the reply unusable as a whole.
    """
    base_url, server = httpstub.start(finish=False)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        with pytest.raises(ProviderError) as caught:
            provider.chat([{"role": "user", "content": "hi"}])
        partial = caught.value.partial
        assert partial is not None, "частичный ответ обязан ехать вместе с ошибкой"
        assert partial.text == "ответ"
        assert partial.reasoning == "думаю "
        assert partial.finish_reason is None
    finally:
        httpstub.stop(server)


def test_the_runner_records_the_partial_turn_of_a_cut_stream():
    """P3: the runner appends the partial reply before classifying the error."""
    task = _tool_task("T-1")
    base_url, server = httpstub.start(finish=False)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        run = Runner(RunConfig(), provider).run_task(task)
    finally:
        httpstub.stop(server)

    assert run.transcript.error and "finish_reason" in run.transcript.error
    assert len(run.transcript.turns) == 1
    assert run.transcript.turns[0].reply.text == "ответ"
    assert run.transcript.turns[0].reply.finish_reason is None
    assert run.score.status == "error"


def test_an_sse_error_event_is_not_an_empty_reply():
    base_url, server = httpstub.start(stream_error={"message": "model crashed", "type": "server_error"})
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        with pytest.raises(ProviderError, match="model crashed"):
            provider.chat([{"role": "user", "content": "hi"}])
    finally:
        httpstub.stop(server)


def test_http_errors_are_provider_errors_and_keep_the_infra_flag():
    base_url, server = httpstub.start(status=500)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        with pytest.raises(ProviderError) as caught:
            provider.chat([{"role": "user", "content": "hi"}])
        assert caught.value.infra is True
        assert "HTTP 500" in str(caught.value)
        with pytest.raises(ProviderError):
            provider.chat([{"role": "user", "content": "hi"}], stream=False)
    finally:
        httpstub.stop(server)


def test_a_client_error_is_not_marked_infrastructure():
    base_url, server = httpstub.start(status=400)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        with pytest.raises(ProviderError) as caught:
            provider.chat([{"role": "user", "content": "hi"}])
        assert caught.value.infra is False
    finally:
        httpstub.stop(server)


def _demo_run(
    directory: Path,
    model: str,
    scores: list[dict],
    *,
    sha: str,
    per_task: list[dict],
) -> Path:
    """A minimal on-disk run the HTML report can load."""
    directory.mkdir(parents=True)
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({"model": model}), encoding="utf-8")
    (directory / "suite.json").write_text(
        json.dumps({"tasks": len(per_task), "sha256": sha, "per_task": per_task}), encoding="utf-8"
    )
    return directory


def test_an_infra_http_503_is_not_ranked_as_a_quality_zero(tmp_path):
    """H12(b): a 503 is an unavailable provider, not a model answer."""
    task = _tool_task("T-1")
    base_url, server = httpstub.start(status=503)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        run = Runner(RunConfig(), provider).run_task(task)
    finally:
        httpstub.stop(server)

    assert run.score.failure == "infra_failure"
    assert run.score.status == "error"
    assert run.score.possible == task.max_points, "инфраструктурная ошибка остаётся в знаменателе"
    summary = summarise([run.score.as_dict()], {"per_task": [{"id": task.id, "category": task.category}]})
    assert summary["eligible"] is False
    assert "не ранжируется" in summary["verdict"]

    from dailybench.htmlreport import render_html

    weak = _demo_run(
        tmp_path / "run-503", "упавшая", [run.score.as_dict()], sha="contract",
        per_task=[{"id": task.id, "category": task.category}],
    )
    strong = _demo_run(
        tmp_path / "run-ok", "полная",
        [{"task_id": "IF-1", "category": "A", "earned": 2.0, "possible": 2.0, "checks": []}],
        sha="contract", per_task=[{"id": "IF-1", "category": "A", "max_points": 2.0}],
    )
    page = render_html([weak, strong])
    assert page.index("data-model='полная'") < page.index("data-model='упавшая'")
    assert "инфраструктура" in page and "инфраструктура недоступна" in page


def test_a_model_timeout_is_its_own_outcome_not_infrastructure(tmp_path):
    """H12: a declared per-task ceiling is the candidate's outcome."""
    task = _tool_task("T-1")

    class _Hung(ScriptedProvider):
        def chat(self, messages, tools=None, **kwargs):
            time.sleep(float(kwargs.get("wall_clock") or 0) * 1.4)
            return super().chat(messages, tools, **kwargs)

    cfg = RunConfig()
    cfg.request_wall_clock = 0.1
    cfg.task_wall_clock = 0.15
    replies = [_call(index, "read_file", {"path": "README.md"}) for index in range(1, 10)]
    run = Runner(cfg, _Hung(replies)).run_task(task)
    assert run.score.failure == "resource_exhausted"
    assert run.score.status == "error"
    assert not run.transcript.extra.get("infra_error")
    assert run.score.possible == task.max_points, "таймаут не выкидывают из знаменателя"
    summary = summarise([run.score.as_dict()], {"per_task": [{"id": task.id, "category": task.category}]})
    assert summary["eligible"] is True, "модельный исход не делает прогон неранжируемым"
    assert summary["outcomes"]["resource_exhausted"] == 1
    assert summary["eligible"] is True


# --------------------------------------------------------------------------- #
# The outcome dictionary and its eligibility gates
# --------------------------------------------------------------------------- #

OUTCOMES = (
    "success",
    "model_failure",
    "resource_exhausted",
    "infra_failure",
    "evaluator_failure",
    "integrity_violation",
    "unsupported",
)


def test_the_outcome_dictionary_is_complete_and_labeled():
    assert core.FAILURE_KINDS == OUTCOMES
    assert core.STORED_FAILURE_KINDS == OUTCOMES[2:]
    for kind in core.FAILURE_KINDS:
        assert core.failure_label(kind), f"у исхода {kind} нет подписи"
    assert core.INCOMPLETE_FAILURES == {
        "infra_failure", "evaluator_failure", "integrity_violation",
    }
    assert core.MODEL_FAILURES == {"model_failure", "resource_exhausted"}


def test_legacy_names_normalise_to_the_dictionary():
    expected = {
        "provider_unavailable": "infra_failure",
        "transport_unavailable": "infra_failure",
        "incomplete_scenario": "resource_exhausted",
        "model_protocol_failure": "model_failure",
        "candidate_resource_exhausted": "resource_exhausted",
        "evaluator_unavailable": "evaluator_failure",
    }
    for old, canonical in expected.items():
        assert core.canonical_failure(old) == canonical
        assert core.failure_label(old), "старая подпись теряется"
    assert core.canonical_failure(None) is None
    assert core.canonical_failure("network_outage") == "network_outage"
    assert core.is_infra_failure("provider_unavailable")
    assert core.is_infra_failure("integrity_violation")
    assert not core.is_infra_failure("candidate_resource_exhausted")
    assert not core.is_infra_failure("model_protocol_failure")


def test_outcome_of_derives_the_quiet_states():
    passed = {"task_id": "A-1", "category": "A", "checks": [{"name": "x", "passed": True}]}
    wrong = {"task_id": "A-1", "category": "A", "checks": [{"name": "x", "passed": False}]}
    assert core.outcome_of(passed) == "success"
    assert core.outcome_of(wrong) == "model_failure"
    assert core.outcome_of({"skipped": "нет браузера"}) == "unsupported"
    assert core.outcome_of({"failure": "provider_unavailable"}) == "infra_failure"
    assert core.outcome_of({"failure": "integrity_violation"}) == "integrity_violation"
    assert core.outcome_of({"error": "boom"}) == "infra_failure"


@pytest.mark.parametrize(
    "kind, eligible",
    [
        ("infra_failure", False),
        ("evaluator_failure", False),
        ("integrity_violation", False),
        ("resource_exhausted", True),
        ("model_failure", True),
        ("provider_unavailable", False),
        ("candidate_resource_exhausted", True),
    ],
)
def test_each_outcome_has_its_eligibility_gate(kind, eligible):
    score = {
        "task_id": "A-1", "category": "A", "earned": 0.0, "possible": 1.0,
        "error": "прогон прерван", "failure": kind,
        "checks": [{"name": "x", "passed": False, "points": 1.0}],
    }
    summary = summarise([score], {"per_task": [{"id": "A-1", "category": "A"}]})
    assert summary["eligible"] is eligible
    assert summary["incomplete"] is False, "покрытие не путается с допуском"
    assert summary["outcomes"].get(core.canonical_failure(kind)) == 1
    if not eligible:
        assert "не ранжируется" in summary["verdict"]
    if core.canonical_failure(kind) == "integrity_violation":
        assert "удостоверение отозвано" in summary["verdict"]
    if kind in ("resource_exhausted", "candidate_resource_exhausted"):
        assert "исходы модели" in summary["verdict"]


def test_unsupported_leaves_the_denominator_and_names_the_cohort():
    scores = [
        {"task_id": "A-1", "category": "A", "earned": 1.0, "possible": 1.0,
         "checks": [{"name": "x", "passed": True, "points": 1.0}]},
        {"task_id": "A-2", "category": "A", "skipped": "нет браузера", "checks": []},
    ]
    summary = summarise(scores, {
        "per_task": [{"id": "A-1", "category": "A"}, {"id": "A-2", "category": "A"}],
    })
    assert summary["outcomes"]["unsupported"] == 1
    assert summary["outcomes"]["success"] == 1
    assert summary["unsupported"] == 1
    assert summary["skipped_scored"] == 1
    assert summary["eligible"] is True
    assert summary["incomplete"] is True
    assert "unsupported" in summary["verdict"]
    assert "ни в числитель, ни в знаменатель" in summary["verdict"]


def test_a_wrong_answer_is_model_failure_but_not_a_failure_code():
    score = {"task_id": "A-1", "category": "A", "earned": 0.0, "possible": 1.0,
             "checks": [{"name": "x", "passed": False, "points": 1.0}]}
    summary = summarise([score], {"per_task": [{"id": "A-1", "category": "A"}]})
    assert summary["outcomes"]["model_failure"] == 1
    assert summary["failures"] == {}, "тихий исход не попадает в failure-счётчики"
    assert summary["eligible"] is True


def test_a_legacy_bundle_reads_under_the_canonical_kind():
    score = {"task_id": "A-1", "category": "A", "earned": 0.0, "possible": 1.0,
             "error": "HTTP 503", "failure": "provider_unavailable", "checks": []}
    summary = summarise([score], {"per_task": [{"id": "A-1", "category": "A"}]})
    assert summary["failures"] == {"infra_failure": 1}
    assert summary["infra_failures"] == {"infra_failure": 1}
    assert summary["eligible"] is False


def test_a_browser_worker_refusal_is_an_evaluator_error(monkeypatch):
    """H12(c): a dead worker is not 0/4 ``ok``."""
    from dailybench import browser, visual

    spec = visual.VisualSpec(
        scene=[visual.contains_all("objects", ["ground"])],
        simulation=[visual.near("t", 1.0, 1e-3)],
        image=[visual.image_min("changed_fraction", 0.02)],
        steps=60,
        dt=1 / 60,
    )
    task = visual.build_task(
        task_id="K-T", title="K-T", prompt="Собери сцену.", spec=spec,
        golden="<html><script>1</script></html>",
    )
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(
        browser, "render",
        lambda *a, **k: browser.RenderResult(
            ok=False, reason="рендер-процесс не запустился: boom",
            failure="evaluator_failure",
        ),
    )
    transcript = core.Transcript(task_id="K-T", category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(
        text=(
            "```html\n<!DOCTYPE html><html><body><script type='module'>\n"
            "import * as THREE from './vendor/three.module.js';\n"
            "window.__probe = () => ({objects: [{name: 'ground'}]});\n"
            "window.__step = (dt) => dt; window.__ready = true;\n"
            "</script></body></html>\n```"
        )
    )))
    score = visual.evaluate(task, transcript)
    assert score.failure == "evaluator_failure"
    assert score.status == "error"
    assert len([c for c in score.checks if not c.passed]) == len(visual.CHECK_NAMES)
    assert score.possible == 4.0
    summary = summarise([score.as_dict()])
    assert summary["eligible"] is False


# --------------------------------------------------------------------------- #
# Browser hooks under an absolute deadline
# --------------------------------------------------------------------------- #

FAST_PAGE = """<!doctype html><script>
window.__ready = true;
window.__probe = () => ({x: 1});
</script><canvas></canvas>"""

HUNG_HOOK_PAGE = """<!doctype html><script>
function spin(ms) { const t = performance.now(); while (performance.now() - t < ms) {} }
window.__ready = true;
window.__probe = () => { spin(800); return {x: 1}; };
</script><canvas></canvas>"""


def test_a_hung_page_hook_is_killed_at_the_render_deadline():
    """A busy JS hook cannot be pre-empted by a Playwright timeout (R10).

    The whole page pass runs in a killable process, so an 800 ms hook under a
    100 ms render deadline returns control almost immediately instead of
    holding the evaluator.
    """
    from dailybench import browser

    _require_browser(browser)
    warm = browser.render(FAST_PAGE, timeout_ms=10_000, again=False)
    assert warm.ok, warm.reason

    # A control render pays the same browser/context setup under the same
    # budget; the difference is the part the hung hook itself adds.
    control_started = time.monotonic()
    control_result = browser.render(FAST_PAGE, timeout_ms=1_000, again=False)
    control = time.monotonic() - control_started
    assert control_result.ok, control_result.reason

    started = time.monotonic()
    result = browser.render(HUNG_HOOK_PAGE, timeout_ms=100, again=False)
    elapsed = time.monotonic() - started

    assert not result.ok
    assert "дедлайн рендера" in result.reason
    assert result.probe is None
    assert elapsed < 0.5, f"зависший hook держал рендер {elapsed:.3f} s"
    assert elapsed < control + 0.2, (
        f"hook добавил {elapsed - control:.3f} s к обычному проходу ({control:.3f} s)"
    )


def test_the_next_render_works_after_a_deadline_kill():
    from dailybench import browser

    _require_browser(browser)
    browser.render(FAST_PAGE, timeout_ms=10_000, again=False)
    assert not browser.render(HUNG_HOOK_PAGE, timeout_ms=100, again=False).ok

    result = browser.render(FAST_PAGE, timeout_ms=10_000, again=False)
    assert result.ok, result.reason
    assert result.probe == {"x": 1}


# --------------------------------------------------------------------------- #
# Protocol, runtime failures and refused requests are different states
# --------------------------------------------------------------------------- #

CONSOLE_ERROR_PAGE = """<!doctype html><script>
window.__ready = true;
window.__probe = () => ({x: 1});
window.__step = (dt) => dt;
console.error("fixture console error");
</script><canvas></canvas>"""

NO_PROTOCOL_PAGE = """<!doctype html><canvas width="400" height="300"></canvas>"""


def _require_browser(browser) -> None:
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(reason)


def test_a_console_error_is_a_runtime_failure_not_a_dead_render():
    """M09: one ``console.error`` used to set ``ok=False`` and zero every check."""
    from dailybench import browser

    _require_browser(browser)
    result = browser.render(CONSOLE_ERROR_PAGE, again=False)

    assert result.ok, result.reason
    assert result.shot and result.shot.startswith(b"\x89PNG\r\n\x1a\n")
    assert any(error.startswith("console:") for error in result.errors)
    assert result.probe == {"x": 1}


def test_a_missing_protocol_is_flagged_without_killing_the_frame():
    """M09: losing ``__ready``/``__probe``/``__step`` is a protocol state.

    The frame is still taken so the picture can be scored (H17); the flags say
    which part of the protocol is missing, the render itself stays usable.
    """
    from dailybench import browser

    _require_browser(browser)
    result = browser.render(NO_PROTOCOL_PAGE, steps=3, again=False)

    assert result.ok, result.reason
    assert result.shot and result.shot.startswith(b"\x89PNG\r\n\x1a\n")
    assert not result.ready_ok
    assert not result.probe_ok
    assert not result.step_ok


def test_http_and_websocket_escapes_use_one_classifier(refusing_endpoint):
    """M09: both escapes are the same fault and must be recorded the same way."""
    from dailybench import browser

    _require_browser(browser)

    http_result = browser.render(
        '<!doctype html><img src="https://example.invalid/x.png">', mode="live"
    )
    websocket = refusing_endpoint.replace("http", "ws", 1)
    ws_result = browser.render(
        f"<!doctype html><script>new WebSocket('{websocket}/ch')</script>", again=False
    )

    assert browser.external_kinds(http_result.external)[browser.BLOCKED_HTTP] == 1
    assert browser.external_kinds(http_result.external)[browser.BLOCKED_WS] == 0
    assert browser.external_kinds(ws_result.external)[browser.BLOCKED_WS] == 1
    assert browser.external_kinds(ws_result.external)[browser.BLOCKED_HTTP] == 0
    # The classifier is one function: an entry's class depends only on the entry.
    for entry in http_result.external + ws_result.external:
        assert browser.external_class(entry) in (browser.BLOCKED_HTTP, browser.BLOCKED_WS)


# --------------------------------------------------------------------------- #
# Text rules
# --------------------------------------------------------------------------- #


def test_answer_token_requires_a_single_value():
    assert text.answer_token("12") == "12"
    assert text.answer_token("**12**") == "12"
    assert text.answer_token("12\n") == "12"
    assert text.answer_token("Ответ: 12") == "12"
    assert text.answer_token("12 минут", ("минут",)) == "12"
    assert text.answer_token("Вот ответ:\n12") is None, "лишняя строка должна обнулять ответ"
    assert text.answer_token("думал долго\n12") is None
    assert text.answer_token("") is None


def test_unit_stripping_never_eats_a_word():
    """``s`` is seconds in one answer and the last letter of nam in another."""
    assert text.answer_token("helios", ("service",)) == "helios"
    assert text.answer_token("service helios", ("service",)) == "service helios"
    assert text.answer_token("helios", ("service",)) != "helio"
    assert text.answer_token("18s", ("s",)) == "18"
    assert text.answer_token("18 s", ("s",)) == "18"
    assert text.answer_token("740ms", ("ms",)) == "740"
    assert text.answer_token("391") == "391"
    assert text.answer_token("10.14.37.192,10.14.37.255,10.14.37.193,10.14.37.254") is not None


def test_strict_json_separates_format_from_content():
    value, strict = text.strict_json('{"a": 1}')
    assert strict and value == {"a": 1}
    value, strict = text.strict_json('```json\n{"a": 1}\n```')
    assert not strict and value == {"a": 1}
    value, strict = text.strict_json('Вот: {"a": 1}')
    assert not strict and value is None
    assert text.loose_json('Вот: {"a": 1}') == {"a": 1}


def test_numbered_lines_and_known_unknown():
    payload = "\n".join(f"{i}. KNOWN: x{i}" if i % 2 else f"{i}. UNKNOWN" for i in range(1, 11))
    values = text.numbered_lines(payload, 10)
    assert values is not None
    assert values[0] == "KNOWN: x1" and values[1] == "UNKNOWN"
    assert text.numbered_lines("1. a\n2. b", 10) is None
    assert text.known_unknown("KNOWN: Go") == ("known", "Go")
    assert text.known_unknown("UNKNOWN") == ("unknown", "")
    assert text.known_unknown("может быть") is None


# --------------------------------------------------------------------------- #
# Runner: the tool loop
# --------------------------------------------------------------------------- #


def _tool_task(task_id: str) -> core.TaskInstance:
    import random

    for task in tools.generate(random.Random(3), RunConfig()):
        if task.id == task_id:
            return task
    raise AssertionError(task_id)


def _call(index: int, name: str, arguments: dict) -> core.ModelReply:
    return core.ModelReply(
        text="",
        tool_calls=[{"id": f"c{index}", "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}],
    )


def test_tool_loop_scores_a_correct_sequence():
    task = _tool_task("T-3")
    oracle = task.oracle
    provider = ScriptedProvider(
        [
            _call(1, "read_file", {"path": "config/app.yaml"}),
            _call(
                2,
                "write_file",
                {
                    "path": "config/app.yaml",
                    "content": f"server:\n{oracle['after_line']}\n  port: 8080\n  log_level: info\n",
                },
            ),
            _call(3, "run_tests", {"target": "config"}),
            _call(4, "git_diff", {}),
            core.ModelReply(
                text=(
                    f"Готово: workers {oracle['workers_before']} → {oracle['workers_after']}, "
                    "tests прошли, diff проверен."
                )
            ),
        ]
    )
    run = Runner(RunConfig(), provider).run_task(task)

    assert [c["name"] for c in run.transcript.tool_log] == ["read_file", "write_file", "run_tests", "git_diff"]
    assert run.score.possible == 2.0
    assert run.score.earned == 2.0
    assert run.transcript.workspace_after["config/app.yaml"].count(oracle["after_line"]) == 1


def test_tool_loop_detects_a_destructive_call():
    import random

    task = None
    for candidate in tools.generate(random.Random(4), RunConfig()):
        if candidate.id == "T-4":
            task = candidate
    assert task is not None
    protected = task.oracle["protected"][0]
    provider = ScriptedProvider(
        [
            _call(1, "search_code", {"query": "build-cache", "path": "."}),
            _call(2, "delete_file", {"path": protected}),
            core.ModelReply(text=f"Удалил {protected}"),
        ]
    )
    run = Runner(RunConfig(), provider).run_task(task)
    illegal = [c for c in run.transcript.tool_log if c["illegal"]]
    assert illegal and illegal[0]["name"] == "delete_file"
    assert protected not in run.transcript.workspace_after
    # Destroying protected data forfeits the outside snapshot and the safety
    # check: an illegal deletion can never keep full marks, while the
    # denominator stays the task's own 2 points.
    assert run.score.earned == 0.0
    assert run.score.possible == task.max_points == 2.0
    assert any(c.name == "safety" and not c.passed for c in run.score.checks)
    assert sum(c.points for c in run.score.checks) == task.max_points
    assert run.score.earned < task.max_points


def test_a_new_file_inside_the_cache_directory_keeps_the_outside_intact():
    """The unchanged-snapshot check is scoped by directory, not by two names.

    A manifest written under the permitted cache directory is not a violation
    of "touch nothing else", so it must not cost the preservation point.
    """
    task = _tool_task("T-4")
    oracle = task.oracle
    replies = [
        _call(1, "delete_file", {"path": oracle["cache_files"][0]}),
        _call(2, "delete_file", {"path": oracle["cache_files"][1]}),
        _call(3, "write_file", {"path": "tmp/build-cache/manifest.txt", "content": "cleaned\n"}),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(), ScriptedProvider(replies)).run_task(task)
    assert run.transcript.workspace_after.get("tmp/build-cache/manifest.txt") == "cleaned\n"
    checks = {c.name: c.passed for c in run.score.checks}
    assert checks["outside_preserved"] and checks["safety"]
    assert run.score.earned == task.max_points == 2.0


def test_a_root_file_deleted_through_dotdot_is_not_counted_as_inside_the_cache():
    """H15: ``tmp/build-cache/../../README.md`` resolves to the repository root.

    The delete guard used the raw string prefix before resolving ``..``, so the
    destructive call was reported as legal; preservation only watched two named
    files, so a destroyed root README still scored 2/2. The final path decides
    legality and the whole outside-of-cache snapshot decides preservation.
    """
    task = _tool_task("T-4")
    oracle = task.oracle
    root_readme = "tmp/build-cache/../../README.md"
    replies = [
        _call(1, "search_code", {"query": "build-cache", "path": "."}),
        _call(2, "delete_file", {"path": root_readme}),
        _call(3, "delete_file", {"path": oracle["cache_files"][0]}),
        _call(4, "delete_file", {"path": oracle["cache_files"][1]}),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(), provider := ScriptedProvider(replies)).run_task(task)
    assert len(provider.calls) == 5, "все пять реплик должны быть израсходованы"

    dangling = [c for c in run.transcript.tool_log if c["name"] == "delete_file" and c["illegal"]]
    assert [c["arguments"]["path"] for c in dangling] == [root_readme]
    assert "README.md" not in run.transcript.workspace_after, "файл за пределами кэша действительно удалён"
    assert all(path not in run.transcript.workspace_after for path in oracle["cache_files"])

    failed = {c.name for c in run.score.checks if not c.passed}
    assert "outside_preserved" in failed, f"потеря корневого файла должна ломать сохранность: {failed}"
    assert "safety" in failed
    assert run.score.earned < task.max_points
    assert run.score.possible == task.max_points == 2.0


def test_diagnosis_without_successful_evidence_is_not_full_marks():
    """H14: a schema-invalid call is an attempt, not an observation.

    Both calls carry the right service/metric and an extra unsupported
    argument, so the handler raises ``TypeError`` and returns an error. The
    diagnosis is the oracle answer, yet unbounded evidence must not earn it.
    """
    task = _tool_task("T-2")
    oracle = task.oracle
    replies = [
        _call(1, "get_logs", {"service": "api-gateway", "minutes": 15, "level": "ERROR", "limit": 50}),
        _call(2, "get_metric", {"service": "inference", "metric": "gpu_memory_used_percent", "minutes": 15, "since": "now"}),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(), ScriptedProvider(replies)).run_task(task)
    assert [c["error"] for c in run.transcript.tool_log] == [True, True]
    assert all("invalid arguments" in c["result"] for c in run.transcript.tool_log)

    passed = {c.name for c in run.score.checks if c.passed}
    failed = {c.name for c in run.score.checks if not c.passed}
    assert {"logs_attempted", "metrics_attempted", "diagnosis_without_restart"} <= passed
    assert {"logs_observed", "metrics_observed"} <= failed
    assert run.score.earned == 1.5, "попытка и вывод без успешного наблюдения — не полный балл"
    assert run.score.possible == task.max_points == 2.0


def test_a_successful_retry_after_an_invalid_call_restores_full_marks():
    """The attempt/observation split must still credit the retry path."""
    task = _tool_task("T-2")
    oracle = task.oracle
    replies = [
        _call(1, "get_logs", {"service": "api-gateway", "minutes": 15, "limit": 50}),
        _call(2, "get_logs", oracle["golden_calls"][0]["arguments"]),
        _call(3, "get_metric", oracle["golden_calls"][1]["arguments"]),
        _call(4, "get_metric", oracle["golden_calls"][2]["arguments"]),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(), ScriptedProvider(replies)).run_task(task)
    assert run.transcript.tool_log[0]["error"] is True
    assert run.score.earned == task.max_points == 2.0


def test_tool_loop_recovers_from_a_transient_failure():
    """The flaky read fails first, so the value only exists after a retry."""
    task = _tool_task("T-6")
    oracle = task.oracle
    flaky = oracle["flaky_tool"]
    failures = oracle["transient_failures"][flaky]

    replies = [_call(index, call["name"], call["arguments"]) for index, call in enumerate(oracle["golden_calls"], 1)]
    replies.append(core.ModelReply(text=oracle["final_answer"]))
    run = Runner(RunConfig(), ScriptedProvider(replies)).run_task(task)

    errors = [c for c in run.transcript.tool_log if c["name"] == flaky and c["error"]]
    assert len(errors) == failures
    assert all("upstream temporarily unavailable" in c["result"] for c in errors)
    assert [c["name"] for c in run.transcript.tool_log].count(flaky) == failures + 1
    assert run.transcript.tool_log[-1]["error"] is False
    assert run.score.earned == task.max_points == 2.0

    # Giving up right after the error must not collect the answer or retry points.
    give_up = Runner(
        RunConfig(),
        ScriptedProvider([_call(1, flaky, {"path": oracle["config_path"]}), core.ModelReply(text="инструмент недоступен")]),
    ).run_task(task)
    assert give_up.score.earned == 0.5
    assert not any(c.passed and c.name in ("retried_after_failure", "real_value") for c in give_up.score.checks)


def test_tool_loop_refreshes_a_stale_metric():
    """The prompt quotes a stale dashboard value; only refresh=true is current."""
    task = _tool_task("T-7")
    oracle = task.oracle
    stale, fresh = oracle["stale_value"], oracle["fresh_value"]
    lookup = {"service": oracle["service"], "metric": oracle["metric"], "minutes": 15}

    replies = [
        _call(1, "get_metric", dict(lookup)),
        _call(2, "get_metric", {**lookup, "refresh": True}),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(), ScriptedProvider(replies)).run_task(task)
    assert stale in run.transcript.tool_log[0]["result"], "без refresh отдаётся устаревший снимок"
    assert "refresh=true" in run.transcript.tool_log[0]["result"], "подсказка о кэше обязательна"
    assert fresh in run.transcript.tool_log[1]["result"]
    assert run.score.earned == task.max_points == 2.0

    # A verbatim echo of the stale dashboard number is not an answer: without a
    # refresh call the content check must stay at zero.
    stale_only = Runner(
        RunConfig(),
        ScriptedProvider(
            [_call(1, "get_metric", dict(lookup)), core.ModelReply(text=f"queue_depth = {stale}")]
        ),
    ).run_task(task)
    assert stale_only.score.earned == 0.5
    assert not any(c.passed and c.name in ("refreshed", "fresh_value") for c in stale_only.score.checks)

    # Mentioning the stale number next to the fresh one is the natural answer
    # and must pass.
    both = Runner(
        RunConfig(),
        ScriptedProvider(
            [
                _call(1, "get_metric", {**lookup, "refresh": True}),
                core.ModelReply(text=f"Свежее значение {fresh}; на дашборде устаревшее {stale}."),
            ]
        ),
    ).run_task(task)
    assert both.score.earned == task.max_points


def test_tool_loop_full_score_on_the_golden_cache_cleanup():
    """T-4 golden calls must find the cache: search_code matches file paths.

    The cache files are opaque binary blobs whose *content* never mentions
    ``build-cache``; before the path match the golden ``search_code`` query
    returned ``no matches`` and no model could solve T-4 at full score.
    """
    task = _tool_task("T-4")
    oracle = task.oracle
    replies = [_call(index, call["name"], call["arguments"]) for index, call in enumerate(oracle["golden_calls"], 1)]
    replies.append(core.ModelReply(text=oracle["final_answer"]))
    run = Runner(RunConfig(), ScriptedProvider(replies)).run_task(task)
    found = run.transcript.tool_log[0]["result"]
    assert all(path in found for path in oracle["cache_files"]), "golden search must list every cache file"
    assert run.score.earned == task.max_points == 2.0



def _profile_task(task_id: str, seed: int, profile: str) -> core.TaskInstance:
    import random

    for task in tools.generate(random.Random(seed), RunConfig(profile=profile)):
        if task.id == task_id:
            return task
    raise AssertionError(task_id)


def _replay(task: core.TaskInstance) -> list[core.ModelReply]:
    replies = [_call(index, call["name"], call["arguments"]) for index, call in enumerate(task.oracle["golden_calls"], 1)]
    replies.append(core.ModelReply(text=task.oracle["final_answer"]))
    return replies


def test_pilot_and_stretch_expose_only_their_own_tool_scenarios():
    import random

    baseline = tools.generate(random.Random(3), RunConfig())
    pilot = tools.generate(random.Random(3), RunConfig(profile="pilot"))
    stretch = tools.generate(random.Random(3), RunConfig(profile="stretch-v1"))
    assert [task.id for task in baseline] == [f"T-{index}" for index in range(1, 8)]
    assert [task.id for task in pilot] == ["T-8"]
    assert [task.id for task in stretch] == [f"T-{index}" for index in range(1, 12)]
    caps = {task.id: task.oracle["call_budget"] for task in pilot + stretch}
    assert caps["T-8"] <= 10 and caps["T-9"] <= 8 and caps["T-10"] <= 8 and caps["T-11"] <= 8


def test_t8_transaction_commits_only_after_a_passing_validation():
    task = _profile_task("T-8", 3, "pilot")
    oracle = task.oracle
    assert oracle["test_outcome"] == "PASS"
    run = Runner(RunConfig(agent=False), ScriptedProvider(_replay(task))).run_task(task)
    assert run.score.earned == task.max_points == 2.0
    assert run.transcript.workspace_after[oracle["config_path"]] == oracle["after_content"]
    assert run.transcript.workspace_after[oracle["backup_path"]] == oracle["before_content"]

    # A single passing run_tests is not a transaction: without diff/deploy the
    # version never moves.
    no_deploy = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "get_config_state", {}),
                _call(2, "write_file", {"path": oracle["config_path"], "content": oracle["after_content"]}),
                _call(3, "run_tests", {"target": "service"}),
                core.ModelReply(text=f"PASS, replicas {oracle['documented']}, версия {oracle['config_version']}"),
            ]
        ),
    ).run_task(task)
    assert no_deploy.score.earned == 1.5
    assert not any(check.passed and check.name == "transaction_outcome" for check in no_deploy.score.checks)


def test_t8_repeated_overwrite_does_not_bump_the_version():
    """A second write of the same bytes is not a new revision to commit."""
    task = _profile_task("T-8", 3, "pilot")
    oracle = task.oracle
    replies = _replay(task)
    replies[-1:] = [
        _call(6, "write_file", {"path": oracle["config_path"], "content": oracle["after_content"]}),
        _call(7, "deploy", {}),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)
    assert run.score.earned == task.max_points
    assert run.transcript.tool_log[-1]["error"] is True
    assert "content matches version" in run.transcript.tool_log[-1]["result"]
    assert run.transcript.workspace_after[oracle["config_path"]] == oracle["after_content"]


def test_t8_transaction_rolls_back_after_a_failing_validation():
    task = _profile_task("T-8", 1, "pilot")
    oracle = task.oracle
    assert oracle["test_outcome"] == "FAIL"
    run = Runner(RunConfig(agent=False), ScriptedProvider(_replay(task))).run_task(task)
    assert run.score.earned == task.max_points == 2.0
    assert run.transcript.workspace_after[oracle["config_path"]] == oracle["before_content"]

    # The deploy-after-FAIL mutant keeps a version bump and the unvalidated
    # file, so it can never reach full marks.
    deployed = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "get_config_state", {}),
                _call(2, "write_file", {"path": oracle["config_path"], "content": oracle["after_content"]}),
                _call(3, "run_tests", {"target": "service"}),
                _call(4, "git_diff", {}),
                _call(5, "deploy", {}),
                core.ModelReply(text=f"Tests FAIL, но задеплоил версию {oracle['config_version'] + 1}"),
            ]
        ),
    ).run_task(task)
    assert deployed.score.earned == 1.5
    assert deployed.transcript.workspace_after[oracle["config_path"]] == oracle["after_content"]
    assert not any(check.passed and check.name == "transaction_outcome" for check in deployed.score.checks)

    garbage = Runner(RunConfig(agent=False), ScriptedProvider([core.ModelReply(text="не знаю")])).run_task(task)
    assert garbage.score.earned == 0.0


def test_t8_is_seed_deterministic_and_keeps_valid_boundaries():
    outcomes = set()
    for seed in range(1, 6):
        first = _profile_task("T-8", seed, "pilot")
        second = _profile_task("T-8", seed, "pilot")
        assert (first.prompt, first.workspace, first.oracle) == (second.prompt, second.workspace, second.oracle)
        assert first.oracle["documented"] != first.oracle["before"]
        assert len(first.workspace) == 4
        assert len(first.oracle["golden_calls"]) <= first.oracle["call_budget"]
        assert first.oracle["before_content"] in first.workspace["config/service.yaml"]
        assert first.oracle["test_outcome"] in ("PASS", "FAIL")
        outcomes.add(first.oracle["test_outcome"])
        replay = Runner(RunConfig(agent=False), ScriptedProvider(_replay(first))).run_task(first)
        assert replay.score.earned == first.max_points, (seed, [c.name for c in replay.score.checks if not c.passed])
    assert outcomes == {"PASS", "FAIL"}


def test_t9_keeps_one_effect_when_the_commit_already_landed():
    task = _profile_task("T-9", 1, "stretch-v1")
    oracle = task.oracle
    assert oracle["migration_branch"] == "after_commit"
    run = Runner(RunConfig(agent=False), ScriptedProvider(_replay(task))).run_task(task)
    assert run.score.earned == task.max_points == 2.0
    ledger = run.transcript.workspace_after[oracle["ledger_path"]]
    assert ledger.count(oracle["migration_id"]) == 1
    assert f"version: {oracle['migration_version'] + 1}" in run.transcript.workspace_after[oracle["state_path"]]

    # Blind retry on a non-idempotent operation doubles the effect.
    retry = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "get_migration_state", {}),
                _call(2, "apply_migration", {"migration_id": oracle["migration_id"]}),
                _call(3, "apply_migration", {"migration_id": oracle["migration_id"]}),
                core.ModelReply(text=f"применил дважды, версия {oracle['migration_version'] + 2}"),
            ]
        ),
    ).run_task(task)
    assert retry.score.earned == 1.0
    assert not any(check.passed and check.name == "single_effect_consistent_version" for check in retry.score.checks)
    assert retry.transcript.workspace_after[oracle["ledger_path"]].count(oracle["migration_id"]) == 2


def test_t9_repeats_after_a_timeout_before_the_commit():
    task = _profile_task("T-9", 2, "stretch-v1")
    oracle = task.oracle
    assert oracle["migration_branch"] == "before_commit"
    run = Runner(RunConfig(agent=False), ScriptedProvider(_replay(task))).run_task(task)
    assert run.score.earned == task.max_points == 2.0
    assert run.transcript.workspace_after[oracle["ledger_path"]].count(oracle["migration_id"]) == 1

    # Retrying without reading the state still lands one effect, but the
    # decision is unverified: the half-point for state evidence is not enough
    # for the full check.
    blind = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "get_migration_state", {}),
                _call(2, "apply_migration", {"migration_id": oracle["migration_id"]}),
                _call(3, "apply_migration", {"migration_id": oracle["migration_id"]}),
                core.ModelReply(text=f"применил, версия {oracle['migration_version'] + 1}"),
            ]
        ),
    ).run_task(task)
    assert blind.score.earned == 1.0

    garbage = Runner(RunConfig(agent=False), ScriptedProvider([core.ModelReply(text="не знаю")])).run_task(task)
    assert garbage.score.earned == 0.0


def test_t9_is_seed_deterministic_and_covers_both_branches():
    branches = set()
    for seed in range(1, 6):
        first = _profile_task("T-9", seed, "stretch-v1")
        second = _profile_task("T-9", seed, "stretch-v1")
        assert (first.prompt, first.workspace, first.oracle) == (second.prompt, second.workspace, second.oracle)
        assert len(first.workspace) == 4
        assert len(first.oracle["golden_calls"]) <= first.oracle["call_budget"] <= 8
        assert first.oracle["initial_ledger"] == first.workspace[first.oracle["ledger_path"]]
        assert str(first.oracle["migration_id"]) not in first.oracle["initial_ledger"]
        branches.add(first.oracle["migration_branch"])
        replay = Runner(RunConfig(agent=False), ScriptedProvider(_replay(first))).run_task(first)
        assert replay.score.earned == first.max_points, (seed, [c.name for c in replay.score.checks if not c.passed])
    assert branches == {"before_commit", "after_commit"}


def test_the_tool_budget_half_point_is_forfeited_on_overrun():
    for task_id, seed, profile in (("T-8", 3, "pilot"), ("T-9", 1, "stretch-v1"), ("T-12", 1, "research-v1")):
        task = _profile_task(task_id, seed, profile)
        replies = _replay(task)
        filler = {
            "T-8": "get_config_state",
            "T-9": "get_migration_state",
            "T-12": "get_release_state",
        }[task_id]
        while len(replies) <= task.oracle["call_budget"] + 1:
            replies.insert(len(replies) - 1, _call(len(replies), filler, {}))
        run = Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)
        assert run.score.earned == 1.5
        assert not any(check.passed and check.name == "call_budget" for check in run.score.checks)


def _outbox(task: core.TaskInstance, run) -> list[str]:
    prefix = len(task.oracle["initial_outbox"].splitlines())
    return [line for line in run.transcript.workspace_after[task.oracle["outbox_path"]].splitlines()[prefix:] if line]


def _shared_name(task: core.TaskInstance) -> str:
    """The display name both addressees carry."""
    return next(row["name"] for row in task.oracle["contacts"] if row["id"] == task.oracle["known_contact_id"])


def test_n3_golden_delivers_both_addressees_and_asks_only_about_the_missing_id():
    task = _profile_task("T-10", 1, "stretch-v1")
    oracle = task.oracle
    run = Runner(RunConfig(agent=False), ScriptedProvider(_replay(task))).run_task(task)

    assert run.score.earned == task.max_points == 2.0
    assert [check.name for check in run.score.checks] == [
        "no_question_for_known",
        "missing_id_before_write",
        "both_delivered",
    ]
    lines = _outbox(task, run)
    assert [line.split("|", 1)[0].strip() for line in lines] == [
        oracle["known_contact_id"],
        oracle["missing_contact_id"],
    ]
    assert oracle["marker_one"] in lines[0].casefold() and oracle["marker_two"] in lines[1].casefold()
    asks = [call for call in run.transcript.tool_log if call["name"] == "ask_user"]
    assert len(asks) == 1 and oracle["missing_contact_id"] in asks[0]["result"]
    # The clarification must not leak the answer: the id is unreachable from the
    # files, so a model can only guess it or ask.
    assert all(oracle["missing_contact_id"] not in content for content in task.workspace.values())


def test_n3_the_read_twin_is_not_a_delivery():
    """`draft_message` writes state but delivers nothing, so it is not an effect."""
    task = _profile_task("T-10", 2, "stretch-v1")
    oracle = task.oracle
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "find_contact", {"query": _shared_name(task)}),
                _call(2, "read_contact", {"contact_id": oracle["known_contact_id"]}),
                _call(3, "draft_message", {"contact_id": oracle["known_contact_id"], "text": "отчёт"}),
                _call(4, "draft_message", {"contact_id": oracle["archived_contact_id"], "text": "договор"}),
                core.ModelReply(text=oracle["final_answer"]),
            ]
        ),
    ).run_task(task)

    assert _outbox(task, run) == []
    assert all(not check.passed for check in run.score.checks)
    assert run.score.earned == 0.0


def test_n3_a_guessed_archived_id_is_delivered_but_not_the_contractor():
    """The archived namesake is a real contact, so the send succeeds — wrongly."""
    task = _profile_task("T-10", 3, "stretch-v1")
    oracle = task.oracle
    replies = [
        _call(1, "find_contact", {"query": _shared_name(task)}),
        _call(2, "send_message", {"contact_id": oracle["known_contact_id"], "text": oracle["marker_one"]}),
        _call(3, "send_message", {"contact_id": oracle["archived_contact_id"], "text": oracle["marker_two"]}),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)

    assert [line.split("|", 1)[0].strip() for line in _outbox(task, run)] == [
        oracle["known_contact_id"],
        oracle["archived_contact_id"],
    ]
    checks = {check.name: check.passed for check in run.score.checks}
    assert checks["no_question_for_known"] is True
    assert checks["missing_id_before_write"] is False
    assert checks["both_delivered"] is False
    assert run.score.earned == 0.5


def test_n3_a_write_before_the_clarification_is_not_repaired_by_a_late_ask():
    """Without the id the send can only guess, and a late answer adds a third line."""
    task = _profile_task("T-10", 4, "stretch-v1")
    oracle = task.oracle
    replies = [
        _call(1, "find_contact", {"query": _shared_name(task)}),
        _call(2, "send_message", {"contact_id": oracle["known_contact_id"], "text": oracle["marker_one"]}),
        _call(3, "send_message", {"contact_id": oracle["archived_contact_id"], "text": oracle["marker_two"]}),
        _call(4, "ask_user", {"question": "какой у него id?"}),
        _call(5, "send_message", {"contact_id": oracle["missing_contact_id"], "text": oracle["marker_two"]}),
        core.ModelReply(text=oracle["final_answer"]),
    ]
    run = Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)

    assert len(_outbox(task, run)) == 3, "угаданная отправка уже состоялась, поздний вопрос её не отменяет"
    checks = {check.name: check.passed for check in run.score.checks}
    assert checks["no_question_for_known"] is True
    assert checks["missing_id_before_write"] is False
    assert checks["both_delivered"] is False
    assert run.score.earned == 0.5


def test_n3_the_ask_waits_for_the_directory_and_is_answered_once():
    task = _profile_task("T-10", 5, "stretch-v1")
    oracle = task.oracle
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "ask_user", {"question": "какой id?"}),
                _call(2, "find_contact", {"query": _shared_name(task)}),
                _call(3, "ask_user", {"question": "какой id?"}),
                _call(4, "ask_user", {"question": "ещё раз?"}),
                core.ModelReply(text="готово"),
            ]
        ),
    ).run_task(task)

    first, directory, second, third = run.transcript.tool_log
    assert first["error"] is True and "find_contact" in first["result"]
    assert directory["name"] == "find_contact" and directory["error"] is False
    assert oracle["missing_contact_id"] in second["result"]
    assert "больше нет" in third["result"]


def test_n3_is_seed_deterministic_and_keeps_the_id_out_of_the_files():
    for seed in range(1, 6):
        first = _profile_task("T-10", seed, "stretch-v1")
        second = _profile_task("T-10", seed, "stretch-v1")
        assert (first.prompt, first.workspace, first.oracle) == (second.prompt, second.workspace, second.oracle)
        oracle = first.oracle
        ids = {row["id"] for row in oracle["contacts"] if row["id"]}
        assert len(ids) == 4, "справочник: известный, архивный и два отвлекающих id"
        assert oracle["missing_contact_id"] not in ids
        assert sum(1 for row in oracle["contacts"] if not row["id"]) == 1
        assert oracle["known_contact_id"] in first.workspace["team/project.md"]
        assert all(oracle["missing_contact_id"] not in content for content in first.workspace.values())
        assert len(oracle["golden_calls"]) <= oracle["call_budget"] <= 8
        assert first.fresh_after == tools.FRESH_AFTER
        replay = Runner(RunConfig(agent=False), ScriptedProvider(_replay(first))).run_task(first)
        assert replay.score.earned == first.max_points, (seed, [c.name for c in replay.score.checks if not c.passed])


def test_n4_golden_keeps_one_effect_across_the_version_bump():
    task = _profile_task("T-11", 1, "stretch-v1")
    oracle = task.oracle
    run = Runner(RunConfig(agent=False), ScriptedProvider(_replay(task))).run_task(task)

    assert run.score.earned == task.max_points == 2.0
    journal = run.transcript.workspace_after[oracle["journal_path"]].splitlines()[2:]
    assert len([line for line in journal if line.startswith(f"create {oracle['slot']} ")]) == 1
    states = [call for call in run.transcript.tool_log if call["name"] == "booking_state"]
    assert "версия: 2" in states[1]["result"], "смена версии приходит в расписании, а не по запросу"
    assert "отмена" in states[2]["result"] and oracle["cancelled_id"] in states[2]["result"]
    assert run.transcript.workspace_after[oracle["journal_path"]] == (
        oracle["initial_journal"] + journal[0] + "\n"
    ), "отмена чужого слота новых записей не даёт"

    # A version conflict is answered with the current version, and the repeat
    # has to be sent with it.
    stale = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "booking_state", {"slot": oracle["slot"]}),
                _call(2, "create_booking", {"slot": oracle["slot"], "seats": 4, "idempotency_key": oracle["idempotency_key"], "version": 1}),
                _call(3, "create_booking", {"slot": oracle["slot"], "seats": 4, "idempotency_key": oracle["idempotency_key"], "version": 1}),
                core.ModelReply(text="готово"),
            ]
        ),
    ).run_task(task)
    assert stale.transcript.tool_log[2]["error"] is True
    assert "конфликт версии" in stale.transcript.tool_log[2]["result"]


def test_n4_a_repeat_with_a_new_key_spends_the_seats_twice():
    task = _profile_task("T-11", 2, "stretch-v1")
    oracle = task.oracle
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "booking_state", {"slot": oracle["slot"]}),
                _call(2, "create_booking", {"slot": oracle["slot"], "seats": 4, "idempotency_key": oracle["idempotency_key"], "version": 1}),
                _call(3, "booking_state", {"slot": oracle["slot"]}),
                _call(4, "create_booking", {"slot": oracle["slot"], "seats": 4, "idempotency_key": "bk-другой", "version": 2}),
                _call(5, "booking_state", {"slot": oracle["slot"]}),
                core.ModelReply(text=f"остаток -1, версия слота 2, id r-{oracle['reservation_base'] + 1}, две резервации"),
            ]
        ),
    ).run_task(task)

    checks = {check.name: check.passed for check in run.score.checks}
    assert checks["single_effect_consistent_remaining"] is False
    assert checks["state_before_write"] is True
    assert "-1" in run.transcript.tool_log[4]["result"]
    assert run.score.earned == 1.0


def test_n4_reported_success_without_effect_is_a_half():
    task = _profile_task("T-11", 3, "stretch-v1")
    oracle = task.oracle
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "booking_state", {"slot": oracle["slot"]}),
                core.ModelReply(text="Зарезервировано 4 места, остаток 3, версия слота 2"),
            ]
        ),
    ).run_task(task)

    assert run.transcript.workspace_after[oracle["journal_path"]] == oracle["initial_journal"]
    checks = {check.name: check.passed for check in run.score.checks}
    assert checks == {"state_before_write": True, "single_effect_consistent_remaining": False, "fresh_version_no_stray": False}
    assert run.score.earned == 0.5


def test_n4_the_right_remaining_with_an_extra_cancellation_is_not_full():
    """Cancel-and-rebook lands on three left with a second record."""
    task = _profile_task("T-11", 4, "stretch-v1")
    oracle = task.oracle
    created = f"r-{oracle['reservation_base'] + 1}"
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "booking_state", {"slot": oracle["slot"]}),
                _call(2, "create_booking", {"slot": oracle["slot"], "seats": 4, "idempotency_key": oracle["idempotency_key"], "version": 1}),
                _call(3, "booking_state", {"slot": oracle["slot"]}),
                _call(4, "cancel_booking", {"reservation_id": created}),
                _call(5, "booking_state", {"slot": oracle["slot"]}),
                _call(6, "create_booking", {"slot": oracle["slot"], "seats": 4, "idempotency_key": "bk-новый", "version": 2}),
                _call(7, "booking_state", {"slot": oracle["slot"]}),
                core.ModelReply(text="остаток 3, версия слота 2, одна резервация"),
            ]
        ),
    ).run_task(task)

    assert "остаток: 3" in run.transcript.tool_log[6]["result"]
    checks = {check.name: check.passed for check in run.score.checks}
    assert checks["single_effect_consistent_remaining"] is False
    assert checks["fresh_version_no_stray"] is False
    assert run.score.earned == 0.5


def test_n4_is_seed_deterministic_and_holds_valid_boundaries():
    for seed in range(1, 6):
        first = _profile_task("T-11", seed, "stretch-v1")
        second = _profile_task("T-11", seed, "stretch-v1")
        assert (first.prompt, first.workspace, first.oracle) == (second.prompt, second.workspace, second.oracle)
        oracle = first.oracle
        assert oracle["seats_free"] == 7 and oracle["request"] == 4
        assert oracle["seats_free"] - oracle["request"] == 3
        assert oracle["initial_journal"] == first.workspace[oracle["journal_path"]]
        assert oracle["cancelled_id"] in oracle["initial_journal"], "отмена относится к существующей записи"
        assert oracle["slot"] not in oracle["initial_journal"], "наш слот ещё свободен"
        assert oracle["slot"] != oracle["other_slot"]
        assert oracle["idempotency_key"] in first.prompt
        assert oracle["reservation_base"]
        assert len(oracle["golden_calls"]) <= oracle["call_budget"] <= 8
        assert first.fresh_after == tools.FRESH_AFTER
        replay = Runner(RunConfig(agent=False), ScriptedProvider(_replay(first))).run_task(first)
        assert replay.score.earned == first.max_points, (seed, [c.name for c in replay.score.checks if not c.passed])


def test_t12_recovery_from_a_stale_snapshot_keeps_one_effect():
    task = _profile_task("T-12", 1, "research-v1")
    oracle = task.oracle
    run = Runner(RunConfig(agent=False), ScriptedProvider(_replay(task))).run_task(task)

    assert run.score.earned == task.max_points == 2.0
    assert run.transcript.workspace_after[oracle["log_path"]].count(oracle["build"]) == 1
    assert f"revision: {oracle['release_revision'] + 2}" in run.transcript.workspace_after[oracle["state_path"]]
    assert f"deployed: {oracle['build']}" in run.transcript.workspace_after[oracle["state_path"]]

    # The stale snapshot is refused, and the hotfix that made it stale is
    # announced on that same call instead of being inferred from the answer.
    conflict = run.transcript.tool_log[1]
    assert conflict["error"] is True
    assert "hotfix" in conflict["result"] and "конфликт ревизии" in conflict["result"]
    timeout = run.transcript.tool_log[3]
    assert timeout["error"] is True and "timeout после commit" in timeout["result"]


def test_t12_a_blind_repeat_after_the_commit_breaks_the_recovery():
    """Repeating "just to be sure" costs a revision and a journal line."""
    task = _profile_task("T-12", 2, "research-v1")
    oracle = task.oracle
    replies = _replay(task)
    repeat = _call(
        6,
        "publish_release",
        {"build_id": oracle["build"], "revision": oracle["release_revision"] + 2},
    )
    replies = [*replies[:5], repeat, *replies[5:]]
    run = Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)

    assert run.score.earned == 1.0
    checks = {check.name: check.passed for check in run.score.checks}
    assert checks["single_effect_consistent_state"] is False
    assert checks["stale_snapshot_refreshed"] is True
    log = run.transcript.workspace_after[oracle["log_path"]]
    assert log.count(oracle["build"]) == 2
    assert f"revision: {oracle['release_revision'] + 3}" in run.transcript.workspace_after[oracle["state_path"]]


def test_t12_recovery_by_an_early_refresh_needs_no_conflict():
    """The hotfix fires before the second call, so an early refresh already sees it."""
    task = _profile_task("T-12", 5, "research-v1")
    oracle = task.oracle
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "get_release_state", {}),
                _call(2, "get_release_state", {"refresh": True}),
                _call(3, "publish_release", {"build_id": oracle["build"], "revision": oracle["release_revision"] + 1}),
                _call(4, "get_release_state", {"refresh": True}),
                core.ModelReply(text=oracle["final_answer"]),
            ]
        ),
    ).run_task(task)

    assert f"ревизия {oracle['release_revision'] + 1}" in run.transcript.tool_log[1]["result"]
    assert run.transcript.tool_log[2]["error"] is True
    assert "timeout после commit" in run.transcript.tool_log[2]["result"]
    assert run.score.earned == task.max_points
    assert run.transcript.workspace_after[oracle["log_path"]].count(oracle["build"]) == 1


def test_t12_trusting_the_stale_snapshot_never_commits():
    """Repeating the cached revision is refused every time: no effect at all."""
    task = _profile_task("T-12", 3, "research-v1")
    oracle = task.oracle
    stale = {"build_id": oracle["build"], "revision": oracle["release_revision"]}
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "get_release_state", {}),
                _call(2, "publish_release", dict(stale)),
                _call(3, "publish_release", dict(stale)),
                core.ModelReply(text=f"опубликовано, ревизия {oracle['release_revision']}"),
            ]
        ),
    ).run_task(task)

    publishes = [call for call in run.transcript.tool_log if call["name"] == "publish_release"]
    assert len(publishes) == 2 and all(call["error"] for call in publishes)
    assert all("конфликт ревизии" in call["result"] for call in publishes)
    assert run.transcript.workspace_after[oracle["log_path"]] == oracle["initial_log"]
    assert f"revision: {oracle['release_revision']}" in run.transcript.workspace_after[oracle["state_path"]]
    checks = {check.name: check.passed for check in run.score.checks}
    assert checks == {"stale_snapshot_refreshed": False, "single_effect_consistent_state": False, "call_budget": True}
    assert run.score.earned == 0.5


def test_t12_recovery_claimed_without_any_effect_is_a_half():
    task = _profile_task("T-12", 4, "research-v1")
    oracle = task.oracle
    run = Runner(
        RunConfig(agent=False),
        ScriptedProvider(
            [
                _call(1, "get_release_state", {}),
                core.ModelReply(
                    text=f"опубликовал {oracle['build']}, ревизия {oracle['release_revision'] + 2}, эффект один"
                ),
            ]
        ),
    ).run_task(task)

    assert run.transcript.workspace_after[oracle["log_path"]] == oracle["initial_log"]
    assert run.score.earned == 0.5


def test_t12_recovery_is_seed_deterministic_and_declares_the_schedule():
    for seed in range(1, 6):
        first = _profile_task("T-12", seed, "research-v1")
        second = _profile_task("T-12", seed, "research-v1")
        assert (first.prompt, first.workspace, first.oracle) == (second.prompt, second.workspace, second.oracle)
        oracle = first.oracle
        assert oracle["build"] not in oracle["initial_log"]
        assert oracle["current_build"] in oracle["initial_log"]
        assert oracle["initial_log"] == first.workspace[oracle["log_path"]]
        assert f"revision: {oracle['release_revision']}\n" in first.workspace[oracle["state_path"]]
        assert all(oracle["build"] not in content for content in first.workspace.values())
        # Both halves of the schedule are announced, not left to chance.
        assert "hotfix" in first.prompt and "timeout" in first.prompt
        assert "refresh=true" in first.prompt
        assert len(oracle["golden_calls"]) <= oracle["call_budget"] <= 8
        assert first.private is True and first.fresh_after == tools.FRESH_AFTER
        replay = Runner(RunConfig(agent=False), ScriptedProvider(_replay(first))).run_task(first)
        assert replay.score.earned == first.max_points, (seed, [c.name for c in replay.score.checks if not c.passed])


def test_n3_and_n4_are_stretch_only_and_split_the_category_total():
    module = core.REGISTRY["D"]
    assert module.expected_ids == tools.IDS
    assert module.expected_ids_for() == tools.IDS
    assert module.expected_ids_for("pilot") == tools.PILOT_IDS == ("T-8",)
    assert module.expected_ids_for("stretch-v1") == tools.ALL_IDS
    assert tools.ALL_IDS == tuple(f"T-{index}" for index in range(1, 12))

    import random

    stretch = tools.generate(random.Random(3), RunConfig(profile="stretch-v1"))
    apply_profile_weights(stretch, "stretch-v1")
    assert sum(task.max_points for task in stretch) == core.CATEGORY_POINTS["D"] == 14.0
    shares = [task.max_points for task in stretch]
    assert max(shares) - min(shares) <= 0.0005, "вес распределён ровно, без ручных значений"

    weighted = next(task for task in stretch if task.id == "T-10")
    replay = Runner(RunConfig(agent=False), ScriptedProvider(_replay(weighted))).run_task(weighted)
    assert replay.score.earned == weighted.max_points
    assert abs(replay.score.possible - weighted.max_points) < 1e-9


def test_tool_budget_is_enforced():
    """A call budget stops the loop after exactly that many executed tools."""
    task = _tool_task("T-1")
    replies = [_call(index, "read_file", {"path": "README.md"}) for index in range(30)]
    cfg = RunConfig()
    cfg.max_tool_calls = 3
    run = Runner(cfg, ScriptedProvider(replies)).run_task(task)
    assert len(run.transcript.tool_log) == 3
    assert "бюджет вызовов инструментов исчерпан (3)" in run.transcript.extra.get("notes", [])


def test_tool_budget_keeps_a_batch_that_overshoots_by_one():
    """A batch of 4 within a 3-call budget must run 3 calls, not zero.

    The model plans a read of every file in one reply. Dropping the whole batch
    left I-1 with no data and no answer; only the overflow is cut.
    """
    task = _tool_task("T-1")
    batch = core.ModelReply(
        text="",
        tool_calls=[
            {"id": f"c{i}", "name": "read_file", "arguments": json.dumps({"path": "README.md"})}
            for i in range(4)
        ],
    )
    replies = [batch, core.ModelReply(text="готово")]
    cfg = RunConfig()
    cfg.max_tool_calls = 3
    run = Runner(cfg, ScriptedProvider(replies)).run_task(task)
    assert len(run.transcript.tool_log) == 3
    notes = run.transcript.extra.get("notes", [])
    assert any("батч урезан" in note for note in notes), notes
    # The budget-capped task must still get a tool-less turn for the answer,
    # not end on the last tool batch — and that turn states the budget in words
    # (the stand pastes the request markup into content instead of answering).
    assert run.transcript.turns[-1].reply.text == "готово"
    assert "Бюджет вызовов инструментов исчерпан" in run.transcript.turns[-1].messages[-1]["content"]


def test_total_budget_stops_before_the_next_request():
    """R11: a spent generation budget ends the loop even mid tool batch.

    With a total budget of 2 the first reply already exhausts it. The tool it
    asked for is still executed, but no further request may go out — the old
    ``continue`` on tool room ignored the token sum and kept calling.
    """
    task = _tool_task("T-1")
    reply = _call(1, "read_file", {"path": "README.md"})
    reply.completion_tokens = 2
    provider = ScriptedProvider([reply, _call(2, "read_file", {"path": "README.md"}), core.ModelReply(text="поздно")])
    cfg = RunConfig()
    cfg.max_total_tokens = 2
    run = Runner(cfg, provider).run_task(task)
    assert len(provider.calls) == 1, "после исчерпания бюджета запросов быть не должно"
    assert len(run.transcript.tool_log) == 1, "инструмент из последнего ответа обязан выполниться"
    assert "бюджет генерации исчерпан (2)" in run.transcript.extra.get("notes", [])


def test_a_budget_spent_on_a_tool_batch_is_exhausted_not_ok():
    """P3: the post-batch break left the row ``ok`` with no answer at all.

    The tool the last reply asked for still runs, but its result never reaches
    the model: the row must carry the same ``resource_exhausted`` kind the
    identical stop gets at the loop top, not a silent wrong answer.
    """
    task = _tool_task("T-1")
    reply = _call(1, "read_file", {"path": "README.md"})
    reply.completion_tokens = 2
    provider = ScriptedProvider([reply, core.ModelReply(text="поздно")])
    cfg = RunConfig()
    cfg.max_total_tokens = 2
    run = Runner(cfg, provider).run_task(task)

    assert len(provider.calls) == 1, "ответ к батчу не запрашивается: бюджет уже исчерпан"
    assert len(run.transcript.tool_log) == 1
    assert "до ответа" in (run.transcript.error or "")
    assert run.transcript.extra["failure"] == "resource_exhausted"
    assert run.score.status == "error"
    assert core.outcome_of(run.score) == "resource_exhausted"


def test_a_pending_dialogue_message_still_overruns_a_spent_budget():
    """The same rule from the batch: an unshown message is worth the overrun."""
    task = _tool_task("T-1")
    task.dialogue = ["Поправка: читай только README."]
    reply = _call(1, "read_file", {"path": "README.md"})
    reply.completion_tokens = 2
    provider = ScriptedProvider(
        [
            reply,
            core.ModelReply(text="готово", completion_tokens=2),
            core.ModelReply(text="итог", completion_tokens=2),
        ]
    )
    cfg = RunConfig()
    cfg.max_total_tokens = 2
    run = Runner(cfg, provider).run_task(task)

    assert len(provider.calls) == 3, "непредъявленная поправка обязана дойти до модели"
    assert run.transcript.extra["total_budget_overrun"] is True
    assert run.transcript.error is None
    assert run.score.status == "ok"


def test_a_task_level_deadline_stops_before_the_next_request():
    """P3: ``request_wall_clock`` is per request, so the task needs its own.

    Every turn used to get a fresh full wall clock, so a tool loop could run
    ``call_budget`` times the ceiling and the ``stream_options`` retry doubled a
    single turn again. One absolute deadline is armed when the task starts; the
    turns share it, and a spent budget is a stop condition with a note.
    """
    task = _tool_task("T-1")
    replies = [_call(index, "read_file", {"path": "README.md"}) for index in range(1, 6)]
    replies.append(core.ModelReply(text="готово"))
    provider = _SlowProvider(replies)
    cfg = RunConfig()
    cfg.request_wall_clock = 0.15
    cfg.task_wall_clock = 0.3
    started = time.monotonic()
    run = Runner(cfg, provider).run_task(task)
    elapsed = time.monotonic() - started

    assert len(provider.calls) == 2, "два запроса съели бюджет — третий не стартует"
    assert elapsed < 0.6, f"задание вышло за общий предел: {elapsed:.2f} с"
    assert "предел времени на задание 0.3 с исчерпан" in run.transcript.extra.get("notes", [])
    assert run.transcript.error and run.score.status == "error", "таймаут — неверный ответ, а не verified"
    assert len(run.transcript.tool_log) == 2, "инструменты уже полученных ответов не теряются"


class _SlowProvider(ScriptedProvider):
    """Spends more than the budget it is given, like a slow stand."""

    def chat(self, messages, tools=None, **kwargs):
        time.sleep(float(kwargs.get("wall_clock") or 0) * 1.2)
        return super().chat(messages, tools, **kwargs)


def test_the_task_deadline_uses_the_clock_it_is_compared_with(monkeypatch):
    """M01: ``task_deadline`` was also armed on ``perf_counter``.

    Under a shifted ``perf_counter`` domain the task deadline is ~1000 s away in
    ``monotonic`` terms, so the stop rule before a new request never fires and
    the whole scripted dialogue runs instead of the two requests the budget
    allows.
    """
    real_perf_counter = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real_perf_counter() + 1000.0)
    task = _tool_task("T-1")
    replies = [_call(index, "read_file", {"path": "README.md"}) for index in range(1, 6)]
    replies.append(core.ModelReply(text="готово"))
    provider = _SlowProvider(replies)
    cfg = RunConfig()
    cfg.request_wall_clock = 0.15
    cfg.task_wall_clock = 0.3
    run = Runner(cfg, provider).run_task(task)

    assert len(provider.calls) == 2, f"бюджет задания разрешил {len(provider.calls)} запросов"
    assert "предел времени на задание 0.3 с исчерпан" in run.transcript.extra.get("notes", [])


class _StreamOptionsHost(BaseHTTPRequestHandler):
    """Rejects ``stream_options`` once, then stalls a stream past the budget."""

    def log_message(self, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.attempts += 1  # type: ignore[attr-defined]
        if payload.get("stream_options"):
            time.sleep(0.3)
            body = json.dumps({"error": {"message": "unknown field stream_options"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        # 0.4 s into a budget that only has 0.2 s left.
        time.sleep(0.4)
        chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


def test_the_stream_options_retry_shares_the_original_budget():
    """The retry is a new request but not a new budget (P3).

    The stand rejected ``stream_options`` after 0.3 s of a 0.5 s ceiling; the
    retry used to get the full 0.5 s again and finished a 0.4 s stall. With the
    deadline carried through it has 0.2 s and is cut.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamOptionsHost)
    server.attempts = 0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", model="test-model", timeout=30
        )
        started = time.monotonic()
        with pytest.raises(ProviderError, match="предел времени"):
            provider.chat([{"role": "user", "content": "hi"}], wall_clock=0.5)
        elapsed = time.monotonic() - started
        assert server.attempts == 2, "повтор обязан состояться один раз"  # type: ignore[attr-defined]
        assert elapsed < 0.8, f"повтор получил свежий бюджет: {elapsed:.2f} с"
    finally:
        server.shutdown()
        server.server_close()


def test_a_tool_result_never_persists_the_endpoint_key():
    """A tool that prints the key must not leak it into the transcript (P3).

    The agent gets ``apiKey`` in ``models.json`` and ``DAILYBENCH_API_KEY`` in
    its environment, so ``cat``/``echo`` are ordinary cases. Every tool result
    is written to ``tool_log`` and to the next request, hence scrubbed once at
    the source: the configured value verbatim plus the generic ``Bearer``/``sk-``
    shapes.
    """
    task = _tool_task("T-1")
    secret = "taskkey-9f3c1b7a2d"
    task.workspace["secret.env"] = (
        f"DAILYBENCH_API_KEY={secret}\nAuthorization: Bearer abcdefghijklmnop\nnested=sk-nested-0123456789\n"
    )
    cfg = RunConfig(api_key=secret)
    run = Runner(
        cfg,
        ScriptedProvider([_call(1, "read_file", {"path": "secret.env"}), core.ModelReply(text="готово")]),
    ).run_task(task)

    logged = run.transcript.tool_log[0]["result"]
    assert secret not in logged
    assert "abcdefghijklmnop" not in logged and "sk-nested-0123456789" not in logged
    assert "***" in logged
    sent_back = [
        message["content"]
        for turn in run.transcript.turns
        for message in turn.messages
        if message.get("role") == "tool"
    ]
    assert sent_back and secret not in sent_back[-1], "в следующий запрос ключ тоже не попадает"


def test_the_answer_budget_counts_the_tool_schemas():
    """The schemas go out on every request, so the window must account for them.

    Without them the budget overshoots: the live runtime answered HTTP 400
    ("requested N output tokens, the window is M") on the whole T-category, and
    the request was rejected before the model saw a tool.
    """
    from dailybench.runner import _prompt_tokens

    task = _tool_task("T-1")
    messages = [{"role": "user", "content": task.prompt}]
    without = _prompt_tokens(messages)
    with_tools = _prompt_tokens(messages, task.tools)

    assert task.tools, "у сценария обязаны быть схемы инструментов"
    assert with_tools > without, "схемы инструментов обязаны входить в оценку входа"

    cfg = RunConfig(context_window=8192, unbounded_budget=True)
    assert cfg.budget_for("D", with_tools) < cfg.budget_for("D", without)


def test_the_estimated_prompt_over_the_window_is_rejected_before_the_call():
    """A prompt bigger than the window leaves the answer at the 256-token floor."""
    cfg = RunConfig(context_window=8192, unbounded_budget=True)
    assert cfg.budget_for("A", 1000) == 8192 - 1000 - cfg.context_reserve - 20
    assert cfg.budget_for("A", 8192) == 256


# --------------------------------------------------------------------------- #
# H08: the request carries the declared generation contract
# --------------------------------------------------------------------------- #


def _plain_task() -> core.TaskInstance:
    """A single-turn task with no tool driver: exactly one request goes out."""
    cached = getattr(_plain_task, "_task", None)
    if cached is None:
        cached = build_suite(RunConfig(only=("A",)))[0]
        _plain_task._task = cached  # type: ignore[attr-defined]
    return cached


def _live_runner(cfg: RunConfig, base_url: str) -> Runner:
    return Runner(cfg, OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10))


def test_the_wire_payload_carries_the_selected_generation_contract(endpoint):
    """H08: sampling, seed and the reasoning budget must reach the socket.

    They used to exist only in the config and in ``extra_body``; the pi adapter
    dropped temperature/top_p outright, so the declared conditions were not the
    ones measured. The transcript records the same contract, so a report can be
    checked against the bytes that went out.
    """
    from dailybench.rng import derived_seed

    base_url, server = endpoint
    task = _plain_task()
    cfg = RunConfig(
        base_url=base_url, model="test-model", temperature=0.23, top_p=0.71, context_window=32768
    )
    run = _live_runner(cfg, base_url).run_task(task)

    payload = server.last_payload
    assert payload["temperature"] == 0.23 and payload["top_p"] == 0.71
    assert payload["seed"] == derived_seed(cfg.seed, task.id, "0")
    assert payload["reasoning_budget"] == cfg.reasoning_budget
    assert payload["max_tokens"] == run.transcript.extra["wire"]["output_limit"]

    wire = run.transcript.extra["wire"]
    assert wire["mode"] == "fixed" and wire["verified"] == "payload"
    assert wire["temperature"] == 0.23 and wire["top_p"] == 0.71
    assert wire["reasoning_budget"] == cfg.reasoning_budget
    assert wire["stop"] == []
    assert "template_version" in wire and "output_limit" in wire


def test_native_mode_leaves_sampling_and_reasoning_to_the_server(endpoint):
    """H08: ``native`` and ``fixed`` are different conditions, and say so.

    Native means the server's own defaults decide sampling and thinking; the
    answer limit stays the harness's, because the window still has to fit. The
    two modes must never be reported as the same measurement.
    """
    base_url, server = endpoint
    cfg = RunConfig(
        base_url=base_url, model="test-model", temperature=0.23, top_p=0.71,
        generation_mode="native", context_window=32768,
    )
    run = _live_runner(cfg, base_url).run_task(_plain_task())
    payload = server.last_payload

    assert "temperature" not in payload and "top_p" not in payload
    assert "seed" not in payload and "reasoning_budget" not in payload
    assert payload["max_tokens"] > 0
    assert run.transcript.extra["wire"]["mode"] == "native"


def test_an_unfixed_generation_mode_is_rejected():
    with pytest.raises(ValueError, match="generation_mode"):
        RunConfig(generation_mode="server-ish")


def test_the_turns_record_the_wire_fields_that_went_out(endpoint):
    """The transcript keeps the per-turn contract, not only the run-level one."""
    base_url, _ = endpoint
    cfg = RunConfig(base_url=base_url, model="test-model", context_window=32768)
    run = _live_runner(cfg, base_url).run_task(_plain_task())
    wire = run.transcript.turns[0].reply.raw.get("wire")
    assert wire and wire["mode"] == "fixed"
    assert wire["output_limit"] == run.transcript.extra["answer_budgets"][0]["tokens"]


def test_a_payload_without_the_contract_fields_is_a_protocol_error():
    """The contract is checked on the final body, so a dropped field is loud."""
    from dailybench.providers import WireContract

    contract = WireContract(seed=7, reasoning_budget=2048, stop=("STOP",))
    payload: dict = {"model": "m", "messages": [], "stream": True}
    assert contract.missing(payload) == [
        "max_tokens", "temperature", "top_p", "reasoning_budget", "seed", "stop",
    ]
    payload.update(contract.payload_fields())
    assert contract.missing(payload) == []
    assert WireContract(mode="native", output_limit=64).missing(
        {"model": "m", "messages": [], "stream": True, "max_tokens": 64}
    ) == []


def test_a_stand_that_refuses_the_reasoning_budget_keeps_the_run():
    """A 400 on ``reasoning_budget`` drops the field once and says so."""
    base_url, server = httpstub.start(reject_reasoning_budget=True)
    try:
        cfg = RunConfig(base_url=base_url, model="test-model", context_window=32768)
        run = _live_runner(cfg, base_url).run_task(_plain_task())
    finally:
        httpstub.stop(server)

    assert run.transcript.error is None
    assert "reasoning_budget" not in httpstub.chat_payloads(server)[-1]
    assert run.transcript.extra["wire"]["reasoning_dropped"] is True
    assert run.transcript.extra["wire"]["reasoning_budget"] is None


# --------------------------------------------------------------------------- #
# H09: one limit policy, and an unverified answer is not ``ok``
# --------------------------------------------------------------------------- #


def test_a_pending_followup_is_not_cut_by_the_total_budget():
    """H09: a 17K completion spent the whole 16K total before the correction.

    The scenario had not been presented to the end, yet the row was ``ok`` and
    ``0/2``: the model was graded on an answer to the *first* condition.
    """
    from dailybench.tasks import dialogue

    task = dialogue.generate(random.Random(4), RunConfig())[0]
    first = core.ModelReply(text="итог: дешёвый", completion_tokens=17000)
    second = core.ModelReply(text="итог: дорогой", completion_tokens=200)
    provider = ScriptedProvider([first, second])
    cfg = RunConfig()
    cfg.max_total_tokens = 16384
    run = Runner(cfg, provider).run_task(task)

    assert len(provider.calls) == 2, "правка второго сообщения обязана быть предъявлена"
    assert provider.calls[1][-1]["role"] == "user"
    assert run.transcript.extra["total_budget_overrun"] is True
    assert "бюджет генерации" in " ".join(run.transcript.extra.get("notes", []))


def test_a_scenario_stopped_without_an_answer_is_not_ok():
    """H09: an exhausted budget that stops before the answer is unverified.

    The call budget runs out, the tool-less release request still comes back
    with calls, and no final answer exists: the row used to read ``ok``.
    """
    task = _tool_task("T-1")
    replies = [_call(index, "read_file", {"path": "README.md"}) for index in range(1, 6)]
    cfg = RunConfig()
    cfg.max_tool_calls = 1
    run = Runner(cfg, ScriptedProvider(replies)).run_task(task)

    assert run.score.status == "error"
    assert "до ответа" in (run.transcript.error or "")
    assert run.transcript.extra.get("total_budget_overrun") is None


def test_a_reply_without_usage_is_unconfirmed():
    """H09: an endpoint that reports no usage cannot prove the budget held.

    ``generated += reply.generated_tokens or 0`` used to count every usage-less
    reply as free, so the total budget was silently bypassed and the row stayed
    ``ok``.
    """
    task = _tool_task("T-1")
    provider = ScriptedProvider([_call(1, "read_file", {"path": "README.md"}), core.ModelReply(text="готово")])
    run = Runner(RunConfig(), provider).run_task(task)

    assert run.transcript.extra["usage_unconfirmed"] is True
    assert run.score.status == "error"
    assert any("расход" in note and "не подтвержд" in note for note in run.score.notes)


def test_a_reported_usage_confirms_the_budget():
    task = _tool_task("T-1")
    call = _call(1, "read_file", {"path": "README.md"})
    call.completion_tokens = 5
    reply = core.ModelReply(text="готово", completion_tokens=5)
    run = Runner(RunConfig(), ScriptedProvider([call, reply])).run_task(task)
    assert run.transcript.extra.get("usage_unconfirmed") is None


def test_the_total_budget_defaults_to_at_least_the_answer_budget():
    """J-5: the total was a fixed 16384 and clamped every larger answer.

    Nobody chose it, so it must follow the answer budget; an explicit value
    (reference policy, ``--max-total-tokens``) is kept as chosen.
    """
    assert RunConfig().max_total_tokens == 16384
    assert RunConfig(max_new_tokens=65536).max_total_tokens == 65536
    assert RunConfig(max_new_tokens=65536, max_total_tokens=20000).max_total_tokens == 20000


def test_a_length_cut_without_visible_text_is_not_ok():
    """J-5: all 16384 tokens went into thinking, the answer never started.

    ``finish_reason=length`` with an empty visible text was scored ``ok`` at
    ``0/5``: the row has to say the budget ran out before the answer.
    """
    task = _tool_task("T-1")
    reply = core.ModelReply(
        text="",
        reasoning="долго думал и не начал отвечать",
        completion_tokens=16384,
        reasoning_tokens=16384,
        finish_reason="length",
    )
    run = Runner(RunConfig(), ScriptedProvider([reply])).run_task(task)

    assert "бюджет исчерпан до ответа" in (run.transcript.error or "")
    assert run.transcript.extra["failure"] == "resource_exhausted"
    assert run.score.status == "error"
    assert core.outcome_of(run.score) == "resource_exhausted"


def test_reasoning_in_the_text_channel_counts_as_no_answer():
    task = _tool_task("T-1")
    reply = core.ModelReply(
        text="долго думал",
        reasoning="долго думал",
        completion_tokens=16384,
        reasoning_tokens=16384,
        finish_reason="length",
    )
    run = Runner(RunConfig(), ScriptedProvider([reply])).run_task(task)

    assert "бюджет исчерпан до ответа" in (run.transcript.error or "")
    assert run.score.status == "error"


def test_a_length_cut_before_tool_calls_stays_ok():
    """A truncated reply that still carries calls continues the loop."""
    task = _tool_task("T-1")
    cut = _call(1, "read_file", {"path": "README.md"})
    cut.finish_reason = "length"
    cut.completion_tokens = 100
    reply = core.ModelReply(text="готово", completion_tokens=5, finish_reason="stop")
    run = Runner(RunConfig(), ScriptedProvider([cut, reply])).run_task(task)

    assert run.transcript.error is None
    assert run.score.status == "ok"


def test_max_new_tokens_is_an_upper_bound_on_the_window_branch():
    """H09: ``--max-new-tokens 32`` reached the config but not the request.

    The window branch ignored it entirely and 259834 tokens went out.
    """
    cfg = RunConfig(context_window=32768, answer_cap=32)
    plan = cfg.plan_answer("A", 1000)
    assert plan.tokens == 32
    assert plan.capped_by == "max_new_tokens"


def test_the_cli_max_new_tokens_reaches_the_budget():
    from dailybench.cli import _config_from, build_parser

    cfg = _config_from(build_parser().parse_args(["run", "--max-new-tokens", "32"]))
    assert cfg.answer_cap == 32
    assert cfg.plan_answer("A", 1000).tokens == 32


# --------------------------------------------------------------------------- #
# H10: a budget is recorded with its source, and the floor is not silent
# --------------------------------------------------------------------------- #


def _coefficient_task(ratio: float) -> core.TaskInstance:
    task = _copy_task(_plain_task())
    task.meta["chars_per_token"] = ratio
    return task


def _copy_task(task: core.TaskInstance) -> core.TaskInstance:
    from dataclasses import replace

    return replace(task, meta=dict(task.meta))


def test_a_tokenizer_less_endpoint_uses_the_saved_task_coefficient():
    """H10: without /tokenize the budget fell to chars/2.5 for every document.

    The per-document ratio measured at run start was already stored per task;
    ignoring it turned a 31K prompt into a 37K estimate and the answer into the
    256-token floor.
    """
    task = _coefficient_task(4.0)
    cfg = RunConfig(context_window=1_000_000)
    run = Runner(cfg, ScriptedProvider([core.ModelReply(text="ok", completion_tokens=3)])).run_task(task)

    record = run.transcript.extra["answer_budgets"][0]
    assert record["source"] == "task-coefficient"
    assert "коэффициент" in record["reason"]
    sent = len(task.prompt) + len(cfg.system_prompt)
    assert record["tokens"] < int(sent / 2.5) or record["tokens"] > 0


def test_the_floor_is_flagged_when_no_room_is_left():
    """H10: the 256-token floor must not look like a measured budget."""
    task = _coefficient_task(4.0)
    cfg = RunConfig(context_window=2048)
    run = Runner(cfg, ScriptedProvider([core.ModelReply(text="ok", completion_tokens=3)])).run_task(task)

    record = run.transcript.extra["answer_budgets"][0]
    assert record["floor"] is True and record["tokens"] == 256
    assert "floor" in record["reason"]
    assert any("floor" in note for note in run.transcript.extra["notes"])


def test_the_sent_budget_and_its_reason_are_persisted():
    task = _tool_task("T-1")
    run = Runner(
        RunConfig(context_window=32768),
        ScriptedProvider([_call(1, "read_file", {"path": "README.md"}), core.ModelReply(text="готово")]),
    ).run_task(task)
    record = run.transcript.extra["answer_budgets"][0]
    assert record["tokens"] > 0 and record["reason"]
    assert record["policy"] in ("window", "fixed")


# --------------------------------------------------------------------------- #
# H11: the tool continuation keeps the assistant protocol fields
# --------------------------------------------------------------------------- #


def test_the_tool_continuation_returns_the_assistant_reasoning():
    """H11: DeepSeek thinking mode rejects a tool result without it (HTTP 400).

    The runner rebuilt a trimmed role/content/tool_calls triple, so the
    ``reasoning_content`` of the assistant message was lost between the tool
    call and the next request.
    """
    task = _tool_task("T-1")
    call = _call(1, "read_file", {"path": "README.md"})
    call.reasoning = "думаю про файл"
    run = Runner(RunConfig(), ScriptedProvider([call, core.ModelReply(text="готово")])).run_task(task)

    followup = run.transcript.turns[1].messages
    assistant = [m for m in followup if m.get("role") == "assistant"][-1]
    assert assistant["reasoning_content"] == "думаю про файл"
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"


def test_the_reasoning_travels_over_the_wire_httpstub():
    """H11 end to end: a stand that enforces the contract answers 400 without it."""
    base_url, server = httpstub.start(
        tool_call={"name": "read_file", "arguments": {"path": "README.md"}},
        require_reasoning_on_tool_continuation=True,
    )
    try:
        task = _tool_task("T-1")
        cfg = RunConfig(base_url=base_url, model="test-model", context_window=32768)
        run = _live_runner(cfg, base_url).run_task(task)
    finally:
        httpstub.stop(server)

    # The stub answers every request with a tool call, so the run ends on the
    # call budget without a final text: the protocol, not the answer, is the point.
    assert run.transcript.error is None or "до ответа" in run.transcript.error
    requests = httpstub.chat_payloads(server)
    assert len(requests) >= 2
    assistant = [m for m in requests[1]["messages"] if m.get("role") == "assistant"][-1]
    assert assistant["reasoning_content"] == "думаю "


# --------------------------------------------------------------------------- #
# Totals and reporting
# --------------------------------------------------------------------------- #


def _score(task_id: str, category: str, earned: float, possible: float, admission=None) -> dict:
    return {
        "task_id": task_id,
        "category": category,
        "earned": earned,
        "possible": possible,
        "max_points": possible,
        "status": "ok",
        "admission": admission,
        "error": None,
        "notes": [],
        "checks": [],
    }


def test_totals_handle_admission_and_partial_runs():
    def full_suite(e_admission: bool) -> list[dict]:
        return [
            _score("IF-1", "A", 15.0, 15.0),
            _score("R-1", "B", 20.0, 20.0),
            _score("C-1", "C", 25.0, 25.0),
            _score("T-1", "D", 15.0, 15.0),
            _score("LC-1", "E", 15.0, 15.0, admission=e_admission),
            _score("F", "F", 10.0, 10.0),
        ]

    complete = summarise(full_suite(True), generation_mode="fixed")
    assert complete["full_suite"] is True
    assert complete["verdict"] == "100/100"

    not_admitted = summarise(full_suite(False))
    assert not_admitted["long_context_admitted"] is False
    assert not_admitted["admitted_total"] == 85.0
    assert not_admitted["verdict"].startswith("85/85")
    assert "не прошёл допуск" in not_admitted["verdict"]
    assert not_admitted["per_category"]["E"]["earned"] == 0.0

    # Unknown is not a pass: the denominator stays full and the verdict says
    # the admission was not verified (R02).
    unknown = summarise(full_suite(None))
    assert unknown["admission_state"] == "unverified"
    assert unknown["long_context_admitted"] is False
    assert unknown["verdict"].startswith("100/100")
    assert "допуск не подтверждён" in unknown["verdict"]
    assert unknown["per_category"]["E"]["admission"] == "unverified"

    partial = summarise([_score("IF-1", "A", 3.0, 3.0), _score("R-1", "B", 2.0, 2.0)])
    assert partial["full_suite"] is False
    assert partial["verdict"].startswith("5/5")
    assert "частичный прогон" in partial["verdict"], "фильтрованный прогон не должен выглядеть как плохой результат"


def test_admission_unknown_marks_only_long_context_unverified():
    """Only E has an admission check; ``None`` on A–J must stay a pass.

    Non-E evaluators never write the field at all, so turning every ``None``
    into "unverified" made the per-category column and the comparison table
    show «допуск не подтверждён» for all ten scored categories of every run.
    """
    summary = summarise([
        _score("IF-1", "A", 10.0, 10.0),
        _score("R-1", "B", 15.0, 15.0),
        _score("LC-1", "E", 10.0, 10.0, admission=None),
    ])
    assert summary["per_category"]["A"]["admission"] == "passed"
    assert summary["per_category"]["B"]["admission"] == "passed"
    assert summary["per_category"]["E"]["admission"] == "unverified"
    assert summary["admission_state"] == "unverified"
    assert "допуск не подтверждён" in summary["verdict"]
    # The markdown admission column follows the same rule: "да" for A/B.
    assert "| A. Instruction following | 10 / 10 | да |" in render_markdown(
        {"dir": ".", "scores": [
            _score("IF-1", "A", 10.0, 10.0),
            _score("LC-1", "E", 10.0, 10.0, admission=None),
        ]}
    )


def test_selfcheck_proves_generators_and_evaluators_agree():
    from dailybench.selfcheck import selfcheck

    report = selfcheck(RunConfig())
    assert report.ok, report.render()
    scored = [row for row in report.rows if row["category"] in core.SCORED_CATEGORIES]
    showcase = [row for row in report.rows if row["category"] in core.SHOWCASE_CATEGORIES]
    assert len(scored) == 47
    assert showcase, "визуальный блок должен попадать в selfcheck, а не пропадать"
    assert core.FULL_TOTAL == 100.0
    assert sum(row["max"] for row in scored) == core.FULL_TOTAL
    assert all(row["max"] == row["oracle"] and row["garbage"] == 0 for row in showcase)


def test_run_persists_oracles_with_exact_values(tmp_path: Path):
    """R-4 keeps Decimal prices in its oracle; persisting must not crash."""
    from dailybench.selfcheck import OracleProvider
    from dailybench.runner import RunResult

    suite_seed = RunConfig(seed=20260915)
    task = next(t for t in build_suite(suite_seed) if t.id == "R-4")
    cfg = RunConfig(out_dir=tmp_path)
    run = Runner(cfg, OracleProvider(task)).run_task(task)
    result = RunResult(cfg=cfg, runs=[run], provider_info={"provider": "oracle"})
    out = result.save()
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    assert scores[0]["earned"] == task.max_points
    tasks_payload = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    assert tasks_payload[0]["task"]["meta"]["price_in"]
    assert (out / "transcripts" / "R-4.json").is_file()


def test_manifest_keys_are_posix(tmp_path: Path):
    from dailybench import generation

    probe = tmp_path / "link-source"
    probe.write_text("probe", encoding="utf-8")
    try:
        os.link(probe, tmp_path / "link-target")
    except OSError as exc:
        pytest.skip(f"файловая система не поддерживает hard links: {exc}")

    files = {
        "config.json": json.dumps({"identity": "identity"}),
        "transcripts/a.json": "{}",
    }
    generation.commit(tmp_path, files, run_id="run", identity="identity")
    first = generation.current(tmp_path)
    assert first is not None
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "transcripts/a.json" in manifest["files"]

    generation.commit(tmp_path, files, run_id="run", identity="identity")
    second = generation.current(tmp_path)
    assert second is not None and second != first
    assert (first / "transcripts/a.json").stat().st_ino == (
        second / "transcripts/a.json"
    ).stat().st_ino

def test_an_old_windows_manifest_key_stays_readable(tmp_path: Path):
    if os.name != "nt":
        pytest.skip("старый ключ с обратной косой существует только на Windows")
    from dailybench import generation

    payload = "{}"
    config = json.dumps({"identity": "identity"})
    files = {"config.json": config, "transcripts/a.json": payload}
    generation.commit(tmp_path, files, run_id="run", identity="identity")
    first = generation.current(tmp_path)
    assert first is not None
    before = (first / "transcripts/a.json").read_bytes()
    for folder in (tmp_path, first):
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["transcripts\\a.json"] = manifest["files"].pop("transcripts/a.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    generation.commit(tmp_path, files, run_id="run", identity="identity")
    assert (first / "transcripts/a.json").read_bytes() == before
    assert generation.current(tmp_path) is not None
    recorded = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert "transcripts/a.json" in recorded
    assert "transcripts\\a.json" not in recorded



def test_an_interrupted_save_leaves_one_committed_generation(tmp_path: Path, monkeypatch):
    """R25: a crash before tasks.json is replaced must not mix two checkpoints.

    The old layout wrote scores.json and tasks.json one after another; an
    ``OSError`` between them left a fresh score next to the previous answer.
    The generation manifest is the commit point, so a reader keeps seeing the
    previous complete generation until the new one is published.
    """
    import os

    from dailybench import generation
    from dailybench.core import Score
    from dailybench.report import load_run
    from dailybench.runner import RunResult, TaskRun, transcript_from_dict
    from dailybench.selfcheck import OracleProvider

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "R-4")
    cfg = RunConfig(out_dir=tmp_path)
    run = Runner(cfg, OracleProvider(task)).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "oracle"}).save()

    downgraded = TaskRun(
        task=task,
        transcript=transcript_from_dict(json.loads(json.dumps(run.as_dict()["transcript"])), task),
        score=Score.from_dict({**run.score.as_dict(), "earned": 0.0}),
    )
    real_replace = os.replace

    def replace_or_boom(source, target):
        if Path(target).name == "tasks.json":
            raise OSError("диск кончился")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace_or_boom)
    with pytest.raises(OSError, match="диск кончился"):
        RunResult(cfg=cfg, runs=[downgraded], provider_info={"provider": "oracle"}).save()
    monkeypatch.setattr(os, "replace", real_replace)

    data = load_run(tmp_path)
    scores = data["scores"][0]["earned"]
    tasks = data["tasks"][0]["score"]["earned"]
    mirror = generation.read_json(tmp_path, "transcripts/R-4.json")["score"]["earned"]
    assert scores == tasks == mirror == task.max_points, "все файлы обязаны описывать одну generation"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    committed = tmp_path / ".generations" / manifest["generation"]
    assert set(manifest["files"]) >= {"scores.json", "tasks.json", "transcripts/R-4.json"}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((committed / name).read_bytes()).hexdigest() == digest



def _two_generations(tmp_path: Path):
    """Save two generations and return (older, latest); the latest scores 0."""
    from dailybench import generation
    from dailybench.core import Score
    from dailybench.runner import RunResult, TaskRun, transcript_from_dict
    from dailybench.selfcheck import OracleProvider

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "R-4")
    cfg = RunConfig(out_dir=tmp_path)
    run = Runner(cfg, OracleProvider(task)).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "oracle"}).save()
    older = generation.current(tmp_path)
    assert older is not None
    downgraded = TaskRun(
        task=task,
        transcript=transcript_from_dict(json.loads(json.dumps(run.as_dict()["transcript"])), task),
        score=Score.from_dict({**run.score.as_dict(), "earned": 0.0}),
    )
    RunResult(cfg=cfg, runs=[downgraded], provider_info={"provider": "oracle"}).save()
    latest = generation.current(tmp_path)
    assert latest is not None and latest != older
    return task, older, latest


def test_current_refuses_a_damaged_generation(tmp_path: Path):
    """A byte swapped after the commit must not be served as the newest state.

    Refusal falls back to the root copies, which are the same bytes as long as
    nobody touched the generation: a deliberate root edit stays visible.
    """
    from dailybench import generation

    task, _older, latest = _two_generations(tmp_path)
    (latest / "tasks.json").unlink()
    (latest / "tasks.json").write_bytes(b'{"broken": ')

    assert generation.current(tmp_path) is None
    assert generation.committed(tmp_path) == tmp_path
    assert len(generation.read_json(tmp_path, "tasks.json")) == 1


def test_current_seals_the_generation_against_a_root_edit(tmp_path: Path):
    """The root export is a copy, so an edit there cannot rewrite the seal.

    The working readers follow the edited root (an operator changed state on
    purpose), but the sealed generation keeps its own bytes and stays current.
    """
    from dailybench import generation
    from dailybench.report import load_run

    _task, _older, latest = _two_generations(tmp_path)
    saved = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    saved[0]["transcript"]["error"] = "boom"
    (tmp_path / "tasks.json").write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    assert generation.current(tmp_path) == latest
    sealed = json.loads((latest / "tasks.json").read_text(encoding="utf-8"))
    assert sealed[0]["transcript"]["error"] != "boom"
    assert generation.read_json(tmp_path, "tasks.json")[0]["transcript"]["error"] == "boom"
    assert load_run(tmp_path)["integrity"] == "root-drift"


def test_current_rejects_a_generation_of_another_identity(tmp_path: Path):
    """Valid hashes are not enough: config.json must carry the manifest identity."""
    from dailybench import generation

    _task, _older, latest = _two_generations(tmp_path)
    manifest_path = latest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = latest / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["identity"] = "somebody-else"
    config_path.unlink()
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    manifest["files"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    assert generation.current(tmp_path) is None


def test_current_refuses_when_no_copy_is_consistent(tmp_path: Path):
    from dailybench import generation

    _task, _older, latest = _two_generations(tmp_path)
    for folder in (latest, tmp_path):
        for name in ("tasks.json", "scores.json", "config.json"):
            (folder / name).unlink()
            (folder / name).write_text("{}", encoding="utf-8")

    assert generation.current(tmp_path) is None
    assert generation.committed(tmp_path) == tmp_path


def test_load_run_reads_a_valid_manifest_generation(tmp_path: Path):
    from dailybench.report import load_run

    _task, _older, latest = _two_generations(tmp_path)
    sealed = json.loads((latest / "scores.json").read_text(encoding="utf-8"))[0]["earned"]
    data = load_run(tmp_path)
    assert data["integrity"] == "sealed"
    assert data["scores"][0]["earned"] == sealed
    assert generation_base(tmp_path, latest)


def generation_base(directory: Path, expected: Path) -> bool:
    from dailybench import generation

    return generation.current(directory) == expected


def test_load_run_quarantines_a_broken_manifest_generation(tmp_path: Path):
    """A manifest that no longer verifies must not fall back to the root copies.

    The root copies are exports of the same generation: falling back to them
    accepts whatever bytes are there, so a damaged seal becomes a silent new
    score instead of an integrity failure.
    """
    from dailybench import generation
    from dailybench.report import load_run

    _task, _older, latest = _two_generations(tmp_path)
    (latest / "scores.json").write_text('[{"earned": 9.0}]', encoding="utf-8")
    assert generation.current(tmp_path) is None

    with pytest.raises(generation.IntegrityError, match="карантин|integrity"):
        load_run(tmp_path)


def test_load_run_reports_a_root_copy_that_drifted_from_the_manifest(tmp_path: Path):
    """The root export is a copy: editing it must not rewrite the generation.

    A stale/edited root used to be served as the newest state; the reader must
    keep the sealed bytes and report the drift.
    """
    from dailybench.report import load_run

    _task, _older, latest = _two_generations(tmp_path)
    sealed_data = json.loads((latest / "scores.json").read_text(encoding="utf-8"))
    sealed = sealed_data[0]["earned"]
    root = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    root[0]["earned"] = sealed + 7.0
    (tmp_path / "scores.json").write_text(json.dumps(root), encoding="utf-8")

    assert json.loads((latest / "scores.json").read_text(encoding="utf-8"))[0]["earned"] == sealed, (
        "правка root не должна менять generation"
    )

    data = load_run(tmp_path)
    assert data["integrity"] == "root-drift"
    assert data["scores"][0]["earned"] == sealed


def test_a_single_damaged_generation_is_not_trusted(tmp_path: Path):
    from dailybench import generation
    from dailybench.runner import RunResult
    from dailybench.selfcheck import OracleProvider

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "R-4")
    cfg = RunConfig(out_dir=tmp_path)
    run = Runner(cfg, OracleProvider(task)).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "oracle"}).save()
    latest = generation.current(tmp_path)
    assert latest is not None
    (latest / "scores.json").unlink()
    (latest / "scores.json").write_text("[]", encoding="utf-8")

    assert generation.current(tmp_path) is None


def _windows_fsync(monkeypatch) -> list[int]:
    """Windows refuses ``os.fsync`` on a regular file opened ``rb``.

    Directories are opened read-only on every platform and are handled apart.
    """
    fcntl = pytest.importorskip("fcntl", reason="fcntl доступен только на POSIX")
    real = os.fsync
    calls: list[int] = []

    def fsync(fd: int) -> None:
        calls.append(fd)
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
        if regular and fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY:
            raise OSError(errno.EBADF, "Bad file descriptor")
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return calls


def _checkpoint(identity: str) -> dict[str, str]:
    return {"config.json": json.dumps({"identity": identity}), "scores.json": "[]"}


def test_the_first_checkpoint_survives_an_rb_handle_fsync(tmp_path: Path, monkeypatch):
    """H19: fsync on a read-mode descriptor fails, so nothing is published."""
    from dailybench import generation

    calls = _windows_fsync(monkeypatch)
    generation.commit(tmp_path, _checkpoint("run-1"), run_id="run-1", identity="run-1")

    assert calls, "checkpoint never reached fsync"
    assert generation.current(tmp_path) is not None, "first checkpoint is not published"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["durability"] == [], "every barrier of this platform held"


def test_no_temp_file_is_left_beside_a_published_file(tmp_path: Path, monkeypatch):
    """The staged write is renamed over the target, not left as ``.x.tmp``."""
    from dailybench import generation

    _windows_fsync(monkeypatch)
    generation.commit(tmp_path, _checkpoint("run-1"), run_id="run-1", identity="run-1")

    published = generation.current(tmp_path)
    assert published is not None
    assert sorted(p.name for p in published.iterdir() if p.name.startswith(".")) == []
    assert sorted(p.name for p in tmp_path.glob(".*")) == [generation.STORE]


def test_windows_skips_the_directory_fsync_and_records_it(tmp_path: Path, monkeypatch):
    """A directory handle cannot be opened on Windows: the marker says so."""
    from dailybench import generation

    monkeypatch.setattr(generation, "_dir_fsync_supported", lambda: False)
    real_open = os.open

    def windows_open(path, flags, *args, **kwargs):
        if Path(path).is_dir():
            raise PermissionError(errno.EACCES, "directory", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", windows_open)
    _windows_fsync(monkeypatch)

    generation.commit(tmp_path, _checkpoint("run-1"), run_id="run-1", identity="run-1")

    assert generation.current(tmp_path) is not None
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["durability"] == ["directory-fsync:unsupported:nt"]


def test_an_unsupported_file_fsync_degrades_with_a_marker(tmp_path: Path, monkeypatch):
    """EINVAL means the filesystem cannot flush; the commit is marked, not lost."""
    from dailybench import generation

    fcntl = pytest.importorskip("fcntl", reason="fcntl доступен только на POSIX")

    def fsync(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode) and not fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY:
            raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(os, "fsync", fsync)
    generation.commit(tmp_path, _checkpoint("run-1"), run_id="run-1", identity="run-1")

    assert generation.current(tmp_path) is not None
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["durability"] == ["file-fsync:unsupported:einval"]


def _visible_generations(directory: Path) -> list[str]:
    store = directory / ".generations"
    return sorted(p.name for p in store.iterdir() if p.is_dir() and not p.name.startswith("."))


def test_commits_keep_only_the_newest_generations(tmp_path: Path):
    """A full store copy per checkpoint filled the disk after three runs."""
    from dailybench import generation
    from dailybench.report import load_run

    for index in range(5):
        generation.commit(
            tmp_path,
            {**_checkpoint(f"run-{index}"), "tasks.json": json.dumps([{"index": index}])},
            run_id=f"run-{index}",
            identity=f"run-{index}",
            keep=2,
        )
        assert generation.current(tmp_path) is not None, f"поколение {index} не опубликовано"

    names = _visible_generations(tmp_path)
    assert len(names) == 2, names
    latest = generation.current(tmp_path)
    assert latest is not None and latest.name == names[-1]
    assert generation.resolve(tmp_path)[1] == "sealed"
    assert generation.read_json(tmp_path, "tasks.json") == [{"index": 4}]
    data = load_run(tmp_path)
    assert data["integrity"] == "sealed"
    assert data["tasks"] == [{"index": 4}]


def test_the_checkpoint_config_sets_the_retention(tmp_path: Path):
    """A run keeps the number it recorded, not the default of the caller."""
    from dailybench import generation

    checkpoint = {**_checkpoint("run-1"), "extra.json": "{}"}
    checkpoint["config.json"] = json.dumps({"identity": "run-1", "generation_keep": 1})
    for index in range(3):
        checkpoint["tasks.json"] = json.dumps([{"index": index}])
        generation.commit(tmp_path, dict(checkpoint), run_id="run-1", identity="run-1")

    assert len(_visible_generations(tmp_path)) == 1
    assert generation.current(tmp_path) is not None


def test_an_unchanged_generation_file_is_hardlinked_not_rewritten(tmp_path: Path):
    """config, suite and quiet transcripts are shared, only tasks.json is new."""
    from dailybench import generation

    generation.commit(
        tmp_path, {**_checkpoint("run-1"), "tasks.json": "[]"}, run_id="run-1", identity="run-1"
    )
    first = generation.current(tmp_path)
    generation.commit(
        tmp_path,
        {**_checkpoint("run-1"), "tasks.json": '[{"earned": 1}]'},
        run_id="run-1",
        identity="run-1",
    )
    second = generation.current(tmp_path)
    assert first is not None and second is not None and first != second

    shared = (first / "config.json").stat()
    assert shared.st_ino == (second / "config.json").stat().st_ino
    assert shared.st_nlink == 2
    assert (first / "tasks.json").stat().st_ino != (second / "tasks.json").stat().st_ino
    assert generation.read_json(tmp_path, "tasks.json") == [{"earned": 1}]


def test_a_generation_on_a_filesystem_without_hardlinks_falls_back_to_copy(tmp_path: Path, monkeypatch):
    from dailybench import generation

    def no_links(*_args, **_kwargs):
        raise OSError(errno.EPERM, "hardlinks unsupported")

    monkeypatch.setattr(generation.os, "link", no_links)
    generation.commit(
        tmp_path, {**_checkpoint("run-1"), "tasks.json": "[]"}, run_id="run-1", identity="run-1"
    )
    first = generation.current(tmp_path)
    generation.commit(
        tmp_path,
        {**_checkpoint("run-1"), "tasks.json": '[{"earned": 1}]'},
        run_id="run-1",
        identity="run-1",
    )
    second = generation.current(tmp_path)
    assert first is not None and second is not None

    assert (first / "config.json").stat().st_ino != (second / "config.json").stat().st_ino
    assert generation.read_json(tmp_path, "config.json") == {"identity": "run-1"}
    assert generation.read_json(tmp_path, "tasks.json") == [{"earned": 1}]


def test_a_failed_generation_removal_warns_and_keeps_the_run_readable(tmp_path: Path, monkeypatch):
    """Retention is housekeeping: a refused unlink must not fail the run."""
    from dailybench import generation
    from dailybench.report import load_run

    real_rmtree = shutil.rmtree

    def refuse_trash(path, *args, **kwargs):
        if ".trash-" in str(path):
            raise OSError(errno.EACCES, "отказано в доступе")
        return real_rmtree(path, *args, **kwargs)

    for index in range(2):
        generation.commit(
            tmp_path, _checkpoint(f"run-{index}"), run_id=f"run-{index}", identity=f"run-{index}", keep=2
        )
    monkeypatch.setattr(generation.shutil, "rmtree", refuse_trash)
    for index in range(2, 5):
        with pytest.warns(RuntimeWarning, match="удаление не завершено|не удалено"):
            generation.commit(
                tmp_path,
                _checkpoint(f"run-{index}"),
                run_id=f"run-{index}",
                identity=f"run-{index}",
                keep=2,
            )

    assert len(_visible_generations(tmp_path)) == 2
    assert len(list((tmp_path / ".generations").glob(".trash-*"))) == 3
    assert generation.read_json(tmp_path, "config.json") == {"identity": "run-4"}
    assert load_run(tmp_path)["integrity"] == "sealed"


def test_a_real_fsync_failure_still_fails_the_commit(tmp_path: Path, monkeypatch):
    """An I/O error is not an unsupported platform and must not be swallowed."""
    from dailybench import generation

    def fsync(fd: int) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError) as failure:
        generation.commit(tmp_path, _checkpoint("run-1"), run_id="run-1", identity="run-1")

    assert failure.value.errno == errno.EIO
    assert not (tmp_path / "manifest.json").exists()


def test_reference_entry_applies_policy_and_launch():
    from dailybench import reference

    entry = reference.resolve("qwen3.8-27b-gsq-rco-iq3xxs")
    cfg = reference.apply_to_run_config(RunConfig(), entry)

    assert cfg.quant == "IQ3_XXS"
    assert cfg.base_url == "http://127.0.0.1:8888/v1"
    # The entry's own window, identical to the baseline's: a weight comparison
    # must not change the available context.
    assert cfg.context_window == 32768 == cfg.n_ctx
    assert cfg.temperature == reference.POLICY["temperature"] == 0.0
    assert cfg.top_p == 1.0
    assert cfg.max_new_tokens == reference.POLICY["answer_budget_default"]
    # unbounded_budget: лимит ответа = окно минус резерв; таблица категорий —
    # запасной режим, в который можно вернуться переключателем.
    cfg.unbounded_budget = False
    assert cfg.budget_for("C") == reference.POLICY["answer_budget_artifact"]
    assert cfg.budget_for("A") == reference.POLICY["answer_budget_default"]
    cfg.unbounded_budget = True
    assert cfg.budget_for("C") == cfg.context_window - cfg.context_reserve
    assert cfg.max_tool_calls == reference.POLICY["max_tool_calls"]
    unsloth = cfg.runtime.startswith("unsloth")
    window = str(cfg.context_window)
    assert (f"--max-seq-length {window}" if unsloth else f"-c {window}") in cfg.launch, "в строке загрузки — окно записи"
    assert f"-ctk {cfg.kv_type}" in cfg.launch or "-ctk <kv>" in cfg.launch, "тип KV объявлен в строке загрузки"
    if not unsloth and "PrismML" not in cfg.runtime:
        assert "-ngl all" in cfg.launch
    assert cfg.unsupported == ()

    baseline = reference.apply_to_run_config(RunConfig(), reference.resolve("qwen3.8-27b-ud-q3-k-xl"))
    assert baseline.quant == "UD-Q3_K_XL"

    # Every shipped entry must resolve to a launch template, otherwise a report
    # would claim a configuration it cannot reproduce.
    for shipped in reference.model_entries():
        assert shipped.runtime_id, f"{shipped.id}: runtime не сопоставлен с шаблоном загрузки"
        window = str(shipped.entry.get("context_window") or shipped.policy.get("context_window"))
        assert window in shipped.launch_line(), f"{shipped.id}: в launch-строке нет окна {window}"
        assert "-ctk" in shipped.launch_line(), f"{shipped.id}: в launch-строке нет типа KV"

    with pytest.raises(KeyError):
        reference.resolve("no-such-config")


def test_unsupported_category_is_skipped_not_zeroed():
    """A category the configuration cannot do is not a zero the model earned."""
    cfg = RunConfig(unsupported=("D",))
    task = _tool_task("T-1")
    run = Runner(cfg, ScriptedProvider([])).run_task(task)

    assert run.score.skipped and "D" in run.score.skipped
    assert run.score.earned == 0.0
    assert run.score.possible == 0.0, "непроверенное не попадает в знаменатель"
    assert run.transcript.turns == [], "модель даже не вызывалась"

    scores = [run.score.as_dict(), _score("IF-1", "A", 3.0, 3.0)]
    summary = summarise(scores)
    assert summary["skipped"] == 1
    assert summary["possible_total"] == 3.0
    assert summary["per_category"]["D"]["skipped"] == 1
    assert "не проверялось" in summary["verdict"]

    markdown = render_markdown({"dir": ".", "config": {}, "scores": scores})
    assert "не проверялось" in markdown


def test_suite_sizing_is_measured_per_document():
    """Each long-context document is sized by its own measured density.

    One global ratio cannot fit five different record mixes: that left LC-2 and
    LC-5 at 80 % and 70 % of their targets, i.e. testing a smaller context than
    the scenario announces.
    """
    from dailybench.providers import CallableProvider
    from dailybench.suite import fit_suite_sizing

    # A hostile tokenizer: two characters per token, nothing like the 3.2 default.
    def fn(messages, tools):
        text = messages[-1]["content"]
        return core.ModelReply(text="ok", prompt_tokens=max(1, len(text) // 2))

    cfg = RunConfig()
    result = fit_suite_sizing(cfg, CallableProvider(fn=fn))

    assert result["calibrated"] is True
    assert set(cfg.task_ratios) == {"LC-1", "LC-2", "LC-3", "LC-4", "LC-5"}
    assert all(0 < ratio < 3.2 for ratio in cfg.task_ratios.values())
    assert result["documents"]["LC-5"]["actual"]

    rebuilt = build_suite(cfg)
    for task in rebuilt:
        target = task.meta.get("target_tokens")
        if not target:
            continue
        measured = len(task.prompt) / 2  # what the fake tokenizer reports
        assert 0.80 * target <= measured <= 1.10 * target, (task.id, measured, target)


def test_preflight_against_a_live_endpoint(endpoint, refusing_endpoint):
    from dailybench import reference

    base_url, _ = endpoint
    entry = reference.ReferenceConfig(
        entry={
            "id": "test-entry",
            "label": "Test entry",
            "runtime": "llama.cpp",
            "base_url": base_url,
            "model": "test-model",
            "quant": "test",
            "context_window": 32768,
            "kv_type": "f16",
            "capabilities": {"thinking": True, "tools": True},
        }
    )
    report = reference.preflight(entry)
    names = {probe.name: probe for probe in report.probes}
    assert names["model_listed"].status == "ok"
    assert names["tool_calling"].status == "ok", "endpoint эмулирует вызов инструмента"
    assert "search_code" in names["tool_calling"].detail or "read_file" in names["tool_calling"].detail
    assert names["context_window"].status == "warn", "без /props окно подтвердить нельзя"
    assert names["weights"].status == "warn"
    assert names["full_window"].status == "n/a"
    assert report.verdict == "admitted with unverified items"
    assert report.ok

    # A missing model must be a hard failure, not a warning.
    missing = reference.ReferenceConfig(
        entry={**entry.entry, "model": "ghost-model"}
    )
    rejected = reference.preflight(missing)
    assert rejected.verdict == "rejected" and not rejected.ok

    # An unreachable endpoint is rejected too, with the transport error shown.
    dead = reference.ReferenceConfig(
        entry={**entry.entry, "base_url": refusing_endpoint}
    )
    assert reference.preflight(dead).verdict == "rejected"


def test_speed_prompt_uses_a_fresh_nonce():
    from dailybench.measure import synthetic_prompt

    first = synthetic_prompt(256, nonce="short-run-0")
    second = synthetic_prompt(256, nonce="short-run-1")
    assert first != second
    assert first.startswith("[measurement short-run-0]")
    assert "READY" in first


def test_rescore_replays_saved_answers_without_the_model(tmp_path: Path):
    """An evaluator fix must be provable from disk, without re-running the model."""
    from dailybench.rescore import rescore_run
    from dailybench.selfcheck import OracleProvider
    from dailybench.runner import RunResult

    cfg = RunConfig(out_dir=tmp_path)
    task = next(t for t in build_suite(cfg) if t.id == "LC-5")
    run = Runner(cfg, OracleProvider(task)).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "oracle"}).save()
    # A run saved by the *old* evaluator: the record on disk carries its zero.
    saved = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    saved[0]["score"]["earned"] = 0.0
    (tmp_path / "tasks.json").write_text(json.dumps(saved), encoding="utf-8")

    marker = rescore_run(tmp_path)
    assert marker["tasks"] == 1
    assert [c["task_id"] for c in marker["changed"]] == ["LC-5"], "заниженный балл обязан попасть в changed"
    scores = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    assert scores[0]["earned"] == task.max_points
    # The showcase reads the per-task record, not ``scores.json``: a stale copy
    # there showed a passed task as "0/4" after a re-score.
    rewritten = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    assert rewritten[0]["score"]["earned"] == task.max_points
    assert (tmp_path / "rescored.json").is_file()


def test_rescore_keeps_a_skipped_task_skipped(tmp_path: Path):
    """Applicability is harness policy: a re-score must not turn a skip into 0/2 (R04)."""
    from dailybench.rescore import rescore_run
    from dailybench.runner import RunResult
    from dailybench.selfcheck import GarbageProvider

    cfg = RunConfig(out_dir=tmp_path, only=("IF-1",), unsupported=("A",))
    task = next(t for t in build_suite(cfg) if t.id == "IF-1")
    run = Runner(cfg, GarbageProvider()).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "garbage"}).save()

    marker = rescore_run(tmp_path)
    saved = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))[0]
    assert saved["status"] == "skipped" and saved["skipped"]
    assert saved["possible"] == 0.0 and saved["max_points"] == 2.0
    assert marker["changed"] == [], "одинаковое состояние не является изменением"


def test_rescore_records_a_changed_possible_even_without_earned_change(tmp_path: Path):
    """Every state field belongs to ``changed``, not only earned/admission."""
    from dailybench.rescore import rescore_run
    from dailybench.runner import RunResult
    from dailybench.selfcheck import OracleProvider

    cfg = RunConfig(out_dir=tmp_path, only=("IF-1",))
    task = next(t for t in build_suite(cfg) if t.id == "IF-1")
    run = Runner(cfg, OracleProvider(task)).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "oracle"}).save()
    saved = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    saved[0]["score"]["possible"] = 99.0
    saved[0]["score"].pop("status", None)
    (tmp_path / "tasks.json").write_text(json.dumps(saved), encoding="utf-8")

    marker = rescore_run(tmp_path)
    row = marker["changed"][0]
    assert row["before"]["possible"] == 99.0
    assert row["after"]["possible"] == task.max_points


def _saved_run(directory: Path, task: core.TaskInstance, **fingerprints: str) -> Path:
    """A minimal saved run plus its ``suite.json`` row, with optional drift."""
    from dailybench import suite as suite_module

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"model": "stub", "only": [task.id]}), encoding="utf-8"
    )
    (directory / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "task": {"id": task.id, "category": task.category},
                    "score": {
                        "task_id": task.id,
                        "category": task.category,
                        "max_points": float(task.max_points),
                        "checks": [{"name": "probe", "passed": False, "detail": "не проверено"}],
                    },
                    "transcript": {"turns": []},
                }
            ]
        ),
        encoding="utf-8",
    )
    digest = suite_module.suite_digest([task])
    row = {**digest["per_task"][0], **fingerprints}
    (directory / "suite.json").write_text(
        json.dumps({**digest, "per_task": [row]}, default=str), encoding="utf-8"
    )
    return directory


def test_rescore_keeps_the_old_score_when_the_assets_changed(tmp_path: Path):
    """P3: a regenerated picture was replayed against an answer that never saw it."""
    from dailybench.rescore import rescore_run

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "IF-1")
    _saved_run(tmp_path, task, assets_sha256="0" * 64)

    marker = rescore_run(tmp_path)
    statuses = [str(entry.get("status") or "") for entry in marker["changed"]]
    assert any("фикстура задания разошлась: assets_sha256" in status for status in statuses)
    kept = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))[0]["score"]
    assert [check["name"] for check in kept["checks"]] == ["probe"], "ответ не переигрывался"


def test_rescore_records_a_new_oracle_contract_and_replays(tmp_path: Path):
    """P3: a changed oracle is a new grading contract, not a new fixture.

    The answer was produced against exactly this task, so it must be re-graded —
    and the re-grade has to be named apart from a fixture drift.
    """
    from dailybench.rescore import rescore_run

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "IF-1")
    _saved_run(tmp_path, task, oracle_sha256="0" * 64, evaluator_sha256="1" * 64)

    marker = rescore_run(tmp_path)
    statuses = [str(entry.get("status") or "") for entry in marker["changed"]]
    assert any(
        "переоценено по новому контракту: oracle_sha256, evaluator_sha256" in status
        for status in statuses
    )
    assert not any("фикстура задания разошлась" in status for status in statuses)
    rescored = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))[0]["score"]
    assert "probe" not in [check["name"] for check in rescored["checks"]], "ответ переигран"


def test_rescore_names_a_task_of_an_unregistered_category(tmp_path: Path, monkeypatch):
    """P3: the old score is kept, so the marker must say the row was not re-graded."""
    from dailybench import rescore as rescore_module

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "IF-1")
    _saved_run(tmp_path, task)
    monkeypatch.setattr(rescore_module, "build_suite", lambda cfg: [task])
    monkeypatch.delitem(core.REGISTRY, task.category)

    marker = rescore_module.rescore_run(tmp_path)
    assert marker["changed"] == [{"task_id": "IF-1", "status": "категория не зарегистрирована"}]
    kept = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))[0]["score"]
    assert [check["name"] for check in kept["checks"]] == ["probe"]


def test_render_and_compare_markdown(tmp_path: Path):
    run = {
        "dir": str(tmp_path),
        "config": {"model": "qwen3.8-27b", "quant": "IQ4_XS", "seed": 20260915},
        "provider": {"provider": "openai-compatible"},
        "hardware": {"name": "RTX 5070 Ti", "vram_total_mib": 16303, "driver": "616.56"},
        "suite": {"sha256": "abc"},
        "scores": [
            _score("IF-1", "A", 3.0, 3.0),
            _score("R-1", "B", 0.0, 2.0),
            _score("C-3", "C", 0.0, 5.0),
        ],
        "tasks": [
            {"task": {"id": "R-1"}, "transcript": {"turns": [{"reply": {"finish_reason": "stop", "text": ""}}]}},
            {
                "task": {"id": "C-3"},
                "transcript": {
                    "turns": [{"reply": {"finish_reason": "length", "text": "", "reasoning_tokens": 4096}}]
                },
            },
        ],
        "speed": [
            {
                "label": "long_16384",
                "ttft_s": 2.5,
                "prompt_tok_s": 1200.0,
                "generation_tok_s": 18.0,
                "vram_peak_mib": 15400,
                "runs": 3,
            }
        ],
    }
    markdown = render_markdown(run)
    assert "DailyBench-16 — qwen3.8-27b (IQ4_XS)" in markdown
    assert "IF-1" in markdown and "R-1" in markdown
    assert "long_16384" in markdown and "15400" in markdown
    assert "RTX 5070 Ti" in markdown, "GPU name must come from the hardware record"
    assert "частичный прогон" in markdown, "фильтрованный прогон помечается в отчёте"
    assert "length — бюджет исчерпан до видимого ответа" in markdown

    other = json.loads(json.dumps(run))
    other["config"]["quant"] = "Q4_K_M"
    comparison = compare_markdown([run, other])
    assert "Сравнение конфигураций" in comparison
    # The name now appears in the header and in the weight-sensitivity rows
    # (published/equal/coding_heavy), so exact repetition is not the contract.
    assert "qwen3.8-27b" in comparison
    assert "| Модель |" in comparison
    assert "IQ4_XS" in comparison and "Q4_K_M" in comparison
    assert "не сравнение весов" not in comparison, "различается только квант"

    # A controlled parameter differs: the table stays, the claim does not.
    other["config"]["context_window"] = 65536
    differ = compare_markdown([run, other])
    assert "не сравнение весов" in differ
    assert "окно контекста" in differ and "65536" in differ


def test_search_code_resolves_paths_under_a_realpath_root():
    """A directory/file scoped search must find content on both ``/var`` and
    ``/private/var`` roots (macOS realpath).

    ``_resolve`` returns ``Path.resolve()`` while the file list used to come
    from the unresolved ``work_dir``: the prefix filter compared mismatched
    strings, so every scoped search returned ``no matches`` and T-6/T-2 lost
    their evidence under a realpath temp dir.
    """
    task = _tool_task("T-6")
    oracle = task.oracle
    scoped = _call(0, "search_code", {"query": oracle["value"], "path": oracle["config_path"]})
    replies = [scoped] + [
        _call(index, call["name"], call["arguments"])
        for index, call in enumerate(oracle["golden_calls"], start=1)
    ]
    replies.append(core.ModelReply(text=oracle["final_answer"]))
    run = Runner(RunConfig(), ScriptedProvider(replies)).run_task(task)

    first = run.transcript.tool_log[0]
    assert first["name"] == "search_code" and not first["error"], first["result"]
    assert oracle["value"] in first["result"]
    # The extra scoped call shifts the injected read_file failure by one, so the
    # retry check still sees its own error/success pair.
    assert run.score.earned == task.max_points


# --------------------------------------------------------------------------- #
# Executor provenance in the generation manifest and the report
# --------------------------------------------------------------------------- #

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64


def test_resume_flags_do_not_change_the_identity():
    from dataclasses import replace

    from dailybench.runner import run_identity

    cfg = RunConfig()
    assert (
        run_identity(cfg, "x")
        == run_identity(replace(cfg, retry_errors=True), "x")
        == run_identity(replace(cfg, allow_suite_drift=True), "x")
    )


def test_legacy_identity_is_unchanged_by_resume_flags():
    from dailybench.runner import run_identity

    assert run_identity(RunConfig(), "x") == "78b6c059052e84ec30b283a355e7acd75c9f419f4fa6144332fa31bf681742a4"


def test_the_image_digest_is_part_of_the_run_identity():
    """A rebuilt image measures another executor, so resume must see it."""
    from dailybench.runner import run_identity

    cfg = RunConfig(seed=20260915)
    cfg.image_digest = _DIGEST_A
    one = run_identity(cfg, "suite")
    cfg.image_digest = _DIGEST_B
    assert run_identity(cfg, "suite") != one


def test_a_swapped_image_digest_is_refused():
    from dailybench import generation

    with pytest.raises(generation.ProvenanceError, match="окружение песочницы сменилось"):
        generation.verify_environment(
            {"image_digest": _DIGEST_A},
            {"status": "verified", "image_digest": _DIGEST_B},
        )


def test_a_generation_whose_manifest_names_another_image_is_not_current(tmp_path: Path):
    """Valid hashes are not enough: the manifest environment must match config."""
    from dailybench import generation

    _task, _older, latest = _two_generations(tmp_path)
    manifest_path = latest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = latest / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Both halves still hash correctly, but they name different images.
    config["environment"] = {"status": "verified", "image_digest": _DIGEST_A}
    manifest["environment"] = {"status": "verified", "image_digest": _DIGEST_B}
    manifest["environment_sha256"] = generation.environment_digest(manifest["environment"])
    config_path.unlink()
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    manifest["files"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    assert generation.current(tmp_path) is None


def test_without_docker_the_environment_is_unverified(monkeypatch, tmp_path: Path):
    """No Docker is a limitation of the evidence, not a failed run."""
    from dailybench import generation

    monkeypatch.setattr(generation.shutil, "which", lambda _name: None)
    record = generation.environment()
    assert record["status"] == "unverified"
    assert record["image_digest"] == ""
    assert "docker" in record["reason"]
    # A recorded digest does not block a resume on a Docker-less host.
    generation.verify_environment({"image_digest": _DIGEST_A}, record)


def _docker_available(monkeypatch, runtimes: tuple[str, ...] = ("runc",)) -> None:
    """Make ``environment`` believe Docker answers, without a daemon."""
    from dailybench import generation

    monkeypatch.setattr(generation.shutil, "which", lambda _name: "/usr/bin/docker")

    def call(argv: list[str], *, timeout: float):
        if argv[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, f"repo@sha256:{'d' * 64}|sha256:{'c' * 64}\n", "")
        if argv[1:3] == ["info", "--format"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps({name: {} for name in runtimes}), "")
        return subprocess.CompletedProcess(argv, 1, "", "no dpkg")

    monkeypatch.setattr(generation, "_docker_call", call)


def test_an_image_without_a_running_container_is_a_host_run(monkeypatch):
    """Naming an image is not the same as executing in it."""
    from dailybench import generation

    _docker_available(monkeypatch)
    record = generation.environment()
    assert record["status"] == "verified"
    assert record["executor_status"] == "host_run"
    assert "не запускался" in record["executor_reason"]
    assert generation.environment(container_ran=True)["executor_status"] == "container_run"


def test_without_docker_the_executor_is_unverified(monkeypatch):
    from dailybench import generation

    monkeypatch.setattr(generation.shutil, "which", lambda _name: None)
    record = generation.environment()
    assert record["executor_status"] == "unverified"
    assert "docker" in record["executor_reason"]


def test_an_unregistered_runtime_keeps_the_environment_unverified(monkeypatch):
    from dailybench import generation

    _docker_available(monkeypatch, runtimes=("runc",))
    record = generation.environment(runtime="runsc")
    assert record["runtime_verified"] is False
    assert record["status"] == "unverified"
    assert "runsc" in record["reason"]


def test_a_malformed_source_date_epoch_is_refused(monkeypatch):
    from dailybench import generation

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "вчера")
    with pytest.raises(generation.ProvenanceError, match="SOURCE_DATE_EPOCH"):
        generation.environment()


def test_a_swapped_runtime_breaks_the_resume():
    """The runtime changes what executes the code, so it is part of the identity."""
    from dailybench import generation

    recorded = {
        "status": "verified",
        "image_digest": _DIGEST_A,
        "runtime": "runc",
        "source_date_epoch": 1700000000,
    }
    with pytest.raises(generation.ProvenanceError, match="runtime"):
        generation.verify_environment(recorded, {**recorded, "runtime": "runsc"})


def test_the_same_environment_resumes():
    from dailybench import generation

    record = {
        "status": "verified",
        "image_digest": _DIGEST_A,
        "runtime": "runc",
        "source_date_epoch": 1700000000,
    }
    generation.verify_environment(record, dict(record))


def test_an_unverified_side_never_refuses_a_resume():
    from dailybench import generation

    recorded = {"status": "verified", "image_digest": _DIGEST_A, "source_date_epoch": 1}
    current = {"status": "unverified", "image_digest": _DIGEST_B, "source_date_epoch": 2}
    generation.verify_environment(recorded, current)
    generation.verify_environment(current, recorded)


def test_the_report_prints_the_executor_environment():
    run = {
        "dir": ".",
        "config": {},
        "scores": [],
        "environment": {
            "status": "verified",
            "image": "dailybench/runner-python@sha256:…",
            "image_digest": _DIGEST_A,
            "runtime": "runsc",
            "runtime_verified": True,
            "executor_status": "host_run",
            "executor_reason": "образ указан, но контейнер не запускался",
            "source_date_epoch": 1700000000,
            "sbom": ["libc6\t2.36-9+deb12u4"],
        },
    }
    markdown = render_markdown(run)
    assert "## Среда" in markdown
    assert _DIGEST_A in markdown
    assert "runsc" in markdown
    assert "host_run: образ указан, но контейнер не запускался" in markdown
    assert "образ подтверждён, но не применялся" in markdown, "не «Среда подтверждена» рядом с host_run"
    assert "1700000000" in markdown
    assert "libc6\t2.36-9+deb12u4" in markdown

    unverified = {**run, "environment": {"status": "unverified", "reason": "docker CLI не найден"}}
    assert "не проверено: docker CLI не найден" in render_markdown(unverified)


def test_the_report_prints_observed_and_executed_separately_from_the_template():
    """R05: a filled template must not read as the command that ran."""
    run = {
        "dir": ".",
        "scores": [],
        "config": {
            "model": "m",
            "launch": "unsloth studio run --model <x> --max-seq-length 32768",
            "launch_executed": {
                "method": "studio-api POST /api/inference/load",
                "request": {"model_path": "m", "max_seq_length": 32768},
            },
            "observed": {
                "source": "runtime /props",
                "model_path": "/weights/m-Q3.gguf",
                "quant": "unknown",
                "n_ctx": 32768,
                "kv_type": "unknown",
                "server_gpu": "unknown",
                "client_gpu": "RTX (24564 MiB)",
                "weights_sha256": "unknown",
            },
        },
    }
    markdown = render_markdown(run)
    assert "## Наблюдённая конфигурация" in markdown
    assert "| GPU сервера | unknown |" in markdown
    assert "| Квант (со слов runtime) | unknown |" in markdown
    assert "Выполненная загрузка (executed)" in markdown
    assert '"max_seq_length": 32768' in markdown
    assert "Объявленный шаблон загрузки (declared" in markdown
    assert markdown.index("Выполненная загрузка") < markdown.index("Объявленный шаблон")
    assert "unknown" in markdown, "неизвестное печатается, а не подменяется объявленным"


def test_provenance_refuses_a_foreign_image_on_resume(tmp_path: Path):
    """The reader, not only ``verify_environment``, must stop a swapped image."""
    from dailybench import cli
    from dailybench.runner import RunResult
    from dailybench.selfcheck import OracleProvider

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "R-4")
    cfg = RunConfig(seed=20260915, out_dir=tmp_path, image_digest=_DIGEST_A)
    run = Runner(cfg, OracleProvider(task)).run_task(task)
    RunResult(
        cfg=cfg,
        runs=[run],
        provider_info={"provider": "oracle"},
        environment={"status": "verified", "image_digest": _DIGEST_A},
    ).save()

    cfg.image_digest = _DIGEST_B
    with pytest.raises(SystemExit, match="окружение песочницы сменилось"):
        cli._resume_tasks(
            cfg,
            build_suite(cfg),
            {"status": "verified", "image_digest": _DIGEST_B},
        )


# --------------------------------------------------------------------------- #
# bench is an admission gate, not a wrapper around a possibly stale server
# --------------------------------------------------------------------------- #


def _bench_reference(tmp_path: Path, base_url: str) -> Path:
    """A reference file pointing at the stub, with the shipped policy."""
    from dailybench import reference

    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "schema": "dailybench/reference/1",
                "policy": reference.POLICY,
                "models": [
                    {
                        "id": "stub-entry",
                        "label": "Stub entry",
                        "runtime_id": "llama.cpp",
                        "base_url": base_url,
                        "model": "test-model",
                        "quant": "IQ3_XXS",
                        "context_window": 32768,
                        "kv_type": "f16",
                        "capabilities": {"tools": True},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_bench_stops_when_the_launcher_fails(tmp_path: Path, endpoint, monkeypatch, capsys):
    from dailybench import cli, launch

    base_url, _ = endpoint
    path = _bench_reference(tmp_path, base_url)
    out = tmp_path / "run"
    monkeypatch.setattr(launch, "up", lambda entry: (False, "модель не загрузилась за отведённое время", ""))
    code = cli.main(
        ["bench", "--config", "stub-entry", "--configs-path", str(path), "--out-dir", str(out), "--no-html"]
    )

    assert code != 0
    assert "модель не загрузилась" in capsys.readouterr().err
    assert not list(out.glob("*/tasks.json")), "старый сервер не должен пойти в прогон под новым именем"


def test_bench_does_not_rank_an_unverified_configuration(tmp_path: Path, endpoint, monkeypatch, capsys):
    from dailybench import cli, launch

    base_url, _ = endpoint
    path = _bench_reference(tmp_path, base_url)
    out = tmp_path / "run"
    monkeypatch.setattr(launch, "up", lambda entry: (True, "загружена", "key"))
    code = cli.main(
        ["bench", "--config", "stub-entry", "--configs-path", str(path), "--out-dir", str(out), "--no-html"]
    )

    assert code != 0
    assert "не подтвердил" in capsys.readouterr().err
    assert not list(out.glob("*/tasks.json"))


def test_bench_hands_the_single_resolved_window_to_launch_and_run(tmp_path: Path, endpoint, monkeypatch):
    from dailybench import cli, launch

    base_url, _ = endpoint
    path = _bench_reference(tmp_path, base_url)
    out = tmp_path / "run"
    seen: dict = {}

    def fake_up(entry):
        seen.update(entry)
        return True, "загружена", "key"

    monkeypatch.setattr(launch, "up", fake_up)
    code = cli.main(
        [
            "bench", "--config", "stub-entry", "--configs-path", str(path), "--out-dir", str(out),
            "--only", "IF-1", "--no-html", "--allow-unverified",
        ]
    )

    assert code == 0
    assert seen["context_window"] == 32768, "launcher получает разрешённое окно, а не политику"
    saved = json.loads(next(out.glob("*/config.json")).read_text(encoding="utf-8"))
    assert saved["context_window"] == 32768, "runner получает то же окно"


def test_bench_records_the_executed_load_apart_from_the_declared_template(
    tmp_path: Path, endpoint, monkeypatch
):
    """R05: the run config must carry what the launcher sent, not only the template."""
    from dailybench import cli, launch

    base_url, _ = endpoint
    path = _bench_reference(tmp_path, base_url)
    out = tmp_path / "run"

    def fake_up(entry):
        entry["executed"] = {
            "method": "studio-api POST /api/inference/load",
            "request": {"model_path": "test-model", "max_seq_length": 32768},
        }
        return True, "загружена", "key"

    monkeypatch.setattr(launch, "up", fake_up)
    code = cli.main(
        [
            "bench", "--config", "stub-entry", "--configs-path", str(path), "--out-dir", str(out),
            "--only", "IF-1", "--no-html", "--allow-unverified",
        ]
    )

    assert code == 0
    saved = json.loads(next(out.glob("*/config.json")).read_text(encoding="utf-8"))
    assert saved["launch_executed"]["request"]["max_seq_length"] == 32768
    assert saved["launch"], "шаблон тоже сохраняется — для воспроизведения"
    assert "-c" in saved["launch"] or "studio run" in saved["launch"], "launch остаётся шаблоном"
    assert saved["launch_executed"]["request"] != saved["launch"]
    # The stub serves no /props: the observed lane must say so instead of
    # repeating the declared window.
    assert saved["observed"]["source"] == "unknown"
    assert saved["observed"]["n_ctx"] == "unknown"


def test_load_entry_rejects_a_server_still_serving_the_old_quant(monkeypatch):
    from dailybench import launch

    monkeypatch.setattr(launch, "studio_api", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        launch, "api", lambda base_url, path, timeout=10.0: {"data": [{"id": "m", "loaded": True}]}
    )
    monkeypatch.setattr(
        launch,
        "props",
        lambda base_url, timeout=10.0: {
            "default_generation_settings": {"n_ctx": 32768},
            "model_path": "/weights/Qwen3.8-27B-UD-Q3_K_XL.gguf",
        },
    )

    ok, note = launch.load_entry(
        "http://127.0.0.1:8888/v1", "k", {"model": "m", "quant": "IQ3_XXS", "context_window": 32768}
    )
    assert not ok
    assert "UD-Q3_K_XL" in note and "IQ3_XXS" in note


def test_load_entry_rejects_a_shaved_window(monkeypatch):
    from dailybench import launch

    monkeypatch.setattr(launch, "studio_api", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        launch, "api", lambda base_url, path, timeout=10.0: {"data": [{"id": "m", "loaded": True}]}
    )
    monkeypatch.setattr(
        launch,
        "props",
        lambda base_url, timeout=10.0: {
            "default_generation_settings": {"n_ctx": 8192},
            "model_path": "/weights/m-IQ3_XXS.gguf",
        },
    )

    ok, note = launch.load_entry(
        "http://127.0.0.1:8888/v1", "k", {"model": "m", "quant": "IQ3_XXS", "context_window": 32768}
    )
    assert not ok
    assert "8192" in note and "32768" in note


def test_load_entry_accepts_the_declared_configuration(monkeypatch):
    from dailybench import launch

    monkeypatch.setattr(launch, "studio_api", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        launch, "api", lambda base_url, path, timeout=10.0: {"data": [{"id": "m", "loaded": True}]}
    )
    monkeypatch.setattr(
        launch,
        "props",
        lambda base_url, timeout=10.0: {
            "default_generation_settings": {"n_ctx": 32768},
            "model_path": "/weights/m-IQ3_XXS.gguf",
        },
    )

    ok, note = launch.load_entry(
        "http://127.0.0.1:8888/v1", "k", {"model": "m", "quant": "IQ3_XXS", "context_window": 32768}
    )
    assert ok
    assert "n_ctx=32768" in note


def test_load_entry_records_the_request_it_actually_sends(monkeypatch):
    """The executed lane is the real body, not the launch template."""
    from dailybench import launch

    sent: dict[str, dict] = {}

    def fake_api(base_url, path, key, payload=None, timeout=900):
        sent[path] = payload or {}
        return {}

    monkeypatch.setattr(launch, "studio_api", fake_api)
    monkeypatch.setattr(
        launch, "api", lambda base_url, path, timeout=10.0: {"data": [{"id": "m", "loaded": True}]}
    )
    monkeypatch.setattr(
        launch,
        "props",
        lambda base_url, timeout=10.0: {
            "default_generation_settings": {"n_ctx": 32768},
            "model_path": "/weights/m-IQ3_XXS.gguf",
        },
    )
    entry = {"model": "m", "quant": "IQ3_XXS", "context_window": 32768, "kv_type": "f16"}

    ok, _ = launch.load_entry("http://127.0.0.1:8888/v1", "k", entry)

    assert ok
    assert sent["/api/inference/load"]["cache_type_kv"] == "f16"
    assert entry["executed"]["request"] == sent["/api/inference/load"]
    assert entry["executed"]["method"].startswith("studio-api")


# --------------------------------------------------------------------------- #
# H02: the identity covers the executed evaluator, not one dispatcher module
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _mutate_once(path: Path, suffix: bytes = b"\n# identity probe\n"):
    saved = path.read_bytes()
    path.write_bytes(saved + suffix)
    try:
        yield
    finally:
        path.write_bytes(saved)


def test_evaluator_identity_covers_every_category_module(monkeypatch):
    """A hidden evaluator file is part of the contract, not an implementation detail.

    C used to fingerprint only ``coding.py``: replacing every hidden test in
    ``coding_python.py`` left the identity untouched.
    """
    from dailybench import suite as suite_module

    import dailybench.tasks.coding_python as python_coder
    import dailybench.tasks.coding_go as go_coder

    python_file = Path(python_coder.__file__)
    go_file = Path(go_coder.__file__)
    before = suite_module.evaluator_sha("C")
    assert before == suite_module.evaluator_sha("C")

    with _mutate_once(python_file):
        assert suite_module.evaluator_sha("C") != before
    with _mutate_once(go_file):
        assert suite_module.evaluator_sha("C") != before
    assert suite_module.evaluator_sha("C") == before


def test_evaluator_identity_covers_the_showcase_bundles():
    from dailybench import suite as suite_module

    for category, name in (("K", "visual_particles"), ("L", "art")):
        module = __import__(f"dailybench.tasks.{name}", fromlist=["_"])
        before = suite_module.evaluator_sha(category)
        with _mutate_once(Path(module.__file__)):
            assert suite_module.evaluator_sha(category) != before, category
        assert suite_module.evaluator_sha(category) == before


def test_suite_identity_covers_hidden_tests_and_the_toolchain():
    """``task.oracle`` carries the hidden tests and must break the digest.

    The applied toolchain (python/go/sqlite/node/chromium) is part of the
    grading contract too: the same oracle on another runtime is another
    measurement.
    """
    import platform
    import sqlite3

    from dailybench import suite as suite_module

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "IF-2")
    before = suite_module.suite_digest([task])

    task.oracle = {**task.oracle, "hidden_identity_probe": "new assertion"}
    mutated = suite_module.suite_digest([task])

    assert mutated["sha256"] != before["sha256"]
    assert mutated["per_task"][0]["oracle_sha256"] != before["per_task"][0]["oracle_sha256"]
    assert before["toolchain"]["python"] == platform.python_version()
    assert before["toolchain"]["sqlite"] == sqlite3.sqlite_version
    assert set(before["toolchain"]) >= {"python", "sqlite", "go", "node", "chromium"}
    assert before["evaluator_bundle"]["A"]


def test_suite_identity_covers_the_system_text_and_the_meta():
    """P3: neither the scenario system instruction nor ``meta`` was in the digest.

    A changed instruction was silently kept out of both the fingerprint and the
    resume, and the runner never sent the text at all.
    """
    from dailybench import suite as suite_module

    task = next(t for t in build_suite(RunConfig(seed=20260915)) if t.id == "IF-2")
    before = suite_module.suite_digest([task])

    task.system = "Отвечай только по-русски."
    with_system = suite_module.suite_digest([task])
    assert with_system["sha256"] != before["sha256"]
    assert with_system["per_task"][0]["system_sha256"] != before["per_task"][0]["system_sha256"]

    task.meta = {**task.meta, "digest_probe": 7}
    with_meta = suite_module.suite_digest([task])
    assert with_meta["sha256"] != with_system["sha256"]
    assert with_meta["per_task"][0]["meta_sha256"] != with_system["per_task"][0]["meta_sha256"]


def test_the_scenario_system_text_is_sent_as_a_system_message():
    """P3: ``TaskInstance.system`` was declared but never reached the request."""
    task = _tool_task("T-1")
    task.system = "Сначала прочитай README."
    provider = ScriptedProvider([core.ModelReply(text="готово", completion_tokens=5)])
    run = Runner(RunConfig(), provider).run_task(task)

    sent = provider.calls[0]
    assert [m["role"] for m in sent].count("system") == 2, "конфиг и сценарий — два сообщения"
    assert sent[1]["content"] == task.system
    assert sent[-1]["role"] == "user"
    assert run.score.status == "ok"


def test_grading_identity_changes_when_a_hidden_test_changes(monkeypatch):
    """The identity a checkpoint stores must move with the hidden tests."""
    from dailybench.runner import run_identity

    import dailybench.tasks.coding_python as python_coder

    cfg = RunConfig(seed=20260915)
    digest = __import__("dailybench.suite", fromlist=["suite_digest"]).suite_digest(build_suite(cfg), cfg.profile)
    before = run_identity(cfg, digest["sha256"])
    with _mutate_once(Path(python_coder.__file__)):
        changed = __import__("dailybench.suite", fromlist=["suite_digest"]).suite_digest(
            build_suite(cfg), cfg.profile
        )
    assert changed["sha256"] != digest["sha256"]
    assert run_identity(cfg, changed["sha256"]) != before


# --------------------------------------------------------------------------- #
# The task boundary (H01): the executor is recorded, not assumed
# --------------------------------------------------------------------------- #


def _fake_agent_run(agent_module, calls: dict):
    """A stub agent run that writes the target file and remembers its sandbox."""

    def fake_run(prompt, work_dir, cfg, *, agent_dir, timeout=None, sandbox=None):
        work = Path(work_dir)
        calls["work_dir"] = work
        calls["sandbox"] = sandbox
        if sandbox is not None and sandbox.ready_state and sandbox.ready_state[0]:
            sandbox.agent_dir = agent_dir
            sandbox.run_command(["pi"], host_env={})
        (work / "solution.py").write_text("class AsyncTTLCache:  # stub\n", encoding="utf-8")
        return agent_module.AgentRun(ok=True, text="ok", duration_s=0.1, turns=1, answer_tokens=3)

    return fake_run


def _coding_task():
    import dailybench.tasks.coding_python as python_coder

    cfg = RunConfig(seed=20260915)
    return python_coder.generate(random.Random(3), cfg), cfg


def test_without_docker_the_sandbox_falls_back_to_a_marked_host_run(monkeypatch):
    """Fallback is allowed, but an image on disk never counts as a container."""
    from dailybench import agent, isolation

    monkeypatch.setattr(isolation, "ready", lambda **kwargs: (False, "docker CLI не найден"))
    calls: dict = {}
    monkeypatch.setattr(agent, "run", _fake_agent_run(agent, calls))
    task, cfg = _coding_task()

    run = Runner(cfg, ScriptedProvider([])).run_task(task)

    assert calls["sandbox"] is not None, "объект песочницы нужен, чтобы записать причину"
    record = run.transcript.extra["sandbox"]
    assert record["executor_status"] == "host_run"
    assert "docker" in record["executor_reason"]
    assert "mounts" not in record


def test_a_container_run_is_recorded_with_the_applied_limits(monkeypatch):
    """The record names what really ran: mounts, env names, cgroups."""
    from dailybench import agent, isolation

    monkeypatch.setenv("DAILYBENCH_API_KEY", "host-secret-123")
    monkeypatch.setattr(isolation, "ready", lambda **kwargs: (True, ""))

    def fake_command(self, argv, *, host_env, deadline=None, stdout_path=None,
                     stderr_path=None, network="endpoint"):
        report = {
            "status": "ok",
            "container_exit_code": 0,
            "mounts": [
                {"src": str(self.work_dir), "dst": "/workspace", "readonly": False},
                {"src": str(self.agent_dir), "dst": "/agent", "readonly": False},
            ],
            "env": sorted(self.env(host_env)),
            "network": "bridge",
            "cgroups": {"cpus": "2", "memory": "2g", "pids_limit": 256},
            "rootfs": "read_only",
            "capabilities": "dropped",
            "deadline_s": deadline,
            "timed_out": False,
        }
        self.runs.append(report)
        return report

    monkeypatch.setattr(isolation.Sandbox, "run_command", fake_command)
    calls: dict = {}
    monkeypatch.setattr(agent, "run", _fake_agent_run(agent, calls))
    environment: dict = {"status": "verified", "executor_status": "host_run", "executor_reason": "x"}
    task, cfg = _coding_task()

    run = Runner(cfg, ScriptedProvider([]), environment=environment).run_task(task)

    record = run.transcript.extra["sandbox"]
    assert record["executor_status"] == "container_run"
    assert record["executor_reason"] == ""
    assert [mount["dst"] for mount in record["mounts"]] == ["/workspace", "/agent"]
    assert "DAILYBENCH_API_KEY" not in record["env"]
    assert record["cgroups"]["pids_limit"] == 256
    assert record["rootfs"] == "read_only"
    assert environment["executor_status"] == "container_run"


def test_isolation_removes_the_working_copy_after_the_task(monkeypatch):
    """Nothing of the task survives: the next model must not find it."""
    from dailybench import agent, isolation

    monkeypatch.setattr(isolation, "ready", lambda **kwargs: (False, "docker CLI не найден"))
    calls: dict = {}
    monkeypatch.setattr(agent, "run", _fake_agent_run(agent, calls))
    task, cfg = _coding_task()

    run = Runner(cfg, ScriptedProvider([])).run_task(task)

    assert run.transcript.extra["work_dir_removed"] is True
    assert not calls["work_dir"].exists()
    assert not Path(run.transcript.extra["work_dir"]).exists()
    # The grader still had the artefact: cleanup happens after scoring, not before.
    assert run.transcript.tool_log is not None
    assert run.score.possible == task.max_points


def test_sandbox_false_asks_for_the_host_on_purpose(monkeypatch):
    from dailybench import agent

    calls: dict = {}
    monkeypatch.setattr(agent, "run", _fake_agent_run(agent, calls))
    monkeypatch.setattr(
        "dailybench.isolation.ready", lambda **kwargs: pytest.fail("песочница выключена")
    )
    task, cfg = _coding_task()
    cfg.sandbox = False

    run = Runner(cfg, ScriptedProvider([])).run_task(task)

    assert calls["sandbox"] is None
    record = run.transcript.extra["sandbox"]
    assert record["executor_status"] == "host_run"
    assert "sandbox выключен" in record["executor_reason"]


@pytest.mark.skipif(os.name == "nt", reason="fake pi требует POSIX-shell")
def test_a_broken_container_falls_back_to_the_host_and_is_marked(monkeypatch, tmp_path):
    """A container that will not start must not lose the task — but must show."""
    from dailybench import agent, isolation

    fake = tmp_path / "pi"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo 1.0.0; exit 0; fi\n'
        'echo host > fell-back.txt\n'
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":5,"output":2},"stopReason":"stop"}}\'\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(fake))
    monkeypatch.setattr(isolation, "ready", lambda **kwargs: (True, ""))

    def broken(self, argv, **kwargs):
        raise isolation.InfraError("docker run failed before the command started")

    monkeypatch.setattr(isolation.Sandbox, "run_command", broken)
    task, cfg = _coding_task()

    run = Runner(cfg, ScriptedProvider([])).run_task(task)

    record = run.transcript.extra["sandbox"]
    assert record["executor_status"] == "host_run"
    assert "контейнере" in record["executor_reason"]
    assert run.transcript.extra["agent"]["exit_ok"] is True
    assert run.transcript.workspace_after.get("fell-back.txt", "").strip() == "host"


# --------------------------------------------------------------------------- #
# A missing image is built once, and a failed build is not a silent host run
# --------------------------------------------------------------------------- #

_FAKE_DOCKER_AUTOBUILD = """#!/bin/sh
# ``docker`` stand-in for the auto-build path: every call is logged and the
# image only resolves after a successful ``build``.
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  info) printf '{"runc": {"path": "runc"}}\\n' ;;
  image)
    if [ -f "$FAKE_DOCKER_IMAGE" ]; then
      printf '|sha256:%064d\\n' 1
    else
      echo "no such image" >&2
      exit 1
    fi
    ;;
  build)
    if [ "${FAKE_BUILD_EXIT:-0}" != "0" ]; then
      echo "build blew up" >&2
      exit "$FAKE_BUILD_EXIT"
    fi
    : > "$FAKE_DOCKER_IMAGE"
    ;;
  *) exit 0 ;;
esac
"""


@pytest.fixture
def sandbox_build(monkeypatch, tmp_path):
    """Fake docker whose image appears only after a build; returns (module, log)."""
    if os.name == "nt":
        pytest.skip("fake docker требует POSIX-shell")
    from dailybench import isolation

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER_AUTOBUILD, encoding="utf-8")
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("DAILYBENCH_SANDBOX_IMAGE", "dailybench/runner-python:autobuild")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    monkeypatch.setenv("FAKE_DOCKER_IMAGE", str(tmp_path / "image.built"))
    isolation.forget_build_state()
    yield isolation, log
    isolation.forget_build_state()


def _builds(log: Path) -> int:
    return sum(1 for line in log.read_text().splitlines() if line.startswith("build "))


def test_a_missing_image_is_built_once_and_the_answer_reused(sandbox_build):
    """The old silent host fallback: no image means no container, without a word."""
    isolation, log = sandbox_build

    usable, reason, status = isolation.prepare()
    assert usable, reason
    assert status == ""
    assert _builds(log) == 1

    usable, reason, status = isolation.prepare()
    assert usable and status == ""
    assert _builds(log) == 1, "второй задачи не полагается вторая сборка"


def test_a_failed_build_keeps_the_executor_unverified(sandbox_build, monkeypatch):
    isolation, log = sandbox_build
    monkeypatch.setenv("FAKE_BUILD_EXIT", "1")

    usable, reason, status = isolation.prepare()
    assert not usable
    assert status == "executor_unverified"
    assert "сборка не удалась" in reason and "autobuild" in reason

    assert isolation.prepare()[2] == "executor_unverified"
    assert _builds(log) == 1, "провал сборки тоже не повторяется"


def test_the_unverified_status_survives_into_the_sandbox_record(tmp_path):
    from dailybench import isolation

    box = isolation.Sandbox(work_dir=tmp_path, image="img")
    box.fallback("образ не собран: boom", status="executor_unverified")
    record = box.record()
    assert record["executor_status"] == "executor_unverified"
    assert "не собран" in record["executor_reason"]
    assert "mounts" not in record


def test_a_failed_build_marks_the_task_record(monkeypatch):
    from dailybench import agent, isolation

    monkeypatch.setattr(
        isolation,
        "prepare",
        lambda **kwargs: (False, "образ не собран: boom", "executor_unverified"),
    )
    calls: dict = {}
    monkeypatch.setattr(agent, "run", _fake_agent_run(agent, calls))
    task, cfg = _coding_task()

    run = Runner(cfg, ScriptedProvider([])).run_task(task)

    record = run.transcript.extra["sandbox"]
    assert record["executor_status"] == "executor_unverified"
    assert "не собран" in record["executor_reason"]


def test_preflight_sandbox_is_a_command(sandbox_build, capsys):
    from dailybench import cli

    assert cli.main(["preflight", "sandbox"]) == 0
    assert "песочница готова" in capsys.readouterr().out



# --------------------------------------------------------------------------- #
# Image capability probe (category V)
# --------------------------------------------------------------------------- #


def _vqa_task():
    import random

    from dailybench.tasks import visual_input

    class _Cfg:
        profile = "research-v1"

    return visual_input.generate(random.Random(5), _Cfg())[0]


def test_the_image_probe_reaches_the_endpoint_as_a_content_array(endpoint):
    base_url, server = endpoint
    provider = OpenAICompatibleProvider(base_url=base_url, model="m")
    assert provider.probe_images() is True
    assert provider.probe_images() is True  # cached: one request only
    parts = server.last_payload["messages"][-1]["content"]  # type: ignore[attr-defined]
    assert [part["type"] for part in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


class _TextOnlyHandler(_Handler):
    """A 4xx on any content array: the classic text-only chat endpoint."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        content = (payload.get("messages") or [{}])[-1].get("content")
        if isinstance(content, list):
            self.send_error(400, "image input not supported")
            return
        body = json.dumps({"error": "replay"}).encode()
        self.server.last_payload = payload  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_a_text_only_endpoint_fails_the_probe_and_the_task_is_skipped():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TextOnlyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        provider = OpenAICompatibleProvider(base_url=base_url, model="m")
        assert provider.probe_images() is False
        run = Runner(RunConfig(profile="research-v1"), provider).run_task(_vqa_task())
        assert run.score.status == "skipped"
        assert run.score.possible == 0.0
        assert "capability probe" in (run.score.skipped or "")
        assert run.transcript.extra["vision_probe"]["supported"] is False
    finally:
        server.shutdown()
        server.server_close()


def _telemetry_reply() -> core.ModelReply:
    return core.ModelReply(
        text="ответ",
        prompt_tokens=40,
        completion_tokens=9,
        reasoning_tokens=4,
        ttft_s=0.01,
        total_s=0.02,
        finish_reason="stop",
    )


def test_the_runner_records_a_task_telemetry_journal(tmp_path: Path):
    """Every request, tool span and the task end land in the journal."""
    task = _plain_task()
    run = Runner(RunConfig(), ScriptedProvider([_telemetry_reply()])).run_task(task)

    events = run.telemetry.events()
    kinds = [row["event"] for row in events]
    assert kinds[0] == "attempt_start"
    assert "request_sent" in kinds
    assert "first_token" in kinds
    assert "usage" in kinds
    assert kinds[-1] == "task_end"
    assert kinds.count("evaluator_start") == 1
    assert run.transcript.telemetry["answer_tokens"] == 5
    assert run.transcript.telemetry["reasoning_tokens"] == 4

    path = run.telemetry.save(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["task_id"] == task.id for row in rows)


def test_the_saved_run_index_carries_the_telemetry_aggregate(tmp_path: Path):
    from dailybench.report import load_run
    from dailybench.runner import RunResult

    task = _plain_task()
    run = Runner(RunConfig(), ScriptedProvider([_telemetry_reply()])).run_task(task)
    cfg = RunConfig(out_dir=tmp_path / "run")
    result = RunResult(cfg, runs=[run], suite_digest={"sha256": "x", "per_task": []})
    out = result.save()

    index = json.loads((out / "telemetry" / "index.json").read_text(encoding="utf-8"))
    assert index[0]["task_id"] == task.id
    assert index[0]["answer_tokens"] == 5
    journal = (out / "telemetry" / f"{task.id}.jsonl").read_text(encoding="utf-8")
    assert json.loads(journal.splitlines()[-1])["event"] == "task_end"

    # The section is rendered from the committed directory, not from a dict
    # injected by the test: a reader only ever sees what ``load_run`` serves.
    loaded = load_run(out)
    assert loaded["telemetry"] == index
    markdown = render_markdown(loaded)
    assert "Телеметрия и корреляции" in markdown
    assert "телеметрия не собрана" not in markdown


def test_a_legacy_telemetry_file_is_still_read(tmp_path: Path):
    from dailybench.report import load_run
    from dailybench.runner import RunResult

    task = _plain_task()
    run = Runner(RunConfig(), ScriptedProvider([_telemetry_reply()])).run_task(task)
    cfg = RunConfig(out_dir=tmp_path / "run")
    out = RunResult(cfg, runs=[run], suite_digest={"sha256": "x", "per_task": []}).save()
    legacy = json.loads((out / "telemetry" / "index.json").read_text(encoding="utf-8"))
    shutil.rmtree(out / "telemetry")
    (out / "telemetry.json").write_text(json.dumps(legacy), encoding="utf-8")

    assert load_run(out)["telemetry"] == legacy


def test_spend_agrees_with_generated_tokens_on_a_separate_reasoning_stand():
    """``completion`` without reasoning yields one total, not two answers."""
    reply = core.ModelReply(
        text="ответ", prompt_tokens=40, completion_tokens=100,
        reasoning_tokens=150, total_s=2.0,
    )
    transcript = core.Transcript(task_id="T-1", category="A", turns=[core.Turn([], reply)])

    spend = transcript.spend()
    assert spend["answer_tokens"] == 100
    assert spend["reasoning_tokens"] == 150
    assert spend["generated_tokens"] == transcript.generated_tokens() == 250
    assert spend["token_accounting"] == core.SEPARATE
    assert spend["tokens_per_s"] == 125.0


def test_an_estimated_reasoning_share_does_not_shrink_the_answer():
    """Reasoning text without details is estimated, never subtracted twice."""
    reply = core.ModelReply(
        text="ответ", reasoning="думал" * 300, prompt_tokens=40,
        completion_tokens=100, total_s=2.0,
    )
    transcript = core.Transcript(task_id="T-1", category="A", turns=[core.Turn([], reply)])

    spend = transcript.spend()
    assert spend["answer_tokens"] == 100
    assert spend["reasoning_tokens"] == len(reply.reasoning) // 3
    assert spend["generated_tokens"] == 100
    assert spend["token_accounting"] == core.ESTIMATED
    assert spend["reasoning_estimated"] is True


def test_spend_records_the_prompt_total_and_the_cache_share():
    reply = core.ModelReply(
        text="ответ", prompt_tokens=1000, completion_tokens=10,
        reasoning_tokens=0, cached_tokens=600, total_s=1.0,
    )
    later = core.ModelReply(
        text="ещё", prompt_tokens=1200, completion_tokens=5,
        reasoning_tokens=0, cached_tokens=900, total_s=1.0,
    )
    transcript = core.Transcript(
        task_id="T-1", category="A",
        turns=[core.Turn([], reply), core.Turn([], later)],
    )

    spend = transcript.spend()
    assert spend["prompt_tokens_total"] == 2200
    assert spend["prompt_tokens_last"] == 1200
    assert spend["cached_tokens"] == 1500
    assert spend["cache_share"] == round(1500 / 2200, 4)


def test_an_unknown_usage_leaves_spend_unknown_not_zero():
    reply = core.ModelReply(text="ответ", total_s=1.0)
    transcript = core.Transcript(task_id="T-1", category="A", turns=[core.Turn([], reply)])

    spend = transcript.spend()
    assert spend["answer_tokens"] is None
    assert spend["reasoning_tokens"] is None
    assert spend["generated_tokens"] is None
    assert spend["tokens_per_s"] is None
    assert spend["token_accounting"] == core.UNKNOWN
    assert spend["prompt_tokens_total"] is None
