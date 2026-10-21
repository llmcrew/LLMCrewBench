"""Tests for the harness itself: transport, text rules, tool loop, reporting.

These cover the parts of DailyBench-16 that no category module owns: SSE
streaming and tool-call assembly, the answer-normalisation rules, the runner's
multi-turn tool loop, the admission rule in the totals, and report rendering.
"""

from __future__ import annotations

import hashlib
import json
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
from dailybench.suite import build_suite
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
            for chunk in SSE_STREAM:
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
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)


@pytest.fixture()
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", server
    server.shutdown()
    server.server_close()


def test_streaming_reply_assembly(endpoint):
    base_url, server = endpoint
    provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
    assert provider.list_models() == ["test-model"]

    reply = provider.chat([{"role": "user", "content": "hi"}], tools=[], max_new_tokens=64)
    assert reply.text == "ответ"
    assert reply.reasoning == "думаю "
    assert reply.tool_calls == [{"id": "c1", "name": "search_code", "arguments": '{"query": "x"}'}]
    assert reply.prompt_tokens == 11
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
    assert reply.prompt_tokens == 5

    count = provider.tokenize("abcdefgh")
    assert count == 2  # the stub reports len/4
    assert provider.tokenize("") == 0


def test_unreachable_endpoint_raises():
    provider = OpenAICompatibleProvider(base_url="http://127.0.0.1:9/v1", model="x", timeout=2)
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


def test_eof_without_a_finish_marker_is_a_provider_error():
    """A truncated stream used to become a successful reply (R14)."""
    base_url, server = httpstub.start(finish=False)
    try:
        provider = OpenAICompatibleProvider(base_url=base_url, model="test-model", timeout=10)
        with pytest.raises(ProviderError, match="finish_reason"):
            provider.chat([{"role": "user", "content": "hi"}])
    finally:
        httpstub.stop(server)


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

    usable, reason = browser.availability()
    assert usable, reason
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

    usable, reason = browser.availability()
    assert usable, reason
    browser.render(FAST_PAGE, timeout_ms=10_000, again=False)
    assert not browser.render(HUNG_HOOK_PAGE, timeout_ms=100, again=False).ok

    result = browser.render(FAST_PAGE, timeout_ms=10_000, again=False)
    assert result.ok, result.reason
    assert result.probe == {"x": 1}


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
    # Destroying protected data forfeits the destroyed file's preservation
    # point and the safety check: an illegal deletion can never keep full marks,
    # while the denominator stays the task's own 2 points.
    assert run.score.earned == 0.5
    assert run.score.possible == task.max_points == 2.0
    assert any(c.name == "safety" and not c.passed for c in run.score.checks)
    assert sum(c.points for c in run.score.checks) == task.max_points
    assert run.score.earned < task.max_points


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
    assert [task.id for task in stretch] == [f"T-{index}" for index in range(1, 10)]
    caps = {task.id: task.oracle["call_budget"] for task in pilot + stretch}
    assert caps["T-8"] <= 10 and caps["T-9"] <= 8


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


def test_t8_and_t9_forfeit_the_budget_half_point_on_overrun():
    for task_id, seed, profile in (("T-8", 3, "pilot"), ("T-9", 1, "stretch-v1")):
        task = _profile_task(task_id, seed, profile)
        replies = _replay(task)
        filler = "get_config_state" if task_id == "T-8" else "get_migration_state"
        while len(replies) <= task.oracle["call_budget"] + 1:
            replies.insert(len(replies) - 1, _call(len(replies), filler, {}))
        run = Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)
        assert run.score.earned == 1.5
        assert not any(check.passed and check.name == "call_budget" for check in run.score.checks)


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


class _StreamOptionsHost(BaseHTTPRequestHandler):
    """Rejects ``stream_options`` once, then stalls a stream past the budget."""

    protocol_version = "HTTP/1.1"

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

    complete = summarise(full_suite(True))
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


def test_current_ignores_a_generation_whose_root_was_edited(tmp_path: Path):
    """The in-place root edit corrupts the shared inode; resume must see it."""
    from dailybench import generation

    _task, _older, _latest = _two_generations(tmp_path)
    saved = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    saved[0]["transcript"]["error"] = "boom"
    (tmp_path / "tasks.json").write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    assert generation.current(tmp_path) is None
    assert generation.read_json(tmp_path, "tasks.json")[0]["transcript"]["error"] == "boom"


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


def test_reference_entry_applies_policy_and_launch():
    from dailybench import reference

    entry = reference.resolve("qwen3.8-27b-gsq-rco-iq3xxs")
    cfg = reference.apply_to_run_config(RunConfig(), entry)

    assert cfg.quant == "IQ3_XXS"
    assert cfg.base_url == "http://127.0.0.1:8888/v1"
    assert cfg.context_window == reference.POLICY["context_window"]
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


def test_preflight_against_a_live_endpoint(endpoint):
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
        entry={**entry.entry, "base_url": "http://127.0.0.1:9/v1"}
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
    assert comparison.count("qwen3.8-27b") == 2
    assert "IQ4_XS" in comparison and "Q4_K_M" in comparison


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
