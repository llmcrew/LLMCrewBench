"""Tests for the harness itself: transport, text rules, tool loop, reporting.

These cover the parts of DailyBench-16 that no category module owns: SSE
streaming and tool-call assembly, the answer-normalisation rules, the runner's
multi-turn tool loop, the admission rule in the totals, and report rendering.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

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
            content = payload.get("content", "")
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
    assert reply.generated_tokens == 10
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
    assert run.score.earned == 3.0
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
    assert run.score.earned < task.max_points


def test_tool_budget_is_enforced():
    task = _tool_task("T-1")
    replies = [_call(index, "read_file", {"path": "README.md"}) for index in range(30)]
    cfg = RunConfig()
    cfg.max_tool_calls = 3
    run = Runner(cfg, ScriptedProvider(replies)).run_task(task)
    assert len(run.transcript.tool_log) == 3
    assert "бюджет вызовов инструментов исчерпан (3)" in run.transcript.extra.get("notes", [])


# --------------------------------------------------------------------------- #
# Totals and reporting
# --------------------------------------------------------------------------- #


def _score(task_id: str, category: str, earned: float, possible: float, admission=None) -> dict:
    return {
        "task_id": task_id,
        "category": category,
        "earned": earned,
        "possible": possible,
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

    partial = summarise([_score("IF-1", "A", 3.0, 3.0), _score("R-1", "B", 2.0, 2.0)])
    assert partial["full_suite"] is False
    assert partial["verdict"].startswith("5/5")
    assert "частичный прогон" in partial["verdict"], "фильтрованный прогон не должен выглядеть как плохой результат"


def test_selfcheck_proves_generators_and_evaluators_agree():
    from dailybench.selfcheck import selfcheck

    report = selfcheck(RunConfig())
    assert report.ok, report.render()
    assert len(report.rows) == 29
    assert sum(row["max"] for row in report.rows) == core.FULL_TOTAL


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
    assert scores[0]["earned"] == 2.0
    tasks_payload = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    assert tasks_payload[0]["task"]["meta"]["price_in"]
    assert (out / "transcripts" / "R-4.json").is_file()


def test_reference_config_applies_the_policy():
    from dailybench import reference

    entry = reference.resolve("qwen3.8-27b-gsq-rco-iq3xxs")
    cfg = reference.apply_to_run_config(RunConfig(), entry)

    assert cfg.reference == "qwen3.8-27b-gsq-rco-iq3xxs"
    assert cfg.quant == "IQ3_XXS"
    assert cfg.base_url == "http://127.0.0.1:8888/v1"
    assert cfg.context_window == reference.POLICY["context_window"] == 32768
    assert cfg.temperature == reference.POLICY["temperature"] == 0.0
    assert cfg.top_p == 1.0
    assert cfg.max_new_tokens == reference.POLICY["answer_budget_default"]
    assert cfg.budget_for("C") == reference.POLICY["answer_budget_artifact"]
    assert cfg.budget_for("A") == reference.POLICY["answer_budget_default"]
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
        window = str(shipped.entry.get("context_window"))
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

    marker = rescore_run(tmp_path)
    assert marker["tasks"] == 1
    assert marker["changed"] == [], "повторный расчёт по тем же ответам должен совпасть"
    scores = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    assert scores[0]["earned"] == task.max_points
    assert (tmp_path / "rescored.json").is_file()


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
