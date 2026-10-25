"""Telemetry journal: valid JSONL, explicit unknowns, deterministic reduction."""

from __future__ import annotations

import json
import threading

import pytest

import httpstub

from dailybench import telemetry
from dailybench.providers import OpenAICompatibleProvider
from dailybench.report import load_run, render_markdown, summarise


class Clock:
    """Deterministic monotonic clock; every read advances it."""

    def __init__(self, step: float = 0.25) -> None:
        self.now = 100.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _journal(*, clock=None, **kwargs) -> telemetry.Telemetry:
    return telemetry.Telemetry(
        "T-1", category="A", window=1000, clock=clock or Clock(), **kwargs
    )


def test_events_are_valid_jsonl_with_monotonic_time():
    journal = _journal()
    journal.attempt_start(budget=8, reason="window", prompt_tokens=120, unbounded=False)
    journal.request_sent(budget=8, reason="window", prompt_tokens=120)
    journal.first_token(ttft_s=0.1, tokens=3)
    journal.decode_chunk(interval_s=0.05, tokens=3, chars=12)
    journal.tool_call_start(name="read_file")
    journal.tool_call_end(name="read_file", ok=True, duration_s=0.01)
    journal.tool_call_end(name="write", ok=False, duration_s=0.02, error=True)
    journal.evaluator_start()
    journal.evaluator_end(ok=True, duration_s=0.03)
    journal.browser_stage("hook", status="start")
    journal.browser_stage("render", status="ok", duration_s=0.4)
    journal.usage(prompt_tokens=120, completion_tokens=9, reasoning_tokens=4, exact=True)
    journal.task_end(outcome="ok")

    lines = journal.to_jsonl().splitlines()
    assert lines
    rows = [json.loads(line) for line in lines]
    assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["task_id"] == "T-1" for row in rows)
    assert all(row["event"] in telemetry.EVENTS for row in rows)
    times = [row["t"] for row in rows]
    assert times == sorted(times)
    assert rows[-1]["event"] == "task_end"

    restored = telemetry.Telemetry.from_jsonl(journal.to_jsonl(), task_id="T-1")
    restored_aggregate = restored.aggregate()
    assert restored_aggregate["turns"] == 1
    assert restored_aggregate["tools"] == 2
    assert restored_aggregate["tool_errors"] == 1
    assert restored_aggregate["browser_s"] == 0.4
    assert restored_aggregate["answer_tokens"] == 5


def test_unknown_usage_stays_unknown_not_zero():
    journal = _journal()
    journal.attempt_start(budget=10, reason="fixed")
    journal.request_sent(budget=10, reason="fixed")
    journal.first_token(ttft_s=0.2)
    journal.usage(exact=False)

    aggregate = journal.aggregate()
    assert aggregate["usage_known"] is False
    assert aggregate["answer_tokens"] is None
    assert aggregate["reasoning_tokens"] is None
    assert aggregate["cached_tokens"] is None
    assert aggregate["ttft_s"] == 0.2


def test_reasoning_is_not_counted_twice():
    journal = _journal()
    journal.attempt_start(budget=10, reason="fixed")
    journal.request_sent(budget=10, reason="fixed")
    journal.usage(
        prompt_tokens=50,
        completion_tokens=100,
        reasoning_tokens=30,
        cached_tokens=10,
        exact=True,
    )

    aggregate = journal.aggregate()
    # completion_tokens already includes reasoning: 100 total, 30 thinking,
    # 70 visible answer.
    assert aggregate["answer_tokens"] == 70
    assert aggregate["reasoning_tokens"] == 30
    assert aggregate["cached_tokens"] == 10


def test_decode_rate_uses_exact_tokens_when_usage_exists(tmp_path):
    journal = _journal(clock=Clock(step=1.0))
    journal.attempt_start(budget=10, reason="window")
    journal.request_sent(budget=10, reason="window")
    journal.first_token(ttft_s=1.0)
    journal.decode_chunk(interval_s=1.0, tokens=5)
    journal.decode_chunk(interval_s=1.0, tokens=5)
    journal.usage(completion_tokens=10, reasoning_tokens=0, exact=True)

    aggregate = journal.aggregate()
    assert aggregate["decode_tokens_per_s"] is not None
    assert aggregate["decode_tokens_estimated"] is False


def test_an_empty_journal_is_all_unknown():
    aggregate = _journal().aggregate()
    for key in (
        "turns",
        "tools",
        "ttft_s",
        "answer_tokens",
        "wall_s",
        "queue_s",
        "context_occupancy",
        "unbounded",
    ):
        assert aggregate[key] is None
    assert aggregate["usage_known"] is False


def _rows() -> list[dict]:
    return [
        {"task_id": str(index), "category": "A", "earned": index, "possible": 10,
         "share": index * 10.0, "spend_tokens": 100 + index * 10,
         "wall_s": 10 - index, "context_occupancy": index / 100}
        for index in range(1, 5)
    ]


def test_correlations_are_deterministic_under_reordering():
    rows = _rows()
    first = telemetry.correlations(rows)
    second = telemetry.correlations(list(reversed(rows)))
    assert first == second
    spend = [row for row in first if row["metric"] == "spend_tokens"][0]
    wall = [row for row in first if row["metric"] == "wall_s"][0]
    assert spend["rho"] == pytest.approx(1.0)
    assert wall["rho"] == pytest.approx(-1.0)
    assert spend["n"] == 4


def test_correlation_needs_three_pairs():
    rows = _rows()[:2]
    assert all(row["rho"] is None for row in telemetry.correlations(rows))


def test_unknown_shares_count_none():
    rows = _rows()
    rows[0]["spend_tokens"] = None
    shares = telemetry.unknown_shares(rows)
    assert shares["A"]["spend_tokens"] == {"unknown": 1, "total": 4, "share": 0.25}
    assert shares["A"]["wall_s"]["unknown"] == 0


def test_cost_per_point_only_on_a_complete_measurement():
    rows = _rows()
    assert telemetry.cost_per_point(rows, complete=True)["tokens"] == sum(
        row["spend_tokens"] for row in rows
    )
    assert telemetry.cost_per_point(rows, complete=False) is None
    rows[1]["spend_tokens"] = None
    assert telemetry.cost_per_point(rows, complete=True) is None


def test_summary_index_is_compact_and_stable():
    rows = _rows()
    index = telemetry.summary_index(rows)
    assert index[0]["n"] == 4
    assert index[0]["ttft_s"] is None
    assert index[0]["unknown_spend"] == 0
    assert index == telemetry.summary_index(list(reversed(rows)))


def _score(index: int, earned: float) -> dict:
    return {"task_id": f"T-{index}", "category": "A", "earned": earned,
            "possible": 10.0, "checks": []}


def _run(directory, telemetry_rows: list[dict] | None) -> dict:
    """A saved run on disk, read through ``load_run`` like a real report does."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "scores.json").write_text(
        json.dumps([_score(index, index) for index in range(1, 4)]), encoding="utf-8"
    )
    if telemetry_rows is not None:
        (directory / "telemetry").mkdir(exist_ok=True)
        (directory / "telemetry" / "index.json").write_text(
            json.dumps(telemetry_rows), encoding="utf-8"
        )
    return load_run(directory)


def test_report_does_not_crash_without_telemetry(tmp_path):
    markdown = render_markdown(_run(tmp_path, None))
    assert "Телеметрия не собрана" in markdown
    assert "Стоимость балла: не рассчитана" in markdown


def test_report_renders_correlations_and_cost_only_when_complete(tmp_path):
    rows = [
        {"task_id": f"T-{index}", "category": "A", "answer_tokens": 100 * index,
         "reasoning_tokens": 10, "wall_s": float(index), "context_occupancy": 0.1 * index,
         "usage_known": True}
        for index in range(1, 4)
    ]
    markdown = render_markdown(_run(tmp_path, rows))
    assert "Корреляция Спирмена" in markdown
    assert "проверенных" not in markdown
    assert "телеметрия не собрана" not in markdown
    # The synthetic run is complete (three of three scored tasks measured).
    summary = summarise(_run(tmp_path, rows)["scores"], {"per_task": [
        {"id": f"T-{index}", "category": "A"} for index in range(1, 4)
    ]})
    assert not summary["incomplete"]
    assert "Стоимость балла:" in markdown


def test_a_browser_render_records_its_stage(monkeypatch):
    seen: list[str] = []

    class Result:
        ok = True
        reason = ""
        failure = None
        duration_s = 0.0

    def fake_render(*args, **kwargs):
        journal = telemetry.active()
        journal.browser_stage("render", status="ok", duration_s=0.1)
        return Result()

    monkeypatch.setattr("dailybench.browser.render", fake_render)
    journal = _journal()
    from dailybench import browser

    with telemetry.bind(journal):
        browser.render("<html></html>")
    assert [row["stage"] for row in journal.events() if row["event"] == "browser_stage"] == [
        "render"
    ]


def test_usage_without_numbers_stays_unknown_not_partial():
    journal = _journal()
    journal.attempt_start(budget=5, reason="fixed")
    journal.request_sent(budget=5, reason="fixed")
    journal.usage(completion_tokens=100, reasoning_tokens=10, exact=False)

    aggregate = journal.aggregate()
    assert aggregate["usage_known"] is False
    # The numbers arrived but the runtime did not vouch for them: a derived
    # total over an inexact row would read as measured.
    assert aggregate["answer_tokens"] is None
    assert aggregate["reasoning_tokens"] is None
    assert aggregate["cached_tokens"] is None


def test_a_half_measured_task_does_not_sum_the_known_half():
    journal = _journal()
    for _ in range(2):
        journal.attempt_start(budget=4, reason="fixed")
        journal.request_sent(budget=4, reason="fixed")
    journal.usage(
        prompt_tokens=10, completion_tokens=7, reasoning_tokens=2, cached_tokens=1, exact=True
    )

    aggregate = journal.aggregate()
    assert aggregate["turns"] == 2
    # One of two requests reported; the missing one must not read as zero.
    assert aggregate["answer_tokens"] is None
    assert aggregate["reasoning_tokens"] is None
    assert aggregate["cached_tokens"] is None


def test_answer_clamps_at_zero_when_reasoning_exceeds_completion():
    journal = _journal()
    journal.attempt_start(budget=4, reason="fixed")
    journal.request_sent(budget=4, reason="fixed")
    journal.usage(completion_tokens=10, reasoning_tokens=40, cached_tokens=0, exact=True)

    aggregate = journal.aggregate()
    assert aggregate["answer_tokens"] == 0
    assert aggregate["reasoning_tokens"] == 40
    # Cache hits travel separately and stay a real zero here.
    assert aggregate["cached_tokens"] == 0


def test_queue_is_unknown_without_a_send_or_when_replayed():
    never_sent = _journal()
    never_sent.attempt_start(budget=5, reason="fixed")
    assert never_sent.aggregate()["queue_s"] is None

    replayed = _journal()
    replayed.attempt_start(budget=5, reason="agent", queue_known=False)
    replayed.request_sent(budget=5, reason="agent")
    assert replayed.aggregate()["queue_s"] is None


def test_overhead_is_unknown_when_a_component_is_unknown():
    journal = _journal()
    journal.attempt_start(budget=5, reason="fixed")
    journal.request_sent(budget=5, reason="fixed")
    journal.first_token(ttft_s=0.1)
    journal.usage(completion_tokens=5, reasoning_tokens=0, exact=True)

    # No evaluator span was recorded, so the unexplained remainder is unknown
    # rather than the whole wall time.
    assert journal.aggregate()["overhead_s"] is None


def test_overhead_is_the_remainder_and_never_negative():
    measured = _journal()
    measured.attempt_start(budget=5, reason="fixed")
    measured.request_sent(budget=5, reason="fixed")
    measured.first_token(ttft_s=0.1)
    measured.usage(completion_tokens=5, reasoning_tokens=0, exact=True)
    measured.tool_call_end(name="read_file", ok=True, duration_s=0.0)
    measured.evaluator_end(ok=True, duration_s=0.0)
    measured.task_end(outcome="ok")
    aggregate = measured.aggregate()
    assert aggregate["overhead_s"] is not None
    assert 0.0 < aggregate["overhead_s"] <= aggregate["wall_s"]

    overrun = _journal()
    overrun.attempt_start(budget=5, reason="fixed")
    overrun.request_sent(budget=5, reason="fixed")
    overrun.first_token(ttft_s=0.1)
    overrun.usage(completion_tokens=5, reasoning_tokens=0, exact=True)
    overrun.evaluator_end(ok=True, duration_s=99.0)
    overrun.task_end(outcome="ok")
    assert overrun.aggregate()["overhead_s"] == 0.0


def test_ttft_is_read_from_the_first_request_only():
    journal = _journal()
    journal.attempt_start(budget=4, reason="fixed")
    journal.request_sent(budget=4, reason="fixed")
    journal.first_token(ttft_s=0.1)
    journal.attempt_start(budget=4, reason="fixed")
    journal.request_sent(budget=4, reason="fixed")
    journal.first_token(ttft_s=0.9)

    assert journal.aggregate()["ttft_s"] == 0.1

    later = _journal()
    later.attempt_start(budget=4, reason="fixed")
    later.request_sent(budget=4, reason="fixed")
    later.attempt_start(budget=4, reason="fixed")
    later.request_sent(budget=4, reason="fixed")
    later.first_token(ttft_s=0.4)
    # The first request never produced a token; the next one is the real first.
    assert later.aggregate()["ttft_s"] == 0.4


def test_ttft_falls_back_to_the_recorded_send_before_the_token():
    journal = _journal()
    journal.attempt_start(budget=4, reason="fixed")
    journal.request_sent(budget=4, reason="fixed")
    journal.first_token(ttft_s=None)

    aggregate = journal.aggregate()
    # The send stamp to the token stamp spans two clock reads: 0.5 s.
    assert aggregate["ttft_s"] == pytest.approx(0.5)


def test_budget_mode_is_three_valued():
    empty = _journal().aggregate()
    assert empty["unbounded"] is None
    assert empty["budgeted"] is None

    mixed = _journal()
    for flag in (True, False, None):
        mixed.attempt_start(budget=1, reason="fixed", unbounded=flag)
    aggregate = mixed.aggregate()
    # "Unbounded" only when every attempt was; a partly unknown mix still
    # resolves, because the known modes are all that the row can claim.
    assert aggregate["unbounded"] is False
    assert aggregate["budgeted"] is True

    unbounded = _journal()
    unbounded.attempt_start(budget=1, reason="fixed", unbounded=True)
    assert unbounded.aggregate()["unbounded"] is True
    assert unbounded.aggregate()["budgeted"] is False

    budgeted = _journal()
    budgeted.attempt_start(budget=1, reason="fixed", unbounded=False)
    assert budgeted.aggregate()["budgeted"] is True


def test_clean_drops_non_finite_floats_and_rounds_the_rest():
    assert telemetry._clean(float("nan")) is None
    assert telemetry._clean(float("inf")) is None
    assert telemetry._clean(float("-inf")) is None
    assert telemetry._clean(1.23456789) == 1.234568
    assert telemetry._clean({"a": [float("inf"), 1.5]}) == {"a": [None, 1.5]}


def test_jsonl_never_writes_bare_nan():
    journal = _journal()
    journal.browser_stage("render", status="ok", duration_s=float("nan"))

    text = journal.to_jsonl()
    assert "NaN" not in text
    assert json.loads(text.strip())["duration_s"] is None


def test_seq_is_unique_and_ordered_across_threads():
    journal = _journal()

    def write() -> None:
        for _ in range(50):
            journal.record("tool_call_start", name="read_file")

    threads = [threading.Thread(target=write) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    seqs = [row["seq"] for row in journal.events()]
    assert sorted(seqs) == list(range(1, 101))


def test_an_unknown_event_name_is_rejected():
    with pytest.raises(ValueError):
        _journal().record("nope")


def test_record_after_task_end_raises():
    journal = _journal()
    journal.task_end(outcome="ok")
    with pytest.raises(RuntimeError):
        journal.record("first_token", ttft_s=0.1)


def test_task_end_is_idempotent():
    journal = _journal()
    journal.task_end(outcome="ok")
    journal.task_end(outcome="error", error="later")

    ends = [row for row in journal.events() if row["event"] == "task_end"]
    assert len(ends) == 1
    aggregate = journal.aggregate()
    assert aggregate["outcome"] == "ok"
    assert aggregate["error"] is None


def test_bind_restores_the_outer_journal():
    outer = _journal()
    inner = _journal()
    assert telemetry.active() is None
    with telemetry.bind(outer):
        assert telemetry.active() is outer
        with telemetry.bind(inner):
            assert telemetry.active() is inner
        assert telemetry.active() is outer
    assert telemetry.active() is None


def test_a_fresh_thread_starts_without_a_journal():
    journal = _journal()
    seen: list[object] = []
    with telemetry.bind(journal):
        thread = threading.Thread(target=lambda: seen.append(telemetry.active()))
        thread.start()
        thread.join()
    assert seen == [None]


def test_save_keeps_path_like_ids_inside_the_telemetry_directory(tmp_path):
    for task_id in ("../outside", "a/b", "..", "A-1"):
        path = telemetry.Telemetry(task_id).save(tmp_path)
        assert path.parent == tmp_path / "telemetry"
        assert path.exists()
    assert sorted(file.name for file in (tmp_path / "telemetry").iterdir()) == [
        "...jsonl",
        ".._outside.jsonl",
        "A-1.jsonl",
        "a_b.jsonl",
    ]


def test_the_provider_records_first_token_chunks_and_usage():
    base_url, server = httpstub.start(reply_text="четыре", completion_tokens=7, reasoning_tokens=3)
    provider = OpenAICompatibleProvider(base_url=base_url, model="stub")
    journal = _journal()
    journal.attempt_start(budget=16, reason="test")
    journal.request_sent(budget=16, reason="test")
    try:
        reply = provider.chat([{"role": "user", "content": "привет"}], telemetry=journal)
    finally:
        httpstub.stop(server)

    events = [row["event"] for row in journal.events()]
    assert "first_token" in events
    assert "decode_chunk" in events
    assert "usage" in events
    aggregate = journal.aggregate()
    assert aggregate["answer_tokens"] == reply.completion_tokens - reply.reasoning_tokens
    assert aggregate["reasoning_tokens"] == 3
    assert aggregate["decode_tokens_per_s"] is not None


def test_streamed_fragments_reach_the_journal_with_their_size():
    base_url, server = httpstub.start(reply_text="четыре", completion_tokens=7, reasoning_tokens=3)
    provider = OpenAICompatibleProvider(base_url=base_url, model="stub")
    journal = _journal()
    journal.attempt_start(budget=16, reason="test")
    journal.request_sent(budget=16, reason="test")
    try:
        provider.chat([{"role": "user", "content": "привет"}], telemetry=journal)
    finally:
        httpstub.stop(server)

    rows = journal.events()
    firsts = [row for row in rows if row["event"] == "first_token"]
    # TTFT is stamped once per request, on the very first fragment.
    assert len(firsts) == 1
    assert firsts[0]["ttft_s"] is not None
    chunks = [row for row in rows if row["event"] == "decode_chunk"]
    assert chunks
    assert all(row["chars"] > 0 and row["tokens"] > 0 for row in chunks)
    assert "NaN" not in journal.to_jsonl()


def test_an_agent_stream_replays_exact_ttft_and_unknown_queue():
    from dailybench.agent import AgentRun

    journal = _journal()
    run = AgentRun(
        ok=True,
        events=[
            {"type": "message_start"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {"input": 100, "output": 10, "reasoning": 2},
                    "ttftMs": 250,
                },
            },
            {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read_file"},
            {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "read_file"},
        ],
    )
    run.replay_telemetry(journal)

    attempts = [row for row in journal.events() if row["event"] == "attempt_start"]
    assert [row["queue_known"] for row in attempts] == [False]
    aggregate = journal.aggregate()
    assert aggregate["queue_s"] is None
    assert aggregate["ttft_s"] == pytest.approx(0.25)
    assert aggregate["answer_tokens"] == 8
    assert aggregate["reasoning_tokens"] == 2
    assert aggregate["usage_known"] is True
    assert aggregate["tools"] == 1
    assert aggregate["tool_errors"] == 0


def test_prompt_total_and_cache_share_are_aggregated():
    journal = _journal()
    for prompt, cached in ((100, 40), (300, 250)):
        journal.attempt_start(budget=10, reason="fixed")
        journal.request_sent(budget=10, reason="fixed", prompt_tokens=prompt)
        journal.usage(prompt_tokens=prompt, completion_tokens=5, reasoning_tokens=0,
                      cached_tokens=cached, exact=True)

    aggregate = journal.aggregate()
    assert aggregate["prompt_tokens_total"] == 400
    assert aggregate["prompt_tokens_last"] == 300
    assert aggregate["cached_tokens"] == 290
    assert aggregate["cache_share"] == 0.725

    rows = telemetry.rows([aggregate], [{"task_id": "T-1", "category": "A",
                                         "earned": 1, "possible": 1}])
    assert rows[0]["cache_share"] == 0.725
    assert rows[0]["prompt_tokens_total"] == 400
    assert telemetry.summary_index(rows)[0]["cache_share"] == 0.725


def test_a_half_reported_usage_leaves_the_prompt_total_unknown():
    journal = _journal()
    journal.attempt_start(budget=10, reason="fixed")
    journal.request_sent(budget=10, reason="fixed", prompt_tokens=100)
    journal.usage(prompt_tokens=100, completion_tokens=5, reasoning_tokens=0,
                  cached_tokens=10, exact=True)
    journal.attempt_start(budget=10, reason="fixed")
    journal.request_sent(budget=10, reason="fixed")
    journal.usage(prompt_tokens=None, completion_tokens=5, reasoning_tokens=0,
                  cached_tokens=None, exact=True)

    aggregate = journal.aggregate()
    assert aggregate["prompt_tokens_total"] is None
    assert aggregate["cache_share"] is None


def test_the_report_shows_the_cache_share_next_to_ttft(tmp_path):
    rows = [
        {"task_id": f"T-{index}", "category": "A", "answer_tokens": 100,
         "reasoning_tokens": 0, "prompt_tokens_total": 1000,
         "cached_tokens": 250 * index, "cache_share": 0.25 * index,
         "wall_s": 1.0, "ttft_s": 0.5, "usage_known": True}
        for index in range(1, 4)
    ]
    markdown = render_markdown(_run(tmp_path, rows))
    assert "Медиана cache, %" in markdown
    assert "50 %" in markdown


def test_non_finite_speed_values_print_as_not_available():
    from dailybench import report

    assert report._fmt(float("nan")) == "n/a"
    assert report._fmt(float("inf")) == "n/a"
    assert report._fmt(1.234) == "1.23"
