"""Per-task timing journal, in one monotonic JSONL stream.

The score says whether an answer was right; it says nothing about how the
answer was paid for. This module records that second half — queue time, time
to first token, decode chunks, tool and evaluator spans, browser stages — on a
clock that cannot jump, and reduces it to an aggregate with explicit unknowns.

Rules that are load bearing:

* **Unknown is ``None``.** An endpoint that reports no ``usage`` leaves
  ``answer_tokens``/``reasoning_tokens`` unknown; a zero would read as a free
  answer. Aggregates that need the missing piece stay ``None`` instead of
  summing the half that happened to arrive.
* **Reasoning is not double counted.** OpenAI-compatible ``completion_tokens``
  already contains reasoning; ``answer_tokens`` is
  ``max(0, completion - reasoning)`` and ``reasoning_tokens`` travels apart.
* **The journal is append-only.** Events are numbered and timed from the first
  instant of the task, so a reader can tell order from the ``seq`` field even
  when two spans share a timestamp.

The active journal is also reachable from the evaluator (``active()``), because
the browser renderer is several calls deep inside a category module and has no
handle to the runner.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

#: Event names this journal accepts. A typo in a caller would otherwise become
#: an unnamed row that no reader knows how to interpret.
EVENTS = frozenset(
    {
        "attempt_start",
        "request_sent",
        "first_token",
        "decode_chunk",
        "usage",
        "tool_call_start",
        "tool_call_end",
        "tool_call_ok",
        "evaluator_start",
        "evaluator_end",
        "browser_stage",
        "task_end",
    }
)

#: Metrics the correlation section may read. The order is part of the report.
METRICS = ("spend_tokens", "wall_s", "context_occupancy")

_ACTIVE: contextvars.ContextVar["Telemetry | None"] = contextvars.ContextVar(
    "dailybench_telemetry", default=None
)


def active() -> "Telemetry | None":
    """The journal bound to this task, if any — for evaluators and renderers."""
    return _ACTIVE.get()


@contextlib.contextmanager
def bind(journal: "Telemetry | None"):
    token = _ACTIVE.set(journal)
    try:
        yield journal
    finally:
        _ACTIVE.reset(token)


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


class Telemetry:
    """One task's append-only event journal and its honest aggregate.

    ``clock`` is injectable so tests can drive a deterministic timeline. The
    production clock is ``time.monotonic``: wall-clock jumps must not make a
    decode rate negative.
    """

    def __init__(
        self,
        task_id: str,
        *,
        category: str | None = None,
        window: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.task_id = task_id
        self.category = category
        self.window = int(window) if window else None
        self._clock = clock
        self._started = clock()
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._closed = False
        self._request = -1
        self._attempts: list[dict[str, Any]] = []
        self._requests: list[dict[str, Any]] = []
        self._firsts: dict[int, dict[str, Any]] = {}
        self._chunks: dict[int, list[dict[str, Any]]] = {}
        self._usage: dict[int, dict[str, Any]] = {}
        self._tools: list[dict[str, Any]] = []
        self._evaluators: list[dict[str, Any]] = []
        self._browser: list[dict[str, Any]] = []
        self._end: dict[str, Any] | None = None

    # -- raw journal ------------------------------------------------------- #

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        if event not in EVENTS:
            raise ValueError(f"unknown telemetry event: {event}")
        with self._lock:
            if self._closed and event != "task_end":
                raise RuntimeError("telemetry journal is closed")
            self._seq += 1
            row = {
                "seq": self._seq,
                "t": _clean(self._clock() - self._started),
                "event": event,
                "task_id": self.task_id,
                **_clean(fields),
            }
            self._events.append(row)
            return row

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def to_jsonl(self) -> str:
        with self._lock:
            rows = list(self._events)
        if not rows:
            return ""
        return "\n".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows
        ) + "\n"

    @classmethod
    def from_jsonl(cls, text: str, *, task_id: str = "task") -> "Telemetry":
        """Rebuild a readable journal, spans included, from saved JSONL."""
        journal = cls(task_id)
        journal._events = [json.loads(line) for line in text.splitlines() if line.strip()]
        journal._seq = max((int(row.get("seq") or 0) for row in journal._events), default=0)
        for row in journal._events:
            event = row.get("event")
            t = float(row.get("t") or 0.0)
            if event == "attempt_start":
                journal._request += 1
                journal._attempts.append(
                    {
                        "index": journal._request,
                        "budget": row.get("budget"),
                        "reason": row.get("reason") or "",
                        "prompt_tokens": row.get("prompt_tokens"),
                        "prompt_estimate": row.get("prompt_estimate"),
                        "prompt_source": row.get("prompt_source"),
                        "unbounded": row.get("unbounded"),
                        "turn": row.get("turn"),
                        "queue_known": bool(row.get("queue_known")),
                        "sent_t": None,
                        "t": t,
                    }
                )
            elif event == "request_sent":
                journal._requests.append(
                    {
                        "index": journal._request,
                        "budget": row.get("budget"),
                        "reason": row.get("reason") or "",
                        "prompt_tokens": row.get("prompt_tokens"),
                        "prompt_estimate": row.get("prompt_estimate"),
                        "prompt_source": row.get("prompt_source"),
                        "turn": row.get("turn"),
                        "sent_t": t,
                    }
                )
            elif event == "first_token" and journal._request >= 0:
                journal._firsts.setdefault(
                    journal._request,
                    {"ttft_s": row.get("ttft_s"), "tokens": row.get("tokens"), "t": t},
                )
            elif event == "decode_chunk" and journal._request >= 0:
                journal._chunks.setdefault(journal._request, []).append(
                    {"interval_s": row.get("interval_s"), "tokens": row.get("tokens"), "t": t}
                )
            elif event == "usage" and journal._request >= 0:
                journal._usage[journal._request] = {
                    "prompt_tokens": row.get("prompt_tokens"),
                    "completion_tokens": row.get("completion_tokens"),
                    "reasoning_tokens": row.get("reasoning_tokens"),
                    "cached_tokens": row.get("cached_tokens"),
                    "exact": bool(row.get("exact")),
                    "t": t,
                }
            elif event == "tool_call_end":
                journal._tools.append(
                    {"name": row.get("name"), "ok": row.get("ok"),
                     "duration_s": row.get("duration_s"), "error": row.get("error")}
                )
            elif event == "evaluator_end":
                journal._evaluators.append(
                    {"ok": row.get("ok"), "duration_s": row.get("duration_s"),
                     "error": row.get("error")}
                )
            elif event == "browser_stage":
                journal._browser.append(
                    {"stage": row.get("stage"), "status": row.get("status"),
                     "duration_s": row.get("duration_s")}
                )
            elif event == "task_end":
                journal._end = {"outcome": row.get("outcome"), "error": row.get("error"), "t": t}
                journal._closed = True
        return journal

    # -- spans ------------------------------------------------------------- #

    def attempt_start(
        self,
        *,
        budget: int | None = None,
        reason: str = "",
        prompt_tokens: int | None = None,
        prompt_estimate: int | None = None,
        prompt_source: str | None = None,
        unbounded: bool | None = None,
        turn: int | None = None,
        queue_known: bool = True,
    ) -> int:
        """A request is about to be prepared.

        ``queue_known=False`` is for an event stream replayed after the fact
        (the agent): the order is real, the gap between preparation and send is
        not.
        """
        with self._lock:
            self._request += 1
            index = self._request
        info = {
            "index": index,
            "budget": budget,
            "reason": reason,
            "prompt_tokens": prompt_tokens,
            "prompt_estimate": prompt_estimate,
            "prompt_source": prompt_source,
            "unbounded": unbounded,
            "turn": turn,
            "queue_known": queue_known,
            "sent_t": None,
            "t": self._clock() - self._started,
        }
        self._attempts.append(info)
        self.record("attempt_start", **{k: v for k, v in info.items() if k not in ("sent_t", "t")})
        return index

    def request_sent(
        self,
        *,
        budget: int | None = None,
        reason: str = "",
        prompt_tokens: int | None = None,
        prompt_estimate: int | None = None,
        prompt_source: str | None = None,
        turn: int | None = None,
    ) -> int:
        with self._lock:
            index = self._request
            if index < 0:
                index = 0
                self._request = 0
                info = {
                    "index": 0,
                    "budget": budget,
                    "reason": reason,
                    "prompt_tokens": prompt_tokens,
                    "prompt_estimate": prompt_estimate,
                    "prompt_source": prompt_source,
                    "unbounded": None,
                    "turn": turn,
                    "queue_known": False,
                    "sent_t": None,
                }
                self._attempts.append(info)
            request = {
                "index": index,
                "budget": budget,
                "reason": reason,
                "prompt_tokens": prompt_tokens,
                "prompt_estimate": prompt_estimate,
                "prompt_source": prompt_source,
                "turn": turn,
                "sent_t": self._clock() - self._started,
            }
            self._requests.append(request)
        self.record(
            "request_sent",
            budget=budget,
            reason=reason,
            prompt_tokens=prompt_tokens,
            prompt_estimate=prompt_estimate,
            prompt_source=prompt_source,
            turn=turn,
        )
        return index

    def _requested(self) -> int:
        with self._lock:
            return self._request

    def first_token(self, *, ttft_s: float | None = None, tokens: int | None = None) -> None:
        index = self._requested()
        # Stamped before the event row: the aggregate needs this instant to
        # derive TTFT from the send when the runtime reported no duration.
        row = {"ttft_s": ttft_s, "tokens": tokens, "t": self._clock() - self._started}
        if index >= 0:
            self._firsts.setdefault(index, row)
        self.record("first_token", ttft_s=ttft_s, tokens=tokens)

    def decode_chunk(self, *, interval_s: float | None, tokens: int | None, chars: int = 0) -> None:
        index = self._requested()
        if index >= 0:
            self._chunks.setdefault(index, []).append(
                {"interval_s": interval_s, "tokens": tokens, "t": self._clock() - self._started}
            )
        self.record("decode_chunk", interval_s=interval_s, tokens=tokens, chars=chars)

    def usage(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        cached_tokens: int | None = None,
        exact: bool = True,
    ) -> None:
        """Provider-reported accounting. ``exact=False`` when the runtime said
        nothing, which keeps every derived total unknown."""
        index = self._requested()
        payload = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "exact": bool(exact),
            "t": self._clock() - self._started,
        }
        if index >= 0:
            self._usage[index] = payload
        self.record("usage", **payload)

    def tool_call_start(self, *, name: str, index: int | None = None) -> None:
        self.record("tool_call_start", name=name, index=index)

    def tool_call_end(
        self,
        *,
        name: str,
        ok: bool,
        duration_s: float | None = None,
        error: bool = False,
    ) -> None:
        row = {"name": name, "ok": bool(ok), "duration_s": duration_s, "error": bool(error)}
        self._tools.append(row)
        self.record("tool_call_end", **row)
        if ok:
            self.record("tool_call_ok", name=name)

    def evaluator_start(self, *, category: str | None = None) -> None:
        self.record("evaluator_start", category=category or self.category)

    def evaluator_end(
        self, *, ok: bool, duration_s: float | None = None, error: str | None = None
    ) -> None:
        self._evaluators.append({"ok": bool(ok), "duration_s": duration_s, "error": error})
        self.record("evaluator_end", ok=bool(ok), duration_s=duration_s, error=error)

    def browser_stage(
        self, stage: str, *, status: str = "start", duration_s: float | None = None
    ) -> None:
        self._browser.append({"stage": stage, "status": status, "duration_s": duration_s})
        self.record("browser_stage", stage=stage, status=status, duration_s=duration_s)

    def task_end(self, *, outcome: str, error: str | None = None) -> None:
        if self._end is None:
            self._end = {
                "outcome": outcome,
                "error": error,
                "t": self._clock() - self._started,
            }
            self._closed = True
            self.record("task_end", outcome=outcome, error=error)

    # -- aggregate --------------------------------------------------------- #

    def _request_span(self, index: int) -> float | None:
        """From send to the last observed token/usage, in seconds."""
        request = next((r for r in self._requests if r["index"] == index), None)
        if request is None or request["sent_t"] is None:
            return None
        end = self._usage.get(index, {}).get("t")
        ends = [end] if end is not None else []
        ends.extend(chunk["t"] for chunk in self._chunks.get(index, []))
        first = self._firsts.get(index, {}).get("t")
        if first is not None:
            ends.append(first)
        if not ends:
            return None
        return max(0.0, max(ends) - request["sent_t"])

    def aggregate(self) -> dict[str, Any]:
        """Reduce the journal without pretending an unmeasured field is zero."""
        requests = list(self._requests)
        attempts = list(self._attempts)
        usage = {index: dict(row) for index, row in self._usage.items()}
        # ``t`` on stored usage is stamped here, once, so the span calculation
        # below is independent of call order.
        usages_exact = [row for row in usage.values() if row.get("exact")]
        all_usage = bool(requests) and len(usages_exact) == len(requests)
        known_answer = all_usage and all(
            row.get("completion_tokens") is not None for row in usage.values()
        )
        known_reasoning = all_usage and all(
            row.get("reasoning_tokens") is not None for row in usage.values()
        )
        known_cache = all_usage and all(
            row.get("cached_tokens") is not None for row in usage.values()
        )

        answer = None
        if known_answer:
            answer = sum(
                max(0, int(row["completion_tokens"]) - int(row.get("reasoning_tokens") or 0))
                for row in usage.values()
            )
        reasoning = (
            sum(int(row["reasoning_tokens"]) for row in usage.values()) if known_reasoning else None
        )
        cached = sum(int(row["cached_tokens"]) for row in usage.values()) if known_cache else None

        ttft: float | None = None
        for index in range(len(requests)):
            first = self._firsts.get(index)
            if not first:
                continue
            value = first.get("ttft_s")
            if value is None:
                request = next(r for r in requests if r["index"] == index)
                if request["sent_t"] is not None and first.get("t") is not None:
                    value = first["t"] - request["sent_t"]
            if value is not None:
                ttft = float(value)
                break

        decode_tokens = 0.0
        decode_seconds = 0.0
        decode_estimated = False
        for index, chunks in self._chunks.items():
            if not chunks:
                continue
            start = self._firsts.get(index, {}).get("t")
            if start is None and chunks:
                start = chunks[0]["t"]
            exact = usage.get(index, {}).get("completion_tokens")
            if exact is not None and usage.get(index, {}).get("exact"):
                tokens = float(exact)
            else:
                tokens = float(sum(int(c.get("tokens") or 0) for c in chunks))
                decode_estimated = True
            window = max(0.0, chunks[-1]["t"] - (start if start is not None else chunks[0]["t"]))
            if window > 0 and tokens > 0:
                decode_tokens += tokens
                decode_seconds += window
        decode_rate = (
            round(decode_tokens / decode_seconds, 3) if decode_seconds > 0 and decode_tokens else None
        )

        wall = self._end["t"] if self._end else None
        queue_values: list[float] = []
        queue_known = True
        for info in attempts:
            request = next((r for r in requests if r["index"] == info["index"]), None)
            if not info.get("queue_known") or request is None or request["sent_t"] is None:
                queue_known = False
                continue
            queue_values.append(max(0.0, request["sent_t"] - float(info["t"])))
        queue = round(sum(queue_values), 3) if queue_known and attempts else None

        tools_known = bool(self._tools) and all(
            row.get("duration_s") is not None for row in self._tools
        )
        tool_s = (
            round(sum(float(row["duration_s"]) for row in self._tools), 3)
            if tools_known
            else None
        )
        eval_known = bool(self._evaluators) and all(
            row.get("duration_s") is not None for row in self._evaluators
        )
        evaluator_s = (
            round(sum(float(row["duration_s"]) for row in self._evaluators), 3)
            if eval_known
            else None
        )
        # ``start`` rows carry no duration; only terminal stages measure.
        browser_rows = [row for row in self._browser if row.get("status") != "start"]
        browser_known = bool(browser_rows) and all(
            row.get("duration_s") is not None for row in browser_rows
        )
        browser_s = (
            round(sum(float(row["duration_s"]) for row in browser_rows), 3)
            if browser_known
            else None
        )

        spans = [self._request_span(index) for index in range(len(requests))]
        model_s = sum(span for span in spans if span is not None) if spans else 0.0
        overhead: float | None = None
        if (
            wall is not None
            and spans
            and all(span is not None for span in spans)
            and evaluator_s is not None
            and queue is not None
            and (not self._tools or tool_s is not None)
            and (not browser_rows or browser_s is not None)
        ):
            overhead = round(max(0.0, wall - model_s - evaluator_s - (tool_s or 0.0)
                                 - (browser_s or 0.0) - queue), 3)

        prompt_last = next(
            (r["prompt_tokens"] for r in reversed(requests) if r.get("prompt_tokens") is not None),
            None,
        )
        prompts = [row.get("prompt_tokens") for row in usage.values()]
        prompt_total = (
            sum(int(value) for value in prompts)
            if requests and len(prompts) == len(requests) and None not in prompts
            else None
        )
        """Prompt tokens of every request: the cost side, not just the last turn."""
        occupancy = None
        if prompt_last is not None and self.window:
            occupancy = round(prompt_last / self.window, 4)

        modes = [row.get("unbounded") for row in self._attempts if row.get("unbounded") is not None]
        unbounded: bool | None
        if not modes:
            unbounded = None
        else:
            unbounded = all(bool(mode) for mode in modes)

        return {
            "task_id": self.task_id,
            "category": self.category,
            "turns": len(requests) if requests else None,
            "tools": len(self._tools) if requests else None,
            "tool_errors": sum(1 for row in self._tools if row.get("error")) if requests else None,
            "ttft_s": round(ttft, 6) if ttft is not None else None,
            "decode_tokens_per_s": decode_rate,
            "decode_tokens_estimated": decode_estimated if self._chunks else None,
            "answer_tokens": answer,
            "reasoning_tokens": reasoning,
            "cached_tokens": cached,
            "prompt_tokens_last": prompt_last,
            "prompt_tokens_total": prompt_total,
            "cache_share": (
                round(cached / prompt_total, 4)
                if cached is not None and prompt_total
                else None
            ),
            "context_occupancy": occupancy,
            "context_window": self.window,
            "wall_s": wall,
            "queue_s": queue,
            "overhead_s": overhead,
            "evaluator_s": evaluator_s,
            "tool_s": tool_s,
            "browser_s": browser_s,
            "unbounded": unbounded,
            "budgeted": None if unbounded is None else not unbounded,
            "usage_known": all_usage,
            "outcome": self._end.get("outcome") if self._end else None,
            "error": self._end.get("error") if self._end else None,
            "events": len(self._events),
        }

    # -- persistence ------------------------------------------------------- #

    def save(self, directory: Path | str, name: str | None = None) -> Path:
        out = Path(directory) / "telemetry"
        out.mkdir(parents=True, exist_ok=True)
        path = out / (name or f"{self.task_id.replace('/', '_')}.jsonl")
        path.write_text(self.to_jsonl(), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# Report-side reduction
# --------------------------------------------------------------------------- #


def rows(
    index: Iterable[Mapping[str, Any]], scores: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join telemetry index entries to score rows by task id.

    A score without telemetry becomes a row with unknown metrics rather than
    disappearing: the unknown share is the measurement, not an exception.
    """
    telemetry = {str(row.get("task_id")): row for row in index if row.get("task_id")}
    out: list[dict[str, Any]] = []
    for score in scores:
        task_id = str(score.get("task_id") or "")
        measured = telemetry.get(task_id) or {}
        earned = float(score.get("earned") or 0.0)
        possible = float(score.get("possible") or score.get("max_points") or 0.0)
        answer = measured.get("answer_tokens")
        reasoning = measured.get("reasoning_tokens")
        spend = (
            int(answer) + int(reasoning)
            if answer is not None and reasoning is not None
            else None
        )
        out.append(
            {
                "task_id": task_id,
                "category": str(score.get("category") or measured.get("category") or "?"),
                "earned": earned,
                "possible": possible,
                "skipped": bool(score.get("skipped")),
                "share": (earned / possible * 100.0) if possible > 0 else None,
                "spend_tokens": spend,
                "prompt_tokens_total": measured.get("prompt_tokens_total"),
                "cached_tokens": measured.get("cached_tokens"),
                "cache_share": measured.get("cache_share"),
                "wall_s": measured.get("wall_s"),
                "context_occupancy": measured.get("context_occupancy"),
                "ttft_s": measured.get("ttft_s"),
                "decode_tokens_per_s": measured.get("decode_tokens_per_s"),
                "budgeted": measured.get("budgeted"),
                "usage_known": measured.get("usage_known"),
                "measured": bool(measured),
            }
        )
    return out


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        rank = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = rank
        position = end + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation by hand: no numpy, deterministic, ties averaged.

    ``None`` when a side is constant (the correlation is undefined) or fewer
    than three pairs exist — two points always fit a line.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx = _average_ranks([float(x) for x in xs])
    ry = _average_ranks([float(y) for y in ys])
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    dx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    dy = math.sqrt(sum((y - my) ** 2 for y in ry))
    if dx == 0 or dy == 0:
        return None
    return round(numerator / (dx * dy), 4)


def unknown_shares(
    data: Iterable[Mapping[str, Any]], metrics: Iterable[str] = METRICS
) -> dict[str, dict[str, Any]]:
    """How much of each metric is unmeasured, per category and in total."""
    rows_list = [row for row in data if not row.get("skipped")]
    out: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        for category in sorted({str(row.get("category")) for row in rows_list}):
            scoped = [row for row in rows_list if str(row.get("category")) == category]
            missing = sum(1 for row in scoped if row.get(metric) is None)
            out.setdefault(category, {})[metric] = {
                "unknown": missing,
                "total": len(scoped),
                "share": round(missing / len(scoped), 3) if scoped else None,
            }
    return out


def correlations(
    data: Iterable[Mapping[str, Any]], metrics: Iterable[str] = METRICS
) -> list[dict[str, Any]]:
    """Score × cost metric per category, in a stable order."""
    rows_list = [row for row in data if not row.get("skipped")]
    out: list[dict[str, Any]] = []
    for category in sorted({str(row.get("category")) for row in rows_list}):
        scoped = [row for row in rows_list if str(row.get("category")) == category]
        for metric in metrics:
            pairs = [
                (float(row["share"]), float(row[metric]))
                for row in scoped
                if row.get("share") is not None and row.get(metric) is not None
            ]
            rho = spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            out.append(
                {
                    "category": category,
                    "metric": metric,
                    "rho": rho,
                    "n": len(pairs),
                    "unknown": len(scoped) - len(pairs),
                }
            )
    return out


def cost_per_point(
    data: Iterable[Mapping[str, Any]], *, complete: bool
) -> dict[str, Any] | None:
    """Tokens per earned point, only when every task is fully measured.

    A partial spend divided by a complete score reads as a cheap run that simply
    skipped work, so ``complete=False`` (or one unknown spend) yields ``None``.
    """
    if not complete:
        return None
    rows_list = [row for row in data if not row.get("skipped")]
    if not rows_list:
        return None
    if any(row.get("spend_tokens") is None or row.get("possible", 0) <= 0 for row in rows_list):
        return None
    tokens = sum(int(row["spend_tokens"]) for row in rows_list)
    earned = sum(float(row["earned"]) for row in rows_list)
    if earned <= 0 or tokens <= 0:
        return None
    return {
        "tokens": tokens,
        "earned": round(earned, 3),
        "tokens_per_point": round(tokens / earned, 2),
    }


def summary_index(data: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact per-category medians, for the report's small index figure."""
    rows_list = [row for row in data if not row.get("skipped")]

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        value = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        return round(value, 3)

    out: list[dict[str, Any]] = []
    for category in sorted({str(row.get("category")) for row in rows_list}):
        scoped = [row for row in rows_list if str(row.get("category")) == category]
        out.append(
            {
                "category": category,
                "n": len(scoped),
                "ttft_s": median([float(r["ttft_s"]) for r in scoped if r.get("ttft_s") is not None]),
                "cache_share": median(
                    [float(r["cache_share"]) for r in scoped if r.get("cache_share") is not None]
                ),
                "decode_tokens_per_s": median(
                    [
                        float(r["decode_tokens_per_s"])
                        for r in scoped
                        if r.get("decode_tokens_per_s") is not None
                    ]
                ),
                "wall_s": median([float(r["wall_s"]) for r in scoped if r.get("wall_s") is not None]),
                "unknown_spend": sum(1 for row in scoped if row.get("spend_tokens") is None),
            }
        )
    return out
