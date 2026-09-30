"""Shared data model for DailyBench-16.

Every task module implements two functions and registers them:

    def generate(rng: random.Random, cfg: RunConfig) -> list[TaskInstance]: ...
    def evaluate(task: TaskInstance, tr: Transcript) -> Score: ...

Contract rules:

* ``generate`` is deterministic in ``rng`` and may not read the network or clock.
* Everything an evaluator needs lives in ``task.oracle``; evaluators never see
  the model's reasoning field, only the transcript.
* ``evaluate`` must award points only for observable results (files, test
  outcomes, final answer text), never for the shape of the transcript.
* ``task.oracle["reference"]`` is the canonical answer used by the self-check
  provider; it must score full points through ``evaluate``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

# --------------------------------------------------------------------------- #
# Scoring constants
# --------------------------------------------------------------------------- #

CATEGORY_POINTS: dict[str, float] = {
    "A": 15.0,  # instruction following
    "B": 20.0,  # reasoning
    "C": 25.0,  # coding
    "D": 15.0,  # tool calling / agentic
    "E": 15.0,  # long context
    "F": 10.0,  # hallucination resistance
}

CATEGORY_TITLES: dict[str, str] = {
    "A": "Instruction following",
    "B": "Reasoning",
    "C": "Coding",
    "D": "Tool calling",
    "E": "Long context",
    "F": "Hallucination resistance",
}

# Categories that can still be scored when the long-context admission fails.
ADMISSION_CATEGORIES = ("A", "B", "C", "D", "F")
SHORT_TOTAL = sum(CATEGORY_POINTS[k] for k in ADMISSION_CATEGORIES)  # 85.0
FULL_TOTAL = sum(CATEGORY_POINTS.values())  # 100.0


# --------------------------------------------------------------------------- #
# Task definitions
# --------------------------------------------------------------------------- #


@dataclass
class TaskInstance:
    """One generated scenario, ready to be handed to a model."""

    id: str
    """Stable scenario id, e.g. ``"IF-1"``, ``"R-7"``, ``"LC-5"``, ``"F"``."""

    category: str
    """One of ``"A".."F"``."""

    title: str
    prompt: str
    """The user message sent to the model."""

    max_points: float

    oracle: dict[str, Any] = field(default_factory=dict)
    """Reference answer plus every datum the evaluator needs.

    ``reference`` is required for single-answer tasks: the canonical reply text
    that must score full points (used by the self-check provider).
    """

    system: str | None = None
    """Extra system text. Normally ``None`` — the run uses the model's own
    chat template plus the minimal neutral instruction from ``runner``."""

    workspace: dict[str, str] | None = None
    """Files materialised before the task runs (coding repos, tool fixtures)."""

    tools: list[dict[str, Any]] | None = None
    """OpenAI function-calling schemas, for multi-turn tool scenarios."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Driver hints: ``{"driver": "tools", "target_tokens": 16384, ...}``."""

    def check(self, max_points: float | None = None) -> "TaskInstance":
        if max_points is not None:
            self.max_points = float(max_points)
        return self


@dataclass
class ModelReply:
    """One assistant message plus the measurements that came with it."""

    text: str = ""
    """Visible answer text (reasoning excluded when the API separates it)."""

    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """``[{"id": str, "name": str, "arguments": str|dict}]``."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    ttft_s: float | None = None
    total_s: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def generated_tokens(self) -> int | None:
        """Completion plus reasoning tokens, when the runtime reports both."""
        if self.completion_tokens is None:
            return None
        return self.completion_tokens + (self.reasoning_tokens or 0)


@dataclass
class Turn:
    """One request/response exchange inside a (possibly multi-turn) run."""

    messages: list[dict[str, Any]]
    reply: ModelReply
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Transcript:
    """Everything recorded while a task ran. The only input to ``evaluate``."""

    task_id: str
    category: str
    turns: list[Turn] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    """``[{"turn": int, "name": str, "arguments": dict, "result": str,
    "error": bool, "illegal": bool}]``."""

    workspace_after: dict[str, str | None] = field(default_factory=dict)
    """Relative path -> final content, ``None`` when the file is gone."""

    workspace_before: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    """Set when the run itself broke (transport error, budget abort)."""

    duration_s: float | None = None

    # -- convenience ------------------------------------------------------- #

    @property
    def final_text(self) -> str:
        """Last visible assistant text, or the concatenation of all of them."""
        for turn in reversed(self.turns):
            if turn.reply.text and turn.reply.text.strip():
                return turn.reply.text
        return ""

    @property
    def all_text(self) -> str:
        return "\n".join(t.reply.text for t in self.turns if t.reply.text)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.tool_log

    def tool_names(self) -> list[str]:
        return [c["name"] for c in self.tool_log]

    def called(self, *names: str) -> bool:
        called = set(self.tool_names())
        return any(n in called for n in names)

    def calls_named(self, name: str) -> list[dict[str, Any]]:
        return [c for c in self.tool_log if c["name"] == name]

    def prompt_tokens(self) -> int | None:
        """Prompt tokens of the last turn — the reported context usage."""
        for turn in reversed(self.turns):
            if turn.reply.prompt_tokens is not None:
                return turn.reply.prompt_tokens
        return None

    def generated_tokens(self) -> int | None:
        total = 0
        seen = False
        for turn in self.turns:
            n = turn.reply.generated_tokens
            if n is not None:
                total += n
                seen = True
        return total if seen else None

    def first_turn_text(self) -> str:
        return self.turns[0].reply.text if self.turns else ""


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    points: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "points": self.points,
        }


@dataclass
class Score:
    task_id: str
    category: str
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    admission: bool | None = None
    """``False`` when the configuration failed the 32K admission for a
    long-context task: the category is reported as not admitted instead of
    comparing an out-of-memory run as a quality result."""

    skipped: str | None = None
    """Set when the configuration declared the task's capability as not
    applicable. Skipped tasks count in neither the numerator nor the
    denominator — an unverified category is not a zero."""

    error: str | None = None

    @property
    def earned(self) -> float:
        if self.error or self.skipped:
            return 0.0
        return round(sum(c.points for c in self.checks if c.passed), 4)

    @property
    def possible(self) -> float:
        return round(sum(c.points for c in self.checks), 4)

    def add(self, name: str, passed: bool, points: float, detail: str = "") -> None:
        self.checks.append(Check(name=name, passed=bool(passed), points=points, detail=detail))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "earned": self.earned,
            "possible": self.possible,
            "admission": self.admission,
            "skipped": self.skipped,
            "error": self.error,
            "notes": self.notes,
            "checks": [c.as_dict() for c in self.checks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Score":
        checks = [Check(**c) for c in data.get("checks", [])]
        return cls(
            task_id=data["task_id"],
            category=data["category"],
            checks=checks,
            notes=list(data.get("notes", [])),
            admission=data.get("admission"),
            skipped=data.get("skipped"),
            error=data.get("error"),
        )


# --------------------------------------------------------------------------- #
# Category registry
# --------------------------------------------------------------------------- #


@dataclass
class CategoryModule:
    key: str
    title: str
    points: float
    generate: Callable[[random.Random, Any], list[TaskInstance]]
    evaluate: Callable[[TaskInstance, Transcript], Score]
    # Scenarios this module is expected to emit, for suite integrity checks.
    expected_ids: tuple[str, ...] = ()


REGISTRY: dict[str, CategoryModule] = {}


def register(
    key: str,
    title: str,
    points: float,
    generate: Callable[[random.Random, Any], list[TaskInstance]],
    evaluate: Callable[[TaskInstance, Transcript], Score],
    expected_ids: Sequence[str] = (),
) -> None:
    """Register a category module. Called at import time by ``tasks/*``."""
    if key in REGISTRY:
        raise RuntimeError(f"category {key!r} registered twice")
    if abs(points - CATEGORY_POINTS.get(key, points)) > 1e-9:
        raise RuntimeError(
            f"category {key!r} declares {points} points, expected {CATEGORY_POINTS.get(key)}"
        )
    REGISTRY[key] = CategoryModule(
        key=key,
        title=title,
        points=float(points),
        generate=generate,
        evaluate=evaluate,
        expected_ids=tuple(expected_ids),
    )


def registered_categories() -> list[CategoryModule]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def json_default(value: Any) -> Any:
    """Encoder for fixture values that are exact but not JSON-native.

    Generators are allowed to keep exact values (``Decimal`` for prices,
    ``Fraction`` for probabilities, ``Path`` for sandbox locations) in their
    oracles, because comparisons must be exact. Persisting a run must not fail
    on them, so the transcript writer falls back to their exact string form.
    """
    from decimal import Decimal
    from fractions import Fraction

    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def parse_json_object(text: str) -> Any:
    """Tiny helper for fixture JSON round-tripping in tests."""
    import json

    return json.loads(text)


# --------------------------------------------------------------------------- #
# Tool scenarios (category D)
# --------------------------------------------------------------------------- #


@dataclass
class ToolOutcome:
    """What a simulated tool returned to the model."""

    text: str
    error: bool = False
    illegal: bool = False
    """``True`` when the call itself violated the task's rules (for example a
    deletion outside the permitted directory). The action still happened, so
    the evaluator can score the damage; the flag only makes it visible."""


class ToolDriver(Protocol):
    """Category-owned simulator driven by the runner's tool loop."""

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas exposed to the model."""

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Run one call, mutating whatever state the fixture holds."""


TOOL_DRIVERS: dict[str, Callable[[TaskInstance, "Path"], ToolDriver]] = {}


def register_tool_driver(
    category: str,
    factory: Callable[[TaskInstance, "Path"], ToolDriver],
) -> None:
    """Category D (and later agentic categories) register their simulator."""
    if category in TOOL_DRIVERS:
        raise RuntimeError(f"tool driver for {category!r} registered twice")
    TOOL_DRIVERS[category] = factory


def merge_scores(scores: Iterable[Score]) -> dict[str, dict[str, float]]:
    """Aggregate per-category totals, honouring the admission rule."""
    totals: dict[str, dict[str, float]] = {}
    for s in scores:
        row = totals.setdefault(
            s.category,
            {"earned": 0.0, "possible": 0.0, "admitted": 1.0, "tasks": 0.0},
        )
        row["tasks"] += 1
        row["possible"] += s.possible
        if s.admission is False:
            row["admitted"] = 0.0
        else:
            row["earned"] += s.earned
    return totals
