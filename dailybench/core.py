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
from typing import Any, Callable, Protocol, Sequence

from . import schema

# --------------------------------------------------------------------------- #
# Suite profiles
# --------------------------------------------------------------------------- #

DEFAULT_PROFILE = "baseline-20261020"
"""Frozen suite version every old artefact belongs to."""

PROFILES: tuple[str, ...] = (DEFAULT_PROFILE, "pilot", "stretch-v1")
"""Accepted profile names.

``baseline`` is the published 47-scenario suite; ``pilot`` carries only the
scenarios that are not admitted to the ranking yet; ``stretch`` adds tasks to
the baseline and redistributes the same category totals inside it, so the
A–J denominator stays 100 while ``pilot`` is read per task instead of as a
/new total.
"""


def profile_of(cfg: Any) -> str:
    """Validated profile of a run configuration."""
    profile = getattr(cfg, "profile", DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise ValueError(f"неизвестный профиль {profile!r}; допустимы: {', '.join(PROFILES)}")
    return profile


# --------------------------------------------------------------------------- #
# Scoring constants
# --------------------------------------------------------------------------- #

CATEGORY_POINTS: dict[str, float] = {
    "A": 10.0,  # instruction following
    "B": 15.0,  # reasoning
    "C": 20.0,  # coding
    "D": 14.0,  # tool calling / agentic
    "E": 10.0,  # long context
    "F": 8.0,  # hallucination resistance
    "G": 8.0,  # structured output
    "H": 6.0,  # multi-turn dialogue
    "I": 4.0,  # retrieval over workspace files
    "J": 5.0,  # code review
}

CATEGORY_TITLES: dict[str, str] = {
    "A": "Instruction following",
    "B": "Reasoning",
    "C": "Coding",
    "D": "Tool calling",
    "E": "Long context",
    "F": "Hallucination resistance",
    "G": "Structured output",
    "H": "Multi-turn dialogue",
    "I": "Retrieval over files",
    "J": "Code review",
}

# Categories that can still be scored when the long-context admission fails:
# every category except E, whose scenarios need the full window admitted.
ADMISSION_CATEGORIES = tuple(key for key in CATEGORY_POINTS if key != "E")
SHORT_TOTAL = sum(CATEGORY_POINTS[k] for k in ADMISSION_CATEGORIES)  # 90.0
FULL_TOTAL = sum(CATEGORY_POINTS.values())  # 100.0

SCORED_CATEGORIES = tuple(CATEGORY_POINTS)
"""Categories that add up to the leaderboard verdict."""

SHOWCASE_POINTS: dict[str, dict[str, float]] = {
    DEFAULT_PROFILE: {
        "K": 80.0,  # 20 visual scenarios, 4 checks of 1.0 each
        "L": 29.0,  # 29 art works; information only, not in the verdict
    },
    "pilot": {"K": 12.0, "L": 29.0},
    "stretch-v1": {"K": 120.0, "L": 39.0},
}
"""Categories that are measured and reported but never enter the verdict.

The visual block answers "умеет ли модель собрать работающую сцену" with
pictures, not with points: mixing it into the 100-point total would make the
score depend on a browser build, so it is kept out of numerator and
denominator and shown on its own screen of the report. For K the checks are
objective (coordinates against a Python recomputation), so it keeps its points.
"""

ART_CATEGORY = "L"
"""Арт-блок баллов не получает: он рисуется для людей.

Модель создаёт свободные визуальные работы: у одной темы нет вовсе, у
остальных назван только сюжет; композицию, палитру, тональность и способ
рисования выбирает сама. Отчёт показывает кадр и исходную страницу.
Единственная автоматическая проверка — загрузка HTML; художественное качество
оценивает человек. Специальный API сцены и анимация не требуются.
"""

#: Категории, которые рисуются все, но оцениваются глазами.
ART_CATEGORIES = (ART_CATEGORY,)

SHOWCASE_TITLES: dict[str, str] = {
    "K": "Visual simulations (Three.js)",
    "L": "Free-form art",
}

SHOWCASE_CATEGORIES = ("K",)  # категории витрины (без арт-блока)

#: Все категории, которые показываются кадрами, включая оценку глазами.
VISUAL_CATEGORIES = SHOWCASE_CATEGORIES + ART_CATEGORIES

PROFILE_POINTS: dict[str, dict[str, float]] = {
    "pilot": dict(CATEGORY_POINTS),
    "stretch-v1": dict(CATEGORY_POINTS),
}
"""Category totals per profile where they differ from the baseline.

``pilot`` carries a single admitted scenario per category, so the full category
total rides on that one task; ``stretch`` keeps the A–J total at 100 and only
enlarges the visual showcase (see ``SHOWCASE_POINTS``). The totals are
distributed evenly over the tasks of the profile by ``suite.build_suite``, so
the denominator stays independent of how many new scenarios a category
received.
"""


#: Bumped whenever the evaluator contract changes: the meaning of a check, the
#: shape of ``Score.checks``, or how points map to checks. The suite digest
#: folds it in, so a resume can never mix scores produced under two contracts.
EVALUATOR_CONTRACT = 1


def points_for(key: str, profile: str = DEFAULT_PROFILE) -> float | None:
    """Declared points of a category, if it has any.

    The art block's number is informational: it is drawn, not scored into the
    verdict. A profile may override the total of a category (the stretch
    showcase); the baseline constants stay untouched.
    """
    override = PROFILE_POINTS.get(profile) or {}
    if key in override:
        return override[key]
    if key in CATEGORY_POINTS:
        return CATEGORY_POINTS[key]
    showcase = SHOWCASE_POINTS.get(profile) or SHOWCASE_POINTS[DEFAULT_PROFILE]
    return showcase.get(key)


def profile_total(profile: str = DEFAULT_PROFILE) -> float:
    """Full denominator of a profile, independent of filters or a partial run."""
    keys = set(CATEGORY_POINTS) | set(SHOWCASE_POINTS.get(profile) or SHOWCASE_POINTS[DEFAULT_PROFILE])
    return sum(points_for(key, profile) or 0.0 for key in keys)


def category_title(key: str) -> str:
    return CATEGORY_TITLES.get(key) or SHOWCASE_TITLES.get(key) or key


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

    dialogue: list[str] | None = None
    """Follow-up user messages, for scripted multi-turn scenarios.

    The runner sends ``prompt`` first and, after every assistant reply that
    carries no tool calls, the next entry of this list. The task ends when the
    dialogue is exhausted, so a category can score a correction the model had to
    apply mid-conversation. Categories that need no follow-ups leave it ``None``.
    """

    meta: dict[str, Any] = field(default_factory=dict)
    """Driver hints: ``{"driver": "tools", "target_tokens": 16384, ...}``."""

    block: str | None = None
    """``deterministic`` or ``open``; derived from ``category`` when omitted."""

    family: str | None = None
    """Taxonomy family (``det/math`` … ``vis/aes``); derived from ``category``."""

    languages: list[str] = field(default_factory=lambda: ["ru"])
    """Language tags of the brief. A twin scenario has its own id and fixtures."""

    cultures: list[str] = field(default_factory=list)
    """Cultures the brief is adapted to (calendars, units, names, formats)."""

    fresh_after: str | None = None
    """ISO date after which the scenario counts as a fresh slice; ``None`` = evergreen."""

    private: bool = False
    """A private task is excluded from the published prompt view."""

    holdout_seeds: list[int] = field(default_factory=list)
    """Blind seeds, never used while the scenario was tuned."""

    def __post_init__(self) -> None:
        if self.block is None:
            self.block = schema.block_for(self.category)
        if self.family is None:
            self.family = schema.family_for(self.category)
        if not self.languages:
            self.languages = ["ru"]
        if self.cultures is None:
            self.cultures = []
        if self.holdout_seeds is None:
            self.holdout_seeds = []
        errors = schema.validate_task_fields(self)
        if errors:
            raise ValueError(f"некорректное задание {self.id!r}: " + "; ".join(errors))

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
        """Total generated tokens as the runtime counts them.

        OpenAI-compatible usage includes reasoning in ``completion_tokens``
        and exposes the reasoning share in ``completion_tokens_details``:
        adding them on top double-counts (170 instead of the reported 100).
        Backends that exclude reasoning from completion are recognised by
        ``completion < reasoning`` and summed instead.
        """
        if self.completion_tokens is None:
            return None
        if self.reasoning_tokens is None or self.reasoning_tokens <= self.completion_tokens:
            return self.completion_tokens
        return self.completion_tokens + self.reasoning_tokens


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

    # -- spend (documented, not capped) ------------------------------------ #

    def spend(self) -> dict[str, Any]:
        """What the task cost, in the same fields for every model.

        The run does not cap generation any more: the limit is the window, so
        the numbers below are the honest record of what a model chose to spend
        instead of how much it was allowed to.
        """
        answer = think = 0
        ttft: float | None = None
        total = 0.0
        turns_with_tokens = 0
        for turn in self.turns:
            reply = turn.reply
            if reply.reasoning_tokens is not None:
                think += reply.reasoning_tokens
            else:
                think += len(reply.reasoning or "") // 3
            if reply.completion_tokens is not None:
                turns_with_tokens += 1
                answer += max(0, reply.completion_tokens - (reply.reasoning_tokens or 0))
            total += reply.total_s or 0.0
            if ttft is None and reply.ttft_s is not None:
                ttft = reply.ttft_s
        return {
            "turns": len(self.turns),
            "tool_calls": len(self.tool_log),
            "prompt_tokens_last": self.prompt_tokens(),
            "answer_tokens": answer if turns_with_tokens else None,
            "reasoning_tokens": think if turns_with_tokens or self.turns else None,
            "answer_chars": sum(len(t.reply.text or "") for t in self.turns),
            "reasoning_chars": sum(len(t.reply.reasoning or "") for t in self.turns),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "generate_s": round(total, 3) or None,
            "tokens_per_s": round((answer + think) / total, 2) if total > 0.5 else None,
            "finish_reasons": [t.reply.finish_reason for t in self.turns],
        }

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
    """Set when the run itself broke (transport error, budget abort, agent
    timeout). It marks the run, not a wrong answer: the checks the evaluator
    still managed to make are objectively passed, and an artefact written
    before the agent hit its wall clock is graded from disk like any other."""

    max_points: float | None = None
    """Fixed budget of the task as the harness knows it.

    The harness sets it on every score, so the denominator is a property of the
    task and never of the answer: an evaluator that adds a penalty check cannot
    enlarge ``possible`` beyond the declared maximum.
    """

    @property
    def status(self) -> str:
        """Machine-readable state of the row: ``ok``, ``skipped`` or ``error``."""
        if self.skipped:
            return "skipped"
        if self.error:
            return "error"
        return "ok"

    @property
    def earned(self) -> float:
        if self.skipped:
            return 0.0
        return round(sum(c.points for c in self.checks if c.passed), 4)

    @property
    def possible(self) -> float:
        """Denominator of the task: its fixed budget, or the issued checks.

        A skipped task stays at zero: not verified counts in neither numerator
        nor denominator. Otherwise a score without a harness budget falls back
        to the checks actually issued.
        """
        if self.skipped:
            return 0.0
        if self.max_points is not None:
            return round(float(self.max_points), 4)
        return round(sum(c.points for c in self.checks), 4)

    def add(self, name: str, passed: bool, points: float, detail: str = "") -> None:
        self.checks.append(Check(name=name, passed=bool(passed), points=points, detail=detail))

    def fit_points(self, max_points: float | None = None) -> None:
        """Scale the issued checks to the task budget, keeping their relation.

        Category modules hard-code the baseline per-check points; a profile that
        splits a category total differently must not leave the sum of checks
        above or below ``max_points``. Proportional scaling preserves the
        partial credit of every check and puts the rounding tail on the last
        one, so the sum is exact.
        """
        budget = float(self.max_points if max_points is None else max_points)
        total = sum(check.points for check in self.checks)
        if not self.checks or abs(total - budget) <= 1e-9:
            return
        if total <= 0:
            share = round(budget / len(self.checks), 4)
            for index, check in enumerate(self.checks):
                check.points = share if index < len(self.checks) - 1 else round(
                    budget - share * (len(self.checks) - 1), 4
                )
            return
        ratio = budget / total
        scaled = [round(check.points * ratio, 4) for check in self.checks]
        scaled[-1] = max(0.0, round(budget - sum(scaled[:-1]), 4))
        for check, points in zip(self.checks, scaled):
            check.points = points

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "earned": self.earned,
            "possible": self.possible,
            "max_points": self.max_points,
            "status": self.status,
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
            max_points=data.get("max_points"),
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
    # ``expected_ids`` is the baseline list; a profile may override it.
    expected_ids: tuple[str, ...] = ()
    expected_ids_by_profile: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def expected_ids_for(self, profile: str = DEFAULT_PROFILE) -> tuple[str, ...]:
        """Expected scenario ids in one profile (baseline list by default)."""
        return self.expected_ids_by_profile.get(profile, self.expected_ids)


REGISTRY: dict[str, CategoryModule] = {}


def register(
    key: str,
    title: str,
    points: float,
    generate: Callable[[random.Random, Any], list[TaskInstance]],
    evaluate: Callable[[TaskInstance, Transcript], Score],
    expected_ids: Sequence[str] = (),
    expected_ids_by_profile: dict[str, Sequence[str]] | None = None,
) -> None:
    """Register a category module. Called at import time by ``tasks/*``.

    ``expected_ids`` is the baseline contract; ``expected_ids_by_profile``
    overrides it for a profile that generates more (or other) scenarios.
    """
    if key in REGISTRY:
        raise RuntimeError(f"category {key!r} registered twice")
    declared = points_for(key)
    if declared is not None and abs(points - declared) > 1e-9:
        raise RuntimeError(f"category {key!r} declares {points} points, expected {declared}")
    unknown = set(expected_ids_by_profile or {}) - set(PROFILES)
    if unknown:
        raise RuntimeError(f"category {key!r}: неизвестные профили {sorted(unknown)}")
    REGISTRY[key] = CategoryModule(
        key=key,
        title=title,
        points=float(points),
        generate=generate,
        evaluate=evaluate,
        expected_ids=tuple(expected_ids),
        expected_ids_by_profile={
            name: tuple(ids) for name, ids in (expected_ids_by_profile or {}).items()
        },
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
    if callable(value):
        # Visual scenarios keep their checks as callables; a published oracle
        # must stay readable and byte-stable, so it carries the label instead of
        # a memory address.
        return f"<проверка: {getattr(value, 'check_label', None) or getattr(value, '__qualname__', 'callable')}>"
    return str(value)


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

