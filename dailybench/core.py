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

import base64
import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import schema

# --------------------------------------------------------------------------- #
# Suite profiles
# --------------------------------------------------------------------------- #

DEFAULT_PROFILE = "baseline-20261020"
"""Frozen suite version every old artefact belongs to."""

PROFILES: tuple[str, ...] = (DEFAULT_PROFILE, "pilot", "stretch-v1", "research-v1")
"""Accepted profile names.

``baseline`` is the published 47-scenario suite; ``pilot`` carries only the
scenarios that are not admitted to the ranking yet; ``stretch`` adds tasks to
the baseline and redistributes the same category totals inside it, so the
A–J denominator stays 100 while ``pilot`` keeps the point weights declared by
its task modules instead of being rescaled to 100. ``research-v1`` is deliberately narrow: ten admitted research
scenarios across seven categories plus six VQA showcases outside the
denominator, so its scored total is 71 and it is not comparable with the
baseline per category.
"""

BASELINE_INCOMPARABLE_PROFILES = frozenset({"research-v1", "pilot"})
"""Profiles whose category set or denominator differs: their numbers are read on their own."""


def comparable_with_baseline(profile: str = DEFAULT_PROFILE) -> bool:
    """Whether a profile may be ranked next to baseline runs."""
    return profile not in BASELINE_INCOMPARABLE_PROFILES


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
    "stretch-v1": {"K": 144.0, "L": 39.0},
    # V is an image-input showcase: six scenarios, one point each, outside the
    # 71-point research denominator.
    "research-v1": {"V": 6.0},
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
    "V": "Image input (VQA)",
}

#: Image-input scenarios are a showcase block too: they need a real multimodal
#: endpoint, and a text-only model must not turn them into a fake accuracy.
SHOWCASE_CATEGORIES = ("K", "V")

#: Все категории, которые показываются кадрами, включая оценку глазами.
VISUAL_CATEGORIES = SHOWCASE_CATEGORIES + ART_CATEGORIES

#: One fixed PNG geometry for every presented image. The size is part of the
#: contract: the evaluator and the oracle are written against it, and a fixture
#: that renames or resizes the file is a different input.
ASSET_SIZE: tuple[int, int] = (768, 512)
ASSET_MEDIA_TYPES: tuple[str, ...] = ("image/png",)
"""Media types a fixture may present. PNG only: preprocessing is fixed."""

PILOT_CATEGORY_POINTS: dict[str, float] = {
    "A": 2.0, "B": 1.0, "C": 8.0, "D": 2.0, "E": 2.0,
    "F": 8.0, "G": 2.0, "H": 4.0, "I": 2.0, "J": 2.0,
}
"""Frozen module weights of the pilot wave; unlike other profiles they are not redistributed.

This is a protocol table, not a live sum. ``test_pilot_keeps_module_weights``
guards it against the weights the task modules declare.
"""


PROFILE_POINTS: dict[str, dict[str, float]] = {
    "stretch-v1": dict(CATEGORY_POINTS),
    # Only the seven categories the research wave reaches exist in this profile;
    # B, G and H have no scenario at all, so their totals are absent rather than
    # zero and the denominator is 71.
    "research-v1": {"A": 10.0, "C": 20.0, "D": 14.0, "E": 10.0, "F": 8.0, "I": 4.0, "J": 5.0},
}
"""Category totals per profile where they differ from the baseline.

``pilot`` deliberately stays out: its module weights live in
``PILOT_CATEGORY_POINTS`` and ``suite.build_suite`` leaves them untouched.
``stretch`` keeps the A–J total at 100 and only enlarges the visual showcase
(see ``SHOWCASE_POINTS``); ``research-v1`` names only the categories it carries
and so drops the rest from its denominator. These totals are distributed
evenly over the tasks of the profile by ``suite.build_suite``, so the
denominator stays independent of how many new scenarios a category received.
"""


#: Bumped whenever the evaluator contract changes: the meaning of a check, the
#: shape of ``Score.checks``, or how points map to checks. The suite digest
#: folds it in, so a resume can never mix scores produced under two contracts.
EVALUATOR_CONTRACT = 4

#: Complete outcome dictionary of the measurement protocol. ``success`` and
#: ``model_failure`` are quiet outcomes: they carry no ``Score.failure`` and are
#: derived from the row (``failure`` is ``None``), the rest are stored. The split
#: below decides whether a run may be ranked at all.
FAILURE_KINDS = (
    "success",
    "model_failure",
    "resource_exhausted",
    "infra_failure",
    "evaluator_failure",
    "integrity_violation",
    "unsupported",
)

#: Kinds ``Score.failure`` actually stores: the two quiet outcomes stay ``None``.
STORED_FAILURE_KINDS = FAILURE_KINDS[2:]

#: Names produced before the protocol dictionary existed. Bundles on disk and
#: their reports still carry them, so they are normalised, never re-written.
LEGACY_FAILURE_KINDS = {
    "provider_unavailable": "infra_failure",
    "transport_unavailable": "infra_failure",
    "evaluator_unavailable": "evaluator_failure",
    "model_protocol_failure": "model_failure",
    "candidate_resource_exhausted": "resource_exhausted",
    "incomplete_scenario": "resource_exhausted",
}

FAILURE_LABELS = {
    "success": "успех",
    "model_failure": "неверный ответ модели",
    "resource_exhausted": "исчерпан объявленный бюджет",
    "infra_failure": "инфраструктура недоступна",
    "evaluator_failure": "оценщик не установил результат",
    "integrity_violation": "отказ в удостоверении",
    "unsupported": "capability не поддерживается",
}

#: Legacy spellings keep their old wording: an old run must read unchanged.
FAILURE_LABELS.update({
    "provider_unavailable": "провайдер недоступен",
    "transport_unavailable": "транспорт оборвался",
    "evaluator_unavailable": "evaluator недоступен",
    "model_protocol_failure": "сбой протокола модели",
    "candidate_resource_exhausted": "исчерпан объявленный ресурс кандидата",
    "incomplete_scenario": "сценарий не предъявлен до конца",
})

#: Outcomes that leave no trustworthy verdict: the numbers may be shown but the
#: run must not be ranked. ``resource_exhausted`` is *not* here — an exhausted
#: declared budget is a model outcome and stays in the denominator; so is a
#: plain wrong answer. ``unsupported`` leaves the denominator entirely.
INCOMPLETE_FAILURES = frozenset({"infra_failure", "evaluator_failure", "integrity_violation"})

#: Kept for callers written against the three-name taxonomy.
INFRA_FAILURES = INCOMPLETE_FAILURES

#: Outcomes of a fixed-budget track: a real failure, but not an incomplete run.
MODEL_FAILURES = frozenset({"model_failure", "resource_exhausted"})


def canonical_failure(kind: str | None) -> str | None:
    """Canonical protocol kind of a stored name, or ``None``."""
    text = str(kind or "")
    if not text:
        return None
    return LEGACY_FAILURE_KINDS.get(text, text)


def failure_label(kind: str | None) -> str:
    """Human-readable label of a taxonomy kind, or an empty string."""
    return FAILURE_LABELS.get(str(kind or ""), "")


def is_infra_failure(kind: str | None) -> bool:
    """True when the measurement is incomplete (infra, evaluator, integrity)."""
    return canonical_failure(kind) in INCOMPLETE_FAILURES


def is_capability_gap(kind: str | None) -> bool:
    return canonical_failure(kind) in {"unsupported"}


def outcome_of(score: Any) -> str:
    """One protocol outcome of a score row or :class:`Score`.

    A skipped task is ``unsupported``; a stored failure is its canonical kind.
    An error without a kind is an infrastructure outcome: the run broke before a
    verdict. Otherwise the row is a capability verdict — wrong while any check
    failed, ``success`` only when all of them passed.
    """
    data = score if isinstance(score, Mapping) else score.as_dict()
    if data.get("skipped"):
        return "unsupported"
    kind = canonical_failure(data.get("failure"))
    if kind:
        return kind
    if data.get("error"):
        return "infra_failure"
    failed = any(not check.get("passed") for check in data.get("checks") or [])
    return "model_failure" if failed else "success"


def showcase_points(profile: str = DEFAULT_PROFILE) -> dict[str, float]:
    """Showcase totals of a profile; an empty declared map means no showcase."""
    return SHOWCASE_POINTS.get(profile, SHOWCASE_POINTS[DEFAULT_PROFILE])


def scored_categories(profile: str = DEFAULT_PROFILE) -> tuple[str, ...]:
    """Scored categories the profile carries, in declaration order."""
    override = PROFILE_POINTS.get(profile)
    if override is None:
        return SCORED_CATEGORIES
    return tuple(key for key in CATEGORY_POINTS if key in override)


def points_for(key: str, profile: str = DEFAULT_PROFILE) -> float | None:
    """Declared points of a category, if it has any.

    The art block's number is informational: it is drawn, not scored into the
    verdict. A profile may override the total of a category (the stretch
    showcase) or drop categories from its set entirely (``research-v1``); the
    baseline constants stay untouched.
    """
    if profile == "pilot":
        if key in PILOT_CATEGORY_POINTS:
            return PILOT_CATEGORY_POINTS[key]
        return showcase_points(profile).get(key)
    override = PROFILE_POINTS.get(profile)
    if override is not None:
        if key in override:
            return override[key]
        if key in CATEGORY_POINTS:
            return None
        return showcase_points(profile).get(key)
    if key in CATEGORY_POINTS:
        return CATEGORY_POINTS[key]
    return showcase_points(profile).get(key)


def scored_total(profile: str = DEFAULT_PROFILE) -> float:
    """Sum of the scored category totals of a profile (the showcase is extra)."""
    return sum(points_for(key, profile) or 0.0 for key in scored_categories(profile))


def profile_total(profile: str = DEFAULT_PROFILE) -> float:
    """Full denominator of a profile, independent of filters or a partial run."""
    keys = set(scored_categories(profile)) | set(showcase_points(profile))
    return sum(points_for(key, profile) or 0.0 for key in keys)


def category_title(key: str) -> str:
    return CATEGORY_TITLES.get(key) or SHOWCASE_TITLES.get(key) or key


# --------------------------------------------------------------------------- #
# Task definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Asset:
    """One file presented to the model next to the text prompt.

    The asset is *local*: ``path`` lives inside the fixture workspace and the
    bytes are carried by the fixture itself. There is no external URL, so a
    replay cannot silently fetch a different picture. PNG is the only accepted
    media type and the geometry is fixed by :data:`ASSET_SIZE`; preprocessing is
    therefore the same for every model and every run.
    """

    path: str
    media_type: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def as_dict(self) -> dict[str, str]:
        """JSON form: the filename and the exact bytes, base64-encoded."""
        return {
            "path": self.path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "data": base64.b64encode(self.data).decode("ascii"),
        }


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
    chat template plus the minimal neutral instruction from ``runner``. When
    set, the runner sends it as its own additional system message and it is
    part of the suite digest."""

    workspace: dict[str, str] | None = None
    """Files materialised before the task runs (coding repos, tool fixtures)."""

    assets: list[Asset] | None = None
    """Typed files presented with the prompt (image-input scenarios).

    Paths are relative to the fixture workspace and are materialised into it.
    The runner reads the bytes back off disk and sends them as an ``image_url``
    content part; a category V scenario without assets is a broken fixture.
    """

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
    cached_tokens: int | None = None
    """Prompt tokens the runtime served from its prefix cache, when reported."""
    ttft_s: float | None = None
    total_s: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> "TokenSplit":
        """Answer/reasoning/total of this reply, see ``split_tokens``."""
        return split_tokens(self)

    @property
    def generated_tokens(self) -> int | None:
        """Total generated tokens as the runtime counts them."""
        return self.tokens.total


#: ``TokenSplit.source`` values. The relationship between completion and
#: reasoning is guessed for ``estimated`` and unknown for ``unknown``.
REPORTED = "reported"
SEPARATE = "separate"
ESTIMATED = "estimated"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class TokenSplit:
    """One reply's generated tokens, split without counting reasoning twice.

    ``answer`` and ``reasoning`` never overlap and ``total`` is the number the
    runtime counts, so ``answer + reasoning`` may differ from ``total`` when
    the reasoning share was estimated.
    """

    answer: int | None
    reasoning: int | None
    total: int | None
    source: str = UNKNOWN


def split_tokens(reply: ModelReply) -> TokenSplit:
    """Decompose one reply into visible answer and reasoning.

    OpenAI-compatible usage already includes reasoning in ``completion_tokens``
    and reports the share separately: the answer is the subtraction (170 total
    is 100 answer + 70 reasoning, not 240). A backend that excludes reasoning is
    recognised by ``reasoning > completion`` and the two are added instead.

    Without ``reasoning_tokens`` the reasoning is estimated from its text and
    **not** subtracted: ``answer = completion`` is what the runtime measured,
    the estimate only marks the share and must not shrink a measured total.
    """
    completion = reply.completion_tokens
    if completion is None:
        return TokenSplit(None, reply.reasoning_tokens, None, UNKNOWN)
    reasoning = reply.reasoning_tokens
    if reasoning is not None and reasoning > completion:
        return TokenSplit(completion, reasoning, completion + reasoning, SEPARATE)
    if reasoning is not None:
        return TokenSplit(max(0, completion - reasoning), reasoning, completion, REPORTED)
    estimate = len(reply.reasoning or "") // 3
    return TokenSplit(completion, estimate, completion, ESTIMATED)


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
    telemetry: dict[str, Any] | None = None
    """Reduced timing/accounting aggregate, ``None`` when none was collected.

    Unknown measurements stay ``None``; the report's unknown shares read that
    absence instead of a fake zero.
    """

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

        A count that one turn did not report stays ``None`` rather than summing
        the half that arrived. ``generated_tokens`` is the runtime's own total,
        so a stand that excludes reasoning from ``completion_tokens`` cannot
        produce a second, contradictory answer here.
        """
        split = [turn.reply.tokens for turn in self.turns]
        sources = {part.source for part in split}
        if not split:
            accounting = UNKNOWN
        elif sources <= {REPORTED}:
            accounting = REPORTED
        elif sources <= {REPORTED, SEPARATE}:
            accounting = SEPARATE if SEPARATE in sources else REPORTED
        elif sources <= {ESTIMATED}:
            accounting = ESTIMATED
        elif UNKNOWN in sources:
            accounting = UNKNOWN
        else:
            accounting = "mixed"

        answers = [part.answer for part in split]
        thinkings = [part.reasoning for part in split]
        totals = [part.total for part in split]
        answer = sum(answers) if answers and None not in answers else None
        reasoning = sum(thinkings) if thinkings and None not in thinkings else None
        generated = sum(totals) if totals and None not in totals else None
        estimated = any(part.source == ESTIMATED for part in split)

        prompts = [turn.reply.prompt_tokens for turn in self.turns]
        prompt_total = sum(prompts) if prompts and None not in prompts else None
        caches = [turn.reply.cached_tokens for turn in self.turns]
        cached = sum(caches) if caches and None not in caches else None

        ttft: float | None = None
        total = 0.0
        for turn in self.turns:
            total += turn.reply.total_s or 0.0
            if ttft is None and turn.reply.ttft_s is not None:
                ttft = turn.reply.ttft_s
        return {
            "turns": len(self.turns),
            "tool_calls": len(self.tool_log),
            "prompt_tokens_last": self.prompt_tokens(),
            "prompt_tokens_total": prompt_total,
            "cached_tokens": cached,
            "cache_share": (
                round(cached / prompt_total, 4)
                if cached is not None and prompt_total
                else None
            ),
            "answer_tokens": answer,
            "reasoning_tokens": reasoning,
            "generated_tokens": generated,
            "token_accounting": accounting,
            "reasoning_estimated": estimated,
            "answer_chars": sum(len(t.reply.text or "") for t in self.turns),
            "reasoning_chars": sum(len(t.reply.reasoning or "") for t in self.turns),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "generate_s": round(total, 3) or None,
            "tokens_per_s": round(generated / total, 2)
            if generated is not None and total > 0.5
            else None,
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

    failure: str | None = None
    """Taxonomy kind of why the task produced no verdict, one of
    :data:`FAILURE_KINDS`. ``error`` names *that* something broke; this names
    *what*, so infrastructure failures can be told apart from a model that ran
    out of its declared budget. ``None`` for a wrong answer and for success;
    :func:`outcome_of` derives those two quiet outcomes from the row."""

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
        if self.error or self.failure:
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
            "failure": self.failure,
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
            failure=data.get("failure"),
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
    if isinstance(value, Asset):
        return value.as_dict()
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
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

