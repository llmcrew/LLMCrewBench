"""Run configuration: inference budgets, model identity, output layout."""

from __future__ import annotations

import functools
import json
import os
import platform
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .core import DEFAULT_PROFILE, PROFILES

DEFAULT_SEED = 20260915
DEFAULT_CONTEXT_WINDOW = 262144

#: Lower bound of the per-task generation ceiling when none was chosen.
#:
#: It used to be the ceiling itself, which silently cut an enlarged answer
#: budget: ``--max-new-tokens 65536`` under the default total still sent 16384,
#: because the window branch clamped every answer to the remaining total.
DEFAULT_TOTAL_TOKENS = 16384

#: Smallest answer the harness still calls an answer. Below it the scenario is
#: over the window rather than merely tight, which is why the floor is reported.
BUDGET_FLOOR = 256

#: How the decoding conditions are fixed; see ``RunConfig.generation_mode``.
GENERATION_MODES = ("fixed", "native")


#: Human label for the provenance of the prompt size in the budget reason.
BUDGET_SOURCES = {
    "tokenizer": "точный токенизатор",
    "task-coefficient": "сохранённый коэффициент",
    "estimate": "оценка по 2.5 символа",
}


@dataclass(frozen=True)
class AnswerBudget:
    """One answer budget decision, kept in the transcript with its reason."""

    tokens: int
    policy: str
    """Which limit policy produced the number: ``window`` or ``fixed``.

    Not to be confused with the wire ``mode`` (``fixed``/``native``), which says
    whether the request pins sampling and thinking at all.
    """
    reason: str
    source: str
    """Where the prompt size came from: ``tokenizer``, ``task-coefficient``,
    ``estimate`` or a caller note."""
    floor: bool = False
    capped_by: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "tokens": self.tokens,
            "policy": self.policy,
            "reason": self.reason,
            "source": self.source,
            "floor": self.floor,
            "capped_by": self.capped_by,
        }


#: Neutral system text, identical in wording for every model. The chat template
#: itself is the runtime's own; this only fixes the shared framing.
NEUTRAL_SYSTEM = (
    "You are the model under evaluation. Follow the user's instructions exactly "
    "as written. Do not use outside knowledge unless the task explicitly allows "
    "it. Answer in the language of the task."
)


def _env(name: str) -> str:
    """Local override: endpoints and keys never live in a tracked file."""
    return os.environ.get(name, "").strip()


@dataclass
class RunConfig:
    #: Filled from the environment in :meth:`__post_init__` when left empty.
    #: Endpoints and keys must never require editing a tracked file.
    _ENV_DEFAULTS = {
        "base_url": ("DAILYBENCH_BASE_URL", "http://127.0.0.1:1234/v1"),
        "model": ("DAILYBENCH_MODEL", "local-model"),
        "api_key": ("DAILYBENCH_API_KEY", "not-needed"),
    }

    # --- endpoint ---------------------------------------------------------- #
    #: A clone points these at its own server with env vars; the tracked defaults
    #: are therefore loopback and a placeholder, never a machine path. ``None``
    #: means "read the environment", so exporting a key works even in a process
    #: that imported this module earlier.
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None

    # --- decoding budgets (identical for every model) ---------------------- #
    temperature: float = 0.0
    top_p: float = 1.0
    generation_mode: str = "fixed"
    """How the decoding conditions are fixed: ``fixed`` or ``native``.

    ``fixed`` sends the whole contract (sampling, seed, answer limit, reasoning
    budget, stop) on every request, so two models meet identical conditions and
    the request itself is the evidence. ``native`` leaves sampling and thinking
    to the server's own defaults and only bounds the answer; it is a different
    measurement and a run says which one it was.
    """
    reasoning_budget: int = 2048
    """Server-side cap on thinking tokens (``reasoning_budget``).

    Without it the stand decides: Studio forced 2048 for one client and vLLM
    left its own default, so the same declared configuration produced different
    modes. Sent in ``fixed`` mode only; a stand that rejects the field drops it
    once and the transcript records the drop.
    """
    stop_sequences: tuple[str, ...] = ()
    """Stop sequences sent with every request; empty means the template's own."""
    template_version: str = ""
    """Declared chat-template version, recorded next to the probed digest."""
    max_new_tokens: int = 4096
    """Answer budget per response, thinking included. Applies to every model."""
    #: Artefact categories (a whole file, a frame) need room for the artefact
    #: *plus* thinking: a reasoning model on a visual task spends 4-8K on thinking
    #: before it writes 5K of HTML, so the small default truncates the answer.
    max_new_tokens_by_category: dict[str, int] = field(
        default_factory=lambda: {"C": 12288, "K": 16384, "L": 16384}
    )
    """Per-category override of the answer budget.

    A category that must *produce an artefact* (a whole source file, a SQL
    statement) needs room for it beyond the thinking; categories whose answer is
    a single value do not. The budgets are identical for every model, so the
    comparison stays fair while a thinking model is not starved of the answer.
    """
    unbounded_budget: bool = True
    """Answer limit = what fits in the window (see :meth:`plan_answer`)."""

    answer_cap: int = 0
    """Explicit upper bound on one answer, ``0`` = the window is the only limit.

    ``--max-new-tokens`` sets it. It was ignored by the window branch, so the
    flag changed the config and 259834 went out instead of the requested 32: the
    free window is a *measurement*, the cap is a *choice*, and the smaller wins.
    """

    retry_errors: bool = False
    """Resume re-attempts tasks that ended in an error (transient outages)."""

    allow_suite_drift: bool = False
    """Resume across a changed suite digest (experimental, recorded)."""

    extra_body: dict = field(default_factory=dict)
    """Extra JSON merged into every request payload (server-specific flags)."""

    agent: bool = True
    """Solve the agentic categories with the ``pi`` coding agent, not a chat loop.

    The categories that produce an *artefact* (``agent_categories``: code files
    and the visual pages) are handed to a real agent in a working copy: it reads
    files, edits them and runs their tests, and the result is graded from disk.
    A bare chat endpoint is measured by the same runner, so the switch is the
    only difference between the two regimes.
    """

    agent_categories: tuple[str, ...] = ("C", "K", "L")
    """Which categories the agent owns when :attr:`agent` is on.

    Spelled out rather than "every artefact category": adding a category later
    must not silently move it to the other harness.
    """

    sandbox: bool = True
    """Run the agent in a container that mounts the task fixture and nothing else.

    The agent is a candidate, so the environment it sees is part of the
    measurement: no repository, no ``task.oracle``, no neighbouring working copy,
    an allowlisted environment and no network beyond the endpoint. When Docker or
    the image is unavailable the run falls back to the host and the transcript
    says ``host_run`` with the reason; ``sandbox=False`` asks for the host on
    purpose.
    """

    max_total_tokens: int | None = None
    """Total generation ceiling for one task, identical for every model.

    ``None`` means "nobody chose one": :meth:`__post_init__` resolves it to at
    least :data:`DEFAULT_TOTAL_TOKENS` and never below :attr:`max_new_tokens`,
    so an enlarged answer budget is not clamped back by the remaining total. A
    reference policy or an explicit ``--max-total-tokens`` sets the value and is
    kept as chosen.
    """
    max_tool_calls: int = 12
    context_window: int = DEFAULT_CONTEXT_WINDOW
    request_timeout: float = 1800.0
    request_wall_clock: float = 1800.0
    """Hard ceiling per request, identical for every model.

    A reasoning model can generate for tens of minutes on one task; without a
    ceiling a single runaway answer holds the GPU for the whole sweep. A task
    that hits the ceiling is ``resource_exhausted``: 0 points, the task stays
    in the denominator, and the run remains rankable. It is not an
    infrastructure failure and not an unverified row.
    """

    task_wall_clock: float = 3600.0
    """Hard ceiling for one whole task, retries and tool turns included.

    ``request_wall_clock`` alone bounds neither a multi-turn tool loop (every
    turn got a fresh full budget) nor the ``stream_options`` retry (a second
    request with a fresh full budget): a task could run for hours while the
    wording read as a per-request ceiling. One absolute deadline is armed when
    the task starts, an exhausted budget stops the loop as
    ``resource_exhausted`` (0 points, still in the denominator), and the later
    turns and the retry get only the remaining time.
    """

    # --- suite ------------------------------------------------------------- #
    profile: str = DEFAULT_PROFILE
    """Suite version: ``baseline-20261020``, ``pilot`` or ``stretch-v1``.

    A run carries the profile in its saved config, so a resume or a re-score
    regenerates exactly the same task list instead of mixing suite versions.
    """

    seed: int = DEFAULT_SEED
    only: tuple[str, ...] = ()
    sql_backend: str = "sqlite"  # "sqlite" | "postgres"
    system_prompt: str = NEUTRAL_SYSTEM
    task_ratios: dict[str, float] = field(default_factory=dict)
    """Per-task characters-per-token, measured for each long-context document.

    One global ratio cannot fit five different documents: a record mix that
    tokenizes at 2.2 chars/token leaves another document 20 % under its target,
    and an under-sized document is a different task than the one announced.
    """
    chars_per_token: float = 3.2
    """Sizing constant for the long-context documents.

    Calibrated at run start against the real tokenizer (``/tokenize`` or the
    usage of a real request): repetitive filler can pack 5 characters per token
    where prose packs 3.5, and a document sized by the wrong constant silently
    tests a smaller context than it claims.
    """

    # --- provenance -------------------------------------------------------- #
    quant: str = ""
    runtime: str = ""
    image_digest: str = ""
    """Digest of the sandbox runner image; part of the run identity.

    A rebuilt image is a different executor, so resume must not fold its tasks
    into a run measured under another one.
    """
    weights_sha256: str = ""
    weights_path: str = ""
    n_ctx: int = DEFAULT_CONTEXT_WINDOW
    kv_type: str = ""
    offload: str = "none"
    reference: str = ""
    """Reference-configuration id this run was launched from (see configs/reference.json)."""
    launch: str = ""
    """The declared launch *template* — for reproduction, not a record of what ran."""
    launch_executed: dict[str, Any] = field(default_factory=dict)
    """The request actually sent to the loader (method + body). Empty when the
    endpoint was already serving and nothing was launched."""
    observed: dict[str, Any] = field(default_factory=dict)
    """What the runtime reported about the loaded server (``/props``).

    A field the runtime did not report stays ``unknown``; nothing is filled in
    from the declared entry, so the report can show where the server differs.
    """
    unsupported: tuple[str, ...] = ()
    """Categories this configuration declares as not applicable.

    A category listed here is reported as *not verified* (and excluded from the
    totals) instead of being scored as a zero the model never had a chance to
    earn — for example tool calling on a model whose template has no tool
    format.
    """

    # --- output ------------------------------------------------------------ #
    out_dir: Path = field(default_factory=lambda: Path("runs") / "run")
    generation_keep: int = 3
    """How many committed generations of one run directory to keep.

    ``.generations`` holds a checkpoint of the whole run, and ``tasks.json``
    alone is hundreds of megabytes near the end of a sweep: an unbounded store
    filled the disk after three runs. Only the newest ``generation_keep``
    generations survive a commit; the current one and the one the manifest
    names are never dropped. Unchanged files of a checkpoint are hard links on
    the previous generation, so a kept checkpoint costs mostly its own
    ``tasks.json``.
    """

    def __post_init__(self) -> None:
        """Resolve local overrides: env wins, the tracked default is a fallback."""
        for field_name, (env_name, fallback) in self._ENV_DEFAULTS.items():
            if not getattr(self, field_name):
                setattr(self, field_name, _env(env_name) or fallback)
        if self.max_total_tokens is None:
            # Nobody chose a total: keep the historical floor, but never below
            # the answer budget that was asked for.
            self.max_total_tokens = max(int(self.max_new_tokens), DEFAULT_TOTAL_TOKENS)
        if self.profile not in PROFILES:
            raise ValueError(
                f"неизвестный профиль {self.profile!r}; допустимы: {', '.join(PROFILES)}"
            )
        if self.generation_mode not in GENERATION_MODES:
            raise ValueError(
                f"неизвестный generation_mode {self.generation_mode!r}; "
                f"допустимы: {', '.join(GENERATION_MODES)}"
            )
        if int(self.generation_keep) < 1:
            # Zero would delete the generation the manifest is about to name.
            raise ValueError(f"generation_keep должен быть ≥ 1, получено {self.generation_keep!r}")

    #: Space kept between prompt + answer and the declared window, so a request
    #: is never rejected for exceeding the context by a handful of tokens.
    #:
    #: The prompt is *estimated* (chars/3.2 plus tool schemas), and a runtime
    #: tokenizer counts differently — on a short Russian prompt the estimate can
    #: be off by hundreds of tokens. A 512-token reserve was not enough: the
    #: request landed exactly one token over the window and the runtime answered
    #: HTTP 400. 2048 covers the estimate error with room to spare.
    context_reserve: int = 2048

    #: Доля окна, оставляемая в запасе поверх ``context_reserve``.
    #:
    #: Оценка промпта — по символам, а токенизатор рантайма считает точнее и
    #: на длинном тексте расходится сильнее: на контексте в 200k символов
    #: промах достигал 2k токенов при фиксированном резерве. Процентный запас
    #: масштабируется вместе с промптом и закрывает эту вилку.
    context_reserve_share: float = 0.02

    def plan_answer(
        self,
        category: str,
        prompt_tokens: int | None = None,
        *,
        source: str = "estimate",
    ) -> "AnswerBudget":
        """The answer budget, with the reason it came out that way.

        One policy for both harnesses, identical for every model:

        * ``unbounded_budget`` — the window is the limit: as many tokens as fit
          after the prompt, capped by :attr:`answer_cap` when one was chosen.
          The model is not cut off by a laboratory ceiling, only by physics and
          by the explicit cap;
        * otherwise the per-category table (the old behaviour), kept so a
          published run stays comparable.

        The returned record is persisted: what was sent and why. The floor is
        reported instead of hiding an over-filled context behind a plausible
        number — a 256-token answer for a 31K prompt is a finding, not a budget.
        """
        basis = f" [вход: {BUDGET_SOURCES.get(source, source)}]"
        if not self.unbounded_budget:
            value = int(self.max_new_tokens_by_category.get(category, self.max_new_tokens))
            return AnswerBudget(
                value, "fixed", f"таблица категорий: {category} → {value}{basis}", source
            )
        window = int(self.context_window or 0)
        if window <= 0:
            value = int(self.max_new_tokens_by_category.get(category, self.max_new_tokens))
            return AnswerBudget(
                value, "fixed", f"окно не задано, взята таблица категорий{basis}", source
            )
        prompt = int(prompt_tokens or 0)
        reserve = int(self.context_reserve) + int(prompt * self.context_reserve_share)
        free = window - prompt - reserve
        cap = int(self.answer_cap or 0)
        if cap and free > cap:
            return AnswerBudget(
                cap, "window", f"max_new_tokens {cap} ниже свободного места {free}{basis}", source,
                capped_by="max_new_tokens",
            )
        if free < BUDGET_FLOOR:
            return AnswerBudget(
                BUDGET_FLOOR, "window", f"свободного места {free} < floor {BUDGET_FLOOR}{basis}",
                source, floor=True, capped_by="floor",
            )
        return AnswerBudget(
            free, "window", f"окно {window} − промпт {prompt} − резерв {reserve}{basis}", source,
            capped_by="window",
        )

    def budget_for(self, category: str, prompt_tokens: int | None = None) -> int:
        """The number alone, for callers that do not persist the record."""
        return self.plan_answer(category, prompt_tokens).tokens

    def to_json(self) -> str:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        # The key itself is local; a published run only says whether one was
        # used. ``from_dict`` still accepts the old field and env resolution
        # fills it again, so a saved config remains replayable.
        data.pop("api_key", None)
        data["api_key_set"] = bool(self.api_key and self.api_key != "not-needed")
        return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "out_dir" in data:
            data["out_dir"] = Path(data["out_dir"])
        if "only" in data and data["only"] is not None:
            data["only"] = tuple(data["only"])
        return cls(**data)


# --------------------------------------------------------------------------- #
# Toolchain provenance
# --------------------------------------------------------------------------- #


def _version_line(argv: list[str]) -> str:
    """First line of ``argv --version``, or an empty string when unavailable.

    The evaluators run on the host toolchain (Go, node, Chromium, SQLite), so
    that toolchain is part of the grading contract: the same hidden test on
    another runtime is another measurement. A missing binary is recorded as an
    empty string rather than failing the digest.
    """
    executable = shutil.which(argv[0])
    if not executable:
        return ""
    try:
        proc = subprocess.run(
            [executable, *argv[1:]], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (proc.stdout or proc.stderr or "").strip()
    return text.splitlines()[0].strip() if text else ""


def _chromium_version() -> str:
    """Build string of the Chromium the visual block actually launches."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            executable = Path(pw.chromium.executable_path)
    except Exception:  # pragma: no cover - depends on the machine
        return ""
    if not executable.is_file():
        return ""
    return _version_line([str(executable), "--version"])


@functools.lru_cache(maxsize=1)
def toolchain_identity() -> dict[str, str]:
    """Versions of every runtime an evaluator executes.

    Cached: the values are process constants and probing Chromium starts a
    driver. Callers treat an empty value as "not available here", the same rule
    the evaluators use for a missing toolchain.
    """
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "go": _version_line(["go", "version"]),
        "node": _version_line(["node", "--version"]),
        "sqlite": sqlite3.sqlite_version,
        "chromium": _chromium_version(),
    }

