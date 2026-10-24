"""Run configuration: inference budgets, model identity, output layout."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .core import DEFAULT_PROFILE, PROFILES

DEFAULT_SEED = 20260915
DEFAULT_CONTEXT_WINDOW = 262144

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
    """Answer limit = what fits in the window (see :meth:`budget_for`)."""

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

    max_total_tokens: int = 16384
    max_tool_calls: int = 12
    context_window: int = DEFAULT_CONTEXT_WINDOW
    request_timeout: float = 1800.0
    request_wall_clock: float = 600.0
    """Hard ceiling per request, identical for every model.

    A reasoning model can generate for tens of minutes on one task; without a
    ceiling a single runaway answer holds the GPU for the whole sweep. A task
    that hits the ceiling is reported as unverified, never as a wrong answer.
    """

    task_wall_clock: float = 3600.0
    """Hard ceiling for one whole task, retries and tool turns included.

    ``request_wall_clock`` alone bounds neither a multi-turn tool loop (every
    turn got a fresh full budget) nor the ``stream_options`` retry (a second
    request with a fresh full budget): a task could run for hours while the
    wording read as a per-request ceiling. One absolute deadline is armed when
    the task starts, an exhausted budget stops the loop, and the later turns
    and the retry get only the remaining time.
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
    """The exact server launch line, echoed into the report as provenance."""
    unsupported: tuple[str, ...] = ()
    """Categories this configuration declares as not applicable.

    A category listed here is reported as *not verified* (and excluded from the
    totals) instead of being scored as a zero the model never had a chance to
    earn — for example tool calling on a model whose template has no tool
    format.
    """

    # --- output ------------------------------------------------------------ #
    out_dir: Path = field(default_factory=lambda: Path("runs") / "run")

    def __post_init__(self) -> None:
        """Resolve local overrides: env wins, the tracked default is a fallback."""
        for field_name, (env_name, fallback) in self._ENV_DEFAULTS.items():
            if not getattr(self, field_name):
                setattr(self, field_name, _env(env_name) or fallback)
        if self.profile not in PROFILES:
            raise ValueError(
                f"неизвестный профиль {self.profile!r}; допустимы: {', '.join(PROFILES)}"
            )

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

    def budget_for(self, category: str, prompt_tokens: int | None = None) -> int:
        """Answer budget for one task.

        Two regimes, identical for every model:

        * ``unbounded_budget`` — the window itself is the limit: as many tokens
          as fit after the prompt. The model is not cut off by a laboratory
          ceiling, only by physics, and what it spent is documented instead of
          being silently truncated;
        * otherwise, the per-category table (the old behaviour), kept so a
          published run stays comparable.
        """
        if not self.unbounded_budget:
            return int(self.max_new_tokens_by_category.get(category, self.max_new_tokens))
        window = int(self.context_window or 0)
        if window <= 0:
            return int(self.max_new_tokens_by_category.get(category, self.max_new_tokens))
        prompt = int(prompt_tokens or 0)
        reserve = int(self.context_reserve) + int(prompt * self.context_reserve_share)
        return max(256, window - prompt - reserve)

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
