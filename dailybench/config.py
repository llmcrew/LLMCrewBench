"""Run configuration: inference budgets, model identity, output layout."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_SEED = 20260915

#: Neutral system text, identical in wording for every model. The chat template
#: itself is the runtime's own; this only fixes the shared framing.
NEUTRAL_SYSTEM = (
    "You are the model under evaluation. Follow the user's instructions exactly "
    "as written. Do not use outside knowledge unless the task explicitly allows "
    "it. Answer in the language of the task."
)


@dataclass
class RunConfig:
    # --- endpoint ---------------------------------------------------------- #
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "local-model"
    api_key: str = os.environ.get("DAILYBENCH_API_KEY", "not-needed")

    # --- decoding budgets (identical for every model) ---------------------- #
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 4096
    """Answer budget per response, thinking included. Applies to every model."""
    max_new_tokens_by_category: dict[str, int] = field(default_factory=lambda: {"C": 8192})
    """Per-category override of the answer budget.

    A category that must *produce an artefact* (a whole source file, a SQL
    statement) needs room for it beyond the thinking; categories whose answer is
    a single value do not. The budgets are identical for every model, so the
    comparison stays fair while a thinking model is not starved of the answer.
    """
    max_total_tokens: int = 16384
    max_tool_calls: int = 12
    context_window: int = 32768
    request_timeout: float = 1800.0
    request_wall_clock: float = 600.0
    """Hard ceiling per request, identical for every model.

    A reasoning model can generate for tens of minutes on one task; without a
    ceiling a single runaway answer holds the GPU for the whole sweep. A task
    that hits the ceiling is reported as unverified, never as a wrong answer.
    """

    # --- suite ------------------------------------------------------------- #
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
    weights_sha256: str = ""
    weights_path: str = ""
    n_ctx: int = 32768
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

    def budget_for(self, category: str) -> int:
        """Answer budget for one category (identical for every model)."""
        return int(self.max_new_tokens_by_category.get(category, self.max_new_tokens))

    def to_json(self) -> str:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "out_dir" in data:
            data["out_dir"] = Path(data["out_dir"])
        if "only" in data and data["only"] is not None:
            data["only"] = tuple(data["only"])
        return cls(**data)
