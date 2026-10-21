"""LLMCrew DailyBench — generated, programmatically scored local-LLM suite.

Ten categories, 100 points, one deterministic suite per seed; every point comes
from an observable result (a parsed answer, a file, a hidden test, a tool log),
never from a judge model. See ``README.md`` for the full description.

Quick start::

    make setup                                 # .venv + pytest (uv)
    make selfcheck                             # no GPU needed
    python -m dailybench bench --config qwen3.8-27b-ud-q3-k-xl --speed
    python -m dailybench report runs/qwen3.8-27b-ud-q3-k-xl
"""

from .config import DEFAULT_SEED, RunConfig
from .core import (
    ADMISSION_CATEGORIES,
    CATEGORY_POINTS,
    CATEGORY_TITLES,
    DEFAULT_PROFILE,
    FULL_TOTAL,
    PROFILES,
    SHORT_TOTAL,
    Check,
    ModelReply,
    Score,
    TaskInstance,
    Transcript,
    Turn,
    profile_of,
    registered_categories,
)

__all__ = [
    "ADMISSION_CATEGORIES",
    "CATEGORY_POINTS",
    "CATEGORY_TITLES",
    "DEFAULT_PROFILE",
    "DEFAULT_SEED",
    "FULL_TOTAL",
    "PROFILES",
    "SHORT_TOTAL",
    "Check",
    "ModelReply",
    "RunConfig",
    "Score",
    "TaskInstance",
    "Transcript",
    "Turn",
    "profile_of",
    "registered_categories",
]

__version__ = "0.2.0"
