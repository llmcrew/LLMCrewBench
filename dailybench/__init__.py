"""LLMCrew DailyBench-16 — generated, programmatically scored local-LLM suite.

See ``LLMCrew-DailyBench-16-v0.1.md`` for the specification this package
implements.

Quick start::

    python -m dailybench selfcheck            # no GPU needed
    python -m dailybench run --base-url http://127.0.0.1:1234/v1 --model qwen3.8-27b
    python -m dailybench report runs/<run-dir>
"""

from .config import DEFAULT_SEED, RunConfig
from .core import (
    ADMISSION_CATEGORIES,
    CATEGORY_POINTS,
    CATEGORY_TITLES,
    FULL_TOTAL,
    SHORT_TOTAL,
    Check,
    ModelReply,
    Score,
    TaskInstance,
    Transcript,
    Turn,
    registered_categories,
)

__all__ = [
    "ADMISSION_CATEGORIES",
    "CATEGORY_POINTS",
    "CATEGORY_TITLES",
    "DEFAULT_SEED",
    "FULL_TOTAL",
    "SHORT_TOTAL",
    "Check",
    "ModelReply",
    "RunConfig",
    "Score",
    "TaskInstance",
    "Transcript",
    "Turn",
    "registered_categories",
]

__version__ = "0.1.0"
