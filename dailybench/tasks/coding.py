"""Category C — coding, 25 points across three independent scenarios.

``C-1`` Python (10 hidden checks), ``C-2`` Go (10 hidden checks) and ``C-3`` SQL
(5 generated databases). Scoring is behavioural: the hidden tests and the real
result sets decide the points, never the shape of the patch or a judge model.
"""

from __future__ import annotations

import random
from typing import Any

from .. import core
from . import coding_go, coding_python, coding_sql

IDS = ("C-1", "C-2", "C-3")

_HELPERS = {
    "C-1": coding_python,
    "C-2": coding_go,
    "C-3": coding_sql,
}


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    return [_HELPERS[scenario].generate(rng, cfg) for scenario in IDS]


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    return _HELPERS[task.id].evaluate(task, tr)


core.register(
    key="C",
    title="Coding",
    points=25.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
)
