"""Category C — coding.

Baseline: ``C-1`` Python (10 hidden checks), ``C-2`` Go (10 hidden checks) and
``C-3`` SQL (5 generated databases). The pilot adds ``C-4`` (generational async
cache) and ``stretch-v1`` adds ``C-5`` (bounded Go pool with acknowledged stop)
and ``C-6`` (SQL order-journal snapshot). Scoring is behavioural: hidden tests
and real result sets decide the points, never the shape of the patch or a judge
model.
"""

from __future__ import annotations

import random
from typing import Any

from .. import core
from . import coding_go, coding_python, coding_sql

BASELINE_IDS = ("C-1", "C-2", "C-3")
PILOT_IDS = ("C-4",)
STRETCH_IDS = BASELINE_IDS + ("C-4", "C-5", "C-6")

IDS = BASELINE_IDS

PRIVATE_IDS = frozenset({"C-1", "C-2"})
"""Private layer: the two baseline scenarios whose prompt never ships."""

_SCENARIOS = {
    "C-1": (coding_python.generate, coding_python.evaluate),
    "C-2": (coding_go.generate, coding_go.evaluate),
    "C-3": (coding_sql.generate, coding_sql.evaluate),
    "C-4": (coding_python.generate_c4, coding_python.evaluate_c4),
    "C-5": (coding_go.generate_c5, coding_go.evaluate_c5),
    "C-6": (coding_sql.generate_c6, coding_sql.evaluate_c6),
}


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    profile = core.profile_of(cfg)
    if profile == "pilot":
        scenarios = PILOT_IDS
    elif profile == "stretch-v1":
        scenarios = STRETCH_IDS
    else:
        scenarios = BASELINE_IDS
    tasks = [_SCENARIOS[scenario][0](rng, cfg) for scenario in scenarios]
    for task in tasks:
        task.private = task.id in PRIVATE_IDS
    return tasks


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    return _SCENARIOS[task.id][1](task, tr)


core.register(
    key="C",
    title="Coding",
    points=20.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=BASELINE_IDS,
    expected_ids_by_profile={"pilot": PILOT_IDS, "stretch-v1": STRETCH_IDS},
)
