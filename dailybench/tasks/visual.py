"""Category K — visual simulations on Three.js, measured outside the score.

Twenty scenarios: physics and fields (K-1..K-11), cells and machines
(K-12..K-16), and visualisations (K-17..K-20). Each one asks for a working page
and is checked by *running* it in a headless Chromium (see ``dailybench/visual``
and ``dailybench/browser``).

The block is reported on its own screen and never enters the leaderboard
verdict: a browser build on the measurement machine must not be able to move a
quality score. That is why its points live in ``core.SHOWCASE_POINTS`` rather
than in ``CATEGORY_POINTS``.
"""

from __future__ import annotations

import importlib
import importlib.util
import random
from typing import Any

from .. import core, visual
from ..config import RunConfig

#: Scenario modules. Each one owns a handful of ids and renders its own
#: reference pages; a slice can be written and verified on its own.
MODULES: tuple[str, ...] = (
    "visual_projectiles",
    "visual_dynamics",
    "visual_particles",
    "visual_fields",
    "visual_visuals",
)

IDS = (
    "K-1", "K-2", "K-3", "K-4", "K-5",
    "K-6", "K-7", "K-8", "K-9", "K-10",
    "K-11", "K-12", "K-13", "K-14", "K-15",
    "K-16", "K-17", "K-18", "K-19", "K-20",
)

TITLE = "Visual simulations (Three.js)"


def _present_modules() -> list[Any]:
    found = []
    for name in MODULES:
        if importlib.util.find_spec(f"{__package__}.{name}") is None:
            continue
        found.append(importlib.import_module(f"{__package__}.{name}"))
    return found


def _expected_ids(modules: list[Any]) -> tuple[str, ...]:
    """Strict once the category is complete, tolerant while slices are in flight.

    A finished category must emit exactly ``IDS`` in exactly that order; the
    tolerant branch only exists so a half-written working tree can still run the
    rest of the suite, and it disappears by itself as soon as the last scenario
    module exists.
    """
    if len(modules) == len(MODULES):
        return IDS
    return tuple(task_id for module in modules for task_id in module.IDS)


_MODULES = _present_modules()
_EXPECTED = _expected_ids(_MODULES)


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    """Every scenario, each module drawing from the shared category stream."""
    tasks: list[core.TaskInstance] = []
    for module in _MODULES:
        tasks.extend(module.build(rng))
    ids = tuple(task.id for task in tasks)
    if ids != _EXPECTED:
        raise RuntimeError(f"категория K собрала {ids}, ожидалось {_EXPECTED}")
    if len(_MODULES) == len(MODULES):
        total = sum(task.max_points for task in tasks)
        if abs(total - core.SHOWCASE_POINTS["K"]) > 1e-9:
            raise RuntimeError(f"категория K: {total} баллов вместо {core.SHOWCASE_POINTS['K']}")
    return tasks


def evaluate(task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
    return visual.evaluate(task, transcript)


core.register(
    key="K",
    title=TITLE,
    points=core.SHOWCASE_POINTS["K"],
    generate=generate,
    evaluate=evaluate,
    expected_ids=_EXPECTED,
)
