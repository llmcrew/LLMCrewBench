"""Category L — art visualisations on Three.js, judged by people.

Sixteen scenarios about *pictures* rather than physics: a sunset, a neon grid, a
mandala, a city, a galaxy, fireworks, a lava lamp, liquid metal, an aurora, a
fractal, a tunnel, fog, a cloth, glass, a flow field.

Each one declares a palette (exact colours and their minimum share of the frame),
a composition (horizon, radial falloff, symmetry, tilt) and a motion, and is
scored from the rendered frame in five checks of one point: ``html``,
``palette``, ``composition``, ``detail``, ``motion``. See ``docs/art-contract.md``
for the authoring rules and :mod:`dailybench.art` for the machinery.

Unlike the simulation block, the art block does not enter the leaderboard at all:
its checks are descriptions, and the scenes are judged by people from the frames.
A complete run still has to answer every scene.
"""

from __future__ import annotations

import importlib
import importlib.util
import random
from typing import Any

from .. import art, core
from ..config import RunConfig

MODULES: tuple[str, ...] = (
    "art_sunset",
    "art_geometry",
    "art_particles",
    "art_shaders",
    "art_fields",
    "art_lofi",
    "art_space",
)

IDS = (
    "L-1", "L-2", "L-3", "L-4", "L-5",
    "L-6", "L-7", "L-8", "L-9", "L-10",
    "L-11", "L-12", "L-13", "L-14", "L-15",
    "L-16", "L-17", "L-18", "L-19", "L-20",
    "L-21", "L-22", "L-23", "L-24",
    "L-25", "L-26", "L-27", "L-28", "L-29",
)

TITLE = "Art visualisations (Three.js)"


def _present_modules() -> list[Any]:
    found = []
    for name in MODULES:
        if importlib.util.find_spec(f"{__package__}.{name}") is None:
            continue
        found.append(importlib.import_module(f"{__package__}.{name}"))
    return found


def _expected_ids(modules: list[Any]) -> tuple[str, ...]:
    """Strict once complete, tolerant while slices are still being written."""
    if len(modules) == len(MODULES):
        return IDS
    return tuple(task_id for module in modules for task_id in module.IDS)


_MODULES = _present_modules()
_EXPECTED = _expected_ids(_MODULES)


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    tasks: list[core.TaskInstance] = []
    for module in _MODULES:
        tasks.extend(module.build(rng))
    ids = tuple(task.id for task in tasks)
    if ids != _EXPECTED:
        raise RuntimeError(f"категория L собрала {ids}, ожидалось {_EXPECTED}")
    # Сцены не балльные, но у каждой есть столько же проверок, сколько у других:
    # сумма остаётся ориентиром для отчёта, а её полнота ловит недописанный слайс.
    if len(_MODULES) == len(MODULES):
        total = sum(task.max_points for task in tasks)
        expected = art.POINTS_PER_CHECK * len(art.CHECK_NAMES) * len(IDS)
        if abs(total - expected) > 1e-9:
            raise RuntimeError(f"категория L: {total} проверок вместо {expected}")
    return tasks


def evaluate(task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
    return art.evaluate(task, transcript)


core.register(
    key="L",
    title=TITLE,
    points=art.POINTS_PER_CHECK * len(art.CHECK_NAMES) * len(IDS),
    generate=generate,
    evaluate=evaluate,
    expected_ids=_EXPECTED,
)
