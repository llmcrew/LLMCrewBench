"""Category K — visual simulations on Three.js, measured outside the score.

Thirty scenarios: physics and fields (K-1..K-11), cells and machines
(K-12..K-16), visualisations (K-17..K-20), and the extended slices (K-21..K-30:
gear kinematics, fluids, circuits, a queue, a lift and structures). Each one asks
for a working page and is checked by *running* it in a headless Chromium (see
``dailybench/visual`` and ``dailybench/browser``).

The block is reported on its own screen and never enters the leaderboard
verdict: a browser build on the measurement machine must not be able to move a
quality score. That is why its points live in ``core.SHOWCASE_POINTS`` rather
than in ``CATEGORY_POINTS``.

Profiles pick the scenario set, never the machinery: the baseline runs the
original twenty, ``pilot`` runs the three host scenarios of the pilot slices
(K-21, K-23, K-25), and ``stretch-v1`` runs all thirty. Every profile is checked
against its exact id list and against ``4 * len(ids)`` points, so a slice that
forgets to register or filters wrongly fails loudly instead of shrinking the
block.
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
    "visual_kinematics",
    "visual_fluids",
    "visual_circuits",
    "visual_queue",
    "visual_lift",
    "visual_structures",
)

BASELINE_IDS = (
    "K-1", "K-2", "K-3", "K-4", "K-5",
    "K-6", "K-7", "K-8", "K-9", "K-10",
    "K-11", "K-12", "K-13", "K-14", "K-15",
    "K-16", "K-17", "K-18", "K-19", "K-20",
)

#: One host scenario per pilot slice — the cheapest proof that the extended
#: machinery works before the whole slice joins the stretch suite.
PILOT_IDS = ("K-21", "K-23", "K-25")

ALL_IDS = BASELINE_IDS + (
    "K-21", "K-22", "K-23", "K-24", "K-25",
    "K-26", "K-27", "K-28", "K-29", "K-30",
)

IDS = BASELINE_IDS

PROFILE_IDS: dict[str, tuple[str, ...]] = {
    "pilot": PILOT_IDS,
    "stretch-v1": ALL_IDS,
}

TITLE = "Visual simulations (Three.js)"

_ORDER = {task_id: index for index, task_id in enumerate(ALL_IDS)}


def _present_modules() -> list[Any]:
    found = []
    for name in MODULES:
        if importlib.util.find_spec(f"{__package__}.{name}") is None:
            continue
        found.append(importlib.import_module(f"{__package__}.{name}"))
    return found


_MODULES = _present_modules()


def _profile_ids(profile: str) -> tuple[str, ...]:
    return PROFILE_IDS.get(profile, BASELINE_IDS)


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    """The scenarios of the run's profile, each module drawing from the stream.

    Modules outside the profile are not built at all, so a pilot run pays for
    three scenarios rather than thirty. The result is re-ordered to the
    declaration order of ``ALL_IDS`` and must match the profile exactly.
    """
    profile = core.profile_of(cfg)
    wanted = _profile_ids(profile)
    wanted_set = set(wanted)
    tasks: list[core.TaskInstance] = []
    for module in _MODULES:
        if not wanted_set.intersection(module.IDS):
            continue
        tasks.extend(task for task in module.build(rng) if task.id in wanted_set)
    tasks.sort(key=lambda task: _ORDER[task.id])
    ids = tuple(task.id for task in tasks)
    # A half-written working tree may still run the rest of the suite; once the
    # last module exists the exact list is the only accepted answer.
    if len(_MODULES) == len(MODULES):
        if ids != wanted:
            raise RuntimeError(f"категория K собрала {ids}, ожидалось {wanted} в профиле {profile}")
        total = sum(task.max_points for task in tasks)
        if abs(total - 4 * len(wanted)) > 1e-9:
            raise RuntimeError(f"категория K: {total} баллов вместо {4 * len(wanted)}")
    elif ids != tuple(task_id for task_id in wanted if task_id in set(ids)):
        raise RuntimeError(f"категория K собрала {ids}, ожидалось {wanted} в профиле {profile}")
    return tasks


def evaluate(task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
    return visual.evaluate(task, transcript)


core.register(
    key="K",
    title=TITLE,
    points=core.points_for("K"),
    generate=generate,
    evaluate=evaluate,
    expected_ids=BASELINE_IDS,
    expected_ids_by_profile={"pilot": PILOT_IDS, "stretch-v1": ALL_IDS},
)
