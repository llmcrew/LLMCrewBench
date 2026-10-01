"""Tests for the K-17..K-20 slice — visualisations (bar chart, orrery, IK arm,
spectrum).

The slice test renders its own reference pages directly through
``visual.evaluate`` instead of going through the category registry, so it can be
run while neighbouring slices are still being written.
"""

from __future__ import annotations

import math
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_visuals as module

#: One fixed seed for the "does it pass" half, a second one for the "does the
#: data actually move with the seed" half.
SEED = 20240930
OTHER_SEED = SEED + 7


def tasks(seed: int = SEED) -> list[core.TaskInstance]:
    return module.build(random.Random(seed))


def by_id(seed: int = SEED) -> dict[str, core.TaskInstance]:
    return {task.id: task for task in tasks(seed)}


def transcript_for(text: str, task_id: str = "K-17") -> core.Transcript:
    return core.Transcript(
        task_id=task_id,
        category="K",
        turns=[core.Turn(messages=[], reply=core.ModelReply(text=text))],
    )


@pytest.fixture(scope="module", autouse=True)
def browser_available() -> None:
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")


# --------------------------------------------------------------------------- #
# Shape of the slice
# --------------------------------------------------------------------------- #


def test_ids_and_points() -> None:
    built = tasks()
    assert tuple(task.id for task in built) == module.IDS
    assert module.IDS == ("K-17", "K-18", "K-19", "K-20")
    for task in built:
        assert task.category == "K"
        assert task.max_points == 4.0
        spec = visual.spec_of(task)
        # Every scenario must be able to fail: no empty check groups.
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps > 0 and spec.dt > 0
        assert spec.need_same_frame


def test_check_groups_sum_to_max_points() -> None:
    for task in tasks():
        spec = visual.spec_of(task)
        total = len(visual.CHECK_NAMES) * visual.POINTS_PER_CHECK
        assert abs(total - task.max_points) < 1e-9
        assert len(spec.scene) + len(spec.simulation) + len(spec.image) >= len(visual.CHECK_NAMES)


def test_golden_pages_fit_the_answer_budget() -> None:
    for task in tasks():
        lines = task.oracle["golden_html"].count("\n") + 1
        assert lines <= 150, f"{task.id}: эталон на {lines} строк — не влезет в бюджет ответа"
        assert "./vendor/three.module.js" in task.oracle["golden_html"]
        assert "preserveDrawingBuffer: true" in task.oracle["golden_html"]
        assert "setSize(800, 600, false)" in task.oracle["golden_html"]
        assert "setPixelRatio(1)" in task.oracle["golden_html"]
        assert "Math.random" not in task.oracle["golden_html"]
        assert "performance.now" not in task.oracle["golden_html"]
        assert "__ready = true" in task.oracle["golden_html"]


# --------------------------------------------------------------------------- #
# Determinism, and what the seed actually moves
# --------------------------------------------------------------------------- #


def test_build_is_deterministic() -> None:
    first, second = tasks(), tasks()
    assert [t.prompt for t in first] == [t.prompt for t in second]
    assert [t.oracle["golden_html"] for t in first] == [t.oracle["golden_html"] for t in second]
    assert [t.oracle["meta"] for t in first] == [t.oracle["meta"] for t in second]


def test_seed_changes_the_fixture() -> None:
    first, other = by_id(), by_id(OTHER_SEED)
    for task_id, task in first.items():
        twin = other[task_id]
        assert task.prompt != twin.prompt, f"{task_id}: смена seed не изменила данные"
        assert task.oracle["meta"] != twin.oracle["meta"]
        assert task.oracle["golden_html"] != twin.oracle["golden_html"]

    # ... and the moving part is exactly the data the check reads back.
    k17_a, k17_b = first["K-17"].oracle["meta"], other["K-17"].oracle["meta"]
    assert k17_a["values"] != k17_b["values"]
    k18_a, k18_b = first["K-18"].oracle["meta"], other["K-18"].oracle["meta"]
    assert [p["period"] for p in k18_a["planets"]] != [p["period"] for p in k18_b["planets"]]
    assert first["K-19"].oracle["meta"]["target"] != other["K-19"].oracle["meta"]["target"]
    k20_a, k20_b = first["K-20"].oracle["meta"], other["K-20"].oracle["meta"]
    assert (k20_a["k1"], k20_a["k2"], k20_a["a1"], k20_a["a2"]) != (
        k20_b["k1"],
        k20_b["k2"],
        k20_b["a1"],
        k20_b["a2"],
    )


# --------------------------------------------------------------------------- #
# The fixture assertions, checked from the outside
# --------------------------------------------------------------------------- #


def test_in_frame_rejects_points_outside_the_view() -> None:
    """The visibility proof used by ``build()`` really does reject what it should."""
    camera, look = module.K17_CAMERA, module.K17_LOOK
    assert module.in_frame((0.0, 3.0, 0.0), camera, look)
    assert not module.in_frame((0.0, 300.0, 0.0), camera, look), "точка высоко над кадром"
    assert not module.in_frame((0.0, 3.0, 400.0), camera, look), "точка за камерой"
    assert not module.in_frame((50.0, 3.0, 0.0), camera, look), "точка далеко в стороне"
    assert not module.in_frame((0.0, 3.0, 17.0), camera, look), "точка за спиной камеры"


def test_k17_fixture_is_sound() -> None:
    meta = by_id()["K-17"].oracle["meta"]
    values, labels = meta["values"], meta["labels"]
    assert len(values) == len(labels) and 5 <= len(values) <= 7
    assert len(set(values)) == len(values), "высоты обязаны различаться"
    assert all(value > 0 for value in values)
    assert meta["steps"] * meta["dt"] >= module.K17_GROW + 0.1, "анимация роста должна успеть закончиться"
    assert meta["labels"] == sorted(labels)


def test_k18_fixture_is_sound() -> None:
    meta = by_id()["K-18"].oracle["meta"]
    planets = meta["planets"]
    assert len(planets) == 3
    assert len({planet["radius"] for planet in planets}) == 3
    assert len({planet["period"] for planet in planets}) == 3
    expected = module._orrery(planets, meta["steps"], meta["dt"])
    assert len({theta for theta, _, _ in expected}) == 3
    for (theta, x, z), planet in zip(expected, planets):
        radius = planet["radius"]
        assert abs(math.hypot(x, z) - radius) < 1e-3, "планета остаётся на своей орбите"
        assert abs(theta - planet["theta0"]) > 1.0, "угол обязан сдвинуться от начальной фазы"


def test_k19_fixture_is_sound() -> None:
    meta = by_id()["K-19"].oracle["meta"]
    l1, l2 = meta["l1"], meta["l2"]
    x, y = meta["target"]
    theta1, theta2, ex, ey = module._ik(l1, l2, x, y)
    assert abs(ex - x) < 1e-9 and abs(ey - y) < 1e-9, "аналитика попадает в цель"
    assert theta2 < 0, "локоть вверх: theta2 отрицателен"
    assert abs(theta2) > 0.2 and abs(theta1) > 0.05
    # The naive answer — the arm still stretched along +x, IK never solved —
    # must miss the target, otherwise the check does not discriminate.
    assert math.hypot((l1 + l2) - x, y) > 0.5


def test_k20_fixture_is_sound() -> None:
    meta = by_id()["K-20"].oracle["meta"]
    n, rate = module.K20_SAMPLES, module.K20_RATE
    k1, k2, a1, a2 = meta["k1"], meta["k2"], meta["a1"], meta["a2"]
    bins = meta["bins"]
    samples = module._signal(n, a1, k1, a2, k2)
    assert abs(module._dft_amplitude(samples, k1) - a1) < 1e-9
    assert abs(module._dft_amplitude(samples, k2) - a2) < 1e-9
    assert len(set(bins)) == 3 and k1 in bins and k2 in bins
    extra = [k for k in bins if k not in (k1, k2)]
    assert len(extra) == 1
    assert abs(module._dft_amplitude(samples, extra[0])) < 1e-9, "на лишнем бине тона нет"
    amplitudes = [module._dft_amplitude(samples, k) for k in bins]
    dominant = bins[amplitudes.index(max(amplitudes))]
    assert dominant == (k1 if a1 > a2 else k2), "доминирующая частота определяется данными"
    assert abs(a1 - a2) > 0.5
    assert abs(dominant - dominant * rate / n) < 1e-9  # бины стоят ровно на целых герцах


# --------------------------------------------------------------------------- #
# The real thing: run the reference pages in the browser
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [SEED, OTHER_SEED])
def test_golden_scores_full(seed: int) -> None:
    for task in tasks(seed):
        score = visual.evaluate(task, transcript_for(task.oracle["reference"]))
        failed = [f"{check.name}: {check.detail}" for check in score.checks if not check.passed]
        assert score.earned == task.max_points, f"{task.id} (seed {seed}) — {failed}"
        assert score.possible == task.max_points


@pytest.mark.parametrize("seed", [SEED, OTHER_SEED])
def test_garbage_scores_zero(seed: int) -> None:
    for task in tasks(seed):
        score = visual.evaluate(task, transcript_for("Не знаю, как это сделать, извините."))
        assert score.earned == 0, f"{task.id}: мусорный ответ набрал {score.earned}"
        assert score.possible == task.max_points


def test_frames_are_reproducible_across_seeds() -> None:
    """A different seed means different data — the picture must still be one
    and the same on a repeated load (the ``image`` check requires it)."""
    for seed in (SEED, OTHER_SEED):
        for task in tasks(seed):
            result, _ = visual.measure(task, transcript_for(task.oracle["reference"]))
            assert result is not None and result.ok, f"{task.id}: страница не отрисовалась"
            assert result.same_frame, f"{task.id}: кадр не воспроизводится"
            assert result.shot_sha and result.shot_sha == result.shot_sha_again
