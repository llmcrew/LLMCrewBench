"""Tests for the K-35..K-36 slice: quadratic drag and a Verlet chain.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, points, determinism, and fixtures that move
  with the seed;
* every reference page scores 4/4 in the real browser, on more than one seed,
  which also re-runs the ``image`` predicates against the measured frame;
* the checks are not trivially satisfiable — the probe taken *before* the steps
  fails the ``simulation`` predicates, and the named mutants (linear drag, no
  drag, no constraint relaxation) fail them too.

The drag fixture is additionally held against its analytic limit: the speed
must approach ``sqrt(g/c)`` from below and never pass it, while a free fall over
the same steps runs far past it.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from dailybench import browser, core, visual
from dailybench.config import RunConfig
from dailybench.suite import build_suite
from dailybench.tasks import visual_mech2 as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them,
#: not only for the one the suite happens to use.
SEEDS = (7, 2, 13, 0, 5)

#: The seed the profile suite is built with: the fixture must also hold on a
#: stream nobody picked by hand.
PROFILE_SEED = 20260915

#: Meta keys holding fixture inputs the answer has to hard-code. Everything
#: else in the meta is either derived by the scheme or added by ``build_task``.
FIXTURE_KEYS = {"c", "vx0", "vy0", "x0", "y0", "length", "damp"}


def _tasks(seed: int) -> list[core.TaskInstance]:
    return mod.build(random.Random(seed))


def _task(seed: int, task_id: str) -> core.TaskInstance:
    return next(task for task in _tasks(seed) if task.id == task_id)


def _transcript(task: core.TaskInstance, text: str) -> core.Transcript:
    return core.Transcript(
        task_id=task.id,
        category=task.category,
        turns=[core.Turn(messages=[], reply=core.ModelReply(text=text))],
    )


def _needs_browser() -> None:
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")


def _numbers(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    return []


def _probe(task: core.TaskInstance) -> dict:
    """A probe payload the fixture itself would return after the steps."""
    meta = task.oracle["meta"]
    if task.id == "K-35":
        x, y, vx, vy = meta["expected"]
        return {
            "steps": meta["steps"],
            "t": round(meta["steps"] * mod.DT, 4),
            "objects": ["ground", "ledge", "shell"],
            "camera": list(mod.K35_CAMERA),
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "speed": round(math.hypot(vx, vy), 4),
            "terminal": meta["terminal"],
            "c": meta["c"],
        }
    lengths = meta["lengths"]
    return {
        "steps": meta["steps"],
        "t": round(meta["steps"] * mod.DT, 4),
        "objects": ["ground", "anchor"]
        + [f"node-{i}" for i in range(7)]
        + [f"link-{i}" for i in range(6)],
        "camera": list(mod.K36_CAMERA),
        "nodes": meta["expected"],
        "lengths": lengths,
        "residual": meta["residual"],
        "len": meta["length"],
        "iterations": mod.K36_ITERATIONS,
    }


def _mutants(task: core.TaskInstance) -> list[tuple[str, dict]]:
    """The named mutants, each as a probe built from the same fixture."""
    meta = task.oracle["meta"]
    good = _probe(task)
    out: list[tuple[str, dict]] = []
    if task.id == "K-35":
        x, y, vx, vy = meta["linear"]
        out.append(
            (
                "линейное сопротивление",
                {
                    **good,
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "speed": round(math.hypot(vx, vy), 4),
                },
            )
        )
        x, y, vx, vy = meta["free"]
        out.append(
            (
                "без сопротивления",
                {
                    **good,
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "speed": round(math.hypot(vx, vy), 4),
                },
            )
        )
        return out

    nodes = meta["loose"]
    lengths = [
        round(math.hypot(nodes[i + 1][0] - nodes[i][0], nodes[i + 1][1] - nodes[i][1]), 4)
        for i in range(len(nodes) - 1)
    ]
    out.append(
        (
            "без релаксации",
            {**good, "nodes": nodes, "lengths": lengths, "residual": meta["loose_residual"]},
        )
    )
    return out


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-35", "K-36")
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "K"
        assert task.max_points == visual.POINTS_PER_CHECK * len(visual.CHECK_NAMES)
        spec = visual.spec_of(task)
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps > 0 and spec.dt > 0
        assert spec.need_same_frame
        assert "window.__step" in task.prompt and "window.__probe()" in task.prompt
        assert task.prompt.endswith(visual.HTML_CONTRACT)
        assert task.oracle["golden_html"].startswith("<!DOCTYPE html>")
        assert "./vendor/three.module.js" in task.oracle["golden_html"]


def test_build_is_deterministic() -> None:
    first = _tasks(11)
    second = _tasks(11)
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle["golden_html"] for task in first] == [
        task.oracle["golden_html"] for task in second
    ]
    assert [task.oracle["meta"] for task in first] == [
        task.oracle["meta"] for task in second
    ]


def test_fixtures_move_with_the_seed() -> None:
    for index in range(len(mod.IDS)):
        drawn = {json.dumps(_tasks(seed)[index].oracle["meta"], sort_keys=True) for seed in range(6)}
        assert len(drawn) == 6, f"{mod.IDS[index]}: фикстуры не зависят от seed"


def test_every_fixture_number_is_in_the_prompt() -> None:
    """A scenario whose constants are hidden cannot be answered."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            for key, value in meta.items():
                if key not in FIXTURE_KEYS:
                    continue
                for number in _numbers(value):
                    assert str(number) in task.prompt, (
                        f"{task.id} seed={seed}: числа {number} ({key}) нет в промпте"
                    )


def test_the_published_state_is_the_fixture_itself() -> None:
    """The probe built from the meta is what the reference page returns."""
    for seed in SEEDS:
        for task in _tasks(seed):
            spec = visual.spec_of(task)
            passed, detail = visual._predicates_report(spec.simulation, _probe(task))
            assert passed, f"{task.id} seed={seed}: probe из фикстуры не проходит ({detail})"


def test_golden_page_structure() -> None:
    """The reference answers stay inside the answer budget and are deterministic."""
    for seed in SEEDS:
        for task in _tasks(seed):
            html = task.oracle["golden_html"]
            assert len(html.splitlines()) <= 150, f"{task.id}: эталон длиннее бюджета"
            assert "Math.random" not in html
            assert "performance.now" not in html
            assert "preserveDrawingBuffer: true" in html
            assert "setSize(800, 600, false)" in html
            assert "setPixelRatio(1)" in html
            assert "window.__ready = true" in html


def test_garbage_answer_scores_zero() -> None:
    for task in _tasks(7):
        score = visual.evaluate(task, _transcript(task, "не знаю"))
        assert score.earned == 0, f"{task.id}: мусорный ответ набрал {score.earned}"


def test_profile_seed_keeps_the_fixture_in_frame() -> None:
    """The scenario must stand on the profile stream, not only on the dev seeds."""
    tasks = [
        task
        for task in build_suite(RunConfig(profile="stretch-v1", seed=PROFILE_SEED))
        if task.id in mod.IDS
    ]
    assert [task.id for task in tasks] == list(mod.IDS)
    task = tasks[0]
    meta = task.oracle["meta"]
    for x, y in mod._drag_path(meta["c"], meta["vx0"], meta["steps"]):
        assert mod._in_frame(
            (x, y, 0.0), mod.K35_RADIUS, mod.K35_CAMERA, mod.K35_LOOK
        ), f"снаряд вышел за кадр в ({x:.2f}, {y:.2f})"


def test_named_mutants_are_rejected() -> None:
    """Linear drag, no drag and a chain without relaxation must all fail."""
    for seed in SEEDS:
        for task in _tasks(seed):
            spec = visual.spec_of(task)
            for label, bad in _mutants(task):
                passed, detail = visual._predicates_report(spec.simulation, bad)
                assert not passed, f"{task.id} seed={seed}: мутант «{label}» прошёл ({detail})"


@pytest.mark.parametrize("seed", SEEDS)
def test_speed_approaches_the_terminal_limit(seed: int) -> None:
    """Every fixture's quadratic replay heads for ``sqrt(g/c)`` and never passes it."""
    task = _task(seed, "K-35")
    meta = task.oracle["meta"]
    c = meta["c"]
    terminal = math.sqrt(mod.GRAVITY / c)
    assert abs(meta["terminal"] - round(terminal, 4)) < 1e-3

    vx, vy = meta["vx0"], meta["vy0"]
    peak = 0.0
    for _ in range(600):
        speed = math.hypot(vx, vy)
        peak = max(peak, speed)
        vx += -c * speed * vx * mod.DT
        vy += (-mod.GRAVITY - c * speed * vy) * mod.DT
    speed = math.hypot(vx, vy)
    assert speed < terminal, "снаряд проскочил предельную скорость"
    assert speed > terminal * 0.98, f"скорость {speed} не подошла к пределу {terminal}"
    assert peak < terminal, "предел нарушен на пути"

    # The same fall without drag leaves the limit far behind: the field means
    # something only because the answer has to integrate it.
    free = math.hypot(meta["vx0"], meta["vy0"] + mod.GRAVITY * 600 * mod.DT)
    assert free > terminal * 3.0, "свободное падение не отличимо от предела"


@pytest.mark.parametrize("seed", SEEDS)
def test_chain_without_relaxation_stretches_the_links(seed: int) -> None:
    """The links only keep their length because the scheme relaxes them."""
    task = _task(seed, "K-36")
    meta = task.oracle["meta"]
    lengths = meta["lengths"]
    assert max(abs(value - meta["length"]) for value in lengths) < 4e-3
    loose = [
        math.hypot(
            meta["loose"][i + 1][0] - meta["loose"][i][0],
            meta["loose"][i + 1][1] - meta["loose"][i][1],
        )
        for i in range(len(meta["loose"]) - 1)
    ]
    assert max(loose) > meta["length"] * 1.05
    assert meta["loose_residual"] > 20 * meta["residual"]


@pytest.mark.parametrize("seed", SEEDS)
def test_golden_scores_full_in_browser(seed: int) -> None:
    _needs_browser()
    for task in _tasks(seed):
        score = visual.evaluate(task, _transcript(task, task.oracle["final_answer"]))
        failed = [(check.name, check.detail) for check in score.checks if not check.passed]
        assert score.possible == task.max_points
        assert score.earned == task.max_points, f"{task.id} seed={seed}: {failed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_checks_reject_the_unstepped_probe(seed: int) -> None:
    """The simulation check is about the stepping, not about the scene."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        before = browser.probe_of(result)
        passed, detail = visual._predicates_report(spec.simulation, before)
        assert not passed, f"{task.id}: проверка проходит без шагов ({detail})"
        after = browser.probe_of(result, stepped=True)
        passed_after, detail_after = visual._predicates_report(spec.simulation, after)
        assert passed_after, f"{task.id}: проверка не проходит после шагов ({detail_after})"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_stepped_probe_matches_a_python_replay(seed: int) -> None:
    """The published check is a Python replay of the scheme, not a copy of the page."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        passed, detail = visual._predicates_report(
            spec.simulation, browser.probe_of(result, stepped=True)
        )
        assert passed, f"{task.id}: схема из промпта и эталон разошлись ({detail})"
