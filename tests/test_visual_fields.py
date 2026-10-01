"""Tests for the K-12..K-16 slice: grids, cellular automata and environments.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, points, determinism, fixtures that move with
  the seed, and constants the model can actually read off the prompt;
* every reference page scores 4/4 in the real browser, on more than one seed,
  which also re-runs the ``image`` predicates against the measured frame;
* the published check is a Python replay of the stated scheme: the probe taken
  *before* the steps fails the ``simulation`` predicates, so a page that merely
  echoes its fixture constants cannot pass.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_fields as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them,
#: not only for the one the suite happens to use.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code. Everything
#: else in the meta is either derived by the scheme or added by ``build_task``
#: (token estimates, driver flags), and the model is not told it.
FIXTURE_KEYS = {
    "K-12": ("alpha",),
    "K-13": ("seed", "density"),
    "K-14": ("seed", "density"),
    "K-15": ("amplitude", "centre", "sigma", "c2"),
    "K-16": ("amplitude",),
}

#: K-16 is a static environment: its probe is the hash of the lattice, and the
#: steps change nothing, so "the unstepped probe fails" is not a property this
#: scenario can have. Its discrimination is the exact hash instead.
STATIC_IDS = ("K-16",)


def _tasks(seed: int) -> list[core.TaskInstance]:
    return mod.build(random.Random(seed))


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


def _numbers(value: object) -> list[object]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    return []


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-12", "K-13", "K-14", "K-15", "K-16")
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
    """The fixtures are drawn from the stream, not baked in."""
    for index in range(len(mod.IDS)):
        drawn = {json.dumps(_tasks(seed)[index].oracle["meta"], sort_keys=True) for seed in range(16)}
        # The spaces are small (K-12 has 3 alphas x 3 step counts), so a few
        # collisions over 16 seeds are expected; a constant fixture is not.
        assert len(drawn) >= 6, f"{mod.IDS[index]}: фикстуры почти не зависят от seed ({len(drawn)} из 16)"


def test_every_fixture_number_is_in_the_prompt() -> None:
    """A scenario whose constants are hidden cannot be answered."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            for key in FIXTURE_KEYS[task.id]:
                for number in _numbers(meta[key]):
                    assert str(number) in task.prompt, (
                        f"{task.id} seed={seed}: числа {number} ({key}) нет в промпте"
                    )


def test_golden_page_structure() -> None:
    """The reference answers stay inside the answer budget and are deterministic."""
    for seed in SEEDS:
        for task in _tasks(seed):
            html = task.oracle["golden_html"]
            assert len(html.splitlines()) <= 150, f"{task.id}: эталон длиннее бюджета"
            assert "Math.random" not in html
            assert "performance.now" not in html
            assert "requestAnimationFrame" not in html
            assert "preserveDrawingBuffer: true" in html
            assert "setSize(800, 600, false)" in html
            assert "setPixelRatio(1)" in html
            assert "window.__ready = true" in html
            # K-16 is a relief, not a grid of cells: it is one displaced plane.
            expected = "PlaneGeometry" if task.id == "K-16" else "InstancedMesh"
            assert expected in html, f"{task.id}: эталон обязан рисовать сцену через {expected}"


def test_fixture_assertions_are_exercised() -> None:
    """``build`` proves its own fixture; the numbers it stores survive the trip."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            if task.id == "K-12":
                samples = meta["samples"]
                assert len({round(v, 4) for v in samples}) == 3
                assert 0.0 < samples[0] and 0.0 < samples[1] < samples[0]
            if task.id == "K-13":
                assert meta["burning"] > 5 and meta["burned"] > 4 and meta["trees"] > 0
            if task.id == "K-14":
                assert meta["alive"] > 0 and meta["alive"] != meta["initial_alive"]
            if task.id == "K-15":
                assert len({round(v, 4) for v in meta["samples"]}) == 3
                assert all(abs(v) > 5e-3 for v in meta["samples"])
                assert meta["energy"] > 0.0
            if task.id == "K-16":
                assert meta["min"] < meta["max"]
                assert len({round(v, 4) for v in meta["heights"]}) == 3


def test_garbage_answer_scores_zero() -> None:
    for task in _tasks(7):
        score = visual.evaluate(task, _transcript(task, "не знаю"))
        assert score.earned == 0, f"{task.id}: мусорный ответ набрал {score.earned}"


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
        if task.id in STATIC_IDS:
            continue
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        passed, detail = visual._predicates_report(spec.simulation, browser.probe_of(result))
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
