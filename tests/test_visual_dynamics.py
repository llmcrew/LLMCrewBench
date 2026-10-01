"""Tests for the K-2..K-6 slice: point dynamics and a free rigid body.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, points, determinism, and fixtures that move
  with the seed;
* every reference page scores 4/4 in the real browser, on more than one seed,
  which also re-runs the ``image`` predicates against the measured frame;
* the checks are not trivially satisfiable — the probe taken *before* the steps
  must fail the ``simulation`` predicates, so a page that does not integrate the
  stated scheme cannot pass.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_dynamics as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them,
#: not only for the one the suite happens to use.
SEEDS = (7, 2, 13, 0, 5)

#: Meta keys holding fixture inputs the answer has to hard-code. Everything
#: else in the meta is either derived by the scheme or added by ``build_task``
#: (token estimates, driver flags), and the model is not told it.
FIXTURE_KEYS = {
    "k", "m", "c", "x0", "v0", "L", "theta0", "omega0", "w", "r", "d0",
    "a", "b", "inertia",
}


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


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-2", "K-3", "K-4", "K-5", "K-6")
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "K"
        assert task.max_points == visual.POINTS_PER_CHECK * len(visual.CHECK_NAMES)
        spec = visual.spec_of(task)
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps > 0 and spec.dt > 0
        assert spec.need_same_frame
        # The common contract is appended by build_task, and the answer key is a
        # real page, not a description of one.
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
    """A scenario whose constants are hidden cannot be answered.

    Every number the model has to hard-code (the state it starts from and the
    constants of the scheme) must appear in the prompt text verbatim.
    """
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


@pytest.mark.parametrize("seed", SEEDS)
def test_golden_scores_full_in_browser(seed: int) -> None:
    _needs_browser()
    for task in _tasks(seed):
        transcript = _transcript(task, task.oracle["final_answer"])
        score = visual.evaluate(task, transcript)
        failed = [(check.name, check.detail) for check in score.checks if not check.passed]
        assert score.possible == task.max_points
        assert score.earned == task.max_points, f"{task.id} seed={seed}: {failed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_checks_reject_the_unstepped_probe(seed: int) -> None:
    """The simulation check must be about the stepping, not about the scene.

    The probe read straight after load holds the initial state; a page that
    merely reports its fixture constants must not pass the simulation check.
    """
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(
            task.oracle["golden_html"], spec.steps, spec.dt
        )
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
        passed, detail = visual._predicates_report(spec.simulation, browser.probe_of(result, stepped=True))
        assert passed, f"{task.id}: схема из промпта и эталон разошлись ({detail})"
