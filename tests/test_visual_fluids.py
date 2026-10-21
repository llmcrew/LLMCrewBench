"""Tests for the K-23/K-26 slice: a valve network and communicating tanks.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, points, determinism, fixtures that move with
  the seed, and constants the model can actually read off the prompt;
* every reference page scores 4/4 in the real browser, on more than one seed,
  which also re-runs the ``image`` predicates against the measured frame;
* the published check is a Python replay of the stated law: the probe taken
  *before* the steps fails the ``simulation`` predicates, the stepped probe
  agrees with the oracle, and the named mutants (swapped resistances, a leak
  through the closed valve, flow from the low tank) land outside tolerance.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_fluids as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them,
#: not only for the one the suite happens to use.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code. Everything
#: else in the meta is either derived by the law or added by ``build_task``
#: (token estimates, driver flags), and the model is not told it.
FIXTURE_KEYS = {
    "K-23": ("p0", "c02", "c23", "c21", "c13", "cv", "damping", "steps"),
    "K-26": ("a1", "a2", "h1_0", "h2_0", "g", "dt", "steps"),
}

#: Both scenarios step, so "the unstepped probe fails" holds for both.
STATIC_IDS: tuple[str, ...] = ()


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


def _render(task: core.TaskInstance):
    spec = visual.spec_of(task)
    result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
    assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
    return result


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-23", "K-26")
    assert mod.IDS_BY_PROFILE["pilot"] == ("K-23",)
    assert mod.IDS_BY_PROFILE["stretch-v1"] == mod.IDS
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "K"
        assert task.max_points == visual.POINTS_PER_CHECK * len(visual.CHECK_NAMES)
        spec = visual.spec_of(task)
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps > 0 and spec.dt > 0
        assert spec.need_same_frame
        assert "window.__step" in task.prompt and "window.__probe()" in task.prompt
        assert 'aria-label="Открыть клапан"' in task.prompt
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
            assert "OrthographicCamera" in html, f"{task.id}: в задании ортографическая камера"
            assert '<button id="valve"' in html
            assert "addEventListener('click'" in html
            assert "getBoundingClientRect" in html


def test_fixture_assertions_are_exercised() -> None:
    """``build`` proves its own fixture; the numbers it stores survive the trip."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            if task.id == "K-23":
                closed, opened = meta["closed"], meta["open"]
                assert closed["flows"]["branch"] > 1e-3
                assert closed["flows"]["valve"] == 0.0
                assert opened["flows"]["branch"] < -1e-3
                assert opened["flows"]["valve"] > 1e-3
                assert closed["balance"] <= 1e-4 and opened["balance"] <= 1e-4
                assert opened["pressures"][0] > opened["pressures"][1]
                assert closed["t"] == round(60 / 60, 4)
            if task.id == "K-26":
                initial = [meta["h1_0"], meta["h2_0"]]
                assert 0.0 < initial[1] < meta["levels"][1] < meta["levels"][0] < initial[0]
                assert meta["balance"] <= 1e-9
                assert meta["levels"][0] * meta["a1"] + meta["levels"][1] * meta["a2"] == pytest.approx(
                    initial[0] * meta["a1"] + initial[1] * meta["a2"], abs=1e-3
                )


def test_network_mutants_are_rejected() -> None:
    """Swapped resistances and a leak through the closed valve must be visible."""
    for seed in SEEDS:
        network_task = _tasks(seed)[0]
        meta = network_task.oracle["meta"]
        p0 = meta["p0"]
        c = {key: meta[key] for key in ("c02", "c23", "c21", "c13", "cv")}
        closed = mod.settle(p0, c, False, mod.STEP_CLICK)

        swapped = mod.settle(p0, {key: 1.0 / value for key, value in c.items()}, False, mod.STEP_CLICK)
        assert abs(swapped["pressures"][0] - closed["pressures"][0]) > mod.MUTANT_TOL
        assert abs(swapped["pressures"][1] - closed["pressures"][1]) > mod.MUTANT_TOL

        leaking = mod.network(p0, c, True)
        assert leaking["flows"]["valve"] > 1e-3
        assert abs(leaking["flows"]["branch"] - closed["flows"]["branch"]) > mod.MUTANT_TOL


def test_tank_mutant_flows_uphill() -> None:
    """A flow from the low tank drives the levels apart instead of together."""
    for seed in SEEDS:
        meta = _tasks(seed)[1].oracle["meta"]
        correct = mod.tanks(meta["a1"], meta["a2"], meta["h1_0"], meta["h2_0"], meta["g"], mod.STEPS26)
        mutant = mod.tanks(
            meta["a1"], meta["a2"], meta["h1_0"], meta["h2_0"], meta["g"], mod.STEPS26, invert=True
        )
        assert mutant["levels"][0] > meta["h1_0"] and mutant["levels"][1] < meta["h2_0"]
        assert abs(mutant["levels"][0] - correct["levels"][0]) > mod.MUTANT_TOL
        assert mutant["balance"] <= 1e-9


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
    """The simulation check is about the stepping and the click, not the scene."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result = _render(task)
        passed, detail = visual._predicates_report(spec.simulation, browser.probe_of(result))
        assert not passed, f"{task.id}: проверка проходит без шагов ({detail})"
        after = browser.probe_of(result, stepped=True)
        passed_after, detail_after = visual._predicates_report(spec.simulation, after)
        assert passed_after, f"{task.id}: проверка не проходит после шагов ({detail_after})"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_stepped_probe_matches_a_python_replay(seed: int) -> None:
    """The published check is a Python replay of the law, not a copy of the page."""
    _needs_browser()
    for task in _tasks(seed):
        result = _render(task)
        probe = browser.probe_of(result, stepped=True)
        spec = visual.spec_of(task)
        passed, detail = visual._predicates_report(spec.simulation, probe)
        assert passed, f"{task.id}: схема из промпта и эталон разошлись ({detail})"
        meta = task.oracle["meta"]

        if task.id == "K-23":
            p0 = meta["p0"]
            c = {key: meta[key] for key in ("c02", "c23", "c21", "c13", "cv")}
            closed = mod.settle(p0, c, False, mod.STEP_CLICK)
            opened = mod.settle(
                p0, c, True, mod.STEPS23 - mod.STEP_CLICK,
                (closed["pressures"][0], closed["pressures"][1]),
            )
            for prefix, state in (("closed", closed), ("open", opened)):
                for index in (0, 1):
                    assert probe[prefix]["pressures"][index] == pytest.approx(
                        state["pressures"][index], abs=1e-3
                    ), f"{task.id}: {prefix}.pressures.{index}"
                for name, value in state["flows"].items():
                    assert probe[prefix]["flows"][name] == pytest.approx(value, abs=1e-3), (
                        f"{task.id}: {prefix}.flows.{name}"
                    )
                assert probe[prefix]["balance"] <= 1e-4
            assert probe["closed"]["flows"]["valve"] == pytest.approx(0.0, abs=1e-9)
            assert probe["closed"]["flows"]["branch"] > 0.0 > probe["open"]["flows"]["branch"]
            assert probe["ui"]["tag"] == "BUTTON" and probe["ui"]["aria_pressed"] == "true"
            assert probe["closed"]["t"] == pytest.approx(1.0, abs=1e-4)
        else:
            expected = mod.tanks(meta["a1"], meta["a2"], meta["h1_0"], meta["h2_0"], meta["g"], mod.STEPS26)
            assert probe["at60"] == pytest.approx([meta["h1_0"], meta["h2_0"]], abs=1e-4)
            assert probe["levels"] == pytest.approx(expected["levels"], abs=1e-3)
            assert probe["volumes"] == pytest.approx(expected["volumes"], abs=1e-3)
            assert probe["balance"] <= 1e-6
            assert probe["levels"][0] < meta["h1_0"] < meta["h1_0"] + 1
            assert probe["levels"][1] > meta["h2_0"]
            assert probe["q"] > 0.0
            assert probe["t"] == pytest.approx(mod.STEPS26 * mod.STEP_DT, abs=1e-4)
