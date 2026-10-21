"""Tests for the K-28 slice: лифт с FIFO-очередью и настоящими кнопками этажей.

Three things are proved here rather than assumed:

* the slice is well-formed — id and profile export, determinism, fixtures that
  move with the seed, and constants the model can read off the prompt;
* the reference page scores 4/4 in the real browser, and its probe before the
  steps fails the ``simulation`` predicates, so the check is about the machine
  and not about the scene; the stepped probe matches the Python replica;
* the three published defects (lost request, nearest-first, early door close)
  are visible: each mutant fails the simulation predicates while the golden
  page passes them.

The browser is optional: without Playwright the render-dependent tests skip.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_lift as mod

#: Seeds the fixtures are replayed on.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code.
FIXTURE_KEYS = {"K-28": ("presses", "steps")}

#: The three mutants of the state machine, as page edits and their label.
MUTANTS = (
    ("const LOST = false;", "const LOST = true;", "потеря запроса"),
    ("const NEAREST = false;", "const NEAREST = true;", "nearest-first"),
    ("const DOOR_TICKS = 2;", "const DOOR_TICKS = 1;", "раннее закрытие двери"),
)


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


def _render(task: core.TaskInstance, html: str) -> browser.RenderResult:
    spec = visual.spec_of(task)
    result, _ = visual._cached_render(html, spec.steps, spec.dt)
    assert result.ok, f"{task.id}: рендер не удался — {result.reason}"
    return result


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-28",)
    assert mod.IDS_BY_PROFILE == {"pilot": ("K-28",), "stretch-v1": ("K-28",)}
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "K"
        assert task.max_points == visual.POINTS_PER_CHECK * len(visual.CHECK_NAMES)
        spec = visual.spec_of(task)
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps > 0 and spec.dt > 0
        assert spec.need_same_frame
        assert "window.__step" in task.prompt and "window.__probe()" in task.prompt
        assert "data-floor" in task.prompt and "journal" in task.prompt
        assert "FIFO" in task.prompt and "OrthographicCamera" in task.prompt
        assert task.prompt.endswith(visual.HTML_CONTRACT)
        assert task.oracle["answer_file"] == visual.PAGE_FILE
        assert task.oracle["golden_html"].startswith("<!DOCTYPE html>")
        assert "./vendor/three.module.js" in task.oracle["golden_html"]


def test_build_is_deterministic() -> None:
    first = _tasks(11)
    second = _tasks(11)
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle["golden_html"] for task in first] == [
        task.oracle["golden_html"] for task in second
    ]
    assert [task.oracle["meta"] for task in first] == [task.oracle["meta"] for task in second]


def test_fixtures_move_with_the_seed() -> None:
    """The pressed floors and the probe tick are drawn from the stream."""
    drawn = {json.dumps(_tasks(seed)[0].oracle["meta"], sort_keys=True) for seed in range(16)}
    assert len(drawn) >= 6, f"фикстуры почти не зависят от seed ({len(drawn)} из 16)"


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
    """The reference answer stays inside the budget and clicks real buttons."""
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
            assert "createElement('button')" in html
            assert "btn.dataset.floor" in html
            assert "btn.addEventListener('click'" in html
            assert "buttons[f].click()" in html


def test_fixture_assertions_are_exercised() -> None:
    """The fixture stores the machine state it proved to itself."""
    for seed in SEEDS:
        task = _tasks(seed)[0]
        meta = task.oracle["meta"]
        assert [click["tick"] for click in meta["clicks"]] == [5, 12, 30]
        assert meta["queue"] == [meta["presses"]["30"]]
        assert meta["served"] == [meta["presses"]["5"], meta["presses"]["12"]]
        assert meta["door"] == "closed" and meta["car"] != meta["queue"][0]
        assert meta["steps"] > 30
        # A page that never steps cannot satisfy the final state.
        start = mod.lift({}, 0)
        assert start["car"] == 0 and start["journal"] == []


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
    """The simulation check is about the machine, not about the scene."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result = _render(task, task.oracle["golden_html"])
        passed, detail = visual._predicates_report(spec.simulation, browser.probe_of(result))
        assert not passed, f"{task.id}: проверка проходит без шагов ({detail})"
        passed_after, detail_after = visual._predicates_report(
            spec.simulation, browser.probe_of(result, stepped=True)
        )
        assert passed_after, f"{task.id}: проверка не проходит после шагов ({detail_after})"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_stepped_probe_matches_a_python_replay(seed: int) -> None:
    """The published check is the Python machine, not a copy of the page."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result = _render(task, task.oracle["golden_html"])
        passed, detail = visual._predicates_report(
            spec.simulation, browser.probe_of(result, stepped=True)
        )
        assert passed, f"{task.id}: автомат из промпта и эталон разошлись ({detail})"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("old,new,label", MUTANTS)
def test_mutants_fail_the_simulation(seed: int, old: str, new: str, label: str) -> None:
    _needs_browser()
    for task in _tasks(seed):
        html = task.oracle["golden_html"]
        assert old in html, f"{task.id}: в эталоне нет {old}"
        spec = visual.spec_of(task)
        result = _render(task, html.replace(old, new))
        passed, detail = visual._predicates_report(
            spec.simulation, browser.probe_of(result, stepped=True)
        )
        assert not passed, f"{task.id} seed={seed}: мутант «{label}» проходит проверку ({detail})"
