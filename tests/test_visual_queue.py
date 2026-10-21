"""Tests for the K-27 slice: a FIFO channel network with delays and capacities.

Three things are proved here rather than assumed:

* the slice is well-formed — id, points, determinism, a fixture that moves with
  the seed, and every fixture number readable off the prompt;
* the reference page scores 4/4 in the real browser on more than one seed,
  which also re-runs the ``image`` predicates against the measured frame;
* the published check is a Python replay of the stated rule: the probe taken
  *before* the ticks fails the ``simulation`` predicates, so a page that merely
  echoes its fixture constants cannot pass, and the three classic mistakes
  (duplicating at the branch, losing on a full channel, serving the queue
  back-to-front) each move the checked numbers.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_queue as mod

#: Seeds the fixtures are replayed on.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code. The arrival
#: table is checked separately: it is a table, not a number.
FIXTURE_KEYS = {"K-27": ("ttl",)}


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
    assert mod.IDS == ("K-27",)
    assert mod.IDS_BY_PROFILE["pilot"] == ("K-27",)
    assert mod.IDS_BY_PROFILE["stretch-v1"] == mod.IDS
    assert set(mod.IDS_BY_PROFILE) == {"pilot", "stretch-v1"}
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "K"
        assert task.max_points == visual.POINTS_PER_CHECK * len(visual.CHECK_NAMES)
        spec = visual.spec_of(task)
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps == mod.TICKS and spec.steps > 0 and spec.dt > 0
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
    """The arrival table is drawn from the stream, not baked in."""
    for index in range(len(mod.IDS)):
        drawn = {
            json.dumps(_tasks(seed)[index].oracle["meta"], sort_keys=True)
            for seed in range(16)
        }
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
            table = {
                tick: count
                for tick, count in enumerate(meta["arrivals"])
                if count
            }
            for tick, count in sorted(table.items())[:5]:
                assert f"\n{tick}: {count}\n" in task.prompt, (
                    f"{task.id} seed={seed}: строки прибытий {tick}: {count} нет в промпте"
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
            assert "OrthographicCamera(-10, 10, 7.5, -7.5, 0.1, 100)" in html
            assert "window.__ready = true" in html
            assert "0xff3bd0" in html and "0xffb347" in html
            assert "PlaneGeometry(0.28, 0.28)" in html


def test_fixture_assertions_are_exercised() -> None:
    """``build`` proves its own fixture; the numbers it stores survive the trip."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            assert meta["balance"] is True
            assert meta["created"] + meta["dropped"] == meta["offered"]
            assert meta["created"] == (
                meta["delivered"] + meta["in_flight"] + meta["expired"] + sum(meta["queues"])
            )
            assert meta["dropped"] > 3 and meta["expired"] > 3 and meta["delivered"] > 3
            assert len(meta["frames"]) == len(mod.FRAMES)
            assert meta["frames"][0] == list(mod.PRELOAD_QUEUES)
            assert len(meta["arrivals"]) == mod.TICKS + 1
            assert all(v == int(v) for v in meta["queues"])
            assert len(set(map(tuple, meta["frames"]))) >= 2


def test_garbage_answer_scores_zero() -> None:
    for task in _tasks(7):
        score = visual.evaluate(task, _transcript(task, "не знаю"))
        assert score.earned == 0, f"{task.id}: мусорный ответ набрал {score.earned}"


def _state(seed: int) -> dict:
    task = _tasks(seed)[0]
    meta = task.oracle["meta"]
    return mod.replica(meta["arrivals"], meta["ttl"])


@pytest.mark.parametrize("seed", SEEDS)
def test_mutants_depart_from_the_oracle(seed: int) -> None:
    """A duplicate, a loss on full and a LIFO queue each move a checked number."""
    task = _tasks(seed)[0]
    meta = task.oracle["meta"]
    truth = _state(seed)
    assert truth["offered"] == meta["offered"]

    duplicated = mod.replica(meta["arrivals"], meta["ttl"], dup=True)
    assert not duplicated["balance"], f"{task.id}: дубль на развилке не нарушил баланс"
    duplicated_alive = (
        duplicated["delivered"] + duplicated["in_flight"]
        + duplicated["expired"] + sum(duplicated["queues"])
    )
    assert duplicated_alive > duplicated["created"], f"{task.id}: дубля не видно в балансе"

    lost = mod.replica(meta["arrivals"], meta["ttl"], lose=True)
    assert lost["dropped"] > truth["dropped"], f"{task.id}: потеря на полном канале не видна"

    lifo = mod.replica(meta["arrivals"], meta["ttl"], lifo=True)
    assert lifo["frames"] != truth["frames"], f"{task.id}: LIFO не сдвинул кадры"


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
    """The simulation check is about the ticks, not about the initial scene."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        passed, detail = visual._predicates_report(spec.simulation, browser.probe_of(result))
        assert not passed, f"{task.id}: проверка проходит без тиков ({detail})"
        after = browser.probe_of(result, stepped=True)
        passed_after, detail_after = visual._predicates_report(spec.simulation, after)
        assert passed_after, f"{task.id}: проверка не проходит после тиков ({detail_after})"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_stepped_probe_matches_a_python_replay(seed: int) -> None:
    """The published check is a Python replay of the rule, not a copy of the page."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        passed, detail = visual._predicates_report(
            spec.simulation, browser.probe_of(result, stepped=True)
        )
        assert passed, f"{task.id}: правило из промпта и эталон разошлись ({detail})"
