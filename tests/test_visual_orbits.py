"""Tests for the K-31..K-32 slice: two-body gravity and a series RLC step.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, profile exports, points, determinism, fixtures
  that move with the seed, and constants the model can read off the prompt;
* every reference page scores 4/4 in the real browser on more than one seed,
  which also re-runs the ``image`` predicates against the measured frame;
* the published checks are a Python replay of the stated schemes: the probe
  taken *before* the steps fails the ``simulation`` predicates, and the mutants
  named by the ticket — a missing force, a flipped force sign, no damping and a
  flipped damping sign — are rejected by the same predicates.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_orbits as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them,
#: not only for the one the suite happens to use.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code. Everything
#: else is in the meta because the scheme derives it.
FIXTURE_KEYS = {
    "K-31": ("m1", "m2", "d", "g", "steps", "dt", "half"),
    "K-32": ("vin", "r", "l", "c", "steps", "dt"),
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


def _probe(task: core.TaskInstance, *, mutant: str = "good") -> dict:
    """A probe payload built from the fixture by the published replay."""
    meta = task.oracle["meta"]
    if task.id == "K-31":
        state = mod.kepler(
            meta["m1"],
            meta["m2"],
            meta["d"],
            mod.STEPS31,
            force=mutant != "no_force",
            sign=-1.0 if mutant == "sign" else 1.0,
        )
        stage = state["half"]
        return {
            "steps": mod.STEPS31,
            "t": mod.round6(mod.STEPS31 * mod.DT31),
            "objects": ["body_a", "body_b", "trail_a", "trail_b"],
            "camera": list(mod.CAMERA),
            "bodies": [
                {key: mod.round6(body[key]) for key in ("mass", "x", "y", "vx", "vy")}
                for body in state["bodies"]
            ],
            "energy": mod.round6(state["energy"]),
            "energy0": mod.round6(state["energy0"]),
            "momentum": [mod.round6(v) for v in state["momentum"]],
            "distance": mod.round6(state["distance"]),
            "half": {
                **{key: mod.round6(stage[key]) for key in ("x1", "y1", "x2", "y2")},
                "energy": mod.round6(stage["energy"]),
                "momentum": [mod.round6(v) for v in stage["momentum"]],
                "distance": mod.round6(stage["distance"]),
                "vx1": 0.0, "vy1": 0.0, "vx2": 0.0, "vy2": 0.0,
            },
        }
    state = mod.rlc(
        meta["vin"],
        meta["r"],
        meta["l"],
        meta["c"],
        damping=mutant != "no_damping",
        sign=1.0 if mutant == "sign" else -1.0,
    )
    return {
        "steps": mod.STEPS32,
        "t": mod.round6(mod.STEPS32 * mod.DT32),
        "objects": ["board", "source", "inductor", "resistor", "capacitor", "gauge", "wires"],
        "camera": list(mod.CAMERA),
        "vc0": 0.0,
        "i0": 0.0,
        **{key: mod.round6(state[key]) for key in ("vc1", "i1", "vc2", "i2", "vc", "i", "peak")},
        "peak_step": state["peak_step"],
        "crossings": state["crossings"],
        "gauge": {"x": 3.4, "y": 0.2},
    }


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-31", "K-32")
    assert mod.IDS_BY_PROFILE["pilot"] == ("K-31",)
    assert mod.IDS_BY_PROFILE["stretch-v1"] == mod.IDS
    assert set(mod.IDS_BY_PROFILE["pilot"]) <= set(mod.IDS)
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
            # Both scenes are flat diagrams under a fixed orthographic camera.
            assert "OrthographicCamera" in html, f"{task.id}: камера обязана быть ортографической"
            if task.id == "K-31":
                assert "half = {" in html and "Math.hypot" in html
            else:
                assert "/ C" in html and "crossings" in html


def test_fixture_assertions_are_exercised() -> None:
    """``build`` proves its own fixture; the stored numbers survive the trip."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            if task.id == "K-31":
                assert meta["energy0"] < -0.1
                assert abs(meta["energy"] - meta["energy0"]) < 1e-4
                assert abs(meta["distance"] - meta["d"]) < 2e-3
                good = _probe(task)
                assert max(abs(v) for v in good["momentum"]) < 1e-3
                for mutant in ("no_force", "sign"):
                    bad = _probe(task, mutant=mutant)
                    assert max(
                        abs(bad["bodies"][i][axis] - good["bodies"][i][axis])
                        for i in range(2)
                        for axis in ("x", "y")
                    ) > 0.15, f"{task.id} seed={seed}: мутант «{mutant}» не отличился"
            else:
                vin = meta["vin"]
                assert 1.3 * vin <= meta["peak"] <= 1.85 * vin
                assert abs(meta["vc"] - vin) < 0.3 * vin
                assert meta["crossings"] >= 2
                assert meta["undamped_peak"] > 1.9 * vin
                assert abs(meta["wrong_sign"][0] - meta["vc1"]) > 0.3


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


@pytest.mark.parametrize("seed", SEEDS)
def test_named_mutants_are_rejected(seed: int) -> None:
    """No force, a flipped sign, no damping and a flipped damping sign all fail."""
    mutants = {"K-31": ("no_force", "sign"), "K-32": ("no_damping", "sign")}
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        good = _probe(task)
        passed, detail = visual._predicates_report(spec.simulation, good)
        assert passed, f"{task.id}: собранный probe обязан проходить ({detail})"
        for mutant in mutants[task.id]:
            passed, detail = visual._predicates_report(spec.simulation, _probe(task, mutant=mutant))
            assert not passed, f"{task.id} seed={seed}: мутант «{mutant}» прошёл ({detail})"
