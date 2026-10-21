"""Tests for the K-29..K-30 slice: truss equilibrium and clamped PD servo.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, profile exports, points, determinism, fixtures
  that move with the seed, and constants the model can actually read off the
  prompt;
* every reference page scores 4/4 in the real browser, on more than one seed,
  which also re-runs the ``image`` predicates against the measured frame;
* the published check is a Python replay of the stated scheme: the probe taken
  *before* the steps fails the ``simulation`` predicates, and the mutants named
  by the ticket — a flipped tension/compression sign, a missing support, a
  disabled clamp and a stale target — are rejected by the same predicates.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_structures as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them,
#: not only for the one the suite happens to use.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code. Everything
#: else in the meta is either derived by the scheme or added by ``build_task``
#: (token estimates, driver flags), and the model is not told it.
FIXTURE_KEYS = {
    "K-29": ("nodes", "load"),
    "K-30": ("x0", "target_a", "target_b", "kp", "kd", "u_max", "steps"),
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


def _probe(task: core.TaskInstance) -> dict:
    """A probe payload the fixture itself would return after the steps."""
    meta = task.oracle["meta"]
    if task.id == "K-29":
        nodes = {name: tuple(point) for name, point in meta["nodes"].items()}
        reactions = list(meta["reactions"])
        forces = list(meta["forces"])
        return {
            "nodes": meta["nodes"],
            "members": list(mod.MEMBERS),
            "load": meta["load"],
            "reactions": reactions,
            "forces": forces,
            "states": list(meta["states"]),
            "residual": list(meta["residual"]),
            "steps": 1,
        }
    return {
        "t": 2.0,
        "steps": mod.STEPS30,
        "target": meta["target"],
        "x": meta["x"],
        "v": meta["v"],
        "u": meta["u"],
        "saturated_steps": meta["saturated_steps"],
        "rail_half": mod.RAIL_HALF,
    }


def _mutants(task: core.TaskInstance, good: dict) -> list[tuple[str, dict]]:
    """The named mutants, each as a probe built from the same fixture."""
    meta = task.oracle["meta"]
    out: list[tuple[str, dict]] = []
    if task.id == "K-29":
        nodes = {name: tuple(point) for name, point in meta["nodes"].items()}
        flipped = copy.deepcopy(good)
        flipped["forces"][0] = -flipped["forces"][0]
        flipped["states"][0] = "compression"
        flipped["residual"] = mod.truss_residual(nodes, flipped["reactions"], flipped["forces"], meta["load"])
        out.append(("tension/compression", flipped))

        dropped = copy.deepcopy(good)
        dropped["reactions"][2] = 0.0
        dropped["residual"] = mod.truss_residual(nodes, dropped["reactions"], dropped["forces"], meta["load"])
        out.append(("пропуск опоры", dropped))

        relabelled = copy.deepcopy(good)
        relabelled["states"][0] = "compression"
        out.append(("состояние стержня", relabelled))
        return out

    without_clamp = copy.deepcopy(good)
    x, v, u, target, saturated = meta["without_clamp"]
    without_clamp.update({"x": x, "v": v, "u": u, "target": target, "saturated_steps": saturated})
    out.append(("без clamp", without_clamp))

    old_target = copy.deepcopy(good)
    x, v, u, target, saturated = meta["old_target"]
    old_target.update({"x": x, "v": v, "u": u, "target": target, "saturated_steps": saturated})
    out.append(("старая цель", old_target))
    return out


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-29", "K-30")
    assert mod.IDS_BY_PROFILE == {"pilot": mod.IDS, "stretch-v1": mod.IDS}
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
            # Both scenes are flat structural diagrams under a fixed orthographic
            # camera, so the frame does not depend on the object distance.
            assert "OrthographicCamera" in html, f"{task.id}: камера обязана быть ортографической"
            if task.id == "K-30":
                assert "bench:target" in html and "CustomEvent" in html
                assert f"step === {mod.RETARGET_AT}" in html
            else:
                assert "residual" in html and "CylinderGeometry" in html


def test_fixture_assertions_are_exercised() -> None:
    """``build`` proves its own fixture; the numbers it stores survive the trip."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            if task.id == "K-29":
                reactions, forces = meta["reactions"], meta["forces"]
                assert reactions[1] > 0.0 and reactions[2] > 0.0
                assert abs(reactions[1] + reactions[2] - meta["load"]) < 1e-9
                assert meta["states"] == ["tension", "compression", "compression"]
                nodes = {name: tuple(point) for name, point in meta["nodes"].items()}
                assert meta["residual"] == [0.0, 0.0, 0.0]
                assert max(abs(v) for v in mod.truss_residual(nodes, reactions, forces, meta["load"])) < 1e-4
            if task.id == "K-30":
                assert 20 < meta["saturated_steps"] < meta["steps"]
                assert meta["target_b"] == -meta["target_a"]
                assert abs(meta["without_clamp"][0] - meta["x"]) > 0.2
                assert abs(meta["old_target"][3] - meta["target"]) > 0
                # The fixture really is the stated controller, not a copy.
                x, v, u, target, saturated = mod.servo(
                    meta["target_a"], meta["target_b"], meta["kp"], meta["kd"]
                )
                assert [mod.round6(x), mod.round6(v), u, target, saturated] == [
                    meta["x"], meta["v"], meta["u"], meta["target"], meta["saturated_steps"]
                ]


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
    """A flipped sign, a dropped support, no clamp and a stale target all fail."""
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        good = _probe(task)
        passed, detail = visual._predicates_report(spec.simulation, good)
        assert passed, f"{task.id}: собранный probe обязан проходить ({detail})"
        for label, bad in _mutants(task, good):
            passed, detail = visual._predicates_report(spec.simulation, bad)
            assert not passed, f"{task.id} seed={seed}: мутант «{label}» прошёл ({detail})"
