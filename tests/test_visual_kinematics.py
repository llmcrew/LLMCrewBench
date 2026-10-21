"""Tests for the K-21..K-22 slice: gear kinematics and a closed linkage.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, profile split, points, determinism, fixtures
  that move with the seed, and constants the model can read off the prompt;
* every reference page scores 4/4 in the real browser, on more than one seed;
* the published checks are a Python replay of the stated geometry: the probe
  taken *before* the steps fails, the mirror branch of the linkage fails even
  though it closes perfectly, and a gear pair with the wrong direction or a
  unit ratio fails as well.

The browser is optional: without Playwright the render-dependent tests skip.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_kinematics as mod

SEEDS = (7, 2, 13, 0, 5, 11)

FIXTURE_KEYS = {
    "K-21": ("r1", "r2", "omega1", "theta1_0", "theta2_0"),
    "K-22": ("a", "b", "c", "d", "theta0", "omega"),
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


def _reject(task: core.TaskInstance, probe: dict) -> None:
    """The published simulation predicates must reject a wrong mechanism."""
    passed, detail = visual._predicates_report(visual.spec_of(task).simulation, probe)
    assert not passed, f"{task.id}: неверная модель прошла проверку ({detail})"


def _gear_probe(meta: dict, *, same_direction: bool = False, unit_ratio: bool = False) -> dict:
    frames = mod.gear_state(
        meta["r1"], meta["r2"], meta["omega1"], meta["theta1_0"], meta["theta2_0"],
        meta["steps"], same_direction=same_direction, unit_ratio=unit_ratio,
    )
    ratio = (-1.0 if unit_ratio else -meta["r1"] / meta["r2"])
    if same_direction:
        ratio = -ratio
    last = frames[-1]
    return {
        "count": 2,
        "r": [meta["r1"], meta["r2"]],
        "omega": [meta["omega1"], mod.round4(ratio * meta["omega1"])],
        "ratio": mod.round4(ratio),
        "angles": [mod.round4(last[1]), mod.round4(last[2])],
        "marks": [[mod.round4(last[3]), mod.round4(last[4])],
                  [mod.round4(last[5]), mod.round4(last[6])]],
        "steps": meta["steps"],
        "frames": [[mod.round4(v) for v in frame] for frame in frames],
    }


def _branch_probe(meta: dict, branch: int) -> dict:
    frames = mod.fourbar_frames(
        meta["a"], meta["b"], meta["c"], meta["d"], meta["theta0"], meta["omega"], branch
    )
    joints, inp, out, closure = mod.fourbar(
        meta["a"], meta["b"], meta["c"], meta["d"], meta["theta0"] + meta["omega"] * 2, branch
    )
    return {
        "count": 4,
        "lengths": [meta["a"], meta["b"], meta["c"], meta["d"]],
        "joints": [[mod.round4(v) for v in joint] for joint in joints],
        "input_angle": mod.round4(inp),
        "output_angle": mod.round4(out),
        "closure": mod.round4(closure),
        "steps": meta["steps"],
        "frames": [[mod.round4(v) for v in frame] for frame in frames],
    }


def test_ids_profile_split_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-21", "K-22")
    assert mod.IDS_BY_PROFILE == {"pilot": ("K-21",), "stretch-v1": mod.IDS}
    assert set(mod.IDS_BY_PROFILE["pilot"]) <= set(mod.IDS)
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "K"
        assert task.max_points == visual.POINTS_PER_CHECK * len(visual.CHECK_NAMES)
        spec = visual.spec_of(task)
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps == 120 and abs(spec.dt - 1 / 60) < 1e-12
        assert spec.need_same_frame
        assert "window.__step" in task.prompt and "window.__probe()" in task.prompt
        assert "OrthographicCamera" in task.prompt
        assert task.prompt.endswith(visual.HTML_CONTRACT)
        golden = task.oracle["golden_html"]
        assert golden.startswith("<!DOCTYPE html>")
        assert "./vendor/three.module.js" in golden
        assert "OrthographicCamera" in golden


def test_build_is_deterministic() -> None:
    first, second = _tasks(11), _tasks(11)
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle["golden_html"] for task in first] == [
        task.oracle["golden_html"] for task in second
    ]
    assert [task.oracle["meta"] for task in first] == [task.oracle["meta"] for task in second]


def test_fixtures_move_with_the_seed() -> None:
    for index in range(len(mod.IDS)):
        drawn = {
            json.dumps(_tasks(seed)[index].oracle["meta"], sort_keys=True) for seed in range(16)
        }
        assert len(drawn) >= 6, f"{mod.IDS[index]}: фикстуры почти не зависят от seed"


def test_every_fixture_number_is_in_the_prompt() -> None:
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            for key in FIXTURE_KEYS[task.id]:
                for number in _numbers(meta[key]):
                    assert str(number) in task.prompt, (
                        f"{task.id} seed={seed}: числа {number} ({key}) нет в промпте"
                    )


def test_golden_page_structure() -> None:
    for seed in SEEDS:
        for task in _tasks(seed):
            html = task.oracle["golden_html"]
            assert len(html.splitlines()) <= 150, f"{task.id}: эталон длиннее бюджета"
            for absent in ("Math.random", "performance.now", "requestAnimationFrame"):
                assert absent not in html
            assert "preserveDrawingBuffer: true" in html
            assert "setSize(800, 600, false)" in html
            assert "setPixelRatio(1)" in html
            assert "window.__ready = true" in html
            expected = "RingGeometry" if task.id == "K-21" else "BoxGeometry(1, 0.18"
            assert expected in html


def test_fixture_assertions_are_exercised() -> None:
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            if task.id == "K-21":
                assert meta["omega2"] < 0.0
                assert abs(meta["ratio"] + meta["r1"] / meta["r2"]) < 1e-4
                assert abs(meta["r1"] + meta["r2"]) > 0.0
            if task.id == "K-22":
                assert meta["a"] + meta["b"] <= meta["c"] + meta["d"], "не кривошип-коромысло"
                joints = meta["joints"]
                assert len({(round(x, 4), round(y, 4)) for x, y in joints}) == 4
                assert abs(joints[3][0] - meta["d"]) < 1e-6 and joints[0] == [0.0, 0.0]


def test_fourbar_oracle_stays_closed_and_keeps_its_lengths() -> None:
    for seed in range(16):
        task = _tasks(seed)[1]
        meta = task.oracle["meta"]
        for step in (0, 30, 60, 90, 120):
            joints, inp, out, closure = mod.fourbar(
                meta["a"], meta["b"], meta["c"], meta["d"], meta["theta0"] + meta["omega"] * step / 60
            )
            assert closure <= 1e-3
            (ax, ay), (bx, by), (cx, cy), (dx, dy) = joints
            assert abs(math.hypot(bx - ax, by - ay) - meta["a"]) <= 1e-3
            assert abs(math.hypot(cx - bx, cy - by) - meta["b"]) <= 1e-3
            assert abs(math.hypot(cx - dx, cy - dy) - meta["c"]) <= 1e-3
            assert abs(inp - mod.norm_angle(meta["theta0"] + meta["omega"] * step / 60)) <= 1e-9
            assert 0.0 <= out < mod.TAU


def test_gears_mutant_direction_and_ratio_are_rejected() -> None:
    for seed in SEEDS:
        task = _tasks(seed)[0]
        meta = task.oracle["meta"]
        _reject(task, _gear_probe(meta, same_direction=True))
        _reject(task, _gear_probe(meta, unit_ratio=True))


def test_fourbar_mirror_branch_is_rejected() -> None:
    for seed in SEEDS:
        task = _tasks(seed)[1]
        mirror = _branch_probe(task.oracle["meta"], -1)
        # The mirror closes just as well: only the branch differs.
        assert mirror["closure"] <= 1e-3
        _reject(task, mirror)


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
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        passed, detail = visual._predicates_report(
            spec.simulation, browser.probe_of(result, stepped=True)
        )
        assert passed, f"{task.id}: схема из промпта и эталон разошлись ({detail})"
