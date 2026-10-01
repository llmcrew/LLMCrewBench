"""Scenario K-1 (ballistics) pinned end to end, including the rendered frame.

The reference page is the answer the self-check replays, so it is rendered here
in the same browser the evaluator uses: a scenario whose own reference does not
score four out of four is a broken scenario, and that must fail in the test
suite rather than in a model run.
"""

from __future__ import annotations

import random

import pytest

from dailybench import browser, core, visual
from dailybench.config import RunConfig
from dailybench.rng import task_rng
from dailybench.suite import build_suite
from dailybench.tasks import visual_projectiles as vp

pytestmark = pytest.mark.skipif(
    not browser.availability()[0], reason=f"браузер недоступен: {browser.availability()[1]}"
)


def _task(seed: int = 20260915) -> core.TaskInstance:
    return vp.build(task_rng(seed, "K"))[0]


def _answer(text: str) -> core.Transcript:
    transcript = core.Transcript(task_id="K-1", category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=text)))
    return transcript


def test_k1_registers_in_the_suite_with_its_points():
    tasks = [task for task in build_suite(RunConfig()) if task.id == "K-1"]
    assert len(tasks) == 1
    task = tasks[0]
    assert task.category == "K"
    assert task.max_points == 4.0
    assert task.max_points == core.SHOWCASE_POINTS["K"] / 20


def test_k1_is_deterministic_in_the_seed_and_varies_with_it():
    assert _task(7).prompt == _task(7).prompt
    assert _task(7).prompt != _task(8).prompt
    assert _task(7).oracle["meta"] != _task(8).oracle["meta"]


def test_the_prompt_carries_every_fixture_number():
    task = _task()
    meta = task.oracle["meta"]
    assert f"гравитация — {meta['gravity']:.2f}" in task.prompt
    for index, (x, y, vy) in enumerate(meta["expected"]):
        spec = meta["launches"][index]
        assert f"`projectile-{index}`: x={spec['x0']}, y={spec['y0']}" in task.prompt
    assert "window.__step" in task.prompt and "window.__probe" in task.prompt


def test_the_reference_scores_four_out_of_four_in_the_browser():
    task = _task()
    score = visual.evaluate(task, _answer(task.oracle["final_answer"]))
    assert score.earned == score.possible == 4.0, score.as_dict()
    assert [check.name for check in score.checks] == list(visual.CHECK_NAMES)


def test_garbage_scores_zero():
    task = _task()
    score = visual.evaluate(task, _answer("Не знаю. Наверное, надо посчитать."))
    assert score.earned == 0.0 and score.possible == 4.0
    assert not score.skipped


def test_the_frame_matches_the_stated_physics_not_only_the_reference():
    """The published checks are driven by the fixture, not by the reference page."""
    task = _task()
    spec = visual.spec_of(task)
    result = browser.render(task.oracle["golden_html"], steps=spec.steps, dt=spec.dt)
    assert result.ok and result.same_frame

    # The simulation check is recomputed in Python: same three positions.
    ok, detail = visual._predicates_report(spec.simulation, browser.probe_of(result, True))
    assert ok, detail
    # A projectile that ignored gravity would land elsewhere: the designed
    # distractor must fail the same predicates.
    wrong = dict(browser.probe_of(result, True))
    wrong["projectiles"] = [{"x": p["x"], "y": p["y"] + 0.5, "vy": p["vy"]} for p in wrong["projectiles"]]
    assert not visual._predicates_report(spec.simulation, wrong)[0]


def test_image_thresholds_keep_a_margin_over_the_measured_frame():
    task = _task()
    spec = visual.spec_of(task)
    result = browser.render(task.oracle["golden_html"], steps=spec.steps, dt=spec.dt)
    warm = float((result.pixels or {}).get("warm", {}).get("fraction") or 0.0)
    threshold = next(
        float(getattr(p, "check_label", "").split(">= ")[-1].rstrip(")"))
        for p in spec.image
        if str(getattr(p, "check_label", "")).startswith("кадр: оранжевые снаряды")
    )
    assert warm >= 2 * threshold, f"кадр даёт {warm}, порог {threshold} — запас меньше двух раз"
    assert (result.pixels or {}).get("changed_fraction", 0) > 0.3


def test_every_seed_in_a_range_stays_renderable():
    for seed in (1, 2, 3, 11, 20260915):
        task = vp.build(task_rng(seed, "K"))[0]
        spec = visual.spec_of(task)
        result = browser.render(task.oracle["golden_html"], steps=spec.steps, dt=spec.dt, again=False)
        assert result.ok, (seed, result.reason, result.errors[:2])
        assert visual._predicates_report(spec.scene, browser.probe_of(result))[0]
        assert visual._predicates_report(spec.image, result.pixels or {})[0]
