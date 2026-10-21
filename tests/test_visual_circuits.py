"""Tests for the K-24/K-25 slice: RC step response and thermostat hysteresis.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, points, determinism, fixtures that move with
  the seed, and constants the model can actually read off the prompt;
* every reference page scores 4/4 in the real browser on more than one seed;
* the published checks discriminate: the probe taken *before* the steps fails
  ``simulation``, the stepped probe equals a Python replay of the stated rule,
  and the three K-24 mutants (frozen voltage, wrong polarity, the analytic
  formula instead of Euler) and both K-25 mutants (one threshold, switching
  after the step) lose at least one point.

The browser is optional: without Playwright the render-dependent tests skip,
because "not verified" is not a zero.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_circuits as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them,
#: not only for the one the suite happens to use.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code.
FIXTURE_KEYS = {
    "K-24": ("vin", "r", "c"),
    "K-25": ("t_start", "t_low", "t_high", "tout0", "tout_event", "heat", "k"),
}

#: The K-24 Euler line, replaced to produce the wrong answers a model may give.
EULER = "  vc += dt * (VIN - vc) / (R * C);\n  clock += dt;\n"

#: The K-25 switch block, exactly as the reference page writes it.
HYSTERESIS = """  if (T <= TLOW) {
    if (!on) { onCount += 1; onSteps.push(step); }
    on = true;
  } else if (T >= THIGH) {
    if (on) { offCount += 1; offSteps.push(step); }
    on = false;
  }
"""

HEAT_STEP = """  if (on) heatingSteps += 1;
  T += (on ? HEAT : -K * (T - outside)) * dt;
"""


def _tasks(seed: int) -> list[core.TaskInstance]:
    return mod.build(random.Random(seed))


def _transcript(task: core.TaskInstance, text: str) -> core.Transcript:
    return core.Transcript(
        task_id=task.id,
        category=task.category,
        turns=[core.Turn(messages=[], reply=core.ModelReply(text=text))],
    )


def _fenced(html: str) -> str:
    return f"```html\n{html}\n```"


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
    assert mod.IDS == ("K-24", "K-25")
    assert mod.IDS_BY_PROFILE == {"pilot": ("K-25",), "stretch-v1": mod.IDS}
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
    assert visual.spec_of(tasks[0]).steps == mod.SNAP_THREE
    assert visual.spec_of(tasks[1]).steps == 180


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
            expected = "gauge" if task.id == "K-24" else "heater"
            assert f"'{expected}'" in html, f"{task.id}: эталон обязан рисовать {expected}"
            assert EULER in html or HYSTERESIS in html


def test_fixture_assertions_are_exercised() -> None:
    """``build`` proves its own fixture; the stored numbers survive the trip."""
    for seed in SEEDS:
        tasks = _tasks(seed)
        rc, thermostat = tasks
        meta = rc.oracle["meta"]
        assert 0.0 < meta["v1"] < meta["v3"] < meta["vin"]
        assert meta["i1"] > meta["i3"] > 0.0
        assert meta["i1"] < meta["vin"] / meta["r"]
        exact = [mod.rc_exact(meta["vin"], meta["r"], meta["c"], t) for t in (1.0, 3.0)]
        assert abs(meta["v1"] - mod.round4(exact[0])) > 5e-4
        assert abs(meta["v3"] - mod.round4(exact[1])) > 1e-3

        meta = thermostat.oracle["meta"]
        assert meta["switches_on"] >= 3 and meta["switches_off"] >= 3
        assert len(meta["on_steps"]) == meta["switches_on"]
        assert len(meta["off_steps"]) == meta["switches_off"]
        assert meta["on_steps"][0] < meta["off_steps"][0]
        assert 0 < meta["heating_steps"] < meta["steps"]
        assert meta["temperature"] <= meta["t_high"] + 0.4
        assert meta["outside"] == meta["tout_event"]


def test_hysteresis_is_not_a_single_threshold() -> None:
    """The fixture itself proves the two thresholds behave differently."""
    for seed in SEEDS:
        meta = _tasks(seed)[1].oracle["meta"]
        single = mod.thermostat(
            meta["t_start"], meta["t_low"], meta["t_high"], meta["tout0"],
            meta["tout_event"], meta["heat"], meta["k"], meta["steps"], meta["dt"],
            single=True,
        )
        after = mod.thermostat(
            meta["t_start"], meta["t_low"], meta["t_high"], meta["tout0"],
            meta["tout_event"], meta["heat"], meta["k"], meta["steps"], meta["dt"],
            after=True,
        )
        both = (meta["switches_on"], meta["switches_off"], tuple(meta["on_steps"]), tuple(meta["off_steps"]))
        assert both not in {
            (single[3], single[4], tuple(single[5]), tuple(single[6])),
            (after[3], after[4], tuple(after[5]), tuple(after[6])),
        }


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
    """The published check is a Python replay of the stated scheme."""
    _needs_browser()
    for task in _tasks(seed):
        spec = visual.spec_of(task)
        result, _ = visual._cached_render(task.oracle["golden_html"], spec.steps, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        probe = browser.probe_of(result, stepped=True)
        meta = task.oracle["meta"]
        if task.id == "K-24":
            _vc, v1, i1, v3, i3 = mod.rc_step(meta["vin"], meta["r"], meta["c"], meta["steps"], meta["dt"])
            expected = {"v1": v1, "v3": v3, "i1": i1, "i3": i3}
        else:
            T, _on, heating, on_count, off_count, on_steps, off_steps, outside, *_ = mod.thermostat(
                meta["t_start"], meta["t_low"], meta["t_high"], meta["tout0"],
                meta["tout_event"], meta["heat"], meta["k"], meta["steps"], meta["dt"],
            )
            expected = {
                "temperature": T, "heating_steps": heating, "switches_on": on_count,
                "switches_off": off_count, "outside": outside,
            }
            assert probe.get("on_steps") == on_steps
            assert probe.get("off_steps") == off_steps
            assert probe.get("on") is _on
        for key, value in expected.items():
            actual = probe.get(key)
            assert isinstance(actual, (int, float)), f"{task.id}: probe.{key} = {actual!r}"
            assert abs(actual - mod.round4(value)) <= 1e-3, (
                f"{task.id}: probe.{key} = {actual}, схема даёт {mod.round4(value)}"
            )


# --------------------------------------------------------------------------- #
# Mutants: each wrong program must lose at least the ``simulation`` point.
# --------------------------------------------------------------------------- #


def _score_html(task: core.TaskInstance, html: str) -> core.Score:
    return visual.evaluate(task, _transcript(task, _fenced(html)))


@pytest.mark.parametrize(
    "mutation",
    (
        "constant",
        "polarity",
        "exact",
    ),
)
def test_rc_mutants_lose_a_point(mutation: str) -> None:
    _needs_browser()
    task = _tasks(7)[0]
    html = task.oracle["golden_html"]
    if mutation == "constant":
        broken = html.replace(EULER, "  clock += dt;\n")
    elif mutation == "polarity":
        broken = html.replace("  vc += dt *", "  vc -= dt *")
    else:
        broken = html.replace(
            EULER,
            "  clock += dt;\n  vc = VIN * (1 - Math.exp(-clock / (R * C)));\n",
        )
    assert broken != html, f"мутация {mutation} не применилась"
    score = _score_html(task, broken)
    assert score.earned < score.possible, f"мутант {mutation} набрал {score.earned}"
    checks = {check.name: check.passed for check in score.checks}
    assert checks["scene"] is True, "мутант всё ещё обязан рисовать сцену"
    assert checks["simulation"] is False, "мутант обязан провалить сверку схемы"


@pytest.mark.parametrize("mutation", ("single", "after"))
def test_thermostat_mutants_lose_a_point(mutation: str) -> None:
    _needs_browser()
    task = _tasks(7)[1]
    html = task.oracle["golden_html"]
    if mutation == "single":
        broken = html.replace("  } else if (T >= THIGH) {", "  } else {")
    else:
        broken = html.replace(
            HYSTERESIS + HEAT_STEP,
            HEAT_STEP + HYSTERESIS,
        )
    assert broken != html, f"мутация {mutation} не применилась"
    score = _score_html(task, broken)
    assert score.earned < score.possible, f"мутант {mutation} набрал {score.earned}"
    checks = {check.name: check.passed for check in score.checks}
    assert checks["scene"] is True, "мутант всё ещё обязан рисовать сцену"
    assert checks["simulation"] is False, "мутант обязан провалить сверку схемы"
