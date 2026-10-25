"""Tests for the K-33..K-34 slice: a Neumann heat plate and an Atwood machine.

Three things are proved here rather than assumed:

* the slice is well-formed — ids, profile, points, determinism, fixtures that
  move with the seed, and constants the model can read off the prompt;
* the oracle is a real one: the heat replay keeps the total energy exactly (a
  mirrored border conserves it, a clamped one and a mass loss do not), and the
  Atwood acceleration matches the closed form only with the inertia term in it;
* in the real browser every reference page scores 4/4, the probe taken *before*
  the steps fails the ``simulation`` predicates, and the named page mutants fail
  them too while the golden page passes.

The browser is optional: without Playwright the render-dependent tests skip.
"""

from __future__ import annotations

import json
import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_pde as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them.
SEEDS = (7, 2, 13, 0, 5)

#: Meta keys holding fixture inputs the answer has to hard-code.
FIXTURE_KEYS = {
    "K-33": ("alpha", "hot1", "hot2"),
    "K-34": ("m1", "m2", "J", "R", "x1_0", "x2_0"),
}

#: Page mutations of the published schemes: (id, old, new, label).
MUTANTS = (
    (
        "K-33",
        """  const ii = i < 0 ? 0 : (i > N - 1 ? N - 1 : i);
  const jj = j < 0 ? 0 : (j > N - 1 ? N - 1 : j);
  return T[ii * N + jj];""",
        """  if (i < 0 || j < 0 || i > N - 1 || j > N - 1) return 0;
  return T[i * N + j];""",
        "край Дирихле вместо нулевого потока",
    ),
    (
        "K-33",
        "const swap = T; T = next; next = swap;",
        "for (let k = 0; k < N * N; k++) next[k] *= 0.9995;\n  const swap = T; T = next; next = swap;",
        "потеря массы",
    ),
    (
        "K-34",
        "M1 + M2 + J / (R * R)",
        "M1 + M2",
        "шкив без инерции",
    ),
    (
        "K-34",
        "(M1 - M2)",
        "(M2 - M1)",
        "перевёрнутый знак разности масс",
    ),
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


def _task(task_id: str, seed: int = 7) -> core.TaskInstance:
    return next(task for task in _tasks(seed) if task.id == task_id)


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("K-33", "K-34")
    assert mod.IDS_BY_PROFILE == {"stretch-v1": mod.IDS}
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "K"
        assert task.max_points == visual.POINTS_PER_CHECK * len(visual.CHECK_NAMES)
        spec = visual.spec_of(task)
        assert spec.scene and spec.simulation and spec.image
        assert spec.steps > 0 and spec.dt > 0
        assert spec.need_same_frame
        assert task.prompt.endswith(visual.HTML_CONTRACT)
        golden = task.oracle["golden_html"]
        assert golden.startswith("<!DOCTYPE html>")
        assert "./vendor/three.module.js" in golden
    # The plate is seen at an angle, the machine through an orthographic front
    # view: the pan columns must not depend on depth.
    assert "PerspectiveCamera" in tasks[0].oracle["golden_html"]
    assert "OrthographicCamera" in tasks[1].oracle["golden_html"]
    assert "OrthographicCamera" in tasks[1].prompt


def test_build_is_deterministic() -> None:
    first, second = _tasks(11), _tasks(11)
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
            json.dumps(_tasks(seed)[index].oracle["meta"], sort_keys=True) for seed in range(6)
        }
        assert len(drawn) == 6, f"{mod.IDS[index]}: фикстуры не зависят от seed"


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


def test_the_heat_oracle_conserves_energy() -> None:
    """The mirrored border is the invariant the scenario is built on."""
    for seed in SEEDS:
        task = _tasks(seed)[0]
        meta = task.oracle["meta"]
        initial = meta["hot1"] + meta["hot2"]
        assert abs(meta["energy"] - initial) <= 1e-6, "нулевой поток обязан сохранять энергию"


def test_a_clamped_border_and_a_mass_loss_leak_energy() -> None:
    """The two published defects are visible in the invariant itself."""
    for seed in SEEDS:
        task = _tasks(seed)[0]
        meta = task.oracle["meta"]
        good = mod.heat_state(meta["alpha"], meta["steps"], meta["hot1"], meta["hot2"])
        initial = meta["hot1"] + meta["hot2"]

        clamped = mod.heat_state(meta["alpha"], meta["steps"], meta["hot1"], meta["hot2"], clamp=True)
        assert sum(clamped) < sum(good) - 1.0, "край Дирихле обязан уносить тепло"
        assert sum(clamped) < initial - 1.0

        leaky = [value * 0.9995 ** meta["steps"] for value in good]
        assert sum(leaky) < initial - 1.0, "потеря массы обязана уносить тепло"

        # The leaked energy must land outside the published tolerance.
        assert abs(sum(leaky) - initial) > 1e-3


def test_the_stepped_heat_probe_passes_the_published_checks() -> None:
    for seed in SEEDS:
        task = _tasks(seed)[0]
        meta = task.oracle["meta"]
        grid = mod.heat_state(meta["alpha"], meta["steps"], meta["hot1"], meta["hot2"])
        probe = _heat_probe(grid, meta)
        passed, detail = visual._predicates_report(visual.spec_of(task).simulation, probe)
        assert passed, f"K-33 seed={seed}: верный probe не проходит ({detail})"


def test_clamped_and_leaky_probes_are_rejected() -> None:
    for seed in SEEDS:
        task = _tasks(seed)[0]
        meta = task.oracle["meta"]
        spec = visual.spec_of(task)
        clamped = mod.heat_state(
            meta["alpha"], meta["steps"], meta["hot1"], meta["hot2"], clamp=True
        )
        leaky_grid = [
            value * 0.9995 ** meta["steps"]
            for value in mod.heat_state(meta["alpha"], meta["steps"], meta["hot1"], meta["hot2"])
        ]
        for label, grid in (("Дирихле", clamped), ("потеря массы", leaky_grid)):
            passed, detail = visual._predicates_report(spec.simulation, _heat_probe(grid, meta))
            assert not passed, f"K-33 seed={seed}: мутант «{label}» прошёл ({detail})"


def _heat_probe(grid: list[float], meta: dict) -> dict:
    size = mod.SIZE
    samples = [grid[i * size + j] for i, j in mod.SAMPLES]
    total = sum(grid)
    return {
        "grid": size,
        "steps": meta["steps"],
        "t": mod.round4(meta["steps"] / 60),
        "mean": mod.round4(total / (size * size)),
        "max": mod.round4(max(grid)),
        "energy": mod.round4(total),
        "samples": [mod.round4(value) for value in samples],
    }


def test_the_atwood_oracle_needs_the_inertia_term() -> None:
    for seed in SEEDS:
        task = _tasks(seed)[1]
        meta = task.oracle["meta"]
        acc, _v, _x1, _x2 = mod.atwood_state(
            meta["m1"], meta["m2"], meta["J"], meta["R"], meta["x1_0"], meta["x2_0"], 0
        )
        naive = (meta["m1"] - meta["m2"]) * mod.GRAVITY / (meta["m1"] + meta["m2"])
        assert abs(acc - naive) > 0.5, "шкив без инерции обязан разойтись с формулой"
        assert meta["a"] == mod.round4(acc)


def test_wrong_mass_difference_and_missing_inertia_are_rejected() -> None:
    for seed in SEEDS:
        task = _tasks(seed)[1]
        meta = task.oracle["meta"]
        spec = visual.spec_of(task)
        good = _atwood_probe(meta)
        passed, detail = visual._predicates_report(spec.simulation, good)
        assert passed, f"K-34 seed={seed}: верный probe не проходит ({detail})"

        no_inertia = _atwood_probe(meta, inertia=0.0)
        signed = _atwood_probe(meta, sign=-1.0)
        for label, probe in (("без инерции", no_inertia), ("знак", signed)):
            passed, detail = visual._predicates_report(spec.simulation, probe)
            assert not passed, f"K-34 seed={seed}: мутант «{label}» прошёл ({detail})"


def _atwood_probe(meta: dict, *, inertia: float | None = None, sign: float = 1.0) -> dict:
    j = meta["J"] if inertia is None else inertia
    acc = sign * (meta["m1"] - meta["m2"]) * mod.GRAVITY / (
        meta["m1"] + meta["m2"] + j / (meta["R"] * meta["R"])
    )
    t = meta["steps"] / 60
    return {
        "steps": meta["steps"],
        "t": mod.round4(t),
        "x1": mod.round4(meta["x1_0"] + acc * t * t / 2),
        "x2": mod.round4(meta["x2_0"] - acc * t * t / 2),
        "v": mod.round4(acc * t),
        "a": mod.round4(acc),
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
    """The check must be about the stepping, not about the scene."""
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
        assert passed, f"{task.id}: схема из промпта и эталон разошлись ({detail})"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task_id,old,new,label", MUTANTS)
def test_page_mutants_fail_the_simulation(
    seed: int, task_id: str, old: str, new: str, label: str
) -> None:
    _needs_browser()
    task = _task(task_id, seed)
    html = task.oracle["golden_html"]
    assert html.count(old) == 1, f"{task_id}: в эталоне нет ровно одного {old!r}"
    spec = visual.spec_of(task)
    result = _render(task, html.replace(old, new))
    passed, detail = visual._predicates_report(
        spec.simulation, browser.probe_of(result, stepped=True)
    )
    assert not passed, f"{task_id} seed={seed}: мутант «{label}» прошёл проверку ({detail})"
