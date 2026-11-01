"""Slice K-7..K-11 — particles, agents and flows.

The test renders each reference page directly (through the same evaluator the
benchmark uses), not through the category registry, so it stays honest while
sibling slices are still being written.  A browser that cannot run is a
``skip``: "not checked" is not a zero.
"""

from __future__ import annotations

import random

import pytest

from dailybench import browser, core, visual
from dailybench.tasks import visual_particles as module

IDS = module.IDS

_usable, _reason = browser.availability()
requires_browser = pytest.mark.skipif(not _usable, reason=f"браузер недоступен: {_reason}")


def _tasks(seed: int = 0) -> list[core.TaskInstance]:
    return module.build(random.Random(seed))


def _fenced(html: str) -> str:
    return f"```html\n{html.strip()}\n```"


def _score(task: core.TaskInstance, text: str) -> core.Score:
    transcript = core.Transcript(
        task_id=task.id,
        category=task.category,
        turns=[core.Turn(messages=[], reply=core.ModelReply(text=text))],
    )
    return visual.evaluate(task, transcript)


# --------------------------------------------------------------------------- #
# Shape of the slice
# --------------------------------------------------------------------------- #


def test_ids_titles_and_points():
    tasks = _tasks()
    assert tuple(task.id for task in tasks) == IDS
    assert [task.category for task in tasks] == ["K"] * len(IDS)
    assert [task.max_points for task in tasks] == [4.0] * len(IDS)
    assert len({task.title for task in tasks}) == len(IDS)
    for task in tasks:
        assert task.meta.get("driver") == "visual"
        assert task.oracle["golden_html"].strip()
        assert task.oracle["reference"] == task.oracle["final_answer"]


def test_every_task_carries_four_checks_summing_to_max_points():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")
    for task in _tasks():
        score = _score(task, "не знаю")
        assert [check.name for check in score.checks] == list(visual.CHECK_NAMES)
        assert score.possible == task.max_points == 4.0


def test_prompt_states_the_probe_the_step_and_the_lcg():
    for task in _tasks():
        prompt = task.prompt
        assert "__probe()" in prompt, f"{task.id}: в промпте нет __probe()"
        assert "__step(dt)" in prompt, f"{task.id}: в промпте нет __step(dt)"
        assert "1664525" in prompt and "1013904223" in prompt and "4294967296" in prompt, \
            f"{task.id}: в промпте нет констант ЛЦГ"


# --------------------------------------------------------------------------- #
# The fixture is deterministic and seed-sensitive
# --------------------------------------------------------------------------- #

#: The meta key that holds the numbers a scenario actually simulates.
SIMULATED_KEY = {"K-7": "first3", "K-8": "positions", "K-9": "positions",
                 "K-10": "balls", "K-11": "positions"}


def test_build_is_deterministic_for_one_seed():
    first = _tasks(7)
    second = _tasks(7)
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle["golden_html"] for task in first] == [task.oracle["golden_html"] for task in second]
    assert [task.oracle["meta"] for task in first] == [task.oracle["meta"] for task in second]


def test_another_seed_changes_every_scenario():
    base = {task.id: task for task in _tasks(7)}
    other = {task.id: task for task in _tasks(8)}
    for task_id in IDS:
        assert base[task_id].prompt != other[task_id].prompt, f"{task_id}: промпт не зависит от seed"
        assert base[task_id].oracle["golden_html"] != other[task_id].oracle["golden_html"], \
            f"{task_id}: эталон не зависит от seed"


def test_another_seed_moves_the_simulated_numbers():
    base = {task.id: task.oracle["meta"] for task in _tasks(3)}
    other = {task.id: task.oracle["meta"] for task in _tasks(4)}
    for task_id in IDS:
        key = SIMULATED_KEY[task_id]
        assert base[task_id][key] != other[task_id][key], f"{task_id}.{key}: числа не меняются от seed"


# --------------------------------------------------------------------------- #
# The reference page is a real answer; garbage earns nothing
# --------------------------------------------------------------------------- #


@requires_browser
@pytest.mark.parametrize("task_id", IDS)
def test_golden_scores_four_out_of_four(task_id):
    task = next(t for t in _tasks() if t.id == task_id)
    score = _score(task, _fenced(task.oracle["golden_html"]))
    details = "; ".join(f"{c.name}:{'ok' if c.passed else 'FAIL ' + c.detail}" for c in score.checks)
    assert score.earned == score.possible == 4.0, f"{task_id}: {details}"


@requires_browser
@pytest.mark.parametrize("task_id", IDS)
def test_garbage_scores_zero(task_id):
    task = next(t for t in _tasks() if t.id == task_id)
    assert _score(task, "не знаю").earned == 0.0


@requires_browser
def test_a_page_that_never_steps_fails_the_simulation_check():
    """A page that renders but does not advance must not pass ``simulation``."""
    task = next(t for t in _tasks() if t.id == "K-8")
    broken = task.oracle["golden_html"].replace("  steps += 1;\n", "")
    assert broken != task.oracle["golden_html"], "эталон K-8 обязан продвигать steps"
    checks = {check.name: check.passed for check in _score(task, _fenced(broken)).checks}
    assert checks["html"] is True, "сломанная страница всё ещё рисует кадр"
    assert checks["simulation"] is False, "замороженная симуляция обязана провалить сверку"
