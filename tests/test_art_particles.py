"""Slice L-6..L-9 — particles, sky and light (galaxy, fireworks, aurora, night).

The test renders each reference page through the same evaluator the benchmark
uses, so it stays honest while sibling slices are still being written. A browser
that cannot run is a ``skip``: "not checked" is not a zero.
"""

from __future__ import annotations

import random
import re

import pytest

from dailybench import art, browser, core, visual
from dailybench.tasks import art_particles as module

IDS = module.IDS

_usable, _reason = browser.availability()
requires_browser = pytest.mark.skipif(not _usable, reason=f"браузер недоступен: {_reason}")

#: The weakest share measured on the reference pages over ten fixtures, per
#: colour. Every declared threshold sits at half of this, so another seed — the
#: self-check regenerates the suite — cannot fall under its own oracle.
MEASURED = {
    "L-6": {"#ffbf00": 0.0032, "#ff0080": 0.0102, "#0cb7f0": 0.0277, "#1d1638": 0.9514},
    "L-7": {"#ff0051": 0.0185, "#ffc300": 0.0159, "#19cdff": 0.0141, "#280f54": 0.7946},
    "L-8": {"#18f051": 0.0273, "#2197db": 0.2442, "#ff33ee": 0.0847, "#1d1638": 0.8498},
    "L-9": {"#150b38": 0.6506, "#00b1d9": 0.2317, "#f2f6ff": 0.0020, "#ffd699": 0.0167},
}


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
    return art.evaluate(task, transcript)


def _palette_of(task: core.TaskInstance) -> dict[str, float]:
    """Declared colours and shares, read back out of the prompt."""
    found = re.findall(r"^\* (#[0-9a-f]{6}) — .*не меньше ([\d.]+)%", task.prompt, re.M)
    assert found, f"{task.id}: в промпте нет объявленной палитры"
    return {colour: round(float(share) / 100, 6) for colour, share in found}


def _share_of(pixels: dict, hex_value: str) -> float:
    """The framed share of one declared colour, by the evaluator's own metric."""
    total = 0.0
    for hue, sat, val, fraction in pixels.get("hue_histogram") or []:
        if visual.hsv_error(hex_value, hue, sat, val) <= 0.22:
            total += float(fraction)
    return total


# --------------------------------------------------------------------------- #
# Shape of the slice
# --------------------------------------------------------------------------- #


def test_ids_titles_and_points():
    tasks = _tasks()
    assert tuple(task.id for task in tasks) == IDS
    assert [task.category for task in tasks] == ["L"] * len(IDS)
    assert [task.max_points for task in tasks] == [5.0] * len(IDS)
    assert len({task.title for task in tasks}) == len(IDS)
    for task in tasks:
        assert task.meta.get("driver") == "art"
        assert task.oracle["golden_html"].strip()
        assert task.oracle["reference"] == task.oracle["final_answer"]


def test_every_task_carries_five_checks_summing_to_max_points():
    for task in _tasks():
        score = _score(task, "не знаю")
        assert [check.name for check in score.checks] == list(art.CHECK_NAMES)
        assert score.possible == task.max_points == 5.0


def test_prompts_state_the_probe_the_step_and_the_lcg():
    for task in _tasks():
        prompt = task.prompt
        assert "__probe()" in prompt, f"{task.id}: в промпте нет __probe()"
        assert "__step(dt)" in prompt, f"{task.id}: в промпте нет __step(dt)"
        assert "1664525" in prompt and "1013904223" in prompt and "4294967296" in prompt, \
            f"{task.id}: в промпте нет констант ЛЦГ"
        assert "Math.random()" in prompt, f"{task.id}: промпт не запрещает Math.random()"


# --------------------------------------------------------------------------- #
# The fixture is deterministic, seed-sensitive and stated in the prompt
# --------------------------------------------------------------------------- #


#: The meta key that holds this scenario's fixture numbers.
FIXTURE_KEY = {"L-6": "seed", "L-7": "spots", "L-8": "seed", "L-9": "ridge"}


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


def test_another_seed_moves_the_fixture():
    base = {task.id: task.oracle["meta"] for task in _tasks(3)}
    other = {task.id: task.oracle["meta"] for task in _tasks(4)}
    for task_id in IDS:
        key = FIXTURE_KEY[task_id]
        assert base[task_id][key] != other[task_id][key], f"{task_id}.{key}: фикстура не меняется от seed"


def test_fixture_numbers_are_in_the_prompt_verbatim():
    for task in _tasks(5):
        prompt = task.prompt
        meta = task.oracle["meta"]
        assert str(meta["seed"]) in prompt, f"{task.id}: seed не в промпте"
        if task.id == "L-7":
            for x, y in meta["spots"]:
                assert f"({x:.3f}, {y:.3f})" in prompt, f"{task.id}: точка запуска ({x}, {y}) не в промпте"
        if task.id == "L-9":
            for value in meta["ridge"]:
                assert f"{value:.3f}" in prompt, f"{task.id}: высота горы {value} не в промпте"
            for value in meta["moon"]:
                assert f"{value:.3f}" in prompt, f"{task.id}: луна {value} не в промпте"


# --------------------------------------------------------------------------- #
# The palette is declared strictly and with a margin over the measured frame
# --------------------------------------------------------------------------- #


def test_palette_is_declared_with_hue_separation_and_a_margin():
    import itertools

    for task in _tasks(11):
        palette = _palette_of(task)
        assert 3 <= len(palette) <= 4, f"{task.id}: палитра из {len(palette)} цветов"
        for share in palette.values():
            assert 0 < share <= 0.6, f"{task.id}: доля {share} вне разумных границ"
        hexes = list(palette)
        for a, b in itertools.combinations(hexes, 2):
            distance = visual.colour_error(a, visual._hex_to_rgb(b))
            assert distance >= 0.2, f"{task.id}: цвета {a} и {b} неразличимы по тону ({distance:.3f})"
        for colour, share in palette.items():
            measured = MEASURED[task.id][colour]
            assert share <= measured * 0.85, f"{task.id}: порог {colour} {share} без запаса от {measured}"


def test_palette_checks_live_in_the_palette_family():
    for task in _tasks(11):
        spec = art.spec_of(task)
        assert len(spec.palette) == 4, f"{task.id}: проверок палитры {len(spec.palette)}"
        assert len(spec.composition) >= 2
        assert len(spec.detail) >= 3
        assert spec.motion >= 60, "движение обязано быть заметным на кадре"


# --------------------------------------------------------------------------- #
# The reference page is a real answer; garbage earns nothing
# --------------------------------------------------------------------------- #


@requires_browser
@pytest.mark.parametrize("task_id", IDS)
def test_golden_scores_five_out_of_five(task_id):
    task = next(t for t in _tasks() if t.id == task_id)
    score = _score(task, _fenced(task.oracle["golden_html"]))
    details = "; ".join(f"{c.name}:{'ok' if c.passed else 'FAIL ' + c.detail}" for c in score.checks)
    assert score.earned == score.possible == 5.0, f"{task_id}: {details}"


@requires_browser
@pytest.mark.parametrize("task_id", IDS)
def test_garbage_scores_zero(task_id):
    task = next(t for t in _tasks() if t.id == task_id)
    assert _score(task, "не знаю").earned == 0.0


@requires_browser
def test_thresholds_hold_on_another_fixture():
    """Every seed must clear the declared shares, with room to spare.

    A threshold planted exactly on the measured value passes on the default
    fixture and fails on the next one, so this walks several seeds, renders each
    scene exactly where the evaluator measures it, and records the weakest
    measured share: the declared threshold has to sit at or below half of it.
    """
    floor = {task_id: {} for task_id in IDS}
    for seed in (5, 2, 7, 9):
        for task in _tasks(seed):
            spec = art.spec_of(task)
            result = browser.render(task.oracle["golden_html"], steps=spec.motion, mode="time")
            assert result.ok, f"{task.id} seed={seed}: {result.reason}"
            pixels = result.pixels or {}
            for predicate in spec.palette + spec.composition + spec.detail:
                good, detail = predicate(pixels)
                assert good, f"{task.id} seed={seed}: {detail}"
            for colour, share in _palette_of(task).items():
                measured = _share_of(pixels, colour)
                best = floor[task.id].get(colour)
                floor[task.id][colour] = measured if best is None else min(best, measured)
                assert share * 2 <= floor[task.id][colour] + 1e-9, (
                    f"{task.id} seed={seed}: порог {colour} {share} без двойного запаса "
                    f"от измеренного {floor[task.id][colour]:.4f}"
                )


@requires_browser
def test_palette_check_rejects_a_wrong_hue():
    """A page painted in the wrong hue must lose the palette point."""
    task = next(t for t in _tasks() if t.id == "L-6")
    page = task.oracle["golden_html"].replace("'#0cb7f0'", "'#ff0000'")
    assert page != task.oracle["golden_html"], "в эталоне L-6 обязан быть cyan"
    checks = {check.name: check.passed for check in _score(task, _fenced(page)).checks}
    assert checks["palette"] is False, "подмена тона на красный обязана провалить палитру"
    assert checks["html"] is True, "страница с подменённым цветом всё ещё рисует кадр"
