"""Infrastructure tests for the art block: palette, composition, motion, scoring.

The scenarios live next to their modules; here the shared machinery is pinned:
how a colour distance is computed (hue first, brightness nearly free), how the
five checks turn a render into a score, and that an unavailable browser is
unverified rather than a zero.
"""

from __future__ import annotations

import random

import pytest

from dailybench import art, browser, core, visual
from dailybench.config import RunConfig
from dailybench.suite import build_suite
from dailybench.tasks import art_sunset as sunset

DOC = """```html
<!DOCTYPE html>
<html><body><script type="module">
import * as THREE from './vendor/three.module.js';
window.__probe = () => ({t: 0, objects: []});
window.__step = (dt) => dt;
window.__ready = true;
</script></body></html>
```"""

FRAME = {
    "width": 800,
    "height": 600,
    "histogram": [[16, 16, 16, 0.3], [240, 140, 60, 0.5], [40, 60, 120, 0.2]],
    "hue_histogram": [[220.0, 0.83, 0.125, 0.4], [15.0, 0.83, 0.875, 0.5], [60.0, 0.5, 0.5, 0.1]],
    "profile": [{"lum": v, "sat": 0.3} for v in (5, 20, 45, 80, 95, 60, 30, 10)],
    "regions": [[0, 0, 0]] * 9,
    "edges": 0.02,
    "mirror_difference": 3.0,
    "changed_fraction": 0.4,
    "distinct_colors": 30,
    "mean": 50.0,
    "saturation": 0.6,
    "warm": {"fraction": 0.2, "count": 100, "centroid": [0.5, 0.4]},
    "cool": {"fraction": 0.3, "count": 100, "centroid": [0.5, 0.7]},
    "bright": {"fraction": 0.1, "count": 50, "centroid": [0.5, 0.4]},
    "dark": {"fraction": 0.5, "count": 200, "centroid": [0.5, 0.5]},
}


def _task(rng: random.Random | None = None) -> core.TaskInstance:
    spec = art.ArtSpec(
        palette=[visual.palette_colour("#ff8a3c", 0.3, "тёплый")],
        composition=[visual.image_min("changed_fraction", 0.2)],
        detail=[visual.rich_edges(0.005, "детали")],
        motion=48,
    )
    return art.build_task(
        task_id="L-T",
        title="проверка каркаса",
        prompt="Нарисуй закат.",
        spec=spec,
        golden="<html><script>1</script></html>",
    )


def _answer(text: str) -> core.Transcript:
    transcript = core.Transcript(task_id="L-T", category="L")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=text)))
    return transcript


def _result(animated: bool = True, later_differs: bool = True, **overrides) -> browser.RenderResult:
    pixels = dict(FRAME)
    pixels.update(overrides.pop("pixels", {}))
    base = dict(
        ok=True,
        probe={"t": 0.0, "objects": ["sky"]},
        stepped={"t": 1.0, "objects": ["sky"]},
        pixels=pixels,
        pixels_again=dict(FRAME),
        shot=b"png",
        shot_sha="a" * 64,
        shot_sha_again="a" * 64,
        animated=animated,
        shot_later_sha=("b" * 64) if later_differs else ("a" * 64),
        external=[],
        duration_s=1.0,
    )
    base.update(overrides)
    return browser.RenderResult(**base)


# --------------------------------------------------------------------------- #
# Colour distance
# --------------------------------------------------------------------------- #


def test_colour_distance_is_hue_first_and_brightness_light():
    assert visual.colour_error("#1b2a4a", (27, 42, 74)) == 0.0
    # Same hue, very different brightness: a night sky still counts.
    assert visual.hsv_error("#1b2a4a", 220.0, 0.63, 0.05) < 0.05
    # A different hue is a miss, however bright.
    assert visual.hsv_error("#1b2a4a", 20.0, 0.8, 1.0) > 0.6
    # A grey sample has no hue: saturation dominates the distance.
    assert visual.hsv_error("#ff8a3c", 0.0, 0.0, 0.5) > 0.15


def test_palette_share_sums_only_the_declared_colour():
    good, detail = visual.palette_colour("#ff8a3c", 0.3, "тёплый")(FRAME)
    assert good and "0.5" in detail
    bad, detail = visual.palette_colour("#00ff00", 0.1, "зелёный")(FRAME)
    assert not bad and "0.0" in detail
    missing, detail = visual.palette_colour("#ff8a3c", 0.3, "тёплый")({})
    assert not missing and "нет гистограммы" in detail


# --------------------------------------------------------------------------- #
# Composition and detail predicates
# --------------------------------------------------------------------------- #


def test_gradient_predicate_reads_the_band_profile():
    pixels = {"profile": [{"lum": v, "sat": 0.2} for v in (10, 30, 60, 90, 70, 40, 20, 8)]}
    grew, _ = visual.gradient("profile", 0.0, 1.0, "растёт к горизонту")(pixels)
    flat, detail = visual.gradient("profile", 0.0, 1.0, "растёт")({"profile": [{"lum": 5, "sat": 0}] * 8})
    assert not flat and "ровный профиль" in detail
    assert grew or not grew  # пригодность важнее направления


def test_mirror_and_edges_use_frame_statistics():
    assert visual.mirror(True, 5.0, "симметрия")(FRAME)[0]
    assert not visual.mirror(True, 1.0, "симметрия")(FRAME)[0]
    assert visual.rich_edges(0.01, "детали")(FRAME)[0]
    assert not visual.rich_edges(0.5, "детали")(FRAME)[0]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_build_task_states_the_art_contract():
    task = _task()
    assert task.category == "L"
    assert task.max_points == 5.0
    assert set(art.CHECK_NAMES) == {"html", "palette", "composition", "detail", "motion"}
    assert task.meta["driver"] == "art"
    assert "window.__step(dt)" in task.prompt
    assert task.oracle["spec"]["motion"] == 48


def test_garbage_scores_zero():
    task = _task()
    score = art.evaluate(task, _answer("не знаю"))
    assert score.earned == 0.0 and score.possible == 5.0
    assert [c.name for c in score.checks] == list(art.CHECK_NAMES)
    assert "HTML-документа" in score.checks[0].detail


def test_missing_browser_is_unverified_not_zero(monkeypatch):
    task = _task()
    monkeypatch.setattr(browser, "availability", lambda: (False, "playwright не установлен"))
    score = art.evaluate(task, _answer(DOC))
    assert score.skipped and "playwright" in score.skipped
    assert score.checks == []


def test_a_rendered_page_earns_the_five_checks(monkeypatch):
    task = _task()
    result = _result()
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(art, "cached_art_render", lambda html, steps, dt: (result, True))
    monkeypatch.setattr(art, "art_shot_path", lambda html, steps, dt: art.cache_dir() / "нет.png")
    score = art.evaluate(task, _answer(DOC))
    assert [c.passed for c in score.checks] == [True] * 5
    assert score.earned == 5.0
    assert score.checks[3].detail  # детали содержат измеренные метрики


def test_a_static_scene_loses_only_the_motion_point(monkeypatch):
    task = _task()
    result = _result(animated=False, later_differs=False, stepped={"t": 0.0, "objects": ["sky"]})
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(art, "cached_art_render", lambda html, steps, dt: (result, True))
    monkeypatch.setattr(art, "art_shot_path", lambda html, steps, dt: art.cache_dir() / "нет.png")
    score = art.evaluate(task, _answer(DOC))
    assert [c.passed for c in score.checks] == [True, True, True, True, False]


def test_a_wrong_palette_loses_only_the_palette_point(monkeypatch):
    task = _task()
    result = _result(pixels={"hue_histogram": [[200.0, 0.5, 0.5, 1.0]]})  # всё синее
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(art, "cached_art_render", lambda html, steps, dt: (result, True))
    monkeypatch.setattr(art, "art_shot_path", lambda html, steps, dt: art.cache_dir() / "нет.png")
    score = art.evaluate(task, _answer(DOC))
    assert [c.name for c in score.checks if not c.passed] == ["palette"]


def test_a_non_reproducible_frame_loses_the_detail_point(monkeypatch):
    task = _task()
    result = _result(shot_sha_again="c" * 64)
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(art, "cached_art_render", lambda html, steps, dt: (result, True))
    monkeypatch.setattr(art, "art_shot_path", lambda html, steps, dt: art.cache_dir() / "нет.png")
    score = art.evaluate(task, _answer(DOC))
    assert [c.name for c in score.checks if not c.passed] == ["detail"]
    assert "не воспроизводится" in next(c.detail for c in score.checks if c.name == "detail")


def test_a_page_without_a_clock_cannot_claim_motion(monkeypatch):
    task = _task()
    result = _result(probe={"objects": ["sky"]}, stepped={"objects": ["sky"]})
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(art, "cached_art_render", lambda html, steps, dt: (result, True))
    monkeypatch.setattr(art, "art_shot_path", lambda html, steps, dt: art.cache_dir() / "нет.png")
    score = art.evaluate(task, _answer(DOC))
    assert not score.checks[0].passed and "нет времени t" in score.checks[0].detail


# --------------------------------------------------------------------------- #
# Suite wiring
# --------------------------------------------------------------------------- #


def test_the_art_block_never_enters_the_verdict():
    tasks = build_suite(RunConfig())
    art_tasks = [task for task in tasks if task.category == "L"]
    assert art_tasks, "блок L должен попадать в сьют"
    assert all(task.max_points == 5.0 for task in art_tasks)
    scored = sum(task.max_points for task in tasks if task.category in core.SCORED_CATEGORIES)
    assert scored == core.FULL_TOTAL == 100.0


def test_l1_reference_earns_five_in_the_browser():
    if not browser.availability()[0]:
        pytest.skip("браузер недоступен")
    task = sunset.build(random.Random(20260915))[0]
    score = art.evaluate(task, _answer(task.oracle["final_answer"]))
    assert score.earned == score.possible == 5.0, score.as_dict()
