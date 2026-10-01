"""Infrastructure tests for the visual block: extraction, predicates, scoring.

The scenarios themselves are tested next to their modules
(``tests/test_visual_<slice>.py``); here the shared machinery is pinned: what
counts as a document, how a probe payload is read, how the four checks are
turned into a score, and what happens when the browser is missing (unverified,
never zero).
"""

from __future__ import annotations

import random

import pytest

from dailybench import browser, core, visual

FENCE_DOC = """Вот решение:

```html
<!DOCTYPE html>
<html><body><script type="module">
import * as THREE from './vendor/three.module.js';
window.__probe = () => ({ok: true});
window.__step = (dt) => dt;
window.__ready = true;
</script></body></html>
```

Готово.
"""


def test_extract_html_prefers_the_fenced_document():
    html = visual.extract_html(FENCE_DOC)
    assert html.startswith("<!DOCTYPE html>")
    assert "three.module.js" in html
    assert "Вот решение" not in html


def test_extract_html_takes_a_bare_document():
    bare = "<!DOCTYPE html>\n<html><body><script>1</script></body></html>"
    assert visual.extract_html(bare) == bare
    assert visual.extract_html("   <html><script>x</script></html>  ").startswith("<html>")


def test_extract_html_keeps_the_largest_block_and_ignores_prose():
    two = "```html\n<html><script>a</script></html>\n```\n```html\n<html><script>b</script><p>x</p></html>\n```"
    assert "b" in visual.extract_html(two)
    assert visual.extract_html("не знаю") == ""
    assert visual.extract_html("```python\nprint(1)\n```") == ""


def test_looks_like_document_rejects_shortcuts():
    ok, _ = visual.looks_like_document(visual.extract_html(FENCE_DOC))
    assert ok
    bad, why = visual.looks_like_document("<script>1</script>")
    assert not bad and "html" in why
    bad, why = visual.looks_like_document("<html><script>1</script></html>")
    assert not bad and "three.module.js" in why
    bad, why = visual.looks_like_document(
        '<html><script src="https://cdn.example/three.module.js"></script></html>'
    )
    assert not bad and "внешние ресурсы" in why


def test_path_of_walks_dicts_and_lists():
    payload = {"camera": [1, 2, 3], "scene": {"balls": [{"x": 5}]}}
    assert visual.path_of(payload, "camera.1") == 2
    assert visual.path_of(payload, "scene.balls.0.x") == 5
    assert visual.path_of(payload, "scene.balls.-1.x") == 5
    assert visual.path_of(payload, "camera.9", "нет") == "нет"
    assert visual.path_of(payload, "scene.nope", None) is None


def test_predicates_read_the_probe_and_report_the_reason():
    payload = {
        "objects": ["ground", "ball"],
        "balls": [{"x": 1.0, "y": 2.0}],
        "camera": [0.0, 6.0, 14.0],
    }
    assert visual.near("balls.0.x", 1.0, 1e-3)(payload)[0]
    assert not visual.near("balls.0.x", 1.0005, 1e-6)(payload)[0]
    assert visual.between("balls.0.y", 1.0, 3.0)(payload)[0]
    assert not visual.between("balls.0.y", 3.0, 4.0)(payload)[0]
    assert visual.contains_all("objects", ["ground"])(payload)[0]
    assert not visual.contains_all("objects", ["missing"])(payload)[0]
    assert visual.size_between("objects", 2, 4)(payload)[0]
    assert not visual.size_between("objects", 3, 4)(payload)[0]
    good, detail = visual.near("balls.0.x", 9.0)(payload)
    assert not good and "balls.0.x = 1.0" in detail


def test_predicates_on_a_malformed_probe_fail_instead_of_raising():
    report = visual._predicates_report([visual.near("camera.0", 1.0)], {"camera": [None, 6, 14]})
    assert report == (False, "- camera.0: не число (None)")
    wrong_type = visual._predicates_report([visual.near("camera.0", 1.0)], {"camera": "нет"})
    assert wrong_type == (False, "- camera.0: в probe нет такого поля")
    missing = visual._predicates_report([visual.near("gyro.x", 1.0)], {"camera": []})
    assert missing == (False, "- gyro.x: в probe нет такого поля")
    boolean = visual._predicates_report([visual.between("flag", 0, 1)], {"flag": True})
    assert boolean == (False, "- flag: не число (True)")


def test_image_predicates_use_frame_statistics():
    pixels = {
        "changed_fraction": 0.51,
        "distinct_colors": 21,
        "warm": {"fraction": 0.0082, "count": 700, "centroid": [0.6, 0.24]},
        "regions": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [44, 53, 65]] * 1
        + [[44, 53, 65], [44, 53, 65]],
    }
    assert visual.image_min("changed_fraction", 0.15)(pixels)[0]
    assert not visual.image_min("changed_fraction", 0.9)(pixels)[0]
    assert visual.image_color("warm", 0.002, "видны объекты")(pixels)[0]
    assert not visual.image_color("warm", 0.05, "видны объекты")(pixels)[0]
    assert visual.image_centroid("warm", 1, 0.0, 0.8, "в кадре")(pixels)[0]
    assert not visual.image_centroid("warm", 1, 0.9, 1.0, "в кадре")(pixels)[0]
    assert visual.image_region(8, visual.cool, "низ кадра")(pixels)[0]
    assert not visual.image_region(0, visual.warm, "верх")(pixels)[0]


def _task(rng: random.Random) -> core.TaskInstance:
    spec = visual.VisualSpec(
        scene=[visual.contains_all("objects", ["ground"])],
        simulation=[visual.near("t", 1.0, 1e-3)],
        image=[visual.image_min("changed_fraction", 0.1)],
        steps=60,
        dt=1 / 60,
    )
    return visual.build_task(
        task_id="K-T",
        title="проверка каркаса",
        prompt="Собери сцену.",
        spec=spec,
        golden="<html><script>1</script></html>",
    )


def test_build_task_keeps_the_contract_of_a_visual_scenario():
    task = _task(random.Random(1))
    assert task.category == "K"
    assert task.max_points == 4.0
    assert task.oracle["spec"]["steps"] == 60
    assert task.oracle["reference"].startswith("```html")
    assert task.prompt.startswith("Собери сцену.")
    assert "window.__probe()" in task.prompt  # общий контракт дописан каркасом
    assert task.meta["driver"] == "visual"


def test_garbage_answer_scores_zero_without_touching_the_browser(monkeypatch):
    task = _task(random.Random(1))
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(
        core.Turn(messages=[], reply=core.ModelReply(text="не знаю"))
    )
    score = visual.evaluate(task, transcript)
    assert score.earned == 0.0
    assert score.possible == 4.0
    assert [c.name for c in score.checks] == list(visual.CHECK_NAMES)
    assert "HTML-документа" in score.checks[0].detail


def test_missing_browser_is_unverified_not_zero(monkeypatch):
    task = _task(random.Random(1))
    monkeypatch.setattr(browser, "availability", lambda: (False, "playwright не установлен"))
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=FENCE_DOC)))
    score = visual.evaluate(task, transcript)
    assert score.skipped and "playwright" in score.skipped
    assert score.earned == 0.0
    assert score.checks == [], "непроверенное задание не должно выглядеть как проваленные проверки"


def test_a_broken_page_loses_all_four_checks(monkeypatch):
    task = _task(random.Random(1))
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(browser, "render", lambda *a, **kw: browser.RenderResult(ok=False, reason="страница не отрисовалась"))
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=FENCE_DOC)))
    score = visual.evaluate(task, transcript)
    assert [c.passed for c in score.checks] == [False, False, False, False]
    assert "рендер не удался" in score.checks[0].detail


def test_an_external_request_is_reported_as_a_failure(monkeypatch):
    task = _task(random.Random(1))
    result = browser.RenderResult(
        ok=True,
        external=["https://cdn.example/three.module.js"],
        probe={"objects": ["ground"], "t": 1.0},
        stepped={"objects": ["ground"], "t": 1.0},
        pixels={"width": 800, "height": 600, "changed_fraction": 0.5, "warm": {"fraction": 0.0, "count": 0, "centroid": None}, "regions": [[0, 0, 0]] * 9},
        shot=b"png",
        shot_sha="a",
        shot_sha_again="a",
    )
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(browser, "render", lambda *a, **kw: result)
    monkeypatch.setattr(visual, "_cached_render", lambda html, steps, dt: (result, True))
    monkeypatch.setattr(visual, "shot_path", lambda html, steps, dt: visual.cache_dir() / "нет.png")
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=FENCE_DOC)))
    score = visual.evaluate(task, transcript)
    assert not score.checks[0].passed
    assert "внешние запросы" in score.checks[0].detail
    assert score.checks[1].passed and score.checks[2].passed and score.checks[3].passed
    assert transcript.extra["visual"]["same_frame"] is True


def test_render_without_a_document_is_refused():
    result = browser.render("")
    assert not result.ok and "пустой ответ" in result.reason


def test_availability_reports_a_reason_when_usable():
    usable, reason = browser.availability()
    assert usable is True, reason
    assert reason == ""


@pytest.mark.parametrize("name", ["three.module.js", "three.core.js", "PROVENANCE.md"])
def test_the_pinned_library_is_present(name):
    assert (browser.VENDOR / name).is_file()
