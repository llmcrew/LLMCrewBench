"""Behavioural checks for free-form artwork capture and its frame cache."""

from __future__ import annotations

import json
import random

import pytest

from dailybench import art, browser, core, visual
from dailybench.config import RunConfig
from dailybench.report import summarise
from dailybench.tasks import art as art_block

SVG_PAGE = """<!DOCTYPE html><html><body style="margin:0;background:#182032">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
<defs><linearGradient id="g"><stop stop-color="#eeccaa"/>
<stop offset="1" stop-color="#223366"/></linearGradient></defs>
<circle cx="400" cy="300" r="200" fill="url(#g)"/>
</svg></body></html>"""


def _task() -> core.TaskInstance:
    return art.build_task(task_id="L-T", title="art", prompt="Создай свою работу.", golden=SVG_PAGE)


def _answer(page: str) -> core.Transcript:
    return core.Transcript(
        task_id="L-T", category="L", turns=[core.Turn(messages=[], reply=core.ModelReply(text=page))]
    )


def _frame(**overrides) -> browser.RenderResult:
    values = {"ok": True, "shot": b"captured-frame", "shot_sha": "a" * 64,
              "pixels": None, "probe": None, "animated": False}
    values.update(overrides)
    return browser.RenderResult(**values)


def test_native_static_svg_needs_no_scene_api_or_palette(monkeypatch, tmp_path):
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(browser, "render", lambda html, **kwargs: _frame())
    transcript = _answer(SVG_PAGE)
    score = art.evaluate(_task(), transcript)
    assert score.earned == score.possible == 1.0
    assert transcript.extra["visual"]["capture_mode"] == "live"
    assert art.art_shot_path(SVG_PAGE).read_bytes() == b"captured-frame"


def test_missing_agent_file_cannot_be_replaced_by_chat(monkeypatch, tmp_path):
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    transcript = _answer(SVG_PAGE)
    transcript.extra = {"harness": "agent", "work_dir": str(tmp_path),
                        "artefact": {"file": "index.html", "text": "", "missing": True}}
    score = art.evaluate(_task(), transcript)
    assert score.earned == 0.0 and score.possible == 1.0


def test_unavailable_browser_remains_unverified(monkeypatch):
    monkeypatch.setattr(browser, "availability", lambda: (False, "chromium unavailable"))
    score = art.evaluate(_task(), _answer(SVG_PAGE))
    assert score.skipped and score.possible == 0.0


def test_page_error_preserves_the_captured_work(monkeypatch, tmp_path):
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(browser, "render", lambda html, **kwargs: _frame(ok=False, errors=["pageerror: broken"]))
    transcript = _answer(SVG_PAGE)
    score = art.evaluate(_task(), transcript)
    assert score.earned == 0.0 and score.possible == 1.0
    assert "broken" in score.checks[0].detail
    assert transcript.extra["visual"]["shot"] == str(art.art_shot_path(SVG_PAGE))
    assert art.art_shot_path(SVG_PAGE).read_bytes() == b"captured-frame"


def test_successful_svg_capture_reuses_cache_without_pixels(monkeypatch, tmp_path):
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    captures = []
    def capture(html, **kwargs):
        captures.append(html)
        return _frame()
    monkeypatch.setattr(browser, "render", capture)
    first, cached = art.cached_art_render(SVG_PAGE)
    again, reused = art.cached_art_render(SVG_PAGE)
    # First render is a cache write, second is a hit; exactly one live capture.
    assert cached and reused and len(captures) == 1
    assert first.shot == again.shot == b"captured-frame"
    assert again.pixels is None


def test_failed_capture_is_retried_and_then_reused(monkeypatch, tmp_path):
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    captures = []
    def capture(html, **kwargs):
        captures.append(html)
        return _frame(ok=len(captures) > 1)
    monkeypatch.setattr(browser, "render", capture)
    failed, _ = art.cached_art_render(SVG_PAGE)
    healed, _ = art.cached_art_render(SVG_PAGE)
    reused, hit = art.cached_art_render(SVG_PAGE)
    assert not failed.ok and healed.ok and reused.ok
    assert hit and len(captures) == 2


def test_corrupt_capture_metadata_does_not_lose_the_work(monkeypatch, tmp_path):
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(browser, "render", lambda html, **kwargs: _frame())
    art.cached_art_render(SVG_PAGE)
    metadata = art.art_shot_path(SVG_PAGE).with_suffix(".json")
    metadata.write_text("{broken", encoding="utf-8")
    healed, recovered = art.cached_art_render(SVG_PAGE)
    # Corrupt metadata triggers a fresh capture that heals both frame and blob.
    assert recovered and healed.shot == b"captured-frame"
    assert json.loads(metadata.read_text(encoding="utf-8"))["result"]["ok"]


def test_render_check_does_not_add_artistic_quality_points():
    scores = [{"task_id": "IF-1", "category": "A", "earned": 2.0, "possible": 2.0},
              {"task_id": "L-1", "category": "L", "earned": 1.0, "possible": 1.0}]
    summary = summarise(scores)
    assert summary["headline_earned"] == summary["headline_possible"] == 2.0
    assert summary["showcase_earned"] == summary["showcase_possible"] == 0.0


def test_block_asks_every_topic_once_and_stays_free_form():
    tasks = art_block.generate(random.Random(1), RunConfig())
    assert tuple(task.id for task in tasks) == art_block.IDS_BASELINE
    assert len(tasks) == 29
    assert [task.id for task in tasks] == [f"L-{i}" for i in range(1, 30)]
    # L-1 без темы; остальные называют только сюжет, а не рецепт.
    assert "Идею, сюжет, стиль" in tasks[0].prompt
    assert "тему «неоновая сетка»" in tasks[1].prompt
    assert "Тему" not in tasks[1].prompt.split("\n")[0]
    for task in tasks:
        assert task.max_points == 1.0
        assert task.meta["seed_sensitive"] is False
        assert "palette" not in task.prompt and "__probe" not in task.prompt
        assert task.oracle["golden_html"].lstrip().lower().startswith("<!doctype html")
        assert task.oracle["answer_file"] == "index.html"


def test_goldens_differ_between_topics():
    tasks = art_block.generate(__import__("random").Random(7), RunConfig())
    pages = {task.oracle["golden_html"] for task in tasks}
    assert len(pages) == len(tasks)


STRETCH_IDS = tuple(f"L-{index}" for index in range(30, 40))

STRETCH_TOPICS = (
    "Архив древних карт",
    "Коралловый риф без людей",
    "Заброшенный карьер",
    "Читальный зал после закрытия",
    "Мастерская часовщика",
    "Путь перелётных птиц через водно-болотные угодья",
    "Палеонтологический раскоп",
    "Рынок на рассвете",
    "Детская площадка в масштабе насекомого",
    "Радиотелескоп на пустынном плато",
)


def test_stretch_appends_the_new_themes_to_the_baseline():
    baseline = [task.id for task in art_block.generate(random.Random(2), RunConfig())]
    stretch = [
        task.id
        for task in art_block.generate(random.Random(2), RunConfig(profile="stretch-v1"))
    ]
    assert stretch == list(art_block.IDS_BASELINE) + list(STRETCH_IDS)
    assert baseline == list(art_block.IDS_BASELINE)
    assert art_block.IDS_BY_PROFILE["pilot"] == art_block.IDS_BASELINE
    assert art_block.IDS_BY_PROFILE["stretch-v1"] == art_block.IDS
    assert art_block.IDS == art_block.IDS_BASELINE + STRETCH_IDS


def test_new_topics_match_the_roadmap_and_never_repeat():
    assert tuple(art_block.TOPICS[task_id] for task_id in STRETCH_IDS) == STRETCH_TOPICS
    assert len(set(art_block.TOPICS.values())) == len(art_block.TOPICS)


def test_each_new_topic_has_a_smoke_golden_and_is_delivered(monkeypatch, tmp_path):
    tasks = {
        task.id: task
        for task in art_block.generate(random.Random(3), RunConfig(profile="stretch-v1"))
    }
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(browser, "render", lambda html, **kwargs: _frame())
    for task_id in STRETCH_IDS:
        task = tasks[task_id]
        page = task.oracle["golden_html"]
        assert page.lstrip().lower().startswith("<!doctype html")
        assert str(task.prompt).split("\n")[0]
        score = art.evaluate(task, _answer(page))
        assert score.earned == score.possible == 1.0


@pytest.mark.parametrize("src", ["data:image/png;base64,AAAA", "blob:http://page/1"])
def test_inline_image_is_not_art(monkeypatch, src):
    """A ready-made image is not a drawing, even when the browser never fetches it."""
    page = "<!doctype html><html><body><img src=\"" + src + "\"></body></html>"
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(
        browser, "render",
        lambda *args, **kwargs: browser.RenderResult(
            ok=True, shot=b"\x89PNG\r\n\x1a\n",
            probe_ok=True, ready_ok=True, step_ok=True,
        ),
    )
    score = art.evaluate(_task(), _answer(page))
    assert score.earned == 0.0
    assert any("изображен" in check.detail for check in score.checks)
