"""Behavioural checks for free-form artwork capture and its frame cache."""

from __future__ import annotations

import json

from dailybench import art, browser, core, visual
from dailybench.report import summarise

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
    first, hit = art.cached_art_render(SVG_PAGE)
    again, reused = art.cached_art_render(SVG_PAGE)
    assert not hit and reused and len(captures) == 1
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
    healed, hit = art.cached_art_render(SVG_PAGE)
    assert not hit and healed.shot == b"captured-frame"
    assert json.loads(metadata.read_text(encoding="utf-8"))["result"]["ok"]


def test_render_check_does_not_add_artistic_quality_points():
    scores = [{"task_id": "IF-1", "category": "A", "earned": 2.0, "possible": 2.0},
              {"task_id": "L-1", "category": "L", "earned": 1.0, "possible": 1.0}]
    summary = summarise(scores)
    assert summary["headline_earned"] == summary["headline_possible"] == 2.0
    assert summary["showcase_earned"] == summary["showcase_possible"] == 0.0
