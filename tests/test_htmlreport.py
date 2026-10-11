"""The HTML report: tabs, charts, live showcase, and self-containment.

The report is the artefact people read and record video from, so three things
must hold no matter how the page is opened: the facts match ``report.md``, the
showcase entries carry the answer page next to their frame, and nothing is
fetched from the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dailybench import core, gallery
from dailybench.htmlreport import render_html, write_html, write_report_html
from dailybench.report import summarise

FRAME = b"\x89PNG\r\n\x1a\n"


def _run(tmp_path: Path, name: str, model: str = "тест-модель") -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    scores = [
        {"task_id": "IF-1", "category": "A", "earned": 2.0, "possible": 2.0, "checks": [
            {"name": "json_format", "passed": True, "detail": "ок", "points": 2.0}]},
        {"task_id": "R-1", "category": "B", "earned": 0.0, "possible": 1.0, "checks": [
            {"name": "answer", "passed": False, "detail": "ответ не сошёлся", "points": 1.0}]},
        {"task_id": "K-1", "category": "K", "earned": 4.0, "possible": 4.0, "checks": [
            {"name": "html", "passed": True, "detail": "ок", "points": 4.0}]},
        {"task_id": "L-1", "category": "L", "earned": 0.0, "possible": 5.0,
         "skipped": "браузер недоступен", "checks": []},
    ]
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({
        "model": model, "quant": "Q4", "base_url": "https://example.test/v1",
        "context_window": 262144, "seed": 20260915,
    }, ensure_ascii=False), encoding="utf-8")
    (directory / "suite.json").write_text(json.dumps({
        "tasks": 2, "sha256": "abc",
        "per_task": [{"id": "IF-1", "category": "A", "prompt_sha256": "x"},
                     {"id": "K-1", "category": "K", "prompt_sha256": "y"}],
    }), encoding="utf-8")
    shot = directory / "shot.png"
    shot.write_bytes(FRAME)
    page = (
        "<!DOCTYPE html><html><body><script type='module'>"
        "import * as THREE from './vendor/three.module.js';"
        "window.__probe = () => ({t: 0}); window.__step = (dt) => dt; window.__ready = true;"
        "</script></body></html>"
    )
    (directory / "tasks.json").write_text(json.dumps([
        {"task": {"id": "IF-1", "category": "A", "title": "Форма"}, "transcript": {"turns": []}, "score": scores[0]},
        {"task": {"id": "K-1", "category": "K", "title": "K-1 — Баллистика"},
         "transcript": {"turns": [{"reply": {"text": f"```html\n{page}\n```"}}],
                        "spend": {"answer_tokens": 1500, "reasoning_tokens": 9000, "generate_s": 30.0, "tool_calls": 2},
                        "extra": {"visual": {"shot": str(shot), "steps": 48}}},
         "score": scores[2]},
    ], ensure_ascii=False), encoding="utf-8")
    return directory


def test_page_has_three_tabs_and_the_facts(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    page = render_html([directory])
    for marker in ("Решения моделей", "Решения моделей по сьюту", "Графики", "Витрина решений",
                   "plotly.min.js", "const DATA =", "ch-scored", "suite-table"):
        assert marker in page, marker
    assert "тест-модель" in page
    assert "ответ не сошёлся" in page
    assert "браузер недоступен" in page
    assert "IF-1" in page and "K-1" in page and "L-1" in page


def test_page_is_offline(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    page = render_html([directory])
    assert "example.test" in page  # endpoint напечатан как хост, без схемы
    external = [chunk for chunk in page.split("src='") if chunk.startswith(("http:", "https:"))]
    assert not external, external[:2]


def test_runs_are_ranked_and_named(tmp_path: Path):
    first = _run(tmp_path, "run-a", model="сильная")
    second = _run(tmp_path, "run-b", model="слабая")
    page = render_html([first, second])
    assert page.count("data-model='сильная'") >= 1 and page.count("data-model='слабая'") >= 1
    assert page.index("сильная") < page.index("слабая")  # сильная выше по баллам


def test_showcase_copies_the_answer_page_and_vendor(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    out = write_html([directory], tmp_path / "comparison.html")
    assert out.is_file()
    copied = list((tmp_path / "_showcase").rglob("K-1.html"))
    assert copied, "страница ответа должна лежать рядом с отчётом"
    body = copied[0].read_text(encoding="utf-8")
    assert "addEventListener('message'" in body, "страница обязана слушать 'step' из отчёта"
    # Страницы сцен собираются standalone: показ идёт через классический бандл
    # three.js, поэтому копии ES-модулей рядом с моделью больше не нужны, а сам
    # бандл лежит один на отчёт.
    assert "window.THREE" in body and "./three.global.js" in body or "_vendor/three.global.js" in body
    assert (tmp_path / "_vendor" / "three.global.js").is_file()
    assert not (copied[0].parent / "vendor").exists()
    assert list((tmp_path / "_showcase").rglob("K-1.png")), "кадр тоже копируется"
    assert (tmp_path / "_vendor" / "plotly.min.js").is_file()


def test_spend_is_documented_per_task_and_per_run(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    page = render_html([directory])
    assert "Токены reasoning" in page
    assert "9000" in page  # расход задания попал в таблицу
    assert "Балл за минуту" in page


def test_html_and_markdown_agree_on_the_verdict(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    scores = json.loads((directory / "scores.json").read_text(encoding="utf-8"))
    summary = summarise(scores)
    page = render_html([directory])
    assert summary["verdict"].split(" (")[0] in page
    assert summary["showcase_earned"] == 4.0
    assert core.FULL_TOTAL == 100.0


def test_gallery_extracts_the_answer_and_survives_a_missing_frame(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    run = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    entry = run[1]
    assert gallery.answer_html(entry).startswith("<!DOCTYPE html>")
    entry["transcript"]["extra"]["visual"]["shot"] = str(tmp_path / "нет.png")
    index = gallery.materialise(directory, tmp_path / "out")
    assert index["tasks"][0]["page"], "страница ответа обязана попасть в галерею"
    # кадр пропал — карточка обязана остаться рабочей, просто без картинки
    page = render_html([directory])
    assert "кадров нет" in page or "missing" in page


# --------------------------------------------------------------------------- #
# Per-run report and the comparison selector
# --------------------------------------------------------------------------- #


def test_a_per_run_report_describes_one_model(tmp_path: Path):
    directory = _run(tmp_path, "run-a", model="одиночка")
    page = render_html([directory], per_run=True, page_name="report.html")
    assert "<title>DailyBench — одиночка</title>" in page
    assert "Провенанс" in page and "p-prov" in page
    assert "Сьюта SHA-256" in page and "Окно контекста" in page
    assert "Прогон —" in page
    # the comparison wording must not leak into a one-model page
    assert "решения моделей на сьюте" not in page.split("</title>")[0]
    assert "report.html" in page, "file://-подсказка обязана звать ту же страницу"


def test_per_run_refuses_more_than_one_run(tmp_path: Path):
    first = _run(tmp_path, "run-a")
    second = _run(tmp_path, "run-b")
    with pytest.raises(ValueError, match="один прогон"):
        render_html([first, second], per_run=True)


def test_write_report_html_is_self_contained(tmp_path: Path):
    directory = _run(tmp_path, "run-a", model="одиночка")
    page = write_report_html(directory)
    assert page == directory / "report.html" and page.is_file()
    text = page.read_text(encoding="utf-8")
    assert "одиночка" in text
    assert (directory / "_vendor" / "plotly.min.js").is_file()
    assert list((directory / "_showcase").rglob("K-1.png")), "кадры витрины обязаны быть рядом"


def test_compare_selects_explicit_models(tmp_path: Path):
    first = _run(tmp_path, "run-a", model="сильная")
    second = _run(tmp_path, "run-b", model="слабая")
    page = render_html([first, second], compare=["слабая"])
    assert "data-model='слабая'" in page
    assert "data-model='сильная'" not in page


def test_compare_keeps_the_requested_order(tmp_path: Path):
    first = _run(tmp_path, "run-a", model="сильная")
    second = _run(tmp_path, "run-b", model="слабая")
    page = render_html([first, second], compare=["слабая", "сильная"])
    assert page.index("data-model='слабая'") < page.index("data-model='сильная'")


def test_compare_rejects_an_unknown_model(tmp_path: Path):
    directory = _run(tmp_path, "run-a", model="сильная")
    with pytest.raises(ValueError, match="неизвестные модели"):
        render_html([directory], compare=["нет-такой"])


def test_the_header_counts_come_from_the_run_not_a_constant(tmp_path: Path):
    """A shipped count used to be hard-coded and drifted from the suite."""
    directory = _run(tmp_path, "run-a")
    # The fixture's suite digest names exactly two tasks: one scored, one showcase.
    page = render_html([directory], per_run=True, page_name="report.html")
    assert "Прогон — 2 заданий" in page
    assert "1 балльных" in page and "1 витринных" in page


def test_compare_accepts_the_directory_name(tmp_path: Path):
    """``--compare qwen36`` must work: that is the name the user typed to run it."""
    directory = _run(tmp_path, "run-a", model="метка модели")
    page = render_html([directory], compare=["run-a"])
    assert "метка модели" in page


def test_a_run_of_the_suite_is_named_after_the_run_not_a_guess(tmp_path: Path):
    """The per-run page names the run it belongs to, so two models never collide."""
    first = _run(tmp_path, "run-a", model="модель-а")
    second = _run(tmp_path, "run-b", model="модель-б")
    page_a = render_html([first], per_run=True, page_name="report.html")
    page_b = render_html([second], per_run=True, page_name="report.html")
    assert "модель-а" in page_a and "модель-а" not in page_b
    assert "модель-б" in page_b


def test_publish_keeps_a_page_per_run(tmp_path: Path):
    """A published artefact must let a reader open one model without hunting."""
    from dailybench.cli import main

    first = _run(tmp_path, "run-a", model="модель-а")
    code = main(["publish", str(first), "--out", str(tmp_path / "pub")])
    assert code == 0
    own = tmp_path / "pub" / "runs" / "run-a" / "index.html"
    assert own.is_file()
    assert "модель-а" in own.read_text(encoding="utf-8")
    assert "runs/run-a/index.html" in (tmp_path / "pub" / "index.html").read_text(encoding="utf-8")


def test_a_showcase_only_run_does_not_claim_a_zero_point_suite(tmp_path: Path):
    """``--only K,L`` must not read as "the suite is worth 0 points"."""
    directory = _run(tmp_path, "run-show", model="витринная")
    (directory / "suite.json").write_text(json.dumps({
        "tasks": 1, "sha256": "abc", "per_task": [{"id": "K-1", "category": "K"}],
    }), encoding="utf-8")
    page = render_html([directory], per_run=True, page_name="report.html")
    assert "балльных категорий нет" in page
    assert f"из {core.FULL_TOTAL:.0f} баллов сьюта" in page


def test_a_lost_frame_is_rendered_again_for_the_showcase(monkeypatch, tmp_path: Path):
    """A purged frame cache must not silently empty the showcase.

    The frame is rendered again from the raw answer — the exact page the checks
    scored; on a machine without a browser the recorded path stays instead.
    """
    from dailybench import browser, visual

    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path / "cache"))
    directory = _run(tmp_path, "run-a")
    entries = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    entry = entries[1]  # the K-1 entry with a page and a shot
    entry["transcript"]["extra"]["visual"]["shot"] = str(tmp_path / "потерян.png")
    run = {"config": {"model": "m"}, "tasks": entries}

    # Без браузера: рендер не удался, карточка остаётся с записанным путём.
    monkeypatch.setattr(
        visual, "_cached_render",
        lambda html, steps, dt: (browser.RenderResult(ok=False), False),
    )
    shot = next(i["shot"] for i in gallery.showcase_entries(run) if i["id"] == "K-1")
    assert shot.endswith("потерян.png"), "без браузера остаётся записанный путь"

    # Браузер доступен: витрина берёт восстановленный кадр, а не пустоту.
    def fake_render(html: str, steps: int, dt: float):
        path = visual.shot_path(html, steps, dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FRAME)
        return browser.RenderResult(ok=True, shot=FRAME), False

    monkeypatch.setattr(visual, "_cached_render", fake_render)
    item = next(i for i in gallery.showcase_entries(run) if i["id"] == "K-1")
    assert item["shot"] == str(visual.shot_path(item["raw"], 48, 1.0 / 60))
    assert "потерян.png" not in item["shot"]


def test_a_frameless_timeout_keeps_its_sample_point(monkeypatch, tmp_path: Path):
    """A run that died before ``_observe`` must still reproduce its frame.

    ``extra.visual`` stays empty when the screenshot times out, but ``meta.steps``
    holds the sample point (both blocks step at 1/60). Without that fallback the
    page kept its live scene and silently lost its picture — 19 pages across the
    five published runs.
    """
    from dailybench import browser, visual

    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path / "cache"))
    directory = _run(tmp_path, "run-a")
    entries = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    entry = entries[1]
    entry["transcript"]["extra"]["visual"] = {}
    entry["task"]["meta"] = {"steps": 48}
    seen: dict[str, float] = {}

    def fake_render(html: str, steps: int, dt: float):
        seen["steps"], seen["dt"] = steps, dt
        path = visual.shot_path(html, steps, dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FRAME)
        return browser.RenderResult(ok=True, shot=FRAME), False

    monkeypatch.setattr(visual, "_cached_render", fake_render)
    item = next(
        i for i in gallery.showcase_entries({"config": {"model": "m"}, "tasks": entries})
        if i["id"] == "K-1"
    )
    assert seen == {"steps": 48, "dt": 1.0 / 60}
    assert item["steps"] == 48 and item["dt"] == 1.0 / 60
    assert Path(item["shot"]).is_file(), "кадр обязан восстановиться, а не потеряться"
