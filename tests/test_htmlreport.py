"""The HTML report: tabs, charts, live showcase, and self-containment.

The report is the artefact people read and record video from, so three things
must hold no matter how the page is opened: the facts match ``report.md``, the
showcase entries carry the answer page next to their frame, and nothing is
fetched from the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from dailybench import core, gallery
from dailybench.htmlreport import render_html, write_html
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
    (directory / "suite.json").write_text(json.dumps({"tasks": 2, "sha256": "abc"}), encoding="utf-8")
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
