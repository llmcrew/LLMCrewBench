"""The HTML report: tabs, charts, live showcase, and self-containment.

The report is the artefact people read and record video from, so three things
must hold no matter how the page is opened: the facts match ``report.md``, the
showcase entries carry the answer page next to their frame, and nothing is
fetched from the network.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from dailybench import core, gallery
from dailybench.browser import REPO
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


# --------------------------------------------------------------------------- #
# Run identity, injection, containment and rebuild safety (R06/R16/R17/R24)
# --------------------------------------------------------------------------- #


def test_two_runs_of_one_model_are_two_columns(tmp_path: Path):
    """R06: the matrix is keyed by run id, never by the display label."""
    first = _run(tmp_path, "run-a", model="одна модель")
    second = _run(tmp_path, "run-b", model="одна модель")
    scores = json.loads((second / "scores.json").read_text(encoding="utf-8"))
    for score in scores:
        score["earned"] = 0.0
    (second / "scores.json").write_text(json.dumps(scores), encoding="utf-8")

    page = render_html([first, second])
    head = page.split("<table class='grid suite-table'>")[1].split("</thead>")[0]
    assert "<th>одна модель · run-a</th>" in head
    assert "<th>одна модель · run-b</th>" in head
    row = next(chunk for chunk in page.split("<tr>") if "IF-1" in chunk)
    # Первая колонка — 2/2, вторая — 0/2. При ключе по label обе показывали 2.
    assert "2.0" in row and "0.0" in row


def test_imported_metadata_is_escaped_everywhere(tmp_path: Path):
    """R16: markup, ``</script>`` and quotes must not break the document."""
    directory = _run(tmp_path, "run-a")
    label = 'модель "<b>x</b>"\u2028</script>'
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config["model"] = label
    (directory / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    tasks[0]["task"]["title"] = '</script><script>alert("t")</script>'
    (directory / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

    page = render_html([directory], per_run=True, page_name="report.html")
    assert "<h1>DailyBench — модель &quot;&lt;b&gt;x&lt;/b&gt;&quot;" in page
    assert page.count("<script") == page.count("</script>")
    data = page.split("const DATA = ", 1)[1].split(";\n\n", 1)[0]
    assert "<" not in data and "\\u2028" in data
    assert json.loads(data)["models"] == [label]
    # Viewer собирается DOM API, а не innerHTML.
    assert "framesEl.innerHTML" not in page
    assert "createElement('iframe')" in page


def test_an_id_outside_the_schema_does_not_escape_pages(tmp_path: Path):
    """R17: ``../review-escape`` must not write next to ``pages/``."""
    directory = _run(tmp_path, "run-a")
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    tasks[1]["task"]["id"] = "../review-escape"
    (directory / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="недопустимый id задания"):
        gallery.materialise(directory, tmp_path / "out")
    assert not (tmp_path / "out" / "review-escape.html").is_file()
    assert not (tmp_path / "out" / "pages" / "review-escape.html").is_file()


def test_a_frame_outside_the_run_and_cache_is_not_copied(monkeypatch, tmp_path: Path):
    """R17: a recorded ``visual.shot`` is only trusted inside allowed roots."""
    from dailybench import browser, visual

    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(visual, "_cached_render", lambda *a: (browser.RenderResult(ok=False), False))
    directory = _run(tmp_path, "run-a")
    canary = tmp_path / "canary.png"
    canary.write_bytes(FRAME + b"canary")
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    tasks[1]["transcript"]["extra"]["visual"]["shot"] = str(canary)
    (directory / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

    write_report_html(directory)
    frames = list((directory / "_showcase").rglob("*.png"))
    assert not frames, "локальный файл вне run/cache не должен попасть в публикацию"


def test_rebuild_without_tasks_keeps_published_pages_and_frames(tmp_path: Path):
    """R24: a published run lost its tasks.json; rebuild is read-only."""
    directory = _run(tmp_path, "run-a")
    write_report_html(directory)
    showcase = directory / "_showcase" / "run-a"
    assert (showcase / "pages" / "K-1.html").is_file()
    assert (showcase / "frames" / "K-1.png").is_file()

    (directory / "tasks.json").unlink()
    write_report_html(directory)

    assert (showcase / "pages" / "K-1.html").is_file()
    assert (showcase / "frames" / "K-1.png").is_file()
    page = (directory / "report.html").read_text(encoding="utf-8")
    assert "_showcase/run-a/pages/K-1.html" in page


# --------------------------------------------------------------------------- #
# Self-contained publication and wheel assets (R18)
# --------------------------------------------------------------------------- #


def test_published_report_has_no_broken_local_links(tmp_path: Path):
    from dailybench.cli import main

    directory = _run(tmp_path, "run-a")
    out = tmp_path / "pub"
    assert main(["publish", str(directory), "--out", str(out)]) == 0

    missing: list[str] = []
    for page in out.rglob("*.html"):
        for ref in re.findall(r"""(?:src|href)='([^']+)'""", page.read_text(encoding="utf-8")):
            if ref.startswith(("http:", "https:", "data:", "mailto:", "#")):
                continue
            if not (page.parent / ref).resolve().exists():
                missing.append(f"{page.relative_to(out)} -> {ref}")
    assert not missing


def _wheel_python() -> str | None:
    """An interpreter able to build this wheel without the network."""
    for candidate in (sys.executable, shutil.which("python3"), shutil.which("python")):
        if not candidate:
            continue
        probe = subprocess.run([candidate, "-c", "import setuptools, wheel"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return candidate
    return None


def test_wheel_contains_the_visual_assets(tmp_path: Path):
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")

    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = config["tool"]["setuptools"]["data-files"]
    assert "vendor/three/three.module.js" in declared["vendor/three"]
    assert "vendor/plotly/plotly.min.js" in declared["vendor/plotly"]

    python = _wheel_python()
    if python is None:
        pytest.skip("нет интерпретатора с setuptools/wheel")
    out = tmp_path / "wheel"
    subprocess.run(
        [python, "setup.py", "egg_info", "--egg-base", str(tmp_path),
         "build", "--build-base", str(tmp_path / "build"),
         "bdist_wheel", "--dist-dir", str(out)],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    wheel = next(out.glob("*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert any(name.endswith("vendor/three/three.module.js") for name in names), names[:5]
    assert any(name.endswith("vendor/plotly/plotly.min.js") for name in names), names[:5]


# --------------------------------------------------------------------------- #
# Verdict caveats and per-profile suite figures (P2b/P3c/P3d)
# --------------------------------------------------------------------------- #


def _with_e(directory: Path, admission: bool | None) -> None:
    """Append a long-context score, optionally with an unknown admission."""
    path = directory / "scores.json"
    scores = json.loads(path.read_text(encoding="utf-8"))
    scores.append({
        "task_id": "LC-1", "category": "E", "earned": 4.0 if admission is not False else 0.0,
        "possible": 4.0, "admission": admission, "checks": [],
    })
    path.write_text(json.dumps(scores), encoding="utf-8")


def test_the_verdict_caveats_reach_the_card_and_the_table(tmp_path: Path):
    """An admission failure changes the denominator; the page must say why."""
    for admission, marker in ((True, None), (False, "не прошёл допуск"),
                              (None, "допуск не подтверждён")):
        directory = _run(tmp_path, f"run-{admission}", model=f"модель-{admission}")
        _with_e(directory, admission)
        page = render_html([directory])
        if marker:
            assert marker in page, marker
        else:
            assert "не прошёл допуск" not in page
            assert "допуск не подтверждён" not in page


def test_the_long_context_cell_names_the_admission_state(tmp_path: Path):
    """The E percentage alone reads as an ordinary zero for a failed admission."""
    failed = _run(tmp_path, "run-failed", model="недопущенная")
    _with_e(failed, False)
    page = render_html([failed])
    assert "admission failed'" in page and "не допущен" in page

    unknown = _run(tmp_path, "run-unknown", model="неизвестная")
    _with_e(unknown, None)
    page = render_html([unknown])
    assert "admission unverified'" in page and "допуск не подтверждён" in page

    admitted = _run(tmp_path, "run-ok", model="допущенная")
    _with_e(admitted, True)
    page = render_html([admitted])
    assert "class='num admission" not in page
    assert "100%" in page  # 4/4 у E и остальные полные


def test_the_header_weighs_the_tasks_on_offer_not_the_category_ceiling(tmp_path: Path):
    """``--only IF-1`` offered 2 points, not A's whole 10."""
    directory = _run(tmp_path, "run-only")
    (directory / "suite.json").write_text(json.dumps({
        "tasks": 1, "sha256": "abc",
        "per_task": [{"id": "IF-1", "category": "A", "max_points": 2.0}],
    }), encoding="utf-8")
    page = render_html([directory], per_run=True, page_name="report.html")
    assert "1 балльных (2 баллов качества)" in page


def test_the_header_follows_the_profile_weights(tmp_path: Path):
    """A pilot K block is 12 points, a stretch one 144 — not the baseline 80."""
    for profile, ceiling in (("pilot", 12), ("stretch-v1", 144)):
        directory = _run(tmp_path, f"run-{profile}")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config["profile"] = profile
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "suite.json").write_text(json.dumps({
            "tasks": 2, "sha256": "abc", "profile": profile,
            "per_task": [{"id": "IF-1", "category": "A", "max_points": 2.0},
                         {"id": "K-1", "category": "K", "max_points": 0.6}],
        }), encoding="utf-8")
        page = render_html([directory], per_run=True, page_name="report.html")
        assert f"потолок блока {ceiling}" in page, profile


def test_the_visual_gallery_escapes_and_contains_imported_values(tmp_path: Path):
    """P3d: ``<b>`` must not become markup and ``../`` must not escape ``visual/``."""
    from dailybench.report import write_visual_gallery

    directory = _run(tmp_path, "run-a")
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    tasks[1]["task"]["id"] = "../K-<b>1</b>"
    tasks[1]["task"]["title"] = "Кадр <b>жирный</b>"
    tasks[1]["score"]["task_id"] = "../K-<b>1</b>"
    scores = json.loads((directory / "scores.json").read_text(encoding="utf-8"))
    scores[2]["task_id"] = "../K-<b>1</b>"
    write_visual_gallery({"dir": str(directory), "tasks": tasks, "scores": scores})

    page = (directory / "visual.html").read_text(encoding="utf-8")
    assert "&lt;b&gt;жирный&lt;/b&gt;" in page
    assert "<b>жирный</b>" not in page
    assert "src=\"visual/_K-_b_1_b_.png\"" in page
    assert (directory / "visual" / "_K-_b_1_b_.png").is_file()
    assert "visual/../" not in page


def test_the_visual_gallery_skips_a_frame_outside_the_run(tmp_path: Path):
    """An imported ``visual.shot`` outside the run and cache is not copied."""
    from dailybench.report import write_visual_gallery

    directory = _run(tmp_path, "run-a")
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    outside = tmp_path / "вне.png"
    outside.write_bytes(FRAME)
    tasks[1]["transcript"]["extra"]["visual"]["shot"] = str(outside)
    scores = json.loads((directory / "scores.json").read_text(encoding="utf-8"))
    write_visual_gallery({"dir": str(directory), "tasks": tasks, "scores": scores})

    page = (directory / "visual.html").read_text(encoding="utf-8")
    assert "кадров нет" not in page
    assert "кадра нет: страница не отрисовалась" in page
    assert not (directory / "visual" / "K-1.png").exists()


# --------------------------------------------------------------------------- #
# Uncertainty bands (spec 5 / 7.1): decodable values, small-sample caveat
# --------------------------------------------------------------------------- #


def _ci_run(tmp_path: Path, name: str, model: str, earned: list[float]) -> Path:
    directory = _run(tmp_path, name, model=model)
    scores = [
        {"task_id": f"IF-{index}", "category": "A", "earned": value, "possible": 10.0,
         "checks": [{"name": "json_format", "passed": value == 10.0, "detail": "", "points": 10.0}]}
        for index, value in enumerate(earned)
    ]
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    return directory


def _headline_bands(page: str) -> list[tuple[str, float, float]]:
    """Model of each card and the first (headline) band that follows it."""
    cards = list(re.finditer(r"class='card' data-model='([^']+)'", page))
    found = []
    for index, card in enumerate(cards):
        end = cards[index + 1].start() if index + 1 < len(cards) else len(page)
        band = re.search(r"data-ci-low='([\d.]+)' data-ci-high='([\d.]+)'", page[card.end():end])
        if band:
            found.append((card.group(1), float(band.group(1)), float(band.group(2))))
    return found


def test_uncertainty_is_encoded_in_data_attributes(tmp_path: Path):
    strong = _ci_run(tmp_path, "run-a", "сильная", [10.0, 9.0] * 20)
    weak = _ci_run(tmp_path, "run-b", "слабая", [2.0, 1.0] * 20)
    page = render_html([strong, weak])

    bands = {model: (low, high) for model, low, high in _headline_bands(page)}
    for directory in (strong, weak):
        summary = summarise(json.loads((directory / "scores.json").read_text(encoding="utf-8")))
        model = json.loads((directory / "config.json").read_text(encoding="utf-8"))["model"]
        assert bands[model][0] <= bands[model][1]
        assert bands[model] == pytest.approx(summary["headline_ci95"])


def test_category_scales_carry_a_band_for_each_measured_metric(tmp_path: Path):
    run = _ci_run(tmp_path, "run-a", "одна", [10.0, 9.0] * 3)
    page = render_html([run])
    summary = summarise(json.loads((run / "scores.json").read_text(encoding="utf-8")))
    expected = sum(1 for row in summary["per_category"].values() if row["ci95"]) + 1  # + headline
    bands = re.findall(r"data-ci-low='([\d.]+)' data-ci-high='([\d.]+)'", page)
    assert len(bands) == expected
    assert all(float(low) <= float(high) for low, high in bands)


def test_compare_publishes_the_paired_difference_and_weight_sensitivity(tmp_path: Path):
    left = _ci_run(tmp_path, "run-a", "ровная", [6.0 + (index % 5) for index in range(40)])
    right = _ci_run(tmp_path, "run-b", "чуть-слабее",
                    [6.0 + (index % 5) - 0.05 for index in range(40)])
    page = render_html([left, right])
    assert "Парное сравнение: разность и её интервал" in page
    assert "Вектор категорий" in page
    assert "Чувствительность к весам" in page
    assert "повторы и сиды не измеряются" in page
    assert re.search(r"\+\d+\.\d\d … \+\d+\.\d\d п\.п\.", page)
    assert ">значимо</td>" in page


def test_a_small_sample_is_marked_provisional_on_the_page(tmp_path: Path):
    run = _ci_run(tmp_path, "run-a", "одна", [10.0, 2.0, 9.0])
    page = render_html([run])
    assert re.search(r"ci-caption'>[^<]*ориентировочно", page)
    band = re.search(r"class='ci-band'[^>]*data-n='(\d+)'", page)
    assert band and int(band.group(1)) == 3


def test_a_sound_sample_has_no_provisional_note(tmp_path: Path):
    run = _ci_run(tmp_path, "run-a", "одна", [10.0, 9.0] * 16)
    page = render_html([run])
    assert not re.search(r"ci-caption'>[^<]*ориентировочно", page)
    assert re.search(r"ci-caption'>ДИ 95%", page)


# --------------------------------------------------------------------------- #
# Visual figures (spec 7.2/7.3): compare gets the radar and heatmap,
# a per-run page gets verdict, completeness, clusters and diffs
# --------------------------------------------------------------------------- #


def test_compare_has_a_radar_and_a_value_bearing_heatmap(tmp_path: Path):
    strong = _run(tmp_path, "run-a", model="сильная")
    weak = _run(tmp_path, "run-b", model="слабая")
    page = render_html([strong, weak])
    assert "viz radar" in page and "viz heatmap" in page
    assert "Радар компетенций" in page and "Heatmap ошибок" in page
    # каждая ячейка несёт модель, семейство и число, а не только цвет
    assert "data-model='сильная'" in page
    assert "data-family='det/logic'" in page and "data-pct=" in page
    assert "role='img'" in page and "<title>" in page


def test_an_empty_compare_page_says_no_data_and_does_not_crash(tmp_path: Path):
    first = _run(tmp_path, "run-a", model="пустая-а")
    second = _run(tmp_path, "run-b", model="пустая-б")
    for directory in (first, second):
        (directory / "scores.json").write_text("[]", encoding="utf-8")
    page = render_html([first, second])
    assert "нет данных для радара" in page
    assert "нет данных для heatmap" in page


def test_a_per_run_page_has_verdict_completeness_clusters_and_diffs(tmp_path: Path):
    directory = _run(tmp_path, "run-a", model="одиночка")
    page = render_html([directory], per_run=True, page_name="report.html")
    markers = ["verdict-strip", "viz completeness", "viz clusters", "Визуальные диффы"]
    positions = [page.index(marker) for marker in markers]
    assert positions == sorted(positions), "порядок обязан идти за §7.2"
    assert "3 вывода" in page and "3 риска" in page
    assert "data-status='checked'" in page and "не запускалось" in page
    assert "data-status='not_run' data-count='1'" in page


def test_the_per_run_diff_names_the_task_and_survives_a_missing_reference(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    page = render_html([directory], per_run=True, page_name="report.html")
    assert "K-1" in page.split("Визуальные диффы")[1]
    assert "нет данных" in page.split("Визуальные диффы")[1]


def test_a_shot_outside_the_run_and_cache_is_not_embedded(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    outside = tmp_path / "чужой-кадр.png"
    outside.write_bytes(FRAME)
    entries = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    entries[1]["transcript"]["extra"]["visual"]["shot"] = str(outside)
    (directory / "tasks.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    page = render_html([directory], per_run=True, page_name="report.html")
    diffs = page.split("Визуальные диффы")[1]
    assert "data:image" not in diffs
    assert "нет данных" in diffs


def test_a_private_task_is_marked_and_its_oracle_detail_collapsed(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    (directory / "suite.json").write_text(json.dumps({
        "per_task": [{"id": "IF-1", "category": "A", "private": True},
                     {"id": "K-1", "category": "K"}],
    }), encoding="utf-8")
    entries = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    entries[0]["task"]["oracle"] = {"reference": "ОЖИДАЕМОЕ-42"}
    entries[0]["score"]["checks"][0]["detail"] = "ожидалось ОЖИДАЕМОЕ-42"
    (directory / "tasks.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    page = render_html([directory], per_run=True, page_name="report.html")
    row = re.search(r"<tr class='ok'><td>IF-1</td>.*?</tr>", page, re.S).group(0)
    assert "<span class='private'" in row
    visible = re.sub(r"<details.*?</details>", "", row, flags=re.S)
    assert "ОЖИДАЕМОЕ-42" not in visible
    hidden = re.search(r"<details.*?</details>", row, re.S).group(0)
    assert "ОЖИДАЕМОЕ-42" in hidden and "оракул" in hidden
    assert "Приватный слой" in page


def test_provenance_marks_an_unverified_runtime(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    (directory / "environment.json").write_text(json.dumps({
        "image": "dailybench/runner-python:local", "status": "verified",
        "runtime": "firecracker", "runtime_verified": False,
    }), encoding="utf-8")
    page = render_html([directory], per_run=True, page_name="report.html")
    assert "firecracker (не подтверждён)" in page


def test_a_per_run_page_without_any_data_still_renders(tmp_path: Path):
    directory = _run(tmp_path, "run-a")
    (directory / "scores.json").write_text("[]", encoding="utf-8")
    page = render_html([directory], per_run=True, page_name="report.html")
    assert "нет данных: ни одного измеренного балла" in page
    assert "нет данных: ни одного задания" in page
    assert "нет данных: провалов нет" in page


# --------------------------------------------------------------------------- #
# Eligibility before ranking (H13)
# --------------------------------------------------------------------------- #


def _tier_run(
    tmp_path: Path,
    name: str,
    model: str,
    passed: int,
    total: int,
    *,
    sha: str,
) -> Path:
    """A run with ``total`` one-point tasks on its own contract."""
    directory = tmp_path / name
    directory.mkdir(parents=True)
    scores = [
        {
            "task_id": f"A-{index}", "category": "A",
            "earned": 1.0 if index <= passed else 0.0, "possible": 1.0,
            "checks": [{"name": "answer", "passed": index <= passed, "detail": "", "points": 1.0}],
        }
        for index in range(1, total + 1)
    ]
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({"model": model}), encoding="utf-8")
    (directory / "suite.json").write_text(json.dumps({
        "tasks": total, "sha256": sha,
        "per_task": [{"id": f"A-{i}", "category": "A", "max_points": 1.0} for i in range(1, total + 1)],
    }), encoding="utf-8")
    return directory


def test_a_partial_one_of_one_does_not_outrank_a_full_suite(tmp_path: Path):
    """PartialOne 1/1 used to take place 1 above Full95 95/100."""
    partial = _tier_run(tmp_path, "run-partial", "PartialOne", 1, 1, sha="sha-partial")
    # The partial run was built for the same kind of suite but only one task ran:
    # its own denominator is honest, its coverage is not the full contract.
    (partial / "suite.json").write_text(json.dumps({
        "tasks": 100, "sha256": "sha-partial",
        "per_task": [{"id": f"A-{i}", "category": "A", "max_points": 1.0} for i in range(1, 101)],
    }), encoding="utf-8")
    full = _tier_run(tmp_path, "run-full", "Full95", 95, 100, sha="sha-full")

    page = render_html([partial, full])
    assert page.index("data-model='Full95'") < page.index("data-model='PartialOne'")
    # The partial run is shown in the incomplete table, not as a rank.
    assert "class='grid incomplete-table'" in page
    assert "неполный прогон" in page
    assert "data-model='PartialOne'" in page
    # Full95 stays the leader of its own contract.
    assert "data-model='Full95'" in page.split("<table class='grid incomplete-table'>")[0]
    assert "95/100" in page


def test_different_suite_contracts_get_separate_tables(tmp_path: Path):
    """Two different suite SHA must not share one ranked table."""
    first = _tier_run(tmp_path, "run-one", "первая", 4, 5, sha="aaa111")
    second = _tier_run(tmp_path, "run-two", "вторая", 9, 10, sha="bbb222")

    page = render_html([first, second])
    assert page.count("Сравнение по контракту") == 2
    # Each model appears inside the section of its own contract: the stronger
    # contract leads, and the two never mix in one table.
    head_second = page.index("Сравнение по контракту <code>bbb222")
    head_first = page.index("Сравнение по контракту <code>aaa111")
    assert head_second < head_first
    assert "вторая" in page[head_second:head_first]
    assert "первая" in page[head_first:]
    assert "2 разных контракта" in page


def _outcome_run(tmp_path: Path, name: str, model: str, failure: str, sha: str = "sha-outcome") -> Path:
    """A one-task run whose only task carries a stored outcome kind."""
    directory = tmp_path / name
    directory.mkdir(parents=True)
    scores = [{
        "task_id": "A-1", "category": "A", "earned": 0.0, "possible": 1.0,
        "error": "прогон прерван", "failure": failure,
        "checks": [{"name": "answer", "passed": False, "detail": "", "points": 1.0}],
    }]
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({"model": model}), encoding="utf-8")
    (directory / "suite.json").write_text(json.dumps({
        "tasks": 1, "sha256": sha,
        "per_task": [{"id": "A-1", "category": "A", "max_points": 1.0}],
    }), encoding="utf-8")
    return directory


def test_an_integrity_violation_refuses_attestation(tmp_path: Path):
    """An integrity outcome is not plain infrastructure; the page says so."""
    good = _tier_run(tmp_path, "run-good", "чистая", 1, 1, sha="sha-mix")
    bad = _outcome_run(tmp_path, "run-bad", "подмена", "integrity_violation", sha="sha-mix")

    page = render_html([good, bad])
    assert "удостоверение отозвано" in page
    assert "отказ в удостоверении" in page
    assert page.index("data-model='чистая'") < page.index("data-model='подмена'")


def test_legacy_failure_names_keep_their_label_and_gate(tmp_path: Path):
    """An old bundle is counted under the canonical kind and stays unrankable."""
    good = _tier_run(tmp_path, "run-good", "чистая", 1, 1, sha="sha-old")
    bad = _outcome_run(tmp_path, "run-bad", "старая", "provider_unavailable", sha="sha-old")

    page = render_html([good, bad])
    assert core.failure_label("provider_unavailable") == "провайдер недоступен"
    assert "инфраструктура недоступна" in page
    assert "не ранжируется" in page
    assert page.index("data-model='чистая'") < page.index("data-model='старая'")


def test_unknown_spend_renders_as_not_measured_not_zero(tmp_path: Path):
    """An endpoint without usage must not read as a free, instant run."""
    directory = _run(tmp_path, "run-a")
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    for entry in tasks:
        if entry["task"]["id"] == "K-1":
            entry["transcript"]["spend"] = {
                "answer_tokens": None, "reasoning_tokens": None,
                "generate_s": None, "tool_calls": 1,
            }
    (directory / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

    page = render_html([directory])
    assert "токенов: <b>н/п</b>" in page
    assert "(н/п у 1 из 1)" in page
    assert "балл/мин <b>н/п</b>" in page
    assert "<td class='num small'>н/п</td>" in page
    assert '"per_min": null' in page
    # Не 10^11: без известного времени балл за минуту не печатается числом.
    assert "токенов: <b>0K</b>" not in page


# --------------------------------------------------------------------------- #
# Paired delta, baseline identity, rank denominator and heatmap cover (P1-P3)
# --------------------------------------------------------------------------- #


def _run_with(
    tmp_path: Path,
    name: str,
    model: str,
    rows: list[tuple[str, float, float]],
    sha: str = "sha-pair",
) -> Path:
    """A run with explicit ``(category, earned, possible)`` score rows."""
    directory = tmp_path / name
    directory.mkdir(parents=True)
    scores = [
        {"task_id": f"{category}-{index}", "category": category,
         "earned": earned, "possible": possible, "checks": []}
        for index, (category, earned, possible) in enumerate(rows)
    ]
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({"model": model}), encoding="utf-8")
    (directory / "suite.json").write_text(json.dumps({
        "tasks": len(rows), "sha256": sha,
        "per_task": [{"id": score["task_id"], "category": score["category"],
                      "max_points": score["possible"]} for score in scores],
    }), encoding="utf-8")
    return directory


def _append_scores(directory: Path, rows: list[tuple[str, float, float]]) -> None:
    path = directory / "scores.json"
    scores = json.loads(path.read_text(encoding="utf-8"))
    for offset, (category, earned, possible) in enumerate(rows):
        scores.append({"task_id": f"{category}-x{offset}", "category": category,
                       "earned": earned, "possible": possible, "checks": []})
    path.write_text(json.dumps(scores), encoding="utf-8")


def _comparison_row(page: str, model: str) -> str:
    """The row of one model in the paired-comparison table."""
    block = page.split("Парное сравнение: разность и её интервал")[1].split("</table>")[0]
    return re.search(rf"<tr><td class='model'>{re.escape(model)}</td>.*?</tr>", block, re.S).group(0)


def test_the_compare_delta_is_paired_and_lies_inside_its_interval(tmp_path: Path):
    """Δ must belong to the sample the interval is built from."""
    base = _run_with(tmp_path, "run-a", "база", [("A", 9.5, 10.0)] * 40)
    other = _run_with(tmp_path, "run-b", "другая", [("A", 10.0, 10.0)] * 40)
    _append_scores(base, [("A", 0.0, 10.0)] * 40)   # база мерила больше и провалила
    page = render_html([base, other])

    row = _comparison_row(page, "другая")
    cells = re.findall(r"<td class='num[^']*'>(.*?)</td>", row)
    delta = float(cells[0])
    band = re.match(r"([+-][\d.]+) … ([+-][\d.]+)", cells[1])
    assert band and float(band.group(1)) <= delta <= float(band.group(2)), cells[:2]
    assert delta == pytest.approx(-5.0)             # не предельные 47.5 − 100
    assert cells[2] == "47.5" and cells[3] == "100.0"   # предельные доли отдельно


def test_the_comparison_baseline_is_the_first_caller_run(tmp_path: Path):
    """comparison.md uses the caller's first run; HTML must use the same one."""
    weak = _run_with(tmp_path, "run-a", "первая", [("A", 5.0, 10.0)] * 20)
    strong = _run_with(tmp_path, "run-b", "вторая", [("A", 10.0, 10.0)] * 20)
    page = render_html([weak, strong])
    assert "База — <b>первая</b>" in page
    assert "<td class='num'>-50.00</td>" in _comparison_row(page, "вторая")
    # ...хотя ранг всё равно ставит вторую выше.
    ranked = page.split("Парное сравнение")[0]
    assert ranked.index("data-model='вторая'") < ranked.index("data-model='первая'")


def test_a_comparison_without_shared_tasks_prints_n_a(tmp_path: Path):
    first = _run_with(tmp_path, "run-a", "первая", [("A", 10.0, 10.0)] * 3)
    second = _run_with(tmp_path, "run-b", "вторая", [("B", 10.0, 10.0)] * 3)
    page = render_html([first, second])
    row = _comparison_row(page, "вторая")
    assert "<td class='num'>n/a</td><td class='num'>n/a</td>" in row
    assert "<td class='num'>n/a</td>" in row        # и «Общих задач»


def test_an_incomplete_run_is_unranked_in_the_paired_table(tmp_path: Path):
    base = _run_with(tmp_path, "run-a", "полная", [("A", 10.0, 10.0)] * 20)
    broken = _run_with(tmp_path, "run-b", "обрыв", [("A", 0.0, 10.0)] * 20)
    path = broken / "scores.json"
    scores = json.loads(path.read_text(encoding="utf-8"))
    scores[0]["error"] = "обрыв соединения"
    scores[0]["failure"] = "provider_unavailable"
    path.write_text(json.dumps(scores), encoding="utf-8")

    page = render_html([base, broken])
    row = _comparison_row(page, "обрыв")
    # Полосы у пары сильные, но вердикт не выносится.
    assert "не ранжируется: инфраструктура" in row
    assert ">незначимо<" not in row and ">значимо<" not in row


def test_a_run_of_another_contract_is_unranked_in_the_paired_table(tmp_path: Path):
    first = _run_with(tmp_path, "run-a", "первая", [("A", 10.0, 10.0)] * 20, sha="aaa")
    second = _run_with(tmp_path, "run-b", "чужая", [("A", 5.0, 10.0)] * 20, sha="bbb")
    page = render_html([first, second])
    row = _comparison_row(page, "чужая")
    assert "не ранжируется: другой контракт сьюта" in row
    assert ">значимо<" not in row


def test_the_rank_counts_earned_points_against_the_profile_denominator(tmp_path: Path):
    """85/90 must not outrank 92/100 by its smaller denominator."""
    more = _run_with(tmp_path, "run-a", "92 из 100", [("A", 1.0, 1.0)] * 92 + [("A", 0.0, 1.0)] * 8)
    fewer = _run_with(tmp_path, "run-b", "85 из 90", [("A", 1.0, 1.0)] * 85 + [("A", 0.0, 1.0)] * 5)
    page = render_html([more, fewer])
    ranked = page.split("Парное сравнение")[0]
    assert ranked.index("data-model='92 из 100'") < ranked.index("data-model='85 из 90'")
    assert re.search(r"data-model='92 из 100'><div class='rank'>1<", page)


def test_the_place_restarts_in_each_contract(tmp_path: Path):
    page = render_html([
        _run_with(tmp_path, "run-1", "а1", [("A", 1.0, 1.0)] * 10, sha="aaa"),
        _run_with(tmp_path, "run-2", "а2", [("A", 1.0, 1.0)] * 5 + [("A", 0.0, 1.0)] * 5, sha="aaa"),
        _run_with(tmp_path, "run-3", "б1", [("A", 1.0, 1.0)] * 8, sha="bbb"),
        _run_with(tmp_path, "run-4", "б2", [("A", 1.0, 1.0)] * 2, sha="bbb"),
    ])
    sections = page.split("Сравнение по контракту")
    assert len(sections) == 3
    for section in sections[1:]:
        assert "<div class='rank'>1<" in section
        assert "<div class='rank'>3<" not in section


def test_a_failed_admission_sorts_after_the_admitted_runs(tmp_path: Path):
    """Its own headline denominator (90) is smaller, not better."""
    admitted = _run_with(tmp_path, "run-a", "допущенная",
                         [("A", 1.0, 1.0)] * 95 + [("A", 0.0, 1.0)] * 5)
    short = _run_with(tmp_path, "run-b", "недопущенная", [("A", 1.0, 1.0)] * 90)
    path = short / "scores.json"
    scores = json.loads(path.read_text(encoding="utf-8"))
    scores.append({"task_id": "E-1", "category": "E", "earned": 0.0, "possible": 10.0,
                   "admission": False, "checks": []})
    path.write_text(json.dumps(scores), encoding="utf-8")

    page = render_html([short, admitted])
    ranked = page.split("Парное сравнение")[0]
    assert ranked.index("data-model='допущенная'") < ranked.index("data-model='недопущенная'")
    assert re.search(r"data-model='недопущенная'><div class='rank'>2<", page)


def test_the_heatmap_counts_measured_tasks_apart_from_skips(tmp_path: Path):
    """A skip must not be reported as a task behind the percentage."""
    first = _run(tmp_path, "run-a", model="измеряющая")
    second = _run(tmp_path, "run-b", model="вторая")

    def cell(page: str) -> str:
        return re.search(
            r"<g class='hm [^']*' data-model='измеряющая' data-family='det/logic'[^>]*>", page
        ).group(0)

    before = cell(render_html([first, second]))
    assert "data-measured='1'" in before and "data-tasks='1'" in before

    _append_scores(first, [("A", 0.0, 10.0)])
    scores = json.loads((first / "scores.json").read_text(encoding="utf-8"))
    scores[-1]["skipped"] = "нет времени"
    (first / "scores.json").write_text(json.dumps(scores), encoding="utf-8")

    after = cell(render_html([first, second]))
    assert "data-pct='100.0'" in after           # skip не улучшил ячейку...
    assert "data-measured='1'" in after          # ...а измерено по-прежнему одно
    assert "data-tasks='2'" in after             # всего заданий видно


def test_the_weight_table_marks_unmeasured_weight(tmp_path: Path):
    """An unmeasured category keeps its weight instead of becoming a free pass."""
    left = _run_with(tmp_path, "run-a", "только-A", [("A", 1.0, 1.0)] * 5)
    right = _run_with(tmp_path, "run-b", "A-и-B",
                      [("A", 0.5, 1.0)] * 5 + [("B", 1.0, 1.0)] * 5)
    page = render_html([left, right])
    published = page.split("Чувствительность к весам")[1].split("</tr>")[1]
    assert "не измерено 90% веса" in published
    assert ">right<" not in published and "right ·" in published
