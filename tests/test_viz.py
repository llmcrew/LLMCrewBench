"""Чистые фигуры: числа → геометрия, а не пиксели.

Каждый тест декодирует ``data-*``/``<title>`` и сверяет координаты с
арифметикой, поэтому переезд вёрстки не ломает смысл, а пустой вход обязан
сказать «нет данных».
"""

from __future__ import annotations

import re

import pytest
from PIL import Image

from dailybench import viz


def _points(svg: str) -> list[tuple[float, float]]:
    raw = re.search(r"<polygon points='([^']+)' fill='#79c0ff'", svg)
    assert raw, svg
    return [
        (float(x), float(y))
        for x, y in (chunk.split(",") for chunk in raw.group(1).split(" "))
    ]


def test_radar_vertices_are_computed_from_the_values():
    svg = viz.radar(
        [{"name": "модель", "values": {"A": 0.0, "B": 50.0, "C": 100.0}}],
        ["A", "B", "C"],
        size=400,
    )
    points = _points(svg)
    expected = [
        viz.polar(200, 200, 144, index, 3, share)
        for index, share in enumerate((0.0, 0.5, 1.0))
    ]
    for got, want in zip(points, expected):
        assert got[0] == pytest.approx(want[0], abs=0.01)
        assert got[1] == pytest.approx(want[1], abs=0.01)
    # Ноль — в центре, сотня — на полном радиусе.
    assert points[0] == (200.0, 200.0)
    assert round(points[2][0], 2) == round(200 - 144 * 0.8660254, 2)


def test_radar_vertices_carry_category_and_value():
    svg = viz.radar(
        [{"name": "м", "values": {"A": 25.0, "B": 80.0}}],
        ["A", "B"],
    )
    values = re.findall(r"data-category='([^']+)' data-value='([^']*)'", svg)
    assert values == [("A", "25.0"), ("B", "80.0")]
    assert "<title>" in svg and "role='img'" in svg


def test_radar_marks_an_unmeasured_axis_as_empty():
    svg = viz.radar(
        [{"name": "м", "values": {"A": 100.0, "B": None}}],
        ["A", "B"],
    )
    assert ("B", "") in re.findall(r"data-category='([^']+)' data-value='([^']*)'", svg)
    assert "B=—" in svg


def test_radar_without_measured_values_says_no_data():
    assert "нет данных" in viz.radar([{"name": "м", "values": {"A": None}}], ["A"])
    assert "нет данных" in viz.radar([], ["A"])


def test_heatmap_cells_are_placed_by_index_and_carry_the_number():
    svg = viz.heatmap(
        [{"name": "модель", "values": {"det/logic": {"pct": 20.0, "tasks": 5, "failed": 4},
                                        "det/math": {"pct": 90.0, "tasks": 10, "failed": 1}}}],
        ["det/logic", "det/math"],
    )
    cells = re.findall(
        r"<g class='hm (\w+)' data-model='([^']+)' data-family='([^']+)' "
        r"data-pct='([^']*)' data-tasks='(\d+)' data-failed='(\d+)'>",
        svg,
    )
    assert cells == [("zero", "модель", "det/logic", "20.0", "5", "4"),
                     ("full", "модель", "det/math", "90.0", "10", "1")]
    xs = [float(x) for x in re.findall(r"<rect x='([\d.]+)'", svg)[:2]]
    assert xs == [119.0, 185.0]  # 118 + колонка * 66 + 1
    assert "20%" in svg and "90%" in svg


def test_heatmap_unmeasured_cells_are_dashed_and_worded():
    svg = viz.heatmap(
        [{"name": "м", "values": {"det/logic": {"tasks": 0}}}],
        ["det/logic"],
    )
    assert "data-pct=''" in svg and "—" in svg and "нет задач" in svg


def test_heatmap_without_rows_says_no_data():
    assert "нет данных" in viz.heatmap([], ["det/logic"])


def test_completeness_widths_follow_the_counts():
    svg = viz.completeness({"checked": 1, "not_run": 3})
    segments = re.findall(
        r"data-status='(\w+)' data-count='(\d+)' data-share='([\d.]+)'", svg
    )
    assert [s[0] for s in segments] == ["checked", "not_run"]
    assert [s[2] for s in segments] == ["0.2500", "0.7500"]
    widths = [float(w) for w in re.findall(r"<rect x='[\d.]+' y='\d+' width='([\d.]+)'", svg)]
    assert widths == pytest.approx([160.0, 480.0], abs=0.01)


def test_completeness_without_tasks_says_no_data():
    assert "нет данных" in viz.completeness({})


def test_status_legend_words_and_marks_every_status():
    svg = viz.status_legend({"not_run": 2})
    for status in viz.STATUS_ORDER:
        assert f"data-status='{status}'" in svg
        assert viz.STATUS_MARKS[status] in svg
        assert viz.STATUS_LABELS[status] in svg
    assert "не запускалось · 2" in svg


def test_status_classification_keeps_the_cause_apart():
    assert viz.classify({"skipped": "нет браузера"}) == "not_run"
    assert viz.classify({"error": "evaluator failed: boom"}) == "evaluator_unavailable"
    assert viz.classify({"error": "ReadTimeout"}) == "model_failed"
    assert viz.classify({"checks": [{"name": "html", "passed": False,
                                     "detail": "рендер не удался: таймаут"}]}) == "capture_failed"
    assert viz.classify({"checks": [{"name": "image", "passed": False,
                                     "detail": "кадра нет"}]}) == "media_missing"
    assert viz.classify({"checks": [{"name": "answer", "passed": False, "detail": "мимо"}]}) == "checked"
    assert viz.task_statuses([{"skipped": "x"}, {"checks": []}]) == {
        "checked": 1, "not_run": 1, "model_failed": 0,
        "capture_failed": 0, "evaluator_unavailable": 0, "media_missing": 0,
    }


def test_cluster_map_is_deterministic_and_buckets_similar_texts():
    items = [
        {"id": "A-1", "model": "м", "category": "A", "text": "ответ не сошёлся по формату"},
        {"id": "A-2", "model": "м", "category": "A", "text": "ответ не сошёлся по формату"},
        {"id": "B-1", "model": "м", "category": "B", "text": "деление на ноль в решении"},
    ]
    first = viz.cluster_map(items)
    again = viz.cluster_map(items)
    assert first == again
    assert "char-триграммы" in first and "seed" in first
    points = re.findall(r"data-id='([^']+)' data-model='[^']+' data-category='[^']+' "
                        r"data-x='([\d.]+)' data-y='([\d.]+)' data-cluster='(C\d+)' data-count='(\d+)'", first)
    assert [p[0] for p in points] == ["A-1", "A-2", "B-1"]
    same = {p[3] for p in points if p[0].startswith("A-")}
    assert len(same) == 1, "одинаковый текст обязан попасть в один кластер"
    assert {p[3] for p in points if p[0] == "B-1"} != same
    assert all(0 <= float(p[1]) <= 640 and 0 <= float(p[2]) <= 320 for p in points)
    assert "A-1 · м: ответ не сошёлся" in first


def test_cluster_map_seed_changes_the_hash_geometry():
    items = [{"id": f"A-{i}", "model": "м", "category": "A", "text": f"ошибка {i}"} for i in range(6)]
    assert viz.cluster_map(items, seed=1) != viz.cluster_map(items, seed=2)


def test_cluster_map_without_failures_says_no_data():
    assert "нет данных" in viz.cluster_map([])


def _png(path, *, size=(40, 20), box=None):
    image = Image.new("RGB", size, "black")
    if box:
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                image.putpixel((x, y), (255, 255, 255))
    image.save(path)
    return path


def test_diff_bbox_is_arithmetic_not_guesswork(tmp_path):
    reference = _png(tmp_path / "ref.png")
    answer = _png(tmp_path / "ans.png", box=(5, 2, 9, 6))
    info = viz.diff_bbox(reference, answer)
    assert info["bbox"] == (5, 2, 9, 6)
    assert info["changed"] == 16
    assert (info["width"], info["height"]) == (40, 20)
    assert info["reason"] is None


def test_diff_bbox_normalises_a_size_mismatch(tmp_path):
    reference = _png(tmp_path / "ref.png", size=(40, 20))
    answer = _png(tmp_path / "ans.png", size=(80, 40), box=(10, 4, 18, 12))
    info = viz.diff_bbox(reference, answer)
    assert info["width"] == 40 and info["height"] == 20
    assert info["bbox"] is not None


def test_diff_bbox_without_change_or_file(tmp_path):
    same = _png(tmp_path / "same.png")
    empty = viz.diff_bbox(same, same)
    assert empty["bbox"] is None and empty["reason"] == "расхождений нет"
    missing = viz.diff_bbox(tmp_path / "нет.png", same)
    assert missing["bbox"] is None and "нет данных" in missing["reason"]


def test_visual_diff_boxes_scales_and_insets_the_region(tmp_path):
    reference = _png(tmp_path / "ref.png", size=(400, 200))
    answer = _png(tmp_path / "ans.png", size=(400, 200), box=(100, 40, 140, 80))
    svg = viz.visual_diff(reference, answer, thumb_w=200)
    assert "data-bbox='100,40,140,80'" in svg
    assert "data-changed='1600'" in svg
    # scale = 200 / 400 = 0.5
    assert "x='280.00'" in svg and "width='20.00'" in svg and "height='20.00'" in svg
    assert svg.count("data:image/png;base64,") == 3  # эталон, ответ, врезка
    assert "расхождение x 100–140, y 40–80" in svg


def test_visual_diff_without_data_says_so(tmp_path):
    svg = viz.visual_diff(tmp_path / "нет.png", tmp_path / "тоже.png")
    assert "нет данных" in svg and "data:image" not in svg


def test_frames_are_decoded_under_an_explicit_pixel_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    reference = _png(tmp_path / "ref.png")
    viz.diff_bbox(reference, reference)
    # PIL's own limit only warns; the panel applies its own, explicit one.
    assert Image.MAX_IMAGE_PIXELS == viz.MAX_IMAGE_PIXELS


def test_a_frame_that_does_not_fit_memory_says_no_data(monkeypatch, tmp_path):
    from PIL import Image

    reference = _png(tmp_path / "ref.png")
    answer = _png(tmp_path / "ans.png", box=(5, 2, 9, 6))
    monkeypatch.setattr(Image, "open", lambda *a, **k: (_ for _ in ()).throw(MemoryError("огромный кадр")))
    info = viz.diff_bbox(reference, answer)
    assert info["bbox"] is None and "нет данных" in info["reason"]
    assert "нет данных" in viz.visual_diff(reference, answer)


def test_a_truncated_frame_says_no_data(tmp_path):
    reference = _png(tmp_path / "ref.png")
    broken = tmp_path / "ans.png"
    broken.write_bytes(reference.read_bytes()[:40])  # заголовок есть, данных нет
    info = viz.diff_bbox(reference, broken)
    assert info["bbox"] is None and "нет данных" in info["reason"]
    assert "нет данных" in viz.visual_diff(reference, broken)


def _summary(**over):
    base = {
        "headline_earned": 40.0,
        "headline_possible": 100.0,
        "headline_ci95": (32.0, 48.0),
        "headline_n": 40,
        "headline_provisional": False,
        "per_category": {
            "A": {"earned": 2.0, "possible": 10.0, "tasks": 2, "skipped": 0},
            "B": {"earned": 8.0, "possible": 10.0, "tasks": 2, "skipped": 0},
            "E": {"earned": 0.0, "possible": 10.0, "tasks": 2, "skipped": 2},
        },
        "admission_state": "passed",
        "errors": 0,
        "skipped": 2,
        "full_suite": True,
    }
    base.update(over)
    return base


def test_verdict_strip_has_three_conclusions_and_three_risks_from_numbers():
    svg = viz.verdict_strip(_summary(), {"capture_failed": 1}, titles={"A": "инструкции", "B": "рассуждения"})
    conclusions = re.findall(r"data-kind='conclusion' data-index='\d+' data-text='([^']*)'", svg)
    risks = re.findall(r"data-kind='risk' data-index='\d+' data-text='([^']*)'", svg)
    assert len(conclusions) == 3 and len(risks) == 3
    assert "Качество 40 из 100 · ДИ 95% 32–48%" in conclusions[0]
    assert "Сильнейшая ось B рассуждения: 80%" in conclusions[1]
    assert "Слабейшая ось A инструкции: 20%" in conclusions[2]
    assert risks[0] == "категории не проверялись: E"
    assert risks[1] == "слабых осей (&lt;50%): 1 (A)"
    assert risks[2] == "страниц не отрисовалось: 1"


def test_verdict_strip_names_admission_and_small_sample():
    svg = viz.verdict_strip(_summary(admission_state="failed", headline_provisional=True, headline_n=4))
    risks = re.findall(r"data-kind='risk' data-index='\d+' data-text='([^']*)'", svg)
    assert "не прошёл допуск" in risks[0]
    assert "малый n=4" in risks[1]


def test_verdict_strip_survives_an_empty_summary():
    svg = viz.verdict_strip({})
    assert "нет данных" in svg
    assert len(re.findall(r"data-kind='(?:conclusion|risk)'", svg)) == 6
