"""Tests for the L-13..L-16 slice — поля, вода и следы.

Каждый эталон обязан набрать 5/5 в браузере, а мусорный ответ — 0; пороги
палитры обязаны иметь запас от измеренной доли, а фикстура — лежать в промпте
дословно. Браузер недоступен → ``pytest.skip``.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from dailybench import art, browser, core
from dailybench.tasks import art_fields as mod

#: Seeds the fixtures are replayed on: the module must hold for all of them.
SEEDS = (7, 2, 13, 0, 5, 11)

#: Meta keys holding fixture inputs the answer has to hard-code.
FIXTURE_KEYS = {
    "L-13": ("count", "bands"),
    "L-14": ("horizon", "wave", "phase", "pitch"),
    "L-15": ("count", "phase", "seed", "speed", "life"),
    "L-16": ("count", "trail"),
}


def _tasks(seed: int) -> list[core.TaskInstance]:
    return mod.build(random.Random(seed))


def _transcript(task: core.TaskInstance, text: str) -> core.Transcript:
    return core.Transcript(
        task_id=task.id,
        category=task.category,
        turns=[core.Turn(messages=[], reply=core.ModelReply(text=text))],
    )


def _needs_browser() -> None:
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")


def _numbers(value: object) -> list[object]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    return []


# --------------------------------------------------------------------------- #
# Форма слайса
# --------------------------------------------------------------------------- #


def test_ids_points_and_prompt_contract() -> None:
    tasks = _tasks(7)
    assert mod.IDS == ("L-13", "L-14", "L-15", "L-16")
    assert [task.id for task in tasks] == list(mod.IDS)
    for task in tasks:
        assert task.category == "L"
        assert task.max_points == art.POINTS_PER_CHECK * len(art.CHECK_NAMES)
        spec = art.spec_of(task)
        assert spec.palette and spec.composition and spec.detail
        assert spec.motion > 0 and spec.dt > 0
        assert spec.need_same_frame
        assert "window.__step" in task.prompt and "window.__probe()" in task.prompt
        assert task.prompt.endswith(art.ART_CONTRACT)
        assert task.oracle["golden_html"].startswith("<!DOCTYPE html>")
        assert "./vendor/three.module.js" in task.oracle["golden_html"]


def test_every_check_has_its_point() -> None:
    """Пять проверок по 1.0: у каждой сцены свой набор предикатов."""
    for task in _tasks(7):
        spec = art.spec_of(task)
        assert len(spec.palette) >= 4, f"{task.id}: палитра из {len(spec.palette)} цветов"
        assert len(spec.composition) >= 2
        assert len(spec.detail) >= 3
        assert task.max_points == 5.0


def test_build_is_deterministic() -> None:
    first = _tasks(11)
    second = _tasks(11)
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle["golden_html"] for task in first] == [
        task.oracle["golden_html"] for task in second
    ]
    assert [task.oracle["meta"] for task in first] == [task.oracle["meta"] for task in second]


def test_fixtures_move_with_the_seed() -> None:
    for index in range(len(mod.IDS)):
        drawn = {json.dumps(_tasks(seed)[index].oracle["meta"], sort_keys=True) for seed in range(16)}
        assert len(drawn) >= 6, f"{mod.IDS[index]}: фикстуры почти не зависят от seed ({len(drawn)} из 16)"


def test_every_fixture_number_is_in_the_prompt() -> None:
    """Сценарий, у которого константы спрятаны, нерешаем."""
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            for key in FIXTURE_KEYS[task.id]:
                for number in _numbers(meta[key]):
                    assert str(number) in task.prompt, (
                        f"{task.id} seed={seed}: числа {number} ({key}) нет в промпте"
                    )


def test_palette_colours_are_distinguishable_by_hue() -> None:
    """В палитре нет коридоров: у каждой сцены минимум три разных тона.

    Внутри одной роли цвета могут стоять близко (тёмный и светлый оттенок воды,
    полоса заката и диск солнца), но кластеры обязаны быть различимы: проверка
    палитры сравнивает по тону и близкая пара не различила бы подмену.
    """
    for task in _tasks(7):
        colours = [check.check_label.split()[2] for check in art.spec_of(task).palette]
        hues = sorted(mod.visual._rgb_to_hsv(*mod.visual._hex_to_rgb(c))[0] for c in colours)
        # Кластеры: сосед ближе 25° — тот же тон, иначе новый кластер.
        clusters: list[float] = []
        for hue in hues:
            if not clusters or all(min(abs(hue - c), 360 - abs(hue - c)) >= 25.0 for c in clusters):
                clusters.append(hue)
        assert len(clusters) >= 3, f"{task.id}: всего {len(clusters)} различимых тона в {colours}"
        for i, first in enumerate(colours):
            for second in colours[i + 1 :]:
                ha = mod.visual._rgb_to_hsv(*mod.visual._hex_to_rgb(first))[0]
                hb = mod.visual._rgb_to_hsv(*mod.visual._hex_to_rgb(second))[0]
                gap = min(abs(ha - hb), 360 - abs(ha - hb))
                assert gap >= 8.0, f"{task.id}: цвета {first} и {second} различаются на {gap:.1f}°"
        gaps = sorted(
            min(abs(a - b), 360 - abs(a - b)) for i, a in enumerate(clusters) for b in clusters[i + 1 :]
        )
        assert gaps and max(gaps) >= 60.0, f"{task.id}: самый широкий разрыв тона {max(gaps):.0f}°"


# --------------------------------------------------------------------------- #
# Эталон: структура и бюджет
# --------------------------------------------------------------------------- #


def test_golden_pages_fit_the_answer_budget() -> None:
    for seed in SEEDS:
        for task in _tasks(seed):
            html = task.oracle["golden_html"]
            assert len(html.splitlines()) <= 160, f"{task.id}: эталон длиннее бюджета"
            assert "Math.random" not in html
            assert "performance.now" not in html
            assert "requestAnimationFrame" not in html
            assert "preserveDrawingBuffer: true" in html
            assert "setSize(1920, 1080, false)" in html
            assert "setPixelRatio(1)" in html
            assert "window.__ready = true" in html
            assert "__step" in html and "__probe" in html


def test_golden_draws_the_declared_scene() -> None:
    """Каждая сцена рисуется тем, что объявлено в промпте."""
    expected = {
        "L-13": ("PlaneGeometry", "smoothstep", "uBlobs"),
        "L-14": ("PlaneGeometry", "reflect", "uSunDir"),
        "L-15": ("InstancedMesh", "setColorAt", "atan2"),
        "L-16": ("SphereGeometry", "Points", "cos"),
    }
    for seed in SEEDS:
        for task in _tasks(seed):
            html = task.oracle["golden_html"]
            for token in expected[task.id]:
                assert token in html, f"{task.id}: эталон обязан рисовать сцену через {token}"


def test_golden_is_deterministic_in_the_prompt_numbers() -> None:
    """Числа фикстуры дословно уезжают и в эталон, а не только в промпт."""
    for seed in SEEDS:
        for task in _tasks(seed):
            html = task.oracle["golden_html"]
            if task.id == "L-13":
                for blob in task.oracle["meta"]["blobs"]:
                    for number in blob[:5]:
                        assert str(number) in html, f"{task.id}: {number} нет в эталоне"
            if task.id == "L-14":
                assert f"{task.oracle['meta']['pitch']:.4f}" in html
            if task.id == "L-15":
                assert str(task.oracle["meta"]["seed"]) in html
            if task.id == "L-16":
                for planet in task.oracle["meta"]["planets"]:
                    for number in planet[:4]:
                        assert str(number) in html, f"{task.id}: {number} нет в эталоне"


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #


def test_fixture_assertions_are_exercised() -> None:
    for seed in SEEDS:
        for task in _tasks(seed):
            meta = task.oracle["meta"]
            if task.id == "L-13":
                assert 6 <= meta["count"] <= 9
                assert len(meta["blobs"]) == meta["count"]
                assert all(b[3] * mod.STEPS / 60 > 0.08 for b in meta["blobs"])
                assert len(meta["bands"]) == 3
                assert all(a < b for a, b in meta["bands"])
            if task.id == "L-14":
                assert 0.3 <= meta["horizon"] <= 0.45
                assert meta["pitch"] > 0.0
                assert min(w / k for _, k, w, _ in mod.WAVES14) * mod.STEPS / 60 > 2.0
            if task.id == "L-15":
                assert 1200 <= meta["count"] <= 2500
                assert meta["speed"] * min(w[0] for w in mod.FIELD15) * mod.STEPS / 60 > 0.05
            if task.id == "L-16":
                assert 5 <= meta["count"] <= 8
                assert len(meta["planets"]) == meta["count"]
                assert len({p[4] for p in meta["planets"]}) == len(mod.COLOURS16)
                assert min(2 * math.pi * mod.STEPS / 60 / p[3] for p in meta["planets"]) > 0.6


# --------------------------------------------------------------------------- #
# Настоящая проверка: эталоны в браузере
# --------------------------------------------------------------------------- #


def test_garbage_answer_scores_zero() -> None:
    for task in _tasks(7):
        score = art.evaluate(task, _transcript(task, "не знаю"))
        assert score.earned == 0, f"{task.id}: мусорный ответ набрал {score.earned}"


@pytest.mark.parametrize("seed", SEEDS)
def test_golden_scores_full_in_browser(seed: int) -> None:
    _needs_browser()
    for task in _tasks(seed):
        score = art.evaluate(task, _transcript(task, task.oracle["final_answer"]))
        failed = [(check.name, check.detail) for check in score.checks if not check.passed]
        assert score.possible == task.max_points
        assert score.earned == task.max_points, f"{task.id} seed={seed}: {failed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_palette_thresholds_hold_with_margin(seed: int) -> None:
    """Порог палитры стоит ~вдвое ниже измеренной доли, а не вплотную."""
    _needs_browser()
    for task in _tasks(seed):
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        pixels = result.pixels or {}
        for check in spec.palette:
            good, detail = check(pixels)
            assert good, f"{task.id}: порог палитры не держится — {detail}"
            measured = float(detail.rsplit("занимает ", 1)[1].split()[0])
            wanted = float(detail.rsplit("нужно >= ", 1)[1].rstrip(")"))
            assert measured >= wanted, f"{task.id}: {detail}"
            assert measured >= 1.2 * wanted, f"{task.id}: нет запаса — {detail}"


@pytest.mark.parametrize("seed", SEEDS)
def test_unstepped_frame_is_flat_and_stepping_moves(seed: int) -> None:
    """Кадр обязан быть детальным с первого момента, а шаги — менять его."""
    _needs_browser()
    for task in _tasks(seed):
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        assert result.ok, f"{task.id}: рендер эталона не удался — {result.reason}"
        assert result.animated, f"{task.id}: сцена не двигается"
        assert result.shot_later_sha and result.shot_later_sha != result.shot_sha, (
            f"{task.id}: кадры на t и t+{spec.motion} совпали"
        )
        probe = browser.probe_of(result)
        stepped = browser.probe_of(result, stepped=True)
        assert isinstance(probe.get("t"), (int, float))
        assert float(stepped["t"]) > float(probe["t"]), f"{task.id}: время в __probe() не растёт"
        assert result.same_frame, f"{task.id}: кадр не воспроизводится между загрузками"


@pytest.mark.parametrize("seed", [7, 13])
def test_probe_reports_the_declared_state(seed: int) -> None:
    """__probe() обязан отдавать объявленные поля, а не только t."""
    _needs_browser()
    wanted = {"L-13": "blobs", "L-14": "wave", "L-15": "particles", "L-16": "orbits"}
    for task in _tasks(seed):
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        probe = browser.probe_of(result)
        assert wanted[task.id] in probe, f"{task.id}: в __probe() нет {wanted[task.id]}"
        assert isinstance(probe["objects"], list) and probe["objects"]
