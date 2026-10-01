"""Космический слайс (L-25..L-29): станция, Земля, катастрофы.

Проверки слайса держат то, что важно для человека, смотрящего кадр: планета и
ореол на месте, вспышка и обломки видны, сцена двигается и воспроизводится.
Баллов у арт-блока нет — здесь проверяется, что эталон и генератор согласованы.
"""

from __future__ import annotations

import random
import re

import pytest

from dailybench import art, browser, core, visual
from dailybench.tasks import art_space as module

SEED = 20260915
OTHER_SEED = 20261001
IDS = ("L-25", "L-26", "L-27", "L-28", "L-29")


def tasks(seed: int = SEED) -> list[core.TaskInstance]:
    return module.build(random.Random(seed))


def _answer(text: str) -> core.Transcript:
    transcript = core.Transcript(task_id="L-T", category="L")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=text)))
    return transcript


def test_ids_and_titles() -> None:
    assert tuple(task.id for task in tasks()) == IDS
    for task in tasks():
        assert task.category == "L"
        assert task.title.startswith(task.id)
        assert task.max_points == art.POINTS_PER_CHECK * len(art.CHECK_NAMES)


def test_prompts_name_the_scene_and_protocol() -> None:
    for task in tasks():
        prompt = task.prompt
        assert "window.__step" in prompt and "window.__probe()" in prompt
        assert "Math.random()" in prompt, f"{task.id}: промпт не запрещает Math.random()"
        assert "1103515245" in prompt, f"{task.id}: LCG не объявлен"
        assert "#" in prompt, f"{task.id}: палитра не названа"


def test_fixture_numbers_are_in_the_prompt() -> None:
    for task in tasks():
        meta = task.oracle["meta"]
        for key, value in meta.items():
            if key in ("driver", "browser", "steps", "estimated_tokens", "answer_tokens_estimate"):
                continue
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, list):
                for item in value:
                    text = f"{float(item):.3f}".rstrip("0").rstrip(".")
                    assert text in task.prompt or f"{float(item):.2f}" in task.prompt, (
                        f"{task.id}: {key} = {value} нет в промпте"
                    )
                continue
            assert str(value) in task.prompt or any(
                form in task.prompt for form in (f"{value}", f"{float(value):.2f}", f"{float(value):.3f}")
            ), f"{task.id}: {key} = {value} нет в промпте"


def test_palette_colours_are_declared() -> None:
    """Каждый цвет проверки палитры назван в промпте дословно."""
    # Цвет проверки виден только в её метке (detail при пустых пикселях общий).
    for task in tasks():
        for predicate in art.spec_of(task).palette:
            label = getattr(predicate, "check_label", "")
            found = re.findall(r"#[0-9a-f]{6}", label)
            assert found, f"{task.id}: в проверке палитры нет цвета ({label!r})"
            for hex_value in found:
                assert hex_value in task.prompt, f"{task.id}: {hex_value} не назван в промпте"


def test_geometry_stays_inside_the_frame() -> None:
    """Планета, станция и эффекты не вылезают за кадр и не вырождаются."""
    for task in tasks():
        meta = task.oracle["meta"]
        if task.id == "L-25":
            for key in ("boom_a", "boom_b"):
                assert len(meta[key]) == 2
                assert -0.6 <= meta[key][0] <= 0.6 and -0.5 <= meta[key][1] <= 0.6
            assert 0.0 < meta["shake"] <= 0.05
        elif task.id == "L-26":
            assert 1.0 < meta["impact_at"] < 3.0, "удар обязан случиться в пределах сцены"
            assert all(v < 0 for v in meta["meteor_v"]), "метеорит летит к планете"
        elif task.id == "L-27":
            assert meta["hit_at"] > 1.5, "планеты не сталкиваются мгновенно"
            assert len(meta["planet_a"]) == 2 and len(meta["planet_b"]) == 2
        elif task.id == "L-28":
            assert 1.5 <= meta["near"] <= 4.0
        elif task.id == "L-29":
            assert meta["travel"] > 3.0, "отъезд обязан быть медленным"
            assert len(meta["camera_to"]) == 2 and len(meta["camera_from"]) == 2


@pytest.mark.parametrize("seed", [SEED, OTHER_SEED])
def test_golden_scores_full(seed: int) -> None:
    if not browser.availability()[0]:
        pytest.skip("браузер недоступен")
    for task in tasks(seed):
        score = art.evaluate(task, _answer(task.oracle["final_answer"]))
        failed = [f"{c.name}: {c.detail}" for c in score.checks if not c.passed]
        assert score.earned == score.possible, f"{task.id} (seed {seed}) — {failed}"


@pytest.mark.parametrize("seed", [SEED, OTHER_SEED])
def test_garbage_scores_zero(seed: int) -> None:
    if not browser.availability()[0]:
        pytest.skip("браузер недоступен")
    for task in tasks(seed):
        score = art.evaluate(task, _answer("мусор без html"))
        assert score.earned == 0.0, f"{task.id}: мусор набрал {score.earned}"


def test_build_is_deterministic() -> None:
    first = tasks(SEED)
    second = tasks(SEED)
    assert [t.prompt for t in first] == [t.prompt for t in second]
    assert [t.oracle["golden_html"] for t in first] == [t.oracle["golden_html"] for t in second]


def test_seed_changes_the_fixture() -> None:
    first = {t.id: t.oracle["meta"] for t in tasks(SEED)}
    second = {t.id: t.oracle["meta"] for t in tasks(OTHER_SEED)}
    for task_id in IDS:
        assert first[task_id] != second[task_id], f"{task_id}: seed не влияет на фикстуру"


def test_goldens_are_offline_and_deterministic() -> None:
    for task in tasks():
        html = task.oracle["golden_html"]
        assert "./vendor/three.module.js" in html
        assert "preserveDrawingBuffer: true" in html
        assert "setSize(1920, 1080, false)" in html
        assert "Math.random" not in html
        assert "performance.now" not in html
        assert "__ready = true" in html


def test_palette_thresholds_keep_a_margin() -> None:
    """Порог палитры заметно ниже измеренного на эталоне."""
    if not browser.availability()[0]:
        pytest.skip("браузер недоступен")
    for task in tasks():
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        pixels = result.pixels or {}
        for predicate in spec.palette:
            ok, detail = predicate(pixels)
            assert ok, f"{task.id}: {detail}"


def test_atmosphere_card_is_filled() -> None:
    if not browser.availability()[0]:
        pytest.skip("браузер недоступен")
    for task in tasks():
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        card = art.atmosphere_scores(result.pixels or {}, result.animated)
        assert [row["criterion"] for row in card] == [row[0] for row in art.ATMOSPHERE_CRITERIA]
        for row in card:
            assert row["score"] >= 2, f"{task.id}: {row['criterion']} = {row['score']}"
