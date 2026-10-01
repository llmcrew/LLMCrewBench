"""Tests for the L-2..L-5 slice — геометрия и архитектура (неон, мандала, город,
мост).

Слайс проверяется своим тестом через ``art.evaluate``, а не через реестр
категории: так он работает, пока соседние слайсы ещё пишутся. Половина тестов
доказывает, что каждый эталон набирает 5/5 и что пороги стоят с запасом, вторая
— что смена seed двигает именно те числа, которые уходят в промпт.
"""

from __future__ import annotations

import random

import pytest

from dailybench import art, browser, core
from dailybench.tasks import art_geometry as module

#: Один фиксированный seed для «проходит ли эталон», второй — для «двигает ли
#: seed данные». Оба обязаны быть разными, иначе проверка фикстуры пустая.
SEED = 20240930
OTHER_SEED = SEED + 7
#: Полоса сидов для проверки запаса порогов: эталон обязан держать 5/5 на всех.
SWEEP = (0, 1, 2, 5, 7, 11, 13, 17, 23, 42)


def tasks(seed: int = SEED) -> list[core.TaskInstance]:
    return module.build(random.Random(seed))


def by_id(seed: int = SEED) -> dict[str, core.TaskInstance]:
    return {task.id: task for task in tasks(seed)}


def transcript_for(text: str, task_id: str = "L-2") -> core.Transcript:
    return core.Transcript(
        task_id=task_id,
        category="L",
        turns=[core.Turn(messages=[], reply=core.ModelReply(text=text))],
    )


@pytest.fixture(scope="module", autouse=True)
def browser_available() -> None:
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")


# --------------------------------------------------------------------------- #
# Форма слайса
# --------------------------------------------------------------------------- #


def test_ids_and_points() -> None:
    built = tasks()
    assert [task.id for task in built] == list(module.IDS)
    assert module.IDS == ("L-2", "L-3", "L-4", "L-5")
    for task in built:
        assert task.category == "L"
        assert task.max_points == len(art.CHECK_NAMES) * art.POINTS_PER_CHECK
        spec = art.spec_of(task)
        # Каждая группа обязана уметь провалиться: пустых групп нет.
        assert spec.palette and spec.composition and spec.detail
        assert spec.motion > 0 and spec.dt > 0
        assert spec.need_same_frame


def test_check_groups_cover_every_named_check() -> None:
    for task in tasks():
        spec = art.spec_of(task)
        groups = len(spec.palette) + len(spec.composition) + len(spec.detail)
        assert groups >= len(art.CHECK_NAMES)
        for check in (*spec.palette, *spec.composition, *spec.detail):
            assert getattr(check, "check_label", ""), "у предиката нет метки"


def test_golden_pages_fit_the_answer_budget() -> None:
    for task in tasks():
        golden = task.oracle["golden_html"]
        lines = golden.count("\n") + 1
        assert lines <= 165, f"{task.id}: эталон на {lines} строк — не влезет в бюджет ответа"
        assert "./vendor/three.module.js" in golden
        assert "preserveDrawingBuffer: true" in golden
        assert "setSize(1920, 1080, false)" in golden
        assert "setPixelRatio(1)" in golden
        assert "Math.random(" not in golden
        assert "performance.now" not in golden
        assert "__ready = true" in golden
        assert "window.__probe" in golden


# --------------------------------------------------------------------------- #
# Детерминизм и что именно двигает seed
# --------------------------------------------------------------------------- #


def test_build_is_deterministic() -> None:
    first, second = tasks(), tasks()
    assert [t.prompt for t in first] == [t.prompt for t in second]
    assert [t.oracle["golden_html"] for t in first] == [t.oracle["golden_html"] for t in second]
    assert [t.oracle["meta"] for t in first] == [t.oracle["meta"] for t in second]


def test_seed_changes_the_fixture() -> None:
    first, other = by_id(), by_id(OTHER_SEED)
    for task_id, task in first.items():
        twin = other[task_id]
        assert task.prompt != twin.prompt, f"{task_id}: смена seed не изменила данные"
        assert task.oracle["meta"] != twin.oracle["meta"]
        assert task.oracle["golden_html"] != twin.oracle["golden_html"]

    # ... и двигается ровно та фикстура, которую видит проверка: числа попали
    # и в промпт, и в эталонную страницу.
    l2_a, l2_b = first["L-2"].oracle["meta"], other["L-2"].oracle["meta"]
    assert (l2_a["speed"], l2_a["sun_y"], l2_a["sun_fall"]) != (l2_b["speed"], l2_b["sun_y"], l2_b["sun_fall"])
    l3_a, l3_b = first["L-3"].oracle["meta"], other["L-3"].oracle["meta"]
    assert (l3_a["ring_r"], l3_a["short_r"]) != (l3_b["ring_r"], l3_b["short_r"])
    l4_a, l4_b = first["L-4"].oracle["meta"], other["L-4"].oracle["meta"]
    assert [h["h"] for h in l4_a["houses"]] != [h["h"] for h in l4_b["houses"]]
    assert (l4_a["phase"], l4_a["haze_y"]) != (l4_b["phase"], l4_b["haze_y"])
    l5_a, l5_b = first["L-5"].oracle["meta"], other["L-5"].oracle["meta"]
    assert (l5_a["arch_r"], l5_a["omega"]) != (l5_b["arch_r"], l5_b["omega"])


# --------------------------------------------------------------------------- #
# Фикстура: числа обязаны быть в промпте и осмысленны
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", SWEEP)
def test_fixture_numbers_are_in_the_prompt(seed: int) -> None:
    import re

    number = re.compile(r"\d+(?:\.\d+)?")
    for task in tasks(seed):
        meta = task.oracle["meta"]
        wanted: list[float] = []
        for key, value in meta.items():
            if key in ("estimated_tokens", "answer_tokens_estimate", "steps"):
                continue
            if isinstance(value, float):
                wanted.append(value)
            elif isinstance(value, list):
                wanted += [float(house[field]) for house in value for field in ("x", "w", "h")]
        present = sorted({float(token) for token in number.findall(task.prompt)})
        for value in wanted:
            # число может стоять в промпте как доля или как процент — важно,
            # что модель видит ровно ту цифру, которую ждёт проверка.
            near = any(
                abs(item - value) < 0.005
                or abs(item - value * 100) < 0.05
                or abs(item - abs(value)) < 0.005
                for item in present
            )
            assert near, f"{task.id}: {value} ({value * 100:.1f}%) нет в промпте"
        for colour in task.oracle["spec"]["palette"]:
            assert getattr(colour, "check_label", "")


def test_l2_fixture_is_sound() -> None:
    for seed in SWEEP:
        meta = by_id(seed)["L-2"].oracle["meta"]
        assert 0.30 <= meta["sun_y"] <= 0.45, "солнце обязано стоять над горизонтом"
        travel = meta["speed"] * module.STEPS / 60
        assert travel >= 0.6, "за проверку сетка обязана сдвинуться"
        assert meta["sun_fall"] * module.STEPS / 60 >= 0.03, "солнце обязано опуститься"


def test_l3_fixture_is_sound() -> None:
    import math

    for seed in SWEEP:
        meta = by_id(seed)["L-3"].oracle["meta"]
        assert module.L3_R_IN < meta["short_r"] <= module.L3_R_OUT
        assert module.L3_R_IN < meta["ring_r"] < module.L3_R_OUT
        # Поворот за проверку кратен 15°: иначе лучи не совпадут сами с собой и
        # зеркальная симметрия кадра развалится.
        turn = math.degrees(module.L3_OMEGA * module.STEPS / 60)
        assert abs((turn + 7.5) % 15.0 - 7.5) < 0.05, "поворот не кратен 15°"
        assert turn >= 10.0


def test_l4_fixture_is_sound() -> None:
    for seed in SWEEP:
        meta = by_id(seed)["L-4"].oracle["meta"]
        houses = meta["houses"]
        assert 6 <= len(houses) <= 7
        assert all(house["h"] > 8.0 for house in houses)
        assert len({house["h"] for house in houses}) == len(houses), "высоты обязаны различаться"
        assert max(abs(house["x"]) + house["w"] / 2 for house in houses) <= 14.0, "ряд влезает в кадр"
        assert meta["blink"] * module.STEPS / 60 >= 0.6, "фаза окон обязана сместиться"


def test_l5_fixture_is_sound() -> None:
    import math

    for seed in SWEEP:
        meta = by_id(seed)["L-5"].oracle["meta"]
        assert 2.8 <= meta["arch_r"] <= 4.0
        assert abs(meta["hang"] - meta["arch_r"] * 0.72) < 0.01
        # подвес упирается в арку: hang² + arch_h² = arch_r² с точностью округления
        assert abs(meta["arch_h"] ** 2 + meta["hang"] ** 2 - meta["arch_r"] ** 2) < 0.05, "подвес упирается в арку"
        rotation = math.degrees(meta["omega"] * module.STEPS / 60)
        assert rotation >= 5.5, "поворот обязан быть виден на кадре"


# --------------------------------------------------------------------------- #
# Настоящая проверка: эталон обязан набрать всё, мусор — ничего
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", (SEED, OTHER_SEED, *SWEEP))
def test_golden_scores_full(seed: int) -> None:
    for task in tasks(seed):
        score = art.evaluate(task, transcript_for(task.oracle["reference"], task.id))
        failed = [f"{check.name}: {check.detail}" for check in score.checks if not check.passed]
        assert score.earned == task.max_points, f"{task.id} (seed {seed}) — {failed}"
        assert score.possible == task.max_points


@pytest.mark.parametrize("seed", (SEED, OTHER_SEED))
def test_garbage_scores_zero(seed: int) -> None:
    for task in tasks(seed):
        score = art.evaluate(task, transcript_for("Не знаю, как это сделать, извините.", task.id))
        assert score.earned == 0, f"{task.id}: мусорный ответ набрал {score.earned}"
        assert score.possible == task.max_points


def test_frames_are_reproducible() -> None:
    """Арт-задание требует побитово одинакового кадра на двух загрузках."""
    for seed in (SEED, OTHER_SEED):
        for task in tasks(seed):
            spec = art.spec_of(task)
            result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
            assert result.ok, f"{task.id}: страница не отрисовалась — {result.reason}"
            assert result.same_frame, f"{task.id}: кадр не воспроизводится"
            assert result.shot_sha and result.shot_sha == result.shot_sha_again


# --------------------------------------------------------------------------- #
# Пороги стоят с запасом от измеренного, а не вплотную
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", (SEED, OTHER_SEED))
def test_palette_thresholds_have_margin(seed: int) -> None:
    """Объявленная доля обязана быть примерно вдвое ниже измеренной: иначе
    вырожденная сцена пройдёт случайно, а живая — не пройдёт вовсе."""
    for task in tasks(seed):
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        assert result.ok and result.pixels
        for check in spec.palette:
            ok, detail = check(result.pixels)
            assert ok, f"{task.id}: палитра провалена — {detail}"
            share = float(detail.rsplit("нужно >= ", 1)[1].rstrip(")"))
            assert share <= 0.55, f"{task.id}: порог выше половины кадра — сцена однотонна"


@pytest.mark.parametrize("seed", (SEED, OTHER_SEED))
def test_composition_thresholds_are_loose(seed: int) -> None:
    """Композиция — широкие пороги: ни один предикат не должен сидеть на самом
    краю (иначе соседний seed его сломает)."""
    for task in tasks(seed):
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        assert result.ok and result.pixels
        pixels = result.pixels
        profile = [entry["lum"] for entry in pixels["profile"]]
        assert len(profile) == 8
        # Кадр не однотонный и не шум: профиль ходит, но плавно.
        assert max(profile) - min(profile) >= 15.0, f"{task.id}: профиль слишком ровный"
        assert pixels["changed_fraction"] >= 0.10, f"{task.id}: сцена на однотонном фоне"
        assert 0.05 <= pixels["mirror_difference"] <= 25.0, f"{task.id}: кадр либо симметричен точно, либо нет"


def test_motion_is_visible_and_time_grows() -> None:
    """Кадр на t+motion шагов обязан отличаться, а время в пробе — расти."""
    for task in tasks():
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        assert result.ok and result.animated, f"{task.id}: сцена не двигается"
        assert result.shot_later_sha and result.shot_later_sha != result.shot_sha
        assert browser.probe_of(result, stepped=True)["t"] > browser.probe_of(result)["t"]


@pytest.mark.parametrize("seed", (SEED, OTHER_SEED))
def test_prompt_states_the_palette_and_the_motion(seed: int) -> None:
    for task in tasks(seed):
        prompt = task.prompt
        meta = task.oracle["meta"]
        assert "Палитра" in prompt and "Проверяется" in prompt
        assert str(meta["steps"]) in prompt, f"{task.id}: число шагов проверки не объявлено"
        assert "window.__step" in prompt and "window.__probe" in prompt
        # Движение описано словами, а не оставлено на догадку модели.
        assert "Анимация" in prompt
