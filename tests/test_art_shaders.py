"""Тесты слайса L-10..L-12 — шейдеры и материалы.

Слайс проверяется сам по себе (через ``art.evaluate``), не через реестр: соседние
модули блока ещё пишутся. Половина тестов — форма слайса и фикстура, половина —
настоящий рендер эталонов в headless-браузере.
"""

from __future__ import annotations

import random

import pytest

from dailybench import art, browser, core
from dailybench.tasks import art_shaders as module

SEED = 20240930
OTHER_SEED = SEED + 7


def tasks(seed: int = SEED) -> list[core.TaskInstance]:
    return module.build(random.Random(seed))


def by_id(seed: int = SEED) -> dict[str, core.TaskInstance]:
    return {task.id: task for task in tasks(seed)}


def transcript_for(text: str, task_id: str = "L-10") -> core.Transcript:
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
    assert tuple(task.id for task in built) == module.IDS
    assert module.IDS == ("L-10", "L-11", "L-12")
    for task in built:
        assert task.category == "L"
        assert task.max_points == 5.0
        spec = art.spec_of(task)
        # Каждая группа проверок непустая: сценарий обязан уметь падать.
        assert spec.palette and spec.composition and spec.detail
        assert spec.motion > 0 and spec.dt > 0
        assert spec.need_same_frame


def test_check_groups_sum_to_max_points() -> None:
    for task in tasks():
        spec = art.spec_of(task)
        total = len(art.CHECK_NAMES) * art.POINTS_PER_CHECK
        assert abs(total - task.max_points) < 1e-9
        assert len(spec.palette) + len(spec.composition) + len(spec.detail) >= len(art.CHECK_NAMES)


def test_golden_pages_fit_the_answer_budget() -> None:
    for task in tasks():
        html = task.oracle["golden_html"]
        assert html.count("\n") + 1 <= 150, f"{task.id}: эталон не влезает в бюджет ответа"
        assert "./vendor/three.module.js" in html
        assert "preserveDrawingBuffer: true" in html
        assert "setSize(1920, 1080, false)" in html
        assert "setPixelRatio(1)" in html
        assert "Math.random" not in html
        assert "performance.now" not in html
        assert "__ready = true" in html


def test_slice_uses_shaders_and_physical_material() -> None:
    """Слайс — про шейдеры и материалы: GLSL и MeshPhysicalMaterial в эталонах."""
    html = " ".join(task.oracle["golden_html"] for task in tasks())
    assert "ShaderMaterial" in html
    assert "gl_FragColor" in html
    assert "MeshPhysicalMaterial" in html
    assert "DataTexture" in html


# --------------------------------------------------------------------------- #
# Детерминизм и что именно двигает seed
# --------------------------------------------------------------------------- #


def test_build_is_deterministic() -> None:
    first, second = tasks(), tasks()
    assert [t.prompt for t in first] == [t.prompt for t in second]
    assert [t.oracle["golden_html"] for t in first] == [t.oracle["golden_html"] for t in second]
    assert [t.oracle["meta"] for t in first] == [t.oracle["meta"] for t in second]


#: Which fixture numbers the seed is expected to move, per scenario.
MOVING = {
    "L-10": ("zoom", "rate"),
    "L-11": ("roughness", "spin", "light_spin", "phase"),
    "L-12": ("density", "speed"),
}


def test_seed_changes_the_fixture() -> None:
    first, other = by_id(), by_id(OTHER_SEED)
    for task_id, task in first.items():
        twin = other[task_id]
        assert task.prompt != twin.prompt, f"{task_id}: смена seed не изменила данные"
        assert task.oracle["meta"] != twin.oracle["meta"]
        assert task.oracle["golden_html"] != twin.oracle["golden_html"]


def test_seed_moves_every_fixture_number() -> None:
    """Ни одно число фикстуры не приколочено: у каждого есть хотя бы два значения."""
    seen: dict[str, dict[str, set]] = {tid: {key: set() for key in keys} for tid, keys in MOVING.items()}
    for seed in range(1, 40):
        for task in tasks(seed):
            for key in MOVING[task.id]:
                seen[task.id][key].add(task.oracle["meta"][key])
    for task_id, groups in seen.items():
        for key, values in groups.items():
            assert len(values) >= 2, f"{task_id}: {key} не зависит от seed ({values})"


# --------------------------------------------------------------------------- #
# Доказательства фикстуры: числа в промпте, попадание в кадр
# --------------------------------------------------------------------------- #


def test_fixture_numbers_are_in_the_prompt() -> None:
    for task in tasks():
        prompt = task.prompt
        meta = task.oracle["meta"]
        for key, value in meta.items():
            if key in ("driver", "browser", "steps", "estimated_tokens", "answer_tokens_estimate"):
                continue
            if isinstance(value, float):
                forms = [str(value), f"{value:.2f}", f"{value:.3f}", f"{value:g}"]
                assert any(form in prompt for form in forms), f"{task.id}: {key} = {value} нет в промпте дословно"
            elif isinstance(value, str):
                assert value in prompt, f"{task.id}: {key} = {value} нет в промпте"
            elif isinstance(value, list):
                for item in value:
                    assert f"{item:.3f}" in prompt, f"{task.id}: {key} = {item} нет в промпте дословно"
            else:
                assert str(value) in prompt, f"{task.id}: {key} = {value} нет в промпте"


def test_palette_colours_are_declared_and_distinct() -> None:
    """Палитра — 3..5 цветов, различимых по тону: иначе проверка не различит подмену."""
    from dailybench.visual import _hex_to_rgb, _rgb_to_hsv

    for task in tasks():
        spec = art.spec_of(task)
        assert 3 <= len(spec.palette) <= 5, f"{task.id}: палитра вне 3..5 цветов"
        for hex_value, share in {
            "L-10": module.L10_PALETTE,
            "L-11": module.L11_PALETTE,
            "L-12": module.L12_PALETTE,
        }[task.id].items():
            assert hex_value in task.prompt, f"{task.id}: {hex_value} не назван в промпте"
            assert f"{share:.0%}" in task.prompt, f"{task.id}: доля {share} не названа в промпте"


def test_palette_hues_are_far_apart() -> None:
    from dailybench.visual import _hex_to_rgb, _rgb_to_hsv

    for task in tasks():
        palette = {
            "L-10": module.L10_PALETTE,
            "L-11": module.L11_PALETTE,
            "L-12": module.L12_PALETTE,
        }[task.id]
        hues = [(_rgb_to_hsv(*_hex_to_rgb(hex_value))[0], hex_value) for hex_value in palette]
        for i, (hue_a, name_a) in enumerate(hues):
            for hue_b, name_b in hues[i + 1 :]:
                delta = min(abs(hue_a - hue_b), 360 - abs(hue_a - hue_b))
                assert delta >= 60 or _rgb_to_hsv(*_hex_to_rgb(name_a))[1] < 0.3 or _rgb_to_hsv(
                    *_hex_to_rgb(name_b)
                )[1] < 0.3, f"{task.id}: {name_a} и {name_b} различаются на {delta:.0f}°"


def test_motion_is_visible() -> None:
    """За объявленное движение фикстура обязана сдвинуться заметно."""
    ten = by_id()["L-10"].oracle["meta"]
    assert ten["rate"] * module.L10_STEPS / 60.0 > 0.1, "зум не успеет вырасти"
    eleven = by_id()["L-11"].oracle["meta"]
    assert eleven["spin"] * module.L11_STEPS / 60.0 > 0.5, "сфера не успеет повернуться"
    twelve = by_id()["L-12"].oracle["meta"]
    assert twelve["speed"] * module.L12_STEPS / 60.0 > module.L12_PITCH * 1.5, "кольца не пролетят и шага"


# --------------------------------------------------------------------------- #
# Настоящий рендер: эталон 5/5, мусор 0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [SEED, OTHER_SEED])
def test_golden_scores_full(seed: int) -> None:
    for task in tasks(seed):
        score = art.evaluate(task, transcript_for(task.oracle["reference"], task.id))
        failed = [f"{check.name}: {check.detail}" for check in score.checks if not check.passed]
        assert score.earned == task.max_points, f"{task.id} (seed {seed}) — {failed}"
        assert score.possible == task.max_points


@pytest.mark.parametrize("seed", [SEED, OTHER_SEED])
def test_garbage_scores_zero(seed: int) -> None:
    for task in tasks(seed):
        score = art.evaluate(task, transcript_for("Не знаю, как это сделать, извините.", task.id))
        assert score.earned == 0, f"{task.id}: мусорный ответ набрал {score.earned}"
        assert score.possible == task.max_points


@pytest.mark.parametrize("seed", [SEED, OTHER_SEED])
def test_frames_are_reproducible(seed: int) -> None:
    for task in tasks(seed):
        spec = art.spec_of(task)
        result, _ = art.cached_art_render(task.oracle["golden_html"], spec.motion, spec.dt)
        assert result.ok, f"{task.id}: страница не отрисовалась — {result.reason or result.errors[:2]}"
        assert not result.external, f"{task.id}: внешние запросы {result.external[:2]}"
        assert result.same_frame, f"{task.id}: кадр не воспроизводится"
        assert result.shot_sha and result.shot_sha == result.shot_sha_again
        assert result.animated, f"{task.id}: сцена не движется"


# --------------------------------------------------------------------------- #
# Геометрия фикстуры: заворот туннеля и область фрактала
# --------------------------------------------------------------------------- #


def test_ring_wrap_keeps_rings_inside_the_span() -> None:
    """Формула заворота из промпта держит z в объявленном коридоре."""
    count, pitch, far = module.L12_COUNT, module.L12_PITCH, module.L12_Z_FAR
    span = count * pitch
    for speed in module.L12_SPEEDS:
        for step in range(0, 600, 7):
            for i in range(count):
                z = far + ((i * pitch + (step / 60) * speed) % span)
                assert far <= z <= far + span
                assert z < module.L12_CAMERA_Z, "кольцо не должно оказаться за камерой"


def test_fractal_covers_the_requested_region() -> None:
    """Кадр с орто-камерой -1..1 покрывает объявленную область, а зум её приближает."""
    for zoom in module.L10_ZOOMS:
        half = module.L10_HALF / zoom
        width = 2 * half * (800 / 600)
        assert 0 < half <= module.L10_HALF, "зум только приближает"
        assert width <= 4.0, "кадр не должен вылезать за квадрат 2x2"
        # центр области остаётся внутри кадра при любом зуме, и сама область
        # целиком попадает в квадрат 2x2 (граница множества |c| <= 2)
        cx, cy = module.L10_CENTRE
        assert abs(cx) + half * (800 / 600) < 2.0 and abs(cy) + half < 2.0
        assert 0.0 < zoom <= 1.5 and module.L10_ITER >= 64


def test_log_scale_colour_bands_are_ordered() -> None:
    """Полосы раскраски монотонны, лежат внутри шкалы и совпадают с промптом.

    Границы вычитаны из самого шейдера: если их сдвинуть в странице и забыть
    в промпте (или наоборот), тест падает.
    """
    import re

    for seed in (SEED, OTHER_SEED):
        task = by_id(seed)["L-10"]
        prompt, html = task.prompt, task.oracle["golden_html"]
        edges = [(float(low), float(high)) for low, high in re.findall(r"smoothstep\((\d\.\d+), (\d\.\d+), t\)", html)]
        assert len(edges) == 4, f"в шейдере {len(edges)} переходов, ожидалось 4"
        assert [low for low, _ in edges] == sorted(low for low, _ in edges), "переходы не по порядку"
        assert edges[0][0] > 0.0 and edges[-1][1] < 1.0, "переходы обязаны лежать внутри шкалы t"
        for low, high in edges:
            assert low < high
            assert f"{low:.2f}..{high:.2f}" in prompt, f"границы {low:.2f}..{high:.2f} не объявлены в промпте"
        # Шкала t нормирована: log(161) в шейдере и в промпте — одно число.
        assert "log(161.0)" in html and str(module.L10_ITER + 1) in prompt
