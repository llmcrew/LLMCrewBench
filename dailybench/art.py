"""Shared machinery for the art block (category L, Three.js).

The simulation block (K) asks "is the physics right": every point comes from a
number recomputed in Python. The art block asks a different question — "is this
a working, deliberate picture?" — so it is scored from the frame itself, in five
checks of one point:

``html``
    the page is a whole offline document, loads without console errors and
    answers ``window.__probe()``;
``palette``
    the colours the prompt declared really cover the declared share of the frame;
``composition``
    the declared geometry is visible: a horizon and a gradient profile, a
    radial falloff, or a symmetry the frame actually has;
``detail``
    the frame is not a flat fill and not noise: edge density, colour count and
    saturation are inside the declared band;
``motion``
    the scene *animates*: frames at two moments differ, while every load
    reproduces its own frame byte for byte.

The palette is checked strictly (a model that paints the wrong hue loses the
point), the composition loosely (broad thresholds, so taste is not punished).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import browser, core, visual
from .visual import (
    HTML_CONTRACT,
    _predicates_report,
    cache_dir,
    extract_html,
    looks_like_document,
    page_of,
)

POINTS_PER_CHECK = 1.0
CHECK_NAMES = ("html", "palette", "composition", "detail", "motion")

#: The file an agent writes the scene into. Graded from disk: an agent run is
#: judged on what it produced, not on what it said.
PAGE_FILE = "index.html"

#: The free-form part of the art contract, reused verbatim when the prompt goes
#: to an agent instead of a chat endpoint.
_ART_FREEDOM = """Как это оценивают. Баллов у арт-блока нет — сцены смотрит человек. Проверки
только описывают кадр: палитра, композиция, детализация, движение. Поэтому
форма свободна: рисуй телами (коробки, сферы, торы), ставь свет и материалы,
строй настоящую перспективу — важно, чтобы кадр читался объёмным и со светом, а
не как плоская заливка. Цвета задания — ориентир по тону, точный процент не
требуется.
"""

ART_CONTRACT = (
    """Требования к странице (общие для всех арт-заданий):
1. Ответ — один самодостаточный документ `<html>`, без внешних ресурсов.
   Библиотека рядом: `import * as THREE from './vendor/three.module.js'`.
2. Рендерер: `new THREE.WebGLRenderer({preserveDrawingBuffer: true})`,
   `renderer.setPixelRatio(1)`, `renderer.setSize(1920, 1080, false)`, canvas в
   body. Кадр 1920x1080 при devicePixelRatio 1.
3. Детерминизм: две загрузки дают побитово одинаковый кадр для одного момента.
   Никаких `Math.random()` без явного seed, никакой зависимости от
   `performance.now()`.
4. Время двигает только `window.__step(dt)` — ровно на `dt` секунд, и
   возвращает `window.__probe()`. Кадр после нуля шагов — начало сцены; кадр
   после шагов обязан отличаться. Свой `requestAnimationFrame` для анимации не
   нужен.
5. `window.__probe()` возвращает `{"t": <4 знака>, "objects": [...]}` плюс поля,
   перечисленные в задании.
6. `window.__ready = true` — после первой отрисовки, когда `__probe()` отвечает.

"""
    + _ART_FREEDOM
)


@dataclass
class ArtSpec:
    """What one art scenario is scored against."""

    palette: list[Callable[[dict[str, Any]], tuple[bool, str]]]
    composition: list[Callable[[dict[str, Any]], tuple[bool, str]]]
    detail: list[Callable[[dict[str, Any]], tuple[bool, str]]]
    motion: int = 48
    """Steps taken before the second frame: the scene must move by then."""
    dt: float = 1 / 60
    need_same_frame: bool = True


#: Версия формата кадра. Меняется, когда в ``pixels`` появляются новые метрики:
#: старые записи кэша тогда не отдаются, и кадр считается заново.
FRAME_VERSION = 7


def _frame_key(html: str, steps: int, dt: float, mode: str) -> str:
    size = f"{browser.ART_VIEWPORT[0]}x{browser.ART_VIEWPORT[1]}"
    return hashlib.sha256(
        f"art: v{FRAME_VERSION}:{mode}:{size}:{steps}:{dt}:{html}".encode("utf-8")
    ).hexdigest()[:32]


def cached_art_render(html: str, steps: int, dt: float) -> tuple[browser.RenderResult, bool]:
    """Render once per unique page, with the animation pass included.

    A *failed* render never overwrites a good cache entry. The screenshot can
    time out under load while the page itself is fine, and the failure carries no
    probe and no pixels — caching it next to the good frame from an earlier
    attempt poisoned that page for every later run (``L-14`` seed 13 kept
    answering "рендер не удался" from an empty entry while a fresh render of the
    same page took 12 s and passed).
    """
    key = _frame_key(html, steps, dt, "time")
    folder = cache_dir()
    blob = folder / f"{key}.json"
    shot = folder / f"{key}.png"
    if blob.is_file():
        try:
            data = json.loads(blob.read_text(encoding="utf-8"))
            result = browser.RenderResult(**data["result"])
            if shot.is_file():
                result.shot = shot.read_bytes()
            if result.ok:
                return result, True
        except Exception:
            pass
    result = browser.render(html, steps=steps, dt=dt, mode="time", viewport=browser.ART_VIEWPORT)
    folder.mkdir(parents=True, exist_ok=True)
    # Кадр — до блоба и своим try: probe, который не сериализуется, не должен
    # оставлять галерею без картинки (см. тот же приём в visual._cached_render).
    if result.shot:
        shot.write_bytes(result.shot)
    try:
        payload = {"result": {
            "ok": result.ok,
            "reason": result.reason,
            "errors": result.errors,
            "external": result.external,
            "probe": result.probe,
            "stepped": result.stepped,
            "pixels": result.pixels,
            "pixels_again": result.pixels_again,
            "shot_sha": result.shot_sha,
            "shot_sha_again": result.shot_sha_again,
            "animated": result.animated,
            "shot_later_sha": result.shot_later_sha,
            "duration_s": result.duration_s,
        }}
        blob.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return result, False


def art_shot_path(html: str, steps: int, dt: float) -> Path:
    return cache_dir() / f"{_frame_key(html, steps, dt, 'time')}.png"



def _observe(transcript: core.Transcript, result: browser.RenderResult, spec: "ArtSpec", html: str) -> None:
    """Сохранить кадр и наблюдения для витрины — даже если проверки провалены.

    Витрина показывает сцену человеку: кадр нужен всегда, когда страница
    отрисовалась. Ранние выходы (нет ``__probe``, нет времени в нём) раньше
    оставляли ``transcript.extra['visual']`` пустым, и сцена пропадала из отчёта.
    """
    pixels = result.pixels if isinstance(result.pixels, dict) else {}
    path = art_shot_path(html, spec.motion, spec.dt)
    transcript.extra["visual"] = {
        "atmosphere": atmosphere_scores(pixels, result.animated),
        "shot": str(path) if path.is_file() else "",
        "steps": spec.motion,
        "dt": spec.dt,
        "pixels": pixels,
        "same_frame": result.same_frame,
        "animated": result.animated,
        "duration_s": round(result.duration_s, 3),
    }


def evaluate(task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
    """The five checks every art scenario shares."""
    score = core.Score(task_id=task.id, category=task.category)
    spec = ArtSpec(**task.oracle["spec"])

    usable, unavailable = browser.availability()
    if not usable:
        score.skipped = f"арт-блок не проверялся: {unavailable}"
        score.notes.append(unavailable)
        return score

    html = extract_html(page_of(task, transcript))
    static_ok, static_detail = looks_like_document(html)
    if not static_ok:
        score.add("html", False, POINTS_PER_CHECK, static_detail)
        for name in CHECK_NAMES[1:]:
            score.add(name, False, POINTS_PER_CHECK, "проверять нечего: страницы нет")
        score.notes.append("Ответ не содержит исполняемого документа")
        return score

    result, _ = cached_art_render(html, spec.motion, spec.dt)
    if not result.ok or result.pixels is None:
        reason = result.reason or "; ".join(result.errors[:3]) or "страница не отрисовалась"
        score.add("html", False, POINTS_PER_CHECK, f"рендер не удался: {reason}")
        for name in CHECK_NAMES[1:]:
            score.add(name, False, POINTS_PER_CHECK, "проверять нечего: страница не работает")
        score.notes.append(reason)
        return score

    _observe(transcript, result, spec, html)

    probe = browser.probe_of(result)
    breaks = not isinstance(probe.get("t"), (int, float))
    size = result.pixels or {}
    detail = f"кадр {size.get('width')}x{size.get('height')}"
    if result.external:
        score.add("html", False, POINTS_PER_CHECK, f"внешние запросы: {result.external[:3]}")
    elif breaks:
        # Страница рисует кадр, но не отвечает протоколом: сцену видно, движения
        # и состояния не проверить. Раньше это рушило весь проход и витрина
        # оставалась без кадра.
        score.add("html", False, POINTS_PER_CHECK, f"в __probe() нет времени t; {detail} отрисован")
    else:
        score.add("html", True, POINTS_PER_CHECK, f"документ загружен без ошибок; {detail}")

    if breaks:
        for name in ("palette", "composition", "detail"):
            ok, why = _predicates_report(spec.__getattribute__(name), result.pixels or {})
            score.add(name, ok, POINTS_PER_CHECK, why)
        score.add("motion", False, POINTS_PER_CHECK, "нет __probe: движение не подтверждено")
        score.notes.append("кадр снят, но протокол __probe не отвечает")
        path = art_shot_path(html, spec.motion, spec.dt)
        if path.is_file():
            score.notes.append(f"кадр сохранён: {path}")
        return score

    pixels = result.pixels if isinstance(result.pixels, dict) else {}
    palette_ok, palette_detail = _predicates_report(spec.palette, pixels)
    score.add("palette", palette_ok, POINTS_PER_CHECK, palette_detail)
    comp_ok, comp_detail = _predicates_report(spec.composition, pixels)
    score.add("composition", comp_ok, POINTS_PER_CHECK, comp_detail)
    detail_ok, detail_detail = _predicates_report(spec.detail, pixels)
    if spec.need_same_frame and not result.same_frame:
        detail_ok, detail_detail = False, f"кадр не воспроизводится: {detail_detail}"
    score.add("detail", detail_ok, POINTS_PER_CHECK, detail_detail)

    moved = result.shot_later_sha and result.shot_later_sha != result.shot_sha
    stepped = browser.probe_of(result, stepped=True)
    moved_probe = isinstance(stepped.get("t"), (int, float)) and float(stepped["t"]) > float(probe["t"])
    if result.animated and moved and moved_probe:
        score.add(
            "motion",
            True,
            POINTS_PER_CHECK,
            f"сцена движется: t={probe['t']} → {stepped['t']}, кадры различаются",
        )
    else:
        score.add(
            "motion",
            False,
            POINTS_PER_CHECK,
            f"сцена не движется: animated={result.animated}, кадры совпадают={not moved}, "
            f"t={probe.get('t')} → {stepped.get('t')}",
        )

    path = art_shot_path(html, spec.motion, spec.dt)
    score.notes.append(f"кадр сохранён: {path}")
    _observe(transcript, result, spec, html)
    return score


#: Человеческие критерии для атмосферных сцен (дождь, неон, уют): те же метрики
#: кадра, но сведённые в понятную карточку рядом с баллом.
ATMOSPHERE_CRITERIA = (
    ("свет", "shade", "плавные перепады яркости: затенение и градиенты, а не заливка"),
    ("объём", "volume", "форма читается: у поверхностей несколько ступеней яркости"),
    ("неон", "saturation", "насыщенность кадра: неон читается"),
    ("слои", "changed_fraction", "кадр не залит одним цветом: есть отражения и слои"),
    ("контраст тепло/холод", "contrast", "присутствуют и тёплые, и холодные тона"),
    ("детали", "distinct_colors", "мелкие детали: много различимых цветов"),
    ("движение", "animated", "кадр меняется во времени"),
)


def atmosphere_scores(pixels: dict[str, Any], animated: bool | None) -> list[dict[str, Any]]:
    """Human-readable verdict per criterion, computed from the frame statistics."""
    warm = float((pixels.get("warm") or {}).get("fraction") or 0.0)
    cool = float((pixels.get("cool") or {}).get("fraction") or 0.0)
    saturation = float(pixels.get("saturation") or 0.0)
    changed = float(pixels.get("changed_fraction") or 0.0)
    colors = float(pixels.get("distinct_colors") or 0.0)

    def scale(value: float, low: float, high: float) -> int:
        # 1 — «пусто», 5 — «выражено»; любое заметное значение получает минимум 2,
        # иначе узкий низ шкалы склеивает разные сцены в одну оценку.
        if value <= low:
            return 1
        if value >= high:
            return 5
        return max(2, 1 + round(4 * (value - low) / (high - low)))

    shade = float(pixels.get("shade") or 0.0)
    # Неон: доля насыщенных пикселей, а не средняя насыщенность кадра — в космосе
    # фон съедает среднее, хотя светящиеся акценты есть.
    vivid = sum(
        float(fraction) for _h, sat, val, fraction in (pixels.get("hue_histogram") or [])
        if sat >= 0.35 and val >= 0.35
    )
    volume = float((pixels.get("volume") or {}).get("shaded_share") or 0.0)
    verdicts = {
        "свет": scale(shade, 0.01, 0.18),
        "объём": scale(volume, 0.03, 0.80),
        "неон": scale(vivid, 0.004, 0.25),
        "слои": scale(changed, 0.05, 0.6),
        "контраст тепло/холод": 5 if warm > 0.002 and cool > 0.05 else (3 if warm > 0.0005 or cool > 0.02 else 1),
        "детали": scale(colors, 6, 60),
        "движение": 5 if animated else 1,
    }
    return [{"criterion": name, "score": verdicts[name], "note": note} for name, _key, note in ATMOSPHERE_CRITERIA]


def build_task(
    *,
    task_id: str,
    title: str,
    prompt: str,
    spec: ArtSpec,
    golden: str,
    fixture: dict[str, Any] | None = None,
) -> core.TaskInstance:
    """One art scenario, with its reference page as the oracle answer."""
    answer = f"```html\n{golden.strip()}\n```"
    from .providers import estimate_tokens

    prompt_text = f"{prompt.strip()}\n\n{ART_CONTRACT}"
    meta = dict(fixture or {})
    meta.update({"driver": "art", "browser": True, "steps": spec.motion})
    meta["estimated_tokens"] = estimate_tokens(prompt_text)
    meta["answer_tokens_estimate"] = estimate_tokens(answer)
    return core.TaskInstance(
        id=task_id,
        category="L",
        title=title,
        prompt=prompt_text,
        max_points=POINTS_PER_CHECK * len(CHECK_NAMES),
        oracle={
            "spec": {
                "palette": spec.palette,
                "composition": spec.composition,
                "detail": spec.detail,
                "motion": spec.motion,
                "dt": spec.dt,
                "need_same_frame": spec.need_same_frame,
            },
            "reference": answer,
            "final_answer": answer,
            "golden_html": golden,
            "label": title,
            "meta": meta,
            # An agent writes the page to this file and the pinned library is
            # staged next to it; a chat run ignores both.
            "answer_file": PAGE_FILE,
            "assets": ["vendor/three.module.js"],
        },
        meta=meta,
    )


def spec_of(task: core.TaskInstance) -> ArtSpec:
    return ArtSpec(**task.oracle["spec"])


def shared_html_contract() -> str:
    """Both blocks state the same page rules; the art block adds nothing to them."""
    return HTML_CONTRACT


def palette_hint() -> str:
    """The art-only part of the contract, for the agent prompt tail."""
    return _ART_FREEDOM.strip()


def contracts_differ_only_by_art() -> Sequence[str]:
    return CHECK_NAMES
