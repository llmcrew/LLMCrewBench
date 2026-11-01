"""Capture a model's free-form artwork; artistic quality is judged by people.

The only automatic check is whether the submitted HTML page renders. No scene
API, reference palette, composition, animation or reproducibility is required.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import agent, browser, core
from .visual import _frame_key as _visual_frame_key
from .visual import cache_dir, extract_html

POINTS_PER_CHECK = 1.0
CHECK_NAMES = ("html",)
PAGE_FILE = "index.html"
FRAME_VERSION = 8

ART_CONTRACT = """Результат — самодостаточная HTML-страница. Способ рисования выбирай сам:
HTML/CSS, SVG, Canvas, WebGL или их сочетание. Если нужен Three.js, локальная
библиотека доступна по пути `./vendor/three.module.js`. Готовые изображения и
другие внешние ресурсы не подставляй вместо собственной работы.
"""


def _frame_key(html: str) -> str:
    """Общий render fingerprint плюс версия арт-контракта и live-режим."""
    return _visual_frame_key(
        html, 0, 1 / 60, viewport=browser.ART_VIEWPORT, mode="live", kind=f"art:v{FRAME_VERSION}"
    )


def art_shot_path(html: str) -> Path:
    return cache_dir() / f"{_frame_key(html)}.png"


def cached_art_render(html: str) -> tuple[browser.RenderResult, bool]:
    """Keep a captured frame, not an assertion that later loads look identical."""
    key = _frame_key(html)
    folder = cache_dir()
    blob = folder / f"{key}.json"
    shot = folder / f"{key}.png"
    if blob.is_file() and shot.is_file():
        try:
            data = json.loads(blob.read_text(encoding="utf-8"))
            result = browser.RenderResult(**data["result"])
            if result.ok:
                result.shot = shot.read_bytes()
                # Хеш — от сохранённых байт: кадр остаётся доказательством и
                # без возможности перерисовать страницу заново.
                if result.shot:
                    result.shot_sha = hashlib.sha256(result.shot).hexdigest()
                return result, True
        except (OSError, ValueError, TypeError):
            pass
    result = browser.render(html, mode="live", viewport=browser.ART_VIEWPORT, again=False)
    folder.mkdir(parents=True, exist_ok=True)
    if result.shot:
        shot.write_bytes(result.shot)
    try:
        payload = {key: value for key, value in vars(result).items() if key != "shot"}
        blob.write_text(json.dumps({"result": payload}, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError, TypeError):
        # A captured frame remains usable even when its metadata cannot be saved.
        pass
    # A fresh render that captured a frame is a cache write, not a miss: the
    # next identical call must hit it. Returning False unconditionally made
    # every first render report a miss even when the frame was saved.
    return result, bool(result.ok and result.shot)


def _page_of(task: core.TaskInstance, transcript: core.Transcript) -> str:
    if transcript.extra.get("harness") == "agent":
        return agent.answer_text(task, transcript, "").strip()
    return extract_html(transcript.final_text)


def _observe(transcript: core.Transcript, result: browser.RenderResult, html: str) -> None:
    path = art_shot_path(html)
    transcript.extra["visual"] = {
        "shot": str(path) if path.is_file() else "",
        "capture_mode": "live",
        "duration_s": round(result.duration_s, 3),
    }


def _inline_image(html: str) -> str:
    """Prefix of a ready-made image embedded in the page, or empty."""
    match = re.search(
        r"""(?:src|href)\s*=\s*['"]?(data:image/|blob:)""", html, re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"""url\(\s*['"]?(data:image/|blob:)""", html, re.IGNORECASE,
        )
    return match.group(1) if match else ""


def evaluate(task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
    """Check delivery/rendering only; do not assign a score to artistic taste."""
    score = core.Score(task_id=task.id, category=task.category)
    usable, unavailable = browser.availability()
    if not usable:
        score.skipped = f"арт-блок не проверялся: {unavailable}"
        score.notes.append(unavailable)
        return score

    html = _page_of(task, transcript)
    lowered = html.lstrip().lower()
    if not lowered.startswith(("<!doctype html", "<html")):
        score.add("html", False, POINTS_PER_CHECK, "нет самостоятельного HTML-документа")
        return score
    inline = _inline_image(html)
    if inline:
        score.add(
            "html", False, POINTS_PER_CHECK,
            f"готовое изображение ({inline}) не является работой",
        )
        return score

    result, _ = cached_art_render(html)
    _observe(transcript, result, html)
    loaded = result.ok and bool(result.shot) and not result.external
    reason = result.reason or "; ".join(result.errors[:3])
    if result.external:
        reason = f"внешние ресурсы: {result.external[:3]}"
    score.add(
        "html",
        loaded,
        POINTS_PER_CHECK,
        "страница загружена, кадр сохранён; художественное качество оценивает человек"
        if loaded else reason or "страница не отрисовалась",
    )
    score.notes.append("Цвет, тон, композиция и выразительность не оцениваются автоматическими баллами.")
    return score


def build_task(
    *,
    task_id: str,
    title: str,
    prompt: str,
    golden: str,
) -> core.TaskInstance:
    """A free brief plus a rendering fixture, not a canonical artistic answer."""
    from .providers import estimate_tokens

    answer = f"```html\n{golden.strip()}\n```"
    prompt_text = f"{prompt.strip()}\n\n{ART_CONTRACT}"
    return core.TaskInstance(
        id=task_id,
        category="L",
        title=title,
        prompt=prompt_text,
        max_points=POINTS_PER_CHECK,
        oracle={
            "reference": answer,
            "final_answer": answer,
            "golden_html": golden,
            "answer_file": PAGE_FILE,
            "assets": ["vendor/three.module.js"],
        },
        meta={
            "driver": "art",
            "browser": True,
            "capture_mode": "live",
            "estimated_tokens": estimate_tokens(prompt_text),
            "answer_tokens_estimate": estimate_tokens(answer),
        },
    )
