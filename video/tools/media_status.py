#!/usr/bin/env python3
"""Единый контракт статуса медиа витрины.

Статус сцены хранится маркером `<модель>__<сцена>.missing` (содержимое — причина)
и переносится в `src/data/showcase.json` (единственный источник для витрины и
preflight). `public/clips.json` — производная проекция статусов для внешних
потребителей, отдельного источника истины у него нет.

Причины не смешиваются: `not_run` — задачи нет в прогоне, `model_failed` —
задача есть, но страница не доставлена или падает до кадра, `capture_failed` —
страница жива, а запись сорвалась, `media_missing` — клипа нет на диске.
Исторические маркеры с содержимым `1` разрешаются по scores прогона.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

#: Все допустимые статусы. Первые два — живая запись и штатно пропущенная задача.
STATUSES = (
    "ok",
    "skipped",
    "not_run",
    "model_failed",
    "evaluator_unavailable",
    "capture_failed",
    "media_missing",
)
#: Статусы, которым соответствует плашка и маркер (клип при этом — заглушка).
PLACEHOLDER_STATUSES = frozenset(
    {"skipped", "not_run", "model_failed", "evaluator_unavailable", "capture_failed"}
)
#: Историческое содержимое маркера: причина не записана.
LEGACY_MARKER = "1"


def clip_path(out_dir: Path, model: str, scene: str) -> Path:
    return out_dir / f"{model}__{scene}.mp4"


def marker_path(out_dir: Path, model: str, scene: str) -> Path:
    return out_dir / f"{model}__{scene}.missing"


def task_status(model: str, scene: str, runs_root: Path | None) -> str | None:
    """Статус задачи по scores прогона; ``None`` — данных прогона нет."""
    if runs_root is None:
        return None
    path = runs_root / model / "scores.json"
    if not path.is_file():
        return None
    try:
        scores = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for row in scores:
        if row.get("task_id") != scene:
            continue
        if row.get("skipped"):
            return "skipped"
        error = (row.get("error") or "").lower()
        if any(word in error for word in ("evaluator", "недоступ", "unavailable")):
            return "evaluator_unavailable"
        return "model_failed" if row.get("error") or not row.get("possible") else "ok"
    return "not_run"


def status_of(
    out_dir: Path, model: str, scene: str, runs_root: Path | None = None
) -> str:
    """Статус пары модель/сцена: маркер главнее файлов, отсутствие клипа — ошибка."""
    marker = marker_path(out_dir, model, scene)
    clip = clip_path(out_dir, model, scene)
    if marker.is_file():
        try:
            raw = marker.read_text(encoding="utf-8").strip()
        except OSError:
            raw = ""
        if raw in STATUSES and raw != "ok":
            return raw
        # Причина может нести улику после двоеточия (`model_failed: диалог: …`):
        # статус — только её первое слово.
        head = raw.split(":", 1)[0].strip()
        if head in STATUSES and head != "ok":
            return head
        # Legacy `1`: причину восстанавливаем по прогону. Нет данных — честнее
        # назвать сбой модели, чем выдавать его за отсутствие задачи.
        return task_status(model, scene, runs_root) or "model_failed"
    if clip.is_file() and clip.stat().st_size > 0:
        return "ok"
    # Клипа ещё нет: если задача не запускалась или упала — это причина, а не
    # «нет медиа»; живая страница без клипа — именно media_missing.
    reason = task_status(model, scene, runs_root)
    if reason and reason != "ok":
        return reason
    return "media_missing"


def write_marker(out_dir: Path, model: str, scene: str, reason: str, detail: str = "") -> None:
    """Причина в маркере; ``detail`` — улика после двоеточия (текст диалога).

    Статус из маркера читается по первое двоеточие, поэтому улика его не ломает.
    """
    text = reason if not detail else f"{reason}: {' '.join(detail.split())[:200]}"
    marker_path(out_dir, model, scene).write_text(text + "\n", encoding="utf-8")


def clear_marker(out_dir: Path, model: str, scene: str) -> None:
    marker_path(out_dir, model, scene).unlink(missing_ok=True)


def scan(out_dir: Path, scenes: list[str], models: list[str], runs_root: Path | None) -> dict[tuple[str, str], str]:
    return {
        (model, scene): status_of(out_dir, model, scene, runs_root)
        for scene in scenes
        for model in models
    }


def main(argv: list[str] | None = None) -> int:
    """Печать статусов текущего дерева (диагностика без remotion)."""
    del argv
    video = Path(__file__).resolve().parents[1]
    meta = video / "src" / "data" / "scenes.json"
    if not meta.is_file():
        print(f"нет {meta}", file=sys.stderr)
        return 2
    scenes = [row["id"] for row in json.loads(meta.read_text(encoding="utf-8"))]
    out = video / "public" / "clips"
    for scene in scenes:
        for model in MODEL_NAMES:
            print(f"{model}/{scene}: {status_of(out, model, scene, video.parent / 'runs')}")
    return 0


MODEL_NAMES = ("deepseek-v4.1-flash-unc-1021", "deepseek-v4.1-flash-unc-1017-1021")


if __name__ == "__main__":
    sys.exit(main())
