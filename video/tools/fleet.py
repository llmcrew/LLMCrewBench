"""Единый источник флота витрины: модели и сцены.

Флот живёт в данных, а не в коде: `src/data/models.json` (запись = каталог
прогона + подпись + акцент) и `src/data/scenes.json` (id, категория, подпись).
Любой список моделей в инструментах выводится отсюда, поэтому третья модель
записывается без правок кода - достаточно дописать строку в JSON или отдать
свой файл через `--models`.
"""
from __future__ import annotations

import json
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
MODELS_JSON = VIDEO / "src" / "data" / "models.json"
SCENES_JSON = VIDEO / "src" / "data" / "scenes.json"


def _path(path: Path | str | None, default: Path) -> Path:
    return default if path is None else Path(path).expanduser()


def scenes(path: Path | str | None = None) -> list[dict]:
    """Сцены витрины; каждая обязана иметь уникальный `id`."""
    source = _path(path, SCENES_JSON)
    try:
        rows = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"не читается {source}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{source}: пустой список сцен")
    ids: list[str] = []
    for row in rows:
        scene = str(row.get("id") or "")
        if not scene:
            raise SystemExit(f"{source}: сцена без id")
        if scene in ids:
            raise SystemExit(f"{source}: сцена {scene!r} повторяется")
        ids.append(scene)
    return rows


def scene_ids(path: Path | str | None = None) -> list[str]:
    return [row["id"] for row in scenes(path)]


def models(path: Path | str | None = None) -> list[dict]:
    """Модели флота; каждая обязана иметь уникальный каталог `dir`."""
    source = _path(path, MODELS_JSON)
    try:
        rows = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"не читается {source}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{source}: пустой список моделей")
    dirs: list[str] = []
    for row in rows:
        model = str(row.get("dir") or "")
        if not model:
            raise SystemExit(f"{source}: модель без dir")
        if model in dirs:
            raise SystemExit(f"{source}: модель {model!r} повторяется")
        dirs.append(model)
    return rows


def model_names(path: Path | str | None = None) -> list[str]:
    return [row["dir"] for row in models(path)]


def check_pair(model: str, scene: str, models_source: Path | str | None = None,
               scenes_source: Path | str | None = None) -> None:
    """Проверить пару по данным флота, перечислив допустимое в ошибке."""
    names = model_names(models_source)
    if model not in names:
        raise SystemExit(f"unknown model {model!r}: {', '.join(names)}")
    ids = scene_ids(scenes_source)
    if scene not in ids:
        known = ", ".join(ids) if len(ids) <= 12 else f"{len(ids)} сцен из {_path(scenes_source, SCENES_JSON)}"
        raise SystemExit(f"unknown scene {scene!r}: {known}")
