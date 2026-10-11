#!/usr/bin/env python3
"""Форматированные имена моделей в данных видео.

Прогоны хранят технические метки (`deepseek-v4.1-flash-uncensored`, `qwen3.8`);
в кадре они читаются плохо. Скрипт приводит подписи в `src/data/*.json` к
человеческому виду и идемпотентен: повторный запуск ничего не портит.

  python3 tools/format_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "video" / "src" / "data"

NAMES = {
    "deepseek-v4.1-flash-uncensored": "DeepSeek 4.1 Flash",
    "deepseek-v4.1-flash-unc": "DeepSeek 4.1 Flash",
    "glm-5.3": "GLM 5.3",
    "Qwen3.6-35B-A3B": "Qwen 3.6 35B A3B",
    "qwen3.8": "Qwen 3.8",
}


def patch(path: Path, key: str) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    changed = 0

    def walk(node):
        nonlocal changed
        if isinstance(node, dict):
            value = node.get(key)
            if isinstance(value, str) and value in NAMES and node[key] != NAMES[value]:
                node[key] = NAMES[value]
                changed += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(raw)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> int:
    total = 0
    for name in ("models.json", "showcase.json", "runs.json"):
        path = DATA / name
        if path.is_file():
            n = patch(path, "label")
            total += n
            print(f"{name}: заменено {n}")
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
