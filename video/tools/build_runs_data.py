#!/usr/bin/env python3
"""Данные прогонов для видео: цифры и вердикт берутся из того же кода, что и отчёт.

Вердикт хранится строкой (`92/100`) ровно как в `report.md`: у Python и JS
разное округление половины (92.5 → 92 против 93), и без этого кадр расходился
с опубликованным отчётом.

  python3 tools/build_runs_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dailybench import core, report  # noqa: E402

RUNS = ROOT / "runs"
DIRS = ["deepseek-v4.1-flash-unc", "glm-5.3", "qwen36", "qwen38"]
LABELS = {
    "deepseek-v4.1-flash-unc": "DeepSeek 4.1 Flash",
    "glm-5.3": "GLM 5.3",
    "qwen36": "Qwen 3.6 35B A3B",
    "qwen38": "Qwen 3.8",
}
ACCENTS = {
    "deepseek-v4.1-flash-unc": "#4cc2ff",
    "glm-5.3": "#f4a3ff",
    "qwen36": "#3ddc97",
    "qwen38": "#b48cff",
}


def model_row(directory: str) -> dict:
    run = report.load_run(RUNS / directory)
    labels = run.get("config", {}).get("label") or LABELS[directory]
    totals = {
        cat: {
            "earned": sum(s["earned"] for s in run["scores"] if s["category"] == cat),
            "possible": sum(s["possible"] for s in run["scores"] if s["category"] == cat),
        }
        for cat in core.CATEGORY_POINTS
    }
    spend = {"answer": 0, "reasoning": 0, "tools": 0, "minutes": 0.0}
    for task in run.get("tasks", []):
        tr = task["transcript"]
        for turn in tr.get("turns") or []:
            reply = turn.get("reply") or {}
            spend["answer"] += reply.get("completion_tokens") or 0
            spend["reasoning"] += reply.get("reasoning_tokens") or 0
        spend["tools"] += len(tr.get("tool_log") or [])
        spend["minutes"] += (tr.get("duration_s") or 0) / 60
    summary = report.summarise(run["scores"], run.get("suite"))
    config = run.get("config") or {}
    return {
        "dir": directory,
        "label": LABELS[directory],
        "title": labels,
        "accent": ACCENTS[directory],
        "quant": config.get("quant") or "—",
        "context": int(config.get("context_window") or 0),
        # `score` — число для сортировки и пропорций, `verdict` — строка отчёта.
        "score": round(sum(t["earned"] for cat, t in totals.items() if cat in core.SCORED_CATEGORIES), 3),
        "verdict": summary["verdict"],
        "errors": sum(1 for s in run["scores"] if s.get("error")),
        "tasks": len(run["scores"]),
        "K": {
            "e": sum(s["earned"] for s in run["scores"] if s["category"] == "K"),
            "p": sum(s["possible"] for s in run["scores"] if s["category"] == "K"),
        },
        "L": {
            "e": sum(s["earned"] for s in run["scores"] if s["category"] == "L"),
            "p": sum(s["possible"] for s in run["scores"] if s["category"] == "L"),
        },
        "totals": totals,
        "spend": {k: round(v, 1) for k, v in spend.items()},
    }


def main() -> int:
    rows = [model_row(d) for d in DIRS]
    target = ROOT / "video/src/data/runs.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['label']}: {r['verdict']} (score {r['score']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
