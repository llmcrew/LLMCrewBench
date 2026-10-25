#!/usr/bin/env python3
"""Данные прогонов для видео: цифры и вердикт берутся из того же кода, что и отчёт.

Вердикт хранится строкой (`92/100`) ровно как в `report.md`: у Python и JS
разное округление половины (92.5 → 92 против 93), и без этого кадр расходился
с опубликованным отчётом.

  python3 tools/build_runs_data.py [--models FILE]
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dailybench import core, report  # noqa: E402
import fleet  # noqa: E402

RUNS = ROOT / "runs"


REPLY_FIELDS = {f.name for f in dataclasses.fields(core.ModelReply)}


def transcript(value: dict[str, Any]) -> core.Transcript:
    """Восстановить transcript из run JSON для канонического `spend()`."""
    turns = [
        core.Turn(
            messages=turn.get("messages") or [],
            reply=core.ModelReply(**{k: v for k, v in (turn.get("reply") or {}).items() if k in REPLY_FIELDS}),
            tool_results=turn.get("tool_results") or [],
        )
        for turn in value.get("turns") or []
    ]
    return core.Transcript(
        task_id=value.get("task_id") or "",
        category=value.get("category") or "",
        turns=turns,
        tool_log=value.get("tool_log") or [],
        duration_s=value.get("duration_s"),
    )


def spend_of(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Расход прогона одной семантикой с `core.Transcript.spend()`.

    Неизвестное остаётся `None`: нет tasks/transcripts или у turn нет usage -
    это не ноль. `answer` уже без reasoning (completion − reasoning).
    """
    answer = 0
    reasoning = 0
    answer_known = reasoning_known = bool(tasks)
    tools = 0
    minutes: float | None = None
    for task in tasks:
        value = task.get("transcript")
        if not value:
            answer_known = reasoning_known = False
            continue
        cost = transcript(value).spend()
        if cost["answer_tokens"] is None:
            answer_known = False
        else:
            answer += cost["answer_tokens"]
        if cost["reasoning_tokens"] is None:
            reasoning_known = False
        else:
            reasoning += cost["reasoning_tokens"]
        tools += cost["tool_calls"]
        if cost["generate_s"] is not None:
            minutes = (minutes or 0.0) + cost["generate_s"] / 60
    return {
        "answer": answer if answer_known else None,
        "reasoning": reasoning if reasoning_known else None,
        "tools": tools,
        "minutes": round(minutes, 1) if minutes is not None else None,
    }


def model_row(entry: dict) -> dict:
    directory = entry["dir"]
    label = entry.get("label") or directory
    run = report.load_run(RUNS / directory)
    totals = {
        cat: {
            "earned": sum(s["earned"] for s in run["scores"] if s["category"] == cat),
            "possible": sum(s["possible"] for s in run["scores"] if s["category"] == cat),
        }
        for cat in core.CATEGORY_POINTS
    }
    spend = spend_of(run.get("tasks") or [])
    summary = report.summarise(run["scores"], run.get("suite"))
    config = run.get("config") or {}
    return {
        "dir": directory,
        "label": label,
        "title": label,
        "accent": entry.get("accent") or "#888888",
        "quant": config.get("quant") or "-",
        "context": int(config.get("context_window") or 0),
        # `score` - число для сортировки и пропорций, `verdict` - строка отчёта.
        # Берём headline из `summarise`: там же, где вердикт, поэтому провал
        # admission у E не расходится с подписью на экране.
        "score": round(summary["headline_earned"], 3),
        "possible": summary["headline_possible"],
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
        "spend": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in spend.items()},
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Данные прогонов для видео")
    parser.add_argument("--models", type=Path, help="свой models.json вместо src/data/models.json")
    args = parser.parse_args(argv)
    rows = [model_row(entry) for entry in fleet.models(args.models)]
    target = ROOT / "video/src/data/runs.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['label']}: {r['verdict']} (score {r['score']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
