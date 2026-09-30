"""Re-score a saved run against the current evaluators.

An evaluator fix must be provable without spending GPU hours again: the answers
are already on disk. This rebuilds the same suite from the run's own recorded
configuration, replays each saved transcript through the category evaluator and
writes fresh scores. The model is never called, so the artefacts are marked as a
re-score — a reader must be able to tell a re-scored result from a fresh run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import core
from .config import RunConfig
from .report import write_reports
from .runner import transcript_from_dict
from .suite import build_suite


def rescore_run(directory: Path | str) -> dict[str, Any]:
    directory = Path(directory)
    config_path = directory / "config.json"
    tasks_path = directory / "tasks.json"
    if not config_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"{directory}: нужны config.json и tasks.json (прогон не сохранён)")

    cfg = RunConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    suite = {task.id: task for task in build_suite(cfg)}
    saved = json.loads(tasks_path.read_text(encoding="utf-8"))

    scores: list[core.Score] = []
    changed: list[dict[str, Any]] = []
    for entry in saved:
        task_id = (entry.get("task") or {}).get("id")
        task = suite.get(str(task_id))
        if task is None:
            # The current suite no longer generates this task: keep the old score.
            scores.append(core.Score.from_dict(entry["score"]))
            changed.append({"task_id": task_id, "status": "нет задания в текущем сьюте"})
            continue
        transcript = transcript_from_dict(entry.get("transcript") or {}, task)
        module = core.REGISTRY.get(task.category)
        if module is None:
            scores.append(core.Score.from_dict(entry["score"]))
            continue
        fresh = module.evaluate(task, transcript)
        if transcript.error:
            fresh.error = transcript.error
        before = float((entry.get("score") or {}).get("earned", 0.0))
        after = fresh.earned
        if abs(before - after) > 1e-9 or (entry.get("score") or {}).get("admission") != fresh.admission:
            changed.append(
                {
                    "task_id": task_id,
                    "earned_before": before,
                    "earned_after": after,
                    "admission_before": (entry.get("score") or {}).get("admission"),
                    "admission_after": fresh.admission,
                }
            )
        scores.append(fresh)

    (directory / "scores.json").write_text(
        json.dumps([score.as_dict() for score in scores], ensure_ascii=False, indent=2, default=core.json_default),
        encoding="utf-8",
    )
    marker = {
        "rescored_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Баллы пересчитаны по сохранённым ответам текущими evaluator'ами; модель не вызывалась.",
        "tasks": len(scores),
        "changed": changed,
    }
    (directory / "rescored.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, default=core.json_default), encoding="utf-8"
    )
    write_reports([directory])
    return marker
