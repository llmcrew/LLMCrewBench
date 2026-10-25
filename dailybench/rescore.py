"""Re-score a saved run against the current evaluators.

An evaluator fix must be provable without spending GPU hours again: the answers
are already on disk. This rebuilds the same suite from the run's own recorded
configuration, replays each saved transcript through the category evaluator and
writes fresh scores. The model is never called, so the artefacts are marked as a
re-score — a reader must be able to tell a re-scored result from a fresh run.
The marker keeps both grading contracts (the recorded one and the applied one)
and the source generation; the checkpoint identity is recomputed from the new
contract instead of being carried over, so a stale identity cannot pose as the
current one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import core, generation
from .config import RunConfig
from .report import write_reports
from .runner import run_identity, transcript_from_dict
from .suite import build_suite, suite_digest


def _differs(before: Any, after: Any) -> bool:
    """State comparison: numbers with tolerance, everything else exactly.

    ``bool`` is an ``int`` subclass, so a None/True admission pair must not go
    through the numeric branch.
    """
    numeric = (int, float)
    if (
        isinstance(before, numeric)
        and isinstance(after, numeric)
        and not isinstance(before, bool)
        and not isinstance(after, bool)
    ):
        return abs(float(before) - float(after)) > 1e-9
    return before != after


def rescore_run(directory: Path | str) -> dict[str, Any]:
    directory = Path(directory)
    config_path = directory / "config.json"
    tasks_path = directory / "tasks.json"
    if not config_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"{directory}: нужны config.json и tasks.json (прогон не сохранён)")

    cfg = RunConfig.from_dict(generation.read_json(directory, "config.json"))
    suite = {task.id: task for task in build_suite(cfg)}
    saved = generation.read_json(directory, "tasks.json")
    suite_path = directory / "suite.json"
    saved_fingerprints: dict[str, dict[str, Any]] = {}
    saved_suite: dict[str, Any] = {}
    whole_drift = False
    # The contract the answers were produced under and the one applied now are
    # both recorded: a re-score is a new grading build, not the old run with a
    # marker attached.
    current_digest = suite_digest(list(suite.values()), cfg.profile)
    if suite_path.is_file():
        try:
            saved_suite = generation.read_json(directory, "suite.json")
        except (json.JSONDecodeError, UnicodeDecodeError):
            saved_suite = {}
        if not isinstance(saved_suite, dict):
            saved_suite = {}
        rows = saved_suite.get("per_task")
        if isinstance(rows, list) and rows:
            saved_fingerprints = {str(row.get("id")): row for row in rows}
        else:
            # Legacy suite.json without per-task rows: the whole-suite hash is
            # the only drift evidence, and a mismatch means the answers no
            # longer belong to these fixtures.
            saved_sha = str(saved_suite.get("sha256") or "")
            whole_drift = bool(saved_sha) and saved_sha != current_digest["sha256"]

    scores: list[core.Score] = []
    replays: list[Any] = []
    changed: list[dict[str, Any]] = []
    for entry in saved:
        task_id = (entry.get("task") or {}).get("id")
        task = suite.get(str(task_id))
        if task is None:
            # The current suite no longer generates this task: keep the old score.
            scores.append(core.Score.from_dict(entry["score"]))
            replays.append(None)
            changed.append({"task_id": task_id, "status": "нет задания в текущем сьюте"})
            continue
        if whole_drift:
            # No per-task rows to localise the drift: keep every old score
            # rather than replay answers against fixtures they never saw.
            scores.append(core.Score.from_dict(entry["score"]))
            replays.append(None)
            changed.append({"task_id": task_id, "status": "фикстура разошлась (sha сьюта)"})
            continue
        # A changed fixture is a different task: replaying an answer against it
        # would compare incomparable things, so the old score is kept and the
        # mismatch is recorded.
        expected = saved_fingerprints.get(str(task_id))
        if expected:
            current = suite_digest([task])["per_task"][0]
            # Old suite rows recorded only the prompt hash: an absent key means
            # "not recorded then", not "changed now", so it must not make every
            # legacy task look drifted.
            drift = [
                key
                for key in ("prompt_sha256", "dialogue_sha256", "workspace_sha256", "tools_sha256")
                if key in expected and expected.get(key) != current.get(key)
            ]
            if drift:
                scores.append(core.Score.from_dict(entry["score"]))
                replays.append(None)
                changed.append(
                    {"task_id": task_id, "status": f"фикстура задания разошлась: {', '.join(drift)}"}
                )
                continue
        transcript = transcript_from_dict(entry.get("transcript") or {}, task)
        module = core.REGISTRY.get(task.category)
        if module is None:
            scores.append(core.Score.from_dict(entry["score"]))
            replays.append(None)
            continue
        if task.category in cfg.unsupported:
            # Applicability is harness policy, not an evaluator result: replaying
            # the evaluator directly used to turn a skip into a 0/max row.
            reason = (
                f"категория {task.category} объявлена неприменимой для этой конфигурации "
                "(конфигурация не поддерживает её — результат не проверялся)"
            )
            fresh = core.Score(
                task_id=task.id,
                category=task.category,
                skipped=reason,
                notes=[reason],
                max_points=task.max_points,
            )
            replays.append(None)
        else:
            fresh = module.evaluate(task, transcript)
            fresh.max_points = task.max_points
            fresh.fit_points()
            replays.append(transcript)
            if transcript.error:
                fresh.error = transcript.error
                fresh.failure = transcript.extra.get("failure")
        before_score = entry.get("score") or {}
        before = {
            "earned": before_score.get("earned", 0.0),
            "possible": before_score.get("possible", 0.0),
            "status": before_score.get("status")
            or ("skipped" if before_score.get("skipped") else "error" if before_score.get("error") else "ok"),
            "admission": before_score.get("admission"),
            "skipped": before_score.get("skipped"),
            "failure": before_score.get("failure"),
        }
        after = {
            "earned": fresh.earned,
            "possible": fresh.possible,
            "status": fresh.status,
            "admission": fresh.admission,
            "skipped": fresh.skipped,
            "failure": fresh.failure,
        }
        if any(_differs(before[key], after[key]) for key in after):
            changed.append({"task_id": task_id, "before": before, "after": after})
        scores.append(fresh)

    # Пересчёт обновляет и записи ответов: наблюдения сцены (кадр, метрики)
    # живут в transcript.extra, а витрина читает их именно оттуда. Там же
    # лежит и балл: витрина берёт его из записи задачи, а не из scores.json —
    # забытый score оставлял кадр «все проверки пройдены · 0/4».
    for entry, replay, fresh in zip(saved, replays, scores):
        if replay is not None:
            transcript = entry.get("transcript")
            if isinstance(transcript, dict):
                for key, value in replay.extra.items():
                    transcript.setdefault("extra", {})[key] = value
        entry["score"] = fresh.as_dict()
    config = generation.read_json(directory, "config.json") or {}
    source_identity = str(config.get("identity") or "")
    source_generation = str((generation._manifest(directory) or {}).get("generation") or "root")
    # A checkpoint identity always describes the whole profile, independent of
    # the --only slice (see ``RunResult.save``); the re-score must not invent a
    # slice identity a resume could never match again.
    contract_digest = current_digest
    if cfg.only:
        contract_digest = suite_digest(build_suite(replace(cfg, only=())), cfg.profile)
    identity = run_identity(cfg, contract_digest["sha256"])
    rows_by_id = {str(row.get("id")): row for row in saved_fingerprints.values()}
    marker = {
        "rescored_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Баллы пересчитаны по сохранённым ответам текущими evaluator'ами; модель не вызывалась.",
        "profile": core.profile_of(cfg),
        "tasks": len(scores),
        "changed": changed,
        # A reader must be able to tell the observation from the re-score: the
        # old identity/generation stays as provenance, the new build is ours.
        "source_identity": source_identity,
        "source_generation": source_generation,
        "identity": identity,
        "grading_contract": {
            "source": {
                "contract": saved_suite.get("contract"),
                "sha256": saved_suite.get("sha256"),
                "identity": saved_suite.get("identity"),
                "evaluator": {
                    task_id: row.get("evaluator_sha256")
                    for task_id, row in rows_by_id.items()
                    if row.get("evaluator_sha256")
                },
            },
            "current": {
                "contract": current_digest["contract"],
                "sha256": current_digest["sha256"],
                "identity": identity,
                "evaluator": current_digest["evaluator_bundle"],
            },
        },
    }

    def dump(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=core.json_default)

    config["identity"] = identity
    config["run_id"] = identity
    config["rescored_from"] = source_identity
    validating_suite = dict(contract_digest)
    validating_suite["identity"] = identity
    validating_suite["source_identity"] = source_identity
    validating_suite["source_generation"] = source_generation
    for entry in saved:
        entry["identity"] = identity
    files = {
        "tasks.json": dump(saved),
        "scores.json": dump([score.as_dict() for score in scores]),
        "rescored.json": dump(marker),
        "config.json": dump(config),
        "suite.json": dump(validating_suite),
    }
    # Mirrors are part of the checkpoint: leaving the old ones would let a
    # reader see pre-rescore scores for the same task.
    for entry in saved:
        task_id = str((entry.get("task") or {}).get("id"))
        if task_id:
            files[f"transcripts/{task_id.replace('/', '_')}.json"] = dump(entry)
    generation.commit(directory, files, run_id=identity or "rescore", identity=identity)
    write_reports([directory])
    return marker
