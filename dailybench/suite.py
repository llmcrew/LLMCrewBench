"""Suite assembly: one deterministic task list per (seed, category set)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from . import core
from .config import RunConfig
from .rng import task_rng
from .tasks import load_all


def build_suite(cfg: RunConfig) -> list[core.TaskInstance]:
    """Generate every task of the run, deterministically from ``cfg.seed``.

    Categories run in registration order and each gets its own RNG stream, so
    adding a category never changes another category's fixtures.
    """
    load_all()
    selected = {item.strip().upper() for item in cfg.only if item.strip()}
    tasks: list[core.TaskInstance] = []
    for category in core.registered_categories():
        module_tasks = category.generate(task_rng(cfg.seed, category.key), cfg)
        ids = tuple(task.id for task in module_tasks)
        if category.expected_ids and ids != category.expected_ids:
            raise RuntimeError(
                f"category {category.key} generated {ids}, expected {category.expected_ids}"
            )
        for task in module_tasks:
            if selected and task.id.upper() not in selected and category.key.upper() not in selected:
                continue
            tasks.append(task)
    return tasks


def category_of(task: core.TaskInstance) -> core.CategoryModule:
    return core.REGISTRY[task.category]


def validate_suite(tasks: Iterable[core.TaskInstance]) -> list[str]:
    """Structural checks the self-check relies on. Returns warnings."""
    warnings: list[str] = []
    by_category: dict[str, list[core.TaskInstance]] = {}
    for task in tasks:
        by_category.setdefault(task.category, []).append(task)
    for key, group in by_category.items():
        # The visual block is checked by its own module: its points only add up
        # once all twenty scenarios exist, and a slice in flight must not look
        # like a broken suite.
        expected = core.CATEGORY_POINTS.get(key)
        total = sum(task.max_points for task in group)
        if expected is not None and abs(total - expected) > 1e-9 and key != "E":
            warnings.append(f"категория {key}: сумма max_points {total} != {expected}")
    return warnings


def fit_suite_sizing(cfg: RunConfig, provider: Any) -> dict[str, Any]:
    """Size the long-context documents by measurement, not by estimate.

    Two measurements, both cheap next to the run itself: the characters-per-token
    ratio of the real documents, then the real token count of the largest
    document. The largest document decides the coefficient, because it is the one
    that must fit the window together with the answer reserve — over-shooting it
    turns a 27K scenario into an admission failure, and under-shooting it turns a
    27K scenario into a smaller test than it claims.
    """
    from .measure import calibrate_chars_per_token, measure_prompt_tokens

    result: dict[str, Any] = {"calibrated": False, "chars_per_token": cfg.chars_per_token, "note": ""}

    provisional = build_suite(cfg)
    sample = next((task.prompt for task in provisional if task.id == "LC-1"), None)
    if sample is None:
        result["note"] = "в сьюте нет long-context заданий"
        return result

    ratio, how = calibrate_chars_per_token(provider, sample)
    if not ratio:
        result["note"] = f"калибровка не удалась: {how}"
        return result

    cfg.chars_per_token = round(ratio, 4)
    result.update({"calibrated": True, "measured_ratio": round(ratio, 4), "how": how})

    # Every document has its own tokenizer density, so every document gets its
    # own measured ratio: one global constant left LC-2 and LC-5 at 80 % and
    # 70 % of their targets, which is a different task than the one announced.
    rebuilt = build_suite(cfg)
    documents = [task for task in rebuilt if task.meta.get("target_tokens")]
    ratios: dict[str, float] = {}
    measured: dict[str, dict[str, float]] = {}
    for task in documents:
        target = int(task.meta["target_tokens"])
        actual, how_one = measure_prompt_tokens(provider, task.prompt)
        used = float(task.meta.get("chars_per_token") or cfg.chars_per_token)
        if not actual:
            measured[task.id] = {"target": target, "actual": None}
            continue
        # 3 % margin: the real count must stay below target + reserve.
        ratios[task.id] = round(used * (target * 0.97) / actual, 4)
        measured[task.id] = {"target": target, "actual": actual, "ratio": ratios[task.id]}

    if ratios:
        cfg.task_ratios = ratios
        result["task_ratios"] = ratios
        worst = min(
            (row["actual"] / row["target"] for row in measured.values() if row.get("actual")),
            default=None,
        )
        note = (
            f"{ratio:.2f} символов/токен ({how}); документов откалибровано {len(ratios)}/"
            f"{len(documents)}, худшее попадание до правки "
            f"{worst * 100:.0f}% от цели → коэффициент на документ {min(ratios.values()):.2f}…{max(ratios.values()):.2f}"
        )
    else:
        note = f"{ratio:.2f} символов/токен ({how}); размеры документов измерить не удалось"
    result["documents"] = measured
    result["chars_per_token"] = cfg.chars_per_token
    result["note"] = note
    return result


def suite_digest(tasks: Iterable[core.TaskInstance]) -> dict[str, Any]:
    """Public fingerprint of the generated fixtures (no answers included).

    The fingerprint covers everything the model actually sees: the prompt, the
    follow-up dialogue, the workspace contents and the tool schemas. A change
    in any of them used to leave the digest intact (the plan observed a seed
    change re-rendering C-3 databases with the same prompt and digest).
    """
    payload = [
        {
            "id": task.id,
            "category": task.category,
            "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
            "dialogue_sha256": hashlib.sha256(
                json.dumps(task.dialogue or [], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "workspace_sha256": hashlib.sha256(
                json.dumps(task.workspace or {}, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "tools_sha256": hashlib.sha256(
                json.dumps(task.tools or [], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
        }
        for task in tasks
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {
        "tasks": len(payload),
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "per_task": payload,
    }


def public_prompts(tasks: Iterable[core.TaskInstance]) -> list[dict[str, Any]]:
    """Model-visible view of the suite: safe to publish before/after a run.

    Includes the follow-up dialogue and full tool schemas, so a reader can
    reconstruct the real task the model faced; no oracle fields are exposed.
    """
    return [
        {
            "id": task.id,
            "category": task.category,
            "title": task.title,
            "max_points": task.max_points,
            "prompt": task.prompt,
            "prompt_chars": len(task.prompt),
            "estimated_tokens": task.meta.get("estimated_tokens"),
            "dialogue": task.dialogue or [],
            "workspace_files": sorted(task.workspace) if task.workspace else [],
            "tools": task.tools or [],
        }
        for task in tasks
    ]


def grading_material(tasks: Iterable[core.TaskInstance]) -> list[dict[str, Any]]:
    """Answers and oracles — kept out of the published prompt file."""
    return [
        {
            "id": task.id,
            "category": task.category,
            "oracle": task.oracle,
            "meta": task.meta,
        }
        for task in tasks
    ]
