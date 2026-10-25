"""Suite assembly: one deterministic task list per (seed, category set)."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterable

from . import core
from .config import RunConfig, toolchain_identity
from .rng import task_rng
from .tasks import load_all


def build_suite(cfg: RunConfig) -> list[core.TaskInstance]:
    """Generate every task of the run, deterministically from ``cfg.seed``.

    Categories run in registration order and each gets its own RNG stream, so
    adding a category never changes another category's fixtures. The profile
    decides which scenarios are expected and, for ``stretch``, how the fixed
    category totals are distributed over them.
    """
    load_all()
    profile = core.profile_of(cfg)
    selected = {item.strip().upper() for item in cfg.only if item.strip()}
    generated: list[core.TaskInstance] = []
    for category in core.registered_categories():
        module_tasks = category.generate(task_rng(cfg.seed, category.key), cfg)
        ids = tuple(task.id for task in module_tasks)
        expected = category.expected_ids_for(profile)
        # An empty set is a declaration too: a profile that drops a category
        # must generate nothing for it.
        if (expected or profile in category.expected_ids_by_profile) and ids != expected:
            raise RuntimeError(
                f"category {category.key} generated {ids}, expected {expected} in profile {profile}"
            )
        generated.extend(module_tasks)
    # Weights belong to the whole profile: a filtered slice keeps the weight the
    # task has in a complete run of that suite version.
    apply_profile_weights(generated, profile)
    tasks: list[core.TaskInstance] = []
    for task in generated:
        if selected and task.id.upper() not in selected and task.category.upper() not in selected:
            continue
        tasks.append(task)
    return tasks


def apply_profile_weights(tasks: list[core.TaskInstance], profile: str) -> None:
    """Distribute a profile's category total evenly over its tasks.

    A profile total (the stretch A–J split, the visual showcase of every
    profile) is spread over the category's tasks, and the last task takes the
    rounding tail so the sum stays exactly the declared one. Categories without
    a total and the A–J categories of the baseline keep the per-task weights
    their modules declare.
    """
    totals = {**core.PROFILE_POINTS.get(profile, {}), **core.showcase_points(profile)}
    if not totals:
        return
    grouped: dict[str, list[core.TaskInstance]] = {}
    for task in tasks:
        grouped.setdefault(task.category, []).append(task)
    for key, total in totals.items():
        group = grouped.get(key) or []
        if not group:
            continue
        share = round(total / len(group), 4)
        for index, task in enumerate(group):
            task.max_points = share if index < len(group) - 1 else round(total - share * (len(group) - 1), 4)



def category_of(task: core.TaskInstance) -> core.CategoryModule:
    return core.REGISTRY[task.category]


def validate_suite(tasks: Iterable[core.TaskInstance], profile: str = core.DEFAULT_PROFILE) -> list[str]:
    """Structural checks the self-check relies on. Returns warnings."""
    warnings: list[str] = []
    by_category: dict[str, list[core.TaskInstance]] = {}
    for task in tasks:
        by_category.setdefault(task.category, []).append(task)
    for key, group in by_category.items():
        # The visual block is checked by its own module: its points only add up
        # once all twenty scenarios exist, and a slice in flight must not look
        # like a broken suite.
        expected = core.points_for(key, profile)
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
    # Any long-context document proves the density; a profile may not contain LC-1.
    sample = next((task.prompt for task in provisional if task.category == "E"), None)
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


_PACKAGE_ROOT = Path(__file__).resolve().parent
_PACKAGE_NAME = _PACKAGE_ROOT.name

#: category → (signature of the bundle files, digest). The signature is
#: ``(path, mtime_ns, size)`` per file, so a rebuild of the same bytes still
#: verifies while an edit invalidates the entry. Hashing the K bundle for every
#: task of a suite digest was otherwise a per-task file walk.
_EVALUATOR_CACHE: dict[str, tuple[tuple[Any, ...], str]] = {}


def _dotted(path: Path) -> str:
    parts = path.relative_to(_PACKAGE_ROOT).with_suffix("").parts
    return ".".join((_PACKAGE_NAME, *parts))


def _category_files(category: str) -> list[Path]:
    """Every source file of the category's executable evaluator.

    The registered ``evaluate`` names the dispatcher; its siblings with the
    same prefix carry the real checks (``coding_python.py``, ``coding_go.py``,
    ``visual_*.py``). Hashing only the dispatcher let a replaced hidden test
    keep the identity intact.
    """
    module = core.REGISTRY.get(category)
    name = getattr(getattr(module, "evaluate", None), "__module__", "")
    registered = sys.modules.get(name)
    source = inspect.getsourcefile(registered) if registered is not None else None
    if not source:
        return []
    path = Path(source).resolve()
    if not path.is_file():
        return []
    # A category can have its dispatcher at the package root (L: ``art.py``)
    # while the scenario files live in ``tasks/``; both sides belong to the
    # bundle. The glob also keeps a newly split evaluator file covered without
    # registering it anywhere.
    directories = {path.parent}
    if (_PACKAGE_ROOT / "tasks").is_dir():
        directories.add(_PACKAGE_ROOT / "tasks")
    files = {path}
    for directory in directories:
        files.update(item for item in directory.glob(f"{path.stem}*.py") if item.is_file())
    return sorted(files)


def _bundle_files(category: str) -> list[Path]:
    """Category files plus the shared helpers they import, transitively.

    Only modules inside this package count: the point is a closed fingerprint
    of the code that can decide a score, not of the standard library.
    """
    pending: list[types.ModuleType] = []
    for path in _category_files(category):
        try:
            pending.append(importlib.import_module(_dotted(path)))
        except Exception:  # pragma: no cover - a broken tree is still hashed by path
            continue
    seen: set[Path] = set()
    while pending:
        module = pending.pop()
        source = inspect.getsourcefile(module)
        if not source:
            continue
        path = Path(source).resolve()
        if path in seen or not path.is_file() or _PACKAGE_ROOT not in path.parents:
            continue
        seen.add(path)
        for value in vars(module).values():
            if isinstance(value, types.ModuleType) and getattr(value, "__file__", None):
                if Path(value.__file__).resolve().parent == _PACKAGE_ROOT or (
                    _PACKAGE_ROOT in Path(value.__file__).resolve().parents
                ):
                    pending.append(value)
    return sorted(seen)


def evaluator_sha(category: str) -> str:
    """Fingerprint of the whole executable evaluator bundle of a category.

    Every module that can decide a score is hashed byte for byte: the
    dispatcher, the per-language/per-scenario evaluator files, and the shared
    helpers they import. The closed oracle (hidden tests) is hashed per task by
    :func:`suite_digest`, and the toolchain by :func:`toolchain_identity`.
    """
    files = _bundle_files(category)
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
    cached = _EVALUATOR_CACHE.get(category)
    if cached and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256(category.encode() + b"\0")
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        digest.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode())
        digest.update(b"\0" + hashlib.sha256(raw).digest())
    value = digest.hexdigest()
    _EVALUATOR_CACHE[category] = (signature, value)
    return value


def oracle_sha(task: core.TaskInstance) -> str:
    """Fingerprint of the closed oracle, hidden tests included.

    ``task.oracle`` is what the evaluator compares against; changing a hidden
    test without touching the evaluator module is a different grading contract.
    """
    try:
        text = json.dumps(task.oracle, sort_keys=True, ensure_ascii=False, default=core.json_default)
    except TypeError:  # a fixture kept a value even the JSON fallback rejects
        text = repr(task.oracle)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def suite_digest(tasks: Iterable[core.TaskInstance], profile: str = core.DEFAULT_PROFILE) -> dict[str, Any]:
    """Public fingerprint of the fixtures and the grading contract.

    The fingerprint covers everything the model actually sees: the prompt, the
    scenario's own system instruction, the follow-up dialogue, the workspace
    contents and the tool schemas, plus the meta fields that steer the budget.
    A change in any of them used to leave the digest intact (the plan observed a
    seed change re-rendering C-3 databases with the same prompt and digest).

    It also covers the denominator, the executable evaluator bundle, the closed
    oracle (hidden tests included) and the applied toolchain: the same fixtures
    scored out of a different ``max_points``, by other code, on another runtime
    are a different measurement, and a resume used to mix the two silently.
    """
    tasks = list(tasks)
    payload = [
        {
            "id": task.id,
            "category": task.category,
            "block": task.block,
            "family": task.family,
            "languages": task.languages,
            "cultures": task.cultures,
            "fresh_after": task.fresh_after,
            "private": task.private,
            "holdout_seeds": task.holdout_seeds,
            "max_points": task.max_points,
            "evaluator_sha256": evaluator_sha(task.category),
            "oracle_sha256": oracle_sha(task),
            "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
            # The scenario's own system instruction and its meta (budget
            # headroom, step count, timing) change the task the model faces and
            # its admission, so they belong to the fingerprint.
            "system_sha256": hashlib.sha256((task.system or "").encode()).hexdigest(),
            "meta_sha256": hashlib.sha256(
                json.dumps(task.meta or {}, ensure_ascii=False, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "dialogue_sha256": hashlib.sha256(
                json.dumps(task.dialogue or [], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "workspace_sha256": hashlib.sha256(
                json.dumps(task.workspace or {}, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "assets_sha256": hashlib.sha256(
                json.dumps(
                    [asset.as_dict() for asset in (task.assets or [])],
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "tools_sha256": hashlib.sha256(
                json.dumps(task.tools or [], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
        }
        for task in tasks
    ]
    toolchain = toolchain_identity()
    blob = json.dumps(
        {"contract": core.EVALUATOR_CONTRACT, "toolchain": toolchain, "tasks": payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "profile": profile,
        "contract": core.EVALUATOR_CONTRACT,
        "toolchain": toolchain,
        "evaluator_bundle": {task.category: evaluator_sha(task.category) for task in tasks},
        "tasks": len(payload),
        "possible_total": round(sum(task.max_points for task in tasks), 4),
        "profile_total": core.profile_total(profile),
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "per_task": payload,
    }


def public_prompts(
    tasks: Iterable[core.TaskInstance], include_private: bool = False
) -> list[dict[str, Any]]:
    """Model-visible view of the suite: safe to publish before/after a run.

    Includes the follow-up dialogue and full tool schemas, so a reader can
    reconstruct the real task the model faced; no oracle fields are exposed.
    A private task contributes only its id, category, denominator and layer
    hashes by default — its prompt never leaves the machine.
    """
    from .decontam import is_private, prompt_hashes

    entries: list[dict[str, Any]] = []
    for task in tasks:
        if is_private(task):
            if not include_private:
                entries.append(
                    {
                        "id": task.id,
                        "category": task.category,
                        "block": task.block,
                        "family": task.family,
                        "languages": list(task.languages),
                        "cultures": list(task.cultures),
                        "fresh_after": task.fresh_after,
                        "holdout_seeds": list(task.holdout_seeds),
                        "max_points": task.max_points,
                        "private": True,
                        **prompt_hashes(task),
                    }
                )
                continue
        entries.append(
            {
                "id": task.id,
                "category": task.category,
                "block": task.block,
                "family": task.family,
                "languages": list(task.languages),
                "cultures": list(task.cultures),
                "fresh_after": task.fresh_after,
                "holdout_seeds": list(task.holdout_seeds),
                "title": task.title,
                "max_points": task.max_points,
                "prompt": task.prompt,
                "prompt_chars": len(task.prompt),
                "estimated_tokens": task.meta.get("estimated_tokens"),
                "dialogue": task.dialogue or [],
                "workspace_files": sorted(task.workspace) if task.workspace else [],
                "tools": task.tools or [],
                **({"private": True} if is_private(task) else {}),
            }
        )
    return entries


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
