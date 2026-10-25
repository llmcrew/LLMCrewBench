"""Reporting: per-run markdown, JSON totals and a cross-model comparison table.

Quality is reported per category and as a total; a configuration whose
long-context tasks failed the admission check is shown out of the short total
(90) with the reason, never folded into a 100-point number it did not earn. The
visual block (category K) is a separate screen: measured, rendered, and kept out
of the verdict.
"""

from __future__ import annotations

import base64
import html
import json
import math
import re
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from . import core, gallery, generation, schema, stats, telemetry, viz

SHORT_ORDER = core.ADMISSION_CATEGORIES


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _file_name(task_id: str) -> str:
    """One path component for an id that is not ``K-1``-shaped."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).lstrip(".") or "task"


def load_run(directory: Path | str) -> dict[str, Any]:
    directory = Path(directory)
    # A committed generation is one consistent checkpoint. A broken manifest is
    # quarantined instead of falling back to the root copies, and an edited
    # root export is reported while the sealed bytes keep being served.
    base, integrity, problems = generation.resolve(directory)
    scores_path = base / "scores.json"
    if not scores_path.is_file():
        raise FileNotFoundError(f"{scores_path} not found — run the suite first")
    data: dict[str, Any] = {
        "dir": str(directory),
        "integrity": integrity,
        "integrity_problems": problems,
        "scores": json.loads(scores_path.read_text(encoding="utf-8")),
    }
    for name in (
        "config",
        "provider",
        "hardware",
        "suite",
        "speed",
        "tasks",
        "sizing",
        "rescored",
        "environment",
    ):
        path = base / f"{name}.json"
        if path.is_file():
            data[name] = json.loads(path.read_text(encoding="utf-8"))
    # The journal is committed as an index under ``telemetry/``; the bare
    # ``telemetry.json`` is the pre-index name and stays readable.
    for path in (base / "telemetry" / "index.json", base / "telemetry.json"):
        if path.is_file():
            data["telemetry"] = json.loads(path.read_text(encoding="utf-8"))
            break
    return data


def _stop_reason(run_entry: dict[str, Any] | None) -> str:
    """Why generation stopped, plus the reasoning-heavy case called out."""
    if not run_entry:
        return "—"
    turns = (run_entry.get("transcript") or {}).get("turns") or []
    if not turns:
        return "—"
    reply = turns[-1].get("reply") or {}
    reason = reply.get("finish_reason") or "—"
    generated = reply.get("completion_tokens")
    reasoning = reply.get("reasoning_tokens")
    if reason == "length" and not (reply.get("text") or "").strip():
        return "length — бюджет исчерпан до видимого ответа"
    if reasoning:
        return f"{reason} ({generated} ток., из них reasoning {reasoning})"
    return reason


def _interval_of(row: dict[str, Any]) -> tuple[float, float] | None:
    """95 % interval of one row's ratio, in percent; ``None`` without samples."""
    if not row["samples"]:
        return None
    low, high = stats.bootstrap_ratio(row["samples"])
    return (round(100.0 * low, 2), round(100.0 * high, 2))


def _canonical_counts(counts: dict[str, int]) -> dict[str, int]:
    """Merge failure counts under their canonical protocol kind."""
    merged: dict[str, int] = {}
    for kind, count in counts.items():
        canonical = core.canonical_failure(kind) or kind
        merged[canonical] = merged.get(canonical, 0) + count
    return merged


def summarise(
    scores: Iterable[dict[str, Any]],
    suite_tasks: int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Totals per category, with the visual block kept out of the verdict.

    ``suite_tasks`` may be the suite digest, so the "partial run" note counts
    only the tasks that can move the score — a skipped browser must not make a
    complete run look partial.
    """
    scored = set(core.SCORED_CATEGORIES)
    showcase = set(core.VISUAL_CATEGORIES)
    profile = core.DEFAULT_PROFILE
    if isinstance(suite_tasks, dict):
        profile = str(suite_tasks.get("profile") or core.DEFAULT_PROFILE)
    # A profile that carries a subset of the categories is read on its own: its
    # denominator is smaller, so its percentage says nothing about a baseline one.
    rankable = core.comparable_with_baseline(profile)

    def blank() -> dict[str, Any]:
        return {
            "earned": 0.0,
            "possible": 0.0,
            "admitted": True,
            "admission": "passed",
            "tasks": 0,
            "errors": 0,
            "skipped": 0,
            "skip_reasons": [],
            "n": 0,
            "ci95": None,
            "provisional": False,
            "samples": [],
            "units": {},
            "failures": {},
            "outcomes": {},
        }

    per_category: dict[str, dict[str, Any]] = {key: blank() for key in core.CATEGORY_POINTS}
    per_showcase: dict[str, dict[str, Any]] = {key: blank() for key in core.VISUAL_CATEGORIES}
    for score in scores:
        category = score["category"]
        if category in showcase:
            row = per_showcase.setdefault(category, blank())
        else:
            row = per_category.setdefault(category, blank())
        row["tasks"] += 1
        # The outcome vector covers skipped tasks too: ``unsupported`` is a
        # protocol state even though it leaves the denominator untouched.
        outcome = core.outcome_of(score)
        row["outcomes"][outcome] = row["outcomes"].get(outcome, 0) + 1
        if score.get("skipped"):
            # Not verified is not a zero: it leaves both the numerator and the
            # denominator untouched, and the reason stays visible.
            row["skipped"] += 1
            row["skip_reasons"].append(score["skipped"])
            continue
        # A broken answer is still an answer: an evaluator crash or a
        # transport error must not shrink the denominator (R1) — an errored
        # task keeps its max points in ``possible`` unless the run itself is
        # explicitly marked skipped above.
        possible = score.get("possible", 0.0)
        if score.get("error") and not possible:
            # An errored task with no checks (evaluator crash, dead agent)
            # still counts at its max: the model did not earn the points.
            possible = float(score.get("max_points") or 0.0)
        admission = score.get("admission")
        row["possible"] += possible
        if possible > 0:
            # One resampling unit per observed task, with its own weight: the
            # interval of a category resamples tasks, not shares. A task that
            # failed admission earned nothing in this metric.
            row["samples"].append((float(score.get("earned", 0.0)
                                        if admission is not False else 0.0), float(possible)))
            # The same task keeps its id so a later comparison can pair it with
            # the identical instance instead of averaging two unrelated draws.
            row["units"][str(score.get("task_id") or f"{category}-{row['tasks']}")] = (
                float(score.get("earned", 0.0) if admission is not False else 0.0),
                float(possible),
            )
            row["n"] += 1
        if score.get("error"):
            row["errors"] += 1
        kind = score.get("failure")
        if kind:
            row["failures"][kind] = row["failures"].get(kind, 0) + 1
        if admission is False:
            row["admission"] = "failed"
            row["admitted"] = False
        elif category == "E" and admission is None and row["admission"] == "passed":
            # Only the long-context block has an admission check; every other
            # evaluator leaves the field unset, so ``None`` there is no signal.
            # For E, unknown is not a pass: the measurement may still turn out
            # bad, so the state stays unverified until a real verdict arrives.
            row["admission"] = "unverified"
            row["admitted"] = False
        if admission is not False:
            row["earned"] += score.get("earned", 0.0)

    admission_state = per_category.get("E", {}).get("admission", "passed")
    long_context_admitted = admission_state == "passed"
    admitted_total = round(sum(per_category[key]["earned"] for key in core.SCORED_CATEGORIES), 3)
    short_total = round(sum(per_category[key]["earned"] for key in SHORT_ORDER), 3)
    possible_total = round(sum(row["possible"] for row in per_category.values()), 3)
    errors = sum(row["errors"] for row in per_category.values())
    ran = sum(row["tasks"] for row in per_category.values())
    ran_showcase = sum(row["tasks"] for row in per_showcase.values())
    skipped_scored = sum(row["skipped"] for row in per_category.values())
    skipped_showcase = sum(row["skipped"] for row in per_showcase.values())
    skipped_total = skipped_scored + skipped_showcase
    failure_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for row in list(per_category.values()) + list(per_showcase.values()):
        for kind, count in row["failures"].items():
            failure_counts[kind] = failure_counts.get(kind, 0) + count
        for outcome, count in row["outcomes"].items():
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + count
    # Every stored failure is shown with the same canonical name it is counted
    # under: a legacy bundle must not split ``infra_failure`` in two rows.
    failure_counts = _canonical_counts(failure_counts)
    # Incomplete means the numbers say nothing about the model: such a run is
    # shown, but never ranked against complete ones.
    infra_failures = {
        kind: count for kind, count in failure_counts.items() if core.is_infra_failure(kind)
    }
    integrity_failures = {
        kind: count for kind, count in failure_counts.items()
        if core.canonical_failure(kind) == "integrity_violation"
    }
    # Capability gaps are outside the denominator by design and form their own
    # cohort; they are named in the verdict, not spread over the percentage.
    unsupported = outcome_counts.get("unsupported", 0)
    # Every dictionary kind is present, so a consumer never has to guess whether
    # a missing key means zero or "not counted".
    outcome_counts = {
        **{kind: outcome_counts.get(kind, 0) for kind in core.FAILURE_KINDS},
        **{kind: count for kind, count in outcome_counts.items() if kind not in core.FAILURE_KINDS},
    }
    eligible = not infra_failures

    showcase_earned = round(sum(row["earned"] for key, row in per_showcase.items() if key in core.SHOWCASE_CATEGORIES), 3)
    showcase_possible = round(sum(row["possible"] for key, row in per_showcase.items() if key in core.SHOWCASE_CATEGORIES), 3)

    # Every published metric carries its interval and its sample size. A row
    # without observed tasks has no interval: a missing measurement must not
    # read as a zero-width one.
    for row in list(per_category.values()) + list(per_showcase.values()):
        row["ci95"] = _interval_of(row)
        row["provisional"] = bool(row["n"]) and stats.is_small(row["n"])

    expected_scored: int | None = None
    if isinstance(suite_tasks, dict):
        expected_scored = sum(
            1 for entry in suite_tasks.get("per_task") or [] if entry.get("category") in scored
        )
    elif isinstance(suite_tasks, int):
        expected_scored = suite_tasks

    if admission_state == "failed":
        headline_earned = short_total
        headline_possible = round(sum(per_category[key]["possible"] for key in SHORT_ORDER), 3)
    else:
        # Unknown admission keeps the full numerator and denominator: an
        # unmeasured input size is not a failed capability (R02).
        headline_earned, headline_possible = admitted_total, possible_total

    headline_keys = SHORT_ORDER if admission_state == "failed" else core.SCORED_CATEGORIES
    headline_samples = [sample for key in headline_keys for sample in per_category[key]["samples"]]
    headline_n = len(headline_samples)
    if headline_samples:
        low, high = stats.bootstrap_ratio(headline_samples)
        headline_ci95: tuple[float, float] | None = (round(100.0 * low, 2), round(100.0 * high, 2))
    else:
        headline_ci95 = None

    if expected_scored:
        full_suite = ran >= expected_scored
    else:
        full_suite = possible_total >= core.scored_total(profile) - 1e-9
    verdict = f"{headline_earned:.0f}/{headline_possible:.0f}"
    if admission_state == "failed":
        verdict += " (из доступных баллов: long context не прошёл допуск)"
    elif admission_state == "unverified":
        verdict += " (допуск не подтверждён: runtime не сообщил размер входа)"
    if not full_suite:
        verdict += (
            f" (частичный прогон: {possible_total:.0f} из {core.scored_total(profile):.0f}"
            " баллов сьюта)"
        )
    if skipped_total:
        skipped_categories = [
            key for key, row in per_category.items() if row["skipped"] and row["skipped"] == row["tasks"]
        ]
        verdict += (
            f" (не проверялось: {', '.join(skipped_categories) or skipped_total} —"
            " результат не входит ни в числитель, ни в знаменатель)"
        )
    if not rankable:
        verdict += (
            f" (не ранжируется с baseline: профиль {profile} — свой набор категорий,"
            f" знаменатель {core.scored_total(profile):.0f})"
        )
    if unsupported:
        verdict += (
            f" (unsupported: {unsupported} — отдельная capability, вне знаменателя)"
        )
    if integrity_failures:
        # An attestation refusal outranks the rest: the verdict itself is not
        # trustworthy, so only the integrity journal is named.
        names = ", ".join(
            f"{core.failure_label(kind)} x{count}"
            for kind, count in sorted(integrity_failures.items())
        )
        verdict += f" (не ранжируется: удостоверение отозвано — {names})"
    elif infra_failures:
        names = ", ".join(
            f"{core.failure_label(kind)} x{count}" for kind, count in sorted(infra_failures.items())
        )
        verdict += f" (не ранжируется: инфраструктура — {names})"
    elif failure_counts:
        # A model outcome (exhausted budget, wrong answer) is a real verdict and
        # stays in the denominator; it is named, not excused.
        names = ", ".join(
            f"{core.failure_label(kind)} x{count}" for kind, count in sorted(failure_counts.items())
        )
        verdict += f" (исходы модели: {names})"

    return {
        "per_category": per_category,
        "showcase": per_showcase,
        "showcase_earned": showcase_earned,
        "showcase_possible": showcase_possible,
        "admitted_total": admitted_total,
        "short_total": short_total,
        "possible_total": possible_total,
        "full_suite": full_suite,
        "headline_earned": headline_earned,
        "headline_possible": headline_possible,
        "headline_ci95": headline_ci95,
        "headline_n": headline_n,
        "headline_provisional": bool(headline_n) and stats.is_small(headline_n),
        "long_context_admitted": long_context_admitted,
        "admission_state": admission_state,
        "errors": errors,
        "ran": ran,
        "ran_showcase": ran_showcase,
        "skipped": skipped_total,
        "skipped_scored": skipped_scored,
        "incomplete": skipped_scored > 0 or not full_suite,
        "failures": failure_counts,
        "outcomes": outcome_counts,
        "unsupported": unsupported,
        "integrity_failures": integrity_failures,
        "infra_failures": infra_failures,
        "eligible": eligible,
        "profile": profile,
        "rankable_with_baseline": rankable,
        "verdict": verdict,
    }


WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    "published": dict(core.CATEGORY_POINTS),
    "equal": {key: 1.0 for key in core.CATEGORY_POINTS},
    # A second, deliberately different policy: if the ordering only survives
    # one weight mixture it is a property of the mixture, not of the model.
    "coding_heavy": {key: 2.0 if key == "C" else 1.5 if key == "B" else 1.0
                     for key in core.CATEGORY_POINTS},
}
"""Weight mixtures the verdict is recomputed under."""


def _stratum(category: str) -> str:
    """Resampling stratum of a category: its semantic family.

    Ten renamed entities of one family are not ten independent draws, so the
    interval is a cluster-level one. The category is only the fallback for an
    off-schema category.
    """
    return schema.family_for(category) or "прочее"


def _paired_units(
    left: dict[str, Any] | None, right: dict[str, Any] | None, key: str
) -> list[stats.PairedUnit]:
    """Shared task instances of two category rows, tagged with the family."""
    left_units = (left or {}).get("units") or {}
    right_units = (right or {}).get("units") or {}
    stratum = _stratum(key)
    units: list[stats.PairedUnit] = []
    for task_id, (left_earned, left_possible) in left_units.items():
        pair = right_units.get(task_id)
        if pair is None or left_possible <= 0 or pair[1] <= 0:
            continue
        units.append((stratum, left_earned, left_possible, pair[0], pair[1]))
    return units


def _overlaps(
    left: tuple[float, float] | None, right: tuple[float, float] | None
) -> bool | None:
    """Whether the two marginal intervals touch; ``None`` when one is missing."""
    if not left or not right:
        return None
    return not stats.disjoint(left, right)


def _weighted_pct(
    summary: dict[str, Any], weights: dict[str, float]
) -> tuple[float | None, frozenset[str]]:
    """Score under one weight mixture, and the categories nobody measured.

    The denominator is the *whole* profile weight, so an unmeasured category is
    not a free pass: its weight stays in the denominator and is reported as
    unknown instead of being re-normalised away. Two runs measuring different
    sets are then still on one scale.
    """
    total = sum(weights.values())
    earned = 0.0
    unknown: set[str] = set()
    for key, weight in weights.items():
        row = summary["per_category"].get(key) or {}
        possible = row.get("possible") or 0.0
        if possible <= 0:
            unknown.add(key)
            continue
        earned += weight * row.get("earned", 0.0) / possible
    return (round(100.0 * earned / total, 2) if total else None), frozenset(unknown)


def _weight_sensitivity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Whether the ordering survives a different policy, not just the published one."""
    schemes: dict[str, Any] = {}
    for name, weights in WEIGHT_SCHEMES.items():
        scheme_total = sum(weights.values())
        left_pct, left_unknown = _weighted_pct(left, weights)
        right_pct, right_unknown = _weighted_pct(right, weights)
        unknown = left_unknown | right_unknown
        delta = None if left_pct is None or right_pct is None else round(left_pct - right_pct, 2)
        schemes[name] = {
            "weights": dict(weights),
            "left": left_pct,
            "right": right_pct,
            "delta": delta,
            # Weight that at least one side did not measure: the delta above is
            # capped by that much unknown, so it is printed, not hidden.
            "unknown": round(100.0 * sum(weights[key] for key in unknown) / scheme_total, 1)
            if unknown and scheme_total else 0.0,
            "winner": None if delta is None else ("left" if delta > 0 else "right" if delta < 0 else "tie"),
        }
    published = schemes["published"]["winner"]
    return {
        "schemes": schemes,
        "winner": published,
        "stable": published is not None and all(
            entry["winner"] == published for entry in schemes.values()
        ),
    }


def _category_vector(summary: dict[str, Any]) -> dict[str, float | None]:
    """The per-category percentages as a standalone published vector."""
    return {key: _pct_of(summary["per_category"].get(key)) for key in sorted(core.CATEGORY_POINTS)}


def _compare_pair(
    left: dict[str, Any] | None, right: dict[str, Any] | None, key: str
) -> dict[str, Any]:
    """One category's paired difference.

    Significance comes from the interval of the paired difference, restricted
    to the shared tasks of that category. ``overlap`` only records whether the
    two marginal intervals happen to touch.
    """
    left_ci = (left or {}).get("ci95")
    right_ci = (right or {}).get("ci95")
    left_pct = _pct_of(left)
    right_pct = _pct_of(right)
    test = stats.paired_ratio_difference(_paired_units(left, right, key))
    delta_ci = test["ci95"]
    delta = test["delta"]
    return {
        "left_ci95": left_ci,
        "right_ci95": right_ci,
        "left_pct": left_pct,
        "right_pct": right_pct,
        # The published difference is the *paired* one: it lives on the shared
        # ids, exactly like the interval printed beside it. The two marginal
        # percentages are context columns, not the estimate.
        "delta": None if delta is None else round(100.0 * delta, 2),
        "delta_ci95": None if delta_ci is None
        else (round(100.0 * delta_ci[0], 2), round(100.0 * delta_ci[1], 2)),
        "pairs": test["n_pairs"],
        "significant": test["significant"],
        "overlap": _overlaps(left_ci, right_ci),
        "provisional": bool((left or {}).get("provisional") or (right or {}).get("provisional")),
    }


def _pct_of(row: dict[str, Any] | None) -> float | None:
    if not row or not row.get("possible"):
        return None
    return round(100.0 * row.get("earned", 0.0) / row["possible"], 2)


def contract_key(run: dict[str, Any]) -> str:
    """Immutable contract identity of a run: the suite digest sha256.

    Runs built from different suite bytes, profiles or evaluator contracts have
    different denominators; their percentages are not ranks of one another.
    """
    return str((run.get("suite") or {}).get("sha256") or "")


def rankable(summary: dict[str, Any]) -> bool:
    """Only a complete, infrastructure-clean measurement may be ranked.

    A skipped *scored* category shrinks the denominator and would silently
    reward the run, so an incomplete run is not ranked either. A skipped
    showcase task (K/L) does not touch the verdict.
    """
    return (
        bool(summary.get("eligible"))
        and bool(summary.get("full_suite"))
        and not summary.get("skipped_scored")
    )


def unranked_reason(summary: dict[str, Any]) -> str:
    """Short reason why a run cannot take a place."""
    if summary.get("integrity_failures"):
        return "удостоверение отозвано"
    if not summary.get("eligible"):
        return "инфраструктура"
    return "неполный прогон"


def _comparison_gate(left: dict[str, Any], right: dict[str, Any], *, same_contract: bool) -> str:
    """Why this pair is not a rank comparison; empty when it is one."""
    if not same_contract:
        return "другой контракт сьюта"
    if not rankable(left):
        return f"база — {unranked_reason(left)}"
    if not rankable(right):
        return unranked_reason(right)
    return ""


def _headline_keys(summary: dict[str, Any]) -> tuple[str, ...]:
    """Categories the headline of that run counts, mirroring :func:`summarise`."""
    return SHORT_ORDER if summary.get("admission_state") == "failed" else core.SCORED_CATEGORIES


def compare_summaries(
    left: dict[str, Any], right: dict[str, Any], *, same_contract: bool = True
) -> dict[str, Any]:
    """Compare two summaries on shared tasks; ``significant`` is a paired test.

    The estimand is the difference of the two ratios over the *same* task
    instances, stratified by family. Two separate intervals overlapping does
    not block the verdict: a consistent 0.05-per-task edge is significant even
    when both marginals share a band. Which side wins is published per weight
    mixture, because the published weights are product policy.

    ``same_contract`` and rankability gate the *verdict*, not the arithmetic:
    runs from another suite contract, or ones that are incomplete or
    infrastructure-broken, still show their difference and interval but are
    marked ``ranked = False`` and take no part in the significance claim.
    """
    keys = tuple(key for key in _headline_keys(left) if key in _headline_keys(right))
    units = [unit for key in keys
             for unit in _paired_units(left["per_category"].get(key), right["per_category"].get(key), key)]
    test = stats.paired_ratio_difference(units)
    delta_ci = test["ci95"]
    delta = test["delta"]
    gate = _comparison_gate(left, right, same_contract=same_contract)
    overall = {
        "left_ci95": left.get("headline_ci95"),
        "right_ci95": right.get("headline_ci95"),
        "left_pct": round(100.0 * left["headline_earned"] / left["headline_possible"], 2)
        if left.get("headline_possible") else None,
        "right_pct": round(100.0 * right["headline_earned"] / right["headline_possible"], 2)
        if right.get("headline_possible") else None,
        "delta": None if delta is None else round(100.0 * delta, 2),
        "delta_ci95": None if delta_ci is None
        else (round(100.0 * delta_ci[0], 2), round(100.0 * delta_ci[1], 2)),
        "pairs": test["n_pairs"],
        "significant": test["significant"] and not gate,
        "ranked": not gate,
        "unranked_reason": gate,
        "overlap": _overlaps(left.get("headline_ci95"), right.get("headline_ci95")),
        "provisional": bool(left.get("headline_provisional") or right.get("headline_provisional")),
    }
    return {
        "method": "paired stratified bootstrap",
        "overall": overall,
        "sample_plan": stats.sample_plan(pairs=test["n_pairs"], strata=test["strata"]),
        "category_vector": {"left": _category_vector(left), "right": _category_vector(right)},
        "weight_sensitivity": _weight_sensitivity(left, right),
        "categories": {
            key: _compare_pair(left["per_category"].get(key), right["per_category"].get(key), key)
            for key in sorted(core.CATEGORY_POINTS)
        },
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        # ``nan``/``inf`` are not numbers a reader can compare; older saved
        # speed blocks still carry them.
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _environment(record: dict[str, Any]) -> list[str]:
    """The «Среда» section: what the numbers were executed on.

    An unverified host (no Docker) is stated, not hidden — it is a limitation
    of the evidence, not a reason to withhold the report.
    """
    if not record:
        return []
    verified = str(record.get("status") or "") == "verified"
    container = str(record.get("executor_status") or "") == "container_run"
    # A verified image says the tag resolves to an immutable id — not that the
    # run used it. Reporting "подтверждено" next to ``host_run`` read as a
    # contradiction, so the status names the executor too.
    if verified and container:
        status = "подтверждено"
    elif verified:
        status = "образ подтверждён, но не применялся"
    else:
        status = "не проверено"
    if record.get("reason"):
        status += f": {record['reason']}"
    lines = [
        "## Среда",
        "",
        "| Поле | Значение |",
        "| --- | --- |",
        f"| Образ | `{record.get('image') or '—'}` |",
        f"| Digest образа | `{record.get('image_digest') or '—'}` |",
        f"| Статус образа | {status} |",
        f"| Runtime | `{record.get('runtime') or '—'}`"
        + ("" if record.get("runtime_verified") else " (не подтверждён)")
        + " |",
        f"| Исполнитель | {record.get('executor_status') or '—'}"
        + (f": {record.get('executor_reason')}" if record.get("executor_reason") else "")
        + " |",
        f"| SOURCE_DATE_EPOCH | {record.get('source_date_epoch', '—')} |",
    ]
    packages = [str(item) for item in record.get("sbom") or []]
    if packages:
        digest = sha256("\n".join(packages).encode("utf-8")).hexdigest()
        lines.append(f"| SBOM | {len(packages)} пакетов, SHA-256 `{digest[:16]}` |")
    lines.append("")
    if packages:
        lines.append("<details><summary>SBOM (пакет / версия)</summary>")
        lines.append("")
        lines.append("```")
        lines.extend(packages[:200])
        if len(packages) > 200:
            lines.append(f"… ещё {len(packages) - 200}")
        lines.append("```")
        lines.append("</details>")
        lines.append("")
    return lines


def _observed(observed: dict[str, Any]) -> list[str]:
    """Observed lane: what the runtime said about the loaded server.

    ``unknown`` is written out, never omitted: a short table would otherwise
    read as "nothing to report" and the declared value would fill the gap by
    itself.
    """
    from .reference import OBSERVED_FIELDS

    if not observed:
        return []
    lines = [
        "## Наблюдённая конфигурация",
        "",
        "Со слов runtime (`/props`). `unknown` — не измерено, а не равно объявленному.",
        "",
        "| Поле | Значение |",
        "| --- | --- |",
    ]
    for key, title in OBSERVED_FIELDS:
        lines.append(f"| {title} | {observed.get(key) or 'unknown'} |")
    lines.append("")
    return lines


def _launch_lanes(config: dict[str, Any], lines: list[str]) -> None:
    """Executed load first, declared template second — never one as the other."""
    executed = config.get("launch_executed") or {}
    if executed:
        lines.append("Выполненная загрузка (executed):")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(executed, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    if config.get("launch"):
        lines.append("Объявленный шаблон загрузки (declared — не запись исполнения):")
        lines.append("")
        lines.append("```")
        lines.append(str(config["launch"]))
        lines.append("```")
        lines.append("")


_METRIC_LABELS = {
    "spend_tokens": "расход токенов",
    "wall_s": "время задания",
    "context_occupancy": "занятость контекста",
}


def _telemetry_section(run: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    """Cost/time correlations, unknown shares and the guarded cost per point.

    A run without a telemetry index still renders: every metric is then simply
    unknown, which is the point — it must not look like free work.
    """
    lines = ["## Телеметрия и корреляции", ""]
    index = run.get("telemetry") or []
    data = telemetry.rows(index, run.get("scores") or [])
    scored_rows = [row for row in data if not row.get("skipped")]
    measured = [row for row in scored_rows if row.get("measured")]
    if not measured:
        lines.append(
            "Телеметрия не собрана: стоимость, TTFT и занятость контекста неизвестны "
            "(не нули)."
        )
        lines.append("")
        lines.append("Стоимость балла: не рассчитана — нет расхода токенов.")
        lines.append("")
        return lines

    counts = {"budgeted": 0, "unbounded": 0, "unknown": 0}
    for row in scored_rows:
        mode = row.get("budgeted")
        counts["unknown" if mode is None else ("budgeted" if mode else "unbounded")] += 1
    lines.append(
        f"Режим бюджета: budgeted {counts['budgeted']}, unbounded {counts['unbounded']}, "
        f"неизвестно {counts['unknown']}. Неизвестное значение остаётся пустым и не "
        "складывается с нулём."
    )
    lines.append("")
    compact = telemetry.summary_index(measured)
    lines.append(viz.telemetry_index(compact))
    lines.append("")
    cache_unknown = {
        category: sum(
            1
            for row in measured
            if row.get("category") == category and row.get("cache_share") is None
        )
        for category in {row.get("category") for row in measured}
    }
    lines.append(
        "| Категория | Задач | Медиана TTFT, с | Медиана cache, % | Медиана tok/s | "
        "Медиана времени, с |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in compact:
        share = row.get("cache_share")
        cache = "н/п" if share is None else f"{share * 100:.0f} %"
        unknown_cache = cache_unknown.get(row["category"]) or 0
        if unknown_cache:
            cache += f" (н/п у {unknown_cache})"
        lines.append(
            f"| {row['category']} | {row['n']} | {_fmt(row.get('ttft_s'))} | {cache} | "
            f"{_fmt(row.get('decode_tokens_per_s'), 1)} | {_fmt(row.get('wall_s'))} |"
        )
    lines.append("")

    shares = telemetry.unknown_shares(scored_rows)
    categories = sorted(shares)
    metrics = list(telemetry.METRICS)
    lines.append("Доля неизвестных значений:")
    lines.append("")
    lines.append("| Категория | " + " | ".join(_METRIC_LABELS[m] for m in metrics) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in metrics) + " |")
    for category in categories:
        cells = []
        for metric in metrics:
            row = shares[category][metric]
            share = row.get("share")
            cells.append("—" if share is None else f"{share * 100:.0f} % ({row['unknown']}/{row['total']})")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines.append("")

    correlations = [row for row in telemetry.correlations(scored_rows) if row["n"]]
    if correlations:
        lines.append("Корреляция Спирмена: балл × стоимость/время/контекст.")
        lines.append("")
        lines.append("| Категория | Метрика | ρ | n | Неизвестно |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in correlations:
            rho = "—" if row["rho"] is None else f"{row['rho']:+.2f}"
            lines.append(
                f"| {row['category']} | {_METRIC_LABELS[row['metric']]} | {rho} | "
                f"{row['n']} | {row['unknown']} |"
            )
        lines.append("")

    complete = (
        bool(scored_rows)
        and not summary.get("incomplete")
        and summary.get("eligible", True)
        and all(row.get("spend_tokens") is not None for row in scored_rows)
    )
    cost = telemetry.cost_per_point(scored_rows, complete=complete)
    if cost is not None:
        lines.append(
            f"Стоимость балла: {cost['tokens_per_point']:.1f} токенов на балл "
            f"({cost['tokens']} токенов на {cost['earned']:.0f} баллов; измерение полное)."
        )
    else:
        reason = (
            "есть неизвестный расход"
            if any(row.get("spend_tokens") is None for row in scored_rows)
            else "прогон неполон или не ранжируется"
        )
        lines.append(f"Стоимость балла: не рассчитана — {reason}.")
    lines.append("")
    return lines


def render_markdown(run: dict[str, Any]) -> str:
    config = run.get("config", {})
    provider = run.get("provider", {})
    hardware = run.get("hardware", {})
    suite = run.get("suite", {})
    # The digest, not its ``tasks`` count: the count includes the visual block,
    # which never enters the verdict, so an int would make a full run look
    # partial (47 scored tasks "against" 96).
    summary = summarise(run["scores"], suite)

    lines: list[str] = []
    model = config.get("model", "unknown")
    quant = config.get("quant") or "n/a"
    lines.append(f"# DailyBench-16 — {model} ({quant})")
    lines.append("")
    lines.append("| Поле | Значение |")
    lines.append("| --- | --- |")
    lines.append(f"| Модель | `{model}` |")
    lines.append(f"| Квант | {quant} |")
    lines.append(f"| Runtime | {config.get('runtime') or provider.get('provider', 'n/a')} |")
    lines.append(f"| Endpoint | `{config.get('base_url', '')}` |")
    lines.append(f"| Профиль | `{config.get('profile') or core.DEFAULT_PROFILE}` |")
    lines.append(f"| Веса SHA-256 | `{config.get('weights_sha256') or 'не указан'}` |")
    lines.append(f"| KV-cache | {config.get('kv_type') or 'n/a'} |")
    lines.append(f"| Окно контекста | {config.get('context_window')} |")
    sizing = run.get("sizing") or {}
    if sizing.get("calibrated"):
        lines.append(
            f"| Калибровка long context | {sizing['chars_per_token']:.2f} символов/токен "
            f"({sizing.get('how', 'измерено')}) |"
        )
    elif config.get("chars_per_token") and abs(float(config["chars_per_token"]) - 3.2) > 1e-9:
        lines.append(f"| Калибровка long context | {float(config['chars_per_token']):.2f} символов/токен (задано, не измерено) |")
    lines.append(f"| Offload | {config.get('offload', 'n/a')} |")
    if config.get("agent"):
        categories = ", ".join(config.get("agent_categories") or ())
        lines.append(
            f"| Каркас артефактов | агент pi, категории {categories}: читает и правит рабочую копию, "
            "оценка по файлу на диске (`--no-agent` переключает на чат) |"
        )
    lines.append(f"| Seed | {config.get('seed')} |")
    lines.append(f"| Бюджеты | max_new_tokens={config.get('max_new_tokens')}, "
                 f"max_total_tokens={config.get('max_total_tokens')}, max_tool_calls={config.get('max_tool_calls')} |")
    if hardware:
        lines.append(
            f"| GPU сэмплера (хост харнесса) | {hardware.get('gpu') or hardware.get('name', 'unknown')} "
            f"({hardware.get('vram_total_mib', '?')} MiB, driver {hardware.get('driver', '?')}) |"
        )
    lines.append(f"| Suite SHA-256 | `{suite.get('sha256', 'n/a')}` |")
    if config.get("reference"):
        lines.append(f"| Reference config | `{config['reference']}` |")
    if config.get("unsupported"):
        lines.append(f"| Не проверялось | {', '.join(config['unsupported'])} |")
    lines.append("")
    lines.extend(_environment(run.get("environment") or {}))
    lines.extend(_observed(config.get("observed") or {}))
    _launch_lanes(config, lines)
    lines.append(f"**Итог: {summary['verdict']}**")
    lines.append("")
    lines.append(
        "Ранг: "
        + (
            f"не ранжируется с baseline (профиль `{summary.get('profile')}` — свой набор "
            "категорий, знаменатель не совпадает)"
            if not summary.get("rankable_with_baseline", True)
            else "допущен (полное измерение, инфраструктура чиста)"
            if not summary.get("incomplete") and summary.get("eligible")
            else "не ранжируется: " + (
                "удостоверение отозвано"
                if summary.get("integrity_failures")
                else "инфраструктурный провал"
                if not summary.get("eligible")
                else "прогон неполон — знаменатель урезан"
            )
        )
    )
    if summary.get("headline_ci95"):
        low, high = summary["headline_ci95"]
        note = stats.note(summary.get("headline_n") or 0)
        lines.append("")
        lines.append(
            f"95 % ДИ итога: {low:.0f}–{high:.0f} % от возможного (n={summary.get('headline_n') or 0}"
            + (f", {note}" if note else "") + ")."
        )
    rescored = run.get("rescored") or {}
    if rescored:
        lines.append("")
        lines.append(
            f"> Баллы пересчитаны {rescored.get('rescored_at', '?')} по сохранённым ответам "
            f"(изменений: {len(rescored.get('changed') or [])}); модель повторно не вызывалась."
        )
    lines.append("")
    lines.append("| Категория | Баллы | Допуск | Ошибок прогона | 95 % ДИ |")
    lines.append("| --- | ---: | --- | ---: | --- |")
    for key in sorted(core.CATEGORY_POINTS):
        row = summary["per_category"].get(key)
        if not row or not row["tasks"]:
            continue
        title = core.CATEGORY_TITLES.get(key, key)
        if row["skipped"] == row["tasks"]:
            lines.append(f"| {key}. {title} | не проверялось | — | {row['errors']} | — |")
            continue
        score_text = f"{row['earned']:.0f} / {row['possible']:.0f}"
        if row["skipped"]:
            score_text += f" (не проверялось: {row['skipped']})"
        admitted = {"passed": "да", "unverified": "не подтверждён"}.get(row["admission"], "нет")
        interval = "—"
        if row.get("ci95"):
            low, high = row["ci95"]
            interval = f"{low:.0f}–{high:.0f} %"
            if row.get("provisional"):
                interval += f" ({stats.note(row['n'])})"
        lines.append(f"| {key}. {title} | {score_text} | {admitted} | {row['errors']} | {interval} |")
    lines.append("")
    lines.append("## Задания")
    lines.append("")
    lines.append("| ID | Балл | Стоп-причина | Ошибка прогона | Проваленные чеки |")
    lines.append("| --- | ---: | --- | --- | --- |")
    runs_by_id = {entry.get("task", {}).get("id"): entry for entry in (run.get("tasks") or [])}
    for score in run["scores"]:
        if score.get("skipped"):
            lines.append(f"| {score['task_id']} | не проверялось | — | — | {score['skipped']} |")
            continue
        failed = [c["name"] for c in score.get("checks", []) if not c["passed"]]
        error_text = score.get("error") or "—"
        if score.get("failure"):
            error_text = f"{core.failure_label(score['failure'])}: {error_text}"
        if score.get("category") in core.ART_CATEGORIES:
            status = "отрисовано" if not failed and score.get("earned") else "не отрисовалось"
            lines.append(
                f"| {score['task_id']} | {status} | {_stop_reason(runs_by_id.get(score['task_id']))} | "
                f"{error_text} | {', '.join(failed) if failed else '—'} |"
            )
            continue
        lines.append(
            f"| {score['task_id']} | {score.get('earned', 0):.0f}/{score.get('possible', 0):.0f} | "
            f"{_stop_reason(runs_by_id.get(score['task_id']))} | "
            f"{error_text} | {', '.join(failed) if failed else '—'} |"
        )
    lines.append("")

    showcase = summary.get("showcase") or {}
    if any(row["tasks"] for row in showcase.values()):
        lines.append("## Витрина: картинки вне общего счёта")
        lines.append("")
        lines.append(
            f"Симуляции K — {summary['showcase_earned']:.0f} / {summary['showcase_possible']:.0f} "
            "баллов вне итоговой суммы. V — чтение изображений (график, пространственные "
            "отношения, трассировка схемы): проверяется ответ по картинке. L — свободные "
            "арт-работы: сохраняются страница и кадр, цвет, тональность и художественное "
            "качество оценивает человек."
        )
        lines.append("")
        for key in sorted(showcase):
            row = showcase[key]
            if not row["tasks"]:
                continue
            lines.append(f"### {key}. {core.SHOWCASE_TITLES.get(key, key)}")
            lines.append("")
            if row["skipped"] == row["tasks"]:
                lines.append("не проверялось")
                lines.append("")
                continue
            if key in core.ART_CATEGORIES:
                lines.append("Без художественных баллов. Автоматически проверяется только загрузка HTML.")
                lines.append("")
                continue
            lines.append(
                f"Итог блока: {row['earned']:.0f} / {row['possible']:.0f}"
                + (f", не проверялось: {row['skipped']}" if row["skipped"] else "")
            )
            lines.append("")
            lines.append("| ID | Баллы | Проваленные проверки |")
            lines.append("| --- | ---: | --- |")
            for score in run["scores"]:
                if score.get("category") != key:
                    continue
                if score.get("skipped"):
                    lines.append(f"| {score['task_id']} | не проверялось | {score['skipped']} |")
                    continue
                failed = [c["name"] for c in score.get("checks", []) if not c["passed"]]
                lines.append(
                    f"| {score['task_id']} | {score.get('earned', 0):.0f}/{score.get('possible', 0):.0f} | "
                    f"{', '.join(failed) if failed else '—'} |"
                )
            lines.append("")
        run_dir = run.get("dir")
        if run_dir and (Path(run_dir) / "visual.html").is_file():
            lines.append("Снимки кадров: [`visual.html`](visual.html).")
            lines.append("")


    lines.extend(_telemetry_section(run, summary))

    speed = run.get("speed")
    if speed:
        lines.append("## Скорость и память (без баллов качества)")
        lines.append("")
        lines.append("| Профиль | TTFT, с | Prompt tok/s | Generation tok/s | Пик VRAM, MiB | Повторов |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in speed:
            lines.append(
                f"| {row.get('label')} | {_fmt(row.get('ttft_s'))} | {_fmt(row.get('prompt_tok_s'), 1)} | "
                f"{_fmt(row.get('generation_tok_s'), 1)} | {row.get('vram_peak_mib', 'n/a')} | {row.get('runs')} |"
            )
        lines.append("")
    return "\n".join(lines)


def compare_markdown(runs: list[dict[str, Any]]) -> str:
    """Side-by-side table across runs: quality first, then memory and speed."""
    from . import reference as reference_module

    lines = ["# Сравнение конфигураций", ""]
    mismatches = reference_module.controlled_mismatches([run.get("config") or {} for run in runs])
    if mismatches:
        lines.append("**Это не сравнение весов**: контролируемые параметры расходятся, поэтому "
                     "разница в баллах включает их, а не только веса.")
        lines.append("")
        lines.extend(f"- {line}" for line in mismatches)
        lines.append("")
    header = ["Модель", "Профиль", "Квант", "Итог"]
    for key in sorted(core.CATEGORY_POINTS):
        header.append(f"{key}, %")
    header += [f"{key} (визуальный), %" for key in sorted(core.SHOWCASE_CATEGORIES)]
    header += [f"{key} (арт)" for key in core.ART_CATEGORIES]
    header += ["Пик VRAM, MiB", "TTFT long, с", "Gen tok/s", "Ошибок"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    compared: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for run in runs:
        config = run.get("config", {})
        summary = summarise(run["scores"], run.get("suite") or {})
        row = [config.get("model", "?"), config.get("profile") or core.DEFAULT_PROFILE,
               config.get("quant") or "n/a", summary["verdict"]]
        for key in sorted(core.CATEGORY_POINTS):
            entry = summary["per_category"].get(key)
            if not entry or not entry["tasks"]:
                row.append("—")
            elif entry["skipped"] == entry["tasks"]:
                row.append("не проверялось")
            elif entry["admission"] == "failed":
                row.append("не допущен")
            elif entry["admission"] == "unverified":
                row.append("допуск не подтверждён")
            elif entry["possible"]:
                row.append(f"{100 * entry['earned'] / entry['possible']:.0f}")
            else:
                row.append("—")
        for key in sorted(core.SHOWCASE_CATEGORIES):
            entry = (summary.get("showcase") or {}).get(key)
            if not entry or not entry["tasks"]:
                row.append("—")
            elif entry["skipped"] == entry["tasks"]:
                row.append("не проверялось")
            elif entry["possible"]:
                row.append(f"{100 * entry['earned'] / entry['possible']:.0f}")
            else:
                row.append("—")
        for key in core.ART_CATEGORIES:
            entry = (summary.get("showcase") or {}).get(key)
            if not entry or not entry["tasks"]:
                row.append("—")
            elif entry["skipped"] == entry["tasks"]:
                row.append("не проверялось")
            else:
                row.append("оценка человеком")
        speed = run.get("speed") or []
        long_row = next((s for s in speed if str(s.get("label", "")).startswith("long")), None)
        peak = long_row.get("vram_peak_mib") if long_row else None
        row.append(str(peak if peak is not None else "n/a"))
        row.append(_fmt(long_row.get("ttft_s") if long_row else None))
        row.append(_fmt(long_row.get("generation_tok_s") if long_row else None, 1))
        row.append(str(summary["errors"]))
        lines.append("| " + " | ".join(row) + " |")
        compared.append((run, summary))

    lines.append("")
    lines.append("Пик VRAM и TTFT берутся из длинного профиля: это измерение при заполненном "
                 "контексте, а не после загрузки модели.")
    lines.append("")
    lines.append("Колонки витрины показывают визуальный блок: он измеряется отдельно и в итог "
                 "не входит — отсутствие браузера на машине измерения не должно двигать балл.")
    if len(compared) > 1:
        lines.extend(_paired_markdown(compared))
    return "\n".join(lines)


def _label(run: dict[str, Any]) -> str:
    """A markdown-safe model label for the comparison sections."""
    return str((run.get("config") or {}).get("model") or run.get("dir") or "?")


def _paired_markdown(compared: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[str]:
    """Difference intervals, the category vector and the weight sensitivity.

    The first run is the baseline. ``significant`` is the paired test, not the
    overlap of two marginal bands — that overlap is printed next to it so a
    conservative reading is still possible.
    """
    baseline_run, baseline = compared[0]
    results = [
        compare_summaries(baseline, summary, same_contract=contract_key(run) == contract_key(baseline_run))
        for run, summary in compared[1:]
    ]
    lines = [
        "",
        "## Парное сравнение: разность и её интервал",
        "",
        f"База — `{_label(baseline_run)}` (первый прогон в порядке вызова, как и в HTML). "
        "Сравниваются одинаковые task_id, стратифицированно по семействам. Δ — парная разность на "
        "общих задачах; доли каждой стороны от её собственного набора — отдельные колонки. "
        "Значимость — 95 % интервал разности не содержит нуля; перекрытие отдельных ДИ указано "
        "рядом и вердикт не блокирует. Прогон вне контракта базы или неполный места в вердикте "
        "не получает.",
        "",
        "| Модель | Δ (парная), п.п. | 95 % ДИ разности | База, % | Модель, % | Общих задач | "
        "Перекрытие отдельных ДИ | Значимо |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    base_pct = round(
        100.0 * baseline["headline_earned"] / baseline["headline_possible"], 2
    ) if baseline.get("headline_possible") else None
    for (run, _), result in zip(compared[1:], results):
        overall = result["overall"]
        ci = overall["delta_ci95"]
        band = f"{ci[0]:+.2f} … {ci[1]:+.2f}" if ci else "n/a"
        verdict = (
            ("да" if overall["significant"] else "нет") if overall["ranked"]
            else f"не ранжируется: {overall['unranked_reason']}"
        )
        lines.append(
            f"| {_label(run)} | {_fmt(overall['delta'], 2)} | {band} | {_fmt(base_pct)} | "
            f"{_fmt(overall['right_pct'])} | {overall['pairs'] or 'n/a'} | "
            f"{'да' if overall['overlap'] else 'нет'} | {verdict} |"
        )
    plan = results[0]["sample_plan"]
    lines += [
        "",
        f"Популяция: {plan['population']}; оцениваемое: {plan['estimand']}. "
        f"Не измеряются: {', '.join(plan['unmeasured'])}. {plan['note']}",
        "",
        "## Вектор категорий",
        "",
        "Один итог — это смесь весов, поэтому публикуется и вектор по категориям.",
        "",
        "| Модель | " + " | ".join(sorted(core.CATEGORY_POINTS)) + " |",
        "| --- |" + "---:|" * len(core.CATEGORY_POINTS),
    ]
    for run, summary in compared:
        vector = _category_vector(summary)
        lines.append(
            f"| {_label(run)} | "
            + " | ".join(_fmt(vector[key]) for key in sorted(core.CATEGORY_POINTS)) + " |"
        )
    lines += [
        "",
        "## Чувствительность к весам",
        "",
        "Дельта под каждой политикой весов: устойчивый вывод не зависит от смеси. "
        "Знаменатель — полный вес профиля, поэтому непомеренная категория не даёт бесплатного "
        "прохода: её доля веса указана рядом с дельтой.",
        "",
        "| Политика | " + " | ".join(_label(run) for run, _ in compared[1:]) + " |",
        "| --- |" + "---:|" * max(0, len(compared) - 1),
    ]
    for scheme in WEIGHT_SCHEMES:
        cells = []
        for result in results:
            entry = result["weight_sensitivity"]["schemes"][scheme]
            cells.append(
                f"{_fmt(entry['delta'], 2)} ({entry['winner'] or 'n/a'})"
                + (f" — не измерено {entry['unknown']:.0f}% веса" if entry["unknown"] else "")
            )
        lines.append(f"| {scheme} | " + " | ".join(cells) + " |")
    stable = all(result["weight_sensitivity"]["stable"] for result in results)
    lines.append("")
    lines.append(
        "Вывод " + ("устойчив к весам." if stable else "меняется вместе с весами — это свойство смеси.")
    )
    return lines


def _shot_of(entry: dict[str, Any], run_dir: Path) -> str:
    """Path of the frame the visual check stored for one task, if any.

    A recorded path is only trusted when it lies inside the run or the render
    cache, the same roots the gallery copies from: an imported ``tasks.json``
    must not make the page reference (and the writer copy) an arbitrary file.
    """
    transcript = entry.get("transcript") or {}
    info = (transcript.get("extra") or {}).get("visual") or {}
    shot = str(info.get("shot") or "")
    if not shot:
        return ""
    path = Path(shot)
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file() or not gallery._within(path, gallery._allowed_roots(run_dir)):
        return ""
    return str(path.resolve())


def write_visual_gallery(run: dict[str, Any]) -> Path | None:
    """Copy the rendered frames next to the run and write a gallery page.

    The frames are the point of the visual block: the score says whether a page
    works, the picture says what it looks like. They are cached outside the run
    (keyed by page content), so a re-run of the same answer reuses them.
    """
    entries = [
        entry
        for entry in (run.get("tasks") or [])
        if (entry.get("task") or {}).get("category") in core.VISUAL_CATEGORIES
    ]
    if not entries:
        return None
    run_dir = Path(run["dir"])
    target = run_dir / "visual"
    target.mkdir(parents=True, exist_ok=True)
    scores = {score["task_id"]: score for score in run["scores"]}

    cards: list[str] = []
    copied = 0
    for entry in sorted(entries, key=lambda item: (item.get("task") or {}).get("id", "")):
        task = entry.get("task") or {}
        task_id = task.get("id", "?")
        title = str(task.get("title", ""))
        if title.startswith(f"{task_id} —") or title.startswith(f"{task_id} -"):
            title = title.split("—", 1)[-1].strip() if "—" in title else title.split("-", 1)[-1].strip()
        score = scores.get(task_id) or {}
        earned, possible = score.get("earned"), score.get("possible")
        shot = _shot_of(entry, run_dir)
        asset = next(iter(task.get("assets") or []), None)
        if not shot and isinstance(asset, dict) and asset.get("data"):
            # An image-input scenario has no rendered page: the presented
            # picture *is* the frame, and it is stored in the task document.
            try:
                raw = base64.b64decode(str(asset["data"]), validate=True)
            except Exception:
                raw = b""
            if raw:
                name = f"{task_id}.png"
                (target / name).write_bytes(raw)
                copied += 1
                image = f'<img src="visual/{_esc(name)}" alt="{_esc(task_id)}" loading="lazy">'
            else:
                image = '<div class="missing">картинки нет в задании</div>'
        elif shot:
            # The id becomes a file name: ``../x`` would land outside ``visual/``,
            # so anything off-schema is reduced to one path component.
            name = f"{task_id}.png" if gallery._ID_SCHEMA.fullmatch(task_id) else f"{_file_name(task_id)}.png"
            shutil.copyfile(shot, target / name)
            copied += 1
            image = f'<img src="visual/{_esc(name)}" alt="{_esc(task_id)}" loading="lazy">'
        else:
            image = '<div class="missing">кадра нет: страница не отрисовалась</div>'
        failed = [c["name"] for c in score.get("checks", []) if not c["passed"]]
        status = (
            f"{earned:.0f}/{possible:.0f}"
            if isinstance(earned, (int, float)) and isinstance(possible, (int, float))
            else "не проверялось"
        )
        block = core.SHOWCASE_TITLES.get(task.get("category", ""), task.get("category", ""))
        cards.append(
            f'<figure><figcaption><span class="block">{_esc(block)}</span> {_esc(task_id)} — {_esc(title)} '
            f'<span class="score">{_esc(status)}</span></figcaption>{image}'
            f'<div class="checks">{_esc("провалено: " + ", ".join(failed) if failed else "все проверки пройдены")}</div>'
            "</figure>"
        )

    page = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Витрина — {_esc(run_dir.name)}</title>
<style>
 body {{ background:#0d1117; color:#e6edf3; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; margin:24px }}
 h1 {{ font-size:20px; margin:0 0 4px }}
 p.hint {{ color:#8b949e; margin:0 0 20px }}
 figure {{ margin:0 0 24px; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px }}
 figcaption {{ font-weight:600; margin-bottom:8px }}
 .score {{ color:#7ee787; font-weight:400; margin-left:6px }}
 img {{ width:100%; max-width:800px; border-radius:4px; display:block }}
 .checks {{ color:#8b949e; margin-top:8px }}
 .block {{ color:#79c0ff; font-size:12px; letter-spacing:.04em; text-transform:uppercase }}
 .missing {{ padding:40px; text-align:center; color:#8b949e; border:1px dashed #30363d }}
</style></head><body>
<h1>Витрина: кадры вне общего счёта</h1>
<p class="hint">{copied} из {len(entries)} кадров · баллы витрины вне общего счёта</p>
{''.join(cards)}
</body></html>
"""
    path = run_dir / "visual.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_reports(run_dirs: list[Path | str]) -> list[Path]:
    """Render ``report.md`` for every run directory and a comparison when >1."""
    written: list[Path] = []
    runs = [load_run(directory) for directory in run_dirs]
    for run in runs:
        gallery = write_visual_gallery(run)
        path = Path(run["dir"]) / "report.md"
        path.write_text(render_markdown(run), encoding="utf-8")
        written.append(path)
        if gallery is not None:
            written.append(gallery)
    if len(runs) > 1:
        target = Path(runs[0]["dir"]).parent / "comparison.md"
        target.write_text(compare_markdown(runs), encoding="utf-8")
        written.append(target)
    return written
