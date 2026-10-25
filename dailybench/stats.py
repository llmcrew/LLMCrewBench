"""Uncertainty for report numbers: seeded bootstrap, Wilson, and significance.

Every published metric carries an interval. Two rules keep the intervals
honest:

* an interval is computed by resampling the observed units (tasks), never by
  inventing a distribution around a point estimate;
* a difference is estimated on the *same* task instances: the interval of the
  difference comes from a paired, category-stratified bootstrap, so a shared
  effect survives even when the two marginal intervals overlap. Overlap of the
  marginals is recorded, but it no longer decides the verdict;
* the proportion interval of a weighted sum is the bootstrap of that sum, never
  a Wilson interval of an arbitrary ratio.

Small samples are still measured and flagged as provisional, but
:data:`SMALL_SAMPLE` is a hint, not a licence to rank: a difference is called
significant only when its own interval excludes zero.

Nothing here measures repeat calls or independent seeds. The bootstrap
resamples the tasks that were run once; :func:`sample_plan` states that gap.

All randomness comes from an explicitly seeded ``random.Random``, so the same
scores produce the same interval on every rebuild.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Any, Iterable, Sequence

RESAMPLES = 10_000
"""Resamples per interval. Fixed so an interval costs the same on every run."""

SEED = 20261021
"""Default seed: a rebuild of one generation must reproduce every interval."""

LEVEL = 0.95
"""Confidence level of the published intervals."""

SMALL_SAMPLE = 30
"""Below this many units a metric is published with the "ориентировочно" note."""

PROVISIONAL_NOTE = "ориентировочно"

SMALL_SAMPLE_CAVEAT = (
    "порог n=%d — ориентир, а не допуск: ранжирует только интервал разности"
) % SMALL_SAMPLE

MIN_PAIRS = 2
"""Below this many shared tasks the difference interval cannot discriminate."""

PairedUnit = tuple[str, float, float, float, float]
"""``(stratum, left_earned, left_possible, right_earned, right_possible)``."""


def is_small(n: int) -> bool:
    """Whether a sample of ``n`` units is too small to rank on."""
    return n < SMALL_SAMPLE


def note(n: int) -> str:
    """The caveat printed next to a small-sample metric, empty when sound."""
    return PROVISIONAL_NOTE if is_small(n) else ""


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation quantile of an already sorted sequence."""
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low]) * (1.0 - weight) + float(ordered[high]) * weight


def _interval(draws: Sequence[float], level: float) -> tuple[float, float]:
    ordered = sorted(draws)
    tail = (1.0 - level) / 2.0
    low, high = _quantile(ordered, tail), _quantile(ordered, 1.0 - tail)
    return (low, high) if low <= high else (high, low)


def _rng(seed: int | None) -> random.Random:
    return random.Random(SEED if seed is None else seed)


def bootstrap_mean(
    values: Iterable[float],
    *,
    resamples: int = RESAMPLES,
    seed: int | None = None,
    level: float = LEVEL,
) -> tuple[float, float]:
    """Percentile-bootstrap interval of a mean."""
    data = [float(value) for value in values]
    if not data:
        return (0.0, 0.0)
    rng = _rng(seed)
    draws = [sum(data[rng.randrange(len(data))] for _ in range(len(data))) / len(data)
             for _ in range(resamples)]
    return _interval(draws, level)


def bootstrap_ratio(
    samples: Iterable[tuple[float, float]],
    *,
    resamples: int = RESAMPLES,
    seed: int | None = None,
    level: float = LEVEL,
) -> tuple[float, float]:
    """Interval of ``sum(earned) / sum(possible)`` over resampled tasks.

    ``samples`` are per-task ``(earned, possible)`` pairs. Resampling the pairs
    keeps the weighting of a heavy task and the task-level correlation between
    earned and possible, which a mean of shares would lose.
    """
    data = [(float(earned), float(possible)) for earned, possible in samples if possible > 0]
    if not data:
        return (0.0, 0.0)
    rng = _rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        earned = possible = 0.0
        for _ in range(len(data)):
            one, room = data[rng.randrange(len(data))]
            earned += one
            possible += room
        draws.append(earned / possible if possible else 0.0)
    return _interval(draws, level)


def wilson_interval(successes: float, total: float, *, level: float = LEVEL) -> tuple[float, float]:
    """Wilson score interval of a proportion.

    Used where a pass rate is binary: unlike the normal approximation it stays
    inside 0–1 and keeps a sane width at 0/n and n/n.
    """
    n = float(total)
    if n <= 0:
        return (0.0, 0.0)
    share = min(max(float(successes) / n, 0.0), 1.0)
    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    denominator = 1.0 + z * z / n
    centre = (share + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt(share * (1.0 - share) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_proportion(
    successes: int,
    total: int,
    *,
    resamples: int = RESAMPLES,
    seed: int | None = None,
    level: float = LEVEL,
) -> tuple[float, float]:
    """Percentile-bootstrap interval of a pass proportion."""
    if total <= 0:
        return (0.0, 0.0)
    data = [1.0] * max(0, int(successes)) + [0.0] * max(0, total - int(successes))
    return bootstrap_mean(data, resamples=resamples, seed=seed, level=level)


def disjoint(left: tuple[float, float] | None, right: tuple[float, float] | None) -> bool:
    """Whether two 95 % intervals fail to overlap.

    Kept as evidence, not as the verdict: overlapping marginals do not prove
    that the paired difference is zero. :func:`paired_ratio_difference`
    decides significance.
    """
    if not left or not right:
        return False
    return left[1] < right[0] or right[1] < left[0]


def _ratio(pairs: Sequence[tuple[float, float]]) -> float:
    earned = sum(earned for earned, _ in pairs)
    possible = sum(possible for _, possible in pairs)
    return earned / possible if possible else 0.0


def paired_ratio_difference(
    units: Iterable[PairedUnit],
    *,
    resamples: int = RESAMPLES,
    seed: int | None = None,
    level: float = LEVEL,
) -> dict[str, Any]:
    """Paired, stratified percentile bootstrap of a ratio difference.

    ``units`` are the *same* task instances measured under both sides, as
    ``(stratum, left_earned, left_possible, right_earned, right_possible)``.
    Each resample redraws tasks inside its stratum and recomputes both ratios
    from the drawn pairs, so the shared task effect cancels instead of being
    swamped by between-task spread.

    Returns the difference in ratio units (left − right), its interval, and
    ``significant``: the whole interval stays on one side of zero. A single
    shared task cannot discriminate and stays insignificant.
    """
    data: list[PairedUnit] = [
        (str(stratum), float(left_earned), float(left_possible), float(right_earned), float(right_possible))
        for stratum, left_earned, left_possible, right_earned, right_possible in units
        if left_possible > 0 and right_possible > 0
    ]
    strata: dict[str, list[PairedUnit]] = {}
    for unit in data:
        strata.setdefault(unit[0], []).append(unit)
    result: dict[str, Any] = {
        "delta": None,
        "ci95": None,
        "significant": False,
        "n_pairs": len(data),
        "strata": len(strata),
    }
    if not data:
        return result
    delta = _ratio([(unit[1], unit[2]) for unit in data]) - _ratio(
        [(unit[3], unit[4]) for unit in data]
    )
    rng = _rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        left: list[tuple[float, float]] = []
        right: list[tuple[float, float]] = []
        for group in strata.values():
            for _ in range(len(group)):
                unit = group[rng.randrange(len(group))]
                left.append((unit[1], unit[2]))
                right.append((unit[3], unit[4]))
        draws.append(_ratio(left) - _ratio(right))
    low, high = _interval(draws, level)
    result["delta"] = delta
    result["ci95"] = (low, high)
    result["significant"] = len(data) >= MIN_PAIRS and (low > 0.0 or high < 0.0)
    return result


def sample_plan(*, pairs: int, strata: int) -> dict[str, Any]:
    """What the comparison actually resamples, and what it does not.

    The published uncertainty is over task instances only. Repeat calls and
    independent seeds are part of the population the estimate speaks about,
    but they are not measured by a single run — the plan says so instead of
    implying a tighter interval than the evidence supports.
    """
    return {
        "population": "общие task_id двух прогонов",
        "estimand": "разность долей на общих задачах, стратифицированная категориями",
        "method": "paired stratified percentile bootstrap",
        "pairs": pairs,
        "strata": strata,
        "measured": ["task_id", "category"],
        "unmeasured": ["повторы вызова", "независимые сиды"],
        "repeats": None,
        "seeds": None,
        "note": (
            "Один прогон на модель: пересэмплируются задачи, а не повторы и не сиды. "
            + SMALL_SAMPLE_CAVEAT
        ),
    }
