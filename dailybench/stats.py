"""Uncertainty for report numbers: seeded bootstrap, Wilson, and significance.

Every published metric carries an interval. Two rules keep the intervals
honest:

* an interval is computed by resampling the observed units (tasks), never by
  inventing a distribution around a point estimate;
* a difference is called significant only when the two 95 % intervals do not
  overlap — a shared interval is reported as "незначимо", not as a win.

Small samples are still measured, but flagged as provisional: below
:data:`SMALL_SAMPLE` the interval is wide enough that the number guides rather
than ranks.

All randomness comes from an explicitly seeded ``random.Random``, so the same
scores produce the same interval on every rebuild.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Iterable, Sequence

RESAMPLES = 10_000
"""Resamples per interval. Fixed so an interval costs the same on every run."""

SEED = 20261021
"""Default seed: a rebuild of one generation must reproduce every interval."""

LEVEL = 0.95
"""Confidence level of the published intervals."""

SMALL_SAMPLE = 30
"""Below this many units a metric is published with the "ориентировочно" note."""

PROVISIONAL_NOTE = "ориентировочно"


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

    Only then is a model difference called significant; nested or touching
    intervals stay "незначимо".
    """
    if not left or not right:
        return False
    return left[1] < right[0] or right[1] < left[0]
