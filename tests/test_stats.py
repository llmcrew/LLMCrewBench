"""Uncertainty rules: seeded intervals, wide small samples, disjoint significance."""

from __future__ import annotations

import pytest

from dailybench import stats
from dailybench.report import compare_summaries, summarise


def _scores(category: str, earned: list[float], possible: float = 10.0) -> list[dict[str, object]]:
    return [
        {"task_id": f"{category}-{index}", "category": category,
         "earned": value, "possible": possible, "checks": []}
        for index, value in enumerate(earned)
    ]


def test_the_same_seed_reproduces_the_interval():
    values = [0.13, 0.77, 0.42, 0.91, 0.05, 0.66, 0.31, 0.58, 0.24, 0.83]
    first = stats.bootstrap_mean(values, seed=7)
    second = stats.bootstrap_mean(values, seed=7)
    assert first == second
    assert first != stats.bootstrap_mean(values, seed=8)
    assert stats.bootstrap_mean(values) == stats.bootstrap_mean(values)


def test_a_percentile_interval_brackets_the_point_estimate():
    values = [float(index % 2) for index in range(100)]
    low, high = stats.bootstrap_mean(values, seed=1)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert high - low < 0.25  # 100 наблюдений: полоса узкая


def test_a_small_sample_gets_a_wide_interval_and_the_caveat():
    low, high = stats.bootstrap_mean([1.0, 0.0, 1.0])
    assert high - low > 0.5
    assert stats.is_small(3) and not stats.is_small(30)
    assert stats.note(3) == "ориентировочно"
    assert stats.note(30) == ""


def test_wilson_keeps_a_full_house_inside_the_unit_interval():
    low, high = stats.wilson_interval(20, 20)
    assert 0.0 < low < 1.0 and high == pytest.approx(1.0)
    low, high = stats.wilson_interval(0, 5)
    assert low == pytest.approx(0.0) and high < 1.0
    assert stats.wilson_interval(0, 0) == (0.0, 0.0)


def test_a_large_difference_is_significant_and_a_close_one_is_not():
    strong = stats.bootstrap_ratio([(9.0, 10.0) if i % 2 else (10.0, 10.0) for i in range(40)], seed=3)
    weak = stats.bootstrap_ratio([(1.0, 10.0) if i % 2 else (2.0, 10.0) for i in range(40)], seed=3)
    near = stats.bootstrap_ratio([(9.0, 10.0)] * 20 + [(10.0, 10.0)] * 19 + [(5.0, 10.0)], seed=3)
    assert stats.disjoint(strong, weak)
    assert not stats.disjoint(strong, near)
    assert not stats.disjoint(strong, None)


def test_the_ratio_interval_keeps_task_weights():
    # Одна тяжёлая задача с полным баллом не даёт интервалу уехать к единице:
    # ресемпл учитывает вес задачи, а не среднее долей.
    samples = [(10.0, 100.0), (0.0, 1.0)]
    low, high = stats.bootstrap_ratio(samples, seed=5)
    assert low <= 10.0 / 101.0 <= high


def test_a_summary_carries_intervals_and_the_small_sample_note():
    summary = summarise(_scores("A", [10.0, 0.0, 10.0]))
    row = summary["per_category"]["A"]
    assert row["n"] == 3
    assert row["ci95"] is not None
    assert row["provisional"] is True
    assert summary["headline_ci95"] is not None
    assert summary["headline_n"] == 3
    assert summary["headline_provisional"] is True


def test_skipped_and_unobserved_tasks_add_no_samples():
    scores = _scores("A", [10.0]) + [
        {"task_id": "A-skip", "category": "A", "earned": 0.0, "possible": 10.0,
         "skipped": "нет браузера", "checks": []},
    ]
    summary = summarise(scores)
    assert summary["per_category"]["A"]["n"] == 1
    assert summary["per_category"]["B"]["ci95"] is None
    assert summary["per_category"]["B"]["provisional"] is False


def test_large_and_close_summaries_compare_by_interval_overlap():
    strong = summarise(_scores("A", [9.0, 10.0] * 20))
    weak = summarise(_scores("A", [1.0, 2.0] * 20))
    far = compare_summaries(strong, weak)
    assert far["overall"]["significant"] is True
    assert far["categories"]["A"]["significant"] is True
    assert far["overall"]["delta"] == pytest.approx(80.0)

    close = compare_summaries(strong, summarise(_scores("A", [9.0] * 20 + [10.0] * 19 + [5.0])))
    assert close["overall"]["significant"] is False
