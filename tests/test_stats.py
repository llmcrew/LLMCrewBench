"""Uncertainty rules: seeded intervals, wide small samples, disjoint significance."""

from __future__ import annotations

import pytest

from dailybench import stats
from dailybench import report
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


def test_the_paired_difference_keeps_a_constant_edge():
    units = [("A", 10.0, 10.0, 9.5, 10.0)] * 40
    result = stats.paired_ratio_difference(units, seed=2)
    assert result["delta"] == pytest.approx(0.05)
    assert result["ci95"][0] > 0
    assert result["significant"] is True
    assert result["strata"] == 1 and result["n_pairs"] == 40


def test_the_paired_difference_ignores_unpaired_tasks():
    units = [("A", 10.0, 10.0, 9.5, 10.0), ("A", 5.0, 0.0, 1.0, 1.0)]
    result = stats.paired_ratio_difference(units)
    assert result["n_pairs"] == 1
    assert result["significant"] is False


def test_the_sample_plan_states_what_is_not_measured():
    plan = stats.sample_plan(pairs=3, strata=1)
    assert plan["repeats"] is None and plan["seeds"] is None
    assert stats.SMALL_SAMPLE_CAVEAT in plan["note"]
    assert stats.SMALL_SAMPLE == 30


def test_disjoint_intervals_no_longer_decide_the_verdict():
    """A constant per-task edge is significant even when both bands overlap."""
    left = summarise(_scores("A", [6.0 + (index % 5) for index in range(40)]))
    right = summarise(_scores("A", [6.0 + (index % 5) - 0.05 for index in range(40)]))
    result = compare_summaries(left, right)
    assert result["overall"]["overlap"] is True          # отдельные ДИ перекрываются
    assert result["overall"]["significant"] is True      # но парная разность держится
    low, high = result["overall"]["delta_ci95"]
    assert low > 0 and high == pytest.approx(0.5, abs=0.2)
    assert result["categories"]["A"]["significant"] is True
    assert result["overall"]["pairs"] == 40


def test_pairing_survives_a_reordered_task_list():
    left = _scores("A", [6.0 + (index % 5) for index in range(40)])
    right = list(reversed(_scores("A", [6.0 + (index % 5) - 0.05 for index in range(40)])))
    result = compare_summaries(summarise(left), summarise(right))
    assert result["overall"]["pairs"] == 40
    assert result["overall"]["significant"] is True


def test_the_paired_test_reports_its_sample_plan_and_weight_sensitivity():
    left = summarise(_scores("A", [10.0] * 20 + [8.0] * 20))
    right = summarise(_scores("A", [8.0] * 20 + [10.0] * 20))
    result = compare_summaries(left, right)
    assert result["method"] == "paired stratified bootstrap"
    plan = result["sample_plan"]
    assert plan["pairs"] == 40
    assert plan["repeats"] is None and plan["seeds"] is None
    assert "повторы вызова" in plan["unmeasured"]
    assert set(result["weight_sensitivity"]["schemes"]) == set(report.WEIGHT_SCHEMES)
    assert result["category_vector"]["left"]["A"] == pytest.approx(90.0)


def test_a_single_shared_task_cannot_be_significant():
    left = summarise(_scores("A", [10.0]))
    right = summarise(_scores("A", [0.0]))
    result = compare_summaries(left, right)
    assert result["overall"]["pairs"] == 1
    assert result["overall"]["significant"] is False


def test_large_and_close_summaries_compare_by_the_difference_interval():
    strong = summarise(_scores("A", [9.0, 10.0] * 20))
    weak = summarise(_scores("A", [1.0, 2.0] * 20))
    far = compare_summaries(strong, weak)
    assert far["overall"]["significant"] is True
    assert far["categories"]["A"]["significant"] is True
    assert far["overall"]["delta"] == pytest.approx(80.0)

    close = compare_summaries(strong, summarise(_scores("A", [9.0] * 20 + [10.0] * 19 + [5.0])))
    assert close["overall"]["significant"] is False


def test_the_difference_is_zero_when_nothing_is_shared():
    left = summarise(_scores("A", [10.0, 9.0]))
    right = summarise([{"task_id": "B-1", "category": "B", "earned": 15.0,
                        "possible": 15.0, "checks": []}])
    result = compare_summaries(left, right)
    assert result["overall"]["pairs"] == 0
    assert result["overall"]["delta"] is None
    assert result["overall"]["delta_ci95"] is None
    assert result["overall"]["significant"] is False


def test_the_delta_is_paired_and_not_the_marginal_gap():
    """Left measured less but solved the shared tasks better.

    The marginal percentages are 50 % and 95 %, the paired difference over the
    forty shared tasks is +5 p.p. Publishing the marginal gap (``-45``) next to
    an interval built from the shared ids would show a number from another
    sample.
    """
    extra = [
        {"task_id": f"A-{100 + index}", "category": "A", "earned": 0.0,
         "possible": 10.0, "checks": []}
        for index in range(40)
    ]
    left = summarise(_scores("A", [10.0] * 40) + extra)
    right = summarise(_scores("A", [9.5] * 40))
    overall = compare_summaries(left, right)["overall"]
    assert overall["left_pct"] == pytest.approx(50.0)
    assert overall["right_pct"] == pytest.approx(95.0)
    assert overall["delta"] == pytest.approx(5.0)
    low, high = overall["delta_ci95"]
    assert low <= overall["delta"] <= high


def test_the_markdown_publishes_the_paired_delta_next_to_the_marginals():
    def run(model: str, scores: list[dict[str, object]]) -> dict[str, object]:
        return {"dir": f"/tmp/{model}", "config": {"model": model}, "scores": scores,
                "suite": {"sha256": "same", "tasks": 1,
                          "per_task": [{"id": "A-0", "category": "A"}]}}

    extra = [
        {"task_id": f"A-{100 + index}", "category": "A", "earned": 0.0,
         "possible": 10.0, "checks": []}
        for index in range(20)
    ]
    text = report.compare_markdown([
        run("база", _scores("A", [9.5] * 20) + extra),
        run("правая", _scores("A", [10.0] * 20)),
    ])
    section = text.split("## Парное сравнение")[1].split("## Вектор категорий")[0]
    row = next(line for line in section.splitlines() if line.startswith("| правая |"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[1] == "-5.00"       # парная разность, не 47.5 − 100
    assert cells[2] == "-5.00 … -5.00"
    assert cells[3] == "47.50" and cells[4] == "100.00"   # предельные доли


def test_one_shared_task_per_family_is_not_significant():
    """Four strata of one unit redraw the same units and collapse the interval."""
    units = [(f"family-{index}", 10.0, 10.0, 5.0, 10.0) for index in range(4)]
    result = stats.paired_ratio_difference(units, seed=4)
    assert result["smallest_stratum"] == 1
    assert result["ci95"][0] == result["ci95"][1]
    assert result["significant"] is False


def test_strata_are_families_not_categories():
    """A and F share the det/logic family and are one resampling stratum."""
    left = summarise(_scores("A", [10.0] * 4) + _scores("F", [10.0] * 4))
    right = summarise(_scores("A", [5.0] * 4) + _scores("F", [5.0] * 4))
    result = compare_summaries(left, right)
    assert result["sample_plan"]["strata"] == 1
    assert result["sample_plan"]["measured"] == ["task_id", "family"]


def test_an_unmeasured_category_is_not_a_free_pass():
    """Weights are normalised over the whole profile, not over what was measured."""
    left = summarise(_scores("A", [10.0] * 4))
    right = summarise(_scores("A", [5.0] * 4) + _scores("B", [15.0] * 4, possible=15.0))
    published = compare_summaries(left, right)["weight_sensitivity"]["schemes"]["published"]
    # A is 10 of the 100 profile points: 10 %, not 100 % of a re-normalised set.
    assert published["left"] == pytest.approx(10.0)
    assert published["right"] == pytest.approx(20.0)
    assert published["unknown"] == pytest.approx(90.0)
    assert published["winner"] == "right"


def test_a_non_rankable_run_keeps_its_interval_but_loses_the_verdict():
    left = summarise(_scores("A", [10.0] * 20))
    scores = _scores("A", [0.0] * 20)
    scores[0]["error"] = "обрыв соединения"
    scores[0]["failure"] = "provider_unavailable"
    right = summarise(scores)
    result = compare_summaries(left, right)
    assert result["overall"]["pairs"] == 20
    # Полоса пары сильная, но вердикт снят: прогон не ранжируется.
    low, high = result["overall"]["delta_ci95"]
    assert low > 0
    assert result["overall"]["ranked"] is False
    assert result["overall"]["significant"] is False
    assert result["overall"]["unranked_reason"] == "инфраструктура"

    gated = compare_summaries(left, summarise(_scores("A", [0.0] * 20)), same_contract=False)
    assert gated["overall"]["ranked"] is False
    assert gated["overall"]["unranked_reason"] == "другой контракт сьюта"
