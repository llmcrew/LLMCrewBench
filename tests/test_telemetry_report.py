"""Report-side telemetry: join, unknown shares, guarded cost, section branches."""

from __future__ import annotations

import pytest

from dailybench import telemetry
from dailybench.report import _telemetry_section, render_markdown


def _entry(task_id: str, **overrides) -> dict:
    row = {
        "task_id": task_id,
        "category": "A",
        "answer_tokens": 100,
        "reasoning_tokens": 10,
        "wall_s": 1.0,
        "context_occupancy": 0.1,
        "ttft_s": 0.2,
        "decode_tokens_per_s": 50.0,
        "budgeted": False,
        "usage_known": True,
    }
    row.update(overrides)
    return row


def _score(task_id: str, earned: float = 10.0, *, possible: float = 10.0,
           category: str = "A", skipped: bool = False) -> dict:
    return {
        "task_id": task_id,
        "category": category,
        "earned": earned,
        "possible": possible,
        "skipped": skipped,
        "checks": [],
    }


def _section(entries, scores, **summary) -> str:
    run = {"dir": ".", "config": {}, "scores": scores, "telemetry": entries}
    base = {"incomplete": False, "eligible": True}
    base.update(summary)
    return "\n".join(_telemetry_section(run, base))


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #


def test_a_score_without_telemetry_stays_a_row_with_unknown_metrics():
    rows = telemetry.rows([], [_score("T-1", 4.0), _score("T-2", 6.0)])
    assert [row["task_id"] for row in rows] == ["T-1", "T-2"]
    for row in rows:
        assert row["measured"] is False
        for metric in ("spend_tokens", "wall_s", "context_occupancy", "ttft_s",
                       "decode_tokens_per_s", "budgeted", "usage_known"):
            assert row[metric] is None
        assert row["share"] == 40.0 if row["task_id"] == "T-1" else True


def test_rows_pair_by_task_id_and_ignore_stray_entries():
    rows = telemetry.rows(
        [_entry("T-2", answer_tokens=5, reasoning_tokens=1), _entry("T-9")],
        [_score("T-1"), _score("T-2")],
    )
    assert [row["measured"] for row in rows] == [False, True]
    assert rows[1]["spend_tokens"] == 6
    assert rows[0]["spend_tokens"] is None


def test_spend_is_unknown_when_either_side_is():
    rows = telemetry.rows(
        [
            _entry("T-1", answer_tokens=None),
            _entry("T-2", reasoning_tokens=None),
            _entry("T-3", answer_tokens=0, reasoning_tokens=0),
        ],
        [_score("T-1"), _score("T-2"), _score("T-3")],
    )
    assert [row["spend_tokens"] for row in rows] == [None, None, 0]


def test_rows_keep_the_skipped_flag_and_fall_back_to_the_score_category():
    rows = telemetry.rows(
        [_entry("T-1", category="B")],
        [_score("T-1", category="B", skipped=True)],
    )
    assert rows[0]["skipped"] is True
    assert rows[0]["category"] == "B"


# --------------------------------------------------------------------------- #
# unknown_shares
# --------------------------------------------------------------------------- #


def test_unknown_shares_skip_skipped_rows_and_scope_to_the_category():
    rows = [
        {"category": "A", "spend_tokens": None, "wall_s": 1.0, "context_occupancy": None},
        {"category": "A", "spend_tokens": 10, "wall_s": 2.0, "context_occupancy": 0.1},
        {"category": "B", "spend_tokens": 10, "wall_s": 3.0, "context_occupancy": 0.2},
        {"category": "A", "spend_tokens": 99, "wall_s": None, "context_occupancy": None,
         "skipped": True},
    ]
    shares = telemetry.unknown_shares(rows)

    assert set(shares) == {"A", "B"}
    assert shares["A"]["spend_tokens"] == {"unknown": 1, "total": 2, "share": 0.5}
    assert shares["A"]["wall_s"] == {"unknown": 0, "total": 2, "share": 0.0}
    assert shares["B"]["spend_tokens"]["unknown"] == 0


def test_unknown_shares_of_only_skipped_rows_is_empty():
    assert telemetry.unknown_shares([{"category": "A", "skipped": True}]) == {}


# --------------------------------------------------------------------------- #
# cost_per_point
# --------------------------------------------------------------------------- #


def test_cost_per_point_needs_completeness_even_when_everything_is_measured():
    rows = [{"earned": 5.0, "possible": 10.0, "spend_tokens": 100}]
    assert telemetry.cost_per_point(rows, complete=True)["tokens_per_point"] == 20.0
    assert telemetry.cost_per_point(rows, complete=False) is None


def test_cost_per_point_excludes_skips_and_guards_the_denominators():
    rows = [
        {"earned": 5.0, "possible": 10.0, "spend_tokens": 100},
        {"earned": 99.0, "possible": 0.0, "spend_tokens": 1000, "skipped": True},
    ]
    cost = telemetry.cost_per_point(rows, complete=True)
    assert cost == {"tokens": 100, "earned": 5.0, "tokens_per_point": 20.0}

    assert telemetry.cost_per_point(
        [{"earned": 0.0, "possible": 10.0, "spend_tokens": 100}], complete=True
    ) is None
    assert telemetry.cost_per_point(
        [{"earned": 5.0, "possible": 10.0, "spend_tokens": 0}], complete=True
    ) is None
    assert telemetry.cost_per_point(
        [{"earned": 5.0, "possible": 0.0, "spend_tokens": 10}], complete=True
    ) is None
    assert telemetry.cost_per_point([], complete=True) is None


# --------------------------------------------------------------------------- #
# summary_index
# --------------------------------------------------------------------------- #


def test_summary_index_medians_are_even_and_odd():
    rows = [
        {"category": "A", "ttft_s": 1.0, "wall_s": 1.0, "decode_tokens_per_s": 1.0},
        {"category": "A", "ttft_s": 2.0, "wall_s": 2.0, "decode_tokens_per_s": 2.0},
        {"category": "A", "ttft_s": 3.0, "wall_s": 3.0, "decode_tokens_per_s": 3.0},
        {"category": "B", "ttft_s": 1.0, "wall_s": 1.0, "decode_tokens_per_s": 1.0},
        {"category": "B", "ttft_s": 2.0, "wall_s": 2.0, "decode_tokens_per_s": 2.0},
        {"category": "B", "ttft_s": 3.0, "wall_s": 3.0, "decode_tokens_per_s": 3.0},
        {"category": "B", "ttft_s": 4.0, "wall_s": 4.0, "decode_tokens_per_s": 4.0},
        {"category": "C", "skipped": True, "ttft_s": 9.0},
    ]
    index = telemetry.summary_index(rows)

    assert [row["category"] for row in index] == ["A", "B"]
    by_category = {row["category"]: row for row in index}
    assert by_category["A"]["ttft_s"] == 2.0
    assert by_category["A"]["n"] == 3
    assert by_category["B"]["ttft_s"] == 2.5
    assert by_category["B"]["n"] == 4
    assert all("C" not in row["category"] for row in index)


def test_summary_index_counts_unknown_spend_and_keeps_none_medians():
    index = telemetry.summary_index(
        [{"category": "A", "spend_tokens": None, "ttft_s": None},
         {"category": "A", "spend_tokens": 5}]
    )
    assert index[0]["unknown_spend"] == 1
    assert index[0]["ttft_s"] is None
    assert index[0]["decode_tokens_per_s"] is None


# --------------------------------------------------------------------------- #
# spearman
# --------------------------------------------------------------------------- #


def test_spearman_edges():
    assert telemetry.spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert telemetry.spearman([1, 2, 3], [3, 2, 1]) == -1.0
    # Fewer than three pairs is never a correlation.
    assert telemetry.spearman([1, 2], [2, 1]) is None
    assert telemetry.spearman([1, 2, 3, 4], [1, 2, 3]) is None
    # A constant side has no rank spread.
    assert telemetry.spearman([1, 2, 3], [5, 5, 5]) is None


def test_spearman_averages_tied_ranks():
    # x ranks become 1, 2.5, 2.5, 4 — an explicit value, not a dropped pair.
    assert telemetry.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9487)


# --------------------------------------------------------------------------- #
# _telemetry_section
# --------------------------------------------------------------------------- #


def test_the_section_says_telemetry_was_not_collected():
    text = _section([], [_score("T-1")])
    assert "Телеметрия не собрана" in text
    assert "Стоимость балла: не рассчитана — нет расхода токенов." in text
    assert "telemetry-index" not in text


def test_the_section_counts_budget_modes_and_renders_the_index():
    scores = [_score(f"T-{index}") for index in range(1, 4)]
    entries = [
        _entry("T-1", budgeted=False),
        _entry("T-2", budgeted=True),
        _entry("T-3", budgeted=None),
    ]
    text = _section(entries, scores)

    assert "budgeted 1, unbounded 1, неизвестно 1" in text
    assert "class='viz telemetry-index'" in text
    assert "| Категория | Задач | Медиана TTFT, с | Медиана cache, % |" in text
    assert "Корреляция Спирмена" in text
    assert "| A | расчёт токенов" in text.replace("расход токенов", "расчёт токенов")


def test_the_section_marks_unknown_metrics_in_the_shares_table():
    scores = [_score(f"T-{index}") for index in range(1, 4)]
    entries = [_entry("T-1"), {"task_id": "T-2", "category": "A"}, _entry("T-3")]
    text = _section(entries, scores)

    assert "33 % (1/3)" in text
    assert "| A | расход токенов | — | 2 | 1 |" in text


def test_the_section_reports_the_cost_only_for_a_complete_measurement():
    scores = [_score("T-1", 4.0), _score("T-2", 6.0)]
    entries = [_entry("T-1", answer_tokens=100, reasoning_tokens=0),
               _entry("T-2", answer_tokens=100, reasoning_tokens=0)]
    text = _section(entries, scores)

    assert "измерение полное" in text
    assert "20.0 токенов на балл" in text


def test_the_section_names_an_unknown_spend_as_the_cost_reason():
    text = _section(
        [_entry("T-1"), {"task_id": "T-2", "category": "A"}],
        [_score("T-1"), _score("T-2")],
    )
    assert "не рассчитана — есть неизвестный расход" in text


def test_the_section_names_incompleteness_as_the_cost_reason():
    text = _section(
        [_entry("T-1")],
        [_score("T-1")],
        incomplete=True,
    )
    assert "не рассчитана — прогон неполон или не ранжируется" in text


def test_the_section_renders_through_the_report_and_flags_non_finite_values():
    run = {
        "dir": ".",
        "config": {},
        "scores": [_score(f"T-{index}") for index in range(1, 4)],
        "telemetry": [
            _entry("T-1", ttft_s=None, wall_s=None, decode_tokens_per_s=None),
            _entry("T-2", ttft_s=None, wall_s=None, decode_tokens_per_s=None),
            _entry("T-3", ttft_s=None, wall_s=None, decode_tokens_per_s=None),
        ],
    }
    markdown = render_markdown(run)

    assert "Телеметрия не собрана" not in markdown
    assert "| A | 3 | n/a |" in markdown
    assert "n/a | n/a |" in markdown
    assert "NaN" not in markdown


def test_the_section_formats_non_finite_metrics_as_unknown():
    text = _section(
        [_entry("T-1", ttft_s=float("nan"), wall_s=float("nan"),
                decode_tokens_per_s=float("nan"))],
        [_score("T-1")],
    )
    assert "| A | 1 | n/a |" in text
    assert "NaN" not in text
