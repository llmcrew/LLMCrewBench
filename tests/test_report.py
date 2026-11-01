"""Ranking contract: suite bytes and generation mode stay inseparable."""

from __future__ import annotations

from dailybench.report import contract_key, rankable, render_markdown, summarise, unranked_reason


def test_fixed_and_native_runs_are_not_ranked_together():
    fixed = {"suite": {"sha256": "same-suite"}, "config": {"generation_mode": "fixed"}}
    native = {"suite": {"sha256": "same-suite"}, "config": {"generation_mode": "native"}}

    assert contract_key(fixed) == "same-suite:fixed"
    assert contract_key(native) == "same-suite:native"
    assert contract_key(fixed) != contract_key(native)


def test_run_without_generation_mode_is_unranked():
    scores = [{"task_id": "A-1", "category": "A", "earned": 1.0, "possible": 1.0, "checks": []}]
    suite = {
        "sha256": "legacy-suite",
        "per_task": [{"id": "A-1", "category": "A", "max_points": 1.0}],
    }
    run = {"config": {"model": "legacy"}, "suite": suite, "scores": scores}
    summary = summarise(scores, suite, None)

    assert rankable(summary) is False
    assert unranked_reason(summary) == "режим генерации не записан"
    assert "режим генерации не записан" in summary["verdict"]
    assert "режим генерации не записан" in render_markdown(run)


def test_speed_table_names_sizing_and_where_vram_was_measured():
    run = {
        "dir": ".",
        "config": {"model": "m"},
        "scores": [],
        "speed": [{
            "label": "long_32",
            "ttft_s": 0.4,
            "prompt_tok_s": 100.0,
            "generation_tok_s": 40.0,
            "vram_peak_mib": 512.0,
            "runs": 1,
            "sized": "estimate",
            "measured_on": "harness_host",
        }],
    }
    text = render_markdown(run)
    assert "sized" in text
    assert "measured_on" in text
    assert "estimate" in text
    assert "harness_host" in text

