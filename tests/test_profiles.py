"""Suite-profile contract: ids, weights, admission tri-state and denominators.

A profile is a frozen suite version. Its task list and weights are part of the
measurement, so both are asserted here: the baseline stays at the published 47
scenarios and 100 A–J points, ``stretch`` adds scenarios without changing the
100-point total, and a filtered or admission-failed run may never quietly
change what the model is measured against.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dailybench import cli, core
from dailybench.config import RunConfig
from dailybench.providers import ScriptedProvider
from dailybench.selfcheck import GarbageProvider, OracleProvider, selfcheck
from dailybench.report import summarise
from dailybench.runner import Runner
from dailybench.suite import build_suite, validate_suite

BASELINE_IDS = [
    "IF-1", "IF-2", "IF-3", "IF-4", "IF-5",
    *[f"R-{index}" for index in range(1, 16)],
    "C-1", "C-2", "C-3",
    *[f"T-{index}" for index in range(1, 8)],
    *[f"LC-{index}" for index in range(1, 6)],
    "F",
    *[f"G-{index}" for index in range(1, 5)],
    *[f"H-{index}" for index in range(1, 4)],
    "I-1", "I-2",
    "J-1", "J-2",
    *[f"K-{index}" for index in range(1, 21)],
    *[f"L-{index}" for index in range(1, 30)],
]

#: The pilot wave: one host scenario per category, the three cheap K hosts and
#: the untouched baseline art block.
PILOT_IDS = [
    "IF-6", "R-16", "C-4", "T-8", "LC-6", "F-2", "G-5", "H-4", "I-3", "J-3",
    "K-21", "K-23", "K-25",
    *[f"L-{index}" for index in range(1, 30)],
]

#: Stretch: every baseline scenario plus the whole admitted pilot wave and the
#: language/edge-case stratum added by the corpus wave.
STRETCH_IDS = [
    *[f"IF-{index}" for index in range(1, 7)],
    "IF-7", "IF-7-KK", "IF-7-EN", "IF-8", "IF-9", "IF-10",
    *[f"R-{index}" for index in range(1, 17)],
    "R-17", "R-17-KK", "R-18",
    *[f"C-{index}" for index in range(1, 7)], "N-2", "C-7", "N-9",
    *[f"T-{index}" for index in range(1, 12)],
    *[f"LC-{index}" for index in range(1, 9)],
    "F", "F-2", "F-3",
    *[f"G-{index}" for index in range(1, 6)],
    *[f"H-{index}" for index in range(1, 6)],
    *[f"I-{index}" for index in range(1, 6)],
    *[f"J-{index}" for index in range(1, 6)],
    *[f"K-{index}" for index in range(1, 37)],
    *[f"L-{index}" for index in range(1, 40)],
]

TOLERANCE = 1e-3


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= TOLERANCE


def _scored(tasks: list[core.TaskInstance]) -> list[core.TaskInstance]:
    return [task for task in tasks if task.category in core.SCORED_CATEGORIES]


def test_baseline_is_the_frozen_published_suite(suite_for):
    tasks = suite_for(core.DEFAULT_PROFILE)
    assert [task.id for task in tasks] == BASELINE_IDS
    assert len(_scored(tasks)) == 47
    assert sum(task.max_points for task in _scored(tasks)) == core.FULL_TOTAL == 100.0
    assert sum(task.max_points for task in _scored(tasks) if task.category != "E") == core.SHORT_TOTAL == 90.0
    assert core.SHOWCASE_POINTS[core.DEFAULT_PROFILE] == {"K": 80.0, "L": 29.0}
    assert sum(task.max_points for task in tasks if task.category == "K") == core.points_for("K") == 80.0
    assert sum(task.max_points for task in tasks if task.category == "L") == core.points_for("L") == 29.0


def test_pilot_and_stretch_expect_the_declared_ids(suite_for):
    """A module override is profile-specific, and unregistered profiles fall back."""
    module = core.REGISTRY["A"]
    assert module.expected_ids_for() == module.expected_ids
    assert module.expected_ids_for("stretch-v1") == (
        *module.expected_ids, "IF-6", "IF-7", "IF-7-KK", "IF-7-EN", "IF-8", "IF-9", "IF-10"
    )
    assert module.expected_ids_for("pilot") == ("IF-6",)
    assert core.REGISTRY["F"].expected_ids_for("pilot") == ("F-2",)
    assert core.REGISTRY["F"].expected_ids_for("stretch-v1") == ("F", "F-2", "F-3")
    assert core.REGISTRY["K"].expected_ids_for("pilot") == ("K-21", "K-23", "K-25")

    # Every profile is a frozen id list: no extra scenario may slip in and no
    # declared scenario may be dropped.
    assert [task.id for task in suite_for(core.DEFAULT_PROFILE)] == BASELINE_IDS
    assert [task.id for task in suite_for("pilot")] == PILOT_IDS
    assert [task.id for task in suite_for("stretch-v1")] == STRETCH_IDS


def test_profile_category_totals_are_fixed(suite_for):
    """Each profile declares its own exact category totals; none inherit by accident."""
    totals = {
        core.DEFAULT_PROFILE: {"A": 10.0, "B": 15.0, "C": 20.0, "D": 14.0, "E": 10.0,
                               "F": 8.0, "G": 8.0, "H": 6.0, "I": 4.0, "J": 5.0,
                               "K": 80.0, "L": 29.0},
        "stretch-v1": {**core.CATEGORY_POINTS, "K": 144.0, "L": 39.0},
    }
    for profile, expected in totals.items():
        tasks = suite_for(profile)
        for key, total in expected.items():
            group = [task for task in tasks if task.category == key]
            assert _close(sum(task.max_points for task in group), total), (profile, key)
            assert _close(core.points_for(key, profile), total), (profile, key)
        assert all(task.max_points > 0 for task in tasks)
    assert core.SHOWCASE_POINTS["pilot"] == {"K": 12.0, "L": 29.0}
    assert core.SHOWCASE_POINTS[core.DEFAULT_PROFILE] == {"K": 80.0, "L": 29.0}
    assert core.SHOWCASE_POINTS["stretch-v1"] == {"K": 144.0, "L": 39.0}


def test_pilot_keeps_module_weights(suite_for):
    tasks = suite_for("pilot")
    expected = {"A": 2.0, "B": 1.0, "C": 8.0, "D": 2.0, "E": 2.0,
                "F": 8.0, "G": 2.0, "H": 4.0, "I": 2.0, "J": 2.0}
    assert next(task for task in tasks if task.id == "IF-6").max_points == 2.0
    for key, total in expected.items():
        group = [task for task in tasks if task.category == key]
        assert sum(task.max_points for task in group) == total
        assert core.points_for(key, "pilot") == total
    module_total = sum(task.max_points for task in _scored(tasks))
    assert module_total == 33.0
    assert core.scored_total("pilot") == module_total


def test_pilot_report_says_not_rankable_with_baseline(suite_for):
    from dailybench.suite import suite_digest

    tasks = suite_for("pilot")
    scores = [
        {"task_id": task.id, "category": task.category, "earned": task.max_points,
         "possible": task.max_points, "admission": True, "checks": []}
        for task in tasks
    ]
    digest = suite_digest(tasks, "pilot")
    summary = summarise(scores, digest)
    assert summary["rankable_with_baseline"] is False
    assert "не ранжируется с baseline" in summary["verdict"]


def test_stretch_redistributes_within_each_category_point_budget(suite_for):
    tasks = suite_for("stretch-v1")
    assert _close(sum(task.max_points for task in _scored(tasks)), core.FULL_TOTAL)
    for key, total in core.CATEGORY_POINTS.items():
        group = [task for task in tasks if task.category == key]
        assert _close(sum(task.max_points for task in group), total)
        # Even shares with the tail on the last task, never a variable scale.
        assert max(task.max_points for task in group) - min(task.max_points for task in group) <= TOLERANCE
    showcase = {key: sum(t.max_points for t in tasks if t.category == key) for key in core.VISUAL_CATEGORIES}
    assert _close(showcase["K"], 144.0) and _close(showcase["L"], 39.0)
    assert core.points_for("K", "stretch-v1") == 144.0
    assert core.points_for("L", "stretch-v1") == 39.0
    assert validate_suite(tasks, "stretch-v1") == []


def test_a_filtered_slice_keeps_the_profile_weight(suite_for):
    complete = {
        (profile, task.id): task.max_points
        for profile in ("pilot", "stretch-v1")
        for task in suite_for(profile)
    }
    for profile, task_id in (("pilot", "IF-6"), ("stretch-v1", "IF-1")):
        tasks = build_suite(replace(RunConfig(profile=profile), only=(task_id,)))
        assert [(task.id, task.max_points) for task in tasks] == [
            (task_id, complete[(profile, task_id)])
        ]
    # The pilot keeps the weight declared by the task module.
    pilot = build_suite(replace(RunConfig(profile="pilot"), only=("IF-6",)))
    assert pilot[0].max_points == 2.0
    # A whole filtered category keeps the same weight as in a complete run.
    tasks = build_suite(replace(RunConfig(profile="stretch-v1"), only=("A",)))
    assert _close(sum(task.max_points for task in tasks), core.points_for("A", "stretch-v1"))


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        RunConfig(profile="no-such-profile")
    with pytest.raises(ValueError):
        core.profile_of(type("Cfg", (), {"profile": "nope"})())
    assert core.profile_of(RunConfig()) == core.DEFAULT_PROFILE


def _t4():
    import random

    from dailybench.tasks import tools

    return next(task for task in tools.generate(random.Random(4), RunConfig()) if task.id == "T-4")


def test_possible_stays_at_max_points_for_any_answer():
    """The consumer invariant: the denominator never depends on the answer (R03)."""
    task = _t4()
    assert task.max_points == 2.0

    oracle = task.oracle
    answers = [
        [_bulk(task.oracle["golden_calls"]), core.ModelReply(text=oracle["final_answer"])],
        [core.ModelReply(text="не знаю")],
        [_delete(oracle["protected"][0])],
    ]
    for replies in answers:
        run = Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)
        assert run.score.possible == task.max_points == 2.0
        assert sum(check.points for check in run.score.checks) == task.max_points
        assert run.score.as_dict()["max_points"] == 2.0


def test_checks_are_fitted_into_a_redistributed_budget():
    score = core.Score(task_id="K-1", category="K")
    for name in ("a", "b", "c", "d"):
        score.add(name, True, 1.0)
    score.max_points = 6.0
    score.fit_points()
    assert sum(check.points for check in score.checks) == 6.0
    assert score.earned == 6.0
    assert all(check.points == 1.5 for check in score.checks)


def test_a_skipped_task_keeps_zero_possible_but_reports_its_budget(suite_for):
    task = next(task for task in suite_for(core.DEFAULT_PROFILE) if task.id == "IF-1")
    run = Runner(RunConfig(unsupported=("A",)), GarbageProvider()).run_task(task)
    assert run.score.status == "skipped"
    assert run.score.possible == 0.0
    assert run.score.max_points == task.max_points


def test_unknown_admission_is_unverified_not_passed():
    scores = [
        {"task_id": "IF-1", "category": "A", "earned": 10.0, "possible": 10.0, "admission": None},
        {"task_id": "LC-1", "category": "E", "earned": 2.0, "possible": 2.0, "admission": None},
    ]
    summary = summarise(scores)
    assert summary["admission_state"] == "unverified"
    assert summary["long_context_admitted"] is False
    assert summary["headline_possible"] == 12.0, "неизвестный допуск не сокращает знаменатель"
    assert "допуск не подтверждён" in summary["verdict"]

    failed = summarise([scores[0], {**scores[1], "admission": False}])
    assert failed["admission_state"] == "failed"
    assert failed["headline_possible"] == 10.0
    assert "не прошёл допуск" in failed["verdict"]

    passed = summarise([scores[0], {**scores[1], "admission": True}])
    assert passed["admission_state"] == "passed"
    assert "допуск" not in passed["verdict"]


def test_selfcheck_shows_unverified_long_context_without_usage():
    report = selfcheck(RunConfig(only=("E",)))
    assert report.ok, report.render()
    rendered = report.render()
    assert "unverified" in rendered
    assert "ПРОБЛЕМА" not in rendered


def test_rescore_keeps_a_skipped_task_skipped(tmp_path: Path):
    from dailybench.rescore import rescore_run
    from dailybench.runner import RunResult

    cfg = RunConfig(out_dir=tmp_path, only=("IF-1",), unsupported=("A",))
    task = next(task for task in build_suite(cfg) if task.id == "IF-1")
    run = Runner(cfg, GarbageProvider()).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "garbage"}).save()
    before = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))[0]
    assert before["status"] == "skipped" and before["possible"] == 0.0

    marker = rescore_run(tmp_path)
    after = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))[0]
    assert after["status"] == "skipped"
    assert after["skipped"]
    assert after["possible"] == 0.0
    assert after["max_points"] == 2.0
    assert marker["changed"] == []


def test_rescore_keeps_scores_when_a_legacy_suite_drifted(tmp_path: Path):
    """A legacy suite.json has no per-task rows; its whole hash still proves drift.

    Without the per-task fingerprints the old code replayed answers against
    whatever the current generators produce and silently re-scored a fixture
    the model never saw.
    """
    from dailybench.rescore import rescore_run
    from dailybench.runner import RunResult

    cfg = RunConfig(out_dir=tmp_path, only=("IF-1",))
    task = next(task for task in build_suite(cfg) if task.id == "IF-1")
    run = Runner(cfg, GarbageProvider()).run_task(task)
    RunResult(cfg=cfg, runs=[run], provider_info={"provider": "garbage"}).save()
    before = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))[0]
    suite = json.loads((tmp_path / "suite.json").read_text(encoding="utf-8"))
    suite.pop("per_task")
    suite["sha256"] = "0" * 64
    (tmp_path / "suite.json").write_text(json.dumps(suite, ensure_ascii=False), encoding="utf-8")

    marker = rescore_run(tmp_path)
    after = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))[0]
    assert after["earned"] == before["earned"] and after["possible"] == before["possible"]
    assert marker["changed"] == [
        {"task_id": "IF-1", "status": "фикстура разошлась (sha сьюта)"}
    ]


def test_cli_records_the_profile_in_the_run(tmp_path: Path):
    out = tmp_path / "suite"
    assert cli.main(["generate", "--only", "A", "--profile", "stretch-v1", "--out", str(out)]) == 0
    payload = json.loads((out / "suite.json").read_text(encoding="utf-8"))
    assert payload["profile"] == "stretch-v1"
    prompts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    assert sum(entry["max_points"] for entry in prompts) == core.CATEGORY_POINTS["A"]


def _bulk(calls: list[dict]) -> core.ModelReply:
    return core.ModelReply(
        text="",
        tool_calls=[
            {"id": f"c{index}", "name": call["name"], "arguments": json.dumps(call["arguments"])}
            for index, call in enumerate(calls, 1)
        ],
    )


def _delete(path: str) -> core.ModelReply:
    return core.ModelReply(
        text="",
        tool_calls=[{"id": "d1", "name": "delete_file", "arguments": json.dumps({"path": path})}],
    )


def _visual_fixture(task: core.TaskInstance) -> dict:
    """A K task without its closures: ``spec`` holds freshly bound checks."""
    import hashlib

    oracle = {key: value for key, value in task.oracle.items() if key not in ("spec", "golden_html")}
    return {
        "prompt": task.prompt,
        "oracle": oracle,
        "golden": hashlib.sha256(task.oracle["golden_html"].encode("utf-8")).hexdigest(),
    }


def test_visual_scenarios_do_not_depend_on_the_module_order() -> None:
    """A scenario module owns its stream: reordering modules changes no byte."""
    from dailybench.rng import task_rng
    from dailybench.tasks import visual

    cfg = RunConfig(profile="stretch-v1")
    plain = {task.id: task for task in visual.generate(task_rng(31, "K"), cfg)}
    modules = tuple(reversed(visual._MODULES))
    original = visual._MODULES
    visual._MODULES = modules
    try:
        reordered = {task.id: task for task in visual.generate(task_rng(31, "K"), cfg)}
    finally:
        visual._MODULES = original
    assert list(plain) == list(reordered)
    for task_id, task in plain.items():
        assert _visual_fixture(task) == _visual_fixture(reordered[task_id]), task_id


def test_a_visual_scenario_is_the_same_in_every_profile() -> None:
    """The pilot hosts K-21, K-23, K-25; the stretch suite must not move them."""
    from dailybench.rng import task_rng
    from dailybench.tasks import visual

    seed = 8
    pilot = {task.id: task for task in visual.generate(task_rng(seed, "K"), RunConfig(profile="pilot"))}
    stretch = {
        task.id: task
        for task in visual.generate(task_rng(seed, "K"), RunConfig(profile="stretch-v1"))
    }
    for task_id, task in pilot.items():
        assert _visual_fixture(stretch[task_id]) == _visual_fixture(task), task_id


#: The research profile: ten admitted scenarios across seven categories plus six
#: image-input showcase scenarios, no reasoning/structured/dialogue block,
#: no K/L block, denominator 71.
RESEARCH_IDS = [
    "IF-9", "IF-10", "N-2", "T-10", "T-11", "T-12", "LC-8", "F-3", "I-5", "J-5",
    "V-1", "V-2", "V-3", "V-4", "V-5", "V-6",
]

RESEARCH_EXPECTED_IDS: dict[str, tuple[str, ...]] = {
    "A": ("IF-9", "IF-10"),
    "C": ("N-2",),
    "D": ("T-10", "T-11", "T-12"),
    "E": ("LC-8",),
    "F": ("F-3",),
    "I": ("I-5",),
    "J": ("J-5",),
    "B": (),
    "G": (),
    "H": (),
    "K": (),
    "L": (),
    "V": ("V-1", "V-2", "V-3", "V-4", "V-5", "V-6"),
}


def test_research_profile_is_the_narrow_research_set(suite_for):
    tasks = suite_for("research-v1")
    assert [task.id for task in tasks] == RESEARCH_IDS
    assert {task.category for task in tasks} == {"A", "C", "D", "E", "F", "I", "J", "V"}
    # 71 scored points plus the 6-point VQA showcase outside the verdict.
    assert sum(task.max_points for task in tasks) == core.profile_total("research-v1") == 77.0
    assert core.scored_total("research-v1") == 71.0
    for key, total in core.PROFILE_POINTS["research-v1"].items():
        group = [task for task in tasks if task.category == key]
        assert _close(sum(task.max_points for task in group), total), key
    assert core.SHOWCASE_POINTS["research-v1"] == {"V": 6.0}
    assert [task.id for task in tasks if task.category in core.VISUAL_CATEGORIES] == [
        "V-1", "V-2", "V-3", "V-4", "V-5", "V-6"
    ]
    # A category the profile does not carry has no total at all, not a baseline one.
    assert core.points_for("B", "research-v1") is None
    assert core.points_for("K", "research-v1") is None
    assert validate_suite(tasks, "research-v1") == []


def test_every_module_declares_the_research_id_set():
    for key, ids in RESEARCH_EXPECTED_IDS.items():
        assert core.REGISTRY[key].expected_ids_for("research-v1") == ids, key
        assert core.REGISTRY[key].expected_ids_for() == core.REGISTRY[key].expected_ids, key


def test_research_profile_leaves_the_other_profiles_alone(suite_for):
    assert [task.id for task in suite_for(core.DEFAULT_PROFILE)] == BASELINE_IDS
    assert [task.id for task in suite_for("pilot")] == PILOT_IDS
    assert [task.id for task in suite_for("stretch-v1")] == STRETCH_IDS
    assert _close(sum(task.max_points for task in _scored(suite_for(core.DEFAULT_PROFILE))), 100.0)
    assert _close(core.profile_total(), 100.0 + 80.0 + 29.0)
    assert _close(core.profile_total("pilot"), 33.0 + 12.0 + 29.0)
    assert _close(core.profile_total("stretch-v1"), 100.0 + 144.0 + 39.0)
    assert _close(
        sum(task.max_points for task in suite_for("pilot")),
        core.profile_total("pilot"),
    )
    assert _close(
        sum(task.max_points for task in suite_for("stretch-v1")),
        core.profile_total("stretch-v1"),
    )


def test_research_run_is_marked_unrankable_with_the_baseline(suite_for):
    from dailybench.suite import suite_digest

    tasks = suite_for("research-v1")
    scores = [
        {
            "task_id": task.id,
            "category": task.category,
            "earned": task.max_points,
            "possible": task.max_points,
            "admission": True,
        }
        for task in tasks
    ]
    summary = summarise(scores, suite_digest(tasks, "research-v1"))
    assert summary["profile"] == "research-v1"
    assert summary["rankable_with_baseline"] is False
    assert "не ранжируется с baseline" in summary["verdict"]
    assert summary["headline_possible"] == 71.0
    base = summarise(scores, suite_digest(suite_for(core.DEFAULT_PROFILE)))
    assert base["profile"] == core.DEFAULT_PROFILE
    assert base["rankable_with_baseline"] is True
    assert "не ранжируется с baseline" not in base["verdict"]

    # The rendered report carries the same mark, not only the summary field.
    from dailybench.report import render_markdown

    text = render_markdown({"config": {"model": "m", "profile": "research-v1"},
                            "suite": suite_digest(tasks, "research-v1"), "scores": scores})
    assert "не ранжируется с baseline" in text
    baseline_text = render_markdown(
        {"config": {"model": "m"}, "suite": suite_digest(suite_for(core.DEFAULT_PROFILE)), "scores": scores}
    )
    assert "не ранжируется с baseline" not in baseline_text


def test_research_sensitivity_uses_profile_categories(suite_for):
    from dailybench.report import compare_summaries
    from dailybench.suite import suite_digest

    tasks = suite_for("research-v1")
    scores = [
        {"task_id": task.id, "category": task.category, "earned": task.max_points,
         "possible": task.max_points, "admission": True, "checks": []}
        for task in tasks
    ]
    summary = summarise(scores, suite_digest(tasks, "research-v1"))
    result = compare_summaries(summary, summary)
    expected = {"A", "C", "D", "E", "F", "I", "J"}
    for scheme in result["weight_sensitivity"]["schemes"].values():
        assert set(scheme["weights"]) == expected
        assert scheme["unknown"] == 0.0


def test_cli_accepts_the_research_profile(tmp_path: Path):
    assert cli.main(["validate", "--profile", "research-v1"]) == 0
    out = tmp_path / "suite"
    assert cli.main(["generate", "--profile", "research-v1", "--out", str(out)]) == 0
    digest = json.loads((out / "suite.json").read_text(encoding="utf-8"))
    assert digest["profile"] == "research-v1"
    assert digest["profile_total"] == 77.0
    prompts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    # 71 scored points + the 6-point VQA showcase.
    assert sum(entry["max_points"] for entry in prompts) == 77.0
