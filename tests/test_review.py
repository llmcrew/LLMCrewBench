"""Category J (code review): registry contract, determinism and behavioural scoring.

The evaluator is only trusted if the fixture it scores is real: the hidden tests
must fail on the shipped file and pass on the corrected one, the reported line
must be the line that actually changes, and the three checks must stay independent — a
right line with a broken file, or a wrong line with a working file, each keep
exactly the points they earned, while a one-line contract broken by an appended
function costs the shape and the location but not the working repair.

The profile scenarios (J-3 pilot, J-4 stretch) are held to the same bar plus
their own oracle: J-3's allowed-row set must be exact with no foreign leak on
every dev seed, and J-4's stateful counter must prove one effect per key.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from typing import Any

import pytest

from dailybench import ModelReply, RunConfig, TaskInstance, Transcript, Turn, core
from dailybench.rng import task_rng
from dailybench.tasks import review
from dailybench.suite import apply_profile_weights

_SEED = 20260915
_DEV_SEEDS = (17, 18, 19, 20, 21)
_CACHE: dict[tuple[int, str], list[TaskInstance]] = {}


def _tasks(seed: int = _SEED) -> list[TaskInstance]:
    """Generated tasks, shared between tests (generation runs the hidden suite)."""
    if seed not in _CACHE:
        _CACHE[(seed, core.DEFAULT_PROFILE)] = review.generate(task_rng(seed, "J"), RunConfig())
    return _CACHE[(seed, core.DEFAULT_PROFILE)]


def _profile_tasks(profile: str, seed: int = _SEED) -> list[TaskInstance]:
    if (seed, profile) not in _CACHE:
        _CACHE[(seed, profile)] = review.generate(task_rng(seed, "J"), RunConfig(profile=profile))
    return _CACHE[(seed, profile)]


def _extra_tasks() -> list[TaskInstance]:
    """J-3 and J-4, generated once with their profile weights applied."""
    tasks = _profile_tasks("stretch-v1")
    apply_profile_weights(tasks, "stretch-v1")
    return [task for task in tasks if task.id in ("J-3", "J-4")]


def _transcript(task: TaskInstance, answer: str) -> Transcript:
    return Transcript(
        task_id=task.id,
        category="J",
        turns=[
            Turn(
                messages=[{"role": "user", "content": task.prompt}],
                reply=ModelReply(text=answer, completion_tokens=1),
            )
        ],
    )


def _score(task: TaskInstance, answer: str) -> core.Score:
    return review.evaluate(task, _transcript(task, answer))


def _check(score: core.Score, name: str) -> core.Check:
    return next(c for c in score.checks if c.name == name)


def test_review_ids_points_and_registry() -> None:
    tasks = _tasks()
    assert [task.id for task in tasks] == ["J-1", "J-2"]
    assert [task.max_points for task in tasks] == [3.0, 2.0]
    assert sum(task.max_points for task in tasks) == core.CATEGORY_POINTS["J"] == 5.0

    module = core.REGISTRY["J"]
    assert module.title == "Code review"
    assert module.points == 5.0
    assert module.expected_ids == review.IDS == ("J-1", "J-2")
    assert module.generate is review.generate and module.evaluate is review.evaluate

    # Profiles add one code-review situation each and never rewrite the baseline.
    assert module.expected_ids_for("pilot") == ("J-3",)
    assert module.expected_ids_for("stretch-v1") == ("J-1", "J-2", "J-3", "J-4")
    assert module.expected_ids_for("baseline-20261020") == ("J-1", "J-2")
    assert [task.id for task in _profile_tasks("pilot")] == ["J-3"]
    assert [task.id for task in _profile_tasks("stretch-v1")] == [
        "J-1",
        "J-2",
        "J-3",
        "J-4",
    ]


def test_review_fixtures_are_deterministic_and_seeded() -> None:
    first = review.generate(task_rng(17, "J"), RunConfig())
    second = review.generate(task_rng(17, "J"), RunConfig())
    other = review.generate(task_rng(18, "J"), RunConfig())

    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.workspace for task in first] == [task.workspace for task in second]
    assert [task.oracle for task in first] == [task.oracle for task in second]
    assert all(a.prompt != b.prompt for a, b in zip(first, other))
    assert all(a.workspace != b.workspace for a, b in zip(first, other))


def test_review_reported_line_is_the_only_changed_line() -> None:
    """The fixture's own claim: line N is the defect, and the fix touches N only."""
    for task in _tasks():
        buggy = task.workspace[task.oracle["target_file"]].splitlines()
        fixed = task.oracle["reference_source"].splitlines()
        line = task.oracle["line"]
        assert len(buggy) == len(fixed)
        assert [i for i, (a, b) in enumerate(zip(buggy, fixed), 1) if a != b] == [line]
        assert buggy[line - 1].strip() and buggy[line - 1] != fixed[line - 1]


def test_review_hidden_tests_fail_on_the_buggy_source_and_pass_on_the_fix() -> None:
    for task in _tasks():
        buggy = task.workspace[task.oracle["target_file"]]
        rows, note = review._run_test_modules(task, buggy, ("hidden",))
        assert not note, task.id
        hidden = rows["hidden"]
        assert hidden
        assert any(not ok for ok, _ in hidden.values()), f"{task.id}: дефектный файл прошёл скрытые тесты"

        rows, note = review._run_test_modules(task, task.oracle["reference_source"], ("hidden",))
        assert not note, task.id
        assert rows["hidden"] and all(ok for ok, _ in rows["hidden"].values()), task.id


def _spoofed(task: TaskInstance) -> TaskInstance:
    """Hidden module that fakes a perfect report on stdout and exits in atexit."""
    spoof = '''import atexit as _atexit
import json as _json
import os as _os

with open("/tmp/fake.jsonl", "w", encoding="utf-8") as _fake:
    for _label in ("hidden", "public"):
        _fake.write(_json.dumps({"module": _label, "test": _label + ".Forged.test_ok",
                                 "ok": True, "detail": ""}) + "\\n")
print("REPORT /tmp/fake.jsonl", flush=True)
_atexit.register(_os._exit, 0)

'''
    return replace(task, oracle={**task.oracle, "hidden_tests": spoof + task.oracle["hidden_tests"]})


def test_review_atexit_report_spoof_does_not_decide_the_verdict(tmp_path) -> None:
    """The last visible ``REPORT`` line is a forged one; the real run must win."""
    task = _tasks()[0]
    spoofed = _spoofed(task)

    rows, note = review._run_test_modules(spoofed, task.oracle["reference_source"], ("hidden",))
    assert not note
    hidden = rows["hidden"]
    assert hidden and all(ok for ok, _ in hidden.values())
    assert not any("Forged" in name for name in hidden), hidden

    # A real failure still beats the forged all-ok rows.
    buggy = spoofed.workspace[spoofed.oracle["target_file"]]
    rows, note = review._run_test_modules(spoofed, buggy, ("hidden",))
    assert not note
    assert any(not ok for ok, _ in rows["hidden"].values())


def test_review_a_module_that_never_runs_reports_every_test_as_failed() -> None:
    """A broken import shortens the report; every declared test must still fail."""
    task = _tasks()[0]
    broken = "raise ImportError('boom')\n" + task.oracle["hidden_tests"]
    patched = replace(task, oracle={**task.oracle, "hidden_tests": broken})
    rows, note = review._run_test_modules(patched, task.oracle["reference_source"], ("hidden",))
    assert not note
    tree = ast.parse(task.oracle["hidden_tests"])
    expected = sum(
        1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test")
    )
    assert len(rows["hidden"]) == expected
    assert all(not ok and "boom" in detail for ok, detail in rows["hidden"].values())



    for task in _tasks():
        good = _score(task, task.oracle["reference"])
        assert good.possible == task.max_points
        assert good.earned == task.max_points, [c.as_dict() for c in good.checks]
        assert [c.name for c in good.checks] == [
            "one_line_fix",
            "defect_located",
            "fix_passes_hidden_tests",
        ]
        assert not good.notes

        bad = _score(task, "не знаю")
        assert bad.earned == 0.0
        assert not any(c.passed for c in bad.checks)


def test_review_wrong_line_loses_only_the_location_check() -> None:
    for task in _tasks():
        payload: dict[str, Any] = json.loads(task.oracle["reference"])
        payload["line"] = task.oracle["line"] + 1
        score = _score(task, json.dumps(payload, ensure_ascii=False))

        assert score.earned == task.max_points - 1.0
        assert _check(score, "one_line_fix").passed
        assert not _check(score, "defect_located").passed
        assert _check(score, "fix_passes_hidden_tests").passed


def test_review_unfixed_file_scores_zero() -> None:
    """An unchanged file is neither a one-line edit nor a repair, so it earns nothing."""
    for task in _tasks():
        payload: dict[str, Any] = json.loads(task.oracle["reference"])
        payload["fix"] = task.workspace[task.oracle["target_file"]]
        score = _score(task, json.dumps(payload, ensure_ascii=False))

        assert score.earned == 0.0
        assert not _check(score, "one_line_fix").passed
        # The location is only paid inside a proper diff (R05).
        assert not _check(score, "defect_located").passed
        failed = _check(score, "fix_passes_hidden_tests")
        assert not failed.passed and "скрытых тестов" in failed.detail


def test_review_appended_function_fails_the_shape_but_not_the_tests() -> None:
    """``zip`` used to stop at the shorter file, so an appended function passed (R05)."""
    for task in _tasks():
        payload: dict[str, Any] = json.loads(task.oracle["reference"])
        payload["fix"] = task.oracle["reference_source"] + (
            "\n\ndef extra(values: list) -> list:\n    return values\n"
        )
        score = _score(task, json.dumps(payload, ensure_ascii=False))

        shape = _check(score, "one_line_fix")
        assert not shape.passed and "строк" in shape.detail
        assert not _check(score, "defect_located").passed
        assert _check(score, "fix_passes_hidden_tests").passed
        assert score.earned < task.max_points


def test_review_allows_moving_unchanged_lines() -> None:
    """Only the multiset of lines matters: a moved blank line is still one-line."""
    for task in _tasks():
        payload: dict[str, Any] = json.loads(task.oracle["reference"])
        lines = task.oracle["reference_source"].splitlines()
        assert lines[1] == ""
        lines.insert(2, lines.pop(1))
        payload["fix"] = "\n".join(lines)
        score = _score(task, json.dumps(payload, ensure_ascii=False))

        assert _check(score, "one_line_fix").passed
        assert score.earned == task.max_points


def test_review_accepts_a_fenced_fix_and_normalised_paths() -> None:
    """The fix is judged by behaviour, so a fenced block and a ``./`` path still score."""
    for task in _tasks():
        payload = {
            "file": f"./{task.oracle['target_file']}",
            "line": str(task.oracle["line"]),
            "fix": f"```python\n{task.oracle['reference_source']}```",
        }
        assert _score(task, json.dumps(payload, ensure_ascii=False)).earned == task.max_points


def test_review_broken_fix_scores_zero() -> None:
    """A syntax error in the returned file is a failed repair, not a crash."""
    for task in _tasks():
        payload = {
            "file": task.oracle["target_file"],
            "line": task.oracle["line"],
            "fix": "def broken(:\n",
        }
        score = _score(task, json.dumps(payload, ensure_ascii=False))
        assert score.earned == 0.0
        assert not _check(score, "one_line_fix").passed
        assert not _check(score, "defect_located").passed
        assert not _check(score, "fix_passes_hidden_tests").passed


# --------------------------------------------------------------------------- #
# Profile scenarios: J-3 (pilot) and J-4 (stretch)
# --------------------------------------------------------------------------- #


def _mutant(task: TaskInstance, fix: str) -> core.Score:
    payload: dict[str, Any] = json.loads(task.oracle["reference"])
    payload["fix"] = fix
    return _score(task, json.dumps(payload, ensure_ascii=False))


def test_review_profile_scenarios_score_reference_full_and_garbage_zero() -> None:
    for task in _extra_tasks():
        assert task.max_points == pytest.approx(1.25)  # 5 points spread over 4 tasks
        good = _score(task, task.oracle["reference"])
        assert good.possible == pytest.approx(task.max_points)
        assert good.earned == pytest.approx(task.max_points), [c.as_dict() for c in good.checks]
        assert sum(check.points for check in good.checks) == pytest.approx(task.max_points)
        assert [c.name for c in good.checks] == [
            "one_line_fix",
            "defect_located",
            "fix_passes_hidden_tests",
        ]

        bad = _score(task, "не знаю")
        assert bad.earned == 0.0
        assert not any(check.passed for check in bad.checks)


def test_review_profile_fixtures_are_deterministic_and_seeded() -> None:
    for profile, ids in (
        ("pilot", ("J-3",)),
        ("stretch-v1", ("J-1", "J-2", "J-3", "J-4")),
    ):
        first = review.generate(task_rng(17, "J"), RunConfig(profile=profile))
        second = review.generate(task_rng(17, "J"), RunConfig(profile=profile))
        other = review.generate(task_rng(18, "J"), RunConfig(profile=profile))

        assert [task.id for task in first] == list(ids)
        assert [task.prompt for task in first] == [task.prompt for task in second]
        assert [task.workspace for task in first] == [task.workspace for task in second]
        assert [task.oracle for task in first] == [task.oracle for task in second]
        assert all(a.prompt != b.prompt for a, b in zip(first, other))
        assert all(a.workspace != b.workspace for a, b in zip(first, other))


def test_review_new_fixtures_are_valid_on_every_dev_seed() -> None:
    """Each dev seed: independent oracle, red defect, green reference, one line."""
    for seed in _DEV_SEEDS:
        tasks = review.generate(task_rng(seed, "J"), RunConfig(profile="stretch-v1"))
        for task in tasks[2:]:
            buggy = task.workspace[task.oracle["target_file"]]
            reference = task.oracle["reference_source"]
            line = task.oracle["line"]
            assert [i for i, (a, b) in enumerate(zip(buggy.splitlines(), reference.splitlines()), 1) if a != b] == [line]

            rows, note = review._run_test_modules(task, buggy, ("hidden",))
            assert not note
            assert any(not ok for ok, _ in rows["hidden"].values()), task.id

            rows, note = review._run_test_modules(task, reference, ("hidden",))
            assert not note
            assert rows["hidden"] and all(ok for ok, _ in rows["hidden"].values()), task.id

            score = _score(task, task.oracle["reference"])
            assert score.earned == pytest.approx(task.max_points)


def test_review_j3_oracle_is_the_exact_allowed_set_without_leaks() -> None:
    for seed in _DEV_SEEDS:
        task = review.generate(task_rng(seed, "J"), RunConfig(profile="pilot"))[0]
        rows = {row["id"]: row for row in _j3_table(task)}
        us = task.oracle["tenant"]
        them = task.oracle["foreign_tenant"]
        assert us != them
        allowed = task.oracle["allowed_ids"]
        assert allowed
        assert sorted(allowed) == [row_id for row_id, row in rows.items() if row["tenant"] == us]
        assert any(row["tenant"] == them for row in rows.values())

        collected: list[int] = []
        for page in task.oracle["pages"]:
            collected.extend(page)
            assert page
        assert collected == allowed
        assert len({row_id for row_id in collected}) == len(collected)


def _j3_table(task: TaskInstance) -> list[dict]:
    source = task.workspace[task.oracle["target_file"]]
    marker = "_TABLE = (\n"
    start = source.index(marker) + len(marker)
    end = source.index(")\n", start)
    return [
        ast.literal_eval(line.strip().rstrip(","))
        for line in source[start:end].splitlines()
        if line.strip()
    ]


def test_review_j3_deny_everyone_mutant_is_not_full() -> None:
    """Fixing the order but refusing every page passes the shape, not the behaviour."""
    for task in _extra_tasks():
        if task.id != "J-3":
            continue
        buggy = task.workspace[task.oracle["target_file"]]
        source = buggy.replace(
            'return {"items": [dict(row) for row in window], "next_cursor": nxt}',
            'return {"items": [], "next_cursor": None}',
        )
        assert source != buggy
        score = _mutant(task, source)
        assert _check(score, "one_line_fix").passed
        assert _check(score, "defect_located").passed
        assert not _check(score, "fix_passes_hidden_tests").passed
        assert 0.0 < score.earned < task.max_points


def test_review_j3_filter_after_pagination_mutant_is_not_full() -> None:
    for task in _extra_tasks():
        if task.id != "J-3":
            continue
        score = _mutant(task, task.workspace[task.oracle["target_file"]])
        assert score.earned == 0.0
        assert not _check(score, "fix_passes_hidden_tests").passed


def test_review_j4_payload_dedupe_and_double_charge_mutants_are_not_full() -> None:
    """Dedup on the payload and an unguarded effect both keep the shape only."""
    for task in _extra_tasks():
        if task.id != "J-4":
            continue
        buggy = task.workspace[task.oracle["target_file"]]
        payload_dedupe = buggy.replace("    token = (user, amount)", "    token = user")
        double_charge = buggy.replace("    if token in _SEEN:", "    if False:")
        for source in (payload_dedupe, double_charge):
            assert source != buggy
            score = _mutant(task, source)
            assert _check(score, "one_line_fix").passed
            assert not _check(score, "fix_passes_hidden_tests").passed
            assert 0.0 < score.earned < task.max_points


def test_review_j4_fixture_counter_proves_one_effect_per_key() -> None:
    """The shipped module, once fixed, dedupes on the key and not on the payload."""
    for task in _extra_tasks():
        if task.id != "J-4":
            continue
        rows, note = review._run_test_modules(task, task.oracle["reference_source"], ("hidden",))
        assert not note
        assert rows["hidden"] and all(ok for ok, _ in rows["hidden"].values())
        names = " ".join(rows["hidden"])
        assert "new_key" in names and "two_users" in names
