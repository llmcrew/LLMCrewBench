"""Category J (code review): registry contract, determinism and behavioural scoring.

The evaluator is only trusted if the fixture it scores is real: the hidden tests
must fail on the shipped file and pass on the corrected one, the reported line
must be the line that actually changes, and the two checks must be independent —
a right line with a broken file, or a wrong line with a working file, each keep
exactly the points they earned.
"""

from __future__ import annotations

import json
from typing import Any

from dailybench import ModelReply, RunConfig, TaskInstance, Transcript, Turn, core
from dailybench.rng import task_rng
from dailybench.tasks import review

_SEED = 20260915
_CACHE: dict[int, list[TaskInstance]] = {}


def _tasks(seed: int = _SEED) -> list[TaskInstance]:
    """Generated tasks, shared between tests (generation runs the hidden suite)."""
    if seed not in _CACHE:
        _CACHE[seed] = review.generate(task_rng(seed, "J"), RunConfig())
    return _CACHE[seed]


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


def test_review_reference_scores_full_and_garbage_scores_zero() -> None:
    for task in _tasks():
        good = _score(task, task.oracle["reference"])
        assert good.possible == task.max_points
        assert good.earned == task.max_points, [c.as_dict() for c in good.checks]
        assert [c.name for c in good.checks] == ["defect_located", "fix_passes_hidden_tests"]
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
        assert not _check(score, "defect_located").passed
        assert _check(score, "fix_passes_hidden_tests").passed


def test_review_unfixed_file_loses_only_the_fix_check() -> None:
    for task in _tasks():
        payload: dict[str, Any] = json.loads(task.oracle["reference"])
        payload["fix"] = task.workspace[task.oracle["target_file"]]
        score = _score(task, json.dumps(payload, ensure_ascii=False))

        assert score.earned == 1.0
        assert _check(score, "defect_located").passed
        failed = _check(score, "fix_passes_hidden_tests")
        # The unfixed file keeps the one-line contract (no lines changed), so the
        # failure must come from the hidden tests, not the contract check.
        assert not failed.passed and "скрытых тестов" in failed.detail


def test_review_accepts_a_fenced_fix_and_normalised_paths() -> None:
    """The fix is judged by behaviour, so a fenced block and a ``./`` path still score."""
    for task in _tasks():
        payload = {
            "file": f"./{task.oracle['target_file']}",
            "line": str(task.oracle["line"]),
            "fix": f"```python\n{task.oracle['reference_source']}```",
        }
        assert _score(task, json.dumps(payload, ensure_ascii=False)).earned == task.max_points


def test_review_broken_fix_keeps_only_the_location_points() -> None:
    """A syntax error in the returned file is a failed repair, not a crash."""
    for task in _tasks():
        payload = {
            "file": task.oracle["target_file"],
            "line": task.oracle["line"],
            "fix": "def broken(:\n",
        }
        score = _score(task, json.dumps(payload, ensure_ascii=False))
        assert score.earned == 1.0
        assert not _check(score, "fix_passes_hidden_tests").passed
