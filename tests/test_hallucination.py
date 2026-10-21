"""Tests for F-2 — three versioned sources decided by a trust rule.

The scenario is pilot-only: the baseline keeps ``F`` untouched. Covers the
profile contract, seed determinism over five development seeds, the exact
oracle, the three target mutants (stale source, external knowledge, a
contradicting line outside the table) and garbage.
"""

from __future__ import annotations

import random
import re

import pytest

from dailybench import ModelReply, RunConfig, TaskInstance, Transcript, Turn, core
from dailybench.rng import task_rng
from dailybench.runner import Runner
from dailybench.selfcheck import GarbageProvider
from dailybench.suite import apply_profile_weights
from dailybench.tasks import hallucination

DEV_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)


def _tasks(seed: int = DEV_SEEDS[0], profile: str = "pilot") -> list[TaskInstance]:
    return hallucination.generate(task_rng(seed, "F"), RunConfig(seed=seed, profile=profile))


def _task(seed: int = DEV_SEEDS[0]) -> TaskInstance:
    return _tasks(seed)[0]


def _transcript(task: TaskInstance, reply: str) -> Transcript:
    return Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[], reply=ModelReply(text=reply))],
    )


def _failed(score: core.Score) -> list[str]:
    return [check.name for check in score.checks if not check.passed]


def _rows(task: TaskInstance) -> list[list[str]]:
    """Reference table as mutable rows ``[index, value, source]``."""
    return [line.strip("|").split("|") for line in task.oracle["reference"].splitlines()[2:]]


def _render(rows: list[list[str]]) -> str:
    body = "\n".join("| {} | {} | {} |".format(*(cell.strip() for cell in row)) for row in rows)
    return "| # | Значение | Источник |\n| --- | --- | --- |\n" + body


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_profile_contract_and_registry() -> None:
    module = core.REGISTRY["F"]
    assert module.expected_ids == ("F",)
    assert module.expected_ids_for() == ("F",)
    assert module.expected_ids_for("pilot") == ("F-2",)
    assert module.expected_ids_for("stretch-v1") == ("F", "F-2")
    assert module.points == 8.0

    assert [task.id for task in _tasks(profile=core.DEFAULT_PROFILE)] == ["F"]
    assert [task.id for task in _tasks(profile="pilot")] == ["F-2"]
    assert [task.id for task in _tasks(profile="stretch-v1")] == ["F", "F-2"]


def test_f2_task_declares_eight_points_and_its_own_title() -> None:
    task = _task()
    assert task.category == "F"
    assert task.max_points == 8.0
    assert task.title == "Versioned sources and trust rule"
    assert task.meta["scenario"] == "F-2"


def test_stretch_splits_the_category_total_over_both_tasks() -> None:
    tasks = _tasks(profile="stretch-v1")
    apply_profile_weights(tasks, "stretch-v1")
    assert [task.max_points for task in tasks] == [4.0, 4.0]
    assert sum(task.max_points for task in tasks) == core.CATEGORY_POINTS["F"]


# --------------------------------------------------------------------------- #
# Determinism and fixture invariants
# --------------------------------------------------------------------------- #


def test_generation_is_deterministic_but_seeded() -> None:
    first = _tasks(DEV_SEEDS[0])
    second = _tasks(DEV_SEEDS[0])
    other = _tasks(DEV_SEEDS[1])

    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle for task in first] == [task.oracle for task in second]
    assert first[0].prompt != other[0].prompt
    assert first[0].oracle["reference"] != other[0].oracle["reference"]


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_fixture_has_six_facts_and_no_unresolved_tie(seed: int) -> None:
    task = _task(seed)
    questions = task.oracle["questions"]

    assert len(questions) == 6
    assert sorted(q["index"] for q in questions) == [1, 2, 3, 4, 5, 6]
    assert [q["index"] for q in questions] == list(range(1, 7))
    assert len({q["field"] for q in questions}) == 6
    assert task.oracle["unknown_count"] == 2
    assert task.oracle["conflicts"] == 3
    assert task.oracle["revoked"] == 1

    # Every known fact has an exact value from a named source; every unknown
    # fact names the documents it was checked in.
    for question in questions:
        assert question["sources"]
        assert set(question["sources"]) <= {"LIVE", "ARCHIVE", "PARTIAL"}
        if question["kind"] == "known":
            assert question["value"] and question["value"] != "UNKNOWN"
        else:
            assert question["value"] is None

    statuses = [line.split("updated: ")[1] for line in task.oracle["documents"].values() if "updated: " in line]
    assert len(statuses) == len(set(statuses)), "даты документов не должны совпадать"

    # Contested facts really occur in two sources with different values.
    archive = task.oracle["documents"]["ARCHIVE"]
    live = task.oracle["documents"]["LIVE"]
    partial = task.oracle["documents"]["PARTIAL"]
    for question in questions:
        if not question.get("contested"):
            continue
        holders = [doc for doc in (archive, live, partial) if f"{question['field']}:" in doc]
        assert len(holders) >= 2, question["field"]
        assert len({" ".join(holder.splitlines()) for holder in holders}) >= 2


# --------------------------------------------------------------------------- #
# Score bands
# --------------------------------------------------------------------------- #

def _entries(task: TaskInstance, field: str) -> list[tuple[str, str, str, str]]:
    found = []
    for name, document in task.oracle["documents"].items():
        authority = re.search(r"^authority: (\w+)$", document, re.MULTILINE)
        updated = re.search(r"^updated: (\S+)$", document, re.MULTILINE)
        entry = re.search(rf"^- {field}: (.*)$", document, re.MULTILINE)
        if not entry or "revoked: true" in entry.group(1):
            continue
        found.append((authority.group(1), updated.group(1), name, entry.group(1).split(" (")[0].strip()))
    return found


def _apply_rule(task: TaskInstance, field: str) -> tuple[str, list[str]]:
    """Independent implementation of the prompt's trust rule, over the documents."""
    entries = _entries(task, field)
    primary = [entry for entry in entries if entry[0] == "primary"]
    if primary:
        return primary[0][3], [primary[0][2]]
    if len(entries) > 1:
        winner = max(entries, key=lambda entry: entry[1])
        return winner[3], [winner[2]]
    if entries:
        return entries[0][3], [entries[0][2]]
    mentioned = sorted(
        name for name, document in task.oracle["documents"].items()
        if re.search(rf"^- {field}:", document, re.MULTILINE)
    )
    return "UNKNOWN", mentioned or sorted(task.oracle["documents"])


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_oracle_matches_an_independent_rule_application(seed: int) -> None:
    task = _task(seed)
    for question in task.oracle["questions"]:
        value, sources = _apply_rule(task, question["field"])
        assert value == (question["value"] or "UNKNOWN"), question["field"]
        assert set(sources) == set(question["sources"]), question["field"]



@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_reference_scores_the_full_budget(seed: int) -> None:
    task = _task(seed)
    score = hallucination.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert _failed(score) == []
    assert score.earned == 8.0
    assert score.possible == task.max_points == 8.0
    assert sum(check.points for check in score.checks) == task.max_points
    assert len(score.checks) == 8


def test_garbage_scores_zero() -> None:
    task = _task()
    run = Runner(RunConfig(profile="pilot"), GarbageProvider()).run_task(task)
    assert run.score.earned == 0.0
    assert run.score.possible == 8.0
    assert "table_shape" in _failed(run.score)
    assert "no_extra_text" in _failed(run.score)


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_a_stale_source_is_not_full(seed: int) -> None:
    task = _task(seed)
    score = hallucination.evaluate(task, _transcript(task, task.oracle["mutant_stale"]))
    assert score.earned < 8.0
    contested = [q["index"] for q in task.oracle["questions"] if q.get("contested")]
    assert set(f"q{index}" for index in contested) <= set(_failed(score))


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_external_knowledge_is_not_full(seed: int) -> None:
    task = _task(seed)
    score = hallucination.evaluate(task, _transcript(task, task.oracle["mutant_external"]))
    assert score.earned < 8.0
    unknown = [q["index"] for q in task.oracle["questions"] if q["kind"] == "unknown"]
    assert set(f"q{index}" for index in unknown) <= set(_failed(score))


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_a_contradicting_line_outside_the_table_is_not_full(seed: int) -> None:
    task = _task(seed)
    score = hallucination.evaluate(task, _transcript(task, task.oracle["mutant_outside"]))
    assert score.earned == 6.0
    assert _failed(score) == ["table_shape", "no_extra_text"]
    assert all(check.passed for check in score.checks[:6]), "строки таблицы остаются верными"


def test_a_wrong_source_loses_only_that_fact() -> None:
    task = _task()
    rows = _rows(task)
    row = rows[0]
    expected = set(next(q for q in task.oracle["questions"] if str(q["index"]) == "1")["sources"])
    outside = [name for name in ("LIVE", "ARCHIVE", "PARTIAL") if name not in expected]
    row[2] = outside[0] if outside else next(iter(expected))

    score = hallucination.evaluate(task, _transcript(task, _render(rows)))
    assert _failed(score) == ["q1"]
    assert score.earned == 7.0
    assert "источник" in score.checks[0].detail and "ожидалось" in score.checks[0].detail


def test_guessed_value_on_an_unknown_fact_loses_only_that_fact() -> None:
    task = _task()
    rows = _rows(task)
    index = next(
        row[0].strip()
        for row in rows
        if next(q for q in task.oracle["questions"] if str(q["index"]) == row[0].strip())["kind"] == "unknown"
    )
    rows[int(index) - 1][1] = "42"

    score = hallucination.evaluate(task, _transcript(task, _render(rows)))
    assert _failed(score) == [f"q{index}"]
    assert score.earned == 7.0


def test_a_fenced_table_still_scores_full() -> None:
    task = _task()
    fenced = f"```\n{task.oracle['reference']}\n```"
    score = hallucination.evaluate(task, _transcript(task, fenced))
    assert _failed(score) == []
    assert score.earned == 8.0


# --------------------------------------------------------------------------- #
# Baseline is untouched, oracle stays out of the prompt
# --------------------------------------------------------------------------- #


def test_baseline_f_still_scores_full() -> None:
    task = next(task for task in _tasks(profile=core.DEFAULT_PROFILE) if task.id == "F")
    score = hallucination.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == task.max_points == 8.0
    assert len(score.checks) == 10


def test_prompt_keeps_the_answers_out() -> None:
    for seed in DEV_SEEDS:
        task = _task(seed)
        for line in task.oracle["reference"].splitlines()[2:]:
            assert line not in task.prompt
        assert "мутант" not in task.prompt
        assert "| # | Значение | Источник |" in task.prompt


def test_the_baseline_fixture_does_not_depend_on_the_profile() -> None:
    """Stretch generates the baseline scenario first, from the same stream."""
    base = hallucination.generate(random.Random(7), RunConfig(seed=7))[0]
    stretch = hallucination.generate(random.Random(7), RunConfig(seed=7, profile="stretch-v1"))
    assert [task.id for task in stretch] == ["F", "F-2"]
    assert stretch[0].prompt == base.prompt
    assert stretch[0].oracle["reference"] == base.oracle["reference"]
