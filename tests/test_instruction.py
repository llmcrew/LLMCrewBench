"""Category A pilot: IF-6 — CSV log summary with exception precedence.

The scenario is proven from three sides that must not agree by construction:
the oracle is built from structured rows, the rendered prompt is decoded back
with a separate RFC 4180 reader, and the rule itself is recomputed here. The
target mutants are the ones the plan names: a naive comma split, a record the
exclusion should have dropped, and ``>`` in place of ``>=``.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dailybench import RunConfig, Score, TaskInstance, Transcript, Turn, ModelReply, core
from dailybench.rng import task_rng
from dailybench.suite import apply_profile_weights
from dailybench.tasks import instruction

DEV_SEEDS = (1, 2, 3, 4, 5)
HOLDOUT_SEEDS = (101, 202, 303)
HEADER = "id,msg,service,level,ms,flag"


def _transcript(task: TaskInstance, answer: str) -> Transcript:
    return Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[{"role": "user", "content": task.prompt}], reply=ModelReply(text=answer))],
    )


def _checks(score: Score) -> dict[str, bool]:
    return {check.name: check.passed for check in score.checks}


def _task(seed: int = 1) -> TaskInstance:
    return next(task for task in instruction.generate(task_rng(seed, "A"), RunConfig(profile="pilot")) if task.id == "IF-6")


def _body(task: TaskInstance) -> str:
    return task.prompt.split("\n\n", 1)[1].split("\n\nПравила применяются", 1)[0]


def _parse_rfc(body: str) -> list[list[str]]:
    """Minimal RFC 4180 reader: doubled quotes, quoted commas, CRLF or LF."""
    rows: list[list[str]] = []
    row: list[str] = []
    field: list[str] = []
    quoted = False
    index = 0
    body = body.replace("\r\n", "\n")
    while index < len(body):
        char = body[index]
        if quoted:
            if char == '"':
                if index + 1 < len(body) and body[index + 1] == '"':
                    field.append('"')
                    index += 2
                    continue
                quoted = False
            else:
                field.append(char)
        elif char == '"':
            quoted = True
        elif char == ",":
            row.append("".join(field))
            field = []
        elif char == "\n":
            row.append("".join(field))
            rows.append(row)
            row, field = [], []
        else:
            field.append(char)
        index += 1
    if field or row:
        row.append("".join(field))
        rows.append(row)
    return rows


def _rows(task: TaskInstance) -> list[dict[str, str]]:
    header, *rest = _parse_rfc(_body(task))
    assert header == HEADER.split(",")
    return [dict(zip(header, values)) for values in rest]


def _independent_selection(task: TaskInstance) -> tuple[list[int], int, int]:
    target, threshold = task.oracle["target_service"], task.oracle["threshold"]
    picked = [
        row for row in _rows(task)
        if row["flag"] != "ignore"
        and row["service"] == target
        and (row["level"] == "CRITICAL" or (row["level"] == "ERROR" and int(row["ms"]) >= threshold))
    ]
    return sorted(int(row["id"]) for row in picked), len(picked), sum(int(row["ms"]) for row in picked)


def _answer(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_if6_scenario_contract() -> None:
    task = _task()
    assert task.id == "IF-6" and task.category == "A"
    assert task.max_points == 2.0
    score = instruction.evaluate(task, _transcript(task, "не знаю"))
    assert score.possible == 2.0
    assert [check.name for check in score.checks] == ["json_format", "ids", "aggregate"]
    assert [check.points for check in score.checks] == [0.5, 0.75, 0.75]
    assert _checks(score) == {"json_format": False, "ids": False, "aggregate": False}
    assert json.loads(task.oracle["reference"]) == {
        "ids": task.oracle["ids"],
        "count": task.oracle["count"],
        "total_ms": task.oracle["total_ms"],
    }


def test_if6_does_not_leak_the_oracle_into_the_prompt() -> None:
    task = _task()
    assert task.oracle["reference"] not in task.prompt
    assert '"ids":[]' in task.prompt  # the empty shape is a format example, not the answer
    assert "target_service" not in task.prompt


# --------------------------------------------------------------------------- #
# Reference, garbage and seed invariants
# --------------------------------------------------------------------------- #


def test_if6_reference_scores_full_and_garbage_zero() -> None:
    for seed in DEV_SEEDS:
        task = _task(seed)
        good = instruction.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert good.earned == 2.0 and good.possible == 2.0
        assert _checks(good) == {"json_format": True, "ids": True, "aggregate": True}
        assert instruction.evaluate(task, _transcript(task, "не знаю")).earned == 0.0
        assert instruction.evaluate(task, _transcript(task, "")).earned == 0.0


def test_if6_fixture_bounds_and_edge_cases() -> None:
    for seed in DEV_SEEDS:
        task = _task(seed)
        rows = _rows(task)
        assert 20 <= len(rows) <= 40
        assert task.meta["rows"] == len(rows)

        quoted = [line for line in _body(task).splitlines() if line.startswith('"') or ',"' in line]
        assert quoted, "the fixture must quote fields with commas"
        assert any('""' in line for line in _body(task).splitlines()), "quotes must be doubled"

        boundary = task.oracle["boundary_id"]
        critical = task.oracle["critical_id"]
        ignored = task.oracle["ignored_ids"]
        assert boundary in task.oracle["ids"]          # exactly on the threshold: >= keeps it
        assert critical in task.oracle["ids"]          # exception wins over the general rule
        assert all(item not in task.oracle["ids"] for item in ignored)
        assert task.oracle["ids"] == sorted(task.oracle["ids"])
        assert boundary not in ignored and critical not in ignored
        # The ignored records really matched one of the inclusion rules.
        by_id = {int(row["id"]): row for row in rows}
        for item in ignored:
            row = by_id[item]
            assert row["service"] == task.oracle["target_service"]
            assert row["level"] in ("ERROR", "CRITICAL")


def test_if6_oracle_matches_an_independent_reading() -> None:
    for seed in DEV_SEEDS:
        task = _task(seed)
        ids, count, total = _independent_selection(task)
        assert ids == task.oracle["ids"]
        assert count == task.oracle["count"]
        assert total == task.oracle["total_ms"]


def test_if6_is_seed_deterministic() -> None:
    first = instruction.generate(task_rng(17, "A"), RunConfig(profile="pilot"))
    second = instruction.generate(task_rng(17, "A"), RunConfig(profile="pilot"))
    other = instruction.generate(task_rng(18, "A"), RunConfig(profile="pilot"))
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle for task in first] == [task.oracle for task in second]
    assert first[0].prompt != other[0].prompt
    assert first[0].oracle["reference"] != other[0].oracle["reference"]


def test_if6_holdout_seeds_keep_the_same_invariants() -> None:
    """Seeds outside the development set still produce a valid, non-degenerate task."""
    for seed in HOLDOUT_SEEDS:
        task = _task(seed)
        rows = _rows(task)
        assert 20 <= len(rows) <= 40
        assert task.oracle["boundary_id"] in task.oracle["ids"]
        assert task.oracle["critical_id"] in task.oracle["ids"]
        assert all(item not in task.oracle["ids"] for item in task.oracle["ignored_ids"])
        ids, count, total = _independent_selection(task)
        assert (ids, count, total) == (task.oracle["ids"], task.oracle["count"], task.oracle["total_ms"])
        assert instruction.evaluate(task, _transcript(task, task.oracle["reference"])).earned == 2.0
        assert instruction.evaluate(task, _transcript(task, "не знаю")).earned == 0.0


# --------------------------------------------------------------------------- #
# Target mutants
# --------------------------------------------------------------------------- #


def test_if6_naive_csv_split_loses_the_boundary() -> None:
    task = _task()
    picked: list[int] = []
    for line in _body(task).splitlines():
        parts = line.split(",")
        try:
            row_id, service, level, ms = int(parts[0]), parts[2], parts[3], int(parts[4])
        except (IndexError, ValueError):
            continue  # a quoted comma shifts the columns
        target = task.oracle["target_service"]
        if service == target and level == "ERROR" and ms >= task.oracle["threshold"]:
            picked.append(row_id)
    assert sorted(picked) != task.oracle["ids"]
    assert task.oracle["boundary_id"] not in picked

    score = instruction.evaluate(task, _transcript(task, _answer({"ids": sorted(picked), "count": len(picked), "total_ms": 0})))
    assert score.earned < 2.0
    assert _checks(score)["ids"] is False


def test_if6_ignored_record_included_loses_both_content_points() -> None:
    task = _task()
    payload = json.loads(task.oracle["reference"])
    ignored_id = task.oracle["ignored_ids"][0]
    ignored_ms = next(int(row["ms"]) for row in _rows(task) if int(row["id"]) == ignored_id)
    payload["ids"] = sorted(payload["ids"] + [ignored_id])
    payload["count"] += 1
    payload["total_ms"] += ignored_ms

    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert score.earned == 0.5
    assert _checks(score) == {"json_format": True, "ids": False, "aggregate": False}


def test_if6_greater_than_instead_of_at_least_loses_the_boundary() -> None:
    task = _task()
    payload = json.loads(task.oracle["reference"])
    payload["ids"] = sorted(item for item in payload["ids"] if item != task.oracle["boundary_id"])
    payload["count"] -= 1
    payload["total_ms"] -= task.oracle["threshold"]

    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert score.earned == 0.5
    assert _checks(score) == {"json_format": True, "ids": False, "aggregate": False}


def test_if6_right_data_in_the_wrong_format_does_not_score_full() -> None:
    task = _task()
    reference = task.oracle["reference"]

    fenced = instruction.evaluate(task, _transcript(task, "```json\n" + reference + "\n```"))
    assert fenced.earned == 1.5
    assert _checks(fenced) == {"json_format": False, "ids": True, "aggregate": True}

    padded = instruction.evaluate(task, _transcript(task, "Готово:\n" + reference + "\nконец"))
    assert padded.earned == 1.5
    assert _checks(padded)["json_format"] is False

    extra_key = json.loads(reference)
    extra_key["note"] = "ok"
    assert _checks(instruction.evaluate(task, _transcript(task, _answer(extra_key))))["json_format"] is False

    string_count = json.loads(reference)
    string_count["count"] = str(string_count["count"])
    as_string = instruction.evaluate(task, _transcript(task, _answer(string_count)))
    assert as_string.earned == 0.75
    assert _checks(as_string) == {"json_format": False, "ids": True, "aggregate": False}


# --------------------------------------------------------------------------- #
# Profile registration
# --------------------------------------------------------------------------- #


def test_if6_is_a_pilot_scenario_only() -> None:
    module = core.REGISTRY["A"]
    assert module.expected_ids == instruction.IDS
    assert module.expected_ids_for() == instruction.IDS
    assert module.expected_ids_for("pilot") == ("IF-6",)
    assert module.expected_ids_for("stretch-v1") == instruction.STRETCH_IDS == instruction.IDS + ("IF-6",)

    def generated(profile: str) -> list[TaskInstance]:
        return instruction.generate(task_rng(1, "A"), RunConfig(profile=profile))

    assert [task.id for task in generated(core.DEFAULT_PROFILE)] == list(instruction.IDS)
    pilot = generated("pilot")
    assert [(task.id, task.max_points) for task in pilot] == [("IF-6", 2.0)]

    stretch = generated("stretch-v1")
    assert [task.id for task in stretch] == [*instruction.IDS, "IF-6"]
    # The stretch profile splits the fixed category total over its six tasks.
    apply_profile_weights(stretch, "stretch-v1")
    assert sum(task.max_points for task in stretch) == core.CATEGORY_POINTS["A"] == 10.0
    # 10 / 6 leaves a 1e-4-scale rounding tail on the last task.
    assert max(task.max_points for task in stretch) - min(task.max_points for task in stretch) <= 0.0002


def test_if6_registered_evaluator_is_the_dispatched_one() -> None:
    task = _task()
    score = core.REGISTRY["A"].evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == 2.0
    assert [check.name for check in score.checks] == ["json_format", "ids", "aggregate"]
    with pytest.raises(KeyError):
        instruction.evaluate(replace(task, id="IF-9"), _transcript(task, ""))
