"""Category A pilot: IF-6 — CSV log summary with exception precedence.

The scenario is proven from three sides that must not agree by construction:
the oracle is built from structured rows, the rendered prompt is decoded back
with a separate RFC 4180 reader, and the rule itself is recomputed here. The
target mutants are the ones the plan names: a naive comma split, a record the
exclusion should have dropped, and ``>`` in place of ``>=``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

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
    assert module.expected_ids_for("stretch-v1") == instruction.STRETCH_IDS
    assert instruction.STRETCH_IDS == instruction.IDS + (
        "IF-6", "IF-7", "IF-7-KK", "IF-7-EN", "IF-8", "IF-9", "IF-10"
    )

    def generated(profile: str) -> list[TaskInstance]:
        return instruction.generate(task_rng(1, "A"), RunConfig(profile=profile))

    assert [task.id for task in generated(core.DEFAULT_PROFILE)] == list(instruction.IDS)
    pilot = generated("pilot")
    assert [(task.id, task.max_points) for task in pilot] == [("IF-6", 2.0)]

    stretch = generated("stretch-v1")
    assert [task.id for task in stretch] == list(instruction.STRETCH_IDS)
    # The stretch profile splits the fixed category total over its twelve tasks.
    apply_profile_weights(stretch, "stretch-v1")
    assert sum(task.max_points for task in stretch) == pytest.approx(core.CATEGORY_POINTS["A"]) == pytest.approx(10.0)
    # 10 / 12 rounds to four places and parks the remainder tail on the last task.
    assert max(task.max_points for task in stretch) - min(task.max_points for task in stretch) <= 0.0005


def test_if6_registered_evaluator_is_the_dispatched_one() -> None:
    task = _task()
    score = core.REGISTRY["A"].evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == 2.0
    assert [check.name for check in score.checks] == ["json_format", "ids", "aggregate"]
    with pytest.raises(KeyError):
        instruction.evaluate(replace(task, id="IF-99"), _transcript(task, ""))


# --------------------------------------------------------------------------- #
# N7 — composite selection (IF-9/IF-10): rule priority, NFC keys, correction
# --------------------------------------------------------------------------- #

N7_HEADER = ("id", "key", "zone", "severity", "score", "ts", "rev", "muted")


def _n7_task(seed: int, scenario: str = "IF-9") -> TaskInstance:
    tasks = instruction.generate(task_rng(seed, "A"), RunConfig(profile="stretch-v1"))
    return next(task for task in tasks if task.id == scenario)


def _n7_table(task: TaskInstance) -> list[dict[str, str]]:
    header, *rest = task.prompt.split("\n\n")[1].split("\n")
    assert tuple(header.split("\t")) == N7_HEADER
    return [dict(zip(N7_HEADER, line.split("\t"))) for line in rest]


def _n7_bounds(task: TaskInstance) -> tuple[str, int, int]:
    """The zone and both bounds read from the brief itself, not from the oracle."""
    zone = re.search(r"zone = (\w+)", task.prompt).group(1)
    first = int(re.search(r"score >= (\d+)", task.prompt).group(1))
    second = int(re.search(r"теперь (\d+)", task.dialogue[0]).group(1))
    return zone, first, second


def _n7_select(task: TaskInstance, threshold: int, keep_muted: bool = False,
               merge_keys: bool = True) -> list[dict[str, str]]:
    zone, _, _ = _n7_bounds(task)
    kept = [
        row for row in _n7_table(task)
        if row["zone"] == zone and row["severity"] == "critical"
        and int(row["score"]) >= threshold
        and (keep_muted or row["muted"] != "yes")
    ]
    best: dict[str, dict[str, str]] = {}
    for row in kept:
        if not merge_keys:
            best[f"{row['key']}#{row['id']}"] = row
            continue
        key = unicodedata.normalize("NFC", row["key"])
        if key not in best or int(row["rev"]) > int(best[key]["rev"]):
            best[key] = row
    return sorted(best.values(), key=lambda row: (-int(row["score"]), row["ts"], int(row["id"])))


def _n7_answer(rows: list[dict[str, str]]) -> str:
    picks = [{"id": int(row["id"]), "score": int(row["score"])} for row in rows]
    return _answer({"picks": picks, "count": len(picks), "total": sum(item["score"] for item in picks)})


@pytest.mark.parametrize("scenario", ("IF-9", "IF-10"))
def test_n7_scenario_contract(scenario: str) -> None:
    task = _n7_task(1, scenario)
    assert task.id == scenario and task.category == "A"
    assert task.max_points == 1.0
    assert task.private is True and task.fresh_after == instruction.FRESH_AFTER
    assert len(task.dialogue) == 1
    score = instruction.evaluate(task, _transcript(task, "не знаю"))
    assert score.possible == 1.0
    assert [check.name for check in score.checks] == ["json_format", "records", "ordering"]
    assert [check.points for check in score.checks] == [0.25, 0.375, 0.375]
    assert _checks(score) == {"json_format": False, "records": False, "ordering": False}
    assert json.loads(task.oracle["reference"]) == {
        "picks": task.oracle["picks"],
        "count": task.oracle["count"],
        "total": task.oracle["total"],
    }
    assert task.oracle["count"] == len(task.oracle["picks"])


def test_n7_does_not_leak_the_oracle_into_the_prompt() -> None:
    task = _n7_task(1)
    assert task.oracle["reference"] not in task.prompt
    assert '"picks":[{"id":0' in task.prompt  # the shape is an example, not the answer
    assert "Поправка" not in task.prompt  # the correction stays in the dialogue
    assert f"score >= {task.oracle['threshold1']}" in task.prompt
    assert f"score >= {task.oracle['threshold2']}" not in task.prompt


@pytest.mark.parametrize("scenario", ("IF-9", "IF-10"))
def test_n7_reference_scores_full_and_garbage_zero(scenario: str) -> None:
    for seed in DEV_SEEDS:
        task = _n7_task(seed, scenario)
        good = instruction.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert good.earned == good.possible == 1.0
        assert _checks(good) == {"json_format": True, "records": True, "ordering": True}
        for garbage in ("не знаю", ""):
            assert instruction.evaluate(task, _transcript(task, garbage)).earned == 0.0


def test_n7_oracle_matches_an_independent_reading() -> None:
    for seed in DEV_SEEDS:
        select = _n7_task(seed, "IF-9")
        __, first, second = _n7_bounds(select)
        live = _n7_select(select, second)
        assert [int(row["id"]) for row in live] == select.oracle["ids"]
        assert sum(int(row["score"]) for row in live) == select.oracle["total"]
        superseded = [int(row["id"]) for row in _n7_select(select, first)]
        assert superseded == select.oracle["superseded_ids"]
        assert superseded != select.oracle["ids"]

        empty = _n7_task(seed, "IF-10")
        __, first, second = _n7_bounds(empty)
        assert _n7_select(empty, second) == []
        assert (empty.oracle["ids"], empty.oracle["count"], empty.oracle["total"]) == ([], 0, 0)
        assert [int(row["id"]) for row in _n7_select(empty, first)] == empty.oracle["superseded_ids"]


def test_n7_fixture_edges_are_the_declared_ones() -> None:
    for seed in DEV_SEEDS:
        task = _n7_task(seed)
        rows = {int(row["id"]): row for row in _n7_table(task)}
        assert 11 <= len(rows) <= 13
        assert task.meta["rows"] == len(rows)

        muted = rows[task.oracle["muted_id"]]
        assert muted["muted"] == "yes"
        assert int(muted["score"]) == max(int(row["score"]) for row in rows.values())
        assert muted["zone"] == task.oracle["zone"] and muted["severity"] == "critical"

        first_tie, second_tie = task.oracle["tie_ids"]
        assert first_tie == task.oracle["ids"][0] and second_tie == task.oracle["ids"][1]
        assert rows[first_tie]["score"] == rows[second_tie]["score"] == str(task.oracle["threshold2"])
        assert rows[first_tie]["ts"] < rows[second_tie]["ts"]
        assert first_tie > second_tie  # a tie-break by id would invert the expected order

        for low_id in task.oracle["low_rev_ids"]:
            low = rows[low_id]
            twin = next(
                row for row in rows.values()
                if row["id"] != low["id"]
                and unicodedata.normalize("NFC", row["key"]) == unicodedata.normalize("NFC", low["key"])
            )
            assert low["key"] != twin["key"], "both spellings really differ"
            assert int(twin["rev"]) > int(low["rev"])
            assert int(low["id"]) not in task.oracle["ids"]

        assert all(rows[item]["zone"] == task.oracle["zone"] for item in task.oracle["ids"])
        assert len(set(task.oracle["ids"])) == len(task.oracle["ids"])


def test_n7_correction_empties_the_second_variant() -> None:
    for seed in DEV_SEEDS:
        task = _n7_task(seed, "IF-10")
        assert task.oracle["picks"] == []
        assert task.oracle["superseded_ids"]
        assert task.oracle["threshold2"] > max(int(row["score"]) for row in _n7_table(task))
        assert task.oracle["reference"] == '{"picks":[],"count":0,"total":0}'


def test_n7_is_seed_deterministic() -> None:
    for scenario in ("IF-9", "IF-10"):
        first = _n7_task(17, scenario)
        second = _n7_task(17, scenario)
        assert (first.prompt, first.oracle, first.dialogue) == (second.prompt, second.oracle, second.dialogue)
        assert first.prompt != _n7_task(18, scenario).prompt


def test_n7_holdout_seeds_keep_the_same_invariants() -> None:
    for seed in HOLDOUT_SEEDS:
        for scenario in ("IF-9", "IF-10"):
            task = _n7_task(seed, scenario)
            __, first, second = _n7_bounds(task)
            live = [int(row["id"]) for row in _n7_select(task, second)]
            assert live == task.oracle["ids"]
            assert instruction.evaluate(task, _transcript(task, task.oracle["reference"])).earned == 1.0
            assert instruction.evaluate(task, _transcript(task, "не знаю")).earned == 0.0


# --------------------------------------------------------------------------- #
# N7 target mutants
# --------------------------------------------------------------------------- #


def test_n7_almost_right_json_loses_only_the_format_point() -> None:
    task = _n7_task(1)
    reference = task.oracle["reference"]
    reference_payload = json.loads(reference)

    fenced = instruction.evaluate(task, _transcript(task, "```json\n" + reference + "\n```"))
    assert fenced.earned == 0.75
    assert _checks(fenced) == {"json_format": False, "records": True, "ordering": True}

    padded = instruction.evaluate(task, _transcript(task, "Готово:\n" + reference + "\nконец"))
    assert _checks(padded) == {"json_format": False, "records": True, "ordering": True}

    extra_key = {**reference_payload, "note": "ok"}
    assert _checks(instruction.evaluate(task, _transcript(task, _answer(extra_key))))["json_format"] is False

    string_count = {**reference_payload, "count": str(reference_payload["count"])}
    assert _checks(instruction.evaluate(task, _transcript(task, _answer(string_count))))["json_format"] is False

    extra_field = {**reference_payload, "picks": [{**item, "ts": "09:00"} for item in reference_payload["picks"]]}
    assert _checks(instruction.evaluate(task, _transcript(task, _answer(extra_field))))["json_format"] is False


def test_n7_wrong_tie_break_keeps_the_records_but_loses_the_order() -> None:
    task = _n7_task(1)
    payload = json.loads(task.oracle["reference"])
    payload["picks"] = list(reversed(payload["picks"]))
    assert [item["id"] for item in payload["picks"]] != task.oracle["ids"]

    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert score.earned == 0.625
    assert _checks(score) == {"json_format": True, "records": True, "ordering": False}


def test_n7_ignored_correction_answers_the_old_bound() -> None:
    task = _n7_task(1)
    __, first, _second = _n7_bounds(task)
    stale = _n7_select(task, first)
    assert stale

    score = instruction.evaluate(task, _transcript(task, _n7_answer(stale)))
    assert score.earned == 0.25
    assert _checks(score) == {"json_format": True, "records": False, "ordering": False}


def test_n7_muted_record_survives_a_missing_exclusion_rule() -> None:
    task = _n7_task(1)
    __, _first, second = _n7_bounds(task)
    leaked = _n7_select(task, second, keep_muted=True)
    assert task.oracle["muted_id"] in [int(row["id"]) for row in leaked]

    score = instruction.evaluate(task, _transcript(task, _n7_answer(leaked)))
    assert score.earned == 0.25
    assert _checks(score) == {"json_format": True, "records": False, "ordering": False}


def test_n7_naive_nfc_keeps_the_superseded_spellings() -> None:
    task = _n7_task(1)
    __, _first, second = _n7_bounds(task)
    unmerged = _n7_select(task, second, merge_keys=False)
    got = [int(row["id"]) for row in unmerged]
    extra = set(got) - set(task.oracle["ids"])
    assert extra and extra <= set(task.oracle["low_rev_ids"])
    assert got != task.oracle["ids"]

    score = instruction.evaluate(task, _transcript(task, _n7_answer(unmerged)))
    assert score.earned == 0.25
    assert _checks(score) == {"json_format": True, "records": False, "ordering": False}


def test_n7_empty_variant_rejects_the_pre_correction_answer() -> None:
    task = _n7_task(1, "IF-10")
    __, first, _second = _n7_bounds(task)
    stale = _n7_select(task, first)
    assert stale

    score = instruction.evaluate(task, _transcript(task, _n7_answer(stale)))
    assert score.earned == 0.25
    assert _checks(score) == {"json_format": True, "records": False, "ordering": False}

    empty = instruction.evaluate(task, _transcript(task, '{"picks":[],"count":0,"total":0}'))
    assert empty.earned == 1.0

    padded = instruction.evaluate(task, _transcript(task, "Итог: {\"picks\":[],\"count\":0,\"total\":0}."))
    assert _checks(padded) == {"json_format": False, "records": True, "ordering": True}


# --------------------------------------------------------------------------- #
# N7 profile registration
# --------------------------------------------------------------------------- #


def test_n7_is_a_stretch_scenario_only() -> None:
    module = core.REGISTRY["A"]
    assert module.expected_ids_for("stretch-v1") == instruction.STRETCH_IDS
    assert instruction.STRETCH_IDS[-2:] == ("IF-9", "IF-10")
    assert {"IF-9", "IF-10"} <= instruction.PRIVATE_IDS
    assert module.expected_ids_for("pilot") == ("IF-6",)

    def generated(profile: str) -> list[TaskInstance]:
        return instruction.generate(task_rng(1, "A"), RunConfig(profile=profile))

    assert not {"IF-9", "IF-10"} & {task.id for task in generated(core.DEFAULT_PROFILE)}
    assert not {"IF-9", "IF-10"} & {task.id for task in generated("pilot")}

    stretch = generated("stretch-v1")
    assert [task.id for task in stretch][-2:] == ["IF-9", "IF-10"]
    assert all(task.private for task in stretch if task.id in ("IF-9", "IF-10"))
    apply_profile_weights(stretch, "stretch-v1")
    assert sum(task.max_points for task in stretch) == pytest.approx(core.CATEGORY_POINTS["A"]) == pytest.approx(10.0)
    assert max(task.max_points for task in stretch) - min(task.max_points for task in stretch) <= 0.0005


def test_n7_registered_evaluator_is_the_dispatched_one() -> None:
    for scenario in ("IF-9", "IF-10"):
        task = _n7_task(1, scenario)
        score = core.REGISTRY["A"].evaluate(task, _transcript(task, task.oracle["reference"]))
        assert score.earned == 1.0
        assert [check.name for check in score.checks] == ["json_format", "records", "ordering"]
    with pytest.raises(KeyError):
        instruction.evaluate(replace(_n7_task(1), id="IF-99"), _transcript(_n7_task(1), ""))


# --------------------------------------------------------------------------- #
# IF-7 ru/en/kk language pair: one semantic case, three self-contained locales
#
# The twins must share the case (same days, masses, invoice numbers) while each
# states its own calendar, separators and Unicode form. The parsers below are
# written here from the brief, not reused from the module, so a prompt that is
# merely a translated template fails them.
# --------------------------------------------------------------------------- #

PAIR_LANGS = ("ru", "kk", "en")
PAIR_IDS = dict(zip(PAIR_LANGS, instruction.IF7_IDS))
PAIR_THRESHOLD_TENTHS = 12_000

PAIR_MONTHS = {
    "ru": ("января", "февраля", "марта", "апреля", "мая", "июня",
           "июля", "августа", "сентября", "октября", "ноября", "декабря"),
    "kk": ("қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым",
           "шілде", "тамыз", "қыркүйек", "қазан", "қараша", "желтоқсан"),
    "en": ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"),
}

PAIR_DATES = {
    "ru": re.compile(r"(\d{1,2}) (" + "|".join(PAIR_MONTHS["ru"]) + r") (\d{4}) г\."),
    "kk": re.compile(r"(\d{4}) жылғы (\d{1,2}) (" + "|".join(PAIR_MONTHS["kk"]) + r")"),
    "en": re.compile(r"(\d{1,2}) (" + "|".join(PAIR_MONTHS["en"]) + r") (\d{4})"),
}

PAIR_MASSES = {
    "ru": re.compile(r"(\d{1,3}(?: \d{3})*),(\d)"),
    "kk": re.compile(r"(\d{1,3}(?: \d{3})*),(\d)"),
    "en": re.compile(r"(\d{1,3}(?:,\d{3})*)\.(\d)"),
}


def _pair_task(seed: int, lang: str) -> TaskInstance:
    tasks = instruction.generate(task_rng(seed, "A"), RunConfig(profile="stretch-v1"))
    return next(task for task in tasks if task.id == PAIR_IDS[lang])


def _pair_listing(task: TaskInstance) -> list[str]:
    """The fixture block between the two blank lines of the prompt."""
    return task.prompt.split("\n\n")[1].splitlines()


def _pair_read_date(raw: str, lang: str) -> date:
    match = PAIR_DATES[lang].fullmatch(raw)
    assert match, f"{lang} brief would not write {raw!r}"
    if lang == "kk":
        year, day, month = int(match.group(1)), int(match.group(2)), match.group(3)
    else:
        day, month, year = int(match.group(1)), match.group(2), int(match.group(3))
    return date(year, PAIR_MONTHS[lang].index(month) + 1, day)


def _pair_read_mass(raw: str, lang: str) -> int:
    """Tenths of a kilogram under the language's declared separators."""
    match = PAIR_MASSES[lang].fullmatch(raw)
    assert match, f"{lang} brief would not write {raw!r}"
    whole = match.group(1).replace(" ", "").replace(",", "")
    return int(whole) * 10 + int(match.group(2))


def _pair_read_listing(lines: list[str], lang: str) -> list[tuple[str, date, int]]:
    rows = []
    for line in lines:
        invoice, raw_date, raw_mass = line.split("; ")
        rows.append((invoice, _pair_read_date(raw_date, lang), _pair_read_mass(raw_mass, lang)))
    return rows


def _pair_write_date(day: date, lang: str) -> str:
    if lang == "kk":
        return f"{day.year} жылғы {day.day} {PAIR_MONTHS['kk'][day.month - 1]}"
    if lang == "en":
        return f"{day.day} {PAIR_MONTHS['en'][day.month - 1]} {day.year}"
    return f"{day.day} {PAIR_MONTHS['ru'][day.month - 1]} {day.year} г."


def _pair_write_mass(tenths: int, lang: str) -> str:
    whole, fraction = divmod(tenths, 10)
    if lang == "en":
        return f"{whole:,}.{fraction}"
    return f"{whole:,}".replace(",", " ") + f",{fraction}"


def _pair_select(
    rows: list[tuple[str, date, int]], window: tuple[str, str]
) -> list[tuple[str, date, int]]:
    """Independent recomputation: latest record per NFC key, then the window."""
    start, end = (date.fromisoformat(edge) for edge in window)
    latest: dict[str, tuple[str, date, int]] = {}
    for invoice, day, tenths in rows:
        key = unicodedata.normalize("NFC", invoice)
        if key not in latest or day >= latest[key][1]:
            latest[key] = (invoice, day, tenths)
    picked = [
        row for row in latest.values()
        if start <= row[1] <= end and row[2] >= PAIR_THRESHOLD_TENTHS
    ]
    picked.sort(key=lambda row: (row[1], unicodedata.normalize("NFC", row[0])))
    return picked


def _pair_tonnes(tenths: int) -> Decimal:
    return (Decimal(tenths) / 1000).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def test_if7_lang_pair_has_its_own_id_metadata_and_prompt() -> None:
    tasks = {lang: _pair_task(1, lang) for lang in PAIR_LANGS}
    assert [tasks[lang].id for lang in PAIR_LANGS] == list(instruction.IF7_IDS)
    assert tasks["ru"].cultures == ["ru"]
    assert tasks["kk"].cultures == ["kz"]
    assert tasks["en"].cultures == ["en-GB"]
    assert len({tasks[lang].prompt for lang in PAIR_LANGS}) == 3
    assert len({tasks[lang].oracle["reference"] for lang in PAIR_LANGS}) == 3
    for lang in PAIR_LANGS:
        assert tasks[lang].languages == [lang]
        assert tasks[lang].meta["languages"] == [lang]
        assert tasks[lang].meta["cultures"] == tasks[lang].cultures
        assert tasks[lang].private is True
        assert tasks[lang].max_points == 2.0


def test_if7_lang_pair_shares_one_semantic_fixture() -> None:
    """Same seed, same case: the languages may not drift to their own data."""
    for seed in DEV_SEEDS:
        tasks = [_pair_task(seed, lang) for lang in PAIR_LANGS]
        shapes = [
            sorted((row["identity"], row["day"], row["tenths"], row["decomposed"]) for row in task.oracle["records"])
            for task in tasks
        ]
        assert shapes[0] == shapes[1] == shapes[2]
        numbers = [
            {row["identity"]: row["invoice"].split("-")[1].split("/")[0] for row in task.oracle["records"]}
            for task in tasks
        ]
        assert numbers[0] == numbers[1] == numbers[2]
        picked = [
            [(row["date"], row["invoice"].split("-")[1].split("/")[0], row["tonnes"]) for row in task.oracle["rows"]]
            for task in tasks
        ]
        assert picked[0] == picked[1] == picked[2]
        for task in tasks:
            assert len(task.oracle["rows"]) == 3


def test_if7_lang_pair_states_its_own_calendar_and_separators() -> None:
    for seed in DEV_SEEDS:
        for lang in PAIR_LANGS:
            task = _pair_task(seed, lang)
            parsed = _pair_read_listing(_pair_listing(task), lang)
            expected = [
                (row["invoice"], date.fromisoformat(row["day"]), row["tenths"])
                for row in sorted(task.oracle["records"], key=lambda row: (row["day"], row["invoice"]))
            ]
            assert sorted(parsed) == sorted(expected), lang
            for other in PAIR_LANGS:
                if other == lang:
                    continue
                foreign = [month for month in PAIR_MONTHS[other] if month not in PAIR_MONTHS[lang]]
                assert not any(month in task.prompt for month in foreign), (lang, other)
            if lang == "en":
                assert "1,200.0 kg" in task.prompt
                assert "1 200,0" not in task.prompt
                assert "YYYY-MM-DD" in task.prompt and "(half up)" in task.prompt
            elif lang == "kk":
                assert "1 200,0 кг" in task.prompt
                assert "1,200.0" not in task.prompt
                assert "ЖЖЖЖ-АА-КК" in task.prompt
            else:
                assert "1 200,0 кг" in task.prompt
                assert "1,200.0" not in task.prompt
                assert "ГГГГ-ММ-ДД" in task.prompt


def test_if7_lang_pair_oracle_is_recomputed_in_every_language() -> None:
    for seed in DEV_SEEDS:
        for lang in PAIR_LANGS:
            task = _pair_task(seed, lang)
            picked = _pair_select(_pair_read_listing(_pair_listing(task), lang), task.oracle["window"])
            assert [
                (row["date"], row["invoice"], row["tonnes"]) for row in task.oracle["rows"]
            ] == [
                (day.isoformat(), unicodedata.normalize("NFC", invoice), _pair_tonnes(tenths))
                for invoice, day, tenths in picked
            ]
            assert sum((row["tonnes"] for row in task.oracle["rows"]), Decimal("0")) == task.oracle["total_tonnes"]


def test_if7_lang_pair_records_keep_both_unicode_forms() -> None:
    for seed in DEV_SEEDS:
        for lang in PAIR_LANGS:
            task = _pair_task(seed, lang)
            for identity in (2, 3):
                pair = [row for row in task.oracle["records"] if row["identity"] == identity]
                assert len(pair) == 2
                left, right = (unicodedata.normalize("NFC", row["invoice"]) for row in pair)
                assert left == right
                # The twins are printed in the two opposite forms, never alike.
                assert sorted(
                    row["invoice"] == unicodedata.normalize("NFC", row["invoice"]) for row in pair
                ) == [False, True]
                assert pair[0]["invoice"] != pair[1]["invoice"]
                assert sorted((row["decomposed"] for row in pair)) == [False, True]
                assert pair[0]["day"] < pair[1]["day"]
                winner = next(row for row in task.oracle["rows"] if row["invoice"] == left)
                assert winner["date"] == pair[1]["day"] == max(row["day"] for row in pair)
            assert task.oracle["decomposed_rows"] >= 4
            assert any(row["invoice"] != unicodedata.normalize("NFC", row["invoice"])
                       for row in task.oracle["records"])


def test_if7_lang_pair_reference_scores_full_and_garbage_zero() -> None:
    for seed in DEV_SEEDS:
        for lang in PAIR_LANGS:
            task = _pair_task(seed, lang)
            good = instruction.evaluate(task, _transcript(task, task.oracle["reference"]))
            assert good.earned == good.possible == 2.0
            assert _checks(good) == {"json_format": True, "rows": True, "total": True}
            assert instruction.evaluate(task, _transcript(task, "не знаю")).earned == 0.0
            assert instruction.evaluate(task, _transcript(task, "")).earned == 0.0


def test_if7_lang_pair_is_deterministic_per_seed() -> None:
    for lang in PAIR_LANGS:
        for seed in DEV_SEEDS:
            first, second = _pair_task(seed, lang), _pair_task(seed, lang)
            assert (first.prompt, first.oracle, first.meta) == (second.prompt, second.oracle, second.meta)
        assert _pair_task(17, lang).prompt != _pair_task(18, lang).prompt


def test_if7_lang_pair_is_a_stretch_scenario_only() -> None:
    def generated(profile: str) -> list[TaskInstance]:
        return instruction.generate(task_rng(1, "A"), RunConfig(profile=profile))

    assert not set(instruction.IF7_IDS) & {task.id for task in generated(core.DEFAULT_PROFILE)}
    assert not set(instruction.IF7_IDS) & {task.id for task in generated("pilot")}
    assert not set(instruction.IF7_IDS) & {task.id for task in generated("research-v1")}
    stretch = generated("stretch-v1")
    assert [task.id for task in stretch if task.id in instruction.IF7_IDS] == list(instruction.IF7_IDS)
    assert set(instruction.IF7_IDS) <= instruction.PRIVATE_IDS
    assert core.REGISTRY["A"].expected_ids_for("stretch-v1") == instruction.STRETCH_IDS


# --------------------------------------------------------------------------- #
# Language-pair mutants
# --------------------------------------------------------------------------- #


def test_lang_pair_translated_template_without_locale_check_fails() -> None:
    """One Russian rendering for every locale: ru parses, en and kk do not."""
    seed = 1
    for lang in ("en", "kk"):
        task = _pair_task(seed, lang)
        translated = [
            f"{row['invoice']}; {_pair_write_date(date.fromisoformat(row['day']), 'ru')}; "
            f"{_pair_write_mass(row['tenths'], 'ru')}"
            for row in task.oracle["records"]
        ]
        with pytest.raises(AssertionError):
            _pair_read_listing(translated, lang)

    ru_task = _pair_task(seed, "ru")
    left_as_ru = [
        f"{row['invoice']}; {_pair_write_date(date.fromisoformat(row['day']), 'ru')}; "
        f"{_pair_write_mass(row['tenths'], 'ru')}"
        for row in ru_task.oracle["records"]
    ]
    assert sorted(_pair_read_listing(left_as_ru, "ru")) == sorted(_pair_read_listing(_pair_listing(ru_task), "ru"))
    # And the English brief does state its own convention, so the reader can tell.
    assert "comma between thousands" in _pair_task(seed, "en").prompt
    assert "мыңдық разряд бос орынмен" in _pair_task(seed, "kk").prompt


def test_lang_pair_swapped_thousands_separator_fails() -> None:
    """Grouping is semantic: the value is right, its spelling belongs elsewhere."""
    for lang in PAIR_LANGS:
        task = _pair_task(2, lang)
        swap = "en" if lang != "en" else "ru"
        lines = [
            f"{row['invoice']}; {_pair_write_date(date.fromisoformat(row['day']), lang)}; "
            f"{_pair_write_mass(row['tenths'], swap)}"
            for row in task.oracle["records"]
        ]
        with pytest.raises(AssertionError):
            _pair_read_listing(lines, lang)
        # A lenient reader that ignores the declared locale reads another value.
        raw = lines[0].split("; ")[2]
        scrubbed = raw.replace(",", "") if swap == "en" else raw.replace(" ", "").replace(",", ".")
        assert int(Decimal(scrubbed) * 10) == task.oracle["records"][0]["tenths"]
        assert raw != _pair_write_mass(task.oracle["records"][0]["tenths"], lang)


def test_lang_pair_nfd_key_in_the_answer_loses_the_rows() -> None:
    task = _pair_task(1, "en")
    row = next(item for item in task.oracle["rows"] if item["invoice"] != unicodedata.normalize("NFD", item["invoice"]))
    payload = json.loads(task.oracle["reference"])
    payload["rows"][0]["invoice"] = unicodedata.normalize("NFD", row["invoice"])
    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert _checks(score) == {"json_format": True, "rows": False, "total": True}


def test_lang_pair_unmerged_nfd_twin_loses_the_rows() -> None:
    """A reader that never normalises sees two invoices instead of one record."""
    task = _pair_task(1, "ru")
    shadow = next(
        row for row in task.oracle["records"]
        if row["identity"] == 2 and row["day"] < max(
            other["day"] for other in task.oracle["records"] if other["identity"] == 2
        )
    )
    payload = json.loads(task.oracle["reference"])
    payload["rows"].append({
        "date": shadow["day"],
        "invoice": unicodedata.normalize("NFC", shadow["invoice"]),
        "tonnes": float(_pair_tonnes(shadow["tenths"])),
    })
    payload["rows"].sort(key=lambda item: (item["date"], item["invoice"]))
    payload["total_tonnes"] = round(payload["total_tonnes"] + float(_pair_tonnes(shadow["tenths"])), 3)
    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert _checks(score)["rows"] is False


def test_lang_pair_native_date_in_the_answer_loses_the_rows() -> None:
    for lang, sample in (
        ("en", "5 January 2026"),
        ("kk", "2026 жылғы 5 қаңтар"),
        ("ru", "5 января 2026 г."),
    ):
        task = _pair_task(3, lang)
        payload = json.loads(task.oracle["reference"])
        payload["rows"][0]["date"] = sample
        score = instruction.evaluate(task, _transcript(task, _answer(payload)))
        assert _checks(score)["rows"] is False
        assert score.earned < 2.0


def _if8(seed: int = 1) -> TaskInstance:
    return next(
        task
        for task in instruction.generate(task_rng(seed, "A"), RunConfig(profile="stretch-v1"))
        if task.id == "IF-8"
    )


def test_if8_nfd_keys_lose_keys_and_order() -> None:
    """The prompt asks for NFC keys; a decomposed spelling is not the oracle."""
    task = _if8()
    payload = json.loads(task.oracle["reference"])
    assert any(unicodedata.normalize("NFD", item["key"]) != item["key"] for item in payload["keys"])
    decomposed = {
        "keys": [
            {"key": unicodedata.normalize("NFD", item["key"]), "value": item["value"]}
            for item in payload["keys"]
        ]
    }
    score = instruction.evaluate(task, _transcript(task, _answer(decomposed)))
    assert _checks(score) == {"json_format": True, "keys": False, "order": False}

    oracle = instruction.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert oracle.earned == 2.0
    assert _checks(oracle) == {"json_format": True, "keys": True, "order": True}

