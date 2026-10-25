"""Tests for F-2 — three versioned sources decided by a trust rule.

The scenario is pilot-only: the baseline keeps ``F``'s facts untouched. Covers
the profile contract, seed determinism over five development seeds, the exact
oracle, the three target mutants (stale source, external knowledge, a
contradicting line outside the table) and garbage. The baseline block covers
its own closed ten-line protocol and the H16 mutants: an added hallucination,
an extra prose paragraph and a contradicting sentence must all stay below the
maximum, while the reference keeps 8/8.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

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
    assert module.expected_ids_for("stretch-v1") == ("F", "F-2", "F-3")
    assert module.points == 8.0

    assert [task.id for task in _tasks(profile=core.DEFAULT_PROFILE)] == ["F"]
    assert [task.id for task in _tasks(profile="pilot")] == ["F-2"]
    assert [task.id for task in _tasks(profile="stretch-v1")] == ["F", "F-2", "F-3"]


def test_f2_task_declares_eight_points_and_its_own_title() -> None:
    task = _task()
    assert task.category == "F"
    assert task.max_points == 8.0
    assert task.title == "Versioned sources and trust rule"
    assert task.meta["scenario"] == "F-2"


def test_stretch_splits_the_category_total_over_every_task() -> None:
    tasks = _tasks(profile="stretch-v1")
    assert [task.id for task in tasks] == ["F", "F-2", "F-3"]
    apply_profile_weights(tasks, "stretch-v1")
    assert [task.max_points for task in tasks] == [2.6667, 2.6667, 2.6666]
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


def _baseline_task(seed: int = DEV_SEEDS[0]) -> TaskInstance:
    return next(task for task in _tasks(seed, core.DEFAULT_PROFILE) if task.id == "F")


def _baseline_score(task: TaskInstance, suffix: str) -> core.Score:
    return hallucination.evaluate(task, _transcript(task, task.oracle["reference"] + suffix))


def test_baseline_f_still_scores_full() -> None:
    task = _baseline_task()
    score = hallucination.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == task.max_points == 8.0
    assert len(score.checks) == 11
    assert "closed_format" in {check.name for check in score.checks}


def test_an_added_hallucination_line_is_not_full() -> None:
    """H16: a fabricated fact outside the ten lines used to earn 8/8."""
    task = _baseline_task()
    gpu = next(q for q in task.oracle["questions"] if "GPU" in q["text"])
    score = _baseline_score(task, "\n11. GPU workers используют H100.")

    assert score.earned < 8.0
    assert _failed(score) == [f"q{gpu['index']}", "closed_format"]
    assert "H100" in score.checks[gpu["index"] - 1].detail


def test_an_extra_prose_paragraph_is_not_full() -> None:
    task = _baseline_task()
    score = _baseline_score(task, "\n\nИтого: все ответы взяты из документа.")

    assert _failed(score) == ["closed_format"]
    assert score.earned == 7.0


def test_correct_answers_with_a_contradiction_are_not_full() -> None:
    task = _baseline_task()
    question = next(q for q in task.oracle["questions"] if "concurrency" in q["text"])
    other = next(value for value in (2, 3, 4, 5, 6, 8) if str(value) != question["answer"])
    score = _baseline_score(task, f"\nConcurrency равна {other}.")

    assert f"q{question['index']}" in _failed(score)
    assert "closed_format" in _failed(score)
    assert score.earned < 8.0


def test_contradicting_the_unknown_answer_costs_only_that_fact() -> None:
    task = _baseline_task()
    question = next(q for q in task.oracle["questions"] if q["kind"] == "unknown")
    line = f"{question['index']}. KNOWN: {question['tempting'][0]}"
    lines = task.oracle["reference"].splitlines()
    lines[question["index"] - 1] = line

    score = hallucination.evaluate(task, _transcript(task, "\n".join(lines)))
    assert _failed(score) == [f"q{question['index']}"]
    assert score.earned == 7.3


def test_blank_lines_between_answers_are_allowed() -> None:
    task = _baseline_task()
    spaced = "\n\n".join(task.oracle["reference"].splitlines())
    score = hallucination.evaluate(task, _transcript(task, spaced))
    assert _failed(score) == []


def test_a_duplicated_number_breaks_the_closed_grammar() -> None:
    task = _baseline_task()
    duplicated = task.oracle["reference"] + "\n2. KNOWN: лишнее"
    score = hallucination.evaluate(task, _transcript(task, duplicated))

    assert "closed_format" in _failed(score)
    assert score.earned < 8.0


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
    assert [task.id for task in stretch] == ["F", "F-2", "F-3"]
    assert stretch[0].prompt == base.prompt
    assert stretch[0].oracle["reference"] == base.oracle["reference"]


# --------------------------------------------------------------------------- #
# F-3 / N6 — incident journal, corrections, revocation (stretch)
# --------------------------------------------------------------------------- #

_ENTRY = re.compile(r"^- (E\d+) (\w+): (.*?) \(seq: (\d+)\)$", re.MULTILINE)
_CORRECTION = re.compile(r"^- (C\d+) correction supersedes: (E\d+) \(seq: \d+\): (\w+) = (\S+)$", re.MULTILINE)
_RANK = {"primary": 2, "secondary": 1, "tertiary": 0}


def _journal(seed: int = DEV_SEEDS[0]) -> TaskInstance:
    tasks = hallucination.generate(task_rng(seed, "F"), RunConfig(seed=seed, profile="stretch-v1"))
    return next(task for task in tasks if task.id == "F-3")


@dataclass(frozen=True)
class _Record:
    document: str
    authority: str
    entry: str
    field: str
    value: str
    seq: int
    revoked: bool


def _records(task: TaskInstance) -> list[_Record]:
    """Re-read the journal from the prompt text, not from the module."""
    found: list[_Record] = []
    for name, document in task.oracle["documents"].items():
        authority = re.search(r"^authority: (\w+)$", document, re.MULTILINE).group(1)
        revoked = "revoked: true" in document
        for match in _CORRECTION.finditer(document):
            found.append(_Record(name, authority, match.group(1), match.group(3), match.group(4), -1, revoked))
        for match in _ENTRY.finditer(document):
            found.append(_Record(name, authority, match.group(1), match.group(2), match.group(3), int(match.group(4)), revoked))
    return found


def _journal_rule(task: TaskInstance, field: str) -> tuple[str, list[str]]:
    records = _records(task)
    superseded = {
        match.group(2) for document in task.oracle["documents"].values() for match in _CORRECTION.finditer(document)
    }
    live = [
        record for record in records
        if record.field == field and not record.revoked and record.entry not in superseded
    ]
    if live:
        winner = max(live, key=lambda record: (_RANK[record.authority], record.seq))
        return winner.value, [f"{winner.document}/{winner.entry}"]
    mentioned = sorted({record.document for record in records if record.field == field})
    return "UNKNOWN", mentioned or sorted(task.oracle["documents"])


def _answer(task: TaskInstance, overrides: dict[str, tuple[str, list[str]]] | None = None) -> str:
    items = []
    for question in sorted(task.oracle["questions"], key=lambda q: q["index"]):
        value, sources = (overrides or {}).get(question["field"], (question["value"] or "UNKNOWN", question["sources"]))
        items.append({"id": question["index"], "value": value, "sources": sources})
    return json.dumps({"answers": items}, ensure_ascii=False, separators=(",", ":"))


def _index_of(task: TaskInstance, field: str) -> int:
    return next(q["index"] for q in task.oracle["questions"] if q["field"] == field)


def test_n6_is_a_stretch_scenario_with_its_own_point_budget() -> None:
    module = core.REGISTRY["F"]
    assert module.expected_ids == ("F",)
    assert module.expected_ids_for("pilot") == ("F-2",)
    assert module.expected_ids_for("stretch-v1") == ("F", "F-2", "F-3")
    assert "F-3" not in [task.id for task in _tasks(profile="pilot")]
    assert [task.id for task in _tasks(profile=core.DEFAULT_PROFILE)] == ["F"]

    task = _journal()
    assert task.title == "Incident journal, corrections and revocation"
    assert task.category == "F"
    assert task.max_points == 8.0
    assert task.languages == ["ru"]
    assert task.fresh_after == "2026-10-01"
    assert task.meta["scenario"] == "F-3"


def test_n6_generation_is_deterministic_but_seeded() -> None:
    first, second, other = _journal(DEV_SEEDS[0]), _journal(DEV_SEEDS[0]), _journal(DEV_SEEDS[1])
    assert first.prompt == second.prompt
    assert first.oracle == second.oracle
    assert first.oracle["reference"] != other.oracle["reference"]


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n6_fixture_has_corrections_revocation_and_two_unresolvable_fields(seed: int) -> None:
    task = _journal(seed)
    questions = task.oracle["questions"]
    documents = task.oracle["documents"]

    assert len(questions) == 6
    assert [q["index"] for q in questions] == list(range(1, 7))
    assert len({q["field"] for q in questions}) == 6
    assert task.oracle["unknown_count"] == 2
    assert sorted(documents) == ["D1", "D2", "D3", "D4"]

    revoked = [name for name, document in documents.items() if "revoked: true" in document]
    assert len(revoked) == 1, "отозван ровно один документ"
    assert sum(len(_CORRECTION.findall(document)) for document in documents.values()) == 1

    seqs = [record.seq for record in _records(task) if record.seq >= 0]
    assert len(seqs) == len(set(seqs)), "seq обязаны быть уникальными"

    for question in questions:
        assert question["sources"]
        if question["kind"] == "known":
            assert question["value"] and question["value"] != "UNKNOWN"
            assert len(question["sources"]) == 1
            assert re.fullmatch(r"D\d/[EC]\d+", question["sources"][0]), question["sources"]
        else:
            assert question["value"] is None
            assert set(question["sources"]) <= set(documents)

    upload = next(q for q in questions if q["field"] == "max_upload_mib")
    assert upload["kind"] == "unknown"
    assert upload["sources"] == revoked, "поле живёт только в отозванном документе"
    assert next(q for q in questions if q["field"] == "backup_region")["sources"] == sorted(documents)


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n6_oracle_matches_an_independent_rule_application(seed: int) -> None:
    task = _journal(seed)
    for question in task.oracle["questions"]:
        value, sources = _journal_rule(task, question["field"])
        assert value == (question["value"] or "UNKNOWN"), question["field"]
        assert sources == question["sources"], question["field"]


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n6_reference_scores_the_full_budget(seed: int) -> None:
    task = _journal(seed)
    score = hallucination.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert _failed(score) == []
    assert score.earned == task.max_points == 8.0
    assert len(score.checks) == 8
    assert sum(check.points for check in score.checks) == task.max_points


def test_n6_garbage_scores_zero() -> None:
    task = _journal()
    run = Runner(RunConfig(profile="stretch-v1"), GarbageProvider()).run_task(task)
    assert run.score.earned == 0.0
    assert set(_failed(run.score)) >= {"json_shape", "no_extra_text", "q1", "q6"}


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n6_a_value_from_the_revoked_document_is_not_full(seed: int) -> None:
    task = _journal(seed)
    score = hallucination.evaluate(task, _transcript(task, task.oracle["mutant_revoked"]))
    assert score.earned < 8.0
    assert {f"q{_index_of(task, field)}" for field in ("max_upload_mib", "worker_replicas")} <= set(_failed(score))
    assert "json_shape" not in _failed(score)


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n6_a_similar_number_transfer_is_not_full(seed: int) -> None:
    task = _journal(seed)
    score = hallucination.evaluate(task, _transcript(task, task.oracle["mutant_similar"]))
    assert score.earned < 8.0
    retention = _index_of(task, "retention_days")
    ttl = _index_of(task, "session_ttl_min")
    assert {f"q{retention}", f"q{ttl}"} <= set(_failed(score))
    cited = json.loads(task.oracle["mutant_similar"])["answers"][retention - 1]["sources"]
    superseded = next(
        match.group(2) for document in task.oracle["documents"].values() for match in _CORRECTION.finditer(document)
    )
    assert cited == [f"D2/{superseded}"], "мутант обязан опираться на отменённую запись"


def test_n6_correct_fields_plus_an_unsupported_claim_are_not_full() -> None:
    task = _journal()
    score = hallucination.evaluate(task, _transcript(task, task.oracle["mutant_unsupported"]))
    assert _failed(score) == ["json_shape"]
    assert all(check.passed for check in score.checks[:6]), "шесть полей остаются верными"
    assert score.earned == 7.0
    assert any("лишние ключи" in note for note in score.notes)


def test_n6_a_line_outside_the_json_is_not_full() -> None:
    task = _journal()
    score = hallucination.evaluate(task, _transcript(task, task.oracle["mutant_trailing"]))
    assert _failed(score) == ["json_shape", "no_extra_text"]
    assert all(check.passed for check in score.checks[:6])
    assert score.earned == 6.0


def test_n6_a_wrong_basis_loses_only_that_fact() -> None:
    task = _journal()
    index = _index_of(task, "worker_replicas")
    question = next(q for q in task.oracle["questions"] if q["index"] == index)
    wrong = next(
        f"{record.document}/{record.entry}"
        for record in _records(task)
        if record.field == question["field"] and f"{record.document}/{record.entry}" != question["sources"][0]
    )
    score = hallucination.evaluate(task, _transcript(task, _answer(task, {question["field"]: (question["value"], [wrong])})))
    assert _failed(score) == [f"q{index}"]
    assert score.earned == 7.0
    assert "основание" in score.checks[index - 1].detail


def test_n6_a_guess_on_an_unresolvable_field_loses_only_that_fact() -> None:
    task = _journal()
    index = _index_of(task, "max_upload_mib")
    score = hallucination.evaluate(task, _transcript(task, _answer(task, {"max_upload_mib": ("256", ["D3"])})))
    assert _failed(score) == [f"q{index}"]
    assert score.earned == 7.0


def test_n6_json_around_text_keeps_the_facts_but_loses_the_format() -> None:
    task = _journal()
    wrapped = f"Готово:\n{task.oracle['reference']}\nКонец."
    score = hallucination.evaluate(task, _transcript(task, wrapped))
    assert _failed(score) == ["json_shape", "no_extra_text"]
    assert all(check.passed for check in score.checks[:6])

    fenced = hallucination.evaluate(task, _transcript(task, f"```json\n{task.oracle['reference']}\n```"))
    assert _failed(fenced) == []
    assert fenced.earned == 8.0


def test_n6_a_missing_or_shuffled_answer_breaks_the_shape() -> None:
    task = _journal()
    payload = json.loads(task.oracle["reference"])
    payload["answers"] = payload["answers"][:-1]
    missing = hallucination.evaluate(task, _transcript(task, json.dumps(payload)))
    assert "json_shape" in _failed(missing)
    assert "no_extra_text" not in _failed(missing)

    shuffled = json.loads(task.oracle["reference"])
    shuffled["answers"][0], shuffled["answers"][1] = shuffled["answers"][1], shuffled["answers"][0]
    reordered = hallucination.evaluate(task, _transcript(task, json.dumps(shuffled)))
    assert "json_shape" in _failed(reordered)


def test_n6_prompt_carries_the_rule_and_no_answers() -> None:
    task = _journal()
    assert "revoked: true" in task.prompt
    assert "authority: primary" in task.prompt
    assert "correction supersedes" in task.prompt
    assert '"answers"' in task.prompt
    assert "мутант" not in task.prompt
    for question in task.oracle["questions"]:
        assert f"{question['index']}. {question['text']}" in task.prompt
        for source in question["sources"]:
            if "/" in source:
                assert source not in task.prompt
    assert task.oracle["reference"] not in task.prompt


def test_n6_registered_evaluator_is_the_dispatched_one() -> None:
    task = _journal()
    score = core.REGISTRY["F"].evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == 8.0
    assert [check.name for check in score.checks] == [f"q{i}" for i in range(1, 7)] + ["json_shape", "no_extra_text"]
    assert isinstance(core.REGISTRY["F"].evaluate(task, _transcript(task, "не знаю")), core.Score)
