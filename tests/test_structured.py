from __future__ import annotations

import copy
import json

import pytest

from dailybench import ModelReply, RunConfig, Score, TaskInstance, Transcript, Turn, core
from dailybench.rng import task_rng
from dailybench.tasks import structured
from dailybench.tasks.structured import check_schema, evaluate, generate, validate


def _transcript(task: TaskInstance, answer: object) -> Transcript:
    """A single-turn run with ``answer`` as the visible reply text."""
    return Transcript(
        task_id=task.id,
        category=task.category,
        turns=[
            Turn(
                messages=[{"role": "user", "content": task.prompt}],
                reply=ModelReply(text=answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)),
            )
        ],
    )


def _checks(score: Score) -> dict[str, bool]:
    return {check.name: check.passed for check in score.checks}


def _tasks(seed: int = 20260915) -> list[TaskInstance]:
    return generate(task_rng(seed, "G"), RunConfig())


# --------------------------------------------------------------------------- #
# Contract: ids, points, registry
# --------------------------------------------------------------------------- #


def test_structured_ids_points_and_registry() -> None:
    tasks = _tasks()
    assert [task.id for task in tasks] == ["G-1", "G-2", "G-3", "G-4"]
    assert tasks[0].id == structured.IDS[0]
    for task in tasks:
        assert task.category == "G"
        assert task.max_points == 2.0
    assert sum(task.max_points for task in tasks) == 8.0

    module = core.REGISTRY["G"]
    assert module.title == "Structured output"
    assert module.points == 8.0
    assert module.expected_ids == ("G-1", "G-2", "G-3", "G-4")
    assert core.CATEGORY_POINTS["G"] == module.points == 8.0


def test_structured_every_task_has_two_one_point_checks() -> None:
    for task in _tasks():
        score = evaluate(task, _transcript(task, "не знаю"))
        assert score.possible == 2.0
        assert {check.points for check in score.checks} == {1.0}
        assert [check.name for check in score.checks] == ["schema", "content"]
        assert len(score.checks) == 2


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_structured_is_deterministic_and_seeded() -> None:
    first = generate(task_rng(17, "G"), RunConfig())
    second = generate(task_rng(17, "G"), RunConfig())
    other = generate(task_rng(18, "G"), RunConfig())
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle for task in first] == [task.oracle for task in second]
    assert [task.id for task in first] == [task.id for task in other]
    assert any(a.prompt != b.prompt for a, b in zip(first, other))
    assert any(a.oracle["reference"] != b.oracle["reference"] for a, b in zip(first, other))


# --------------------------------------------------------------------------- #
# Reference and garbage baselines
# --------------------------------------------------------------------------- #


def test_structured_reference_scores_full_and_garbage_zero() -> None:
    for task in _tasks(3):
        good = evaluate(task, _transcript(task, task.oracle["reference"]))
        assert good.earned == 2.0
        assert good.possible == 2.0
        assert _checks(good) == {"schema": True, "content": True}
        assert evaluate(task, _transcript(task, "не знаю")).earned == 0.0
        assert evaluate(task, _transcript(task, "")).earned == 0.0


def test_structured_fixture_claims_hold() -> None:
    for task in _tasks(5):
        schema = task.oracle["schema"]
        answer = task.oracle["answer"]
        check_schema(schema)  # the schema uses only the supported subset
        assert validate(answer, schema) == []
        assert json.loads(task.oracle["reference"]) == answer
        assert task.oracle["distractor"] != answer
    # G-4's third rule is only observable while the optional field stays absent.
    g4 = _tasks(5)[3]
    assert "timeout" not in g4.oracle["answer"]
    # G-3's distractor is a well-formed array in the wrong order.
    g3 = _tasks(5)[2]
    assert validate(g3.oracle["distractor"], g3.oracle["schema"]) == []


# --------------------------------------------------------------------------- #
# Traps: one per scenario, with the exact point split
# --------------------------------------------------------------------------- #


def test_structured_g1_fenced_json_loses_schema_keeps_content() -> None:
    task = _tasks()[0]
    fenced = "```json\n" + task.oracle["reference"] + "\n```"
    score = evaluate(task, _transcript(task, fenced))
    assert score.earned == 1.0
    assert score.possible == 2.0
    assert _checks(score) == {"schema": False, "content": True}
    # One extra field is the distractor the fixture warns about: both points go.
    extra = copy.deepcopy(task.oracle["answer"])
    extra["note"] = "заполнено автоматически"
    assert evaluate(task, _transcript(task, extra)).earned == 0.0


def test_structured_g2_wrong_enum_fails_both() -> None:
    task = _tasks()[1]
    broken = copy.deepcopy(task.oracle["answer"])
    broken[0]["severity"] = broken[0]["severity"].lower()  # e.g. "error", not in the enum
    assert broken[0]["severity"] not in task.oracle["schema"]["items"]["properties"]["severity"]["enum"]
    score = evaluate(task, _transcript(task, broken))
    assert score.earned == 0.0
    assert _checks(score) == {"schema": False, "content": False}
    # A wrong date format is also rejected by the pattern.
    undated = copy.deepcopy(task.oracle["answer"])
    undated[0]["ts"] = "04.03.2026 09:15"
    assert evaluate(task, _transcript(task, undated)).earned == 0.0


def test_structured_g3_unsorted_array_keeps_schema_loses_content() -> None:
    task = _tasks()[2]
    answer = task.oracle["answer"]
    unsorted = list(reversed(answer))
    assert unsorted != answer  # the reversal really moved at least two items
    assert validate(unsorted, task.oracle["schema"]) == []
    score = evaluate(task, _transcript(task, unsorted))
    assert score.earned == 1.0
    assert _checks(score) == {"schema": True, "content": False}
    # Duplicates blow the maxItems bound as well as the content.
    duplicated = answer + [answer[0]]
    assert evaluate(task, _transcript(task, duplicated)).earned == 0.0


def test_structured_g4_uncoerced_field_fails_both() -> None:
    task = _tasks()[3]
    broken = copy.deepcopy(task.oracle["answer"])
    broken["port"] = str(broken["port"])  # rule 1 forgotten: still a string
    score = evaluate(task, _transcript(task, broken))
    assert score.earned == 0.0
    assert _checks(score) == {"schema": False, "content": False}
    # An unknown key is dropped by the schema check too.
    leaking = copy.deepcopy(task.oracle["answer"])
    leaking["comment"] = "черновик"
    assert evaluate(task, _transcript(task, leaking)).earned == 0.0


# --------------------------------------------------------------------------- #
# The schema validator itself
# --------------------------------------------------------------------------- #


def test_validator_types_and_enum() -> None:
    assert validate(1, {"type": "integer"}) == []
    assert validate(True, {"type": "integer"}) != []  # bool is not an integer
    assert validate(2.5, {"type": "integer"}) != []
    assert validate(2.5, {"type": "number"}) == []
    assert validate(2, {"type": "number"}) == []
    assert validate("x", {"type": "string"}) == []
    assert validate(None, {"type": "null"}) == []
    assert validate([], {"type": "array"}) == []
    assert validate({}, {"type": "object"}) == []
    assert validate("b", {"type": "string", "enum": ["a", "b"]}) == []
    assert validate("c", {"type": "string", "enum": ["a", "b"]}) != []


def test_validator_structure_and_bounds() -> None:
    schema = {
        "type": "object",
        "required": ["id", "items"],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "pattern": r"^svc-\d{3}$"},
            "items": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "integer", "minimum": 0}},
        },
    }
    assert validate({"id": "svc-042", "items": [0, 1]}, schema) == []
    assert validate({"id": "svc-42", "items": [0]}, schema) != []      # pattern
    assert validate({"id": "svc-042"}, schema) != []                 # required
    assert validate({"id": "svc-042", "items": []}, schema) != []     # minItems
    assert validate({"id": "svc-042", "items": [0, 1, 2]}, schema) != []  # maxItems
    assert validate({"id": "svc-042", "items": [-1]}, schema) != []   # minimum
    assert validate({"id": "svc-042", "items": [0], "extra": 1}, schema) != []  # additionalProperties
    assert validate({"id": "svc-042", "items": ["0"]}, schema) != []  # nested type
    # Nested paths are reported so the evaluator can quote the first error.
    assert validate({"id": "svc-42", "items": [0]}, schema)[0].startswith("$.id")


def test_check_schema_rejects_unsupported_keywords() -> None:
    with pytest.raises(ValueError):
        check_schema({"type": "object", "patternProperties": {}})
    with pytest.raises(ValueError):
        check_schema({"type": "date"})
    with pytest.raises(ValueError):
        check_schema({"type": "array", "items": {"type": "string", "format": "uuid"}})
