"""Task/profile contract: defaults, rejection and a full export round-trip.

The schema layer is the gate the corpus passes through: a direction, family,
language tag or denominator that the report relies on must be valid before a
scenario can run. Defaults are asserted as values, not as incidental dataclass
behaviour, because ``block`` and ``family`` are derived from the category.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dailybench import cli, core, schema
from dailybench.config import RunConfig
from dailybench.suite import build_suite, public_prompts


def _det(**overrides):
    task = core.TaskInstance("R-99", "B", "Reasoning", "Реши задачу", 2.0)
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def _open(**overrides):
    task = core.TaskInstance("K-99", "K", "Scene", "Нарисуй сцену", 1.0)
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def test_derived_defaults_are_the_contract():
    det = core.TaskInstance("R-99", "B", "Reasoning", "p", 2.0)
    assert (det.block, det.family) == ("deterministic", "det/math")
    assert det.languages == ["ru"]
    assert det.cultures == []
    assert det.fresh_after is None
    assert det.private is False
    assert det.holdout_seeds == []

    opened = core.TaskInstance("K-99", "K", "Scene", "p", 1.0)
    assert (opened.block, opened.family) == ("open", "vis/sim")

    # An empty language list still means the default brief language.
    assert core.TaskInstance("R-100", "B", "R", "p", 1.0, languages=[]).languages == ["ru"]


def test_valid_deterministic_and_open_tasks_pass():
    det = _det(languages=["ru", "kk"], cultures=["kz"], fresh_after="2026-07-01", holdout_seeds=[9901])
    assert schema.validate_task(det) == []
    assert schema.validate_task(_open(languages=["ru"])) == []


@pytest.mark.parametrize("block", ["nonsense", "Deterministic", "", None])
def test_invalid_block_is_rejected(block):
    assert any("block" in error for error in schema.validate_task(_det(block=block)))


@pytest.mark.parametrize("family", ["det/nope", "math", "vis/math", ""])
def test_invalid_family_is_rejected(family):
    assert any("family" in error for error in schema.validate_task(_det(family=family)))


def test_block_and_family_must_agree():
    assert any("family" in error for error in schema.validate_task(_det(family="vis/art")))
    assert any("family" in error for error in schema.validate_task(_open(family="det/code")))


def test_block_must_match_the_category():
    assert any("block" in error for error in schema.validate_task(_det(block="open")))


@pytest.mark.parametrize("points", [0, -1.5, "2", True])
def test_invalid_points_are_rejected(points):
    errors = schema.validate_task(
        {
            "schema": schema.SCHEMA_VERSION,
            "id": "R-99",
            "category": "B",
            "block": "deterministic",
            "family": "det/math",
            "points": points,
            "prompt": "p",
        }
    )
    assert any("points" in error for error in errors)


@pytest.mark.parametrize("languages", [[], [""], "ru", ["ru", "ru"], ["t"]])
def test_invalid_languages_are_rejected(languages):
    assert any("languages" in error for error in schema.validate_task(_det(languages=languages)))


def test_task_instance_refuses_invalid_values_at_construction():
    with pytest.raises(ValueError, match="block"):
        core.TaskInstance("R-99", "B", "R", "p", 1.0, block="open")
    with pytest.raises(ValueError, match="points"):
        core.TaskInstance("R-99", "B", "R", "p", 0.0)


def test_private_prompt_stays_out_of_the_public_view():
    tasks = [_det(), _det(private=True), _open()]
    public = public_prompts(tasks)
    private = next(entry for entry in public if entry["id"] == "R-99" and entry.get("private"))
    assert "prompt" not in private and "prompt_sha256" in private
    assert [entry["id"] for entry in public] == ["R-99", "R-99", "K-99"]
    assert all("prompt" in entry for entry in public if not entry.get("private"))

    opened = public_prompts(tasks, include_private=True)
    assert all("prompt" in entry for entry in opened)
    assert next(entry for entry in opened if entry.get("private"))["prompt"] == "Реши задачу"


def test_digest_marks_private_and_new_fields():
    from dailybench.suite import suite_digest

    digest = suite_digest([_det(private=True, languages=["ru", "kk"], holdout_seeds=[7])], "pilot")
    row = digest["per_task"][0]
    assert row["private"] is True
    assert row["block"] == "deterministic"
    assert row["family"] == "det/math"
    assert row["languages"] == ["ru", "kk"]
    assert row["holdout_seeds"] == [7]


def test_export_round_trip_validates(tmp_path: Path):
    target = tmp_path / "suite.json"
    assert cli.main(["export", "--format", "json", "--out", str(target), "--only", "R-1,K-1"]) == 0
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert schema.validate_suite_doc(doc, check_totals=False) == []
    assert cli.main(["validate", "--suite", str(target), "--no-totals"]) == 0


def test_yaml_export_round_trip(tmp_path: Path):
    import yaml

    target = tmp_path / "suite.yaml"
    assert cli.main(["export", "--format", "yaml", "--out", str(target), "--only", "R-1"]) == 0
    doc = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert schema.validate_suite_doc(doc, check_totals=False) == []


def test_export_turns_private_tasks_into_hash_stubs(tmp_path: Path):
    tasks = build_suite(RunConfig(only=("R-1", "K-1")))
    assert not any(task.private for task in tasks)

    first = next(task for task in tasks if task.id == "R-1")
    first.private = True
    hidden = schema.export_document("pilot", tasks)
    assert [task["id"] for task in hidden["tasks"]] == ["R-1", "K-1"]
    stub = hidden["tasks"][0]
    assert stub["private"] is True and stub["prompt"] is None
    assert len(stub["prompt_sha256"]) == 64 and "oracle" not in stub
    assert schema.validate_suite_doc(hidden) == []

    shown = schema.export_document("pilot", tasks, include_private=True)
    full = shown["tasks"][0]
    assert full["prompt"] == first.prompt and "oracle" in full
    assert schema.validate_suite_doc(shown) == []


def test_exported_private_denominator_stays_whole(tmp_path: Path):
    tasks = build_suite(RunConfig(profile="pilot", only=("IF-6",)))
    for task in tasks:
        task.private = True
    doc = schema.export_document("pilot", tasks)
    assert doc["tasks"][0]["points"] == 10.0
    errors = schema.validate_suite_doc(doc, check_totals=True)
    assert errors == [], errors


def test_profile_document_is_valid_and_adds_up():
    doc = schema.profile_document(core.DEFAULT_PROFILE)
    assert schema.validate_profile(doc) == []

    doc["total"] += 1
    assert any("total" in error for error in schema.validate_profile(doc))


def test_exported_suite_reports_a_broken_category_total():
    task = _det()
    doc = schema.export_document("pilot", [task])
    assert schema.validate_suite_doc(doc, check_totals=True) != []


@pytest.mark.parametrize("payload", [None, [], ["A-1"], "задача", 5, True])
def test_non_mapping_document_is_an_error_and_not_a_crash(payload):
    errors = schema.validate_task(payload)
    assert errors, payload
    assert "expected object" in errors[0], errors[0]


def test_validate_reports_empty_and_scalar_yaml_without_a_traceback(tmp_path: Path, capsys):
    for name, text in (("empty.yaml", ""), ("scalar.yaml", "просто строка\n"), ("list.yaml", "- A-1\n")):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert cli.main(["validate", "--task", str(path)]) == 1
        assert "expected object" in capsys.readouterr().err


def test_validate_reports_broken_yaml_as_an_error(tmp_path: Path, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text("id: [A-1\n", encoding="utf-8")
    assert cli.main(["validate", "--task", str(path)]) == 1
    assert "YAML" in capsys.readouterr().err


def test_document_size_cap_is_enforced(tmp_path: Path, capsys, monkeypatch):
    path = tmp_path / "task.yaml"
    path.write_text("id: A-1\n", encoding="utf-8")
    monkeypatch.setattr(cli, "MAX_DOCUMENT_BYTES", 4)
    with pytest.raises(ValueError, match="байт"):
        cli._read_document(path)
    assert cli.main(["validate", "--task", str(path)]) == 1
    assert "больше" in capsys.readouterr().err


def test_oversized_fields_are_rejected():
    assert any("longer than" in error for error in schema.validate_task(_det(title="т" * 201)))
    assert any("longer than" in error for error in schema.validate_task(_det(prompt="п" * 262145)))
    assert any("more than" in error for error in schema.validate_task(_det(languages=["ru"] * 33)))
    assert any("more than" in error for error in schema.validate_task(_det(holdout_seeds=list(range(65)))))


def test_id_uses_the_same_file_name_contract_as_the_gallery():
    from dailybench import gallery

    assert schema.validate_task(_det(id="A-1")) == []
    errors = schema.validate_task(_det(id="A-1/x"))
    assert any("invalid task id" in error and "K-1" in error for error in errors), errors
    assert schema.task_schema()["properties"]["id"]["pattern"] == gallery._ID_SCHEMA.pattern


def test_sort_key_tolerates_an_id_without_a_dash():
    from dailybench import htmlreport

    assert htmlreport._task_sort_key("AB") == (0, "AB")
    assert htmlreport._task_sort_key("K-12") == (12, "K-12")
    assert htmlreport._task_sort_key("K-12-b") == (12, "K-12-b")


def test_private_stub_accepts_a_mapping_and_a_null_prompt():
    doc = schema.task_to_dict(_det(private=True))
    doc["prompt"] = None
    stub = schema._private_stub(doc)
    assert stub["prompt"] is None and len(stub["prompt_sha256"]) == 64
    assert schema.validate_task(stub) == []


def test_export_stubs_a_private_mapping():
    doc = schema.task_to_dict(_det(prompt=None, private=True))
    doc["prompt_sha256"] = "0" * 64
    suite = schema.export_document("pilot", [doc])
    assert suite["tasks"][0]["prompt"] is None
    assert schema.validate_suite_doc(suite) == []
