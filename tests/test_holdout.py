"""Holdout splits and the rights registry.

The suite has to prove three things before a closed comparison is credible:
closed material never appears in the public export, any edit to a frozen input
breaks the manifest, and an external record without a licence refuses the
import. The tests drive the real CLI and the real export filter; fixtures are
generated tasks, not hand-written JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dailybench import cli, holdout, schema
from dailybench.adapters import bigcodebench
from dailybench.config import RunConfig
from dailybench.suite import build_suite

IDS = ("C-1", "C-2", "C-3", "T-1", "T-2", "LC-1", "K-1")
PINNED = {"C-1": "closed-eval", "C-2": "audit", "C-3": "public-dev"}


def _tasks(ids=IDS):
    return build_suite(RunConfig(only=ids))


def _manifest(tasks, **overrides):
    return holdout.build_manifest(tasks, profile="pilot", seed=11, overrides={**PINNED, **overrides})


def _record(**overrides):
    record = {
        "schema": holdout.LICENSE_SCHEMA_VERSION,
        "id": "material",
        "origin": "https://example.com/dataset",
        "revision": "abc123",
        "upstream_id": "org/dataset@abc123:test",
        "code_license": "Apache-2.0",
        "data_license": "CC-BY-4.0",
        "attribution": "Example project",
        "redistribution": "allowed",
        "status": "verified",
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# Assignment and manifest
# --------------------------------------------------------------------------- #


def test_assignment_is_stable_and_disjoint():
    tasks = _tasks()
    first = _manifest(tasks)
    second = _manifest(_tasks())
    assert first == second
    assert first["digest"] == second["digest"]
    ids = [entry["id"] for name in holdout.SPLITS for entry in first["splits"][name]]
    assert sorted(ids) == sorted(IDS)
    assert len(ids) == len(set(ids))
    assert "C-1" in [entry["id"] for entry in first["splits"]["closed-eval"]]
    for identifier, split in PINNED.items():
        assert identifier in [entry["id"] for entry in first["splits"][split]]
    assert holdout.validate_manifest(first) == []


def test_hash_buckets_are_keyed_by_task_id_only():
    first = holdout.split_for("R-1")
    assert all(holdout.split_for("R-1") == first for _ in range(5))
    assert holdout.split_for("R-1", closed_share=0.0, audit_share=0.0) == "public-dev"
    assert holdout.split_for("R-1", closed_share=1.0, audit_share=0.0) == "closed-eval"


def test_unknown_override_and_bad_shares_are_rejected():
    with pytest.raises(ValueError, match="неизвестных"):
        holdout.build_manifest(_tasks(), profile="pilot", overrides={"ZZ-9": "audit"})
    with pytest.raises(ValueError, match="доли"):
        holdout.build_manifest(_tasks(), profile="pilot", closed_share=0.8, audit_share=0.5)


def test_manifest_requires_both_a_public_and_a_closed_part():
    tasks = _tasks()
    every_closed = holdout.build_manifest(
        tasks, profile="pilot", overrides={identifier: "closed-eval" for identifier in IDS}
    )
    assert any("public-dev" in error for error in holdout.validate_manifest(every_closed))


# --------------------------------------------------------------------------- #
# The export gate
# --------------------------------------------------------------------------- #


def test_closed_material_never_reaches_the_export():
    tasks = _tasks()
    tasks[0].holdout_seeds = [9901]
    manifest = _manifest(tasks)
    doc = schema.export_document("pilot", tasks, hidden=holdout.hidden_ids(manifest))
    assert holdout.check_export(doc, manifest) == []
    assert schema.validate_suite_doc(doc) == []

    by_id = {task["id"]: task for task in doc["tasks"]}
    closed = by_id["C-1"]
    assert closed["prompt"] is None
    assert closed["private"] is True
    assert closed["split"] == "closed-eval"
    assert closed["holdout_seed_count"] == 1
    assert "holdout_seeds" not in closed
    for field in ("workspace", "oracle", "dialogue", "tools", "system", "assets"):
        assert closed.get(field) is None, field
    assert closed["prompt_sha256"] == next(
        entry["hashes"]["prompt_sha256"]
        for entry in manifest["splits"]["closed-eval"]
    )

    # Public development keeps the real text; audit is hashed like closed-eval.
    assert by_id["C-3"]["prompt"]
    assert by_id["C-2"]["prompt"] is None


def test_include_private_does_not_open_a_closed_task():
    tasks = _tasks()
    manifest = _manifest(tasks)
    doc = schema.export_document(
        "pilot", tasks, include_private=True, hidden=holdout.hidden_ids(manifest)
    )
    by_id = {task["id"]: task for task in doc["tasks"]}
    assert by_id["C-1"]["prompt"] is None
    assert holdout.check_export(doc, manifest) == []


def test_export_gate_reports_a_leaked_prompt_and_a_leaked_seed():
    tasks = _tasks()
    manifest = _manifest(tasks)
    doc = schema.export_document("pilot", tasks, hidden=holdout.hidden_ids(manifest))
    closed = next(task for task in doc["tasks"] if task["id"] == "C-1")
    closed["prompt"] = "утекло"
    closed["holdout_seeds"] = [9901]
    errors = holdout.check_export(doc, manifest)
    assert any("prompt" in error for error in errors)
    assert any("holdout_seeds" in error for error in errors)


def test_export_without_a_manifest_publishes_closed_text():
    tasks = _tasks()
    doc = schema.export_document("pilot", tasks)
    by_id = {task["id"]: task for task in doc["tasks"]}
    assert by_id["K-1"]["prompt"]


# --------------------------------------------------------------------------- #
# Frozen hashes
# --------------------------------------------------------------------------- #


def test_a_changed_prompt_hash_is_caught():
    tasks = _tasks()
    manifest = _manifest(tasks)
    tasks[0].prompt += " правка"
    errors = holdout.check_manifest(manifest, tasks)
    assert any("prompt_sha256" in error and "изменился" in error for error in errors)


def test_a_tampered_manifest_hash_is_caught_even_with_a_fresh_digest():
    tasks = _tasks()
    manifest = _manifest(tasks)
    entry = manifest["splits"]["closed-eval"][0]
    entry["hashes"]["oracle_sha256"] = "0" * 64
    manifest["composition"] = {
        name: holdout.composition_hash(manifest["splits"][name]) for name in holdout.SPLITS
    }
    manifest["digest"] = holdout.manifest_digest(manifest)
    assert holdout.validate_manifest(manifest) == []
    errors = holdout.check_manifest(manifest, tasks)
    assert any("oracle_sha256" in error for error in errors)


def test_a_manifest_that_skips_a_task_is_caught():
    tasks = _tasks()
    manifest = _manifest(tasks)
    manifest["splits"]["public-dev"] = [
        entry for entry in manifest["splits"]["public-dev"] if entry["id"] != "C-3"
    ]
    manifest["composition"] = {
        name: holdout.composition_hash(manifest["splits"][name]) for name in holdout.SPLITS
    }
    manifest["digest"] = holdout.manifest_digest(manifest)
    assert any("C-3" in error for error in holdout.check_manifest(manifest, tasks))


# --------------------------------------------------------------------------- #
# Rights registry
# --------------------------------------------------------------------------- #


def test_valid_record_is_importable(tmp_path: Path):
    record = _record()
    assert holdout.validate_record(record) == []
    assert holdout.importable(record)
    holdout.require_importable(record)

    path = tmp_path / "material.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert cli.main(["licenses", "check", str(path)]) == 0


@pytest.mark.parametrize("field", ["code_license", "data_license"])
def test_record_without_a_licence_is_rejected(field):
    record = _record()
    del record[field]
    errors = holdout.validate_record(record)
    assert any(field in error for error in errors)
    assert not holdout.importable(record)
    with pytest.raises(ValueError, match=field):
        holdout.require_importable(record)


def test_noassertion_and_unverified_records_are_rejected():
    unknown = _record(data_license="NOASSERTION")
    assert any("data_license" in error for error in holdout.validate_record(unknown))
    unverified = _record(status="unverified")
    assert any("импорт запрещён" in error for error in holdout.validate_record(unverified))
    refused = _record(redistribution="forbidden")
    assert any("распространение" in error for error in holdout.validate_record(refused))
    restricted = _record(redistribution="restricted")
    assert any("note" in error for error in holdout.validate_record(restricted))
    assert holdout.validate_record(_record(redistribution="restricted", note="только с атрибуцией")) == []


def test_the_checked_in_registry_is_importable():
    assert holdout.check_directory() == []
    record = holdout.load_record_by_id("bigcodebench")
    assert record["status"] == "verified"


def test_licenses_check_rejects_an_unlicensed_record_and_a_misnamed_file(tmp_path: Path):
    bad = tmp_path / "material.json"
    bad.write_text(json.dumps(_record(data_license=None)), encoding="utf-8")
    assert cli.main(["licenses", "check", "--dir", str(tmp_path)]) == 1

    other = tmp_path / "not-the-id.json"
    other.write_text(json.dumps(_record()), encoding="utf-8")
    assert cli.main(["licenses", "check", "--dir", str(tmp_path)]) == 1


def test_empty_registry_is_an_error(tmp_path: Path):
    assert cli.main(["licenses", "check", "--dir", str(tmp_path)]) == 1


def test_adapter_prepare_refuses_a_missing_record(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(holdout, "LICENSE_DIR", tmp_path)
    with pytest.raises(bigcodebench.AdapterError, match="импорт отклонён"):
        bigcodebench.prepare(out=tmp_path / "bundle", tasks_path=_fixture_tasks(tmp_path))


def _fixture_tasks(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "BigCodeBench/1",
                "prompt": "prompt",
                "entry_point": "f",
                "canonical_solution": "return 1",
                "libs": [],
                "doc_struct": "d",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# CLI round trip
# --------------------------------------------------------------------------- #


def test_cli_assign_check_and_export_round_trip(tmp_path: Path):
    manifest_path = tmp_path / "holdout.json"
    only = ",".join(IDS)
    pinned = ["--pin", "C-1=closed-eval", "--pin", "C-2=audit", "--pin", "C-3=public-dev"]
    assert cli.main(["holdout", "assign", str(manifest_path), "--only", only, *pinned]) == 0
    manifest = holdout.load_manifest(manifest_path)

    assert cli.main(["holdout", "check", str(manifest_path), "--only", only]) == 0

    out = tmp_path / "suite.json"
    assert (
        cli.main(["export", "--only", only, "--holdout", str(manifest_path), "--out", str(out)]) == 0
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    by_id = {task["id"]: task for task in doc["tasks"]}
    assert by_id["C-1"]["prompt"] is None
    assert by_id["C-3"]["prompt"]
    assert holdout.check_export(doc, manifest) == []


def test_cli_check_catches_an_edited_manifest(tmp_path: Path):
    manifest_path = tmp_path / "holdout.json"
    only = ",".join(IDS)
    pinned = ["--pin", "C-1=closed-eval", "--pin", "C-3=public-dev"]
    assert cli.main(["holdout", "assign", str(manifest_path), "--only", only, *pinned]) == 0
    manifest = holdout.load_manifest(manifest_path)
    manifest["splits"]["closed-eval"][0]["hashes"]["prompt_sha256"] = "1" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert cli.main(["holdout", "check", str(manifest_path), "--only", only]) == 1
