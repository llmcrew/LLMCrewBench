"""Pinned BigCodeBench adapter: identity, upstream argv and external metric.

The unit tests use a two-task fixture and a fake ``bigcodebench.evaluate``, so
they prove the protocol without docker, network or the upstream package. A
single live test is skipped unless the pinned scorer and docker are installed.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from dailybench import cli
from dailybench.adapters import bigcodebench as bcb

FIXTURES = Path(__file__).parent / "fixtures"
TASKS = FIXTURES / "bcb_tasks.jsonl"
FAKE = FIXTURES / "bcb_fake_evaluate.py"


def _prepare(tmp_path: Path, *extra: str) -> Path:
    out = tmp_path / "bundle"
    code = cli.main(
        [
            "adapter",
            "bigcodebench",
            "prepare",
            "--out",
            str(out),
            "--limit",
            "2",
            "--tasks",
            str(TASKS),
            *extra,
        ]
    )
    assert code == 0
    return out


def _samples(tmp_path: Path, first: str = "def f():\n    return 'PASS'", second: str = "def f():\n    return 1") -> Path:
    path = tmp_path / "model_predictions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"task_id": "BigCodeBench/289", "solution": first},
                {"task_id": "BigCodeBench/500", "solution": second},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _score(out: Path, samples: Path, log: Path, *extra: str) -> int:
    return cli.main(
        [
            "adapter",
            "bigcodebench",
            "score",
            "--out",
            str(out),
            "--samples",
            str(samples),
            "--evaluate-cmd",
            shlex.join([sys.executable, str(FAKE)]),
            *extra,
        ]
    )


def test_prepare_freezes_pinned_identity_and_upstream_ids(tmp_path):
    out = _prepare(tmp_path)
    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["status"] == "prepared"
    assert bundle["metric"] is None
    assert bundle["dataset"] == {
        "id": "bigcode/bigcodebench-hard",
        "revision": "v0.1.4",
        "revision_sha": "298d2cc7b96612e15e47313c3603ee124cee0c1f",
        "split": "v0.1.4",
        "prompt_field": "instruct_prompt",
        "upstream_split": "instruct",
        "subset": "hard",
        "task_ids": ["BigCodeBench/289", "BigCodeBench/500"],
    }
    assert bundle["scorer"]["execution"] == "local"
    assert bundle["scorer"]["batch_size"] == 1
    assert bundle["native_100"] is False
    assert bundle["selection"] == {"order": "upstream", "limit": 2, "source": str(TASKS)}
    assert bundle["license"]["spdx"] == "Apache-2.0"
    assert bundle["license"]["applies_to"] == ["dataset", "scorer-code"]
    tasks = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    assert [task["task_id"] for task in tasks] == ["BigCodeBench/289", "BigCodeBench/500"]
    assert tasks[0]["prompt"].startswith("Write a function `sort_dict_by_value`")
    example = (out / "predictions.example.jsonl").read_text(encoding="utf-8")
    assert json.loads(example.splitlines()[0])["task_id"] == "BigCodeBench/289"


def test_score_passes_pinned_argv_and_stores_pass1_as_ratio(tmp_path, monkeypatch):
    out = _prepare(tmp_path)
    log = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_BCB_LOG", str(log))

    code = _score(out, _samples(tmp_path), log)

    assert code == 0
    argv = json.loads(log.read_text(encoding="utf-8"))
    assert argv[argv.index("--execution") + 1] == "local"
    assert argv[argv.index("--selective_evaluate") + 1] == "BigCodeBench/289,BigCodeBench/500"
    assert argv[argv.index("--split") + 1] == "instruct"
    assert argv[argv.index("--subset") + 1] == "hard"
    assert argv[argv.index("--parallel") + 1] == "1"
    assert argv[argv.index("--samples") + 1] == str(out / "samples.jsonl")

    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in samples] == ["BigCodeBench/289", "BigCodeBench/500"]

    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["status"] == "verified"
    assert bundle["metric"]["name"] == "pass@1"
    assert bundle["metric"]["value"] == 0.5
    assert bundle["metric"]["numerator"] == 1
    assert bundle["metric"]["denominator"] == 2
    assert bundle["metric"]["percent_of_100"] is False
    assert [row["status"] for row in bundle["verdicts"]] == ["pass", "fail"]
    assert [row["task_id"] for row in bundle["verdicts"]] == ["BigCodeBench/289", "BigCodeBench/500"]
    assert (out / "evaluate.log").read_text(encoding="utf-8").startswith("$ ")


def test_bcb_score_survives_non_utf8_output(tmp_path):
    out = _prepare(tmp_path)
    stub = tmp_path / "non_utf8_scorer.py"
    stub.write_text("import os\nos.write(1, b'\\xef\\xf0')\nraise SystemExit(1)\n", encoding="utf-8")

    bundle = bcb.score(
        out=out,
        samples=_samples(tmp_path),
        evaluate_cmd=[sys.executable, str(stub)],
    )

    assert bundle["status"] == "unverified"
    assert "кодом 1" in bundle["reason"]
    assert "��" in (out / "evaluate.log").read_text(encoding="utf-8")


def test_missing_package_and_docker_are_unverified_not_zero(tmp_path, monkeypatch, capsys):
    out = _prepare(tmp_path)
    monkeypatch.setattr(bcb.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(bcb.shutil, "which", lambda name: None)
    samples = _samples(tmp_path)

    code = cli.main(
        ["adapter", "bigcodebench", "score", "--out", str(out), "--samples", str(samples)]
    )

    assert code == 1
    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["status"] == "unverified"
    assert bundle["metric"] is None
    assert "нет пакета bigcodebench==0.2.5" in bundle["reason"]
    assert "нет docker" in bundle["reason"]
    assert "unverified" in capsys.readouterr().out


def test_missing_prediction_is_unverified(tmp_path, capsys):
    out = _prepare(tmp_path)
    samples = tmp_path / "partial.jsonl"
    samples.write_text(json.dumps({"task_id": "BigCodeBench/289", "solution": "PASS"}) + "\n", encoding="utf-8")

    code = _score(out, samples, tmp_path / "argv.json")

    assert code == 1
    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["metric"] is None
    assert "BigCodeBench/500" in bundle["reason"]


def test_duplicate_sample_breaks_batch_size_one(tmp_path, capsys):
    out = _prepare(tmp_path)
    samples = tmp_path / "dupes.jsonl"
    samples.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"task_id": "BigCodeBench/289", "solution": "PASS"},
                {"task_id": "BigCodeBench/289", "solution": "PASS"},
                {"task_id": "BigCodeBench/500", "solution": "PASS"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    code = _score(out, samples, tmp_path / "argv.json")

    assert code == 1
    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert "batch size 1" in bundle["reason"]


def test_parse_rejects_foreign_or_partial_upstream_results(tmp_path):
    tasks = ["BigCodeBench/289", "BigCodeBench/500"]
    path = tmp_path / "r_eval_results.json"
    path.write_text(
        json.dumps({"eval": {"BigCodeBench/289": [{"status": "pass"}]}}),
        encoding="utf-8",
    )
    with pytest.raises(bcb.AdapterError, match="нет результата"):
        bcb.parse_results(path, tasks)

    path.write_text(
        json.dumps(
            {
                "eval": {
                    "BigCodeBench/289": [{"status": "pass"}],
                    "BigCodeBench/999": [{"status": "pass"}],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(bcb.AdapterError, match="чужие task_id"):
        bcb.parse_results(path, tasks)


def test_results_path_matches_upstream_convention():
    assert bcb.results_path(Path("run/model.jsonl")) == Path("run/model_eval_results.json")


@pytest.mark.skipif(
    bcb.readiness() != [],
    reason="нужны bigcodebench==0.2.5 и docker",
)
def test_live_golden_answer_is_verified(tmp_path):
    out = tmp_path / "live"
    assert cli.main(["adapter", "bigcodebench", "prepare", "--out", str(out), "--limit", "1"]) == 0
    task = json.loads((out / "tasks.json").read_text(encoding="utf-8"))[0]
    samples = tmp_path / "golden.jsonl"
    samples.write_text(
        json.dumps({"task_id": task["task_id"], "solution": task["canonical_solution"]}) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["adapter", "bigcodebench", "score", "--out", str(out), "--samples", str(samples)]) == 0
    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["status"] == "verified"
    assert bundle["metric"]["value"] == 1.0
