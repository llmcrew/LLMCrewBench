"""Sweep orchestration: the driver must reach the run and record its result.

The sweep once called ``time.time()`` without importing ``time`` (R13), so a
whole orchestration path was dead in practice. These tests run the real
``run_one``/``main`` flow with the server, preflight and CLI replaced, so no
model is loaded, nothing is rendered and nothing waits on the network.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from dailybench import reference

import scripts.sweep as sweep


class _Ref:
    """Minimal stand-in for ``reference.ReferenceConfig``."""

    def __init__(self, entry: dict) -> None:
        self.entry = entry
        self.policy = dict(reference.POLICY)
        self.id = entry["id"]
        self.label = entry.get("label") or entry["id"]

    def launch_line(self) -> str:
        return f"launch {self.id}"

    def effective(self) -> dict:
        merged = dict(self.policy)
        merged.update(self.entry)
        merged.setdefault("context_window", self.policy["context_window"])
        return merged

    @property
    def context_window(self) -> int | None:
        return self.entry.get("context_window", self.policy["context_window"])

    def unsupported_categories(self) -> tuple[str, ...]:
        return ()


def _entry(entry_id: str = "test-entry") -> dict:
    return {
        "id": entry_id,
        "label": "Test entry",
        "base_url": "http://127.0.0.1:9/v1",
        "model": "test-model",
        "quant": "f16",
        "context_window": 32768,
    }


def _report(verdict_probes: list[reference.Probe] | None = None) -> reference.PreflightReport:
    probes = verdict_probes or [reference.Probe(name, "ok", "") for name in reference.REQUIRED_PROBES]
    return reference.PreflightReport(
        reference_id="test-entry",
        label="Test entry",
        probes=probes,
        metrics={"vram_mib": 1000},
    )


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Repo root moved to a tmp dir plus every live dependency replaced."""
    root = tmp_path
    (root / "runs").mkdir()
    monkeypatch.setattr(sweep, "ROOT", root)
    monkeypatch.setattr(sweep, "used_mib", lambda: 12)
    monkeypatch.setattr(sweep, "_suite_size", lambda run_dir: 1)
    monkeypatch.setattr(sweep.launch, "ensure_server", lambda base_url, entry: "key")
    monkeypatch.setattr(sweep.launch, "load_entry", lambda base_url, key, entry: (True, "загружена"))
    monkeypatch.setattr(sweep.reference, "resolve", lambda cfg_id: _Ref(_entry(cfg_id)))
    monkeypatch.setattr(sweep.reference, "preflight", lambda resolved, *, full: _report())
    return root


def _args(**over) -> object:
    class Args:
        no_full = False
        profile = "baseline-20261020"
        limit = 0
        tasks = ""
        skip_calibration = False
        allow_unverified = False

    args = Args()
    for key, value in over.items():
        setattr(args, key, value)
    return args


def test_module_imports_time() -> None:
    module = importlib.import_module("scripts.sweep")
    assert module.time is sys.modules["time"]


def test_run_one_reaches_run_and_records_scores(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_cli(argv: list[str]) -> int:
        calls.append(argv)
        run_dir = sandbox / "runs" / "test-entry"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "scores.json").write_text(
            json.dumps([{"category": "A", "earned": 5.0, "possible": 10.0, "max_points": 10.0}]),
            encoding="utf-8",
        )
        (run_dir / "speed.json").write_text(json.dumps({"tokens_per_second": 12.5}), encoding="utf-8")
        return 0

    monkeypatch.setattr(sweep, "cli_main", fake_cli)

    result = sweep.run_one(_entry(), _args(limit=3, tasks="A1", skip_calibration=True))

    assert calls == [["run", "--config", "test-entry", "--name", "test-entry", "--speed",
                      "--speed-long", "4096", "--speed-repeats", "2",
                      "--profile", "baseline-20261020",
                      "--provenance", str(sandbox / "runs" / "test-entry" / "provenance.json"),
                      "--limit", "3", "--only", "A1", "--no-calibrate"]]
    assert result["loaded"] is True
    assert result["preflight"] == "admitted"
    assert result["run_exit_code"] == 0
    assert isinstance(result["run_seconds"], float)
    assert result["verdict"].startswith("5/10")
    assert result["earned"] == 5.0
    assert result["speed"] == {"tokens_per_second": 12.5}
    assert (sandbox / "runs" / "preflight-test-entry.md").is_file()


def test_run_one_marks_run_without_scores(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep, "cli_main", lambda argv: 7)

    result = sweep.run_one(_entry(), _args())

    assert result["run_exit_code"] == 7
    assert result["verdict"] == "прогон не дал scores.json"


def test_run_one_stops_when_preflight_rejects(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sweep.reference, "preflight", lambda resolved, *, full: _report([reference.Probe("weights", "fail", "другие")])
    )
    monkeypatch.setattr(sweep, "cli_main", lambda argv: pytest.fail("прогон не должен запускаться"))

    result = sweep.run_one(_entry(), _args())

    assert result["verdict"] == "отклонена preflight'ом"
    assert result["preflight_probes"] == {"weights": "fail"}


def test_run_one_reports_load_failure(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep.launch, "load_entry", lambda base_url, key, entry: (False, "OOM"))
    monkeypatch.setattr(sweep.reference, "preflight", lambda resolved, *, full: pytest.fail("preflight не нужен"))

    result = sweep.run_one(_entry(), _args())

    assert result["verdict"] == "не загрузилась: OOM"


def test_main_sweeps_entries_and_writes_summary(sandbox: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    other = _entry("other-entry")
    monkeypatch.setattr(
        sweep.reference, "model_entries", lambda path=None: [_Ref(_entry()), _Ref(other), _Ref(_entry("generic-gguf"))]
    )
    monkeypatch.setattr(sweep, "cli_main", lambda argv: 3)
    monkeypatch.setattr(sys, "argv", ["sweep.py", "--skip", "generic-gguf", "--only", "test-entry,other-entry"])

    assert sweep.main() == 0

    summary = json.loads((sandbox / "runs" / "sweep-summary.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in summary["results"]] == ["test-entry", "other-entry"]
    assert all(row["verdict"] == "прогон не дал scores.json" for row in summary["results"])
    out = capsys.readouterr().out
    assert "Итоговая таблица" in out


def test_main_passes_profile_to_the_run(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(sweep.reference, "model_entries", lambda path=None: [_Ref(_entry())])
    monkeypatch.setattr(sweep, "cli_main", lambda argv: seen.append(argv[argv.index("--profile") + 1]) or 3)
    monkeypatch.setattr(sys, "argv", ["sweep.py", "--profile", "stretch-v1"])

    assert sweep.main() == 0

    assert seen == ["stretch-v1"]
    summary = json.loads((sandbox / "runs" / "sweep-summary.json").read_text(encoding="utf-8"))
    assert summary["results"][0]["profile"] == "stretch-v1"


def test_parser_rejects_unknown_profile(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sweep.py", "--profile", "nope"])

    with pytest.raises(SystemExit):
        sweep.main()


def test_suite_size_reads_recorded_per_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The denominator is the recorded suite, not a rebuilt one."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "suite.json").write_text(
        json.dumps({"per_task": [{"id": "A1"}, {"id": "K-1"}]}),
        encoding="utf-8",
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("build_suite не должен вызываться")

    monkeypatch.setattr("dailybench.suite.build_suite", boom)

    assert sweep._suite_size(run_dir) == 2
