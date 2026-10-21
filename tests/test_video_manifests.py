"""Контракт медиа витрины: статусы, манифест, preflight, spend и запись.

Тесты не требуют node и Chrome. Декодирование ffprobe — один клип, при
отсутствии ffprobe проверка пропускается.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "video" / "tools"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


media_status = load("media_status")
record_clip = load("record_clip")
preflight = load("preflight")
record_all = load("record_all")

sys.path.insert(0, str(ROOT))
build_runs_data = load("build_runs_data")

VIDEO = ROOT / "video"
SHOWCASE = VIDEO / "src" / "data" / "showcase.json"
CLIPS_JSON = VIDEO / "public" / "clips.json"
OUT = VIDEO / "public" / "clips"


def run_tree(tmp_path: Path, scores: dict[str, list[dict]] | None = None) -> Path:
    runs = tmp_path / "runs"
    for model, rows in (scores or {}).items():
        target = runs / model
        target.mkdir(parents=True)
        (target / "scores.json").write_text(json.dumps(rows), encoding="utf-8")
    return runs


# -- R21: та же семантика spend, что у core.Transcript.spend() ---------------- #


def test_spend_subtracts_reasoning_from_completion():
    cost = build_runs_data.spend_of([
        {"transcript": {"task_id": "X", "category": "A", "turns": [
            {"messages": [], "reply": {"completion_tokens": 100, "reasoning_tokens": 60}},
        ]}}
    ])
    assert cost["answer"] == 40
    assert cost["reasoning"] == 60


def test_spend_unknown_is_none_not_zero():
    assert build_runs_data.spend_of([])["answer"] is None
    assert build_runs_data.spend_of([{"transcript": {}}])["answer"] is None
    cost = build_runs_data.spend_of([
        {"transcript": {"turns": [{"messages": [], "reply": {"completion_tokens": 10}}]}},
        {"transcript": {}},
    ])
    assert cost["answer"] is None
    assert cost["reasoning"] is None


# -- R19/R20: статусы и манифест --------------------------------------------- #


def test_current_manifest_matches_markers_and_files():
    scenes = json.loads((VIDEO / "src" / "data" / "scenes.json").read_text(encoding="utf-8"))
    rows = json.loads(SHOWCASE.read_text(encoding="utf-8"))
    assert [r["scene"] for r in rows] == [s["id"] for s in scenes]
    derived = []
    for row in rows:
        assert [run["model"] for run in row["runs"]] == record_all.MODELS
        for run in row["runs"]:
            assert "missing" not in run
            status = run["status"]
            assert status in media_status.STATUSES
            marker = media_status.marker_path(OUT, run["model"], row["scene"])
            clip = media_status.clip_path(OUT, run["model"], row["scene"])
            if status == "ok":
                assert clip.is_file() and not marker.exists()
            else:
                assert marker.is_file() and clip.is_file()
                assert marker.read_text(encoding="utf-8").strip() == status
            derived.append({"model": run["model"], "scene": row["scene"], "status": status})
    assert json.loads(CLIPS_JSON.read_text(encoding="utf-8")) == derived


def test_current_tree_statuses_not_run_for_missing_tasks():
    rows = json.loads(SHOWCASE.read_text(encoding="utf-8"))
    bad = [
        (row["scene"], run["model"], run["status"])
        for row in rows
        for run in row["runs"]
        if row["scene"].startswith("L-") and row["scene"] != "L-1"
        and run["model"].endswith("1017")
        and run["status"] != "not_run"
    ]
    assert not bad, bad


def test_task_status_maps_reasons_not_one_generic_flag(tmp_path):
    runs = run_tree(tmp_path, {"m": [
        {"task_id": "A-1"},
        {"task_id": "A-2", "skipped": "нет браузера"},
        {"task_id": "A-3", "error": "evaluator unavailable"},
        {"task_id": "A-4", "error": "агент не уложился", "possible": 1},
        {"task_id": "A-5", "possible": 1, "earned": 1},
    ]})
    assert media_status.task_status("m", "A-1", runs) == "model_failed"  # 0/0
    assert media_status.task_status("m", "A-2", runs) == "skipped"
    assert media_status.task_status("m", "A-3", runs) == "evaluator_unavailable"
    assert media_status.task_status("m", "A-4", runs) == "model_failed"
    assert media_status.task_status("m", "A-5", runs) == "ok"
    assert media_status.task_status("m", "A-9", runs) == "not_run"


def test_status_resolution_marker_wins_and_legacy_inferred(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    runs = run_tree(tmp_path, {"m": [{"task_id": "K-1", "possible": 1, "earned": 1}]})
    (out / "m__K-1.mp4").write_bytes(b"x")
    assert media_status.status_of(out, "m", "K-1", runs) == "ok"
    media_status.write_marker(out, "m", "K-1", "capture_failed")
    assert media_status.status_of(out, "m", "K-1", runs) == "capture_failed"
    # legacy `1` у отсутствующей задачи — not_run, без данных прогона — сбой модели.
    (out / "m__L-2.mp4").write_bytes(b"x")
    (out / "m__L-2.missing").write_text("1", encoding="utf-8")
    assert media_status.status_of(out, "m", "L-2", runs) == "not_run"
    assert media_status.status_of(out, "m", "L-2", None) == "model_failed"
    assert media_status.status_of(out, "m", "L-3", runs) == "media_missing"


def test_a_marker_detail_keeps_the_status_readable(tmp_path):
    """Улика после двоеточия (текст диалога) не подменяет статус маркера."""
    out = tmp_path / "clips"
    out.mkdir()
    media_status.write_marker(out, "m", "K-1", "model_failed", "диалог: boom\nвторой")
    text = (out / "m__K-1.missing").read_text(encoding="utf-8").strip()
    assert text == "model_failed: диалог: boom второй", "улика складывается в одну строку"
    assert media_status.status_of(out, "m", "K-1") == "model_failed"
    assert media_status.status_of(out, "m", "K-2") == "media_missing"


# -- R19: preflight ------------------------------------------------------------ #


def _fixture(tmp_path, statuses=("not_run", "ok")):
    scenes = [{"id": "K-1", "cat": "K", "title": "K-1 — сцена"}]
    rows = [{
        "scene": "K-1",
        "title": "K-1 — сцена",
        "runs": [
            {"model": m, "label": m, "clip": f"clips/{m}__K-1.mp4", "status": s, "accent": "#fff"}
            for m, s in zip(record_all.MODELS, statuses)
        ],
        "open": 1,
    }]
    out = tmp_path / "clips"
    out.mkdir()
    for (m, s) in zip(record_all.MODELS, statuses):
        if s != "media_missing":
            (out / f"{m}__K-1.mp4").write_bytes(b"x")
        if s != "ok":
            media_status.write_marker(out, m, "K-1", s)
    derived = [{"model": r["model"], "scene": "K-1", "status": r["status"]} for r in rows[0]["runs"]]
    clips = tmp_path / "clips.json"
    clips.write_text(json.dumps(derived), encoding="utf-8")
    return scenes, rows, out, clips


def test_preflight_accepts_consistent_manifest(tmp_path, monkeypatch):
    scenes, rows, out, clips = _fixture(tmp_path)
    runs = run_tree(tmp_path, {record_all.MODELS[1]: [{"task_id": "K-1", "possible": 1, "earned": 1}]})
    monkeypatch.setattr(preflight, "OUT", out)
    monkeypatch.setattr(preflight, "CLIPS_JSON", clips)
    monkeypatch.setattr(preflight, "RUNS", runs)
    errors: list[str] = []
    preflight.validate(rows, scenes, errors, [])
    assert errors == []


def test_preflight_catches_ok_with_stale_marker(tmp_path, monkeypatch):
    scenes, rows, out, clips = _fixture(tmp_path)
    monkeypatch.setattr(preflight, "OUT", out)
    monkeypatch.setattr(preflight, "CLIPS_JSON", clips)
    monkeypatch.setattr(preflight, "RUNS", tmp_path / "none")
    media_status.write_marker(out, record_all.MODELS[1], "K-1", "model_failed")
    errors: list[str] = []
    preflight.validate(rows, scenes, errors, [])
    assert any("маркер" in e for e in errors), errors


def test_preflight_catches_missing_clip(tmp_path, monkeypatch):
    scenes, rows, out, clips = _fixture(tmp_path, ("not_run", "ok"))
    (out / f"{record_all.MODELS[1]}__K-1.mp4").unlink()
    monkeypatch.setattr(preflight, "OUT", out)
    monkeypatch.setattr(preflight, "CLIPS_JSON", clips)
    monkeypatch.setattr(preflight, "RUNS", tmp_path / "none")
    errors: list[str] = []
    preflight.validate(rows, scenes, errors, [])
    assert any("клипа" in e for e in errors), errors


def test_preflight_catches_unknown_status(tmp_path, monkeypatch):
    scenes, rows, out, clips = _fixture(tmp_path)
    rows[0]["runs"][1]["status"] = "nonsense"
    monkeypatch.setattr(preflight, "OUT", out)
    monkeypatch.setattr(preflight, "CLIPS_JSON", clips)
    monkeypatch.setattr(preflight, "RUNS", tmp_path / "none")
    errors: list[str] = []
    preflight.validate(rows, scenes, errors, [])
    assert any("неизвестный статус" in e for e in errors)


def test_preflight_rejects_not_run_for_present_task(tmp_path, monkeypatch):
    scenes, rows, out, clips = _fixture(tmp_path, ("not_run", "ok"))
    runs = run_tree(tmp_path, {
        record_all.MODELS[0]: [{"task_id": "K-1", "possible": 1, "earned": 1}],
        record_all.MODELS[1]: [{"task_id": "K-1", "possible": 1, "earned": 1}],
    })
    monkeypatch.setattr(preflight, "OUT", out)
    monkeypatch.setattr(preflight, "CLIPS_JSON", clips)
    monkeypatch.setattr(preflight, "RUNS", runs)
    errors: list[str] = []
    preflight.validate(rows, scenes, errors, [])
    assert any("not_run" in e and "есть в прогоне" in e for e in errors), errors


def test_cached_placeholder_is_not_success(tmp_path, monkeypatch):
    out = tmp_path / "clips"
    out.mkdir()
    monkeypatch.setattr(record_all, "OUT", out)
    monkeypatch.setattr(record_all, "RUNS", tmp_path / "none")
    media_status.clip_path(out, record_all.MODELS[0], "L-2").write_bytes(b"x")
    media_status.write_marker(out, record_all.MODELS[0], "L-2", "not_run")
    assert record_all.job((record_all.MODELS[0], "L-2"), 23.0, 0.6)[2] == "not_run"


def test_preflight_catches_clips_json_drift(tmp_path, monkeypatch):
    scenes, rows, out, clips = _fixture(tmp_path)
    clips.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(preflight, "OUT", out)
    monkeypatch.setattr(preflight, "CLIPS_JSON", clips)
    monkeypatch.setattr(preflight, "RUNS", tmp_path / "none")
    errors: list[str] = []
    preflight.validate(rows, scenes, errors, [])
    assert any("clips.json" in e for e in errors)


def test_preflight_real_tree_passes_without_decode():
    venv = ROOT / ".venv" / "bin" / "python"
    python = str(venv) if venv.is_file() else sys.executable
    out = subprocess.run([python, str(TOOLS / "preflight.py"), "--no-decode"],
                         cwd=VIDEO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_preflight_decodes_first_and_last_frame():
    exe = preflight.find_ffprobe()
    if exe is None:
        pytest.skip("ffprobe недоступен")
    errors: list[str] = []
    preflight.check_clip(exe, record_all.MODELS[0], "K-1", "ok", errors)
    assert errors == []


# -- R22: containиment CLI и settled-готовность L ----------------------------- #


def test_cli_containment():
    valid = record_clip.check_target(record_all.MODELS[0], "K-1", OUT)
    assert valid == OUT.resolve()
    with pytest.raises(SystemExit):
        record_clip.check_target("нет-такой", "K-1", OUT)
    with pytest.raises(SystemExit):
        record_clip.check_target(record_all.MODELS[0], "K-99", OUT)
    with pytest.raises(SystemExit):
        record_clip.check_target(record_all.MODELS[0], "K-1", ROOT / "out")


class FakeWS:
    events: list[dict] = []


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        self.now += 0.05
        return self.now

    def sleep(self, _s: float) -> None:
        pass


def test_l_settled_without_ready_hook(monkeypatch):
    calls = {"n": 0}

    def call(method, params=None, **kwargs):
        calls["n"] += 1
        return {"result": {"result": {"value": "settled:10|800x600|40"}}}

    monkeypatch.setattr(record_clip, "time", Clock())
    assert record_clip.page_ready(call, FakeWS(), "L-1", timeout=1) == "settled"
    assert calls["n"] > 1


def test_l_settled_needs_stable_markup(monkeypatch):
    def call(method, params=None, **kwargs):
        value = "settled:10|800x600|40" if kwargs.get("n", 0) % 2 else "settled:11|800x600|41"
        kwargs["n"] = kwargs.get("n", 0) + 1
        return {"result": {"result": {"value": value}}}

    monkeypatch.setattr(record_clip, "time", Clock())
    assert record_clip.page_ready(call, FakeWS(), "L-1", timeout=0.5) == ""


def test_k_requires_ready_or_driver(monkeypatch):
    def call(method, params=None, **kwargs):
        return {"result": {"result": {"value": "settled:10|800x600|40"}}}

    monkeypatch.setattr(record_clip, "time", Clock())
    assert record_clip.page_ready(call, FakeWS(), "K-1", timeout=0.3) == ""
