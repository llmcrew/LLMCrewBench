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
sys.path.insert(0, str(TOOLS))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fleet = load("fleet")
media_status = load("media_status")
record_clip = load("record_clip")
preflight = load("preflight")
record_all = load("record_all")

sys.path.insert(0, str(ROOT))
build_runs_data = load("build_runs_data")
verify_clips = load("verify_clips")

VIDEO = ROOT / "video"
SHOWCASE = VIDEO / "src" / "data" / "showcase.json"
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
    assert rows, "showcase.json пуст"
    checked = [
        row for row in rows
        if row["scene"].startswith("L-") and row["scene"] != "L-1"
    ]
    assert checked, "в манифесте нет L-сцен, кроме L-1"
    assert not bad, bad


def test_task_status_maps_reasons_not_one_generic_flag(tmp_path):
    runs = run_tree(tmp_path, {"m": [
        {"task_id": "A-1"},
        {"task_id": "A-2", "skipped": "нет браузера"},
        {"task_id": "A-3", "error": "evaluator unavailable"},
        {"task_id": "A-4", "error": "агент не уложился", "possible": 1},
        {"task_id": "A-5", "possible": 1, "earned": 1},
        {"task_id": "A-6", "failure": "evaluator_failure", "error": "boom", "possible": 1},
        {"task_id": "A-7", "failure": "infra_failure", "error": "обрыв соединения", "possible": 1},
        {"task_id": "A-8", "failure": "transport_unavailable", "error": "unreachable", "possible": 1},
        {"task_id": "A-9", "failure": "model_failure", "error": "неверный ответ", "possible": 1},
        {"task_id": "A-98"},
    ]})
    assert media_status.task_status("m", "A-1", runs) == "model_failed"  # 0/0
    assert media_status.task_status("m", "A-2", runs) == "skipped"
    assert media_status.task_status("m", "A-3", runs) == "evaluator_unavailable"
    assert media_status.task_status("m", "A-4", runs) == "model_failed"
    assert media_status.task_status("m", "A-5", runs) == "ok"
    # Провал берётся из поля `failure`, текст ошибки — только фолбэк: обрыв
    # транспорта и 503 — инфраструктура, а не ответ модели.
    assert media_status.task_status("m", "A-6", runs) == "evaluator_unavailable"
    assert media_status.task_status("m", "A-7", runs) == "infra_failure"
    assert media_status.task_status("m", "A-8", runs) == "infra_failure"
    assert media_status.task_status("m", "A-9", runs) == "model_failed"
    assert media_status.task_status("m", "A-100", runs) == "not_run"
    assert "infra_failure" in media_status.STATUSES
    assert "infra_failure" in media_status.PLACEHOLDER_STATUSES


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
    assert media_status.status_of(out, "m", "L-3", runs) == "not_run", "нет задачи и нет клипа — not_run"


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
    clip = media_status.clip_path(OUT, record_all.MODELS[0], "K-1")
    if not clip.is_file():
        pytest.skip("клипы ещё не записаны в этом дереве")
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






def test_k_requires_ready_or_driver(monkeypatch):
    def call(method, params=None, **kwargs):
        return {"result": {"result": {"value": "settled:10|800x600|40"}}}

    monkeypatch.setattr(record_clip, "time", Clock())
    assert record_clip.page_ready(call, FakeWS(), "K-1", timeout=0.3) == ""


# -- флот из данных, а не из кода -------------------------------------------- #


def write_models(tmp_path: Path, *dirs: str) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps([
        {"dir": name, "label": name.upper(), "accent": f"#00000{index}"}
        for index, name in enumerate(dirs)
    ]), encoding="utf-8")
    return path


def test_the_fleet_and_scenes_come_from_one_data_file():
    names = fleet.model_names()
    assert record_clip.MODELS == tuple(names)
    assert media_status.MODEL_NAMES == tuple(names)
    assert record_all.MODELS == names
    # В коде не осталось литералов каталогов прогонов.
    for module in (record_clip, media_status, record_all, build_runs_data):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert names[0] not in source, module.__name__
    scenes = fleet.scene_ids()
    assert record_clip.SCENES == tuple(scenes)
    assert record_all.SCENES == scenes


def test_a_third_model_is_recordable_without_code_changes(tmp_path):
    models = write_models(tmp_path, "третья-модель", fleet.model_names()[0])
    out = ROOT / "video" / "public" / "clips"
    with pytest.raises(SystemExit):
        record_clip.check_target("третья-модель", "K-1", out)
    assert record_clip.check_target("третья-модель", "K-1", out, models) == out.resolve()
    with pytest.raises(SystemExit):
        record_clip.check_target("третья-модель", "K-99", out, models)


def test_fleet_rejects_duplicates_and_empty_files(tmp_path):
    duplicate = write_models(tmp_path, "m", "m")
    with pytest.raises(SystemExit):
        fleet.models(duplicate)
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        fleet.models(empty)
    with pytest.raises(SystemExit):
        fleet.scenes(empty)


def test_build_runs_data_writes_a_row_for_every_model_in_the_file(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name in ("one", "two"):
        (runs / name).mkdir(parents=True)
        (runs / name / "scores.json").write_text(json.dumps([
            {"task_id": "A-1", "category": "A", "earned": 1, "possible": 1},
        ]), encoding="utf-8")
    (tmp_path / "video" / "src" / "data").mkdir(parents=True)
    monkeypatch.setattr(build_runs_data, "ROOT", tmp_path)
    monkeypatch.setattr(build_runs_data, "RUNS", runs)
    models = write_models(tmp_path, "one", "two")
    assert build_runs_data.main(["--models", str(models)]) == 0
    rows = json.loads((tmp_path / "video" / "src" / "data" / "runs.json").read_text(encoding="utf-8"))
    assert [row["dir"] for row in rows] == ["one", "two"]
    assert [row["label"] for row in rows] == ["ONE", "TWO"]


# -- score не расходится с вердиктом отчёта ---------------------------------- #


def test_score_is_the_headline_of_the_summary_not_the_raw_earned(tmp_path, monkeypatch):
    """Провал admission у E: число на экране обязано совпасть с вердиктом."""
    runs = tmp_path / "runs"
    (runs / "m").mkdir(parents=True)
    (runs / "m" / "scores.json").write_text(json.dumps([
        {"task_id": "A-1", "category": "A", "earned": 5, "possible": 10},
        {"task_id": "E-1", "category": "E", "earned": 10, "possible": 10, "admission": False},
    ]), encoding="utf-8")
    monkeypatch.setattr(build_runs_data, "RUNS", runs)
    row = build_runs_data.model_row({"dir": "m", "label": "M", "accent": "#fff"})
    assert row["score"] == 5.0, "E не прошёл допуск — его баллы не идут в headline"
    assert row["score"] != row["totals"]["A"]["earned"] + row["totals"]["E"]["earned"]
    assert row["verdict"].startswith("5/")
    assert row["possible"] == 10.0
    assert row["label"] == "M" and row["accent"] == "#fff"


# -- fps/crf/preset действительно доезжают ----------------------------------- #


def test_the_injected_clock_renders_the_requested_rate():
    assert "const FPS = 120;" in record_clip.FRAME_CLOCK_JS
    assert "const FPS = 30;" in record_clip.frame_clock(30)
    assert "__FPS__" not in record_clip.frame_clock(30)
    assert record_clip._frame_count(1, fps=30) == 30
    assert record_clip._frame_count(1 / 30, fps=30) == 1
    with pytest.raises(ValueError):
        record_clip._frame_count(0.51, fps=30)


def test_encode_and_placeholder_carry_fps_crf_preset(tmp_path, monkeypatch):
    seen: list[list[str]] = []

    def atomic(out, args):
        seen.append(args)
        out.write_bytes(b"x")

    monkeypatch.setattr(record_clip, "_atomic_output", atomic)
    frames = tmp_path / "frames"
    frames.mkdir()
    record_clip.encode(frames, tmp_path / "a.mp4", 4, fps=30, crf=18, preset="slow")
    record_clip.placeholder(tmp_path / "b.mp4", 0.5, fps=30, crf=18, preset="slow")
    assert seen[0][seen[0].index("-framerate") + 1] == "30"
    assert seen[0][seen[0].index("-crf") + 1] == "18"
    assert seen[0][seen[0].index("-preset") + 1] == "slow"
    assert seen[1][seen[1].index("-r") + 1] == "30"
    assert seen[1][seen[1].index("-frames:v") + 1] == "15"
    assert seen[1].count("medium") == 0


def test_a_non_default_encoding_reaches_the_capture_evidence(tmp_path, monkeypatch):
    video = tmp_path / "video"
    (video / "out").mkdir(parents=True)
    monkeypatch.setattr(record_clip, "VIDEO", video)
    monkeypatch.setattr(record_clip, "ROOT", tmp_path)
    monkeypatch.setattr(record_clip, "locate_source", lambda *args, **kwargs: (None, tmp_path))

    def placeholder(out, secs, **kwargs):
        out.write_bytes(b"clip")

    monkeypatch.setattr(record_clip, "placeholder", placeholder)
    out = video / "out" / "clips"
    model = record_all.MODELS[0]
    code = record_clip.main([model, "K-1", str(out), "0.5", "0",
                             "--fps", "30", "--crf", "18", "--preset", "slow",
                             "--source-root", str(tmp_path)])
    assert code == record_clip.EXIT["model_failed"], "неизвестный исходник -> плашка"
    evidence = json.loads((out / f"{model}__K-1.capture.json").read_text(encoding="utf-8"))
    assert (evidence["fps"], evidence["crf"], evidence["preset"]) == (30, 18, "slow")
    assert evidence["frames"] == 15
    assert evidence["status"] == "model_failed"


def test_preflight_takes_seconds_and_fps_from_the_cli(monkeypatch):
    calls: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = json.dumps({"frames": [{"pkt_pts_time": "0"}]})
        stderr = ""

    monkeypatch.setattr(preflight.subprocess, "run",
                        lambda args, **kwargs: calls.append(args) or Done())
    assert preflight.decode_frame(Path("ffprobe"), Path("a.mp4"), last=True, seconds=5.0) is None
    assert calls[0][calls[0].index("-read_intervals") + 1] == "4.800%+#1"


def test_verify_clips_takes_crf_and_preset_from_the_cli(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = clips / "m__K-1.mp4"
    clip.write_bytes(b"x")
    clip.with_suffix(".capture.json").write_text(json.dumps({
        "status": "ok", "seconds": 1.0, "fps": 10, "frames": 10, "crf": 20,
        "preset": "slow", "captured_frames": 10, "input_format": "png",
        "clock_discontinuities": 0, "clock_start_ms": 0, "clock_end_ms": 900,
    }), encoding="utf-8")
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps([{"model": "m", "scene": "K-1", "status": "ok"}]), encoding="utf-8")
    monkeypatch.setattr(verify_clips, "verify",
                        lambda path, *args, **kwargs: {"file": path.name, "ok": True, "errors": []})
    code = verify_clips.main([str(clips), "--manifest", str(manifest), "--seconds", "1",
                              "--fps", "10", "--crf", "20", "--preset", "slow"])
    report = json.loads((clips.parent / "recording-quality.json").read_text(encoding="utf-8"))
    assert code == 0, report["results"][0]["errors"]
    assert report["results"][0]["capture"]["crf"] == 20


# -- проба не стирает capture_failed без клипа ------------------------------- #


def test_a_ready_page_keeps_capture_failed_without_a_clip(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    media_status.write_marker(out, "m", "K-1", "capture_failed", "ffmpeg упал")
    assert media_status.refresh_after_probe(out, "m", "K-1") == "capture_failed"
    assert media_status.marker_path(out, "m", "K-1").is_file()
    assert media_status.status_of(out, "m", "K-1") == "capture_failed"


def test_a_ready_page_with_a_clip_still_clears_the_marker(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    media_status.clip_path(out, "m", "K-1").write_bytes(b"x")
    media_status.write_marker(out, "m", "K-1", "capture_failed")
    assert media_status.refresh_after_probe(out, "m", "K-1") is None
    assert not media_status.marker_path(out, "m", "K-1").is_file()
    assert media_status.status_of(out, "m", "K-1") == "ok"


def test_an_empty_clip_does_not_count_as_recorded(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    media_status.clip_path(out, "m", "K-1").write_bytes(b"")
    media_status.write_marker(out, "m", "K-1", "capture_failed")
    assert media_status.refresh_after_probe(out, "m", "K-1") == "capture_failed"


def test_other_markers_are_cleared_by_a_ready_page(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    media_status.write_marker(out, "m", "K-1", "model_failed", "страница упала")
    assert media_status.refresh_after_probe(out, "m", "K-1") is None
