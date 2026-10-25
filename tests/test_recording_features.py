"""Свежие фичи записи в самостоятельном наборе: политика, занавес, проверка.

Тесты идут на числах и фейковых кадрах: Chrome не нужен, ffmpeg - только для
двух сквозных проб (они пропускаются без бинарника). Проверяется то, что легко
соврать в отчёте: длительность, порядок показа, чернота, PTS и статусы.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "video" / "docs" / "recording"

sys.path.insert(0, str(RECORDING))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"recording_{name}", RECORDING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


timing = _load("timing")
layout = _load("layout")
manifest = _load("manifest")
verify_clips = _load("verify_clips")
record_scene = _load("record_scene")
record_batch = _load("record_batch")


# --------------------------------------------------------------------------- #
# Контракт: 20 с, мастер 120, показ 60, две полосы, занавес 0,75
# --------------------------------------------------------------------------- #


def test_clip_is_twenty_seconds_of_2400_frames_at_master_rate():
    assert timing.CLIP_SECONDS == 20.0
    assert timing.frame_count(timing.CLIP_SECONDS, timing.MASTER_FPS) == 2400
    # Показ 60 к/с - ровно половина мастера, кадры показа целые.
    assert timing.MASTER_FPS % timing.DISPLAY_FPS == 0
    assert 2400 % timing.DISPLAY_FPS == 0


def test_curtain_is_half_the_lead_and_ninety_frames_at_master():
    assert timing.CURTAIN_SECONDS == timing.SHOWCASE_LEAD / 2
    assert timing.stage_frames(timing.SHOWCASE_LEAD, timing.MASTER_FPS) == 90
    sizes = timing.validate_timing(fps=60)
    assert sizes["curtain_frames"] == 45, "на 60 к/с занавес - те же 0,75 с"


@pytest.mark.parametrize(
    "values",
    [
        {"curtain": 0.5},
        {"display_fps": 90},
        {"secs": 20.0005},
        {"fps": 0},
    ],
)
def test_timing_rejects_a_broken_policy(values):
    with pytest.raises(ValueError):
        timing.validate_timing(**values)


def test_frame_count_refuses_sloppy_durations():
    with pytest.raises(ValueError):
        timing.frame_count(0, 120)
    with pytest.raises(ValueError):
        timing.frame_count(-1, 120)
    with pytest.raises(ValueError):
        timing.frame_count(float("nan"), 120)
    assert timing.frame_count(0, 120, allow_zero=True) == 0


def test_status_vocabulary_is_closed_and_placeholders_are_separate():
    assert "ok" not in timing.PLACEHOLDER_STATUSES
    assert timing.is_placeholder("model_failed")
    assert not timing.is_placeholder("ok")
    assert not timing.valid_status("nonsense")
    assert not timing.valid_status(None)


@pytest.mark.parametrize(
    "status, has_clip, black, state",
    [
        ("ok", True, False, "ok"),
        ("ok", True, True, "ok-black"),
        ("ok", False, False, "missing-clip"),
        ("model_failed", True, True, "placeholder-ok"),
        ("model_failed", True, False, "placeholder-not-black"),
        ("nonsense", True, True, "unknown-status"),
    ],
)
def test_media_state_separates_placeholders_from_recordings(status, has_clip, black, state):
    assert timing.media_state(status, has_clip=has_clip, black=black) == state


# --------------------------------------------------------------------------- #
# Result-notes: провал не выдаётся за успех
# --------------------------------------------------------------------------- #

NOTE = {
    "model": "m", "scene": "L-6", "originalStatus": "failed",
    "reason": "нет страницы", "expected": "галактика из частиц",
}


def test_note_round_trips_with_and_without_repair():
    plain = timing.parse_note(NOTE)
    assert plain.repair is None and plain.original_status == "failed"
    repaired = timing.parse_note({**NOTE, "repair": {
        "clip": "clips/m__L-6-repair.mp4", "change": "дописан рендер",
        "sourceSha256": "a", "repairedSha256": "b",
    }})
    assert repaired.repair.clip.endswith("repair.mp4")
    assert not repaired.repair.same_source


def test_note_errors_are_collected_not_first_only():
    errors = timing.note_errors({"model": "", "scene": "L-6", "originalStatus": "ok"})
    assert len(errors) == 4, errors
    assert any("model" in error for error in errors)
    assert any("originalStatus" in error for error in errors)
    assert any("reason" in error for error in errors)


def test_presentation_clip_prefers_the_repair_and_hides_bare_failures():
    run = {"status": "ok", "clip": "clips/a.mp4", "resultNote": {"originalStatus": "failed"}}
    assert timing.presentation_clip(run) is None
    run["resultNote"]["repair"] = {"clip": "clips/a-repair.mp4"}
    assert timing.presentation_clip(run) == "clips/a-repair.mp4"
    assert timing.presentation_clip({"status": "ok", "clip": "clips/b.mp4"}) == "clips/b.mp4"
    assert timing.presentation_clip({"status": "model_failed", "clip": "clips/c.mp4"}) is None


def test_fullscreen_turn_needs_a_real_code_repair():
    failed = {"status": "ok", "clip": "clips/a.mp4", "resultNote": {"originalStatus": "failed"}}
    assert not timing.can_open_fullscreen(failed), "провал без правки не открывается"
    longer_wait = {**failed, "resultNote": {"originalStatus": "failed", "repair": {
        "clip": "clips/a2.mp4", "change": "дольше ждать", "sourceSha256": "x",
        "repairedSha256": "y",
    }}}
    assert not timing.can_open_fullscreen(longer_wait), "тот же код - тот же провал"
    same_source = {**failed, "resultNote": {"originalStatus": "failed", "repair": {
        "clip": "clips/a3.mp4", "change": "правка", "sourceSha256": "x", "repairedSha256": "x",
    }}}
    assert timing.can_open_fullscreen(same_source)
    assert timing.can_open_fullscreen({"status": "ok", "clip": "clips/z.mp4"})


def test_fullscreen_order_keeps_lane_positions():
    runs = [
        {"status": "ok", "clip": "0.mp4"},
        {"status": "model_failed", "clip": "1.mp4"},
        {"status": "ok", "clip": "2.mp4"},
    ]
    assert timing.fullscreen_order(runs, 0) == [0, 2]
    assert timing.fullscreen_order(runs, 1) == [2, 0], "порядок идёт от текущей полосы"
    assert timing.fullscreen_order(runs, 2) == [2, 0]
    assert timing.fullscreen_order([], 0) == []


def test_timeline_counts_turns_and_keeps_an_empty_section_visible():
    entries = [
        {"scene": "A", "runs": [{"status": "ok", "clip": "a.mp4"}]},
        {"scene": "B", "runs": [{"status": "model_failed", "clip": "b.mp4"}]},
    ]
    timeline = timing.build_timeline(entries)
    assert timeline[0]["seconds"] == timing.SHOWCASE_TURN_SECONDS
    assert timeline[1]["seconds"] == timing.SHOWCASE_EMPTY_SECONDS
    assert [section["from"] for section in timeline] == [
        0.0, timing.SHOWCASE_TURN_SECONDS,
    ]
    assert timing.timeline_errors(timeline, timing.DISPLAY_FPS) == []
    assert timing.timeline_frames(timeline, timing.DISPLAY_FPS) == (
        23 * 60 + 4 * 60
    )

    timeline[1]["from"] += 1
    assert any("!=" in error for error in timing.timeline_errors(timeline, 60))


# --------------------------------------------------------------------------- #
# Раскладка: 960 + 960, занавес, раскадровка
# --------------------------------------------------------------------------- #


def test_two_lanes_cover_the_frame_without_gaps():
    spec = layout.Layout()
    assert spec.lane_width == 960
    assert spec.lane(0) == {"x": 0, "y": 0, "width": 960, "height": 1080}
    assert spec.lane(1)["x"] == 960
    assert spec.errors() == []
    with pytest.raises(ValueError):
        spec.lane(2)


def test_lane_slides_from_its_half_to_the_full_frame():
    spec = layout.Layout()
    assert spec.at(1, 0.0)["x"] == 960
    assert spec.at(1, 1.0) == {"x": 0, "y": 0, "width": 1920, "height": 1080}
    middle = spec.at(1, 0.5)
    assert middle["x"] == 480 and middle["width"] == 1440
    assert all(isinstance(value, int) for value in middle.values())
    with pytest.raises(ValueError):
        spec.at(0, 1.5)


def test_storyboard_expands_exactly_one_curtain_before_the_window():
    turns = layout.storyboard([1, 0], clips=["b.mp4", "a.mp4"])
    first, second = turns
    assert first.lane == 1 and first.clip == "b.mp4"
    assert first.open_from - first.expand_from == 90
    assert first.return_to - first.open_to == 90
    assert first.open_to - first.open_from == 2400
    assert first.turn_end - first.turn_start == 2400 + 2 * 180
    assert second.turn_start == first.turn_end
    assert layout.storyboard_errors(turns) == []
    assert first.facing(first.expand_from, 90) == 0.0
    assert first.facing(first.open_from, 90) == 1.0, "к окну полоса уже во весь кадр"
    assert first.facing(first.return_to - 1, 90) == pytest.approx(1 / 90)
    assert first.facing(first.return_to, 90) == 0.0


def test_storyboard_errors_catch_overlap_and_wrong_open_window():
    turns = layout.storyboard([0, 1])
    turns[1] = record_scene.dataclasses.replace(
        turns[1], turn_start=turns[0].return_to - 1, expand_from=turns[0].return_to - 91,
        open_from=turns[0].return_to - 1, open_to=turns[0].return_to + 2400 - 1,
        return_to=turns[0].return_to + 2400, turn_end=turns[0].return_to + 2400 + 180,
    )
    errors = layout.storyboard_errors(turns)
    assert any("start" in error for error in errors)
    assert any("overlaps" in error for error in errors)


def test_storyboard_rejects_bad_orders():
    with pytest.raises(ValueError):
        layout.storyboard([0, 0])
    with pytest.raises(ValueError):
        layout.storyboard([2])
    with pytest.raises(ValueError):
        layout.storyboard([])
    with pytest.raises(ValueError):
        layout.storyboard([0], curtain=0.5)
    with pytest.raises(ValueError):
        layout.storyboard([0], secs=20.0), "запись меньше хода"


def test_frame_turn_maps_a_frame_to_its_lane_and_curtain():
    turns = layout.storyboard([1])
    turn, facing = layout.frame_turn(turns, turns[0].expand_from)
    assert turn.lane == 1 and facing == 0.0
    turn, facing = layout.frame_turn(turns, turns[0].open_from + 10)
    assert turn.lane == 1 and facing == 1.0
    assert layout.frame_turn(turns, 10 ** 6) == (None, 0.0)


# -- страница считает так же ------------------------------------------------- #


def _node() -> str | None:
    return shutil.which("node")


def _js_storyboard(tmp_path: Path, config: dict) -> list[dict]:
    driver = f"""
const vm = require('node:vm');
const fs = require('node:fs');
const context = {{}};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(RECORDING / 'curtain.js')!r}, 'utf8'), context);
const turns = context.__dbcurtain.build({json.dumps(config)});
process.stdout.write(JSON.stringify(turns));
"""
    script = tmp_path / "storyboard.cjs"
    script.write_text(driver, encoding="utf-8")
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(_node() is None, reason="node недоступен")
@pytest.mark.parametrize(
    "config",
    [
        {"fps": 120, "secs": 5520, "open": 2400, "lanes": 2, "width": 1920, "lead": 180,
         "curtain": 90, "order": [1, 0]},
        {"fps": 60, "secs": 2760, "open": 1200, "lanes": 2, "width": 1920, "lead": 90,
         "curtain": 45, "order": [0, 1]},
        {"fps": 120, "secs": 3120, "open": 600, "lanes": 3, "width": 1920, "lead": 90,
         "curtain": 45, "order": [2, 0]},
    ],
)
def test_page_curtain_matches_python_storyboard(tmp_path, config):
    js = _js_storyboard(tmp_path, config)
    python = layout.storyboard(
        config["order"], secs=config["secs"] / config["fps"],
        open=config["open"] / config["fps"], fps=config["fps"],
        lead=config["lead"] / config["fps"],
        curtain=config["curtain"] / config["fps"], lanes=config["lanes"],
    )
    assert [(turn["lane"], turn["turnStart"], turn["expandFrom"], turn["openFrom"],
             turn["openTo"], turn["returnTo"], turn["turnEnd"]) for turn in js] == [
        (turn.lane, turn.turn_start, turn.expand_from, turn.open_from, turn.open_to,
         turn.return_to, turn.turn_end) for turn in python
    ]


@pytest.mark.skipif(_node() is None, reason="node недоступен")
def test_page_curtain_validates_its_config(tmp_path):
    bad = [
        {"fps": 120, "secs": 2760, "open": 2400, "lanes": 2, "width": 1920, "lead": 180,
         "curtain": 45},
        {"fps": 120, "secs": 2760, "open": 2400, "lanes": 2, "width": 1001, "lead": 180,
         "curtain": 90},
        {"fps": 120, "secs": 2760, "open": 2400, "lanes": 2, "width": 1920, "lead": 180,
         "curtain": 90, "order": [0, 0]},
        {"fps": 120, "secs": 300, "open": 2400, "lanes": 2, "width": 1920, "lead": 180,
         "curtain": 90},
    ]
    script = tmp_path / "bad.cjs"
    script.write_text(
        "const vm=require('node:vm'),fs=require('node:fs');const c={};c.globalThis=c;"
        f"vm.createContext(c);vm.runInContext(fs.readFileSync({str(RECORDING / 'curtain.js')!r},'utf8'),c);"
        "const configs=JSON.parse(process.argv[2]);const bad=[];"
        "for(const cfg of configs){try{c.__dbcurtain.validate(cfg);bad.push('accepted');}"
        "catch(e){bad.push('rejected');}}process.stdout.write(JSON.stringify(bad));",
        encoding="utf-8",
    )
    out = subprocess.run(
        ["node", str(script), json.dumps(bad)], capture_output=True, text=True, check=True,
    )
    assert json.loads(out.stdout) == ["rejected"] * len(bad)


def test_page_driver_carries_the_curtain_stage():
    driver = (RECORDING / "page-driver.js").read_text(encoding="utf-8")
    assert "__dbcurtain" in driver and "dbstage" in driver
    assert "scene.storyboard" in driver and "frameState" in driver
    # Обратная совместимость: без dbstage драйвер ведёт себя как раньше.
    assert "if (staged) {" in driver
    assert "scene.seek = (n) =>" in driver, "штатный seek остаётся для одиночной сцены"


# --------------------------------------------------------------------------- #
# Манифест: уникальность, статусы, причины, починки
# --------------------------------------------------------------------------- #


def _rows(*pairs: tuple[str, str, str]) -> list[dict]:
    return [{"model": model, "scene": scene, "status": status} for model, scene, status in pairs]


def test_manifest_accepts_a_consistent_list():
    rows = _rows(("m", "K-1", "ok"), ("m", "L-2", "not_run"))
    rows[1]["reason"] = "нет задачи"
    assert manifest.check(rows) == []


def test_manifest_catches_duplicates_and_status_lies():
    duplicate = _rows(("m", "K-1", "ok"), ("m", "K-1", "ok"))
    assert len(manifest.by_clip(duplicate)) == 1
    assert any("duplicate" in error for error in manifest.check(duplicate))

    errors = manifest.check(_rows(("m", "K-1", "nonsense")))
    assert any("unknown status" in error for error in errors)

    missing_reason = manifest.check(_rows(("m", "K-1", "model_failed")))
    assert any("without a reason" in error for error in missing_reason)

    ok_with_reason = _rows(("m", "K-1", "ok"))
    ok_with_reason[0]["reason"] = "почему-то"
    assert any("ok carries" in error for error in manifest.check(ok_with_reason))

    bad_flag = _rows(("m", "K-1", "ok"))
    bad_flag[0]["repaired"] = "да"
    assert any("repaired must be a boolean" in error for error in manifest.check(bad_flag))


def test_manifest_notes_must_match_rows():
    rows = _rows(("m", "K-1", "ok"))
    note = {"model": "m", "scene": "K-1", "originalStatus": "failed",
            "reason": "r", "expected": "e"}
    errors = manifest.check(rows, notes=[note])
    assert any("noted as failed" in error for error in errors)

    rows[0]["repaired"] = True
    note = {**note, "scene": "K-9"}
    assert any("not in the manifest" in error for error in manifest.check(rows, notes=[note]))

    note = {**note, "scene": "K-1", "repair": {
        "clip": "c.mp4", "change": "x", "sourceSha256": "a", "repairedSha256": "b"}}
    rows[0]["repaired"] = False
    assert any("without repaired=true" in error for error in manifest.check(rows, notes=[note]))


def test_manifest_files_follow_the_status(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    rows = _rows(("m", "K-1", "ok"), ("m", "L-2", "model_failed"))
    assert len(manifest.check_files(rows, clips)) >= 2
    (clips / "m__K-1.mp4").write_bytes(b"x")
    (clips / "m__K-1.missing").write_text("ok\n", encoding="utf-8")
    (clips / "m__L-2.mp4").write_bytes(b"x")
    (clips / "m__L-2.missing").write_text("model_failed: битая\n", encoding="utf-8")
    errors = manifest.check_files(rows, clips)
    assert any("marker left" in error for error in errors), errors
    (clips / "m__K-1.missing").unlink()
    assert manifest.check_files(rows, clips) == []


# -- исходники: починка раньше, исходник не подменяется --------------------- #


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "_showcase"
    (root / "m" / "pages").mkdir(parents=True)
    (root / "m" / "pages" / "K-1.html").write_text("<html>src</html>", encoding="utf-8")
    return root


def test_repairs_are_preferred_and_reported(tmp_path):
    root = _tree(tmp_path)
    (root.parent / "_repairs" / "m" / "pages").mkdir(parents=True)
    repaired = root.parent / "_repairs" / "m" / "pages" / "K-1.html"
    repaired.write_text("<html>fix</html>", encoding="utf-8")

    page, server_root, is_repaired = manifest.resolve_source("m/K-1", root, "repairs")
    assert page == repaired.resolve() and is_repaired
    assert page.relative_to(server_root).as_posix() == "m/pages/K-1.html"

    plain, _, plain_repaired = manifest.resolve_source("m/K-1", root, "showcase")
    assert plain == (root / "m" / "pages" / "K-1.html").resolve()
    assert not plain_repaired


def test_repairs_mode_falls_back_to_the_original(tmp_path):
    root = _tree(tmp_path)
    page, server_root, repaired = manifest.resolve_source("m/K-1", root, "repairs")
    assert page.name == "K-1.html" and server_root == root.resolve() and not repaired


def test_source_resolution_rejects_bad_names_and_modes(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(ValueError):
        manifest.resolve_source("K-1", root)
    with pytest.raises(ValueError):
        manifest.resolve_source("m/K-1", root, "nonsense")
    with pytest.raises(manifest.ManifestError):
        manifest.resolve_source("m/K-9", root)


def test_flat_tree_layout_is_supported(tmp_path):
    (tmp_path / "K-1.html").write_text("<html>flat</html>", encoding="utf-8")
    page, server_root, repaired = manifest.resolve_source("m/K-1", tmp_path)
    assert page == (tmp_path / "K-1.html").resolve()
    assert server_root == tmp_path.resolve() and not repaired


# --------------------------------------------------------------------------- #
# verify: фейковые кадры без Chrome и ffmpeg
# --------------------------------------------------------------------------- #

SPEC = verify_clips.Spec(seconds=Fraction(1), fps=10, width=32, height=18)


def _framemd5(frames: int, digests: list[str], stamps: list[int] | None = None) -> str:
    timebase = "1/10"
    lines = [f"#tb 0: {timebase}"]
    for index in range(frames):
        stamp = stamps[index] if stamps else index
        lines.append(f"0, {stamp}, {stamp}, 1, 64, {digests[index % len(digests)]}")
    return "\n".join(lines) + "\n"


def _luma(frames: int, values: list[int]) -> bytes:
    size = verify_clips.LUMA_W * verify_clips.LUMA_H
    return b"".join(bytes([values[index % len(values)]]) * size for index in range(frames))


class Vehicle:
    """Подменный ffprobe/ffmpeg: отдаёт заранее собранные ответы."""

    def __init__(self, *, frames: int = 10, digests: tuple[str, ...] = ("aaa", "bbb"),
                 stamps: list[int] | None = None, luma: list[int] = (120,),
                 meta: dict | None = None):
        self.digests = list(digests)
        self.stamps = stamps
        self.luma = list(luma)
        self.frames = frames
        self.meta = meta or {}
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], timeout: int = 300, binary: bool = False):
        self.commands.append(command)
        if Path(command[0]).name.startswith("ffprobe"):
            stream = {
                "codec_name": "h264", "width": SPEC.width, "height": SPEC.height,
                "pix_fmt": "yuv420p", "avg_frame_rate": f"{SPEC.fps}/1",
                "nb_frames": str(self.frames), "bit_rate": "1234",
            }
            stream.update(self.meta.get("stream") or {})
            meta = {
                "streams": [stream],
                "format": {"duration": str(float(SPEC.seconds)), "bit_rate": "1234",
                           **self.meta.get("format", {})},
            }
            return json.dumps(meta)
        if "rawvideo" in command:
            return _luma(self.frames, self.luma)
        return _framemd5(self.frames, self.digests, self.stamps)


def _clip(tmp_path: Path, name: str = "m__K-1.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake mp4")
    return path


def _evidence(path: Path, **overrides) -> dict:
    data = {
        "clip": path.name, "status": "ok", "seconds": 1.0, "fps": SPEC.fps,
        "frames": SPEC.frames, "captured_frames": SPEC.frames, "crf": SPEC.crf,
        "preset": SPEC.preset, "input_format": "png", "clock_start_ms": 0.0,
        "clock_end_ms": (SPEC.frames - 1) * 1000 / SPEC.fps,
        "clock_discontinuities": 0,
    }
    data.update(overrides)
    path.with_suffix(".capture.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def test_verify_accepts_a_healthy_clip(tmp_path):
    row = verify_clips.verify(_clip(tmp_path), SPEC, runner=Vehicle())
    assert row["ok"], row["errors"]
    assert row["decoded_frames"] == SPEC.frames
    assert row["distinct_thumbnails"] == 2
    assert row["black_frames"] == 0
    assert len(row["sha256"]) == 64


@pytest.mark.parametrize(
    "meta, needle",
    [
        ({"stream": {"codec_name": "hevc"}}, "codec_name"),
        ({"stream": {"width": 64}}, "width"),
        ({"stream": {"pix_fmt": "yuv444p"}}, "pix_fmt"),
        ({"stream": {"avg_frame_rate": "20/1"}}, "fps"),
        ({"format": {"duration": "3.0"}}, "duration"),
        ({"stream": {"nb_frames": "9"}}, "container frames"),
    ],
)
def test_verify_catches_a_wrong_container(tmp_path, meta, needle):
    row = verify_clips.verify(_clip(tmp_path), SPEC, runner=Vehicle(meta=meta))
    assert not row["ok"]
    assert any(needle in error for error in row["errors"]), row["errors"]


def test_verify_catches_non_cfr_timestamps(tmp_path):
    stamps = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    row = verify_clips.verify(_clip(tmp_path), SPEC, runner=Vehicle(stamps=stamps))
    assert any("PTS frame 3" in error for error in row["errors"]), row["errors"]


def test_verify_catches_missing_decoded_frames(tmp_path):
    vehicle = Vehicle()
    row = verify_clips.verify(_clip(tmp_path), SPEC, runner=vehicle)
    assert row["ok"]
    row = verify_clips.verify(_clip(tmp_path), SPEC, runner=Vehicle(frames=8))
    assert any("decoded frames" in error for error in row["errors"])


def test_verify_flags_an_all_black_ok_clip(tmp_path):
    row = verify_clips.verify(_clip(tmp_path), SPEC, runner=Vehicle(luma=[0]))
    assert row["ok"] is False
    assert any("entirely black" in error for error in row["errors"])
    assert row["black_ratio"] == 1.0


def test_verify_accepts_a_black_placeholder_and_rejects_a_colorful_one(tmp_path):
    row = verify_clips.verify(
        _clip(tmp_path), SPEC, status="model_failed", runner=Vehicle(luma=[0]),
    )
    assert row["ok"], row["errors"]
    row = verify_clips.verify(
        _clip(tmp_path), SPEC, status="model_failed", runner=Vehicle(luma=[120]),
    )
    assert any("is not black" in error for error in row["errors"])


def test_verify_reports_a_black_run_without_failing(tmp_path):
    row = verify_clips.verify(
        _clip(tmp_path), SPEC, runner=Vehicle(luma=[120, 0, 0, 120]),
    )
    assert row["ok"], row["errors"]
    assert row["black_frames"] == 5
    assert row["longest_black_seconds"] == pytest.approx(0.2)


def test_verify_compares_the_manifest_sha(tmp_path):
    row = verify_clips.verify(_clip(tmp_path), SPEC, runner=Vehicle(), sha256="0" * 64)
    assert any("sha256" in error for error in row["errors"])


def test_verify_requires_proof_of_capture(tmp_path):
    path = _clip(tmp_path)
    row = verify_clips.verify(path, SPEC, runner=Vehicle(), evidence=True)
    assert any("evidence unreadable" in error for error in row["errors"])

    _evidence(path, input_format="jpeg")
    row = verify_clips.verify(path, SPEC, runner=Vehicle(), evidence=True)
    assert any("input_format" in error for error in row["errors"])

    _evidence(path, clock_discontinuities=1)
    row = verify_clips.verify(path, SPEC, runner=Vehicle(), evidence=True)
    assert any("clock is not continuous" in error for error in row["errors"])

    _evidence(path, clock_end_ms=999.0)
    row = verify_clips.verify(path, SPEC, runner=Vehicle(), evidence=True)
    assert any("clock span" in error for error in row["errors"])

    _evidence(path, status="capture_failed")
    row = verify_clips.verify(path, SPEC, runner=Vehicle(), evidence=True)
    assert any("capture status" in error for error in row["errors"])


def test_quality_report_keeps_failures_in_the_count():
    rows = [{"file": "b.mp4", "ok": False}, {"file": "a.mp4", "ok": True}]
    report = verify_clips.quality_report(rows, SPEC)
    assert report["clips"] == 2 and report["passed"] == 1 and report["failed"] == 1
    assert [row["file"] for row in report["results"]] == ["a.mp4", "b.mp4"]
    assert report["black_luma_threshold"] == verify_clips.BLACK_LUMA
    assert "never treated as a capture failure" in report["notes"]


def test_write_report_is_atomic(tmp_path):
    target = tmp_path / "recording-quality.json"
    verify_clips.write_report({"clips": 0}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"clips": 0}
    assert not list(tmp_path.glob(".*.part"))


def test_rows_without_manifest_take_the_status_from_the_marker(tmp_path):
    _clip(tmp_path, "a.mp4")
    _clip(tmp_path, "b.mp4")
    (tmp_path / "b.missing").write_text("model_failed\n", encoding="utf-8")
    pairs = verify_clips.rows_for(tmp_path)
    assert {path.name: row["status"] for path, row in pairs} == {
        "a.mp4": "ok", "b.mp4": "media_missing",
    }


# -- CLI -------------------------------------------------------------------- #


def _manifest(tmp_path: Path, status: str = "ok") -> Path:
    path = tmp_path / "clips.json"
    path.write_text(json.dumps([{"model": "m", "scene": "K-1", "status": status}]),
                    encoding="utf-8")
    return path


def test_cli_writes_a_report_and_passes(tmp_path, monkeypatch, capsys):
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips)
    seen: list[dict] = []

    def fake(path, spec, **kwargs):
        seen.append({"path": path, **kwargs})
        return {"file": Path(path).name, "ok": True, "errors": [], "status": kwargs["status"]}

    monkeypatch.setattr(verify_clips, "verify", fake)
    code = verify_clips.main([str(clips), "--no-evidence", "--output",
                              str(tmp_path / "q.json")])
    assert code == 0
    report = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert report["passed"] == 1 and seen[0]["status"] == "ok"


def test_cli_fails_on_a_failure_row(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips)
    monkeypatch.setattr(
        verify_clips, "verify",
        lambda path, spec, **kwargs: {"file": Path(path).name, "ok": False,
                                      "errors": ["boom"], "status": "ok"},
    )
    assert verify_clips.main([str(clips), "--no-evidence", "--output",
                              str(tmp_path / "q.json")]) == 1


@pytest.mark.parametrize("rows", [
    [{"model": "m", "scene": "K-1", "status": "ok"},
     {"model": "m", "scene": "K-1", "status": "ok"}],
    [{"model": "m", "scene": "K-1", "status": "nonsense"}],
])
def test_cli_rejects_a_broken_manifest(tmp_path, rows, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips)
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(verify_clips, "verify",
                        lambda *a, **k: {"file": "m__K-1.mp4", "ok": True, "errors": []})
    with pytest.raises(SystemExit):
        verify_clips.main([str(clips), "--manifest", str(path), "--no-evidence"])


def test_cli_manifest_requires_clicks_of_every_pair(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    path = _manifest(tmp_path)
    monkeypatch.setattr(verify_clips, "verify",
                        lambda *a, **k: {"file": "m__K-1.mp4", "ok": True, "errors": []})
    code = verify_clips.main([str(clips), "--manifest", str(path), "--no-evidence",
                              "--output", str(tmp_path / "q.json")])
    assert code == 1
    report = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert any("clip missing" in error for error in report["manifest_errors"])


# --------------------------------------------------------------------------- #
# record_scene: параметры, паспорт, манифест
# --------------------------------------------------------------------------- #


def test_options_carry_the_whole_contract():
    options = record_scene.Options()
    assert (options.secs, options.open) == (23.0, 20.0), "запись - целый ход"
    assert (options.fps, options.display_fps) == (120, 60)
    assert (options.lanes, options.curtain, options.lead) == (2, 0.75, 1.5)
    assert options.frames == 2760 and options.turns == 1
    assert options.layout.lane_width == 960
    assert options.dt == pytest.approx(1 / 120)


def test_options_are_frozen_and_validated():
    options = record_scene.Options()
    with pytest.raises(Exception):
        options.secs = 10  # type: ignore[misc]
    with pytest.raises(ValueError):
        record_scene.Options(curtain=0.5)
    with pytest.raises(ValueError):
        record_scene.Options(frame_format="webp")
    with pytest.raises(ValueError):
        record_scene.Options(source_mode="mirror")
    with pytest.raises(ValueError):
        record_scene.Options(width=1001)


def test_stage_url_is_added_only_for_several_lanes():
    options = record_scene.Options()
    plain = record_scene.Options(lanes=1)
    url = "http://host/scene.html?a=1"
    assert record_scene.stage_url(url, plain) == url
    staged = record_scene.stage_url(url, options)
    assert "dbstage=1" in staged and "a=1" in staged and "frames=2760" in staged
    assert "open=20" in staged and "secs=23" in staged
    assert "lanes=2" in staged and "curtain=0.75" in staged


def test_capture_passport_records_the_clock(tmp_path):
    options = record_scene.Options(secs=5.2, open=2.0, fps=10, display_fps=10,
                                   curtain=0.8, lead=1.6)
    out = tmp_path / "m__K-1.mp4"
    target = record_scene.write_capture(out, options, "ok", mode="stage-seek")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["status"] == "ok" and data["frames"] == 52
    assert data["input_format"] == "png" and data["clock_discontinuities"] == 0
    assert data["clock_end_ms"] == pytest.approx(5100.0)
    assert data["clock_modes"] == ["stage-seek"]
    assert not list(tmp_path.glob("*.tmp"))

    failed = record_scene.write_capture(out, options, "model_failed", reason="boom")
    data = json.loads(failed.read_text(encoding="utf-8"))
    assert data["status"] == "model_failed" and data["captured_frames"] == 0
    assert data["input_format"] == "generated-placeholder" and data["reason"] == "boom"


def test_manifest_gate_rejects_unknown_pairs(tmp_path):
    options = record_scene.Options(manifest=tmp_path / "clips.json")
    options.manifest.write_text(
        json.dumps([{"model": "m", "scene": "K-1", "status": "ok"}]), encoding="utf-8",
    )
    record_scene.check_manifest(options, "m", "K-1")
    with pytest.raises(manifest.ManifestError):
        record_scene.check_manifest(options, "m", "K-2")
    options.manifest.write_text(json.dumps([{"model": "m", "scene": "K-1"}]), encoding="utf-8")
    with pytest.raises(manifest.ManifestError):
        record_scene.check_manifest(options, "m", "K-1")
    assert record_scene.Options(manifest=None).manifest is None


def test_batch_discovers_the_page_tree(tmp_path):
    (tmp_path / "m" / "pages").mkdir(parents=True)
    (tmp_path / "m" / "pages" / "K-1.html").write_text("x", encoding="utf-8")
    (tmp_path / "m" / "pages" / "note.txt").write_text("x", encoding="utf-8")
    assert record_batch.discover(tmp_path) == [("m", "K-1")]


# --------------------------------------------------------------------------- #
# Сквозные пробы с настоящим ffmpeg
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
def test_placeholder_is_black_and_exactly_one_second(tmp_path):
    options = record_scene.Options(secs=5.2, open=2.0, fps=10, display_fps=10,
                                   curtain=0.8, lead=1.6, lanes=1)
    out = tmp_path / "plate.mp4"
    record_scene.placeholder(out, options)
    row = verify_clips.verify(out, verify_clips.Spec(seconds=Fraction(52, 10), fps=10),
                              status="model_failed")
    assert row["ok"], row["errors"]
    assert row["black_ratio"] == 1.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
def test_verify_rejects_a_placeholder_claimed_as_ok(tmp_path):
    options = record_scene.Options(secs=5.2, open=2.0, fps=10, display_fps=10,
                                   curtain=0.8, lead=1.6, lanes=1)
    out = tmp_path / "plate.mp4"
    record_scene.placeholder(out, options)
    row = verify_clips.verify(out, verify_clips.Spec(seconds=Fraction(52, 10), fps=10),
                              status="ok")
    assert not row["ok"]
    assert any("entirely black" in error for error in row["errors"])
