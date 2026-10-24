#!/usr/bin/env python3
"""Build speech, music, and AAC master audio from video/src/data/edit.json."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

VIDEO = Path(__file__).resolve().parents[1]
PUBLIC = VIDEO / "public"
EDIT_DEFAULT = VIDEO / "src" / "data" / "edit.json"
RATE = 48_000
FPS = 60
LOOP_CROSSFADE_SECONDS = 4.0
CUE_CROSSFADE_SECONDS = 3.0
VOICE_PREROLL_SECONDS = 0.15
VOICE_RELEASE_SECONDS = 0.55
VOICE_DUCK_DB = -10.0
MUSIC_NORMAL_DBFS = -34.0
MUSIC_DUCKED_DBFS = -44.0
MUSIC_SHOWCASE_DBFS = -27.0
MUSIC_REFERENCE_RMS_DBFS = -20.0
MUSIC_PEAK_CEILING_DBFS = -8.0

TRACKS = {
    "background": {
        "file": "signal-to-noise-no-piano.mp3",
        "title": "Signal to Noise (No Piano Melody)",
        "artist": "Scott Buckley",
        "startSeconds": 0.0,
        "page": "https://www.scottbuckley.com.au/library/signal-to-noise/",
        "download": "https://www.scottbuckley.com.au/library/wp-content/uploads/2020/04/sb_signaltonoise_nomelody.mp3",
        "credit": "'Signal to Noise' by Scott Buckley - released under CC-BY 4.0. www.scottbuckley.com.au",
    },
    "showcase-K": {
        "file": "electric-dreams.mp3",
        "title": "Electric Dreams",
        "artist": "Scott Buckley",
        "startSeconds": 20.0,
        "page": "https://www.scottbuckley.com.au/library/electric-dreams/",
        "download": "https://www.scottbuckley.com.au/library/wp-content/uploads/2020/09/sb_electricdreams.mp3",
        "credit": "'Electric Dreams' by Scott Buckley - released under CC-BY 4.0. www.scottbuckley.com.au",
    },
    "showcase-L": {
        "file": "reparateur.mp3",
        "title": "Reparateur",
        "artist": "Scott Buckley",
        "startSeconds": 20.0,
        "page": "https://www.scottbuckley.com.au/library/reparateur/",
        "download": "http://www.scottbuckley.com.au/wp-content/audio/sb_reparateur.mp3",
        "credit": "'Reparateur' by Scott Buckley - released under CC-BY 4.0. www.scottbuckley.com.au",
    },
}
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def fail(message: str) -> None:
    raise ValueError(message)


def number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        fail(f"{name} must be >= {minimum}")
    return result


def frame_number(value: Any, name: str, *, minimum: int = 0) -> int:
    result = number(value, name, minimum=minimum)
    if not result.is_integer():
        fail(f"{name} must be an integer frame count")
    return int(result)


def fmt(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def db_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def find_binary(env_name: str, name: str) -> str:
    configured = os.environ.get(env_name)
    found = configured or shutil.which(name)
    if not found:
        fail(f"{name} not found; install FFmpeg or set {env_name}")
    return found


def run_capture(command: list[str], *, label: str, stdout: Any = subprocess.PIPE) -> str:
    result = subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode:
        error = result.stderr or "(no FFmpeg diagnostic)"
        raise RuntimeError(f"{label} failed (exit {result.returncode}):\n{error[-6000:]}")
    return result.stderr or ""


def probe_audio(ffprobe: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,duration:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"ffprobe could not read audio source {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        fail(f"No audio stream in source: {path}")
    stream_duration = streams[0].get("duration")
    raw_duration = stream_duration or data.get("format", {}).get("duration")
    if raw_duration is None:
        fail(f"Could not determine audio duration: {path}")
    return {"durationSeconds": float(raw_duration), **streams[0]}


def source_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        fail(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = PUBLIC / path
    path = path.resolve()
    if not path.is_file():
        fail(f"Audio source does not exist: {path}")
    return path


def output_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = VIDEO / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scene_label(scene: dict[str, Any]) -> str:
    value = scene.get("id")
    if not isinstance(value, str) or not value.strip():
        fail("Every scene must have a non-empty string id")
    return value.strip()


def role_for_id(value: str) -> str | None:
    text = value.casefold().replace("_", "-").replace(" ", "-")
    if text == "hero":
        return "showcase-K"
    if "showcase" not in text and text not in {"k", "l"}:
        return None
    if re.search(r"(?:^|-)l(?:$|-|\d)", text) or text == "l":
        return "showcase-L"
    if re.search(r"(?:^|-)k(?:$|-|\d)", text) or text in {"k", "showcase"}:
        return "showcase-K"
    return "showcase-K"


def is_long_showcase_scene(value: str) -> bool:
    text = value.casefold().replace("_", "-").replace(" ", "-")
    return text in {"showcase", "hero"} or bool(re.match(r"^showcase-(?:k|l)(?:$|\d|-)", text))


def merge_ranges(ranges: list[tuple[float, float]], *, gap: float = 0.0) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if not result or start > result[-1][1] + gap:
            result.append((start, end))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], end))
    return result


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
    if not path.is_file():
        fail(f"Edit manifest not found: {path}; parent creates it after transcription")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        fail("edit.json must contain a JSON object")
    fps = number(manifest.get("fps"), "fps", minimum=1)
    if fps != FPS:
        fail("The edit manifest contract uses fps: 60; all from/duration frame values are 60-fps frames")
    total_frames = frame_number(manifest.get("totalFrames"), "totalFrames", minimum=1)
    raw_scenes = manifest.get("scenes")
    raw_narration = manifest.get("narration")
    if not isinstance(raw_scenes, list) or not isinstance(raw_narration, list):
        fail("edit.json must contain scenes[] and narration[] arrays")

    scenes: list[dict[str, Any]] = []
    for index, scene in enumerate(raw_scenes):
        if not isinstance(scene, dict):
            fail(f"scenes[{index}] must be an object")
        identifier = scene_label(scene)
        start = frame_number(scene.get("from"), f"scenes[{index}].from")
        duration = frame_number(scene.get("duration"), f"scenes[{index}].duration", minimum=1)
        if start + duration > total_frames:
            fail(f"Scene {identifier!r} extends beyond totalFrames")
        scenes.append({"id": identifier, "from": start, "duration": duration})

    narration: list[dict[str, Any]] = []
    for index, item in enumerate(raw_narration):
        if not isinstance(item, dict):
            fail(f"narration[{index}] must be an object")
        start = frame_number(item.get("from"), f"narration[{index}].from")
        source_start = number(item.get("sourceStartSeconds"), f"narration[{index}].sourceStartSeconds", minimum=0)
        duration = number(item.get("durationSeconds"), f"narration[{index}].durationSeconds", minimum=0.001)
        if start + duration * FPS > total_frames + 1e-6:
            fail(f"narration[{index}] ends beyond totalFrames")
        src = source_path(item.get("src"), f"narration[{index}].src")
        identifier = item.get("id", f"narration-{index}")
        scene_id = item.get("scene")
        if not isinstance(identifier, str) or not identifier.strip():
            fail(f"narration[{index}].id must be a non-empty string")
        if scene_id is not None and not isinstance(scene_id, str):
            fail(f"narration[{index}].scene must be a string when provided")
        narration.append({"index": index, "id": identifier, "scene": scene_id, "src": src,
                          "manifestSrc": item["src"], "sourceStartSeconds": source_start,
                          "durationSeconds": duration, "from": start,
                          "originalTake": item.get("originalTake")})

    music_cues: list[dict[str, Any]] = []
    cue_lists: list[tuple[str, Any]] = []
    if "musicCues" in manifest:
        cue_lists.append(("musicCues", manifest["musicCues"]))
    video_edit = manifest.get("videoEdit")
    if isinstance(video_edit, dict) and "musicCues" in video_edit:
        cue_lists.append(("videoEdit.musicCues", video_edit["musicCues"]))
    for source, items in cue_lists:
        if not isinstance(items, list):
            fail(f"{source} must be an array of {{id, from}} objects")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                fail(f"{source}[{index}] must be an object")
            identifier = scene_label(item)
            start = frame_number(item.get("from"), f"{source}[{index}].from")
            if start >= total_frames:
                fail(f"Music cue {identifier!r} starts at or after totalFrames")
            music_cues.append({"id": identifier, "from": start, "source": source})
    return manifest, scenes, narration, total_frames, music_cues


def showcase_ranges(scenes: list[dict[str, Any]], cues: list[dict[str, Any]], total_frames: int) -> list[tuple[int, int]]:
    ranges = [
        (scene["from"], scene["from"] + scene["duration"])
        for scene in scenes if is_long_showcase_scene(scene["id"])
    ]
    if not ranges:
        cue_roles = [(cue["from"], role_for_id(cue["id"])) for cue in cues]
        k_starts = [start for start, role in cue_roles if role == "showcase-K"]
        l_starts = [start for start, role in cue_roles if role == "showcase-L"]
        if k_starts and l_starts:
            start = min(k_starts)
            later_non_showcase = [cue["from"] for cue in cues
                                  if cue["from"] > max(l_starts) and role_for_id(cue["id"]) is None]
            end = min(later_non_showcase) if later_non_showcase else total_frames
            ranges.append((start, end))
    return [(int(start), int(end)) for start, end in merge_ranges([(float(a), float(b)) for a, b in ranges])]


def make_music_segments(scenes: list[dict[str, Any]], cues: list[dict[str, Any]], total_frames: int,
                        show_ranges: list[tuple[int, int]]) -> list[dict[str, Any]]:
    events: dict[int, tuple[int, str, str]] = {}

    def add(frame: int, priority: int, role: str, label: str) -> None:
        old = events.get(frame)
        if old is None or priority >= old[0]:
            events[frame] = (priority, role, label)

    add(0, 0, "background", "default background")
    for scene in scenes:
        if not is_long_showcase_scene(scene["id"]):
            continue
        role = role_for_id(scene["id"]) or "showcase-K"
        add(scene["from"], 1, role, scene["id"])
    for start, end in show_ranges:
        add(end, 1, "background", "end of showcase")
    for cue in cues:
        role = role_for_id(cue["id"]) or "background"
        add(cue["from"], 2, role, cue["id"])

    points = sorted(frame for frame in events if frame < total_frames)
    if not points or points[0] != 0:
        points.insert(0, 0)
    segments: list[dict[str, Any]] = []
    for index, start in enumerate(points):
        end = points[index + 1] if index + 1 < len(points) else total_frames
        if end <= start:
            continue
        role = events[start][1]
        labels = [events[start][2]]
        if segments and segments[-1]["role"] == role and segments[-1]["endFrame"] == start:
            segments[-1]["endFrame"] = end
            segments[-1]["cueIds"].extend(labels)
        else:
            segments.append({"role": role, "startFrame": start, "endFrame": end, "cueIds": labels})
    saw_k = False
    k_to_l = False
    for segment in segments:
        if segment["role"] == "showcase-K":
            saw_k = True
        elif segment["role"] == "showcase-L" and saw_k:
            k_to_l = True
    if not k_to_l:
        fail("Music needs a K→L showcase transition. Include showcase-K start and showcase-L boundary cues/scenes (all from values are 60-fps frames).")
    return segments


def validate_narration_outside_showcase(narration: list[dict[str, Any]], ranges: list[tuple[int, int]]) -> None:
    for clip in narration:
        start = clip["from"] / FPS
        end = start + clip["durationSeconds"]
        for show_start, show_end in ranges:
            lo, hi = show_start / FPS, show_end / FPS
            if start < hi and end > lo:
                fail(
                    f"narration[{clip['index']}] overlaps long showcase ({lo:.3f}–{hi:.3f}s). "
                    "The builder will not truncate a speech clip; remove/re-time it in edit.json so showcase narration is zero."
                )


def new_temp_path(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(prefix=f".{destination.stem}.", suffix=destination.suffix,
                                     dir=destination.parent, delete=False) as handle:
        path = Path(handle.name)
    path.unlink()
    return path


def run_ffmpeg(ffmpeg: str, command: list[str], label: str) -> str:
    return run_capture([ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info", *command], label=label)


def parse_astats(log: str) -> dict[str, float | None]:
    def last(pattern: str) -> float | None:
        matches = re.findall(pattern, log, flags=re.IGNORECASE)
        if not matches:
            return None
        raw = matches[-1]
        return None if raw.casefold() == "-inf" else float(raw)

    return {
        "rmsDbfs": last(r"RMS level dB:\s*(-?\d+(?:\.\d+)?|-inf)"),
        "peakDbfs": last(r"Peak level dB:\s*(-?\d+(?:\.\d+)?|-inf)"),
    }


def source_levels(ffmpeg: str, path: Path) -> tuple[float, float]:
    log = run_ffmpeg(ffmpeg, ["-i", str(path), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
                     f"Measuring music source {path.name}")
    rms_values = re.findall(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log, flags=re.IGNORECASE)
    peak_values = re.findall(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log, flags=re.IGNORECASE)
    if not rms_values or not peak_values:
        fail(f"Could not measure RMS and peak levels of music source {path}")
    return float(rms_values[-1]), float(peak_values[-1])


def build_voice(ffmpeg: str, ffprobe: str, narration: list[dict[str, Any]], total_frames: int,
                destination: Path) -> tuple[Path, dict[str, Any]]:
    duration = total_frames / FPS
    final_fade = min(3.0, duration / 2)
    temp = new_temp_path(destination)
    sources: list[Path] = []
    source_info: dict[Path, dict[str, Any]] = {}
    source_index: dict[Path, int] = {}
    for clip in narration:
        if clip["src"] not in source_index:
            source_info[clip["src"]] = probe_audio(ffprobe, clip["src"])
            source_index[clip["src"]] = len(sources)
            sources.append(clip["src"])
        info = source_info[clip["src"]]
        if clip["sourceStartSeconds"] + clip["durationSeconds"] > info["durationSeconds"] + 0.05:
            fail(f"narration[{clip['index']}] asks past the end of {clip['src']} ("
                 f"{info['durationSeconds']:.3f}s available)")

    grouped: dict[Path, list[dict[str, Any]]] = {source: [] for source in sources}
    for clip in narration:
        grouped[clip["src"]].append(clip)

    command: list[str] = ["-y", "-filter_complex_threads", "1"]
    for source in sources:
        command.extend(["-i", str(source)])
    graph: list[str] = []
    for source, clips in grouped.items():
        input_index = source_index[source]
        split_labels = [f"vsrc{input_index}_{number}" for number in range(len(clips))]
        graph.append(
            f"[{input_index}:a:0]aresample={RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS,asplit={len(clips)}" + "".join(f"[{label}]" for label in split_labels)
        )
        for clip, split_label in zip(clips, split_labels):
            label = f"voiceclip{clip['index']}"
            samples = round(clip["durationSeconds"] * RATE)
            graph.append(
                f"[{split_label}]atrim=start={fmt(clip['sourceStartSeconds'])}:duration={fmt(clip['durationSeconds'])},"
                "asetpts=PTS-STARTPTS,highpass=f=70:p=2,"
                "acompressor=threshold=0.18:ratio=1.4:attack=20:release=250:makeup=1,"
                f"apad=whole_len={samples},atrim=end_sample={samples}[{label}]"
            )

    timeline_labels: list[str] = []
    cursor = 0
    total_samples = total_frames * (RATE // FPS)
    points = [
        (clip["from"] * (RATE // FPS), f"[voiceclip{clip['index']}]", round(clip["durationSeconds"] * RATE))
        for clip in sorted(narration, key=lambda item: item["from"])
    ]
    points.append((total_samples, None, 0))
    for start, label, samples in points:
        if start < cursor:
            fail("Narration clips overlap; place each spoken fragment after the preceding one")
        if start > cursor:
            silence = f"silence{len(timeline_labels)}"
            graph.append(
                f"anullsrc=r={RATE}:cl=stereo,atrim=end_sample={start - cursor},"
                f"asetpts=PTS-STARTPTS,aformat=sample_fmts=fltp[{silence}]"
            )
            timeline_labels.append(f"[{silence}]")
        if label is not None:
            timeline_labels.append(label)
        cursor = start + samples
    normalization = "loudnorm=I=-19:TP=-3:LRA=7:linear=false," if narration else ""
    graph.append(
        "".join(timeline_labels) + f"concat=n={len(timeline_labels)}:v=0:a=1,{normalization}"
        f"aresample={RATE},apad=whole_len={total_samples},atrim=end_sample={total_samples},"
        f"afade=t=out:st={fmt(duration - final_fade)}:d={fmt(final_fade)},"
        "astats=metadata=0:reset=0[voice]"
    )
    command.extend(["-filter_complex", ";".join(graph), "-map", "[voice]", "-c:a", "pcm_s24le",
                    "-ar", str(RATE), "-ac", "2", str(temp)])
    try:
        log = run_ffmpeg(ffmpeg, command, "Building narration.wav")
        info = probe_audio(ffprobe, temp)
        if abs(info["durationSeconds"] - duration) > 1 / RATE:
            fail(f"Narration duration mismatch: {info['durationSeconds']:.6f}s, expected {duration:.6f}s")
        if narration and parse_astats(log)["peakDbfs"] is None:
            fail("Narration contains no finite, measurable voice samples")
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return temp, {
        "path": str(destination),
        **parse_astats(log),
        "sourceCount": len(sources),
        "clipCount": len(narration),
        "filters": (["highpass 70 Hz, 2-pole", "soft compressor 1.4:1",
                     "EBU R128 loudness normalization -19 LUFS / -3 dBTP; no time stretching"]
                    if narration else ["zero-filled silence; no narration sources"]),
        "finalFadeOutSeconds": final_fade,
    }

def escape_expr(value: str) -> str:
    return value.replace("'", "\\'")


def duck_expression(narration: list[dict[str, Any]]) -> str:
    intervals = merge_ranges([
        (clip["from"] / FPS, clip["from"] / FPS + clip["durationSeconds"])
        for clip in narration
    ], gap=VOICE_PREROLL_SECONDS + VOICE_RELEASE_SECONDS)
    if not intervals:
        return "1"
    low = db_gain(VOICE_DUCK_DB)
    parts: list[tuple[float, float, str]] = []
    for start, end in intervals:
        pre = max(0.001, VOICE_PREROLL_SECONDS)
        release = max(0.001, VOICE_RELEASE_SECONDS)
        begin = max(0.0, start - pre)
        env = (
            f"if(lt(t,{fmt(start)}),1-((1-{fmt(low)})*(t-{fmt(begin)})/{fmt(max(0.001, start - begin))}),"
            f"if(lt(t,{fmt(end)}),{fmt(low)},"
            f"if(lt(t,{fmt(end + release)}),{fmt(low)}+((1-{fmt(low)})*(t-{fmt(end)})/{fmt(release)}),1)))"
        )
        parts.append((begin, end + release, env))
    expression = "1"
    for begin, end, env in reversed(parts):
        expression = f"if(between(t,{fmt(begin)},{fmt(end)}),{env},{expression})"
    return expression


def segment_window(segment: dict[str, Any], index: int, segments: list[dict[str, Any]]) -> str:
    start = segment["startFrame"] / FPS
    end = segment["endFrame"] / FPS
    if index == 0:
        fade_in = min(2.0, (end - start) / 2)
    else:
        prev_len = (segments[index - 1]["endFrame"] - segments[index - 1]["startFrame"]) / FPS
        fade_in = min(CUE_CROSSFADE_SECONDS, prev_len / 2, (end - start) / 2)
    if index + 1 == len(segments):
        fade_out = 0.0
    else:
        next_len = (segments[index + 1]["endFrame"] - segments[index + 1]["startFrame"]) / FPS
        fade_out = min(CUE_CROSSFADE_SECONDS, (end - start) / 2, next_len / 2)
    if fade_in > 0:
        start_expr = f"if(lt(t,{fmt(start + fade_in)}),sin((t-{fmt(start)})/{fmt(fade_in)}*PI/2),"
    else:
        start_expr = ""
    if fade_out > 0:
        middle_expr = f"if(lt(t,{fmt(end - fade_out)}),1,if(lt(t,{fmt(end)}),cos((t-{fmt(end - fade_out)})/{fmt(fade_out)}*PI/2),0))"
    else:
        middle_expr = f"if(lt(t,{fmt(end)}),1,0)"
    if fade_in > 0:
        return f"if(lt(t,{fmt(start)}),0,{start_expr}{middle_expr}))"
    return f"if(lt(t,{fmt(start)}),0,{middle_expr})"


def make_loop_file(ffmpeg: str, source: Path, track: dict[str, Any], source_duration: float,
                   destination: Path) -> None:
    fade = LOOP_CROSSFADE_SECONDS
    offset = float(track["startSeconds"])
    graph = ";".join([
        (f"[0:a:0]aresample={RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
         "asetpts=PTS-STARTPTS,asplit=3[mid-src][tail-src][head-src]"),
        (f"[mid-src]atrim=start={fmt(offset + fade)}:end={fmt(source_duration - fade)},"
         "asetpts=PTS-STARTPTS[mid]"),
        (f"[tail-src]atrim=start={fmt(source_duration - fade)}:end={fmt(source_duration)},"
         "asetpts=PTS-STARTPTS[tail]"),
        (f"[head-src]atrim=start={fmt(offset)}:end={fmt(offset + fade)},"
         "asetpts=PTS-STARTPTS[head]"),
        f"[tail][head]acrossfade=d={fmt(fade)}:c1=tri:c2=tri[xfade]",
        ("[mid][xfade]concat=n=2:v=0:a=1,asetpts=PTS-STARTPTS,"
         f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo[cycle]"),
    ])
    command = ["-y", "-filter_complex_threads", "1", "-i", str(source), "-filter_complex", graph,
               "-map", "[cycle]", "-c:a", "pcm_s24le", "-ar", str(RATE), "-ac", "2", str(destination)]
    run_ffmpeg(ffmpeg, command, f"Preparing seamless loop for {track['title']}")


def build_music(ffmpeg: str, ffprobe: str, segments: list[dict[str, Any]], narration: list[dict[str, Any]],
                total_frames: int, destination: Path) -> tuple[Path, dict[str, Any]]:
    duration = total_frames / FPS
    temp = new_temp_path(destination)
    roles = sorted({segment["role"] for segment in segments})
    used_tracks = {role: TRACKS[role] for role in roles}
    paths: dict[str, Path] = {}
    source_info: dict[str, dict[str, Any]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    loop_samples: dict[str, int] = {}
    loop_directory = tempfile.TemporaryDirectory(prefix=".audio-loops.", dir=destination.parent)
    try:
        for role, track in used_tracks.items():
            path = (PUBLIC / "audio" / "music" / track["file"]).resolve()
            if not path.is_file():
                fail(f"Selected music asset is missing: {path}")
            info = probe_audio(ffprobe, path)
            offset = float(track["startSeconds"])
            if info["durationSeconds"] <= offset + 2 * LOOP_CROSSFADE_SECONDS + 1:
                fail(f"Music source is too short for a {LOOP_CROSSFADE_SECONDS:g}s loop crossfade: {path}")
            rms, peak = source_levels(ffmpeg, path)
            gain = min(MUSIC_REFERENCE_RMS_DBFS - rms, MUSIC_PEAK_CEILING_DBFS - peak)
            paths[role] = path
            source_info[role] = info
            source_stats[role] = {
                "sourceRmsDbfs": rms,
                "sourcePeakDbfs": peak,
                "durationSeconds": info["durationSeconds"],
                "normalizationGainDb": gain,
                "normalizedRmsDbfs": rms + gain,
                "normalizedPeakDbfs": peak + gain,
            }

        loop_paths: dict[str, Path] = {}
        for index, (role, track) in enumerate(used_tracks.items()):
            loop_path = Path(loop_directory.name) / f"track-{index}.wav"
            make_loop_file(ffmpeg, paths[role], track, source_info[role]["durationSeconds"], loop_path)
            loop_paths[role] = loop_path
            loop_samples[role] = round(probe_audio(ffprobe, loop_path)["durationSeconds"] * RATE)

        command: list[str] = ["-y", "-filter_complex_threads", "1"]
        for role in roles:
            command.extend(["-stream_loop", "-1", "-i", str(loop_paths[role])])
        graph: list[str] = []
        loop_labels: dict[str, str] = {}
        for index, role in enumerate(roles):
            label = f"loop{index}"
            normalized_gain = db_gain(source_stats[role]["normalizationGainDb"])
            graph.append(
                f"[{index}:a:0]aresample={RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"atrim=duration={fmt(duration)},asetpts=PTS-STARTPTS,volume={fmt(normalized_gain)}[{label}]"
            )
            loop_labels[role] = label

        duck = escape_expr(duck_expression(narration))
        final_fade = min(3.0, duration / 2)
        fade_in = min(2.0, duration / 2)
        global_gain = (
            f"if(lt(t,{fmt(fade_in)}),sin(t/{fmt(fade_in)}*PI/2),"
            f"if(lt(t,{fmt(duration - final_fade)}),1,"
            f"if(lt(t,{fmt(duration)}),cos((t-{fmt(duration - final_fade)})/{fmt(final_fade)}*PI/2),0)))"
        )
        branch_labels: list[str] = []
        for role in roles:
            terms: list[str] = []
            for index, segment in enumerate(segments):
                if segment["role"] != role:
                    continue
                window = segment_window(segment, index, segments)
                target_db = MUSIC_SHOWCASE_DBFS if role.startswith("showcase-") else MUSIC_NORMAL_DBFS
                base_gain = db_gain(target_db - MUSIC_REFERENCE_RMS_DBFS)
                level_expr = f"({fmt(base_gain)}*({duck}))"
                terms.append(f"({window})*({level_expr})")
            if not terms:
                continue
            label = f"bed{len(branch_labels)}"
            graph.append(f"[{loop_labels[role]}]volume='{escape_expr('+'.join(terms))}':eval=frame[{label}]")
            branch_labels.append(f"[{label}]")
        if not branch_labels:
            fail("No music cue intervals were produced")
        graph.append(
            "".join(branch_labels) + f"amix=inputs={len(branch_labels)}:duration=longest:normalize=0:dropout_transition=0,"
            f"volume='{global_gain}':eval=frame,"
            f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo,"
            "astats=metadata=0:reset=0[music]"
        )
        command.extend(["-filter_complex", ";".join(graph), "-map", "[music]", "-c:a", "pcm_s24le",
                        "-ar", str(RATE), "-ac", "2", str(temp)])
        log = run_ffmpeg(ffmpeg, command, "Building music.wav")
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    finally:
        loop_directory.cleanup()

    track_sources: list[dict[str, Any]] = []
    for role, track in used_tracks.items():
        track_sources.append({
            "role": role,
            "title": track["title"],
            "artist": track["artist"],
            "file": f"audio/music/{track['file']}",
            "localSourceFile": track["file"],
            "page": track["page"],
            "download": track["download"],
            "license": "CC BY 4.0",
            "licenseUrl": LICENSE_URL,
            "licenseEvidencePage": track["page"],
            "credit": track["credit"],
            "sourceStartSeconds": track["startSeconds"],
            **source_stats[role],
            "loopCrossfadeSeconds": LOOP_CROSSFADE_SECONDS,
            "loopCycleSamplesAt48k": loop_samples[role],
        })
    return temp, {
        "path": str(destination),
        **parse_astats(log),
        "sampleRate": RATE,
        "channels": 2,
        "encoding": "PCM 24-bit WAV",
        "targetRmsDbfs": {"underNarration": MUSIC_DUCKED_DBFS, "normalBed": MUSIC_NORMAL_DBFS,
                           "showcase": MUSIC_SHOWCASE_DBFS},
        "sourcePeakCeilingAfterNormalizationDbfs": MUSIC_PEAK_CEILING_DBFS,
        "voiceDuckDb": VOICE_DUCK_DB,
        "voicePrerollSeconds": VOICE_PREROLL_SECONDS,
        "voiceReleaseSeconds": VOICE_RELEASE_SECONDS,
        "cueCrossfadeSeconds": CUE_CROSSFADE_SECONDS,
        "finalFadeOutSeconds": final_fade,
        "sources": track_sources,
    }


def build_mix(ffmpeg: str, voice_path: Path, music_path: Path, total_frames: int,
              destination: Path) -> tuple[Path, dict[str, Any]]:
    duration = total_frames / FPS
    temp = new_temp_path(destination)
    graph = (
        f"[0:a]aformat=sample_rates={RATE}:channel_layouts=stereo[voice];"
        f"[1:a]aformat=sample_rates={RATE}:channel_layouts=stereo[music];"
        f"[voice][music]amix=inputs=2:duration=longest:normalize=0:dropout_transition=0,"
        f"apad=whole_dur={fmt(duration)},atrim=duration={fmt(duration)},"
        f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo,"
        "astats=metadata=0:reset=0[mix]"
    )
    command = ["-y", "-filter_complex_threads", "1", "-i", str(voice_path), "-i", str(music_path),
               "-filter_complex", graph, "-map", "[mix]", "-c:a", "aac", "-b:a", "320k",
               "-ar", str(RATE), "-ac", "2", "-movflags", "+faststart", str(temp)]
    try:
        log = run_ffmpeg(ffmpeg, command, "Building mix.m4a")
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return temp, {
        "path": str(destination),
        **parse_astats(log),
        "sampleRate": RATE,
        "channels": 2,
        "encoding": "AAC-LC 320 kbit/s",
        "masterGainDb": 0,
        "masterLimiter": "none; pre-encode signal is the direct stem sum",
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp = new_temp_path(path)
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", type=Path, default=EDIT_DEFAULT, help="edit.json (default: video/src/data/edit.json)")
    parser.add_argument("--narration-output", help="default: video/public/audio/narration.wav")
    parser.add_argument("--music-output", help="default: video/public/audio/music.wav")
    parser.add_argument("--mix-output", help="default: video/public/audio/mix.m4a")
    args = parser.parse_args()

    edit_path = args.edit if args.edit.is_absolute() else (VIDEO / args.edit)
    edit_path = edit_path.resolve()
    _, scenes, narration, total_frames, cues = read_manifest(edit_path)
    show_ranges = showcase_ranges(scenes, cues, total_frames)
    if not show_ranges:
        fail("Add a long showcase scene id ('showcase' or 'hero') with frame-based from/duration so speech can be excluded and the K→L change placed.")
    validate_narration_outside_showcase(narration, show_ranges)
    segments = make_music_segments(scenes, cues, total_frames, show_ranges)

    ffmpeg = find_binary("FFMPEG_PATH", "ffmpeg")
    ffprobe = find_binary("FFPROBE_PATH", "ffprobe")
    narration_output = output_path(args.narration_output, PUBLIC / "audio" / "narration.wav")
    music_output = output_path(args.music_output, PUBLIC / "audio" / "music.wav")
    mix_output = output_path(args.mix_output, PUBLIC / "audio" / "mix.m4a")
    if len({narration_output, music_output, mix_output}) != 3:
        fail("Narration, music, and mix outputs must be three different paths")

    temp_paths: list[Path] = []
    try:
        voice_temp, voice_meta = build_voice(ffmpeg, ffprobe, narration, total_frames, narration_output)
        temp_paths.append(voice_temp)
        music_temp, music_meta = build_music(ffmpeg, ffprobe, segments, narration, total_frames, music_output)
        temp_paths.append(music_temp)
        mix_temp, mix_meta = build_mix(ffmpeg, voice_temp, music_temp, total_frames, mix_output)
        temp_paths.append(mix_temp)
        outputs = [(voice_temp, narration_output), (music_temp, music_output), (mix_temp, mix_output)]
        for temporary, _ in outputs:
            info = probe_audio(ffprobe, temporary)
            if (int(info["sample_rate"]) != RATE or info["channels"] != 2
                    or abs(info["durationSeconds"] - total_frames / FPS) > 1 / FPS):
                fail(f"Audio output does not match the 48-kHz stereo timeline: {temporary}")
        for temporary, final in outputs:
            os.replace(temporary, final)
        temp_paths.clear()

        plan = {
            "manifest": str(edit_path),
            "fps": FPS,
            "totalFrames": total_frames,
            "durationSeconds": total_frames / FPS,
            "sampleRate": RATE,
            "channels": 2,
            "narration": voice_meta,
            "music": music_meta,
            "mix": mix_meta,
            "showcaseIntervals": [{"from": start, "to": end, "unit": "frames at 60 fps"}
                                  for start, end in show_ranges],
            "narrationOverlapsShowcase": False,
            "narrationSources": sorted({str(clip["manifestSrc"]) for clip in narration}),
            "narrationClips": [
                {"id": clip["id"], "scene": clip["scene"], "src": clip["manifestSrc"],
                 "sourceStartSeconds": clip["sourceStartSeconds"], "durationSeconds": clip["durationSeconds"],
                 "from": clip["from"], "originalTake": clip["originalTake"]}
                for clip in narration
            ],
            "musicSegments": [
                {"id": segment["cueIds"], "role": segment["role"],
                 "from": segment["startFrame"], "to": segment["endFrame"],
                 "startSeconds": segment["startFrame"] / FPS,
                 "endSeconds": segment["endFrame"] / FPS}
                for segment in segments
            ],
            "licenseNote": "CC BY 4.0 was checked against the locally cached Scott Buckley track pages in the neighboring Remotion project; include each used track's credit and the license URL in the YouTube description. Content ID claims remain possible.",
            "licenseUrl": LICENSE_URL,
        }
        write_json_atomic(PUBLIC / "audio" / "mix-plan.json", plan)
        credits = "\n\n".join(
            f"{track['credit']}\n{track['page']}\n{track['licenseUrl']}"
            for track in music_meta["sources"]
        )
        credits += "\n\nMusic excerpts were edited, looped, crossfaded, and mixed with narration for this video.\n"
        credit_path = VIDEO / "out/edit/music-credits.txt"
        credit_path.parent.mkdir(parents=True, exist_ok=True)
        credit_path.write_text(credits, encoding="utf-8")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    finally:
        for temporary in temp_paths:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        print(f"build_audio: {error}", file=sys.stderr)
        raise SystemExit(1)
