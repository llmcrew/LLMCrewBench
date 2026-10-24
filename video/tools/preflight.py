#!/usr/bin/env python3
"""Preflight медиа витрины: проверить манифест, маркеры и сами клипы до render.

Проверки:
  * showcase.json полон по scenes.json, статусы допустимы, clips.json - его
    производная;
  * статус согласован с маркером и файлом (ok без маркера, плашка с маркером,
    media_missing без клипа);
  * not_run/model_failed согласованы со scores прогона;
  * каждый клип декодируется ffprobe: h264, 1920×1080, 120 fps, 20 с,
    ожидаемое число кадров, читаются первый и последний кадр.

Ошибка печатается до Studio/render и даёт ненулевой код: битое медиа не должно
выясняться кадром 300 в Studio или через час рендера.

  python3 tools/preflight.py [--no-decode] [--no-clips]

``--no-clips`` проверяет манифест без самих клипов: в чистом клоне
``public/clips/`` нет (каталог вне git), а согласованность showcase.json,
scenes.json и производного clips.json проверяема и там.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import media_status  # noqa: E402
from record_all import MODELS, OUT, RUNS  # noqa: E402
from record_clip import FPS, H, W  # noqa: E402

VIDEO = OUT.parents[1]
SHOWCASE = VIDEO / "src" / "data" / "showcase.json"
CLIPS_JSON = OUT.parent / "clips.json"
SCENES_JSON = VIDEO / "src" / "data" / "scenes.json"
SECONDS = 20.0
DURATION_TOL = 0.25
FRAMES = int(round(SECONDS * FPS))


def find_ffprobe() -> Path | None:
    """Рабочий ffprobe: комплектный (если запускается), иначе системный из PATH."""
    candidates = [VIDEO / "bin" / "ffprobe"]
    found = shutil.which("ffprobe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if not (candidate.is_file() and os.access(candidate, os.X_OK)):
            continue
        try:
            probe = subprocess.run([str(candidate), "-version"], cwd=str(candidate.parent),
                                   capture_output=True, text=True)
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    return None


def probe_meta(exe: Path, path: Path) -> dict:
    out = subprocess.run(
        [str(exe), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        cwd=str(exe.parent), capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise ValueError((out.stderr or "ffprobe не прочитал файл").strip().splitlines()[-1])
    return json.loads(out.stdout or "{}")


def decode_frame(exe: Path, path: Path, last: bool = False) -> str | None:
    """Декодировать первый/последний кадр: ffprobe, а не только заголовок."""
    interval = f"{SECONDS - 0.2:.1f}%+#1" if last else "0%+#1"
    out = subprocess.run(
        [str(exe), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pkt_pts_time", "-read_intervals", interval,
         "-of", "json", str(path)],
        cwd=str(exe.parent), capture_output=True, text=True,
    )
    if out.returncode != 0:
        return (out.stderr or "кадр не декодируется").strip().splitlines()[-1]
    try:
        frames = json.loads(out.stdout or "{}").get("frames") or []
    except ValueError:
        return "ffprobe вернул не JSON"
    return None if frames else "кадр не декодирован"


def check_clip(exe: Path, model: str, scene: str, status: str, errors: list[str]) -> None:
    path = media_status.clip_path(OUT, model, scene)
    label = f"{model}/{scene}"
    if not path.is_file():
        if status != "media_missing":
            errors.append(f"{label}: нет файла клипа (статус {status})")
        return
    if path.stat().st_size == 0:
        errors.append(f"{label}: пустой файл клипа")
        return
    if status == "media_missing":
        errors.append(f"{label}: статус media_missing, но клип есть")
    try:
        meta = probe_meta(exe, path)
    except ValueError as exc:
        errors.append(f"{label}: ffprobe: {exc}")
        return
    streams = meta.get("streams") or []
    if not streams:
        errors.append(f"{label}: в контейнере нет видеопотока")
        return
    st = streams[0]
    if st.get("codec_name") != "h264":
        errors.append(f"{label}: кодек {st.get('codec_name')}, ожидался h264")
    if (st.get("width"), st.get("height")) != (W, H):
        errors.append(f"{label}: размер {st.get('width')}x{st.get('height')}, ожидался {W}x{H}")
    fps = st.get("avg_frame_rate") or st.get("r_frame_rate")
    if fps != f"{FPS}/1":
        errors.append(f"{label}: частота {fps}, ожидалась {FPS}/1")
    try:
        duration = float(st.get("duration") or (meta.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    if abs(duration - SECONDS) > DURATION_TOL:
        errors.append(f"{label}: длительность {duration:.3f} с, ожидалось {SECONDS}")
    frames = st.get("nb_frames")
    try:
        frame_count = int(frames) if frames is not None else -1
    except ValueError:
        frame_count = -1
    if abs(frame_count - FRAMES) > 2:
        errors.append(f"{label}: кадров {frames}, ожидалось {FRAMES}")
    first = decode_frame(exe, path)
    if first:
        errors.append(f"{label}: первый кадр: {first}")
    last = decode_frame(exe, path, last=True)
    if last:
        errors.append(f"{label}: последний кадр: {last}")


def validate(rows: list, scenes: list[str], errors: list[str], warnings: list[str],
             *, clips: bool = True) -> None:
    scene_ids = [s["id"] for s in scenes]
    if [r.get("scene") for r in rows] != scene_ids:
        errors.append("showcase.json: набор/порядок сцен не совпадает с scenes.json")
    # Свежее дерево до первой записи: клипов нет вообще, причины плашек придут
    # из прогона и превратятся в маркеры при записи. Это предупреждения, а не
    # ошибки; на записанном дереве те же расхождения - ошибки.
    pre_record = not any(
        media_status.clip_path(OUT, run.get("model", "?"), row.get("scene", "?")) .is_file()
        for row in rows for run in row.get("runs") or []
    )
    derived = []
    for row in rows:
        scene = row.get("scene", "?")
        runs = row.get("runs") or []
        if [r.get("model") for r in runs] != MODELS:
            errors.append(f"{scene}: полосы не совпадают с моделями {MODELS}")
            continue
        if row.get("open") not in range(len(runs)):
            errors.append(f"{scene}: open={row.get('open')} вне полос")
        for run in runs:
            model = run.get("model", "?")
            status = run.get("status")
            derived.append({"model": model, "scene": scene, "status": status})
            if status not in media_status.STATUSES:
                errors.append(f"{model}/{scene}: неизвестный статус {status!r}")
                continue
            marker = media_status.marker_path(OUT, model, scene)
            clip = media_status.clip_path(OUT, model, scene)
            if run.get("clip") != f"clips/{model}__{scene}.mp4":
                errors.append(f"{model}/{scene}: путь клипа {run.get('clip')!r}")
            if clips:
                if status == "media_missing":
                    if clip.is_file():
                        errors.append(f"{model}/{scene}: статус media_missing, но клип есть")
                elif status in media_status.STATUSES and not clip.is_file():
                    (warnings if pre_record else errors).append(
                        f"{model}/{scene}: нет файла клипа (статус {status})")
                if status == "ok" and marker.is_file():
                    errors.append(f"{model}/{scene}: статус ok, но маркер плашки ещё лежит")
                if status in media_status.PLACEHOLDER_STATUSES:
                    if not marker.is_file():
                        (warnings if pre_record and status != "capture_failed" else errors).append(
                            f"{model}/{scene}: нет маркера для статуса {status}")
                    else:
                        reason = marker.read_text(encoding="utf-8").strip()
                        # Улика после двоеточия статус не меняет.
                        if reason.split(":", 1)[0].strip() != status:
                            if reason == media_status.LEGACY_MARKER:
                                warnings.append(f"{model}/{scene}: legacy-маркер без причины, статус из данных")
                            else:
                                errors.append(f"{model}/{scene}: маркер {reason!r} не равен статусу {status!r}")
            expected = media_status.task_status(model, scene, RUNS)
            if expected is not None:
                if status == "not_run" and expected != "not_run":
                    errors.append(f"{model}/{scene}: not_run, но задача есть в прогоне ({expected})")
                if status != "not_run" and expected == "not_run":
                    errors.append(f"{model}/{scene}: статус {status}, но задачи нет в прогоне")
    try:
        actual = json.loads(CLIPS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"clips.json не читается: {exc}")
    else:
        if actual != derived:
            errors.append("clips.json расходится с showcase.json (производный файл устарел)")


def main() -> int:
    clips = "--no-clips" not in sys.argv[1:]
    decode = "--no-decode" not in sys.argv[1:] and clips
    errors: list[str] = []
    warnings: list[str] = []
    try:
        scenes = json.loads(SCENES_JSON.read_text(encoding="utf-8"))
        rows = json.loads(SHOWCASE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"preflight: не читаются данные: {exc}", file=sys.stderr)
        return 2
    validate(rows, scenes, errors, warnings, clips=clips)
    # Осиротевшие файлы: клип без строки манифеста или маркер без пары.
    manifest_pairs = {(r.get("model"), row.get("scene")) for row in rows for r in row.get("runs") or []}
    if not clips:
        files = []
    elif not OUT.is_dir():
        errors.append(f"нет каталога клипов {OUT}")
        files = []
    else:
        files = sorted(OUT.iterdir())
    for path in files:
        if path.name.startswith("_"):
            continue
        suffix = path.suffix
        if suffix not in (".mp4", ".missing"):
            continue
        stem = path.name[: -len(suffix)]
        model, _, scene = stem.partition("__")
        if scene not in {s["id"] for s in scenes} or model not in MODELS or (model, scene) not in manifest_pairs:
            warnings.append(f"лишний файл вне манифеста: {path.name}")
    if errors:
        for line in errors:
            print(f"preflight: ОШИБКА: {line}", file=sys.stderr)
    for line in warnings:
        print(f"preflight: предупреждение: {line}", file=sys.stderr)
    if errors:
        print(f"preflight: {len(errors)} ошибок - исправьте до Studio/render", file=sys.stderr)
        return 1
    if decode:
        exe = find_ffprobe()
        if exe is None:
            print("preflight: ffprobe не найден (PATH или video/bin); "
                  "поставьте ffmpeg или запустите с --no-decode", file=sys.stderr)
            return 1
        for row in rows:
            for run in row["runs"]:
                check_clip(exe, run["model"], row["scene"], run["status"], errors)
        if errors:
            for line in errors:
                print(f"preflight: ОШИБКА: {line}", file=sys.stderr)
            print(f"preflight: {len(errors)} ошибок декодирования", file=sys.stderr)
            return 1
    mode = "декодирование проверено" if decode else (
        "клипы не проверялись" if not clips else "без декодирования")
    print(f"preflight: ok - {len(rows)} сцен, {len(rows) * len(MODELS)} клипов, {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
