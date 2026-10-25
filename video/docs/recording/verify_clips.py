#!/usr/bin/env python3
"""verify_clips: декодировать каждую запись и доказать, что клип - запись.

Проверяется не заголовок, а кадры: длительность, CFR-метки, число кадров,
непрерывность часов съёмки, SHA файла и «кадр вообще меняется». Чёрный клип -
плашка, поэтому чёрнота не ошибка статуса плашки и ошибка статуса ``ok``.

Библиотечно::

    from verify_clips import Spec, verify, quality_report
    row = verify(Path("clips/m__K-1.mp4"), Spec(), status="ok")
    report = quality_report([row], Spec())

``verify`` принимает ``runner`` - подмену запуска ffmpeg/ffprobe, поэтому
логику проверок можно прогнать на фейковых кадрах без бинарников.

CLI::

    python3 verify_clips.py CLIPS [--manifest clips.json] [--seconds 23]
        [--fps 120] [--output recording-quality.json]

Код возврата 0 - все клипы прошли, 1 - есть провал. Отчёт пишется атомарно:
недописанный JSON не выглядит провалом проверки.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manifest  # noqa: E402
import timing  # noqa: E402

#: Средняя яркость кадра (0..255) не выше порога - кадр чёрный.
BLACK_LUMA = 2
LUMA_W, LUMA_H = 96, 54
EVIDENCE_SUFFIX = ".capture.json"


class VehicleError(RuntimeError):
    """ffmpeg/ffprobe не отработали: клип не проверен, а не «сломан»."""


@dataclasses.dataclass(frozen=True)
class Spec:
    """Ожидаемая запись. Замораживается: половина проверок - сравнение с ней."""

    #: По умолчанию - целый ход сетки: окно 20 с и занавес с двух сторон.
    seconds: Fraction = Fraction(int(timing.SHOWCASE_TURN_SECONDS))
    fps: int = timing.MASTER_FPS
    width: int = timing.FRAME_WIDTH
    height: int = timing.FRAME_HEIGHT
    codec: str = "h264"
    pix_fmt: str = "yuv420p"
    crf: int = 16
    preset: str = "medium"

    @property
    def frames(self) -> int:
        return int(self.seconds * self.fps)

    @property
    def tolerance(self) -> Fraction:
        return Fraction(1, self.fps * 2)


def run(command: list[str], timeout: int = 300, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        command, capture_output=True, text=not binary, timeout=timeout, check=False,
    )
    problem = result.stderr if binary else (result.stderr or "").strip()
    if result.returncode or problem:
        if isinstance(problem, bytes):
            problem = problem.decode("utf-8", "replace").strip()
        raise VehicleError(problem or f"command exited {result.returncode}")
    return result.stdout


def fingerprint(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, hashlib.sha256).hexdigest()


def _meta(path: Path, ffprobe: str, runner) -> dict:
    keys = ",".join((
        "codec_name", "width", "height", "pix_fmt", "avg_frame_rate", "nb_frames",
        "duration", "bit_rate",
    ))
    return json.loads(runner([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        f"stream={keys}:format=duration,size,bit_rate", "-of", "json", str(path),
    ]))


def _frames(path: Path, ffmpeg: str, runner) -> tuple[list[Fraction], list[str]]:
    """Метки и хеши кадров: PTS должен идти ровно по 1/fps, без пропусков."""
    decoded = runner([
        ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-threads", "2", "-i", str(path),
        "-map", "0:v:0", "-an", "-vf", f"scale={LUMA_W}:{LUMA_H}:flags=area",
        "-fps_mode", "passthrough", "-threads", "1", "-f", "framemd5", "pipe:1",
    ])
    timebase = None
    stamps: list[Fraction] = []
    digests: list[str] = []
    for line in decoded.splitlines():
        if line.startswith("#tb 0:"):
            timebase = Fraction(line.split(":", 1)[1].strip())
            continue
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or timebase is None:
            raise ValueError(f"invalid decoded frame row: {line}")
        stamps.append(int(fields[2]) * timebase)
        digests.append(fields[5])
    return stamps, digests


def _luma(path: Path, ffmpeg: str, runner) -> list[float]:
    """Средняя яркость каждого кадра: 96×54 серого - достаточно для «чёрный»."""
    data = runner([
        ffmpeg, "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
        "-vf", f"scale={LUMA_W}:{LUMA_H}:flags=area,format=gray",
        "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
    ], binary=True)
    if isinstance(data, str):
        data = data.encode("latin-1")
    size = LUMA_W * LUMA_H
    return [sum(data[start:start + size]) / size
            for start in range(0, len(data) - size + 1, size)]


def _evidence(path: Path, spec: Spec, status: str, errors: list[str]) -> dict:
    """Паспорт съёмки рядом с клипом: без него запись недоказуема."""
    source = path.with_suffix(EVIDENCE_SUFFIX)
    try:
        evidence = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"capture evidence unreadable: {exc}")
        return {}
    expected = {
        "status": status, "seconds": float(spec.seconds), "fps": spec.fps,
        "frames": spec.frames, "crf": spec.crf, "preset": spec.preset,
        "captured_frames": spec.frames if status == "ok" else 0,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            errors.append(f"capture {key}: {evidence.get(key)} != {value}")
    if status == "ok":
        if evidence.get("captured_frames") != expected["captured_frames"]:
            errors.append("capture does not prove every frame was captured")
        if evidence.get("input_format") != "png":
            errors.append(f"capture input_format: {evidence.get('input_format')} != png")
        if evidence.get("clock_discontinuities") != 0:
            errors.append("capture clock is not continuous")
        span = (evidence.get("clock_end_ms") or 0) - (evidence.get("clock_start_ms") or 0)
        if abs(span - (spec.frames - 1) * 1000 / spec.fps) > 1e-5:
            errors.append(f"capture clock span: {span}")
    return evidence


def verify(
    path: Path, spec: Spec, *, status: str = "ok", ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe", runner=run, evidence: bool = False,
    sha256: str | None = None,
) -> dict:
    """Проверить один клип. Возвращает строку отчёта, а не бросает на провал."""
    path = Path(path)
    errors: list[str] = []
    row: dict = {"file": path.name, "ok": False, "errors": errors, "status": status}
    meta = _meta(path, ffprobe, runner)
    streams = meta.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected one video stream, got {len(streams)}")
    stream = streams[0]
    for key, expected in (("codec_name", spec.codec), ("width", spec.width),
                          ("height", spec.height), ("pix_fmt", spec.pix_fmt)):
        if stream.get(key) != expected:
            errors.append(f"{key}: {stream.get(key)} != {expected}")
    if Fraction(stream["avg_frame_rate"]) != spec.fps:
        errors.append(f"fps: {stream['avg_frame_rate']} != {spec.fps}")
    duration = Fraction(meta["format"]["duration"])
    if abs(duration - spec.seconds) > spec.tolerance:
        errors.append(f"duration: {float(duration)} != {float(spec.seconds)}")
    if int(stream.get("nb_frames", -1)) != spec.frames:
        errors.append(f"container frames: {stream.get('nb_frames')} != {spec.frames}")

    stamps, digests = _frames(path, ffmpeg, runner)
    for index, stamp in enumerate(stamps):
        if stamp != Fraction(index, spec.fps):
            errors.append(f"PTS frame {index}: {stamp} != {Fraction(index, spec.fps)}")
    if len(digests) != spec.frames:
        errors.append(f"decoded frames: {len(digests)} != {spec.frames}")

    luma = _luma(path, ffmpeg, runner)
    if len(luma) != len(digests):
        errors.append(f"luma frames: {len(luma)} != {len(digests)}")
    black = [value <= BLACK_LUMA for value in luma]
    black_frames = sum(black)
    all_black = bool(luma) and black_frames == len(luma)
    state = timing.media_state(status, has_clip=True, black=all_black)
    if state == "ok-black":
        errors.append("ok clip is entirely black (looks like a placeholder)")
    elif state == "placeholder-not-black":
        errors.append(f"placeholder {status} is not black (mean luma {max(luma or [0]):.1f})")

    unchanged = longest_unchanged = longest_black = run_black = 0
    previous = None
    for digest, is_black in zip(digests, black):
        unchanged = unchanged + 1 if digest == previous else 1
        longest_unchanged = max(longest_unchanged, unchanged)
        run_black = run_black + 1 if is_black else 0
        longest_black = max(longest_black, run_black)
        previous = digest

    digest = fingerprint(path)
    if sha256 is not None and digest != sha256:
        errors.append("sha256 differs from the manifest")
    row.update(
        seconds=float(spec.seconds), fps=spec.fps, decoded_frames=len(digests),
        bit_rate=int(stream.get("bit_rate") or meta["format"].get("bit_rate") or 0),
        bytes=path.stat().st_size, sha256=digest,
        distinct_thumbnails=len(set(digests)),
        longest_unchanged_seconds=round(longest_unchanged / spec.fps, 4),
        black_frames=black_frames, black_ratio=round(black_frames / max(1, len(luma)), 4),
        longest_black_seconds=round(longest_black / spec.fps, 4),
        sample_hashes=[
            digests[index] for index in sorted({0, len(digests) // 2, len(digests) - 1})
            if 0 <= index < len(digests)
        ],
    )
    if evidence:
        row["capture"] = _evidence(path, spec, status, errors)
    row["ok"] = not errors
    return row


def quality_report(rows: list[dict], spec: Spec) -> dict:
    """Сводка по строкам. Провал остаётся в списке, а не исчезает из счёта."""
    order = sorted(rows, key=lambda row: row.get("file", ""))
    return {
        "seconds": float(spec.seconds), "fps": spec.fps,
        "width": spec.width, "height": spec.height,
        "clips": len(order), "passed": sum(row.get("ok") is True for row in order),
        "failed": sum(row.get("ok") is not True for row in order),
        "black_luma_threshold": BLACK_LUMA,
        "notes": (
            "Unchanged imagery can be a valid static result; it is reported, never treated "
            "as a capture failure. Placeholder clips must be black, real clips must not be "
            "entirely black. Bitrate varies with scene complexity under CRF encoding."
        ),
        "results": order,
    }


def write_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def rows_for(clips: Path, rows: list[dict] | None = None) -> list[tuple[Path, dict]]:
    """Пары «клип, строка манифеста»; без манифеста статус берётся по маркеру."""
    if rows is None:
        return [
            (path, {"status": "media_missing" if path.with_suffix(".missing").is_file() else "ok"})
            for path in sorted(Path(clips).glob("*.mp4"))
        ]
    return [(Path(clips) / manifest.clip_name(row), row) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path)
    parser.add_argument("--manifest", type=Path, help="проверять только пары из clips.json")
    parser.add_argument("--notes", type=Path, help="result-notes.json для проверки статусов")
    parser.add_argument("--seconds", default=str(timing.SHOWCASE_TURN_SECONDS))
    parser.add_argument("--fps", type=int, default=timing.MASTER_FPS)
    parser.add_argument("--width", type=int, default=timing.FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=timing.FRAME_HEIGHT)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-evidence", action="store_true", help="не требовать паспорт съёмки")
    args = parser.parse_args(argv)

    try:
        spec = Spec(seconds=Fraction(args.seconds), fps=args.fps, width=args.width,
                    height=args.height, crf=args.crf, preset=args.preset)
        timing.frame_count(float(spec.seconds), spec.fps)
    except (ValueError, ZeroDivisionError) as exc:
        parser.error(f"invalid spec: {exc}")

    errors: list[str] = []
    rows: list[dict] | None = None
    notes: list[dict] | None = None
    if args.manifest:
        try:
            rows = manifest.load(args.manifest)
            if args.notes:
                notes = json.loads(args.notes.read_text(encoding="utf-8"))
            errors += manifest.check(rows, notes=notes)
        except (manifest.ManifestError, OSError, ValueError) as exc:
            parser.error(str(exc))
        if errors:
            parser.error("; ".join(errors[:5]))
    pairs = rows_for(args.clips, rows)
    if not pairs:
        parser.error("no MP4 clips found")
    if rows is not None:
        errors += [f"manifest: {error}" for error in manifest.check_files(rows, args.clips)]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                verify, path, spec, status=row.get("status", "ok"), ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe, evidence=not args.no_evidence and rows is not None,
                sha256=row.get("sha256"),
            ): (path, row)
            for path, row in pairs
        }
        for future in as_completed(futures):
            path, row = futures[future]
            try:
                result = future.result()
            except (OSError, ValueError, RuntimeError, KeyError, TypeError,
                    subprocess.SubprocessError) as exc:
                result = {
                    "file": path.name, "ok": False, "status": row.get("status"),
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            if row.get("model"):
                result.update(model=row["model"], scene=row["scene"])
            results.append(result)
            print(f"[verify] {len(results)}/{len(pairs)} "
                  f"{'ok' if result['ok'] else 'FAIL'} {path.name}", flush=True)

    report = quality_report(results, spec)
    report["manifest_errors"] = errors
    output = write_report(report, args.output or args.clips.parent / "recording-quality.json")
    print(f"[verify] {report['passed']}/{report['clips']} passed; {output}", flush=True)
    if errors:
        for error in errors:
            print(f"[verify] manifest: {error}", file=sys.stderr)
    return 0 if report["failed"] == 0 and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
