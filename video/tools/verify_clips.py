#!/usr/bin/env python3
"""Decode every clip and verify duration, CFR timestamps and recording evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path


def run(command: list[str], timeout: int = 300) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(result.stderr.strip() or f"command exited {result.returncode}")
    return result.stdout


def verify(path: Path, ffmpeg: str, ffprobe: str, seconds: Fraction, fps: int) -> dict:
    metadata = json.loads(run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,duration,bit_rate:format=duration,size,bit_rate",
        "-of", "json", str(path),
    ]))
    streams = metadata.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream, got {len(streams)}")
    stream = streams[0]
    expected_frames = int(seconds * fps)
    errors = []
    for key, expected in (("codec_name", "h264"), ("width", 1920), ("height", 1080), ("pix_fmt", "yuv420p")):
        if stream.get(key) != expected:
            errors.append(f"{key}: {stream.get(key)} != {expected}")
    if Fraction(stream["avg_frame_rate"]) != fps:
        errors.append(f"fps: {stream['avg_frame_rate']} != {fps}")
    if abs(Fraction(metadata["format"]["duration"]) - seconds) > Fraction(1, fps * 2):
        errors.append(f"duration: {metadata['format']['duration']} != {seconds}")
    if int(stream.get("nb_frames", -1)) != expected_frames:
        errors.append(f"container frames: {stream.get('nb_frames')} != {expected_frames}")

    decoded = run([
        ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-threads", "2", "-i", str(path),
        "-map", "0:v:0", "-an", "-vf", "scale=96:54:flags=area", "-fps_mode", "passthrough",
        "-threads", "1", "-f", "framemd5", "pipe:1",
    ])
    timebase = None
    frame_hashes = []
    max_unchanged = unchanged = 0
    previous_hash = None
    for line in decoded.splitlines():
        if line.startswith("#tb 0:"):
            timebase = Fraction(line.split(":", 1)[1].strip())
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or timebase is None:
            raise ValueError(f"invalid decoded frame row: {line}")
        timestamp = int(fields[2]) * timebase
        expected_timestamp = Fraction(len(frame_hashes), fps)
        if timestamp != expected_timestamp:
            errors.append(f"PTS frame {len(frame_hashes)}: {timestamp} != {expected_timestamp}")
        digest = fields[5]
        unchanged = unchanged + 1 if digest == previous_hash else 1
        max_unchanged = max(max_unchanged, unchanged)
        previous_hash = digest
        frame_hashes.append(digest)
    if len(frame_hashes) != expected_frames:
        errors.append(f"decoded frames: {len(frame_hashes)} != {expected_frames}")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "file": path.name, "ok": not errors, "errors": errors,
        "seconds": float(seconds), "fps": fps, "decoded_frames": len(frame_hashes),
        "bit_rate": int(stream.get("bit_rate") or metadata["format"].get("bit_rate") or 0),
        "bytes": path.stat().st_size, "sha256": digest,
        "distinct_thumbnails": len(set(frame_hashes)),
        "longest_unchanged_seconds": round(max_unchanged / fps, 4),
        "sample_hashes": [frame_hashes[index] for index in sorted({0, len(frame_hashes) // 2, len(frame_hashes) - 1}) if 0 <= index < len(frame_hashes)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path)
    parser.add_argument("--seconds", default="20")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, help="Check only model/scene pairs in clips.json")
    args = parser.parse_args()
    manifest_by_file = {}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        paths = [args.clips / f"{row['model']}__{row['scene']}.mp4" for row in manifest]
        if len(set(paths)) != len(paths):
            parser.error("manifest contains duplicate model/scene pairs")
        manifest_by_file = {path.name: item for path, item in zip(paths, manifest)}
    else:
        paths = sorted(args.clips.glob("*.mp4"))
    if not paths:
        parser.error("no MP4 clips found")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(verify, path, args.ffmpeg, args.ffprobe, Fraction(args.seconds), args.fps): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            row = {"file": path.name, "ok": False, "errors": []}
            try:
                row = future.result()
                if path.name in manifest_by_file:
                    item = manifest_by_file[path.name]
                    evidence = json.loads(path.with_suffix(".capture.json").read_text(encoding="utf-8"))
                    expected_frames = round(Fraction(args.seconds) * args.fps)
                    expected = {
                        "status": item["status"], "seconds": float(Fraction(args.seconds)),
                        "fps": args.fps, "frames": expected_frames, "crf": 16, "preset": "medium",
                        "captured_frames": expected_frames if item["status"] == "ok" else 0,
                    }
                    for key, value in expected.items():
                        if evidence.get(key) != value:
                            row["errors"].append(f"capture {key}: {evidence.get(key)} != {value}")
                    if item["status"] == "ok":
                        if evidence.get("input_format") != "png" or evidence.get("clock_discontinuities") != 0:
                            row["errors"].append("capture does not prove lossless input and continuous frame clock")
                        clock_span = evidence["clock_end_ms"] - evidence["clock_start_ms"]
                        if abs(clock_span - (expected_frames - 1) * 1000 / args.fps) > 1e-5:
                            row["errors"].append(f"capture clock span: {clock_span}")
                    row.update(model=item["model"], scene=item["scene"], media_status=item["status"], capture=evidence)
                    row["ok"] = not row["errors"]
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
                row["ok"] = False
                row["errors"].append(f"{type(error).__name__}: {error}")
            rows.append(row)
            print(f"[verify] {len(rows)}/{len(paths)} {'ok' if row['ok'] else 'FAIL'} {path.name}", flush=True)
    rows.sort(key=lambda row: row["file"])
    report = {
        "seconds": float(Fraction(args.seconds)), "fps": args.fps,
        "clips": len(rows), "passed": sum(row["ok"] for row in rows),
        "notes": "Unchanged imagery can be a valid static result; it is reported, never silently treated as a capture failure. Bitrate varies with scene complexity under CRF encoding.",
        "results": rows,
    }
    output = args.output or args.clips.parent / "recording-quality.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[verify] {report['passed']}/{report['clips']} passed; {output}", flush=True)
    return 0 if report["passed"] == report["clips"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
