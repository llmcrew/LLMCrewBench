#!/usr/bin/env python3
"""Locate sparse K results for presentation without changing source clips.

Sample several times, ignore the page's small top HUD, and retain the union
of visible content. L artwork is not cropped by this tool.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
SOURCE_WIDTH, SOURCE_HEIGHT = 1920, 1080
SAMPLE_WIDTH, SAMPLE_HEIGHT = 480, 270
SAMPLE_TIMES = (0.5, 3, 7, 12, 17, 19.5)


def content_bounds(rgb: bytes, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Bound non-background pixels, excluding only the source's top HUD."""
    if len(rgb) != width * height * 3:
        raise ValueError("incomplete RGB frame")
    top = max(1, round(height * 32 / SOURCE_HEIGHT))
    corners = [(2, top + 2), (width - 3, top + 2), (2, height - 3), (width - 3, height - 3)]
    background = [statistics.median(rgb[(y * width + x) * 3 + channel] for x, y in corners) for channel in range(3)]
    left, right, bottom = width, -1, -1
    background_r, background_g, background_b = background
    first_row = height
    count = 0
    for y in range(top, height):
        for x in range(width):
            offset = (y * width + x) * 3
            if (abs(rgb[offset] - background_r) < 18
                    and abs(rgb[offset + 1] - background_g) < 18
                    and abs(rgb[offset + 2] - background_b) < 18):
                continue
            left, right = min(left, x), max(right, x)
            first_row, bottom = min(first_row, y), max(bottom, y)
            count += 1
    return (left, first_row, right + 1, bottom + 1) if count >= 8 else None


def framing_region(bounds: list[tuple[int, int, int, int]]) -> dict[str, int] | None:
    """Union all samples, pad the result, and cap presentation magnification."""
    if not bounds:
        return None
    ratio = SOURCE_WIDTH / SAMPLE_WIDTH
    left = min(item[0] for item in bounds) * ratio
    top = min(item[1] for item in bounds) * ratio
    right = max(item[2] for item in bounds) * ratio
    bottom = max(item[3] for item in bounds) * ratio
    width = min(SOURCE_WIDTH, max(640, (right - left) * 1.24))
    height = min(SOURCE_HEIGHT, max(360, (bottom - top) * 1.24))
    x = max(0, min(SOURCE_WIDTH - width, (left + right - width) / 2))
    y = max(0, min(SOURCE_HEIGHT - height, (top + bottom - height) / 2))
    # Outward rounding keeps both sides of a sampled object inside the region.
    x0, y0 = math.floor(x), math.floor(y)
    return {"x": x0, "y": y0, "width": math.ceil(x + width) - x0, "height": math.ceil(y + height) - y0}


def clip_framing(ffmpeg: str, clips: Path, row: dict) -> dict:
    name = f"{row['model']}__{row['scene']}.mp4"
    bounds = []
    for seconds in SAMPLE_TIMES:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-ss", str(seconds), "-i", str(clips / name),
             "-frames:v", "1", "-vf", f"scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT}",
             "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
            capture_output=True, check=True, timeout=60,
        )
        box = content_bounds(result.stdout, SAMPLE_WIDTH, SAMPLE_HEIGHT)
        if box is not None:
            bounds.append(box)
    region = framing_region(bounds)
    if region is None:
        raise ValueError(f"no visible content found: {name}")
    return {"clip": f"clips/{name}", **region}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=VIDEO / "public/clips")
    parser.add_argument("--manifest", type=Path, default=VIDEO / "public/clips.json")
    parser.add_argument("--output", type=Path, default=VIDEO / "src/data/framing.json")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("ffmpeg is required")
    rows = [row for row in json.loads(args.manifest.read_text(encoding="utf-8"))
            if row["scene"].startswith("K-") and row["status"] == "ok"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        regions = list(pool.map(lambda row: clip_framing(args.ffmpeg, args.clips, row), rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(regions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"framing: {len(regions)} K clips; source HTML and MP4 unchanged; {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
