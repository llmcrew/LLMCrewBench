"""The rendered voice stem retains delayed fragments and sample-accurate gaps."""
from __future__ import annotations

import array
import importlib.util
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "video/tools/build_audio.py"
spec = importlib.util.spec_from_file_location("build_audio_under_test", TOOL)
audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="needs FFmpeg")
def test_delayed_voice_fragments_survive_with_exact_silent_gaps(tmp_path: Path):
    rate = 48_000
    source = tmp_path / "take.wav"
    tone = array.array("h", (round(4_000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(rate)))
    if sys.byteorder != "little":
        tone.byteswap()
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(tone.tobytes())

    clips = [
        {"index": index, "src": source, "from": frame, "sourceStartSeconds": 0.1, "durationSeconds": 0.6}
        for index, frame in enumerate((15, 120))
    ]
    rendered, _ = audio.build_voice("ffmpeg", "ffprobe", clips, 8 * 60, tmp_path / "narration.wav")
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(rendered), "-ac", "1", "-ar", str(rate),
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    samples = array.array("f", decoded.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    assert len(samples) == 8 * rate
    assert all(math.isfinite(value) for value in samples)
    assert max(abs(value) for value in samples) < 0.9

    def rms(start: float, end: float) -> float:
        section = samples[round(start * rate):round(end * rate)]
        return math.sqrt(sum(value * value for value in section) / len(section))

    assert rms(0.3, 0.8) > 0.01
    assert rms(2.05, 2.5) > 0.01
    assert rms(0, 0.15) < 1e-6
    assert rms(1.1, 1.8) < 1e-6
    assert rms(3, 7.5) < 1e-6
