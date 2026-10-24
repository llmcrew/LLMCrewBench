"""Readiness of a scene page must not hinge on one hook.

A page that draws on its own (rAF loop, no ``__ready``/``__dbscene``) was treated
as dead and got a "модель не справилась" plaque, even though it rendered fine —
that is how the 1017 stand lost L-1. The probe therefore accepts any of three
signs, and only an empty page counts as dead.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "video" / "tools" / "record_clip.py"

pytestmark = pytest.mark.skipif(not TOOL.is_file(), reason="рекордер вне репозитория")


def _module():
    spec = importlib.util.spec_from_file_location("record_clip_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeWS:
    """Подменный клиент: готовности из него нужен только список событий."""

    def __init__(self):
        self.events: list[dict] = []


class FakeCall:
    """CDP ``Runtime.evaluate`` подменик: отдаёт заранее заданный вердикт."""

    def __init__(self, verdict: str):
        self.verdict = verdict

    def __call__(self, method, params=None, **kwargs):
        return {"result": {"result": {"value": self.verdict}}}




def test_empty_page_is_not_ready():
    mod = _module()
    assert mod.page_ready(FakeCall(""), FakeWS(), "L-1", timeout=0.1) == ""




def test_a_broken_evaluate_does_not_crash_the_recorder():
    mod = _module()

    def broken(method, params=None, **kwargs):
        return {"error": {"message": "Execution context was destroyed"}}

    assert mod.page_ready(broken, FakeWS(), "L-1", timeout=0.1) == ""


def test_placeholder_decodes_to_black_not_green(tmp_path):
    exe = shutil.which("ffmpeg")
    if exe is None:
        pytest.skip("ffmpeg не установлен")
    mod = _module()
    out = tmp_path / "placeholder.mp4"
    mod.placeholder(out, 3 / mod.FPS)
    decoded = subprocess.run(
        [exe, "-v", "error", "-i", str(out), "-vf", "scale=1:1",
         "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True,
    )
    assert len(decoded.stdout) == 9
    assert max(decoded.stdout) <= 2
