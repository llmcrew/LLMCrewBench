"""Readiness of a scene page must not hinge on one hook.

A page that draws on its own (rAF loop, no ``__ready``/``__dbscene``) was treated
as dead and got a "модель не справилась" plaque, even though it rendered fine —
that is how the 1017 stand lost L-1. The probe therefore accepts any of three
signs, and only an empty page counts as dead.
"""

from __future__ import annotations

import importlib.util
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


class FakeCall:
    """CDP ``Runtime.evaluate`` подменик: отдаёт заранее заданный вердикт."""

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.expressions: list[str] = []

    def __call__(self, method, params=None, **kwargs):
        self.expressions.append((params or {}).get("expression", ""))
        return {"result": {"result": {"value": self.verdict}}}


@pytest.mark.parametrize("verdict", ["ready", "driver", "canvas"])
def test_any_render_sign_counts_as_ready(verdict: str):
    mod = _module()
    call = FakeCall(verdict)
    assert mod.page_ready(call, timeout=1) is True


def test_empty_page_is_not_ready():
    mod = _module()
    assert mod.page_ready(FakeCall(""), timeout=0.2) is False


def test_probe_checks_the_page_not_only_ready_flag():
    """Регресс: ``__ready`` одного мало — иначе L-1 снова станет плашкой."""
    mod = _module()
    call = FakeCall("")
    mod.page_ready(call, timeout=0.05)
    probe = call.expressions[0]
    assert "__ready" in probe
    assert "__dbscene" in probe
    assert "canvas" in probe


def test_a_broken_evaluate_does_not_crash_the_recorder():
    mod = _module()

    def broken(method, params=None, **kwargs):
        return {"error": {"message": "Execution context was destroyed"}}

    assert mod.page_ready(broken, timeout=0.1) is False
