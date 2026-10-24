"""Presentation framing must retain moving and off-centre scene content."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "video/tools/build_framing.py"
spec = importlib.util.spec_from_file_location("build_framing_under_test", TOOL)
framing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(framing)


def test_top_hud_does_not_turn_an_empty_scene_into_foreground():
    width, height = 64, 36
    pixels = bytearray(width * height * 3)
    pixels[:width * 3] = bytes([255]) * (width * 3)
    assert framing.content_bounds(bytes(pixels), width, height) is None


def test_motion_union_keeps_early_and_late_positions_inside_crop():
    bounds = [(20, 80, 40, 100), (360, 160, 440, 240)]
    region = framing.framing_region(bounds)
    ratio = framing.SOURCE_WIDTH / framing.SAMPLE_WIDTH
    assert region is not None
    for left, top, right, bottom in bounds:
        assert region["x"] <= left * ratio
        assert region["y"] <= top * ratio
        assert region["x"] + region["width"] >= right * ratio
        assert region["y"] + region["height"] >= bottom * ratio
    assert 0 <= region["x"] < region["x"] + region["width"] <= framing.SOURCE_WIDTH
    assert 0 <= region["y"] < region["y"] + region["height"] <= framing.SOURCE_HEIGHT


def test_off_centre_result_is_framed_around_its_content():
    region = framing.framing_region([(40, 110, 160, 150)])
    ratio = framing.SOURCE_WIDTH / framing.SAMPLE_WIDTH
    assert region is not None
    assert region["x"] + region["width"] / 2 == pytest.approx(100 * ratio, abs=1)
    assert region["y"] + region["height"] / 2 == pytest.approx(130 * ratio, abs=1)
    assert region["width"] < framing.SOURCE_WIDTH
    assert region["height"] < framing.SOURCE_HEIGHT
