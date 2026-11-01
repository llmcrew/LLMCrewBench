"""README stretch counts stay equal to the suite the code actually builds."""

from __future__ import annotations

import re
from pathlib import Path

from dailybench.config import RunConfig

def _ci_cfg(**kw):
    """RunConfig with the CI long-context window cap (see conftest)."""
    import os

    cfg = RunConfig(**kw)
    window = int(os.environ.get("DAILYBENCH_LONGCTX_WINDOW") or 0)
    if window and window < cfg.context_window:
        cfg.context_window = window
    return cfg

from dailybench.suite import build_suite

ROOT = Path(__file__).resolve().parents[1]


def _suite(profile: str):
    return build_suite(_ci_cfg(profile=profile))


def _compact_new_ids(new_ids: list[str]) -> str:
    """Suite order, with a contiguous K or L run written as a range."""
    parts: list[str] = []
    index = 0
    while index < len(new_ids):
        match = re.fullmatch(r"([KL])-(\d+)", new_ids[index])
        if match:
            prefix, start = match.group(1), int(match.group(2))
            end = start
            cursor = index
            while cursor + 1 < len(new_ids):
                nxt = re.fullmatch(rf"{prefix}-(\d+)", new_ids[cursor + 1])
                if not nxt or int(nxt.group(1)) != end + 1:
                    break
                end = int(nxt.group(1))
                cursor += 1
            if cursor > index:
                parts.append(f"{prefix}-{start}…{prefix}-{end}")
                index = cursor + 1
                continue
        parts.append(new_ids[index])
        index += 1
    return ", ".join(parts)


def test_readme_stretch_counts_match_build_suite():
    stretch = _suite("stretch-v1")
    baseline_ids = {task.id for task in _suite("baseline-20261020")}
    k = sum(task.category == "K" for task in stretch)
    l = sum(task.category == "L" for task in stretch)
    new_ids = _compact_new_ids([task.id for task in stretch if task.id not in baseline_ids])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_ru = (ROOT / "README.ru.md").read_text(encoding="utf-8")
    assert f"({k} and {l} in the `stretch-v1` profile)" in readme
    assert f"в `stretch-v1` — {k} и {l}" in readme_ru
    assert "30 and 39" not in readme
    assert "30 и 39" not in readme_ru
    assert new_ids in readme
    assert new_ids in readme_ru
    assert f"{k} K and {l} L" in readme
    assert f"{k} K, {l} L" in readme_ru


def test_readmes_state_the_current_measurement_policy():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_ru = (ROOT / "README.ru.md").read_text(encoding="utf-8")
    for text in (readme, readme_ru):
        assert "--request-wall-clock" in text
        assert "resource_exhausted" in text
        assert "1800" in text
        assert "Windows" in text
        assert "resume_log.jsonl" in text
        assert "infra_failure" in text and "evaluator_failure" in text
    assert "not ranked against a baseline" in readme
    assert "6 informational points" in readme and "outside the 71" in readme
    assert "не ранжируется с baseline" in readme_ru
    assert "6 справочных баллов" in readme_ru
    assert "без баллов" not in readme_ru
    assert "forwarder" in readme
    assert "форвардер" in readme_ru or "forwarder" in readme_ru
