"""The ignore rules must not swallow the artefacts a published run depends on.

Two mistakes are easy to make and invisible until someone clones the repo: a
wide pattern (``*.json``) silently drops the reference policy, and a missing
pattern commits megabytes of run output. Both directions are pinned here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git недоступен")


def _ignored(relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "relative",
    [
        "configs/reference.json",
        "suite/20260915/prompts.json",
        "suite/20260915/oracles.json",
        "suite/20260915/suite.json",
        "vendor/three/three.module.js",
        "vendor/three/PROVENANCE.md",
        "docs/art-contract.md",
        "pyproject.toml",
        "Makefile",
        "README.md",
        "dailybench/tasks/art_sunset.py",
        "tests/test_art.py",
        # Опубликованный отчёт лежит в git: он открывается двойным кликом,
        # без прогона и без сервера. runs/ при этом остаётся локальным.
        "reports/comparison/index.html",
        "reports/comparison/_vendor/three.global.js",
    ],
)
def test_published_artefacts_are_tracked(relative: str):
    assert not _ignored(relative), f"{relative} обязан попасть в git"


@pytest.mark.parametrize(
    "relative",
    [
        "runs/plan/scores.json",
        "runs/_servers/studio.log",
        ".venv/bin/python",
        "dailybench/__pycache__/core.cpython-312.pyc",
        "tests/__pycache__/test_harness.cpython-312-pytest-9.1.1.pyc",
        ".pytest_cache/v/cache/lastfailed",
        ".coverage",
        "htmlcov/index.html",
        "dailybench.egg-info/PKG-INFO",
        ".env",
        "secrets.env",
        ".DS_Store",
    ],
)
def test_generated_files_are_ignored(relative: str):
    assert _ignored(relative), f"{relative} — сгенерированное, не должно коммититься"
