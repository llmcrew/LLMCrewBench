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
        "suite/20261020/prompts.json",
        "suite/20261020/oracles.json",
        "suite/20261020/suite.json",
        "vendor/three/three.module.js",
        "vendor/three/PROVENANCE.md",
        "docs/art-contract.md",
        "pyproject.toml",
        "Makefile",
        "README.md",
"dailybench/tasks/art.py",
        "tests/test_art.py",
        # Опубликованный отчёт: в git живёт только выборка первой публикации —
        # читатель клона открывает её двойным кликом, без прогона и сервера.
        # Пересборки и новые файлы — игнорируются (пины ниже); runs/ не в git.
        "reports/comparison/index.html",
        "reports/comparison/_vendor/three.global.js",
        # Опубликованные артефакты: прогон на стенде 1017 и сравнение стендов.
        # Видео (клипы, архив) — вне git: только описание и хеш архива.
        "runs/deepseek-v4.1-flash-unc-1017/report.md",
        "runs/deepseek-v4.1-flash-unc-1017/scores.json",
        "runs/deepseek-v4.1-flash-unc-1017/sizing.json",
        "runs/deepseek-v4.1-flash-unc-1017/_showcase/deepseek-v4.1-flash-unc-1017/frames/K-1.png",
        "reports/comparison-deepseek-1017/index.html",
        "reports/comparison-deepseek-1017/_vendor/plotly.min.js",
        "reports/comparison-deepseek-1017/_vendor/three.global.js",
        "reports/comparison-deepseek-1017/_showcase/deepseek-v4.1-flash-unc-1017/pages/K-1.html",
        "video/archive/RESTORE.md",
        "video/archive/SHA256",
        "docs/ARTIFACTS.md",
        "docs/recording/README.md",
        "docs/recording/record_scene.py",
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
        # Объёмные локальные артефакты ролика и отчёта: не пушатся, см. .gitignore.
        "video/public/clips/deepseek-v4.1-flash-unc__K-1.mp4",
        "video/bin/ffmpeg",
        "video/archive/dailybench-video.zip.part-aa",
        "video/dailybench-video.zip",
        "video/dailybench-video.tar.zst",
        "video/dist/dailybench-video.zip",
        "reports/comparison/_showcase/glm-5.3/pages/K-1.html",
        # Не публикуются: сырые транскрипты, машинные параметры, служебные каталоги.
        "runs/deepseek-v4.1-flash-unc-1017/tasks.json",
        "runs/deepseek-v4.1-flash-unc-1017/transcripts/L-1.json",
        "runs/_logs/deepseek-v4.1-flash-unc-1017.log",
        "runs/smoke-t1/config.json",
    ],
)
def test_generated_files_are_ignored(relative: str):
    assert _ignored(relative), f"{relative} — сгенерированное, не должно коммититься"
