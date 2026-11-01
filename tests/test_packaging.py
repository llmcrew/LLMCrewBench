"""Контракты необязательных зависимостей и legacy-сборки."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _legacy_setup() -> dict:
    program = r'''
import json
import runpy
import sys
import types

captured = {}
setuptools = types.ModuleType("setuptools")
setuptools.setup = lambda **kwargs: captured.update(kwargs)
sys.modules["setuptools"] = setuptools
runpy.run_path("setup.py", run_name="__main__")
print(json.dumps(captured))
'''
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_list_works_without_pillow():
    program = "import sys; sys.modules['PIL']=None; from dailybench.cli import main; raise SystemExit(main(['list']))"
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_v_tasks_without_pillow_explain_vision_extra():
    expected = "для профиля research-v1 нужен Pillow: uv pip install -e .[vision]"
    program = f'''
import random
import sys
sys.modules["PIL"] = None
from dailybench.config import RunConfig
from dailybench.tasks import visual_input
try:
    visual_input.generate(random.Random(1), RunConfig(profile="research-v1"))
except RuntimeError as exc:
    raise SystemExit(0 if str(exc) == {expected!r} else 3)
raise SystemExit(2)
'''
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_visual_extra_carries_frame_stats_dependencies():
    visual = _pyproject()["project"]["optional-dependencies"]["visual"]
    assert any(requirement.startswith("numpy") for requirement in visual)
    assert any(requirement.startswith("Pillow") for requirement in visual)


def test_legacy_setup_packages_match_pyproject():
    config = _pyproject()["tool"]["setuptools"]
    legacy = _legacy_setup()
    assert legacy["packages"] == config["packages"]
    assert "dailybench.adapters" in legacy["packages"]
    assert dict(legacy["data_files"]) == config["data-files"]
    assert "schema/task.schema.json" in dict(legacy["data_files"])["schema"]


def test_no_package_data_points_to_a_missing_file():
    package_data = _pyproject()["tool"]["setuptools"].get("package-data", {})
    for package, patterns in package_data.items():
        directory = ROOT.joinpath(*package.split("."))
        for pattern in patterns:
            matches = list(directory.glob(pattern))
            assert matches, f"package-data {package}: {pattern} не существует"
            assert all(path.is_file() for path in matches)
