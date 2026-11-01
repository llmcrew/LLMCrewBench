"""Сборка wheel через legacy-интерфейс ``setup.py``.

``pyproject.toml`` остаётся единственным источником списков пакетов и ресурсов:
этот файл читает их напрямую, чтобы legacy- и PEP 517-сборки не расходились.
Ресурсы three.js, plotly и JSON-схемы необходимы установленному пакету.
"""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from setuptools._vendor import tomli as tomllib

from setuptools import setup

_CONFIG = tomllib.loads(
    (Path(__file__).resolve().parent / "pyproject.toml").read_text(encoding="utf-8")
)["tool"]["setuptools"]

setup(
    name="dailybench",
    version="0.2.0",
    packages=_CONFIG["packages"],
    data_files=[
        (directory, files)
        for directory, files in _CONFIG["data-files"].items()
    ],
)
