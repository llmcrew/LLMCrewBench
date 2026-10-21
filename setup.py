"""Сборка wheel без PEP 517-изоляции.

``pyproject.toml`` объявляет метаданные и ресурсы, но setuptools < 61 не читает
``[project]`` и ``[tool.setuptools.data-files]``. Этот файл повторяет минимум для
такой сборки; тест сверяет, что список ресурсов в ``pyproject.toml`` не разошёлся
с ним. Без ресурсов wheel ставится без three.js и plotly, и опубликованная
витрина не рисуется (R18).
"""

from setuptools import setup

setup(
    name="dailybench",
    version="0.2.0",
    packages=["dailybench", "dailybench.tasks"],
    data_files=[
        (
            "vendor/three",
            [
                "vendor/three/three.core.js",
                "vendor/three/three.module.js",
                "vendor/three/LICENSE",
                "vendor/three/PROVENANCE.md",
            ],
        ),
        (
            "vendor/plotly",
            ["vendor/plotly/plotly.min.js", "vendor/plotly/LICENSE", "vendor/plotly/PROVENANCE.md"],
        ),
    ],
)
