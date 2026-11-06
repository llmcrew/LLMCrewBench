"""Контракт CI: workflow разбирается, гейты на месте и не привязаны к машине.

Тест держит форму `.github/workflows/ci.yml`, а не её детали оформления:
обязательные шаги перечислены по смыслу, matrix покрывает обе платформы,
секретов и абсолютных путей нет.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"

#: Команда/фрагмент, который обязан остаться в workflow.
REQUIRED_STEPS = {
    "compileall": "python -m compileall",
    "pytest по сьюту": "python -m pytest tests",
    "docker smoke": "tests/test_sandbox.py",
    "схемы — валидный JSON": "json.tool schema/task.schema.json",
    "валидация фикстуры сьюта": "validate --suite tests/fixtures/validate/suite.json",
    "негативные фикстуры": "tests/fixtures/validate/bad-",
    "валидный экспорт": "dailybench export",
    "сборка wheel": "pip wheel .",
    "чистый venv": "python -m venv",
    "установка wheel": ".whl",
    "smoke из установки": "bin/dailybench",
    "замкнутость ссылок отчёта": "broken_local_links",
}


def _load() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(document, dict), "workflow — не YAML-объект"
    return document


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _steps() -> list[dict]:
    return [step for job in _load()["jobs"].values() for step in job.get("steps", [])]


def test_workflow_is_valid_yaml_with_jobs():
    document = _load()
    assert document["jobs"], "нет ни одного job"
    triggers = document.get("on", document.get(True))
    assert triggers, "нет триггеров (PyYAML читает `on` как True)"
    assert {"push", "pull_request"} <= set(triggers), triggers


@pytest.mark.parametrize("gate", sorted(REQUIRED_STEPS))
def test_required_gate_is_present(gate: str):
    needle = REQUIRED_STEPS[gate]
    commands = "\n".join(str(step.get("run", "")) for step in _steps())
    names = "\n".join(str(step.get("name", "")) for step in _steps())
    uses = "\n".join(str(step.get("uses", "")) for step in _steps())
    assert needle in "\n".join((commands, names, uses)), f"нет шага: {gate} ({needle})"


def test_matrix_covers_linux_and_macos():
    matrices = [job["strategy"]["matrix"] for job in _load()["jobs"].values()
                if "matrix" in job.get("strategy", {})]
    assert matrices, "pytest без matrix"
    systems = [os for matrix in matrices for os in matrix.get("os", [])]
    assert any(os.startswith("ubuntu") for os in systems), systems
    assert any(os.startswith("macos") for os in systems), systems


def test_ci_actions_are_pinned_by_sha():
    lines = [line.strip() for line in _text().splitlines() if "uses:" in line]
    assert lines
    for line in lines:
        assert re.search(r"uses:\s*[\w./-]+@[0-9a-f]{40}\s+#\s*v\d", line), line


def test_ci_tests_windows_and_python_310():
    job = _load()["jobs"]["test"]
    matrix = job["strategy"]["matrix"]
    assert set(matrix["os"]) == {"ubuntu-latest", "macos-latest", "windows-latest"}
    assert set(matrix["python"]) == {"3.10", "3.12"}
    windows = [step for step in job["steps"]
               if "Remove-Item Env:PYTHONUTF8" in step.get("run", "")]
    assert windows, "Windows pytest наследует PYTHONUTF8"
    assert all("Remove-Item Env:PYTHONIOENCODING" in step["run"] for step in windows)


def test_ci_installs_locked_dev_dependencies():
    installs = [step for step in _steps() if step.get("name") == "Install"]
    assert installs
    # The pytest matrix additionally installs visual+vision (research-v1
    # fixtures); every install must stay locked and carry dev.
    assert all(
        step.get("run", "").startswith("uv sync --locked --extra dev")
        for step in installs
    )


def test_every_job_is_time_boxed_and_the_matrix_is_fail_fast():
    for name, job in _load()["jobs"].items():
        assert job.get("timeout-minutes") == 30, f"{name}: нужен timeout-minutes: 30"
        if "matrix" in job.get("strategy", {}):
            assert job["strategy"].get("fail-fast") is True, f"{name}: matrix без fail-fast"


def test_pip_cache_is_enabled_everywhere_python_is_set_up():
    setups = [step for step in _steps() if str(step.get("uses", "")).startswith("actions/setup-python")]
    assert setups, "нет actions/setup-python"
    for step in setups:
        assert step.get("with", {}).get("cache") == "pip", "setup-python без кэша pip"
        assert step["with"].get("cache-dependency-path") == "pyproject.toml"


def test_workflow_asks_for_read_only_contents():
    assert _load()["permissions"] == {"contents": "read"}


def test_no_secrets_are_used():
    text = _text()
    assert "secrets." not in text
    assert not re.search(r"(?im)^\s*(password|token|api[_-]?key)\s*:", text)


def test_no_absolute_host_paths():
    """Клон может лежать где угодно: пути только относительные или из env."""
    text = _text()
    assert not re.search(r"(?<![\w.$])/(?:Users|home|opt|mnt|private)/", text)
    assert not re.search(r"[A-Za-z]:\\\\", text)


def test_the_negative_fixtures_exist_and_are_wired_into_ci():
    """A validation gate is only a gate while the fixtures it names are there."""
    directory = ROOT / "tests" / "fixtures" / "validate"
    required = ["valid-task.yaml", "suite.json", "bad-id.yaml", "bad-category.yaml",
                "empty.yaml", "scalar.yaml", "list.yaml", "broken.yaml"]
    missing = [name for name in required if not (directory / name).is_file()]
    assert not missing, f"нет фикстур валидатора: {missing}"
    commands = "\n".join(str(step.get("run", "")) for step in _steps())
    assert "tests/fixtures/validate/bad-*.yaml" in commands, "негативные фикстуры не гоняются пачкой"
    for name in ("valid-task.yaml", "suite.json", "empty.yaml", "scalar.yaml", "list.yaml", "broken.yaml"):
        assert name in commands, f"фикстура {name} не используется в workflow"


def test_makefile_exposes_the_same_gates():
    text = MAKEFILE.read_text(encoding="utf-8")
    assert re.search(r"(?m)^ci:\s*\n(?:\t.*\n)*?\t.*compileall", text), "make ci без compileall"
    assert re.search(r"(?m)^ci:\s*\n(?:\t.*\n)*?\t.*pytest tests", text), "make ci без pytest"
    assert re.search(r"(?m)^docker-smoke:\s*\n\t.*test_sandbox\.py", text), "make docker-smoke без смоука"
