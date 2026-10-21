"""Snapshot/restore of the coder loop must never destroy the repository."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "coder_loop.py"
SPEC = importlib.util.spec_from_file_location("coder_loop", SCRIPT)
coder_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coder_loop)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (project / ".git" / "config").write_text("[core]\n")
    (project / "src" / "app.py").write_text("original\n")
    (project / "README.md").write_text("readme\n")
    return project


def test_snapshot_skips_service_paths(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "__pycache__").mkdir()
    (project / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"pyc")
    keep = tmp_path / "keep"

    coder_loop.snapshot(project, keep)

    assert (keep / "src" / "app.py").read_text() == "original\n"
    assert not (keep / ".git").exists()
    assert not (keep / "__pycache__").exists()


def test_restore_keeps_git_and_rolls_back_without_changes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    keep = tmp_path / "keep"
    coder_loop.snapshot(project, keep)

    # A destructive round: changed, deleted and invented files.
    (project / "src" / "app.py").write_text("broken\n")
    (project / "README.md").unlink()
    (project / "extra").mkdir()
    (project / "extra" / "new.py").write_text("x = 1\n")
    (project / "src" / "__pycache__").mkdir()
    (project / "src" / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"pyc")

    coder_loop.restore(keep, project)

    assert (project / ".git" / "HEAD").read_text() == "ref: refs/heads/main\n"
    assert (project / ".git" / "config").read_text() == "[core]\n"
    assert (project / "src" / "app.py").read_text() == "original\n"
    assert (project / "README.md").read_text() == "readme\n"
    assert not (project / "extra").exists()
    assert (project / "src" / "__pycache__" / "app.cpython-312.pyc").exists()
