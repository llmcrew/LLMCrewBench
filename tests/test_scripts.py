"""coder_loop counts repeats without rereading the log; sweep does not rebuild."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts import sweep

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "coder_loop.py"
_SPEC = importlib.util.spec_from_file_location("coder_loop", _SCRIPT)
coder_loop = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(coder_loop)


def _tool_line(name: str, args: dict) -> str:
    return json.dumps(
        {"type": "tool_execution_start", "toolName": name, "args": args},
        ensure_ascii=False,
    )


def _pid_alive(pid: int) -> bool:
    """True while the pid still has a running (not zombie) process."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87
        try:
            exit_code = ctypes.c_uint32()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("State:"):
                return not line.split()[1].startswith("Z")
    return True


def _force_dead(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _fake_harness(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-harness.py"
    script.write_text(body, encoding="utf-8")
    if os.name == "nt":
        launcher = tmp_path / "fake-harness.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return launcher
    launcher = tmp_path / "fake-harness"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n", encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def test_repeat_counter_counts_fed_lines_without_reopening_the_log(tmp_path, monkeypatch) -> None:
    """A pump-fed counter must not reread the log: that reread is quadratic.

    Breaks if counting reopens the log (the decoy line on disk would be seen,
    or Path.open would fire) or if fed lines are dropped.
    """
    log = tmp_path / "round.log"
    log.write_text(_tool_line("write", {"path": "decoy.py"}) + "\n", encoding="utf-8")
    opened: list[str] = []
    real_open = Path.open

    def tracking_open(self, mode="r", *args, **kwargs):
        if self == log:
            opened.append(mode if isinstance(mode, str) else repr(mode))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    counter = coder_loop._RepeatCounter()
    counter.feed(_tool_line("write", {"content": "x", "path": "a.py"}))
    counter.feed('{"type":"message_update","delta":"token"}')
    counter.feed(_tool_line("write", {"content": "x", "path": "a.py"}))
    counter.feed('{"type":"tool_execution_start" not-json')
    counter.feed(_tool_line("edit", {"path": "b.py"}))

    assert counter.counts == {
        'write:{"content": "x", "path": "a.py"}': 2,
        'edit:{"path": "b.py"}': 1,
    }
    assert opened == []


def test_round_timeout_kills_the_grandchild(tmp_path, monkeypatch) -> None:
    """A round wall-clock must kill the harness tree, not only the root pid.

    Breaks if timeout uses terminate/kill on the root alone: the grandchild
    keeps sleeping afterwards.
    """
    project = tmp_path / "project"
    project.mkdir()
    log = tmp_path / "logs" / "round.log"
    pids = tmp_path / "pids"
    body = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"Path({str(pids)!r}).write_text(f'{{os.getpid()}} {{child.pid}}', encoding='utf-8')\n"
        "time.sleep(120)\n"
    )
    monkeypatch.setattr(coder_loop, "HARNESS", _fake_harness(tmp_path, body))
    recorded: list[int] = []
    try:
        coder_loop.run_harness(
            project, "model", "do it", "read,edit,write", log, round_timeout=0.05
        )
        assert pids.is_file(), "fake did not record the grandchild pid"
        recorded = [int(part) for part in pids.read_text(encoding="utf-8").split()]
        assert len(recorded) == 2
        assert not _pid_alive(recorded[1]), f"grandchild {recorded[1]} still alive"
    finally:
        for pid in recorded:
            if _pid_alive(pid):
                _force_dead(pid)
        deadline = time.monotonic() + 2
        while not pids.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if pids.is_file():
            for part in pids.read_text(encoding="utf-8").split():
                pid = int(part)
                if _pid_alive(pid):
                    _force_dead(pid)


def test_suite_size_reads_recorded_per_task_and_does_not_rebuild(tmp_path, monkeypatch) -> None:
    """The denominator is the recorded suite, not a fresh build_suite.

    Breaks if sweep rebuilds the suite to count tasks, or if per_task length
    is not what _suite_size returns.
    """
    run_dir = tmp_path / "runs" / "model"
    run_dir.mkdir(parents=True)
    (run_dir / "suite.json").write_text(
        json.dumps(
            {
                "profile": "stretch-v1",
                "per_task": [
                    {"id": "A1", "category": "A"},
                    {"id": "K-1", "category": "K"},
                    {"id": "L-1", "category": "L"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("build_suite must not be called to count tasks")

    monkeypatch.setattr("dailybench.suite.build_suite", boom)

    assert sweep._suite_size(run_dir) == 3
