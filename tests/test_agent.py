"""The agent harness: prompt contract, isolation, event parsing, disk grading.

The agent path is proven without a live model: a tiny shell script stands in for
``pi``, speaking the same JSON event stream. That keeps the *contract* under test
(isolated config, disabled session/context, tools allowlist, disk-authoritative
grading) while a real model is only needed for the end-to-end smoke.
"""

from __future__ import annotations

import json
import hashlib
import os
import random
import shlex
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dailybench import agent, art, core, visual
from dailybench.config import RunConfig
from dailybench.runner import Runner
from dailybench.tasks import coding_python, load_all


_FAKE_VERSION_ENV = "DAILYBENCH_TEST_AGENT_VERSION"


def _fake_agent(tmp_path: Path, body: str) -> Path:
    """Build a Python fake behind a native executable launcher."""
    version = os.environ.get(_FAKE_VERSION_ENV, agent.PI_VERSION)
    script = tmp_path / "fake-agent.py"
    script.write_text(
        "import sys\n"
        'if sys.argv[1:] == ["--version"]:\n'
        f"    print({version!r})\n"
        "    raise SystemExit(0)\n" + body,
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = tmp_path / "fake-agent.cmd"
        launcher.write_text(
            f"@chcp 65001>nul\r\n@rem {version}\r\n@\"{sys.executable}\" \"{script}\" %*\r\n",
            encoding="utf-8",
        )
    else:
        launcher = tmp_path / "fake-agent"
        launcher.write_text(
            f"#!/bin/sh\n# {version}\nexec {shlex.quote(sys.executable)} "
            f"{shlex.quote(str(script))} \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return launcher


@pytest.mark.skipif(os.name != "nt", reason="проверяется чтение .cmd через cmd.exe")
def test_fake_agent_launcher_supports_non_ascii_paths(tmp_path):
    folder = tmp_path / "агент"
    folder.mkdir()
    launcher = _fake_agent(folder, "raise SystemExit(0)\n")
    result = subprocess.run(
        [str(launcher), "--version"], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == agent.PI_VERSION


_TASKKILL_BODY = """\
import os
import signal
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("TASKKILL_LOG")
if log:
    with Path(log).open("a", encoding="utf-8") as handle:
        handle.write(" ".join(args) + "\\n")
pid = int(args[args.index("/PID") + 1])
if os.name == "nt":
    real = Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe"
    subprocess.run([str(real), *args], check=False)
else:
    children = subprocess.run(
        ["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True, check=True
    ).stdout
    for line in children.splitlines():
        child, parent = map(int, line.split())
        if parent == pid:
            os.kill(child, signal.SIGKILL)
    os.kill(pid, signal.SIGKILL)
"""


def _pid_alive(pid: int) -> bool:
    """Does the pid exist and is it not a zombie?"""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not handle:
            # ERROR_INVALID_PARAMETER means the pid is gone; access denial does not.
            return ctypes.get_last_error() != 87
        try:
            exit_code = ctypes.c_uint32()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
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
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(state) and not state.startswith("Z")


@pytest.mark.skipif(os.name != "nt", reason="проверяется Win32 OpenProcess/GetExitCodeProcess")
def test_windows_pid_alive_distinguishes_live_and_dead_processes():
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        assert _pid_alive(proc.pid)
    finally:
        proc.kill()
        proc.wait()
    assert not _pid_alive(proc.pid)


def _real_pi() -> str | None:
    """The installed real ``pi`` when it is the pinned release, else ``None``."""
    found = agent.binary()
    if not found:
        return None
    try:
        agent.verify(found)
    except agent.AgentVersionError:
        return None
    return found


def _events(*records: dict) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


def _assistant(text: str = "готово", **extra) -> dict:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {"input": 100, "output": 20, **extra},
            "stopReason": "stop",
        },
    }


# --------------------------------------------------------------------------- #
# Binary resolution and configuration
# --------------------------------------------------------------------------- #


def test_binary_prefers_the_environment(monkeypatch, tmp_path):
    fake = _fake_agent(tmp_path, "raise SystemExit(0)\n")
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(fake))
    assert agent.binary() == str(fake)


def test_missing_binary_is_installed(monkeypatch, tmp_path):
    """A machine without pi gets it installed, not a skipped category."""
    fake = _fake_agent(tmp_path, "raise SystemExit(0)\n")
    monkeypatch.delenv(agent.AGENT_BIN_ENV, raising=False)
    monkeypatch.setattr(agent.shutil, "which", lambda name: None)
    monkeypatch.setenv("FAKE_INSTALL_SOURCE", str(fake))
    monkeypatch.setenv("FAKE_INSTALL_DEST", str(tmp_path / "installed"))
    monkeypatch.setenv(
        agent.AGENT_INSTALL_ENV,
        f'"{sys.executable}" -c "import os,shutil; '
        "shutil.copyfile(os.environ['FAKE_INSTALL_SOURCE'], os.environ['FAKE_INSTALL_DEST'])\"",
    )

    monkeypatch.setattr(
        agent,
        "binary",
        lambda: (str(tmp_path / "installed") if (tmp_path / "installed").is_file() else None),
    )
    ok, note = agent.install()
    assert ok and note == "установлен"


def test_config_dir_holds_one_isolated_provider(tmp_path):
    cfg = RunConfig(base_url="http://127.0.0.1:9999/v1", model="m", context_window=8192)
    folder = agent.config_dir(tmp_path, cfg, budget=8192)
    payload = json.loads((folder / "models.json").read_text(encoding="utf-8"))
    model = payload["providers"]["bench"]["models"][0]
    assert model["id"] == "m"
    assert model["maxTokens"] == 8192
    assert payload["providers"]["bench"]["baseUrl"] == "http://127.0.0.1:9999/v1"
    # Decoding policy travels in ``samplingParams``: pi copies that record onto
    # the request body, while a model-level temperature stays inert.
    assert model["samplingParams"]["temperature"] == 0.0 and model["samplingParams"]["top_p"] == 1.0
    # reasoning_effort is not forced: with it, pi sends its own "medium" and a
    # stand may reject it (DeepSeek V4.1 wants low/high/xhigh/max), while the
    # chat path sends nothing — the harnesses must agree.
    assert model["compat"]["supportsReasoningEffort"] is False
    # The pin is written next to it: compaction, retries and thinking level are
    # the harness's numbers, not whatever release defaults are current (R04).
    assert sorted(p.name for p in folder.iterdir()) == ["models.json", "settings.json", "wire.json"]


def test_the_answer_ceiling_is_the_category_budget(tmp_path):
    """A chat-sized 4096 can be spent on thinking before anything is written."""
    cfg = RunConfig(model="m", max_new_tokens=1024, max_new_tokens_by_category={"C": 8192})
    folder = agent.config_dir(tmp_path, cfg, budget=cfg.max_new_tokens_by_category["C"])
    model = json.loads((folder / "models.json").read_text(encoding="utf-8"))["providers"]["bench"]["models"][0]
    assert model["maxTokens"] == 8192


def test_the_pi_payload_carries_the_whole_generation_contract(tmp_path):
    """H08: the adapter dropped temperature/top_p and never sent a seed.

    The fields travel in ``samplingParams`` because pi copies that record onto
    the request body; a model-level ``temperature`` was inert. The reasoning
    budget has to be pinned too: leaving it out handed the decision to whatever
    the stand had configured.
    """
    cfg = RunConfig(base_url="http://127.0.0.1:9999/v1", model="m", temperature=0.23, top_p=0.71)
    folder = agent.config_dir(tmp_path, cfg, budget=1024, seed=77)
    model = json.loads((folder / "models.json").read_text(encoding="utf-8"))["providers"]["bench"]["models"][0]
    assert model["samplingParams"] == {
        "temperature": 0.23,
        "top_p": 0.71,
        "seed": 77,
        "reasoning_budget": cfg.reasoning_budget,
    }
    assert model["reasoning"] is True
    assert "temperature" not in model, "поля уровня модели адаптер не переносит"


def test_native_mode_leaves_the_pi_sampling_to_the_server(tmp_path):
    cfg = RunConfig(base_url="http://127.0.0.1:9999/v1", model="m", generation_mode="native")
    folder = agent.config_dir(tmp_path, cfg, budget=1024, seed=77)
    model = json.loads((folder / "models.json").read_text(encoding="utf-8"))["providers"]["bench"]["models"][0]
    assert "samplingParams" not in model
    assert model["maxTokens"] == 1024, "limit ответа остаётся за harness в обоих режимах"
    assert model["contextWindow"] == cfg.context_window


def test_the_pi_contract_is_recorded_and_complete(tmp_path):
    """The agent writes no wire log, so the declared contract must be checkable."""
    cfg = RunConfig(base_url="http://127.0.0.1:9999/v1", model="m")
    folder = agent.config_dir(tmp_path, cfg, budget=1024, seed=5)
    contract = json.loads((folder / "wire.json").read_text(encoding="utf-8"))
    assert contract["mode"] == "fixed"
    assert contract["seed"] == 5 and contract["output_limit"] == 1024
    assert contract["reasoning_budget"] == cfg.reasoning_budget
    assert contract["verified"] == "declared"
    assert sorted(p.name for p in folder.iterdir()) == ["models.json", "settings.json", "wire.json"]


def test_config_dir_is_unique_per_task_under_a_shared_parent(tmp_path):
    """R02: every work dir shares one temp parent, the config dir must not."""
    cfg = RunConfig(base_url="http://127.0.0.1:9999/v1", model="m")
    first = agent.config_dir(tmp_path, cfg, name="task-a")
    second = agent.config_dir(tmp_path, cfg, name="task-b")
    assert first != second and first.parent == second.parent == tmp_path
    assert first.name.startswith("task-a.") and second.name.startswith("task-b.")
    # Writes are atomic: no half-written documents are left behind, and the
    # models.json each run reads is its own.
    assert sorted(p.name for p in first.iterdir()) == ["models.json", "settings.json", "wire.json"]
    assert sorted(p.name for p in second.iterdir()) == ["models.json", "settings.json", "wire.json"]


def test_parallel_runs_do_not_overwrite_each_others_models_json(monkeypatch, tmp_path):
    """Two tasks used to share ``<temp>/pi-agent`` and the last writer won."""
    script = """\
import json
import os
import time
from pathlib import Path

time.sleep(0.3)
config = Path(os.environ["PI_CODING_AGENT_DIR"]) / "models.json"
model = json.loads(config.read_text(encoding="utf-8"))["providers"]["bench"]["models"][0]["id"]
event = {
    "type": "message_end",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": model}],
        "usage": {"input": 10, "output": 2},
        "stopReason": "stop",
    },
}
print(json.dumps(event), flush=True)
"""
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, script)))
    work_a, work_b = tmp_path / "task-a", tmp_path / "task-b"
    work_a.mkdir()
    work_b.mkdir()

    def solve(work, model):
        cfg = RunConfig(base_url="http://127.0.0.1:9999/v1", model=model)
        folder = agent.config_dir(work.parent, cfg, name=work.name)
        return agent.run("p", work, cfg, agent_dir=folder, timeout=60)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [f.result() for f in (
            pool.submit(solve, work_a, "model-a"),
            pool.submit(solve, work_b, "model-b"),
        )]
    assert sorted(r.text for r in results) == ["model-a", "model-b"]


def test_the_settings_pin_names_the_document_on_disk(tmp_path):
    """R04: compaction/retry/thinking are declared, and the claim is checkable."""
    cfg = RunConfig(model="m", context_window=32768)
    folder = agent.config_dir(tmp_path, cfg, budget=1024)
    pin = agent.settings_pin(folder)
    on_disk = (folder / "settings.json").read_text(encoding="utf-8")
    assert pin["version"] == agent.PI_SETTINGS_VERSION
    assert hashlib.sha256(on_disk.encode("utf-8")).hexdigest() == pin["sha256"]
    assert json.loads(on_disk) == pin["document"] == agent.pinned_settings(cfg)
    settings = pin["document"]
    assert settings["defaultTools"] == list(agent.AGENT_TOOLS)
    assert settings["defaultThinkingLevel"] == "medium"
    assert settings["compaction"]["enabled"] is True
    # The reserve stays inside the window and leaves room for an answer.
    assert 0 < settings["compaction"]["reserveTokens"] < cfg.context_window
    assert 0 < settings["compaction"]["keepRecentTokens"] < cfg.context_window
    assert settings["retry"]["maxRetries"] == agent.PI_RETRIES


def test_compaction_spend_is_part_of_the_usage(monkeypatch, tmp_path):
    """A summary is a model request; its tokens must not look free."""
    records = [
        {"type": "compaction_start", "reason": "threshold"},
        {
            "type": "compaction_end",
            "reason": "threshold",
            "result": {
                "tokensBefore": 9000,
                "estimatedTokensAfter": 1200,
                "usage": {"input": 9000, "output": 30},
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input": 100, "output": 20},
                "stopReason": "stop",
            },
        },
    ]
    body = "".join(f"print({json.dumps(record)!r}, flush=True)\n" for record in records)
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    cfg = RunConfig(base_url="http://127.0.0.1:9999/v1", model="m")
    work = tmp_path / "work"
    work.mkdir()
    result = agent.run("p", work, cfg, agent_dir=agent.config_dir(tmp_path, cfg))
    assert result.ok and result.turns == 1
    assert result.answer_tokens == 50 and result.summary_tokens == 30
    extra = result.as_extra()["agent"]
    assert extra["summary_tokens"] == 30
    assert extra["compactions"] == [{
        "reason": "threshold",
        "tokens_before": 9000,
        "tokens_after": 1200,
        "output_tokens": 30,
        "aborted": False,
        "error": None,
    }]


# --------------------------------------------------------------------------- #
# The pin: only the verified release runs, and it is named
# --------------------------------------------------------------------------- #


def test_parse_version_reads_the_reported_release():
    assert agent.parse_version("1.0.0\n") == (1, 0, 0)
    assert agent.parse_version("pi/0.80.10") == (0, 80, 10)
    assert agent.parse_version("no-number") is None


def test_a_foreign_release_is_refused_with_the_install_hint(monkeypatch, tmp_path):
    """0.80.10 rejected ``--`` before the first request; it must be named."""
    monkeypatch.setenv(_FAKE_VERSION_ENV, "0.80.10")
    wrong = _fake_agent(tmp_path, "raise SystemExit(0)\n")
    with pytest.raises(agent.AgentVersionError) as caught:
        agent.verify(str(wrong))
    message = str(caught.value)
    assert "0.80.10" in message and agent.PI_VERSION in message
    assert f"@{agent.PI_VERSION}" in message and agent.AGENT_BIN_ENV in message
    assert caught.value.probed is not None and caught.value.probed.version == "0.80.10"


def test_a_newer_release_is_refused_too(monkeypatch, tmp_path):
    monkeypatch.setenv(_FAKE_VERSION_ENV, "1.0.1")
    with pytest.raises(agent.AgentVersionError):
        agent.verify(str(_fake_agent(tmp_path, "raise SystemExit(0)\n")))


def test_ensure_refuses_a_present_but_wrong_release(monkeypatch, tmp_path):
    """``make agent`` must fail on the version, not just install blindly."""
    monkeypatch.setenv(_FAKE_VERSION_ENV, "0.80.10")
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, "raise SystemExit(0)\n")))
    with pytest.raises(agent.AgentVersionError):
        agent.ensure()


def test_an_unreadable_version_is_an_error_not_an_assumption(monkeypatch, tmp_path):
    monkeypatch.setenv(_FAKE_VERSION_ENV, "не число")
    silent = _fake_agent(tmp_path, "raise SystemExit(0)\n")
    with pytest.raises(agent.AgentVersionError):
        agent.verify(str(silent))


def test_the_written_models_json_does_not_replace_the_version_check(monkeypatch, tmp_path):
    """A file on disk proves nothing about the CLI; the binary is still probed."""
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input": 10, "output": 2},
            "stopReason": "stop",
        },
    }
    script = f"print({json.dumps(event)!r}, flush=True)\n"
    monkeypatch.setenv(_FAKE_VERSION_ENV, "0.80.10")
    wrong = _fake_agent(tmp_path, script)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(wrong))
    cfg = RunConfig(base_url="http://x/v1", model="m")
    agent_dir = agent.config_dir(tmp_path, cfg)
    assert (agent_dir / "models.json").is_file()
    work = tmp_path / "work"
    work.mkdir()
    result = agent.run("привет", work, cfg, agent_dir=agent_dir)
    assert not result.ok and result.events == [] and result.turns == 0
    assert "0.80.10" in (result.error or "")
    assert f"@{agent.PI_VERSION}" in (result.error or "")
    extra = result.as_extra()["agent"]
    assert extra["version"] == "0.80.10" and extra["required_version"] == agent.PI_VERSION
    assert len(result.agent_sha256 or "") == 64


def test_the_agent_identity_names_the_binary_and_digest(monkeypatch, tmp_path):
    fake = _fake_agent(tmp_path, "raise SystemExit(0)\n")
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(fake))
    identity = agent.identity()
    assert identity["required"] == agent.PI_VERSION and identity["found"] == agent.PI_VERSION
    assert identity["sha256"] and len(identity["sha256"]) == 64


def test_the_agent_identity_tolerates_a_missing_binary(monkeypatch):
    """``resume``/``doctor`` must be able to hash an identity without a binary."""
    monkeypatch.delenv(agent.AGENT_BIN_ENV, raising=False)
    monkeypatch.setattr(agent.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent, "PI_CANDIDATES", ())
    assert agent.identity() == {
        "required": agent.PI_VERSION,
        "found": None,
        "sha256": None,
        "settings": agent.PI_SETTINGS_VERSION,
    }


def test_the_agent_binary_is_part_of_the_run_identity(monkeypatch, tmp_path):
    """Two checkpoints made by different releases are not one observation."""
    from dailybench.runner import run_identity

    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(folder, "raise SystemExit(0)\n")))
    cfg = RunConfig(agent=True)
    before = run_identity(cfg, "suite")
    monkeypatch.setenv(_FAKE_VERSION_ENV, "0.80.10")
    _fake_agent(folder, "raise SystemExit(0)\n")
    assert run_identity(cfg, "suite") != before
    assert run_identity(RunConfig(agent=False), "suite") == run_identity(RunConfig(agent=False), "suite")


# --------------------------------------------------------------------------- #
# Running the agent: isolation flags and event parsing
# --------------------------------------------------------------------------- #


def test_run_passes_the_isolation_flags(monkeypatch, tmp_path):
    """The harness must be pinned: a stray .pi/ or AGENTS.md must not reach it."""
    seen: dict[str, object] = {}
    event = _assistant("ok")
    event["message"]["usage"]["output"] = 2
    body = (
        "import os\nfrom pathlib import Path\n"
        'Path(os.environ["ARGS_OUT"]).write_text("\\n".join(sys.argv[1:]) + "\\n", encoding="utf-8")\n'
        f"print({json.dumps(event)!r}, flush=True)\n"
    )
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    monkeypatch.setenv("ARGS_OUT", str(tmp_path / "args.txt"))
    cfg = RunConfig(base_url="http://x/v1", model="m")
    work = tmp_path / "work"
    work.mkdir()
    result = agent.run("привет", work, cfg, agent_dir=tmp_path)

    lines = (tmp_path / "args.txt").read_text(encoding="utf-8").splitlines()
    for flag in ("--print", "--mode", "json", "--no-session", "--no-context-files",
                 "--no-approve", "--offline", "--no-extensions", "--no-skills"):
        assert flag in lines, f"{flag} не передан агенту"
    tools = lines.index("--tools")
    assert lines[tools + 1] == "read,bash,edit,write"
    # The separator is the *supported* form for the pinned release, and the
    # prompt comes after it, so a prompt beginning with "-" is not a flag.
    separator = lines.index("--")
    assert lines[separator + 1] == "привет"
    assert result.ok and result.turns == 1 and result.answer_tokens == 2
    # The arrived binary is named in the provenance, not just assumed.
    assert result.as_extra()["agent"]["version"] == agent.PI_VERSION
    assert len(result.agent_sha256 or "") == 64


def test_events_become_tool_calls_and_usage(monkeypatch, tmp_path):
    records = [
        {"type": "tool_execution_start", "toolCallId": "1", "toolName": "write", "args": {"path": "a.py"}},
        {
            "type": "tool_execution_end",
            "toolCallId": "1",
            "toolName": "write",
            "result": {"content": [{"type": "text", "text": "written"}]},
            "isError": False,
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "сделано"}],
                "usage": {"input": 50, "output": 7, "reasoning": 3},
                "stopReason": "stop",
            },
        },
    ]
    body = "".join(f"print({json.dumps(record)!r}, flush=True)\n" for record in records)
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    cfg = RunConfig(base_url="http://x/v1", model="m")
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, cfg, agent_dir=tmp_path)

    assert [c["name"] for c in result.tool_calls] == ["write"]
    assert result.tool_calls[0]["result"] == "written"
    assert result.answer_tokens == 7 and result.reasoning_tokens == 3
    assert result.text == "сделано"


def test_the_endpoint_key_is_scrubbed_from_the_event_stream(monkeypatch, tmp_path):
    """The agent holds the key (``models.json``, ``DAILYBENCH_API_KEY``).

    ``cat``/``echo`` therefore put it into a tool result, and the event log is
    persisted into ``runs/``. Both the configured value and the generic
    ``Bearer``/``sk-`` shapes must be scrubbed before an event is recorded.
    """
    secret = "taskkey-9f3c1b7a2d"
    leaked = {
        "type": "tool_execution_end",
        "toolCallId": "1",
        "toolName": "bash",
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f'{{"apiKey":"{secret}"}} Authorization: Bearer abcdefghijklmnop '
                        "nested=sk-nested-0123456789"
                    ),
                }
            ]
        },
        "isError": False,
    }
    body = (
        f"print({json.dumps(leaked)!r}, flush=True)\n"
        f"print({json.dumps(_assistant())!r}, flush=True)\n"
        f"print('DAILYBENCH_API_KEY={secret}', file=sys.stderr)\n"
    )
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    cfg = RunConfig(base_url="http://x/v1", model="m", api_key=secret)
    result = agent.run("p", work, cfg, agent_dir=tmp_path)

    extra = result.as_extra(secret)["agent"]
    blob = json.dumps(extra["event_log"], ensure_ascii=False) + extra["stderr"] + json.dumps(result.tool_calls)
    assert secret not in blob
    assert "abcdefghijklmnop" not in blob and "sk-nested-0123456789" not in blob
    assert "***" in blob
    assert result.ok
    assert json.dumps(result.events).count(secret) == 0, "сырые события тоже уже очищены"


def test_a_nonzero_exit_keeps_the_events_and_the_reason(monkeypatch, tmp_path):
    event = {"type": "message_end", "message": {"role": "assistant", "content": []}}
    body = (
        f"print({json.dumps(event)!r}, flush=True)\n"
        "print('boom: provider 500', file=sys.stderr)\nraise SystemExit(3)\n"
    )
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert not result.ok and result.events, "события обязаны сохраниться для отчёта"
    assert "кодом 3" in (result.error or "") and "boom" in result.stderr_tail


@pytest.mark.skipif(os.name == "nt", reason="отрицательный код завершения проверяет семантику сигналов POSIX")
def test_a_killed_agent_is_a_resource_outcome_not_a_model_failure(monkeypatch, tmp_path):
    """``exit -9`` is the kernel or a watchdog, not the model's verdict.

    Classified as ``model_failure`` it looked like a wrong answer; the declared
    resource was exhausted (or someone intervened) and the run must say so.
    """
    event = {"type": "message_end", "message": {"role": "assistant", "content": []}}
    body = (
        "import os\nimport signal\n"
        f"print({json.dumps(event)!r}, flush=True)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert not result.ok
    assert result.failure == "candidate_resource_exhausted"
    assert "сигналом 9" in (result.error or "")
    assert result.events, "события до убийства обязаны сохраниться"


def test_a_silent_agent_is_reported_not_scored(monkeypatch, tmp_path):
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, "raise SystemExit(0)\n")))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert not result.ok and result.error == "агент не выдал ни одного события"
    assert result.failure == "infra_failure"


def test_a_stop_reason_error_on_exit_zero_is_an_error(monkeypatch, tmp_path):
    """Real pi answers HTTP 400 with exit 0 and ``stopReason: error`` alone.

    The process state used to be the only signal, so the run read ``ok`` with
    no error and the task was graded as a wrong answer instead of a broken one.
    """
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [],
            "usage": {"input": 10, "output": 0},
            "stopReason": "error",
            "errorMessage": "HTTP 400 from /v1/chat/completions: bad reasoning_content",
        },
    }
    script = f"print({json.dumps(event)!r}, flush=True)\n"
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, script)))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert result.events, "событие обязано сохраниться"
    assert not result.ok
    assert "HTTP 400" in (result.error or "")
    assert result.failure == "model_failure"


def test_an_endpoint_error_stop_is_infrastructure(monkeypatch, tmp_path):
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [],
            "stopReason": "error",
            "errorMessage": "HTTP 503 from /v1/chat/completions: unavailable",
        },
    }
    script = f"print({json.dumps(event)!r}, flush=True)\n"
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, script)))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert not result.ok and result.failure == "infra_failure"


def test_the_agent_does_not_wait_for_stdin(monkeypatch, tmp_path):
    """A piped stdin used to hang the agent for the whole wall clock.

    ``pi`` reads stdin when it is a pipe and waits for EOF; a run started from a
    shell pipeline therefore produced no events at all until the timeout. The
    prompt travels in argv, so stdin must be empty.
    """
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input": 1, "output": 1},
            "stopReason": "stop",
        },
    }
    body = f"sys.stdin.read()\nprint({json.dumps(event)!r}, flush=True)\n"
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    cfg = RunConfig(base_url="http://x/v1", model="m", request_wall_clock=5.0)
    result = agent.run("p", work, cfg, agent_dir=tmp_path)
    assert result.ok, f"агент ждал stdin: {result.error}"


def test_a_timeout_kills_the_whole_process_tree(monkeypatch, tmp_path):
    """The orphaned tool shell is what poisons the *next* task.

    ``subprocess.run(timeout=)`` killed only the agent process: the shell it
    spawned stayed alive (PPID 1), kept its hard-coded debugging port, and the
    next task's identical self-check then hung on the occupied port too. The
    whole group must die, and the pipe must be drained *after* the kill — the
    children hold the stdout pipe open, so draining first blocks again.

    The grandchild runs in its *own session* on purpose: ``pi``'s tool shell is
    started detached, so a plain ``killpg`` on the agent's group misses a
    Chrome a self-check script launched, and only a parent-chain walk reaches it.
    """
    body = """\
import os
import subprocess
import sys
import time
from pathlib import Path

child_code = (
    "import subprocess,sys,time;"
    "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(601.7)']);"
    "print(p.pid,flush=True);time.sleep(600)"
)
detached = {"start_new_session": True} if os.name != "nt" else {}
child = subprocess.Popen(
    [sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True, **detached
)
grandchild = int(child.stdout.readline())
other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(602.9)"])
Path(os.environ["AGENT_TEST_PIDS"]).write_text(
    f"{child.pid} {grandchild} {other.pid}", encoding="utf-8"
)
time.sleep(600)
"""
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    pids = tmp_path / "pids"
    monkeypatch.setenv("AGENT_TEST_PIDS", str(pids))
    work = tmp_path / "w"
    work.mkdir()
    cfg = RunConfig(base_url="http://x/v1", model="m", request_wall_clock=60.0, task_wall_clock=3.0)
    started = time.perf_counter()
    result = agent.run("p", work, cfg, agent_dir=tmp_path)
    elapsed = time.perf_counter() - started
    assert not result.ok and "не уложился" in (result.error or "")
    assert elapsed < 12, f"дочерний процесс держал пайп: {elapsed:.1f} с"
    recorded = [int(pid) for pid in pids.read_text(encoding="utf-8").split()]
    assert len(recorded) == 3
    time.sleep(0.5)
    alive = [pid for pid in recorded if _pid_alive(pid)]
    assert not alive, f"дети пережили таймаут: {alive}"


def test_keyboard_interrupt_kills_the_agent_tree(monkeypatch, tmp_path):
    """Ctrl-C must not leave the fake agent's grandchild running."""
    body = """\
import os
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(611.1)"])
Path(os.environ["AGENT_TEST_PIDS"]).write_text(
    f"{os.getpid()} {child.pid}", encoding="utf-8"
)
time.sleep(600)
"""
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, body)))
    pids = tmp_path / "interrupt-pids"
    monkeypatch.setenv("AGENT_TEST_PIDS", str(pids))
    work = tmp_path / "w"
    work.mkdir()
    launched = []
    real_popen = agent.subprocess.Popen

    def interrupt(_proc, *_args, **_kwargs):
        deadline = time.monotonic() + 5
        while not pids.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pids.is_file(), "фейковый агент не записал pid дерева"
        raise KeyboardInterrupt

    def tracked_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if "--print" not in proc.args:
            return proc
        launched.append(proc)

        def communicate(*communicate_args, **communicate_kwargs):
            return interrupt(proc, *communicate_args, **communicate_kwargs)

        monkeypatch.setattr(proc, "communicate", communicate)
        return proc

    monkeypatch.setattr(agent.subprocess, "Popen", tracked_popen)
    # Isolate communicate's cleanup contract from the independent cwd sweep.
    monkeypatch.setattr(agent, "_kill_strays", lambda _work_dir: None)
    try:
        with pytest.raises(KeyboardInterrupt):
            agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
        recorded = [int(pid) for pid in pids.read_text(encoding="utf-8").split()]
        assert len(recorded) == 2
        alive = [pid for pid in recorded if _pid_alive(pid)]
        assert not alive, f"Ctrl-C оставил процессы агента: {alive}"
    finally:
        if launched and launched[0].poll() is None:
            agent._kill_tree(launched[0])


def test_kill_tree_uses_taskkill_on_a_windows_host(monkeypatch, tmp_path):
    """Windows has neither ``os.getpgid`` nor ``os.killpg``; taskkill covers it."""
    tools = tmp_path / "bin"
    log = tmp_path / "taskkill.log"
    tools.mkdir()
    taskkill = _fake_agent(tools, _TASKKILL_BODY)
    taskkill = taskkill.rename(tools / ("taskkill.cmd" if os.name == "nt" else "taskkill"))
    monkeypatch.setenv("TASKKILL_LOG", str(log))
    monkeypatch.setenv("PATH", os.pathsep.join([str(tools), os.environ.get("PATH", "")]))
    real_run = agent.subprocess.run

    def run_taskkill(command, *args, **kwargs):
        if command and command[0] == "taskkill":
            command = [str(taskkill), *command[1:]]
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(agent.subprocess, "run", run_taskkill)
    monkeypatch.setattr(agent.sys, "platform", "win32")
    # On a real Windows host these attributes do not exist at all.
    monkeypatch.delattr(agent.os, "getpgid", raising=False)
    monkeypatch.delattr(agent.os, "killpg", raising=False)
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(607.1)"])
    try:
        agent._kill_tree(proc)
        assert proc.poll() is not None
        assert f"/PID {proc.pid} /T /F" in log.read_text(encoding="utf-8")
    finally:
        if proc.poll() is None:  # pragma: no cover - a broken path
            proc.kill()
            proc.wait()


@pytest.mark.skipif(os.name != "nt", reason="AssignProcessToJobObject существует только на Windows")
def test_job_attach_failure_still_kills_by_taskkill(monkeypatch, tmp_path):
    calls = []

    class _Kernel32:
        def CreateJobObjectW(self, *_args):
            return 7

        def AssignProcessToJobObject(self, handle, process):
            import ctypes

            ctypes.set_last_error(6)
            calls.append(("assign", handle, process))
            return False

        def CloseHandle(self, handle):
            calls.append(("close", handle))
            return True

    tools = tmp_path / "bin"
    tools.mkdir()
    taskkill = _fake_agent(tools, _TASKKILL_BODY)
    taskkill = taskkill.rename(tools / "taskkill.cmd")
    log = tmp_path / "taskkill.log"
    monkeypatch.setenv("TASKKILL_LOG", str(log))
    monkeypatch.setenv("PATH", os.pathsep.join([str(tools), os.environ.get("PATH", "")]))
    real_run = agent.subprocess.run

    def run_taskkill(command, *args, **kwargs):
        if command and command[0] == "taskkill":
            command = [str(taskkill), *command[1:]]
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(agent.subprocess, "run", run_taskkill)
    monkeypatch.setattr(agent, "_win32", lambda: _Kernel32())
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(607.15)"])
    work = tmp_path / "w"
    work.mkdir()
    try:
        agent._guard_process(proc, work)
        assert calls and calls[0][0] == "assign"
        assert str(work) not in agent._OPEN_JOBS
        agent._kill_tree(proc)
        assert proc.poll() is not None
        assert f"/PID {proc.pid} /T /F" in log.read_text(encoding="utf-8")
        assert agent._JOB_ATTACH_ERRORS[proc.pid] == "не прикреплён: WinError 6"
    finally:
        getattr(agent, "_JOB_ATTACH_ERRORS", {}).pop(proc.pid, None)
        agent._OPEN_JOBS.pop(str(work), None)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.skipif(os.name != "nt", reason="проверяется аварийный Windows Job Object и taskkill")
def test_keyboard_interrupt_with_failed_job_uses_taskkill_and_sweeps(monkeypatch, tmp_path):
    body = """\
import os
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(611.2)"])
Path(os.environ["AGENT_TEST_PIDS"]).write_text(
    f"{os.getpid()} {child.pid}", encoding="utf-8"
)
time.sleep(600)
"""

    class _Kernel32:
        def CreateJobObjectW(self, *_args):
            return 7

        def AssignProcessToJobObject(self, _handle, _process):
            import ctypes

            ctypes.set_last_error(6)
            return False

        def CloseHandle(self, _handle):
            return True

    tools = tmp_path / "bin"
    tools.mkdir()
    taskkill = _fake_agent(tools, _TASKKILL_BODY)
    taskkill = taskkill.rename(tools / "taskkill.cmd")
    log = tmp_path / "taskkill.log"
    monkeypatch.setenv("TASKKILL_LOG", str(log))
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, body)))
    pids = tmp_path / "interrupt-pids"
    monkeypatch.setenv("AGENT_TEST_PIDS", str(pids))
    real_run = agent.subprocess.run

    def run_taskkill(command, *args, **kwargs):
        if command and command[0] == "taskkill":
            command = [str(taskkill), *command[1:]]
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(agent.subprocess, "run", run_taskkill)
    monkeypatch.setattr(agent, "_win32", lambda: _Kernel32())
    launched = []
    real_popen = agent.subprocess.Popen

    def tracked_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if "--print" not in proc.args:
            return proc
        launched.append(proc)

        def communicate(*_args, **_kwargs):
            deadline = time.monotonic() + 5
            while not pids.is_file() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert pids.is_file(), "фейковый агент не записал pid дерева"
            raise KeyboardInterrupt

        monkeypatch.setattr(proc, "communicate", communicate)
        return proc

    monkeypatch.setattr(agent.subprocess, "Popen", tracked_popen)
    work = tmp_path / "w"
    work.mkdir()
    try:
        with pytest.raises(KeyboardInterrupt):
            agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
        assert f"/PID {launched[0].pid} /T /F" in log.read_text(encoding="utf-8")
        recorded = [int(pid) for pid in pids.read_text(encoding="utf-8").split()]
        assert len(recorded) == 2
        alive = [pid for pid in recorded if _pid_alive(pid)]
        assert not alive, f"Ctrl-C оставил процессы агента: {alive}"
    finally:
        if launched:
            getattr(agent, "_JOB_ATTACH_ERRORS", {}).pop(launched[0].pid, None)
            if launched[0].poll() is None:
                agent._kill_tree(launched[0])


def test_a_run_timeout_survives_a_windows_platform_stub(monkeypatch, tmp_path):
    """The R12 reproduction: no ``os.getpgid``/``killpg``, yet the timeout works."""
    tools = tmp_path / "bin"
    tools.mkdir()
    taskkill = _fake_agent(tools, _TASKKILL_BODY)
    taskkill = taskkill.rename(tools / ("taskkill.cmd" if os.name == "nt" else "taskkill"))
    body = f"print({json.dumps(_assistant('до'))!r}, flush=True)\nimport time\ntime.sleep(610.1)\n"
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    monkeypatch.setenv("PATH", os.pathsep.join([str(tools), os.environ.get("PATH", "")]))
    monkeypatch.setattr(agent.sys, "platform", "win32")
    monkeypatch.delattr(agent.os, "getpgid", raising=False)
    monkeypatch.delattr(agent.os, "killpg", raising=False)
    # ``shutil.which`` switches to ``_winapi`` by platform alone.
    monkeypatch.setattr(
        agent.shutil, "_winapi", type("_WinAPI", (), {"NeedCurrentDirectoryForExePath": staticmethod(lambda _n: False)})(),
        raising=False,
    )
    work = tmp_path / "w"
    work.mkdir()
    started = time.perf_counter()
    result = agent.run(
        "p",
        work,
        RunConfig(base_url="http://x/v1", model="m", request_wall_clock=60.0, task_wall_clock=1.0),
        agent_dir=tmp_path,
    )
    assert not result.ok and "не уложился" in (result.error or "")
    assert result.turns == 1 and result.answer_tokens == 20
    assert time.perf_counter() - started < 12


def test_kill_tree_falls_back_to_proc_kill_without_taskkill(monkeypatch, tmp_path):
    """A stripped Windows image has no taskkill; the root must still die."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(agent.sys, "platform", "win32")
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(607.2)"])
    try:
        agent._kill_tree(proc)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:  # pragma: no cover
            proc.kill()
            proc.wait()


def test_a_windows_job_covers_the_tree_even_without_taskkill(monkeypatch, tmp_path):
    """When a job can be attached it is the transport; taskkill is not needed."""
    calls: list[tuple] = []

    class _Kernel32:
        def CreateJobObjectW(self, *_args):
            calls.append(("create",))
            return 7

        def AssignProcessToJobObject(self, handle, pid):
            calls.append(("assign", handle, pid))
            return True

        def TerminateJobObject(self, handle, code):
            calls.append(("terminate", handle, code))
            return True

        def CloseHandle(self, handle):
            calls.append(("close", handle))
            return True

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(agent.sys, "platform", "win32")
    monkeypatch.setattr(agent, "_win32", lambda: _Kernel32())
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(607.3)"])
    if os.name != "nt":
        proc._handle = 99  # POSIX Popen has no handle; the job code must still work
    work = tmp_path / "w"
    work.mkdir()
    try:
        agent._guard_process(proc, work)
        assert str(work) in agent._OPEN_JOBS
        agent._kill_tree(proc)
        assert proc.poll() is not None
        assert ("terminate", 7, 1) in calls and ("close", 7) in calls
        assert not agent._PROC_JOBS
    finally:
        agent._OPEN_JOBS.pop(str(work), None)
        if proc.poll() is None:  # pragma: no cover
            proc.kill()
            proc.wait()


def test_descendants_fall_back_when_ps_is_missing(monkeypatch):
    """``ps`` is optional: without it the walk must not raise."""
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(agent.shutil, "which", lambda _name: None)
    assert os.getpid() in agent._descendants(os.getpid())


def test_a_term_ignoring_descendant_that_changed_cwd_is_killed(monkeypatch, tmp_path):
    """The field shape: the child survives its root, ignores TERM and leaves the cwd."""
    pids = tmp_path / "pids"
    body = """\
import os
import subprocess
import sys
import time
from pathlib import Path

first = subprocess.Popen([
    sys.executable,
    "-c",
    "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
    "os.chdir('/');time.sleep(608.1)",
])
second = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(608.2)"])
Path(os.environ["AGENT_TEST_PIDS"]).write_text(
    f"{first.pid} {second.pid}", encoding="utf-8"
)
time.sleep(600)
"""
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv("AGENT_TEST_PIDS", str(pids))
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run(
        "p",
        work,
        RunConfig(base_url="http://x/v1", model="m", request_wall_clock=60.0, task_wall_clock=1.0),
        agent_dir=tmp_path,
    )
    assert not result.ok and "не уложился" in (result.error or "")
    time.sleep(0.3)
    # Check the exact child pids: a marker regex can hit unrelated long-lived
    # processes (browser argv), while the script writes the real ones down.
    recorded = [int(pid) for pid in pids.read_text().split()]
    assert len(recorded) == 2
    alive = [pid for pid in recorded if _pid_alive(pid)]
    assert not alive, f"дети пережили таймаут: {alive}"


def test_a_timeout_keeps_the_measured_events_and_spend(monkeypatch, tmp_path):
    """A timeout is an outcome, not amnesia: usage and tool traffic stay recorded."""
    records = [
        {
            "type": "tool_execution_start",
            "toolCallId": "1",
            "toolName": "write",
            "args": {"path": "a.py"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "1",
            "toolName": "write",
            "result": {"content": [{"type": "text", "text": "written"}]},
            "isError": False,
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "частично"}],
                "usage": {"input": 50, "output": 7, "reasoning": 3},
                "stopReason": "stop",
            },
        },
    ]
    body = "".join(f"print({json.dumps(record)!r}, flush=True)\n" for record in records)
    body += "import time\ntime.sleep(609.1)\n"
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run(
        "p",
        work,
        RunConfig(base_url="http://x/v1", model="m", request_wall_clock=60.0, task_wall_clock=1.0),
        agent_dir=tmp_path,
    )

    assert not result.ok and "не уложился" in (result.error or "")
    assert result.turns == 1
    assert result.prompt_tokens_last == 50
    assert result.answer_tokens == 7 and result.reasoning_tokens == 3
    assert result.text == "частично"
    assert [call["name"] for call in result.tool_calls] == ["write"]
    assert result.tool_calls[0]["result"] == "written"

    transcript = core.Transcript(task_id="T-1", category="C", extra={})
    transcript.extra.update(result.as_extra())
    extra = transcript.extra["agent"]
    kinds = [event.get("type") for event in extra["event_log"]]
    assert extra["events"] == len(records) and extra["tool_names"] == ["write"]
    assert "tool_execution_start" in kinds and "message_end" in kinds
    assistant = [event for event in extra["event_log"] if event.get("type") == "message_end"][0]
    assert assistant["message"]["usage"]["output"] == 7


@pytest.mark.skipif(os.name == "nt", reason="поиск сирот по cwd через /proc и lsof доступен только на POSIX")
def test_the_exit_hook_sweeps_when_the_stack_never_unwinds(tmp_path):
    """A debugger's ``Py_Exit`` skips ``finally``; the atexit hook must not.

    The killed run proved the gap: ``PyRun_SimpleString`` raising ``SystemExit``
    ends the interpreter without unwinding ``run()``, and the surviving
    ``python -m http.server`` kept its port for an hour. The hook therefore must
    sweep by the registry of open work dirs, not by the stack frame that is about
    to disappear.
    """
    work = tmp_path / "w"
    work.mkdir()
    survivor = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(604.3)"],
        cwd=str(work),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)
    assert agent._strays_in(work), "тест не воспроизвёл ситуацию: сирота не найдена по cwd"
    # Exactly the state at a Py_Exit: the run registered its dir, the stack never
    # reached the ``finally`` that would have swept it.
    agent._OPEN_WORK_DIRS[str(work)] = work
    agent._sweep_on_exit()
    assert survivor.poll() is not None or agent._strays_in(work) == []
    assert str(work) not in agent._OPEN_WORK_DIRS


def test_a_clean_exit_still_sweeps_a_detached_survivor(monkeypatch, tmp_path):
    """A model's own ``(server &)`` outlives a *successful* agent run.

    The timeout path kills the process tree, but a clean exit never walked it:
    the detached server stayed bound to its hard-coded port, and the next task's
    identical script hung on the occupied port for the whole wall clock. After
    reparenting the parent chain is gone, so the sweep keys on the one marker
    that survives: the cwd, a per-task working copy.
    """
    body = """\
import os
import subprocess
import sys
from pathlib import Path

detached = {"start_new_session": True} if os.name != "nt" else {}
survivor = subprocess.Popen(
    [sys.executable, "-c", "import time;time.sleep(603.1)"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    **detached,
)
Path(os.environ["AGENT_TEST_PIDS"]).write_text(str(survivor.pid), encoding="utf-8")
"""
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input": 1, "output": 1},
            "stopReason": "stop",
        },
    }
    body += f"print({json.dumps(event)!r}, flush=True)\n"
    script = _fake_agent(tmp_path, body)
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    pids = tmp_path / "survivor-pid"
    monkeypatch.setenv("AGENT_TEST_PIDS", str(pids))
    work = tmp_path / "w"
    work.mkdir()
    cfg = RunConfig(base_url="http://x/v1", model="m")
    result = agent.run("p", work, cfg, agent_dir=tmp_path)
    assert result.ok, result.error
    survivor = int(pids.read_text(encoding="utf-8"))
    time.sleep(0.3)
    assert not _pid_alive(survivor), f"сирота пережил чистый выход агента (pid {survivor})"


# --------------------------------------------------------------------------- #
# Working copies
# --------------------------------------------------------------------------- #




def test_the_prompt_names_the_graded_file():
    load_all()
    task = coding_python.generate(random.Random(2), RunConfig())
    prompt = agent.prompt_for(task)
    assert task.prompt.rstrip() in prompt
    assert f"`{coding_python.TARGET_FILE}`" in prompt
    assert "осталось на диске" in prompt



# --------------------------------------------------------------------------- #
# Grading is from disk
# --------------------------------------------------------------------------- #


def _agent_transcript(task: core.TaskInstance, work_dir: Path, text: str = "") -> core.Transcript:
    transcript = core.Transcript(task_id=task.id, category=task.category, extra={"harness": "agent"})
    transcript.extra["work_dir"] = str(work_dir)
    if text:
        transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=text)))
    return transcript


def test_the_agent_answer_is_read_from_disk(tmp_path):
    load_all()
    task = coding_python.generate(random.Random(3), RunConfig())
    work = agent.prepare_work_dir(task)
    (work / coding_python.TARGET_FILE).write_text("правильный ответ", encoding="utf-8")
    transcript = _agent_transcript(task, work, text="а в чате я написал другое")
    assert agent.answer_text(task, transcript, transcript.final_text) == "правильный ответ"


def test_an_agent_that_never_wrote_the_file_scores_zero(tmp_path):
    """Leaving the stub in place is a result: it does not implement the task."""
    load_all()
    task = coding_python.generate(random.Random(4), RunConfig())
    work = agent.prepare_work_dir(task)
    transcript = _agent_transcript(task, work, text=task.oracle["reference"])
    assert "NotImplementedError" in agent.answer_text(task, transcript, transcript.final_text)


def test_a_chat_run_grades_the_printed_file_not_the_stub(tmp_path):
    """A chat run also has a working copy — and it holds the untouched stub."""
    load_all()
    task = coding_python.generate(random.Random(5), RunConfig())
    work = agent.prepare_work_dir(task)
    transcript = core.Transcript(
        task_id=task.id, category=task.category, extra={"work_dir": str(work), "harness": "chat"}
    )
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=task.oracle["reference"])))
    assert agent.answer_text(task, transcript, transcript.final_text) == task.oracle["reference"]


def test_the_visual_page_is_read_from_disk(tmp_path):
    load_all()
    task = art.build_task(
        task_id="L-T", title="t", prompt="нарисуй",
        golden="<html>gold</html>",
    )
    work = agent.prepare_work_dir(task)
    (work / visual.PAGE_FILE).write_text("<!DOCTYPE html><html>disk</html>", encoding="utf-8")
    transcript = _agent_transcript(task, work, text="```html\n<html>chat</html>\n```")
    assert "disk" in visual.page_of(task, transcript)


# --------------------------------------------------------------------------- #
# The runner switch
# --------------------------------------------------------------------------- #


def test_the_runner_uses_the_agent_only_for_its_categories(monkeypatch, tmp_path):
    load_all()
    calls: list[str] = []

    def fake_run(prompt, work_dir, cfg, *, agent_dir, timeout=None, sandbox=None):
        calls.append(str(work_dir))
        return agent.AgentRun(ok=True, text="агент", duration_s=0.1, turns=1, answer_tokens=1)

    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, "raise SystemExit(0)\n")))
    monkeypatch.setattr(agent, "run", fake_run)

    cfg = RunConfig(agent=True, agent_categories=("C",))
    coding = coding_python.generate(random.Random(6), cfg)
    run = Runner(cfg, _UnusedProvider()).run_task(coding)
    assert run.transcript.extra["harness"] == "agent"
    assert calls, "агент обязан быть вызван"
    assert run.transcript.turns and run.transcript.turns[0].reply.text == "агент"

    other = _shell_task()
    run_other = Runner(cfg, _UnusedProvider()).run_task(other)
    assert run_other.transcript.extra["harness"] == "chat"


def test_the_settings_pin_reaches_the_provenance(monkeypatch, tmp_path):
    """R04: the transcript names the pinned settings and they match the disk."""
    load_all()
    payload = json.dumps(_assistant("готово"))
    body = f"print({payload!r}, flush=True)\n"
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_agent(tmp_path, body)))
    cfg = RunConfig(agent=True, agent_categories=("C",), model="m", context_window=32768)
    task = coding_python.generate(random.Random(6), cfg)
    run = Runner(cfg, _UnusedProvider()).run_task(task)

    pin = run.transcript.extra["agent"]["settings"]
    assert pin["version"] == agent.PI_SETTINGS_VERSION
    assert pin["document"] == agent.pinned_settings(cfg)
    # The task root is swept after the task, so the declared bytes are checked
    # by rendering the document exactly as ``config_dir`` writes it.
    on_disk = json.dumps(pin["document"], ensure_ascii=False, indent=2)
    assert hashlib.sha256(on_disk.encode("utf-8")).hexdigest() == pin["sha256"]
    assert pin["path"].endswith("settings.json")


def test_no_agent_keeps_the_chat_path(monkeypatch, tmp_path):
    load_all()
    monkeypatch.setattr(agent, "run", lambda *a, **kw: pytest.fail("агент не должен запускаться"))
    cfg = RunConfig(agent=False)
    task = coding_python.generate(random.Random(7), cfg)
    run = Runner(cfg, _UnusedProvider()).run_task(task)
    assert run.transcript.extra["harness"] == "chat"


class _UnusedProvider:
    """The agent talks to a real endpoint through pi; the runner needs a stub."""

    name = "unused"

    def chat(self, messages, tools=None, **kwargs):
        return core.ModelReply(text="не использовался", prompt_tokens=1, completion_tokens=1)

    def tokenize(self, text):
        return None

    def describe(self):
        return {"provider": "unused"}


def _shell_task() -> core.TaskInstance:
    return core.TaskInstance(
        id="A-test", category="A", title="t", prompt="p", max_points=1.0,
        oracle={"reference": "x"},
    )


# --------------------------------------------------------------------------- #
# The real binary
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(_real_pi() is None, reason="нужен установленный pi ровно закреплённой версии")
def test_a_real_pi_reaches_a_local_endpoint(tmp_path):
    """The shipped CLI itself must accept our argv and the written models.json.

    A replayed event script proves the parser; only the real binary proves that
    the flags and provider fields are understood, and that the C task survives
    to the first model request.
    """
    import httpstub

    base_url, server = httpstub.start(reply_text="готово")
    try:
        cfg = RunConfig(base_url=base_url, model="stub-model", max_new_tokens=512, context_window=8192)
        task = coding_python.generate(random.Random(11), cfg)
        work = agent.prepare_work_dir(task, tmp_path)
        agent_dir = agent.config_dir(work.parent, cfg, budget=512)
        result = agent.run(agent.prompt_for(task), work, cfg, agent_dir=agent_dir, timeout=180)
        detail = f"{result.error or ''}\n{result.stderr_tail[-1500:]}"
        payloads = httpstub.chat_payloads(server)
        assert payloads, f"pi {agent.PI_VERSION} не дошёл до первого запроса: {detail}"
        assert result.ok, detail
        assert result.turns == 1
        assert result.agent_version == agent.PI_VERSION
        assert len(result.agent_sha256 or "") == 64
        sent = json.dumps(payloads[0], ensure_ascii=False)
        assert task.prompt.splitlines()[0][:30] in sent
        assert payloads[0]["model"] == "stub-model"
    finally:
        httpstub.stop(server)
