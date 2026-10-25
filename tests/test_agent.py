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


def _fake_pi(tmp_path: Path, script: str, version: str = agent.PI_VERSION) -> Path:
    """A stand-in that answers ``--version`` like the pinned release.

    The harness probes every binary before it runs anything, so a fake that
    stayed silent would be refused before its script was reached.
    """
    path = tmp_path / "fake-pi"
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "{version}"; exit 0; fi\n' + script,
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _pid_alive(pid: int) -> bool:
    """Does the pid exist and is it not a zombie?"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(state) and not state.startswith("Z")


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


def _fake_taskkill(folder: Path) -> Path:
    """A stand-in for the Windows command; it kills the given tree on POSIX."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "taskkill"
    path.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "${TASKKILL_LOG:-/dev/null}"\n'
        "pid=$2\n"
        'for child in $(/bin/ps -Ao pid=,ppid= | /usr/bin/awk -v p="$pid" \'$2==p {print $1}\'); do\n'
        '  /bin/kill -9 "$child"\n'
        "done\n"
        '/bin/kill -9 "$pid"\n',
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


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
    fake = _fake_pi(tmp_path, "exit 0\n")
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(fake))
    assert agent.binary() == str(fake)


def test_missing_binary_is_installed(monkeypatch, tmp_path):
    """A machine without pi gets it installed, not a skipped category."""
    fake = _fake_pi(tmp_path, "exit 0\n")
    monkeypatch.delenv(agent.AGENT_BIN_ENV, raising=False)
    monkeypatch.setattr(agent.shutil, "which", lambda name: None)
    monkeypatch.setenv(agent.AGENT_INSTALL_ENV, f"cp {fake} {tmp_path}/installed && echo {tmp_path}/installed")

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
    script = (
        "sleep 0.3\n"
        'model=$(sed -n \'s/.*"id": "\\([^"]*\\)".*/\\1/p\' "$PI_CODING_AGENT_DIR/models.json")\n'
        'printf \'%s\\n\' "{\\"type\\":\\"message_end\\",\\"message\\":{\\"role\\":\\"assistant\\",'
        '\\"content\\":[{\\"type\\":\\"text\\",\\"text\\":\\"$model\\"}],'
        '\\"usage\\":{\\"input\\":10,\\"output\\":2},\\"stopReason\\":\\"stop\\"}}"\n'
    )
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(tmp_path, script)))
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
    script = _fake_pi(
        tmp_path,
        'printf \'%s\\n\' \'{"type":"compaction_start","reason":"threshold"}\'\n'
        'printf \'%s\\n\' \'{"type":"compaction_end","reason":"threshold","result":'
        '{"tokensBefore":9000,"estimatedTokensAfter":1200,"usage":{"input":9000,"output":30}}}\'\n'
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":100,"output":20},"stopReason":"stop"}}\'\n',
    )
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


def test_a_foreign_release_is_refused_with_the_install_hint(tmp_path):
    """0.80.10 rejected ``--`` before the first request; it must be named."""
    wrong = _fake_pi(tmp_path, "exit 0\n", version="0.80.10")
    with pytest.raises(agent.AgentVersionError) as caught:
        agent.verify(str(wrong))
    message = str(caught.value)
    assert "0.80.10" in message and agent.PI_VERSION in message
    assert f"@{agent.PI_VERSION}" in message and agent.AGENT_BIN_ENV in message
    assert caught.value.probed is not None and caught.value.probed.version == "0.80.10"


def test_a_newer_release_is_refused_too(tmp_path):
    with pytest.raises(agent.AgentVersionError):
        agent.verify(str(_fake_pi(tmp_path, "exit 0\n", version="1.0.1")))


def test_ensure_refuses_a_present_but_wrong_release(monkeypatch, tmp_path):
    """``make agent`` must fail on the version, not just install blindly."""
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(tmp_path, "exit 0\n", version="0.80.10")))
    with pytest.raises(agent.AgentVersionError):
        agent.ensure()


def test_an_unreadable_version_is_an_error_not_an_assumption(tmp_path):
    silent = _fake_pi(tmp_path, "exit 0\n", version="не число")
    with pytest.raises(agent.AgentVersionError):
        agent.verify(str(silent))


def test_the_written_models_json_does_not_replace_the_version_check(monkeypatch, tmp_path):
    """A file on disk proves nothing about the CLI; the binary is still probed."""
    script = (
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":10,"output":2},"stopReason":"stop"}}\'\n'
    )
    wrong = _fake_pi(tmp_path, script, version="0.80.10")
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
    fake = _fake_pi(tmp_path, "exit 0\n")
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
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(folder, "exit 0\n")))
    cfg = RunConfig(agent=True)
    before = run_identity(cfg, "suite")
    _fake_pi(folder, "exit 0\n", version="0.80.10")
    assert run_identity(cfg, "suite") != before
    assert run_identity(RunConfig(agent=False), "suite") == run_identity(RunConfig(agent=False), "suite")


# --------------------------------------------------------------------------- #
# Running the agent: isolation flags and event parsing
# --------------------------------------------------------------------------- #


def test_run_passes_the_isolation_flags(monkeypatch, tmp_path):
    """The harness must be pinned: a stray .pi/ or AGENTS.md must not reach it."""
    seen: dict[str, object] = {}
    script = _fake_pi(
        tmp_path,
        'printf \'%s\\n\' "$@" > "$ARGS_OUT"\n'
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":10,"output":2},"stopReason":"stop"}}\'\n',
    )
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
    script = _fake_pi(
        tmp_path,
        'printf \'%s\\n\' \'{"type":"tool_execution_start","toolCallId":"1","toolName":"write","args":{"path":"a.py"}}\'\n'
        'printf \'%s\\n\' \'{"type":"tool_execution_end","toolCallId":"1","toolName":"write",'
        '"result":{"content":[{"type":"text","text":"written"}]},"isError":false}\'\n'
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"сделано"}],"usage":{"input":50,"output":7,"reasoning":3},"stopReason":"stop"}}\'\n',
    )
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
    script = _fake_pi(
        tmp_path,
        "printf '%s\\n' "
        + f"'{json.dumps(leaked)}'"
        + "\n"
        + "printf '%s\\n' "
        + f"'{json.dumps(_assistant())}'"
        + "\n"
        + f"echo 'DAILYBENCH_API_KEY={secret}' >&2\n",
    )
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
    script = _fake_pi(
        tmp_path,
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant","content":[]}}\'\n'
        'echo "boom: provider 500" >&2\nexit 3\n',
    )
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert not result.ok and result.events, "события обязаны сохраниться для отчёта"
    assert "кодом 3" in (result.error or "") and "boom" in result.stderr_tail


def test_a_silent_agent_is_reported_not_scored(monkeypatch, tmp_path):
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(tmp_path, "exit 0\n")))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert not result.ok and result.error == "агент не выдал ни одного события"
    assert result.failure == "incomplete_scenario"


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
    script = "printf '%s\\n' '" + json.dumps(event) + "'\n"
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(tmp_path, script)))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert result.events, "событие обязано сохраниться"
    assert not result.ok
    assert "HTTP 400" in (result.error or "")
    assert result.failure == "model_protocol_failure"


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
    script = "printf '%s\\n' '" + json.dumps(event) + "'\n"
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(tmp_path, script)))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert not result.ok and result.failure == "provider_unavailable"


def test_the_agent_does_not_wait_for_stdin(monkeypatch, tmp_path):
    """A piped stdin used to hang the agent for the whole wall clock.

    ``pi`` reads stdin when it is a pipe and waits for EOF; a run started from a
    shell pipeline therefore produced no events at all until the timeout. The
    prompt travels in argv, so stdin must be empty.
    """
    script = _fake_pi(
        tmp_path,
        # Blocks unless stdin is at EOF.
        'read line\n'
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":1,"output":1},"stopReason":"stop"}}\'\n',
    )
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
    script = _fake_pi(
        tmp_path,
        # Own session, so killpg cannot see it; both levels outlive the timeout.
        f'{sys.executable} -c "import subprocess;subprocess.Popen([\'sleep\',\'601.7\'],'
        f'start_new_session=True);import time;time.sleep(600)" &\n'
        "sleep 602.9 &\n"
        "wait\n",
    )
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    cfg = RunConfig(base_url="http://x/v1", model="m", request_wall_clock=60.0, task_wall_clock=3.0)
    started = time.perf_counter()
    result = agent.run("p", work, cfg, agent_dir=tmp_path)
    elapsed = time.perf_counter() - started
    assert not result.ok and "не уложился" in (result.error or "")
    assert elapsed < 12, f"дочерний процесс держал пайп: {elapsed:.1f} с"
    time.sleep(0.5)
    for name in ("sleep 601.7", "sleep 602.9"):
        survivors = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True).stdout.strip()
        assert not survivors, f"{name} пережил таймаут (pid {survivors})"


def test_kill_tree_uses_taskkill_on_a_windows_host(monkeypatch, tmp_path):
    """Windows has neither ``os.getpgid`` nor ``os.killpg``; taskkill covers it."""
    tools = tmp_path / "bin"
    log = tmp_path / "taskkill.log"
    _fake_taskkill(tools)
    monkeypatch.setenv("TASKKILL_LOG", str(log))
    monkeypatch.setenv("PATH", str(tools))
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


def test_a_run_timeout_survives_a_windows_platform_stub(monkeypatch, tmp_path):
    """The R12 reproduction: no ``os.getpgid``/``killpg``, yet the timeout works."""
    tools = tmp_path / "bin"
    _fake_taskkill(tools)
    script = _fake_pi(
        tmp_path,
        "printf '%s\\n' "
        + f"'{json.dumps(_assistant('до'))}'"
        + "\nsleep 610.1\n",
    )
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    monkeypatch.setenv("PATH", os.pathsep.join([str(tools), "/bin", "/usr/bin"]))
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
    script = _fake_pi(
        tmp_path,
        f'{sys.executable} -c "import os,signal,time;'
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);os.chdir('/');"
        'time.sleep(608.1)" & echo $! >> ' + str(pids) + "\n"
        "sleep 608.2 & echo $! >> " + str(pids) + "\n"
        "wait\n",
    )
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
    script = _fake_pi(
        tmp_path,
        "printf '%s\\n' "
        + " ".join(f"'{json.dumps(record)}'" for record in records)
        + "\nsleep 609.1\n",
    )
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
    script = _fake_pi(
        tmp_path,
        # Own session and orphaned before the agent exits - the shape of a
        # model's `python -m http.server 8770 &` self-check. Its stdio must be
        # detached too, or it holds the benchmark's stdout pipe open and
        # ``communicate`` waits out the whole sleep instead of the agent exit.
        f'{sys.executable} -c "import subprocess;subprocess.Popen([\'sleep\',\'603.1\'],'
        f'start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)" &\n'
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":1,"output":1},"stopReason":"stop"}}\'\n'
        "exit 0\n",
    )
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    work = tmp_path / "w"
    work.mkdir()
    result = agent.run("p", work, RunConfig(base_url="http://x/v1", model="m"), agent_dir=tmp_path)
    assert result.ok, result.error
    time.sleep(0.3)
    survivors = subprocess.run(["pgrep", "-f", "sleep 603.1"], capture_output=True, text=True).stdout.strip()
    assert not survivors, f"сирота пережил чистый выход агента (pid {survivors})"


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

    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(tmp_path, "exit 0\n")))
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
    payload = json.dumps(_assistant("готово"), ensure_ascii=False).replace("'", "'\\''")
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(_fake_pi(tmp_path, f"printf '%s\\n' '{payload}'\n")))
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
