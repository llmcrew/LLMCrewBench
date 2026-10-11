"""The agent harness: prompt contract, isolation, event parsing, disk grading.

The agent path is proven without a live model: a tiny shell script stands in for
``pi``, speaking the same JSON event stream. That keeps the *contract* under test
(isolated config, disabled session/context, tools allowlist, disk-authoritative
grading) while a real model is only needed for the end-to-end smoke.
"""

from __future__ import annotations

import json
import os
import random
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dailybench import agent, art, core, visual
from dailybench.config import RunConfig
from dailybench.runner import Runner
from dailybench.tasks import coding_python, load_all


def _fake_pi(tmp_path: Path, script: str) -> Path:
    path = tmp_path / "fake-pi"
    path.write_text("#!/bin/sh\n" + script, encoding="utf-8")
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
    # Decoding policy travels as model fields: pi has no CLI flag for them.
    assert model["temperature"] == 0.0 and model["samplingParams"]["top_p"] == 1.0
    # reasoning_effort is not forced: with it, pi sends its own "medium" and a
    # stand may reject it (DeepSeek V4.1 wants low/high/xhigh/max), while the
    # chat path sends nothing — the harnesses must agree.
    assert model["compat"]["supportsReasoningEffort"] is False
    # Nothing else is written, so pi's own defaults apply untouched.
    assert sorted(p.name for p in folder.iterdir()) == ["models.json"]


def test_the_answer_ceiling_is_the_category_budget(tmp_path):
    """A chat-sized 4096 can be spent on thinking before anything is written."""
    cfg = RunConfig(model="m", max_new_tokens=1024, max_new_tokens_by_category={"C": 8192})
    folder = agent.config_dir(tmp_path, cfg, budget=cfg.max_new_tokens_by_category["C"])
    model = json.loads((folder / "models.json").read_text(encoding="utf-8"))["providers"]["bench"]["models"][0]
    assert model["maxTokens"] == 8192


# --------------------------------------------------------------------------- #
# Running the agent: isolation flags and event parsing
# --------------------------------------------------------------------------- #


def test_run_passes_the_isolation_flags(monkeypatch, tmp_path):
    """The harness must be pinned: a stray .pi/ or AGENTS.md must not reach it."""
    seen: dict[str, object] = {}
    script = _fake_pi(
        tmp_path,
        'echo "$@" > "$ARGS_OUT"\n'
        'printf \'%s\\n\' \'{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"ok"}],"usage":{"input":10,"output":2},"stopReason":"stop"}}\'\n',
    )
    monkeypatch.setenv(agent.AGENT_BIN_ENV, str(script))
    monkeypatch.setenv("ARGS_OUT", str(tmp_path / "args.txt"))
    cfg = RunConfig(base_url="http://x/v1", model="m")
    work = tmp_path / "work"
    work.mkdir()
    result = agent.run("привет", work, cfg, agent_dir=tmp_path)

    args = (tmp_path / "args.txt").read_text(encoding="utf-8")
    for flag in ("--print", "--mode json", "--no-session", "--no-context-files",
                 "--no-approve", "--offline", "--no-extensions", "--no-skills"):
        assert flag in args, f"{flag} не передан агенту"
    assert "--tools read,bash,edit,write" in args
    assert result.ok and result.turns == 1 and result.answer_tokens == 2


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
    cfg = RunConfig(base_url="http://x/v1", model="m", request_wall_clock=3.0)
    started = time.perf_counter()
    result = agent.run("p", work, cfg, agent_dir=tmp_path)
    elapsed = time.perf_counter() - started
    assert not result.ok and "не уложился" in (result.error or "")
    assert elapsed < 12, f"дочерний процесс держал пайп: {elapsed:.1f} с"
    time.sleep(0.5)
    for name in ("sleep 601.7", "sleep 602.9"):
        survivors = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True).stdout.strip()
        assert not survivors, f"{name} пережил таймаут (pid {survivors})"


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

    def fake_run(prompt, work_dir, cfg, *, agent_dir, timeout=None):
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
        id="A-тест", category="A", title="t", prompt="p", max_points=1.0,
        oracle={"reference": "x"},
    )


