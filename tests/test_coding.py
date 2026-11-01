"""The C-1 verdict belongs to the harness, never to the candidate.

The candidate is imported into a child process the supervisor spawned, so it
can print whatever it likes — including a ``REPORT <path>`` line pointing at a
JSONL it wrote itself. Stdout is only a journal; the supervisor reads the child
report, requires the end marker the child appends after the whole suite, and
then writes the final report itself from rows it vetted. A candidate's
``os._exit(0)`` therefore only kills the child and leaves no marker, and a
forged report is overwritten by the real rows the child adds afterwards. The
public answer contract (the file is compiled and imported) still holds.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from dailybench import ModelReply, RunConfig, Transcript, Turn
from dailybench.tasks import coding_python


def _transcript(task, answer: str) -> Transcript:
    return Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[{"role": "user", "content": task.prompt}], reply=ModelReply(text=answer))],
    )


def _task():
    return coding_python.generate(random.Random(1), RunConfig())


def _forged_code(fake_path: Path) -> str:
    """No class at all: only a self-written perfect JSONL and a REPORT line."""
    rows = "\n".join(
        json.dumps({"test": f"TTLTests.{test_name}", "ok": True, "detail": ""})
        for _, test_name in coding_python.CHECK_ORDER
    )
    return f'''"""Forged module: pretends the hidden tests already passed."""

import json
from pathlib import Path

_FAKE = Path({str(fake_path)!r})
_FAKE.write_text({rows!r} + "\\n", encoding="utf-8")
print("REPORT", _FAKE)
raise ImportError("AsyncTTLCache is not implemented here")
'''


def test_a_correct_module_scores_full_points():
    task = _task()
    score = coding_python.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == score.possible == 8.0
    assert [check.name for check in score.checks] == [name for name, _ in coding_python.CHECK_ORDER]
    assert all(check.passed for check in score.checks)


def test_a_report_forger_without_the_class_does_not_score_full(tmp_path):
    task = _task()
    score = coding_python.evaluate(task, _transcript(task, _forged_code(tmp_path / "fake.jsonl")))
    assert score.earned == 0.0
    assert not any(check.passed for check in score.checks)
    assert any("ImportError" in check.detail for check in score.checks)


def test_garbage_scores_zero():
    task = _task()
    score = coding_python.evaluate(task, _transcript(task, "def broken(:\n    pass\n"))
    assert score.earned == 0.0


def test_the_candidate_stdout_is_only_a_journal(tmp_path):
    task = _task()
    journal = (
        f"print('REPORT', {str(tmp_path / 'forged.jsonl')!r})\n"
        "print('журнал: загружаю модуль')\n"
    )
    answer = task.oracle["reference"].replace("import time\n", "import time\n\n" + journal, 1)
    score = coding_python.evaluate(task, _transcript(task, answer))
    assert score.earned == 8.0


_FORGED_ROWS = "\n".join(
    json.dumps({"test": f"TTLTests.{test_name}", "ok": True, "detail": ""})
    for _, test_name in coding_python.CHECK_ORDER
)


def _forgery_preamble(exit: bool) -> str:
    """Overwrite the visible child report with a perfect one at import time.

    ``sys.argv[3]`` is the child's report path — the only verdict surface a
    candidate can reach. ``exit`` turns it into the strongest variant: the
    candidate stops the process itself before the child can write anything.
    """
    tail = "os._exit(0)" if exit else ""
    return f'''
import json
import os
import sys

_rows = {_FORGED_ROWS!r}
with open(sys.argv[3], "w", encoding="utf-8") as _fh:
    _fh.write(_rows + "\\n")
{tail}
'''


def test_a_candidate_that_writes_the_child_report_and_exits_does_not_score_full():
    """No class, forged rows, ``os._exit(0)``: the marker the child only writes
    after the suite is missing, so the supervisor fails every check loudly."""
    task = _task()
    score = coding_python.evaluate(task, _transcript(task, _forgery_preamble(exit=True)))
    assert score.earned < 8.0
    assert not any(check.passed for check in score.checks)
    assert any("маркера нет" in check.detail for check in score.checks)


def test_real_results_beat_a_forgery_when_the_candidate_stays_alive():
    """The candidate forges a perfect report but does not exit, and its
    implementation is broken: the rows the child appends afterwards win."""
    task = _task()
    broken = (
        _forgery_preamble(exit=False)
        + '''

class AsyncTTLCache:
    """Stores values but never serves them from the cache."""

    def __init__(self, ttl=1.0, clock=None, sleep=None):
        self.ttl = ttl
        self._entries = {}

    async def get_or_load(self, key, loader):
        value = await loader()
        self._entries[key] = value
        return value
'''
    )
    score = coding_python.evaluate(task, _transcript(task, broken))
    assert score.earned < 8.0
    assert any(not check.passed for check in score.checks)


_RUN_REPLACEMENT = '''
import unittest


def _fake(self, result=None):
    if result is None:
        result = self.defaultTestResult()
    result.addSuccess(self)


unittest.TestCase.run = _fake
unittest.IsolatedAsyncioTestCase.run = _fake


class AsyncTTLCache:
    """Always answers "WRONG", and claims every test succeeded."""

    def __init__(self, ttl, *, clock=None, sleep=None):
        self.ttl = ttl

    async def get_or_load(self, key, loader):
        return "WRONG"
'''


def test_a_candidate_that_replaces_TestCase_run_is_still_judged_on_the_methods():
    """The rebind is undone before every test, so the real assertions run."""
    task = _task()
    score = coding_python.evaluate(task, _transcript(task, _RUN_REPLACEMENT))
    assert score.earned < 8.0
    assert any(not check.passed for check in score.checks)
    assert not any(check.detail.startswith("integrity:") for check in score.checks)


_RECORDER_REAPPLIED = '''
"""Rebinds the recorder from inside the tested call, on every call."""

import sys


def _ok(self, test, err):
    self.addSuccess(test)


class AsyncTTLCache:
    def __init__(self, ttl, *, clock=None, sleep=None):
        self.ttl = ttl

    async def get_or_load(self, key, loader):
        sys.modules["hidden_ttl_tests"].Recorder.addFailure = _ok
        return "WRONG"
'''


def test_a_recorder_patch_reapplied_during_the_test_cannot_hide_failures():
    """Restoring before the test is not enough: the candidate patches again in
    the call under test, so the recorder methods are bound per instance."""
    task = _task()
    score = coding_python.evaluate(task, _transcript(task, _RECORDER_REAPPLIED))
    assert score.earned < 8.0
    assert any(not check.passed for check in score.checks)


def test_a_no_op_callTestMethod_is_restored_before_the_tests():
    """The counting hook lives on both case classes; a stub is undone, and the
    counter proves the methods still run (assertion text survives)."""
    task = _task()
    code = _RUN_REPLACEMENT + """
unittest.TestCase._callTestMethod = lambda self, method: None
unittest.IsolatedAsyncioTestCase._callTestMethod = lambda self, method: None
"""
    score = coding_python.evaluate(task, _transcript(task, code))
    failed = [check for check in score.checks if not check.passed]
    assert failed
    assert "WRONG" in failed[0].detail
    assert not any(check.detail.startswith("integrity:") for check in score.checks)


def test_duplicate_rows_are_last_wins():
    order = (("first_load", "test_first_load"), ("ttl_hit", "test_hit_within_ttl"))
    blob = "\n".join(
        [
            json.dumps({"test": "TTLTests.test_first_load", "ok": True, "detail": ""}),
            json.dumps({"test": "TTLTests.test_hit_within_ttl", "ok": True, "detail": ""}),
            json.dumps({"test": "TTLTests.test_first_load", "ok": False, "detail": "второй прогон"}),
        ]
    )
    mapped = coding_python._map_rows(blob, order)
    assert mapped["first_load"] == (False, "второй прогон")
    assert mapped["ttl_hit"] == (True, "")


def test_a_missing_row_fails_with_the_global_cause():
    blob = json.dumps({"test": "*", "ok": False, "detail": "ImportError: boom"})
    mapped = coding_python._map_rows(blob, (("first_load", "test_first_load"),))
    assert mapped["first_load"][0] is False
    assert "ImportError" in mapped["first_load"][1]


# --------------------------------------------------------------------------- #
# C-4 (pilot), C-5 (stretch) and C-6 (stretch): reference, mutants, garbage.
# --------------------------------------------------------------------------- #

import pytest

from dailybench import core, viz
from dailybench.tasks import coding, coding_go, coding_sql, coding_workflow

C4_POINTS = 8.0
C5_POINTS = 8.0
C6_POINTS = 4.0
DEV_SEEDS = (0, 1, 2, 3, 4)


def _replace(source: str, old: str, new: str) -> str:
    assert old in source, f"mutation anchor is gone: {old!r}"
    return source.replace(old, new)


def _c4_mutant(reference: str, kind: str) -> str:
    if kind == "late_result":
        return _replace(reference, "if self._generation.get(key) == generation:", "if True:")
    if kind == "dirty_inflight":
        return _replace(
            reference,
            "        self._inflight.pop(key, None)\n        self._generation[key]",
            "        self._generation[key]",
        )
    if kind == "caller_owned_loader":
        return _replace(
            reference,
            "        task = loop.create_task(loader())\n"
            "        task.add_done_callback(lambda done: self._finish(key, generation, future, done))\n"
            "        return await asyncio.shield(future)",
            "        try:\n"
            "            value = await loader()\n"
            "        except BaseException:\n"
            "            self._inflight.pop(key, None)\n"
            "            raise\n"
            "        self._inflight.pop(key, None)\n"
            "        self._entries[key] = (value, self._clock() + self.ttl)\n"
            "        return value",
        )
    if kind == "global_lock":
        locked = _replace(
            reference,
            "        self._generation: dict[object, int] = {}",
            "        self._generation: dict[object, int] = {}\n        self._lock = asyncio.Lock()",
        )
        return _replace(
            locked,
            "    async def get_or_load(self, key, loader):\n        entry = self._entries.get(key)",
            "    async def get_or_load(self, key, loader):\n"
            "        async with self._lock:\n"
            "            return await self._load_locked(key, loader)\n\n"
            "    async def _load_locked(self, key, loader):\n"
            "        entry = self._entries.get(key)",
        )
    raise AssertionError(kind)


def _c5_mutant(reference: str, kind: str) -> str:
    if kind == "drop_queued":
        return _replace(
            reference,
            "\t\tcase <-p.ctx.Done():\n"
            "\t\t\t// Accepted Jobs are already in the queue (or inside a blocked\n"
            "\t\t\t// Submit): drain them before leaving.\n"
            "\t\t\tfor job := range p.jobs {\n"
            "\t\t\t\tp.handle(job)\n"
            "\t\t\t}\n"
            "\t\t\treturn",
            "\t\tcase <-p.ctx.Done():\n\t\t\treturn",
        )
    if kind == "background_context":
        return _replace(reference, "err = job.Run(p.ctx)", "err = job.Run(context.Background())")
    raise AssertionError(kind)


def _c6_mutant(reference: str, kind: str) -> str:
    if kind == "no_cutoff":
        return _replace(reference, "WHERE recorded_at <= '2026-05-31 23:59:59'", "WHERE 1 = 1")
    if kind == "no_dedupe":
        return _replace(
            reference, "GROUP BY order_id, revision, kind", "GROUP BY event_id, order_id, revision, kind"
        )
    if kind == "first_revision":
        return _replace(reference, "ORDER BY revision DESC", "ORDER BY revision ASC")
    if kind == "no_tiebreak":
        return _replace(reference, "ORDER BY net_cents DESC, c.category ASC", "ORDER BY net_cents DESC")
    raise AssertionError(kind)


def _score(task, answer):
    return coding.evaluate(task, _transcript(task, answer))


def test_c4_reference_scores_full():
    task = coding_python.generate_c4(random.Random(1), RunConfig(profile="pilot"))
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == C4_POINTS
    assert [check.name for check in score.checks] == [name for name, _ in coding_python.CHECK_ORDER_C4]


@pytest.mark.parametrize("kind", ["late_result", "dirty_inflight", "caller_owned_loader", "global_lock"])
def test_c4_mutant_does_not_score_full(kind):
    task = coding_python.generate_c4(random.Random(1), RunConfig(profile="pilot"))
    score = _score(task, _c4_mutant(task.oracle["reference"], kind))
    assert score.earned < C4_POINTS
    assert any(not check.passed for check in score.checks)


def test_c4_garbage_scores_zero():
    task = coding_python.generate_c4(random.Random(1), RunConfig(profile="pilot"))
    assert _score(task, "def broken(:\n    pass\n").earned == 0.0


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_c4_reference_full_on_every_dev_seed(seed):
    task = coding_python.generate_c4(random.Random(seed), RunConfig(profile="pilot"))
    assert task.id == "C-4" and task.max_points == C4_POINTS
    assert len(task.oracle["checks"]) * coding_python._POINTS_PER_CHECK == C4_POINTS
    assert task.prompt.count("invalidate") >= 3
    assert task.oracle["reference"] not in task.prompt
    assert _score(task, task.oracle["reference"]).earned == C4_POINTS


requires_go = pytest.mark.skipif(
    coding_go._go_binary() is None, reason="go toolchain unavailable"
)
requires_docker = pytest.mark.skipif(
    not coding_sql.PostgresBackend().available(), reason="docker unavailable"
)


@requires_go
def test_c5_reference_scores_full():
    task = coding_go.generate_c5(random.Random(1), RunConfig(profile="stretch-v1"))
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == C5_POINTS
    assert [check.name for check in score.checks] == [name for name, _ in coding_go.CHECK_ORDER_C5]


@requires_go
@pytest.mark.parametrize("kind", ["drop_queued", "background_context"])
def test_c5_mutant_does_not_score_full(kind):
    task = coding_go.generate_c5(random.Random(1), RunConfig(profile="stretch-v1"))
    score = _score(task, _c5_mutant(task.oracle["reference"], kind))
    assert score.earned < C5_POINTS


def _forged_go(reference: str) -> str:
    """``init`` prints a perfect PASS log, then panics before any test runs."""
    lines = "".join(f'\tprintln("--- PASS: {test} (0.00s)")\n' for _, test in coding_go.CHECK_ORDER)
    init = "func init() {\n" + lines + '\tpanic("forged pass")\n}\n'
    return reference.replace("package pool\n", "package pool\n\n" + init, 1)


@requires_go
def test_c5_forged_pass_lines_from_init_do_not_survive_a_panic():
    task = coding_go.generate_c5(random.Random(1), RunConfig(profile="stretch-v1"))
    score = _score(task, _forged_go(task.oracle["reference"]))
    assert score.earned == 0.0
    assert all("кодом" in check.detail for check in score.checks)


_GO_FORGED_INIT = '''
import (
\t"fmt"
\t"os"
)

func init() {
\tfor _, name := range []string{%s} {
\t\tfmt.Printf("{\\\"Action\\\":\\\"run\\\",\\\"Package\\\":\\\"dailybench/pool\\\",\\\"Test\\\":\\\"%%s\\\"}\\n", name)
\t\tfmt.Printf("{\\\"Action\\\":\\\"pass\\\",\\\"Package\\\":\\\"dailybench/pool\\\",\\\"Test\\\":\\\"%%s\\\"}\\n", name)
\t}
\tos.Exit(0)
}
'''


def _go_forged_stub(task) -> str:
    """The panicking stub plus an ``init`` that prints a perfect test2json log
    and exits the process with status 0 before a single test ran."""
    names = ", ".join(f'"{test}"' for _, test in coding_go.CHECK_ORDER)
    stub = task.workspace[coding_go.TARGET_FILE]
    return stub.replace('import "context"\n', 'import "context"\n' + _GO_FORGED_INIT % names, 1)


@requires_go
def test_c2_forged_test2json_from_init_does_not_score_full():
    task = coding_go.generate(random.Random(1), RunConfig())
    score = _score(task, _go_forged_stub(task))
    assert score.earned < 8.0
    assert not any(check.passed for check in score.checks)


# --------------------------------------------------------------------------- #
# C-2: the no-hang check must observe pool goroutines, not an unrelated timer.
# --------------------------------------------------------------------------- #

C2_NO_HANG = "no_hang_after_consumer_stops"


def _c2_mutant(reference: str, kind: str) -> str:
    """Break the cancellation paths a stopped consumer depends on."""
    if kind == "blocking_send":
        # The worker delivers without watching ctx: once nobody reads the
        # output it is parked forever and its goroutine never ends.
        return _replace(
            reference,
            "\t\t\t\t\terr := job.Run(ctx)\n"
            "\t\t\t\t\tselect {\n"
            "\t\t\t\t\tcase out <- Result{ID: job.ID, Err: err}:\n"
            "\t\t\t\t\tcase <-ctx.Done():\n"
            "\t\t\t\t\t\treturn\n"
            "\t\t\t\t\t}",
            "\t\t\t\t\terr := job.Run(ctx)\n"
            "\t\t\t\t\tout <- Result{ID: job.ID, Err: err}",
        )
    if kind == "ignore_cancel":
        # Workers only learn about cancellation from the receive select; with
        # the producer parked on a send they wait for good.
        return _replace(
            reference,
            "\t\t\tfor {\n"
            "\t\t\t\tif ctx.Err() != nil {\n"
            "\t\t\t\t\treturn\n"
            "\t\t\t\t}\n"
            "\t\t\t\tselect {\n"
            "\t\t\t\tcase <-ctx.Done():\n"
            "\t\t\t\t\treturn\n"
            "\t\t\t\tcase job, ok := <-jobs:",
            "\t\t\tfor {\n"
            "\t\t\t\tjob, ok := <-jobs:",
        )
    raise AssertionError(kind)


@requires_go
def test_go_c2_reference_scores_full():
    task = coding_go.generate(random.Random(1), RunConfig())
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == 8.0


@requires_go
@pytest.mark.parametrize("kind", ["blocking_send", "ignore_cancel"])
def test_go_c2_mutant_does_not_score_full_and_fails_the_no_hang_check(kind):
    """Both mutants only mis-handle "the consumer stopped reading after the
    cancellation": the timer-only assertion used to let them through."""
    task = coding_go.generate(random.Random(1), RunConfig())
    score = _score(task, _c2_mutant(task.oracle["reference"], kind))
    assert score.earned < 8.0
    failed = {check.name for check in score.checks if not check.passed}
    assert C2_NO_HANG in failed


def test_go_c2_declares_the_no_hang_check():
    """The observation lives in a check of its own: C-2 orders it by name."""
    assert C2_NO_HANG in {name for name, _ in coding_go.CHECK_ORDER}


# --------------------------------------------------------------------------- #
# R03: the evaluation must not depend on the host's Go environment or network.
# --------------------------------------------------------------------------- #

HOSTILE_GO_ENV = {
    "GOFLAGS": "-mod=vendor",  # no vendor directory is shipped
    "GOTOOLCHAIN": "go1.99.99",  # only resolvable by downloading
    "GOPROXY": "off",
    "GOWORK": "/nonexistent/go.work",
    "GO111MODULE": "off",
}


def test_go_env_overwrites_every_host_key(tmp_path, monkeypatch):
    for key, value in HOSTILE_GO_ENV.items():
        monkeypatch.setenv(key, value)
    env = coding_go._go_env(tmp_path)
    assert env["GOTOOLCHAIN"] == "local"
    assert env["GOFLAGS"] == "-mod=mod"
    assert env["GOPROXY"] == "off"
    assert env["GOWORK"] == "off"
    assert env["GO111MODULE"] == "on"
    assert env["CGO_ENABLED"] == "0"
    for key in ("GOCACHE", "GOMODCACHE", "GOPATH"):
        assert env[key] != ""
        assert Path(env[key]).is_dir()


def test_go_build_cache_is_shared_and_module_cache_stays_per_repository(tmp_path, monkeypatch):
    """A host GOCACHE must not be reused, but grading shares one build cache.

    Break this test by putting GOCACHE back beside each repository, or by
    letting GOMODCACHE leak into the host or into the other repository.
    """
    monkeypatch.delenv("DAILYBENCH_GOCACHE", raising=False)
    monkeypatch.setenv("GOCACHE", str(tmp_path / "host-build"))
    monkeypatch.setenv("GOMODCACHE", str(tmp_path / "host-mod"))
    (tmp_path / "host-build").mkdir()
    (tmp_path / "host-mod").mkdir()
    first = coding_go._go_env(tmp_path / "one")
    second = coding_go._go_env(tmp_path / "two")
    assert first["GOCACHE"] != str(tmp_path / "host-build")
    assert first["GOCACHE"] == second["GOCACHE"]
    assert "dailybench" in Path(first["GOCACHE"]).as_posix()
    assert first["GOMODCACHE"] != str(tmp_path / "host-mod")
    assert first["GOMODCACHE"] != second["GOMODCACHE"]


def _go_version_text() -> str:
    go = coding_go._go_binary()
    assert go, "go toolchain unavailable"
    proc = subprocess.run(
        [go, "version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=True,
    )
    return proc.stdout.strip()


def _expected_user_gocache(version: str) -> Path:
    """The contract path, derived here and not from the implementation."""
    digest = hashlib.sha256(version.encode("utf-8")).hexdigest()[:16]
    if os.name == "nt":
        root = Path(os.environ["LOCALAPPDATA"])
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        raw = os.environ.get("XDG_CACHE_HOME")
        root = Path(raw) if raw else Path.home() / ".cache"
    return root / "dailybench" / "go-build" / digest


@requires_go
def test_gocache_is_the_versioned_user_cache(tmp_path, monkeypatch):
    """Break this by caching beside the repo, or by hashing anything other
    than the stripped ``go version`` line."""
    monkeypatch.delenv("DAILYBENCH_GOCACHE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("GOCACHE", str(tmp_path / "host-build"))
    expected = _expected_user_gocache(_go_version_text())
    env = coding_go._go_env(tmp_path / "repo")
    assert Path(env["GOCACHE"]) == expected
    assert expected.is_dir()
    assert env["GOPROXY"] == "off"
    assert env["GOTOOLCHAIN"] == "local"
    assert env["GOFLAGS"] == "-mod=mod"


def test_dailybench_gocache_overrides_the_versioned_cache(tmp_path, monkeypatch):
    """Break this by ignoring ``DAILYBENCH_GOCACHE`` or by appending the hash."""
    pinned = tmp_path / "pinned-build"
    monkeypatch.setenv("DAILYBENCH_GOCACHE", str(pinned))
    monkeypatch.setenv("GOCACHE", str(tmp_path / "host-build"))
    env = coding_go._go_env(tmp_path / "repo")
    assert Path(env["GOCACHE"]) == pinned
    assert pinned.is_dir()
    assert env["GOPROXY"] == "off"
    assert env["GOTOOLCHAIN"] == "local"


def test_go_env_uses_a_shipped_vendor_tree(tmp_path):
    (tmp_path / "vendor").mkdir()
    assert coding_go._go_env(tmp_path)["GOFLAGS"] == "-mod=vendor"


@requires_go
@pytest.mark.parametrize("builder", [coding_go.generate, coding_go.generate_c5])
def test_go_evaluation_is_hermetic_against_the_host_environment(builder, monkeypatch):
    """With a hostile host env and the network cut off, the reference still
    scores full: the contract is decided by the evaluator, not by the box."""
    for key, value in HOSTILE_GO_ENV.items():
        monkeypatch.setenv(key, value)
    task = builder(random.Random(1), RunConfig(profile="stretch-v1"))
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == 8.0


def _c2_per_job(reference: str) -> str:
    """One goroutine per Job, still closing the output after they finish."""
    return _replace(
        reference,
        "\tvar wg sync.WaitGroup\n"
        "\twg.Add(workers)\n"
        "\tfor i := 0; i < workers; i++ {\n"
        "\t\tgo func() {\n"
        "\t\t\tdefer wg.Done()\n"
        "\t\t\tfor {\n"
        "\t\t\t\tif ctx.Err() != nil {\n"
        "\t\t\t\t\treturn\n"
        "\t\t\t\t}\n"
        "\t\t\t\tselect {\n"
        "\t\t\t\tcase <-ctx.Done():\n"
        "\t\t\t\t\treturn\n"
        "\t\t\t\tcase job, ok := <-jobs:\n"
        "\t\t\t\t\tif !ok {\n"
        "\t\t\t\t\t\treturn\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t\terr := job.Run(ctx)\n"
        "\t\t\t\t\tselect {\n"
        "\t\t\t\t\tcase out <- Result{ID: job.ID, Err: err}:\n"
        "\t\t\t\t\tcase <-ctx.Done():\n"
        "\t\t\t\t\t\treturn\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t}\n"
        "\t\t\t}\n"
        "\t\t}()\n"
        "\t}\n"
        "\n"
        "\tgo func() {\n"
        "\t\twg.Wait()\n"
        "\t\tclose(out)\n"
        "\t}()\n",
        "\tvar wg sync.WaitGroup\n"
        "\tgo func() {\n"
        "\t\tdefer func() {\n"
        "\t\t\twg.Wait()\n"
        "\t\t\tclose(out)\n"
        "\t\t}()\n"
        "\t\tfor {\n"
        "\t\t\tif ctx.Err() != nil {\n"
        "\t\t\t\treturn\n"
        "\t\t\t}\n"
        "\t\t\tselect {\n"
        "\t\t\tcase <-ctx.Done():\n"
        "\t\t\t\treturn\n"
        "\t\t\tcase job, ok := <-jobs:\n"
        "\t\t\t\tif !ok {\n"
        "\t\t\t\t\treturn\n"
        "\t\t\t\t}\n"
        "\t\t\t\twg.Add(1)\n"
        "\t\t\t\tgo func(job Job) {\n"
        "\t\t\t\t\tdefer wg.Done()\n"
        "\t\t\t\t\terr := job.Run(ctx)\n"
        "\t\t\t\t\tselect {\n"
        "\t\t\t\t\tcase out <- Result{ID: job.ID, Err: err}:\n"
        "\t\t\t\t\tcase <-ctx.Done():\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t}(job)\n"
        "\t\t\t}\n"
        "\t\t}\n"
        "\t}()\n",
    )


def _c2_leak(reference: str) -> str:
    """Workers finish, but the pool never closes its output."""
    return _replace(
        reference,
        "\tgo func() {\n\t\twg.Wait()\n\t\tclose(out)\n\t}()\n",
        "\tgo func() {\n\t\twg.Wait()\n\t\tselect {}\n\t}()\n",
    )


def _c2_early_close(reference: str) -> str:
    """Closes the output, but leaves a goroutine stuck after that close."""
    return _replace(
        reference,
        "\tgo func() {\n\t\twg.Wait()\n\t\tclose(out)\n\t}()\n",
        "\tgo func() { select {} }()\n"
        "\tgo func() {\n\t\twg.Wait()\n\t\tclose(out)\n\t}()\n",
    )


def _c5_early_close(reference: str) -> str:
    """Closes Results after the workers, but leaves a goroutine stuck."""
    return _replace(
        reference,
        "\tgo p.supervise()\n",
        "\tgo func() { select {} }()\n\tgo p.supervise()\n",
    )


def _c5_per_job(reference: str) -> str:
    return _replace(reference, "p.handle(job)", "go p.handle(job)")


def _c5_leak(reference: str) -> str:
    """Stop is acknowledged; Results is never closed."""
    return _replace(
        reference,
        "\tclose(p.ack)\n\tp.wg.Wait()\n\tclose(p.results)\n",
        "\tclose(p.ack)\n\tselect {}\n",
    )


def _stage_hidden_go(tmp_path: Path, task, source: str, hidden_name: str) -> Path:
    repo = tmp_path / "repo"
    for rel, content in task.workspace.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo / task.oracle["target_file"]).write_text(source, encoding="utf-8")
    (repo / "pool" / hidden_name).write_text(task.oracle["hidden_tests"], encoding="utf-8")
    return repo


def _run_hidden_go(repo: Path, *, count: int, go_timeout: str, failfast: bool) -> subprocess.CompletedProcess[str]:
    go = coding_go._go_binary()
    assert go
    env = coding_go._go_env(repo)
    env["GOMAXPROCS"] = "1"
    cmd = [go, "test", "-json", f"-count={count}", "-timeout", go_timeout]
    if failfast:
        cmd.append("-failfast")
    cmd.append("./...")
    return subprocess.run(
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=240,
    )


@requires_go
@pytest.mark.parametrize(
    ("scenario", "kind"),
    [
        ("c2", "oracle"),
        ("c2", "per_job"),
        ("c2", "leak"),
        ("c2", "early_close"),
        ("c5", "oracle"),
        ("c5", "per_job"),
        ("c5", "leak"),
        ("c5", "early_close"),
    ],
)
def test_hidden_go_checks_hold_under_gomaxprocs_1_and_count_20(tmp_path, monkeypatch, scenario, kind):
    """The hidden suite must pass the oracle and reject a per-job pool and a
    pool that never signals completion, even with one OS thread and 20 repeats.

    Break this by bringing back a sleep/NumGoroutine bound (it flakes or lets
    the per-job mutant through under GOMAXPROCS=1) or by treating a missing
    completion signal as success.
    """
    monkeypatch.delenv("DAILYBENCH_GOCACHE", raising=False)
    monkeypatch.setenv("GOMAXPROCS", "1")
    if scenario == "c2":
        task = coding_go.generate(random.Random(1), RunConfig())
        hidden = "process_hidden_test.go"
        order = coding_go.CHECK_ORDER
        source = {
            "oracle": task.oracle["reference"],
            "per_job": _c2_per_job(task.oracle["reference"]),
            "leak": _c2_leak(task.oracle["reference"]),
            "early_close": _c2_early_close(task.oracle["reference"]),
        }[kind]
        bound = "no_goroutine_per_job"
        leaked = "no_leak"
    else:
        task = coding_go.generate_c5(random.Random(1), RunConfig(profile="stretch-v1"))
        hidden = "pool_hidden_test.go"
        order = coding_go.CHECK_ORDER_C5
        source = {
            "oracle": task.oracle["reference"],
            "per_job": _c5_per_job(task.oracle["reference"]),
            "leak": _c5_leak(task.oracle["reference"]),
            "early_close": _c5_early_close(task.oracle["reference"]),
        }[kind]
        bound = "concurrency_limit"
        leaked = "no_goroutine_leak"
    repo = _stage_hidden_go(tmp_path, task, source, hidden)
    # Mutants are expected to fail; -failfast keeps a hang from repeating.
    proc = _run_hidden_go(
        repo,
        count=20,
        go_timeout="120s" if kind == "oracle" else "20s",
        failfast=kind != "oracle",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert "syntax error" not in output, output[-500:]
    results = coding_go._verdicts(output, order, proc.returncode)
    if kind == "oracle":
        failed = {name: detail for name, (ok, detail) in results.items() if not ok}
        assert proc.returncode == 0, output[-800:]
        assert not failed, failed
    elif kind == "per_job":
        assert results[bound][0] is False, results[bound]
    elif kind == "early_close":
        assert results[leaked][0] is False, results[leaked]
        assert "goroutine leak" in results[leaked][1], results[leaked]
    else:
        assert results[leaked][0] is False, results[leaked]


def test_a_go_pass_event_needs_a_preceding_run_event():
    order = (("check", "TestCheck"),)
    only_pass = json.dumps({"Action": "pass", "Package": "p", "Test": "TestCheck"})
    results = coding_go._verdicts(only_pass, order, 0)
    assert results["check"][0] is False
    assert "run" in results["check"][1]


def test_a_forged_stream_inside_an_output_event_creates_nothing():
    forged = "\n".join(
        [
            json.dumps({"Action": "run", "Test": "TestCheck"}),
            json.dumps({"Action": "pass", "Test": "TestCheck"}),
            json.dumps({"Action": "pass", "Package": "p"}),
        ]
    )
    output = json.dumps({"Action": "output", "Test": "TestCheck", "Output": forged})
    results = coding_go._verdicts(output, (("check", "TestCheck"),), 0)
    assert results["check"][0] is False


def test_a_stream_without_a_package_pass_is_not_a_completed_run():
    order = (("check", "TestCheck"),)
    output = "\n".join(
        json.dumps(event)
        for event in (
            {"Action": "run", "Test": "TestCheck"},
            {"Action": "pass", "Test": "TestCheck"},
        )
    )
    results = coding_go._verdicts(output, order, 0)
    assert results["check"][0] is False
    assert "пакет" in results["check"][1]


@requires_go
def test_go_skip_is_not_a_pass(tmp_path):
    (tmp_path / "go.mod").write_text("module x\n\ngo 1.21\n", encoding="utf-8")
    (tmp_path / "x_test.go").write_text(
        'package x\n\nimport "testing"\n\n'
        'func TestSkipMe(t *testing.T) { t.Skip("nope") }\n\n'
        "func TestPassMe(t *testing.T) {}\n",
        encoding="utf-8",
    )
    order = (("skip", "TestSkipMe"), ("pass", "TestPassMe"))
    results, _ = coding_go._run_go_tests(tmp_path, timeout=120, order=order)
    assert results["skip"] == (False, "skipped")
    assert results["pass"] == (True, "ok")


@requires_go
def test_c5_garbage_scores_zero():
    task = coding_go.generate_c5(random.Random(1), RunConfig(profile="stretch-v1"))
    assert _score(task, "package pool\n\nfunc broken(\n").earned == 0.0


@requires_go
def test_c5_fixture_is_identical_on_every_dev_seed():
    first = coding_go.generate_c5(random.Random(DEV_SEEDS[0]), RunConfig(profile="stretch-v1"))
    assert len(first.oracle["checks"]) * coding_go._POINTS_PER_CHECK == C5_POINTS
    for seed in DEV_SEEDS[1:]:
        task = coding_go.generate_c5(random.Random(seed), RunConfig(profile="stretch-v1"))
        assert task.prompt == first.prompt
        assert task.workspace == first.workspace
        assert task.oracle == first.oracle


def test_c6_reference_scores_full():
    task = coding_sql.generate_c6(random.Random(1), RunConfig(profile="stretch-v1"))
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == C6_POINTS


@pytest.mark.parametrize("kind", ["no_cutoff", "no_dedupe", "first_revision", "no_tiebreak"])
def test_c6_mutant_does_not_score_full(kind):
    task = coding_sql.generate_c6(random.Random(1), RunConfig(profile="stretch-v1"))
    score = _score(task, _c6_mutant(task.oracle["reference"], kind))
    assert score.earned < C6_POINTS


def test_c6_garbage_scores_zero():
    task = coding_sql.generate_c6(random.Random(1), RunConfig(profile="stretch-v1"))
    assert _score(task, "SELECT * FROM;").earned == 0.0


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_c6_oracle_is_valid_and_bounded_on_every_dev_seed(seed):
    task = coding_sql.generate_c6(random.Random(seed), RunConfig(profile="stretch-v1"))
    assert task.max_points == C6_POINTS
    assert len(task.oracle["databases"]) == coding_sql.C6_DATABASES
    for database, expected in zip(task.oracle["databases"], task.oracle["expected"]):
        assert 10 <= len(database["orders"]) <= 40
        event_counts: dict[int, set[int]] = {}
        for _, order_id, revision, _, _, _ in database["events"]:
            event_counts.setdefault(order_id, set()).add(revision)
        assert all(len(revisions) <= 3 for revisions in event_counts.values())
        assert 1 <= len(expected) <= 2
        nets = [row[2] for row in expected]
        assert nets == sorted(nets, reverse=True)
        assert any(row[3] >= 1 for row in expected)
    assert _score(task, task.oracle["reference"]).earned == C6_POINTS


def test_c6_top_two_are_tied_and_break_alphabetically():
    task = coding_sql.generate_c6(random.Random(2), RunConfig(profile="stretch-v1"))
    for expected in task.oracle["expected"]:
        assert len(expected) == 2
        assert expected[0][2] == expected[1][2]
        assert expected[0][0] < expected[1][0]


# --------------------------------------------------------------------------- #
# Router profiles
# --------------------------------------------------------------------------- #


def test_baseline_profile_keeps_the_three_old_scenarios():
    tasks = coding.generate(random.Random(1), RunConfig())
    assert [task.id for task in tasks] == list(coding.BASELINE_IDS)
    assert sum(task.max_points for task in tasks) == 20.0


def test_pilot_profile_generates_only_the_pilot_scenario():
    tasks = coding.generate(random.Random(1), RunConfig(profile="pilot"))
    assert [task.id for task in tasks] == list(coding.PILOT_IDS)


def test_stretch_profile_generates_baseline_plus_new_scenarios():
    tasks = coding.generate(random.Random(1), RunConfig(profile="stretch-v1"))
    assert [task.id for task in tasks] == list(coding.STRETCH_IDS)
    assert set(coding.BASELINE_IDS) <= {task.id for task in tasks}


def test_registration_declares_the_profile_contract():
    category = core.REGISTRY["C"]
    assert category.expected_ids == coding.BASELINE_IDS
    assert category.expected_ids_for("pilot") == coding.PILOT_IDS
    assert category.expected_ids_for("stretch-v1") == coding.STRETCH_IDS
    assert category.expected_ids_for("baseline-20261020") == coding.BASELINE_IDS


def test_new_scenarios_keep_their_declared_budgets_and_answer_files():
    for builder, points in (
        (coding_python.generate_c4, C4_POINTS),
        (coding_go.generate_c5, C5_POINTS),
        (coding_sql.generate_c6, C6_POINTS),
    ):
        task = builder(random.Random(0), RunConfig(profile="stretch-v1"))
        assert task.max_points == points
        assert task.oracle["answer_file"] == task.oracle["target_file"]
        assert task.oracle["answer_file"] in task.workspace
        assert task.oracle["reference"] not in task.prompt


# --------------------------------------------------------------------------- #
# R01: the PostgreSQL backend must create every database the checks name, and
# a lost connection is an evaluator failure rather than a model zero.
# --------------------------------------------------------------------------- #

_FAKE_DOCKER = '''#!{python}
"""Stand-in for the docker CLI: journals psql traffic, replays fixed rows."""

import json
import os
import sys

args = sys.argv[1:]
command = args[0] if args else ""
log = os.environ["FAKE_DOCKER_LOG"]


def journal(entry):
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\\n")


if command == "info":
    print("16.0")
elif command == "run":
    print("fake-pg")
elif command == "rm":
    pass
elif command == "exec":
    sql = sys.stdin.read()
    database = args[args.index("-d") + 1] if "-d" in args else ""
    journal({"database": database, "sql": sql})
    failing = [name for name in os.environ.get("FAKE_DOCKER_FAIL_LOAD", "").split(",") if name]
    is_query = sql.lstrip().upper().startswith(("SELECT", "WITH"))
    if database in failing and not is_query:
        sys.stderr.write('connection to server failed: database "%s" not rebooted\\n' % database)
        sys.exit(2)
    if is_query:
        for row in json.loads(os.environ["FAKE_DOCKER_ROWS"]).get(database, []):
            print("\\t".join(str(value) for value in row))
else:
    sys.exit(1)
'''


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    """Install a fake ``docker`` on PATH and answer queries from the oracle."""

    def install(task, fail_load: str = "") -> Path:
        binaries = tmp_path / "bin"
        binaries.mkdir(exist_ok=True)
        script = binaries / "docker"
        script.write_text(_FAKE_DOCKER.replace("{python}", sys.executable), encoding="utf-8")
        script.chmod(0o755)
        log = tmp_path / "docker.log"
        log.write_text("", encoding="utf-8")
        rows = {f"bench{index + 1}": rows for index, rows in enumerate(task.oracle["expected"])}
        monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
        monkeypatch.setenv("FAKE_DOCKER_ROWS", json.dumps(rows))
        monkeypatch.setenv("FAKE_DOCKER_FAIL_LOAD", fail_load)
        return log

    return install


def _journal(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _created(journal: list[dict]) -> list[str]:
    created = []
    for entry in journal:
        match = re.match(r'\s*CREATE DATABASE "?(bench\d+)"?;', entry["sql"], re.IGNORECASE)
        if match:
            created.append(match.group(1))
    return created


@requires_docker
@pytest.mark.parametrize(
    "builder, points",
    [(coding_sql.generate, 4.0), (coding_sql.generate_c6, 4.0)],
    ids=["c3", "c6"],
)
def test_postgres_creates_every_database_before_loading_it(fake_docker, builder, points):
    task = builder(random.Random(1), RunConfig(sql_backend="postgres"))
    log = fake_docker(task)
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == points

    journal = _journal(log)
    expected_names = [f"bench{index + 1}" for index in range(len(task.oracle["checks"]))]
    assert _created(journal) == expected_names
    for name in expected_names:
        created_at = next(i for i, entry in enumerate(journal) if _match_create(entry, name))
        first_use = next(i for i, entry in enumerate(journal) if entry["database"] == name)
        assert created_at < first_use


def _match_create(entry: dict, name: str) -> bool:
    return bool(re.match(rf'\s*CREATE DATABASE "?{name}"?;', entry["sql"], re.IGNORECASE))


@requires_docker
@pytest.mark.parametrize(
    "builder",
    [coding_sql.generate, coding_sql.generate_c6],
    ids=["c3", "c6"],
)
def test_postgres_connection_loss_is_evaluator_unavailable(fake_docker, builder):
    task = builder(random.Random(1), RunConfig(sql_backend="postgres"))
    fake_docker(task, fail_load="bench3")
    score = _score(task, task.oracle["reference"])

    assert score.status == "error"
    assert "evaluator" in (score.error or "").lower()
    assert viz.classify(score.as_dict()) == "evaluator_unavailable"
    assert [check.passed for check in score.checks[:2]] == [True, True]
    assert not any(check.passed for check in score.checks[2:])


@requires_docker
@pytest.mark.parametrize(
    "builder",
    [coding_sql.generate, coding_sql.generate_c6],
    ids=["c3", "c6"],
)
def test_postgres_reference_scores_full_on_a_real_server(builder):
    task = builder(random.Random(1), RunConfig(sql_backend="postgres"))
    score = _score(task, task.oracle["reference"])
    assert score.error is None
    assert score.earned == score.possible == 4.0


# --------------------------------------------------------------------------- #
# N-2 (stretch): terminal/data workflow — CSV import, migration, report,
# protected files, declared exit codes and atomicity on an invalid batch.
# --------------------------------------------------------------------------- #

N2_POINTS = 1.0


def _n2(seed: int = 1):
    return coding_workflow.generate(random.Random(seed), RunConfig(profile="stretch-v1"))


def _replace_block(source: str, start: str, end: str, new: str) -> str:
    head = source.index(start)
    tail = source.index(end, head)
    return source[:head] + new + source[tail:]


_REPORT_WITHOUT_DB = '''def command_import(path: str) -> int:
    if not Path(path).is_file():
        return fail(f"invalid batch: missing {path}")
    rows, bad, header_ok = parse_batch(path)
    if bad:
        if not header_ok:
            return fail("invalid batch: 1")
        return fail("invalid batch: " + ",".join(str(number) for number in bad))
    command_report(path)
    print(f"ok: rows={len(rows)}")
    return 0


def command_report(path: str) -> int:
    rows, bad, _header = parse_batch(path)
    if bad:
        return fail("report: invalid batch")
    name = batch_name(path)
    by_region: dict[str, list[int]] = {}
    by_product: dict[str, int] = {}
    for _order_id, region, product, cents, _status in rows:
        bucket = by_region.setdefault(region, [0, 0])
        bucket[0] += 1
        bucket[1] += cents
        by_product[product] = by_product.get(product, 0) + cents
    payload = {
        "batch": name,
        "rows": len(rows),
        "total_cents": sum(row[3] for row in rows),
        "by_region": [
            {"region": region, "rows": by_region[region][0], "cents": by_region[region][1]}
            for region in sorted(by_region)
        ],
        "top_product": min(by_product, key=lambda product: (-by_product[product], product)),
    }
    REPORTS.mkdir(exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\\n", encoding="utf-8")
    return 0


'''

_PARTIAL_INVALID = '''def command_import(path: str) -> int:
    if not Path(path).is_file():
        return fail(f"invalid batch: missing {path}")
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != HEADER:
        return fail("invalid batch: 1")
    regions = read_regions()
    BUILD.mkdir(exist_ok=True)
    connection = connect(DB)
    with connection:
        migrate(connection)
    name = batch_name(path)
    seen: set[int] = set()
    bad: list[int] = []
    count = 0
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        if len(fields) != 5:
            bad.append(number)
            continue
        raw_id, region, product, raw_cents, status = fields
        order_id = int(raw_id) if re.fullmatch(r"[0-9]+", raw_id) else 0
        cents = int(raw_cents) if re.fullmatch(r"[0-9]+", raw_cents) else 0
        if (
            order_id <= 0
            or region not in regions
            or not product
            or cents <= 0
            or status not in STATUSES
            or order_id in seen
        ):
            bad.append(number)
            continue
        seen.add(order_id)
        connection.execute(
            "INSERT INTO orders (order_id, region, product, cents, status, batch)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(order_id) DO UPDATE SET"
            " region = excluded.region, product = excluded.product,"
            " cents = excluded.cents, status = excluded.status, batch = excluded.batch",
            (order_id, region, product, cents, status, name),
        )
        count += 1
    connection.commit()
    connection.close()
    if bad:
        return fail("invalid batch: " + ",".join(str(number) for number in bad))
    print(f"ok: rows={count}")
    return 0


'''


def _n2_mutant(reference: str, kind: str) -> str:
    if kind == "report_without_db":
        return _replace_block(
            reference,
            "def command_import(path: str) -> int:",
            "def main(argv: list[str]) -> int:",
            _REPORT_WITHOUT_DB,
        )
    if kind == "double_import":
        return _replace(
            reference,
            " cents = excluded.cents, status = excluded.status, batch = excluded.batch",
            " cents = orders.cents + excluded.cents, status = excluded.status, batch = excluded.batch",
        )
    if kind == "partial_invalid_batch":
        return _replace_block(
            reference,
            "def command_import(path: str) -> int:",
            "def command_report(path: str) -> int:",
            _PARTIAL_INVALID,
        )
    raise AssertionError(kind)


def test_n2_reference_scores_full():
    task = _n2()
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == N2_POINTS
    assert [check.name for check in score.checks] == task.oracle["checks"]


@pytest.mark.parametrize(
    ("kind", "failed"),
    [
        ("report_without_db", "db_rows"),
        ("double_import", "idempotent_reimport"),
        ("partial_invalid_batch", "invalid_atomicity"),
    ],
)
def test_n2_mutant_does_not_score_full(kind, failed):
    task = _n2()
    score = _score(task, _n2_mutant(task.oracle["reference"], kind))
    assert score.earned < N2_POINTS
    assert not next(check for check in score.checks if check.name == failed).passed


def test_n2_garbage_scores_zero():
    task = _n2()
    assert _score(task, "def broken(:\n    pass\n").earned == 0.0


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n2_reference_is_full_on_every_dev_seed(seed):
    task = _n2(seed)
    assert task.id == "N-2" and task.max_points == N2_POINTS
    assert _score(task, task.oracle["reference"]).earned == N2_POINTS


def test_n2_oracle_report_is_recomputed_independently():
    """The expected report is derived here from the structured rows, not read
    from the module's own helper, so a generator bug cannot agree with itself."""
    task = _n2(3)
    rows = [tuple(row) for row in task.oracle["expected_rows"]]
    by_region: dict[str, list[int]] = {}
    by_product: dict[str, int] = {}
    for _, region, product, cents, _status in rows:
        bucket = by_region.setdefault(region, [0, 0])
        bucket[0] += 1
        bucket[1] += cents
        by_product[product] = by_product.get(product, 0) + cents
    expected = {
        "batch": "batch_2026_10_01.csv",
        "rows": len(rows),
        "total_cents": sum(row[3] for row in rows),
        "by_region": [
            {"region": region, "rows": by_region[region][0], "cents": by_region[region][1]}
            for region in sorted(by_region)
        ],
        "top_product": min(by_product, key=lambda product: (-by_product[product], product)),
    }
    assert task.oracle["expected_report"] == expected
    # The constructed tie is real: two products share the maximum total and the
    # alphabetical one wins; no filler may exceed them on any dev seed.
    totals = sorted(by_product.values(), reverse=True)
    assert totals[0] == totals[1]
    assert task.oracle["expected_report"]["top_product"] == "alpha"


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n2_fixture_is_valid_and_bounded(seed):
    task = _n2(seed)
    regions = set(task.workspace["config/regions.txt"].split())
    lines = task.workspace["data/batch_2026_10_01.csv"].splitlines()
    assert lines[0] == "order_id,region,product,cents,status"
    rows = [line.split(",") for line in lines[1:]]
    assert 14 <= len(rows) <= 20
    ids = [row[0] for row in rows]
    assert len(set(ids)) == len(ids)
    for order_id, region, product, cents, status in rows:
        assert order_id.isdigit() and int(order_id) > 0
        assert region in regions and product and cents.isdigit() and int(cents) > 0
        assert status in ("paid", "refunded")
    assert sorted(int(value) for value in ids) == [row[0] for row in task.oracle["expected_rows"]]
    # The invalid batch has exactly the four declared violations on their lines.
    broken = task.workspace["data/batch_2026_10_02.csv"].splitlines()
    assert task.oracle["invalid_lines"] == [4, 7, 8, 11]
    assert broken[3].split(",")[1] not in regions
    assert not broken[6].split(",")[3].isdigit()
    assert broken[7].split(",")[0] == broken[1].split(",")[0]
    assert broken[10].split(",")[4] not in ("paid", "refunded")


def test_n2_prompt_declares_the_contract():
    task = _n2()
    prompt = task.prompt
    for needle in (
        "build/warehouse.db",
        "reports/summary.json",
        "config/regions.txt",
        "order_id,region,product,cents,status",
        "invalid batch:",
        "код 2",
        "64",
        "томарность",
        "protected",
    ):
        assert needle in prompt
    assert task.oracle["reference"] not in prompt
    assert task.oracle["answer_file"] == task.oracle["target_file"] == "pipeline.py"
    assert task.oracle["answer_file"] in task.workspace
    for rel in coding_workflow.PROTECTED:
        assert rel in task.workspace
    assert task.fresh_after == "2026-10-01"
    assert task.languages == ["ru"]


def test_n2_agent_mode_grades_the_file_on_disk(tmp_path):
    task = _n2()
    work = tmp_path / "copy"
    for rel, content in task.workspace.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (work / task.oracle["answer_file"]).write_text(task.oracle["reference"], encoding="utf-8")
    transcript = Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[], reply=ModelReply(text="готово"))],
        extra={"harness": "agent", "work_dir": str(work)},
    )
    score = coding.evaluate(task, transcript)
    assert score.earned == N2_POINTS


def test_n2_is_stretch_only_and_registered_by_profile():
    stretch = coding.generate(random.Random(1), RunConfig(profile="stretch-v1"))
    pilot = coding.generate(random.Random(1), RunConfig(profile="pilot"))
    assert [task.id for task in stretch] == list(coding.STRETCH_IDS)
    assert "N-2" in coding.STRETCH_IDS
    assert "N-2" not in [task.id for task in pilot]
    category = core.REGISTRY["C"]
    assert category.expected_ids_for("stretch-v1") == coding.STRETCH_IDS
    assert category.expected_ids_for("pilot") == coding.PILOT_IDS
    assert category.expected_ids_for("baseline-20261020") == coding.BASELINE_IDS


# --------------------------------------------------------------------------- #
# C-7 — library composition (base64 ZIP → CSV → SQLite)
# --------------------------------------------------------------------------- #

C7_POINTS = 8.0


def _c7(seed: int = 1):
    return coding_python.generate_c7(random.Random(seed), RunConfig(profile="stretch-v1"))


_C7_PARTIAL = '''def command_import(path: Path) -> int:
    declared = entries()
    header = columns()
    allowed = regions()
    BUILD.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB)
    applied = 0
    try:
        connection.execute(TABLE)
        with zipfile.ZipFile(io.BytesIO(decode(path))) as archive:
            for batch in declared:
                name = str(batch["member"])
                with archive.open(name) as member:
                    with io.TextIOWrapper(member, encoding="utf-8", newline="") as stream:
                        text = stream.read()
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != batch["sha256"]:
                    raise Invalid(f"sha256 {name}")
                rows = []
                reader = csv.reader(io.StringIO(text, newline=""))
                next(reader, None)
                for fields in reader:
                    order_id, region, product, cents, status = fields
                    rows.append((int(order_id), region, product, int(cents), status))
                connection.executemany(
                    "INSERT INTO events (order_id, region, product, cents, status, batch)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(order_id) DO UPDATE SET"
                    " region = excluded.region, product = excluded.product,"
                    " cents = excluded.cents, status = excluded.status, batch = excluded.batch",
                    [(a, b, c, d, e, name) for a, b, c, d, e in rows],
                )
                connection.commit()
                applied += len(rows)
        payload = summary(connection)
    except Invalid as exc:
        return fail(str(exc))
    finally:
        connection.close()
    publish(payload)
    print(f"ok: rows={applied}")
    return 0


'''


def _c7_mutant(reference: str, kind: str) -> str:
    if kind == "leaked_handles":
        source = _replace(
            reference,
            'STATUSES = ("paid", "refunded")',
            'STATUSES = ("paid", "refunded")\n_OPEN: list = []',
        )
        source = _replace(
            source,
            "with zipfile.ZipFile(io.BytesIO(decode(path))) as archive:",
            "archive = zipfile.ZipFile(io.BytesIO(decode(path)))\n"
            "    _OPEN.append(archive)\n"
            "    if True:",
        )
        return _replace(
            source,
            "    finally:\n        connection.close()",
            "    finally:\n        _OPEN.append(connection)",
        )
    if kind == "partial_write":
        return _replace_block(
            reference, "def command_import", "def command_report", _C7_PARTIAL
        )
    if kind == "no_newline":
        source = _replace(
            reference,
            'io.TextIOWrapper(member, encoding="utf-8", newline="")',
            'io.TextIOWrapper(member, encoding="utf-8")',
        )
        return _replace(source, 'io.StringIO(text, newline="")', "io.StringIO(text)")
    if kind == "unsorted_region":
        return _replace(
            reference, "GROUP BY region ORDER BY region", "GROUP BY region ORDER BY region DESC"
        )
    raise AssertionError(kind)


def test_c7_library_composition_reference_scores_full():
    task = _c7()
    score = _score(task, task.oracle["reference"])
    assert score.earned == score.possible == C7_POINTS
    assert [check.name for check in score.checks] == list(task.oracle["checks"])


@pytest.mark.parametrize(
    ("kind", "failed"),
    [
        ("leaked_handles", "resources_closed"),
        ("partial_write", "invalid_sha_atomic"),
        ("no_newline", "pinned_apis"),
        ("unsorted_region", "report_content"),
    ],
)
def test_c7_library_composition_mutant_does_not_score_full(kind, failed):
    task = _c7()
    score = _score(task, _c7_mutant(task.oracle["reference"], kind))
    assert score.earned < C7_POINTS
    assert not next(check for check in score.checks if check.name == failed).passed


def test_c7_library_composition_garbage_scores_zero():
    assert _score(_c7(), "#!/usr/bin/env python3\nmain(\n").earned == 0.0


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_c7_library_composition_reference_is_full_on_every_dev_seed(seed):
    task = _c7(seed)
    assert task.id == "C-7" and task.max_points == C7_POINTS
    assert _score(task, task.oracle["reference"]).earned == C7_POINTS


def test_c7_library_composition_oracle_report_is_recomputed_independently():
    """The report is derived here from the structured rows, not from the
    module's own helper, so the generator cannot agree with itself."""
    task = _c7(3)
    rows = [tuple(row) for row in task.oracle["expected_rows"]]
    by_region: dict[str, list[int]] = {}
    by_product: dict[str, int] = {}
    for _, region, product, cents, _status in rows:
        bucket = by_region.setdefault(region, [0, 0])
        bucket[0] += 1
        bucket[1] += cents
        by_product[product] = by_product.get(product, 0) + cents
    expected = {
        "by_region": [
            {"region": region, "rows": by_region[region][0], "cents": by_region[region][1]}
            for region in sorted(by_region)
        ],
        "rows": len(rows),
        "top_product": min(by_product, key=lambda name: (-by_product[name], name)),
        "total_cents": sum(row[3] for row in rows),
    }
    assert task.oracle["expected_report"] == expected
    totals = sorted(by_product.values(), reverse=True)
    assert totals[0] == totals[1]
    assert task.oracle["expected_report"]["top_product"] == "alpha"


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_c7_library_composition_fixture_is_valid_and_bounded(seed):
    task = _c7()
    manifest = json.loads(task.workspace["data/manifest.json"])
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(task.workspace["data/exports.zip.b64"])))
    try:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    finally:
        archive.close()
    header = task.workspace["config/columns.txt"].strip()
    regions = set(task.workspace["config/regions.txt"].split())
    assert sorted(members) == sorted(batch["member"] for batch in manifest["batches"])
    seen: list[str] = []
    for batch in manifest["batches"]:
        data = members[batch["member"]].decode("utf-8")
        assert hashlib.sha256(data.encode("utf-8")).hexdigest() == batch["sha256"]
        lines = data.splitlines()
        assert lines[0] == header
        rows = [line.split(",") for line in lines[1:]]
        assert len(rows) == batch["rows"] == len(set(map(tuple, rows)))
        for order_id, region, product, cents, status in rows:
            assert order_id.isdigit() and int(order_id) > 0
            assert region in regions and product and cents.isdigit() and int(cents) > 0
            assert status in ("paid", "refunded")
        seen.extend(row[0] for row in rows)
    assert 14 == len(seen) and len(set(seen)) == 12
    assert sorted(map(int, set(seen))) == [row[0] for row in task.oracle["expected_rows"]]


def test_c7_library_composition_prompt_declares_the_contract():
    task = _c7()
    prompt = task.prompt
    for needle in (
        "bin/import.py",
        "build/warehouse.db",
        "reports/summary.json",
        "config/columns.txt",
        "order_id,region,product,cents,status",
        "invalid archive:",
        "код 2",
        "код 64",
        "os.replace",
        'newline=""',
        "top_product",
        "sha256",
    ):
        assert needle in prompt
    assert task.oracle["reference"] not in prompt
    assert task.oracle["answer_file"] == task.oracle["target_file"] == "bin/import.py"
    assert task.oracle["answer_file"] in task.workspace
    assert task.languages == ["ru"]
    assert task.meta["scenario"] == "C-7"


def test_c7_library_composition_agent_mode_grades_the_file_on_disk(tmp_path):
    task = _c7()
    work = tmp_path / "copy"
    for rel, content in task.workspace.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (work / task.oracle["answer_file"]).write_text(task.oracle["reference"], encoding="utf-8")
    transcript = Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[], reply=ModelReply(text="готово"))],
        extra={"harness": "agent", "work_dir": str(work)},
    )
    score = coding.evaluate(task, transcript)
    assert score.earned == C7_POINTS


def test_c7_library_composition_is_stretch_only():
    stretch = coding.generate(random.Random(1), RunConfig(profile="stretch-v1"))
    pilot = coding.generate(random.Random(1), RunConfig(profile="pilot"))
    assert [task.id for task in stretch] == list(coding.STRETCH_IDS)
    assert "C-7" in coding.STRETCH_IDS
    assert "C-7" not in [task.id for task in pilot]
    category = core.REGISTRY["C"]
    assert category.expected_ids_for("stretch-v1") == coding.STRETCH_IDS


# --------------------------------------------------------------------------- #
# N-9 (stretch): program from spec — a package with commands, Decimal money,
# idempotent repeats and an atomic error. Only the public CLI is graded.
# --------------------------------------------------------------------------- #

from decimal import ROUND_HALF_UP, Decimal  # noqa: E402

from dailybench.tasks import coding_program  # noqa: E402

N9_POINTS = 1.0


def _n9(seed: int = 1):
    return coding_program.generate(random.Random(seed), RunConfig(profile="stretch-v1"))


def _n9_answer(files: dict[str, str]) -> str:
    return "\n".join(f"### FILE: {rel}\n```python\n{body}```" for rel, body in files.items())


def _n9_score(task, files: dict[str, str]):
    transcript = Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[], reply=ModelReply(text=_n9_answer(files)))],
    )
    return coding.evaluate(task, transcript)


class _N9Alternative:
    """A second correct solution: another module split and another state file."""

    MONEY = '''"""Rendering of money values."""

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def render(value: Decimal) -> str:
    return format(value.quantize(CENT, rounding=ROUND_HALF_UP), "f")
'''

    MAIN = r'''#!/usr/bin/env python3
"""Ledger over a JSONL journal, written as an independent second solution."""

from __future__ import annotations

import json
import os
import re
import sys
from decimal import Decimal
from pathlib import Path

from .money import render

JOURNAL = Path("state/journal.jsonl")
REPORT = Path("reports/statement.json")
CONFIG = Path("config")
USAGE = "usage: python3 -m ledger post <batch> | statement | unpost <key>"
ID = re.compile(r"TX-[0-9]{4}\Z")
KEY = re.compile(r"[a-z0-9-]{3,32}\Z")
MONEY = re.compile(r"(0|[1-9][0-9]*)\.[0-9]{4}\Z")
FIELDS = ("id", "account", "amount", "key")


def report(message: str, code: int) -> int:
    print(message, file=sys.stderr)
    return code


def config() -> tuple[dict[str, str], dict[str, Decimal]]:
    accounts = {
        item["code"]: item["currency"]
        for item in json.loads((CONFIG / "accounts.json").read_text())["accounts"]
    }
    rates = {
        code: Decimal(value) for code, value in json.loads((CONFIG / "rates.json").read_text()).items()
    }
    return accounts, rates


def journal() -> list[dict[str, str]]:
    if not JOURNAL.exists():
        return []
    return [json.loads(line) for line in JOURNAL.read_text(encoding="utf-8").splitlines() if line]


def rewrite(entries: list[dict[str, str]]) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    staging = JOURNAL.parent / "journal.next"
    staging.write_text(
        "".join(json.dumps(entry, separators=(",", ":")) + "\n" for entry in entries),
        encoding="utf-8",
    )
    os.replace(staging, JOURNAL)


def check(line: str, accounts: dict[str, str], rates: dict[str, Decimal]) -> dict[str, str] | None:
    try:
        item = json.loads(line)
    except ValueError:
        return None
    if type(item) is not dict or tuple(item) != FIELDS:
        return None
    if any(type(item[name]) is not str for name in FIELDS):
        return None
    if not ID.match(item["id"]) or not KEY.match(item["key"]):
        return None
    if item["account"] not in accounts or accounts[item["account"]] not in rates:
        return None
    if not MONEY.match(item["amount"]):
        return None
    if not Decimal("0") < Decimal(item["amount"]) <= Decimal("1000000"):
        return None
    return {name: item[name] for name in FIELDS}


def batch(path: str) -> tuple[list[dict[str, str]] | None, list[int]]:
    target = Path(path)
    if not target.is_file():
        return None, []
    accounts, rates = config()
    good: list[dict[str, str]] = []
    bad: list[int] = []
    keys: set[str] = set()
    ids: set[str] = set()
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        entry = check(line, accounts, rates) if line.strip() else None
        if entry is None or entry["key"] in keys or entry["id"] in ids:
            bad.append(number)
            continue
        keys.add(entry["key"])
        ids.add(entry["id"])
        good.append(entry)
    return good, bad


def main(argv: list[str]) -> int:
    if not argv:
        return report(USAGE, 64)
    if argv[0] == "post":
        if len(argv) != 2:
            return report(USAGE, 64)
        good, bad = batch(argv[1])
        if good is None:
            return report(f"invalid batch: missing {argv[1]}", 2)
        if bad:
            return report("invalid batch: " + ",".join(map(str, bad)), 2)
        current = journal()
        known = {entry["key"] for entry in current}
        applied = skipped = 0
        for entry in good:
            if entry["key"] in known:
                skipped += 1
                continue
            known.add(entry["key"])
            current.append(entry)
            applied += 1
        if applied:
            rewrite(current)
        print(f"ok: applied={applied} skipped={skipped}")
        return 0
    if argv[0] == "unpost":
        if len(argv) != 2:
            return report(USAGE, 64)
        current = journal()
        kept = [entry for entry in current if entry["key"] != argv[1]]
        if len(kept) == len(current):
            return report(f"missing key: {argv[1]}", 3)
        rewrite(kept)
        print("ok: removed=1")
        return 0
    if argv[0] == "statement":
        if len(argv) != 1:
            return report(USAGE, 64)
        accounts, rates = config()
        groups: dict[str, list[Decimal]] = {}
        for entry in journal():
            groups.setdefault(entry["account"], []).append(
                Decimal(entry["amount"]) * rates[accounts[entry["account"]]]
            )
        payload = {
            "accounts": [
                {"account": code, "entries": len(values), "eur_total": render(sum(values, Decimal("0")))}
                for code, values in sorted(groups.items())
            ],
            "rows": sum(len(values) for values in groups.values()),
            "grand_total": render(sum((sum(values, Decimal("0")) for values in groups.values()), Decimal("0"))),
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        staging = REPORT.parent / "statement.next"
        staging.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(staging, REPORT)
        return 0
    return report(USAGE, 64)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''

    @classmethod
    def files(cls) -> dict[str, str]:
        return {
            "ledger/__init__.py": '"""Ledger package."""\n',
            "ledger/money.py": cls.MONEY,
            "ledger/__main__.py": cls.MAIN,
        }


_FLOAT_STATEMENT = '''    totals: dict[str, list[float]] = {}
    for entry in current:
        bucket = totals.setdefault(entry["account"], [0, 0.0])
        bucket[0] += 1
        bucket[1] += float(entry["amount"]) * float(rate_table[known[entry["account"]]])
    grand = sum(bucket[1] for bucket in totals.values())'''

_PARTIAL_WRITE = '''    if bad:
        current = load()
        have = {entry["key"] for entry in current}
        for entry in entries:
            if entry["key"] not in have:
                current.append(entry)
                have.add(entry["key"])
        if entries:
            save(current)
        return fail("invalid batch: " + ",".join(str(number) for number in bad))'''


def _n9_mutant(task, kind: str) -> dict[str, str]:
    files = dict(task.oracle["reference"])
    main = files["ledger/__main__.py"]
    if kind == "float_statement":
        main = _replace(
            main,
            '''    totals: dict[str, list[Any]] = {}
    for entry in current:
        bucket = totals.setdefault(entry["account"], [0, Decimal("0")])
        bucket[0] += 1
        bucket[1] += Decimal(entry["amount"]) * rate_table[known[entry["account"]]]
    grand = sum((bucket[1] for bucket in totals.values()), Decimal("0"))''',
            _FLOAT_STATEMENT,
        )
        main = _replace(
            main,
            '"eur_total": money(totals[account][1])',
            '"eur_total": f"{totals[account][1]:.2f}"',
        )
        main = _replace(main, '"grand_total": money(grand),', '"grand_total": f"{grand:.2f}",')
    elif kind == "duplicate_post":
        main = _replace(main, '    have = {entry["key"] for entry in current}', "    have: set[str] = set()")
    elif kind == "partial_write":
        main = _replace(
            main,
            '''    if bad:
        return fail("invalid batch: " + ",".join(str(number) for number in bad))''',
            _PARTIAL_WRITE,
        )
    else:  # pragma: no cover - guard for a typo in the parametrisation
        raise AssertionError(kind)
    files["ledger/__main__.py"] = main
    return files


def test_n9_program_reference_scores_full():
    task = _n9()
    score = _n9_score(task, task.oracle["reference"])
    assert score.earned == score.possible == N9_POINTS
    assert [check.name for check in score.checks] == task.oracle["checks"]
    assert len(task.oracle["checks"]) * coding_program._POINTS_PER_CHECK == N9_POINTS


def test_n9_program_alternative_implementation_scores_full():
    """Another module split and another state layout still satisfy the spec."""
    task = _n9(2)
    score = _n9_score(task, _N9Alternative.files())
    assert score.earned == score.possible == N9_POINTS


@pytest.mark.parametrize(
    ("kind", "failed"),
    [
        ("float_statement", "statement_values"),
        ("duplicate_post", "idempotent_repost"),
        ("partial_write", "invalid_atomic_restore"),
    ],
)
def test_n9_program_mutant_does_not_score_full(kind, failed):
    task = _n9(1)
    score = _n9_score(task, _n9_mutant(task, kind))
    assert score.earned < N9_POINTS
    assert not next(check for check in score.checks if check.name == failed).passed


def test_n9_program_garbage_scores_zero():
    task = _n9()
    assert _n9_score(task, {"ledger/__init__.py": "", "ledger/__main__.py": "def broken(:\n"}).earned == 0.0
    assert _n9_score(task, {}).earned == 0.0


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n9_program_reference_is_full_on_every_dev_seed(seed):
    task = _n9(seed)
    assert task.id == "N-9" and task.max_points == N9_POINTS
    assert task.oracle["reference"]["ledger/__main__.py"] not in task.prompt
    assert _n9_score(task, task.oracle["reference"]).earned == N9_POINTS


def _n9_independent_report(rows: list[dict[str, str]], accounts: dict[str, str], rates: dict[str, str]) -> dict:
    exact: dict[str, Decimal] = {}
    counts: dict[str, int] = {}
    for row in rows:
        exact[row["account"]] = exact.get(row["account"], Decimal("0")) + Decimal(row["amount"]) * Decimal(
            rates[accounts[row["account"]]]
        )
        counts[row["account"]] = counts.get(row["account"], 0) + 1
    total = sum(exact.values(), Decimal("0"))
    return {
        "accounts": [
            {
                "account": account,
                "entries": counts[account],
                "eur_total": str(exact[account].quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            }
            for account in sorted(counts)
        ],
        "rows": len(rows),
        "grand_total": str(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
    }


def _n9_rates(task) -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(task.workspace[coding_program.ACCOUNTS_PATH])
    rates = json.loads(task.workspace[coding_program.RATES_PATH])
    return {item["code"]: item["currency"] for item in payload["accounts"]}, rates


def test_n9_program_oracle_is_recomputed_independently():
    task = _n9(3)
    accounts, rates = _n9_rates(task)
    rows = task.oracle["entries"]
    assert task.oracle["expected_report"] == _n9_independent_report(rows, accounts, rates)
    # The ties are real: each XDR total ends in a half at two decimals and a
    # binary float sum drifts below it, so half-up and float disagree.
    expected = {item["account"]: item["eur_total"] for item in task.oracle["expected_report"]["accounts"]}
    ties = drifts = 0
    for account in sorted({row["account"] for row in rows}):
        if accounts[account] != "XDR":
            continue
        exact = sum(
            (Decimal(row["amount"]) * Decimal(rates[accounts[account]]) for row in rows if row["account"] == account),
            Decimal("0"),
        )
        assert re.search(r"\.\d\d50$", f"{exact:.4f}")
        drift = sum(
            (float(row["amount"]) * float(rates[accounts[account]]) for row in rows if row["account"] == account),
            0.0,
        )
        ties += 1
        drifts += f"{drift:.2f}" != expected[account]
    assert ties == 2
    assert drifts >= 1
    known = {row["key"] for row in rows}
    fresh = [
        json.loads(line)
        for line in task.workspace[coding_program.MIXED_BATCH].splitlines()
        if json.loads(line)["key"] not in known
    ]
    after_mixed = rows + fresh
    assert task.oracle["expected_mixed_report"] == _n9_independent_report(after_mixed, accounts, rates)
    unposted = [row for row in after_mixed if row["key"] != task.oracle["unpost_key"]]
    assert task.oracle["expected_unpost_report"] == _n9_independent_report(unposted, accounts, rates)


def _n9_valid_line(line: str, accounts: dict[str, str], rates: dict[str, str], seen_keys: set, seen_ids: set) -> bool:
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(item, dict) or list(item) != ["id", "account", "amount", "key"]:
        return False
    if not re.fullmatch(r"TX-[0-9]{4}", item["id"]) or not re.fullmatch(r"[a-z0-9-]{3,32}", item["key"]):
        return False
    if item["account"] not in accounts or accounts[item["account"]] not in rates:
        return False
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.[0-9]{4}", item["amount"]):
        return False
    if not 0 < Decimal(item["amount"]) <= Decimal("1000000"):
        return False
    if item["key"] in seen_keys or item["id"] in seen_ids:
        return False
    seen_keys.add(item["key"])
    seen_ids.add(item["id"])
    return True


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_n9_program_fixtures_are_valid_and_bounded(seed):
    task = _n9(seed)
    accounts, rates = _n9_rates(task)
    valid = task.workspace[coding_program.VALID_BATCH].splitlines()
    assert 12 <= len(valid) <= 18
    seen_keys: set[str] = set()
    seen_ids: set[str] = set()
    for line in valid:
        assert _n9_valid_line(line, accounts, rates, seen_keys, seen_ids)
    invalid = task.workspace[coding_program.INVALID_BATCH].splitlines()
    bad_lines = task.oracle["invalid_lines"]
    assert len(bad_lines) == 8
    seen_keys = set()
    seen_ids = set()
    detected: list[int] = []
    for number, line in enumerate(invalid, start=1):
        if not _n9_valid_line(line, accounts, rates, seen_keys, seen_ids):
            detected.append(number)
    assert detected == bad_lines
    # The valid prefix stays untouched, and every mixed line is fresh or a known key.
    mixed = task.workspace[coding_program.MIXED_BATCH].splitlines()
    assert len(mixed) == 4
    known = {json.loads(line)["key"] for line in valid}
    fresh = [line for line in mixed if json.loads(line)["key"] not in known]
    assert len(fresh) == task.oracle["fresh_entries"] == 2


def test_n9_program_prompt_declares_the_contract():
    task = _n9()
    prompt = task.prompt
    for needle in (
        "python3 -m ledger post",
        "python3 -m ledger statement",
        "python3 -m ledger unpost",
        "invalid batch:",
        "missing key:",
        "64",
        "томарность",
        "decimal.Decimal",
        "половина вверх",
        "### FILE:",
        task.oracle["report_path"],
        task.oracle["state_dir"],
        coding_program.MAIN_FILE,
        coding_program.INIT_FILE,
        "Защищённые файлы",
    ):
        assert needle in prompt
    assert task.oracle["answer_file"] == task.oracle["target_file"] == coding_program.MAIN_FILE
    assert coding_program.MAIN_FILE in task.workspace
    for rel in coding_program.PROTECTED:
        assert rel in task.workspace
        assert rel in prompt
    assert task.fresh_after == "2026-10-15"
    assert task.languages == ["ru"]


def test_n9_program_answer_protocol_ignores_extra_paths():
    reply = (
        "Вот пакет.\n\n"
        "### FILE: notes.txt\n```\nignored\n```\n"
        "### FILE: ledger/__main__.py\n```python\nprint('hi')\n```\n"
        "### FILE: ../escape.py\n```python\nprint('no')\n```\n"
    )
    assert coding_program._parse_answer(reply) == {"ledger/__main__.py": "print('hi')\n"}


def test_n9_program_agent_mode_grades_the_package_on_disk(tmp_path):
    task = _n9()
    work = tmp_path / "copy"
    for rel, content in task.workspace.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for rel, content in task.oracle["reference"].items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    transcript = Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[], reply=ModelReply(text="готово"))],
        extra={"harness": "agent", "work_dir": str(work)},
    )
    assert coding.evaluate(task, transcript).earned == N9_POINTS
    # A missing package on disk is an honest zero, not the printed stub.
    empty = tmp_path / "empty"
    empty.mkdir()
    blank = Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[], reply=ModelReply(text=_n9_answer(task.oracle["reference"])))],
        extra={"harness": "agent", "work_dir": str(empty)},
    )
    assert coding.evaluate(task, blank).earned == 0.0


def test_n9_program_is_stretch_only_and_registered_by_profile():
    stretch = coding.generate(random.Random(1), RunConfig(profile="stretch-v1"))
    pilot = coding.generate(random.Random(1), RunConfig(profile="pilot"))
    research = coding.generate(random.Random(1), RunConfig(profile="research-v1"))
    assert [task.id for task in stretch] == list(coding.STRETCH_IDS)
    assert "N-9" in coding.STRETCH_IDS and coding.STRETCH_IDS[-1] == "N-9"
    assert "N-9" not in [task.id for task in pilot]
    assert "N-9" not in [task.id for task in research]
    category = core.REGISTRY["C"]
    assert category.expected_ids_for("stretch-v1") == coding.STRETCH_IDS
    assert category.expected_ids_for("research-v1") == coding.RESEARCH_IDS == ("N-2",)
