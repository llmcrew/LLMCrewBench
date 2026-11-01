"""Scenario C-1 — Python ``AsyncTTLCache`` (10 hidden checks × 1 point).

The model receives a stub package and must return the complete corrected file.
Ten hidden checks cover the requirements from the specification: load-on-miss,
TTL hit, reload after expiry, single-flight for one key, parallel loading of
different keys, exceptions not cached, cancellation not wedging a key, ``None``
as a real value, hits not extending the TTL and no surviving background tasks.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .. import core, text

#: Points per hidden check; three scenarios share the constant so a check
#: weighs the same wherever it is earned.
_POINTS_PER_CHECK = 0.8


CHECK_ORDER = (
    ("first_load", "test_first_load"),
    ("ttl_hit", "test_hit_within_ttl"),
    ("ttl_expiry", "test_reload_after_ttl"),
    ("single_flight", "test_single_flight_same_key"),
    ("parallel_keys", "test_parallel_different_keys"),
    ("exception_not_cached", "test_exception_not_cached"),
    ("cancel_recovery", "test_cancelled_load_not_stuck"),
    ("none_cached", "test_none_is_cached"),
    ("ttl_not_extended", "test_hit_does_not_extend_ttl"),
    ("no_background_tasks", "test_no_background_tasks"),
)

TARGET_FILE = "ttlcache/cache.py"

_STUB = '''"""Minimal async TTL cache."""

from __future__ import annotations

import asyncio
import time


class AsyncTTLCache:
    """Cache values produced by an async loader for a fixed TTL.

    Public API (must not change):

      * ``__init__(self, ttl, *, clock=time.monotonic, sleep=asyncio.sleep)``
        ``ttl`` is in the same units as ``clock()``. ``clock`` is an injectable
        time source; TTL decisions must use it, not ``time.monotonic`` directly.
      * ``async def get_or_load(self, key, loader)``
        ``loader`` is a zero-argument coroutine function producing the value.
    """

    def __init__(self, ttl: float, *, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.ttl = ttl
        self._clock = clock
        self._sleep = sleep

    async def get_or_load(self, key, loader):
        raise NotImplementedError
'''

_REFERENCE = '''"""Minimal async TTL cache."""

from __future__ import annotations

import asyncio
import time


class AsyncTTLCache:
    """Cache values produced by an async loader for a fixed TTL.

    Public API (must not change):

      * ``__init__(self, ttl, *, clock=time.monotonic, sleep=asyncio.sleep)``
        ``ttl`` is in the same units as ``clock()``. ``clock`` is an injectable
        time source; TTL decisions must use it, not ``time.monotonic`` directly.
      * ``async def get_or_load(self, key, loader)``
        ``loader`` is a zero-argument coroutine function producing the value.
    """

    def __init__(self, ttl: float, *, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.ttl = ttl
        self._clock = clock
        self._sleep = sleep
        self._entries: dict[object, tuple[object, float]] = {}
        self._inflight: dict[object, "asyncio.Future[object]"] = {}

    async def get_or_load(self, key, loader):
        entry = self._entries.get(key)
        if entry is not None:
            value, expires_at = entry
            if self._clock() < expires_at:
                return value
            self._entries.pop(key, None)

        pending = self._inflight.get(key)
        if pending is not None:
            return await asyncio.shield(pending)

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[object]" = loop.create_future()
        self._inflight[key] = future
        try:
            value = await loader()
        except BaseException:
            self._inflight.pop(key, None)
            if not future.done():
                future.cancel()
            raise
        self._inflight.pop(key, None)
        self._entries[key] = (value, self._clock() + self.ttl)
        if not future.done():
            future.set_result(value)
        return value
'''

_PUBLIC_TEST = '''"""Public test showing the intended behaviour (a subset of the requirements)."""

import asyncio
import unittest

from ttlcache.cache import AsyncTTLCache


class PublicTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_call_is_served_from_cache(self) -> None:
        clock = lambda: 0.0  # noqa: E731
        cache = AsyncTTLCache(30.0, clock=clock, sleep=asyncio.sleep)

        calls = []

        async def loader():
            calls.append(1)
            return "value"

        self.assertEqual(await cache.get_or_load("k", loader), "value")
        self.assertEqual(await cache.get_or_load("k", loader), "value")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
'''

_HIDDEN_TESTS = '''"""Hidden acceptance tests for AsyncTTLCache (not visible to the model)."""

import asyncio
import unittest

from ttlcache.cache import AsyncTTLCache


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += float(delta)


def make_cache(ttl: float, clock: FakeClock) -> AsyncTTLCache:
    return AsyncTTLCache(ttl, clock=clock, sleep=asyncio.sleep)


class TTLTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_load(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return "value"

        self.assertEqual(await cache.get_or_load("k", loader), "value")
        self.assertEqual(len(calls), 1)

    async def test_hit_within_ttl(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return len(calls)

        first = await cache.get_or_load("k", loader)
        clock.advance(9.0)
        second = await cache.get_or_load("k", loader)
        self.assertEqual(second, first)
        self.assertEqual(len(calls), 1)

    async def test_reload_after_ttl(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return len(calls)

        await cache.get_or_load("k", loader)
        clock.advance(10.0)
        self.assertEqual(await cache.get_or_load("k", loader), 2)
        self.assertEqual(len(calls), 2)

    async def test_single_flight_same_key(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            await asyncio.sleep(0.02)
            return "shared"

        results = await asyncio.gather(*[cache.get_or_load("k", loader) for _ in range(20)])
        self.assertEqual(len(calls), 1, "loader must run exactly once for one key")
        self.assertTrue(all(r == "shared" for r in results))

    async def test_parallel_different_keys(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = set()
        both_inside = asyncio.Event()

        def make_loader(key):
            async def loader():
                started.add(key)
                if len(started) >= 2:
                    both_inside.set()
                await asyncio.wait_for(both_inside.wait(), timeout=2.0)
                return key

            return loader

        results = await asyncio.gather(
            cache.get_or_load("a", make_loader("a")),
            cache.get_or_load("b", make_loader("b")),
        )
        self.assertEqual(sorted(results), ["a", "b"])

    async def test_exception_not_cached(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def failing():
            calls.append("fail")
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            await cache.get_or_load("k", failing)

        async def ok():
            calls.append("ok")
            return "good"

        self.assertEqual(await cache.get_or_load("k", ok), "good")
        self.assertEqual(calls, ["fail", "ok"])

    async def test_cancelled_load_not_stuck(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(5)
            return "slow"

        task = asyncio.create_task(cache.get_or_load("k", slow))
        try:
            await asyncio.wait_for(started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self.fail("loader was never started")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        async def fresh():
            return "fresh"

        self.assertEqual(await asyncio.wait_for(cache.get_or_load("k", fresh), timeout=1.0), "fresh")

    async def test_none_is_cached(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return None

        self.assertIsNone(await cache.get_or_load("k", loader))
        self.assertIsNone(await cache.get_or_load("k", loader))
        self.assertEqual(len(calls), 1)

    async def test_hit_does_not_extend_ttl(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return len(calls)

        await cache.get_or_load("k", loader)
        clock.advance(9.0)
        await cache.get_or_load("k", loader)
        clock.advance(1.5)
        await cache.get_or_load("k", loader)
        self.assertEqual(len(calls), 2, "hits must not extend the TTL window")

    async def test_no_background_tasks(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        before = set(asyncio.all_tasks())

        async def loader():
            return "v"

        await cache.get_or_load("k", loader)
        clock.advance(11.0)
        await cache.get_or_load("k2", loader)
        await asyncio.sleep(0.05)
        leftover = [t for t in asyncio.all_tasks() if t not in before and not t.done()]
        self.assertEqual(leftover, [], f"background tasks still running: {leftover}")


if __name__ == "__main__":
    unittest.main()
'''

#: The child that actually touches the candidate. Embedded so the supervisor
#: can materialise it in its own trusted directory.
_CHILD_RUNNER = '''"""Child: imports the untrusted candidate and runs the hidden tests.

Writes one JSONL row per executed test to ``argv[3]`` and finishes with an
``eof`` row carrying a nonce that only ever travelled over stdin. The nonce is
how the supervisor tells a finished run from a candidate that wrote a file of
its own and called ``os._exit(0)`` from the import hook.

The candidate shares this interpreter with the grader, so it can rebind
``unittest`` — the very object that reports the verdict. The grader state is
snapshotted *before* the import and restored before every test, tests are run
one at a time (a patched ``TestSuite.run`` cannot swallow them), and every test
must prove it really executed: the counter is bumped in
``TestCase._callTestMethod``, the single place through which both the sync and
the async case invoke their method. A canary test of the same shape runs first,
so a silently skipped method becomes an integrity failure instead of a pass.
The loaded test set must equal the executed one exactly.
"""

import builtins
import importlib.util
import json
import os
import sys
import unittest

_open = builtins.open
_dumps = json.dumps


class Recorder(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.rows = []
        # Bound per instance: the candidate may rebind the class at any moment,
        # including from inside the function under test, i.e. after the restore
        # that preceded the test. An instance attribute wins over the class.
        for name in _RESULT_METHODS:
            setattr(self, name, _ORIG_RESULT[name].__get__(self))

    def addSuccess(self, test) -> None:
        self.rows.append((test.id(), True, ""))

    def addFailure(self, test, err) -> None:
        self.rows.append((test.id(), False, str(err[1])))

    def addError(self, test, err) -> None:
        self.rows.append((test.id(), False, str(err[1])))

    def addSkip(self, test, reason) -> None:
        self.rows.append((test.id(), False, f"skipped: {reason}"))


_RESULT_METHODS = ("addSuccess", "addFailure", "addError", "addSkip")
_ORIG_RESULT = {name: getattr(Recorder, name) for name in _RESULT_METHODS}

_ORIG_LOAD = unittest.TestLoader.loadTestsFromModule
_LOADER = unittest.TestLoader()

_EXECUTED = {"n": 0}
_ORIG_CALL = unittest.TestCase._callTestMethod
_ORIG_ASYNC_CALL = unittest.IsolatedAsyncioTestCase._callTestMethod


def _counting_call(self, method):
    _EXECUTED["n"] += 1
    return _ORIG_CALL(self, method)


def _counting_async_call(self, method):
    _EXECUTED["n"] += 1
    return _ORIG_ASYNC_CALL(self, method)


unittest.TestCase._callTestMethod = _counting_call
unittest.IsolatedAsyncioTestCase._callTestMethod = _counting_async_call

#: ``label, owner, attribute, what must be there``. Everything the candidate
#: may plausibly rebind to fake a verdict is listed — including the Recorder
#: methods, whose class is reachable from ``sys.modules``.
_OWNED = (
    ("TestCase.run", unittest.TestCase, "run", unittest.TestCase.run),
    (
        "IsolatedAsyncioTestCase.run",
        unittest.IsolatedAsyncioTestCase,
        "run",
        unittest.IsolatedAsyncioTestCase.run,
    ),
    ("TestCase._callTestMethod", unittest.TestCase, "_callTestMethod", _counting_call),
    (
        "IsolatedAsyncioTestCase._callTestMethod",
        unittest.IsolatedAsyncioTestCase,
        "_callTestMethod",
        _counting_async_call,
    ),
    ("TestLoader.loadTestsFromModule", unittest.TestLoader, "loadTestsFromModule", _ORIG_LOAD),
) + tuple(
    (f"Recorder.{name}", Recorder, name, _ORIG_RESULT[name]) for name in _RESULT_METHODS
)


def _restore() -> "list[str]":
    """Put the grader state back; report what the candidate had rebound."""
    tampered = []
    for label, owner, attribute, expected in _OWNED:
        if getattr(owner, attribute, None) is not expected:
            tampered.append(label)
        setattr(owner, attribute, expected)
    return tampered


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


_CANARY = {"ran": 0, "async": 0}


class _Canary(unittest.TestCase):
    def test_canary(self) -> None:
        _CANARY["ran"] += 1


class _AsyncCanary(unittest.IsolatedAsyncioTestCase):
    async def test_canary(self) -> None:
        _CANARY["async"] += 1


def _run_one(test) -> "tuple[Recorder, bool]":
    """Run one test against a fresh recorder; say whether its method ran."""
    before = _EXECUTED["n"]
    recorder = Recorder()
    test.run(recorder)
    return recorder, _EXECUTED["n"] > before


def _run_suite(module, tampered: "list[str]") -> "list[tuple[str, bool, str]]":
    # Rebinding is only a journal note: the restored state is what the verdict
    # rests on, and the canary plus the per-test counter prove the restore held.
    rows: "list[tuple[str, bool, str]]" = []
    if tampered:
        rows.append(("*", True, "integrity: кандидат подменил " + ", ".join(tampered)))

    notes: "list[str]" = []
    canary_before = dict(_CANARY)
    _restore()
    _run_one(_Canary("test_canary"))
    _run_one(_AsyncCanary("test_canary"))
    for kind in ("ran", "async"):
        if _CANARY[kind] == canary_before[kind]:
            notes.append(f"канарейка {kind} не исполнилась")

    suite = _ORIG_LOAD(_LOADER, module)
    tests = list(_flatten(suite))
    expected = [test.id() for test in tests]
    if not expected:
        notes.append("hidden-модуль не объявил ни одного теста")
    if len(set(expected)) != len(expected):
        notes.append("состав тестов повторяется")
    if notes:
        detail = "integrity: " + "; ".join(notes)
        rows.extend((name, False, detail[:400]) for name in expected or ["*"])
        return rows

    for test in tests:
        name = test.id()
        _restore()
        recorder, executed = _run_one(test)
        if not executed:
            rows.append((name, False, "integrity: метод теста не исполнился"))
        elif len(recorder.rows) != 1 or recorder.rows[0][0] != name:
            rows.append((name, False, f"integrity: тест оставил {len(recorder.rows)} строк вместо одной"))
        else:
            rows.append(recorder.rows[0])
    return rows


def main() -> None:
    repo, test_file, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
    nonce = sys.stdin.read().strip()
    rows: "list[tuple[str, bool, str]]" = []
    try:
        sys.path.insert(0, repo)
        spec = importlib.util.spec_from_file_location("hidden_ttl_tests", test_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # the untrusted candidate runs here
        rows = _run_suite(module, _restore())
    except BaseException as exc:  # a broken candidate must still leave a report
        rows.append(("*", False, f"{type(exc).__name__}: {exc}"[:400]))
    payload = "".join(
        _dumps({"test": name, "ok": ok, "detail": str(detail)[:400]}, ensure_ascii=False) + "\\n"
        for name, ok, detail in rows
    )
    payload += _dumps({"eof": nonce}) + "\\n"
    with _open(report_path, "w", encoding="utf-8") as report:
        report.write(payload)
        report.flush()
        os.fsync(report.fileno())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
'''

#: Wall clock for the child; the supervisor itself gets a slightly longer one
#: from the evaluator, so a hung candidate is still reported as a failure.
_CHILD_TIMEOUT = 150

_RUNNER_TEMPLATE = '''"""Supervisor: the only process that writes the final verdict.

The candidate is never imported here. It runs in a child process, which is free
to print, forge its own report path (``argv[3]`` of the child), or call
``os._exit`` — all of that only kills the child. This supervisor waits for the
child, requires a zero exit status and the nonce-marked ``eof`` row the child
appends after the whole suite, re-reads the rows and writes them to a second
path (``argv[4]``) that the child was never told about. The final report is
therefore written after any tampering, from data the supervisor vetted itself.
Everything the candidate prints — even a ``REPORT ...`` line — is a journal:
stdout is never parsed.
"""

import json
import os
import secrets
import subprocess
import sys

_CHILD = __CHILD__

CHILD_TIMEOUT = __TIMEOUT__


def _write(path: str, payload: str) -> None:
    with open(path, "w", encoding="utf-8") as report:
        report.write(payload)
        report.flush()
        os.fsync(report.fileno())


def _fail(path: str, detail: str) -> None:
    _write(path, json.dumps({"test": "*", "ok": False, "detail": detail[:400]}, ensure_ascii=False) + "\\n")


def _tail(stderr: str) -> str:
    lines = (stderr or "").strip().splitlines()
    return (": " + lines[-1][-300:]) if lines else ""


def main() -> int:
    repo, test_file, child_report, final_report = sys.argv[1:5]
    child = os.path.join(os.path.dirname(os.path.abspath(final_report)), "_hidden_child.py")
    _write(child, _CHILD)
    nonce = secrets.token_hex(16)
    try:
        proc = subprocess.run(
            [sys.executable, child, repo, test_file, child_report],
            input=nonce,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHILD_TIMEOUT,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        _fail(final_report, f"скрытые тесты не уложились в {CHILD_TIMEOUT}s")
        return 0
    except OSError as exc:
        _fail(final_report, f"дочерний процесс не запустился: {exc}")
        return 0

    try:
        with open(child_report, encoding="utf-8", errors="replace") as report:
            blob = report.read()
    except OSError as exc:
        _fail(final_report, f"дочерний отчёт не создан ({exc}){_tail(proc.stderr)}")
        return 0

    rows = []
    marker = ""
    for line in blob.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if "eof" in row:
            marker = str(row["eof"])
        else:
            rows.append(row)

    if proc.returncode != 0:
        _fail(
            final_report,
            f"дочерний процесс завершился с кодом {proc.returncode} ({len(rows)} строк){_tail(proc.stderr)}",
        )
        return 0
    if marker != nonce:
        # Rows exist but the child never reached its own write: a candidate that
        # wrote the file itself and called os._exit(0) from the import hook.
        _fail(
            final_report,
            f"дочерний процесс не дошёл до конца ({len(rows)} строк, маркера нет){_tail(proc.stderr)}",
        )
        return 0

    _write(
        final_report,
        "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_RUNNER = _RUNNER_TEMPLATE.replace("__CHILD__", repr(_CHILD_RUNNER)).replace(
    "__TIMEOUT__", str(_CHILD_TIMEOUT)
)


_PROMPT = """В проекте есть модуль ttlcache с классом AsyncTTLCache.

Публичный API менять запрещено.

Требования:

1. Первое обращение к отсутствующему key вызывает loader().
2. Результат хранится в кеше в течение TTL.
3. После окончания TTL loader вызывается снова.
4. Если 20 coroutine одновременно запрашивают один отсутствующий key,
   loader должен выполниться ровно один раз.
5. Разные keys должны загружаться параллельно.
6. Исключение loader не кэшируется.
7. Отменённый loader не оставляет key навсегда заблокированным.
8. None является корректным кэшируемым значением.
9. Повторный cache hit не продлевает TTL.
10. Реализация не должна создавать background tasks,
    продолжающих работу после закрытия теста.

Изменяй только реализацию библиотеки.
Тесты изменять запрещено.

Верни ТОЛЬКО полный текст файла {target} целиком, в одном блоке кода.

Файл ttlcache/cache.py в рабочей копии:

```python
{stub}```"""


def _workspace() -> dict[str, str]:
    return {
        "ttlcache/__init__.py": '"""Async TTL cache package."""\n\nfrom .cache import AsyncTTLCache\n\n__all__ = ["AsyncTTLCache"]\n',
        TARGET_FILE: _STUB,
        "tests/test_cache.py": _PUBLIC_TEST,
        "README.md": "# ttlcache\n\nSmall async TTL cache used by the service runtime.\n",
    }


def generate(rng: random.Random, cfg: Any) -> core.TaskInstance:
    return core.TaskInstance(
        id="C-1",
        category="C",
        title="Python AsyncTTLCache",
        prompt=_PROMPT.format(target=TARGET_FILE, stub=_STUB),
        max_points=8.0,
        workspace=_workspace(),
        oracle={
            "reference": _REFERENCE,
            "solution": _REFERENCE,
            "hidden_tests": _HIDDEN_TESTS,
            "runner": _RUNNER,
            "target_file": TARGET_FILE,
            "answer_file": TARGET_FILE,
            "checks": [name for name, _ in CHECK_ORDER],
        },
        meta={"scenario": "C-1"},
    )


def extract_code(reply: str) -> str | None:
    """Full-file answer: the outermost fenced block when present, else the reply.

    A module may legitimately contain a fenced example in its docstring; the
    first inner fence used to win and truncated the file to the example (C2),
    scoring a correct module 0. The whole-file protocol treats the reply as
    one block: everything between the opening fence line and the final closer.
    """
    candidate = text.strip_fence(reply)[0]
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) > 1 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]) or None
        return candidate[3:].lstrip("\n") or None
    return candidate or None


def code_of(task: core.TaskInstance, tr: core.Transcript) -> str | None:
    """The file the agent left, else the file it printed.

    The agent harness grades what is on disk; a chat run has no working copy and
    takes the printed block. Both paths end in the same evaluator, so a scenario
    cannot score differently because of *how* it was answered.
    """
    from .. import agent

    text_answer = agent.answer_text(task, tr, tr.final_text)
    return extract_code(text_answer)


def _map_rows(blob: str, order: tuple[tuple[str, str], ...]) -> dict[str, tuple[bool, str]]:
    """Map supervisor-vetted report rows onto the declared checks.

    A repeated row for one test is last-wins, both while collecting and when
    looking a check up: the file is written after the child is gone, so the
    last row for a test is the one produced furthest into the run. Every
    declared check needs a row; a missing one fails and carries the global
    (``*``) cause, if the supervisor recorded one.
    """
    results: dict[str, tuple[bool, str]] = {}
    global_detail = ""
    for line in blob.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        test_name = str(row.get("test", ""))
        if test_name == "*":
            global_detail = str(row.get("detail", ""))
        elif test_name:
            results[test_name] = (bool(row.get("ok")), str(row.get("detail", "")))

    mapped: dict[str, tuple[bool, str]] = {}
    for name, test_name in order:
        hits = [
            value
            for key, value in results.items()
            if key == test_name or key.endswith("." + test_name)
        ]
        if hits:
            mapped[name] = hits[-1]
        else:
            suffix = f": {global_detail}" if global_detail else ""
            mapped[name] = (False, f"тест {test_name} не выполнился{suffix}"[:400])
    return mapped


def _run_hidden(
    sandbox: Path,
    repo: Path,
    tests_name: str = "test_ttlcache_hidden.py",
    hidden: str = _HIDDEN_TESTS,
    order: tuple[tuple[str, str], ...] = CHECK_ORDER,
) -> dict[str, tuple[bool, str]]:
    runner = sandbox / "run_hidden.py"
    tests = sandbox / tests_name
    runner.write_text(_RUNNER, encoding="utf-8")
    tests.write_text(hidden, encoding="utf-8")
    # Both reports live in a directory the supervisor picked *outside* the
    # sandbox. The child (which imports the candidate) is told only the
    # throwaway one; the final path stays in the supervisor's argv and is
    # written after the child is gone, so the candidate cannot preempt it.
    trusted = Path(tempfile.mkdtemp(prefix="dailybench-c1-trusted-"))
    child_report = trusted / "child.jsonl"
    report_path = trusted / "report.jsonl"
    try:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    str(repo),
                    str(tests),
                    str(child_report),
                    str(report_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                cwd=str(sandbox),
            )
        except subprocess.TimeoutExpired:
            return {name: (False, "hidden tests timed out (180s)") for name, _ in order}
        except OSError as exc:
            return {name: (False, f"cannot run hidden tests: {exc}") for name, _ in order}

        # Stdout is deliberately ignored: a ``REPORT ...`` line printed by the
        # candidate (before or after the tests, even from ``atexit``) carries no
        # authority. Only the supervisor's own final report is read.
        try:
            blob = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            tail = (proc.stderr or "").strip().splitlines()
            reason = tail[-1] if tail else "нет отчёта"
            return {
                name: (
                    False,
                    f"отчёт скрытых тестов не создан ({exc}, код {proc.returncode}): {reason[-200:]}",
                )
                for name, _ in order
            }
        return _map_rows(blob, order)
    finally:
        shutil.rmtree(trusted, ignore_errors=True)


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    code = code_of(task, tr)
    if not code:
        for name, _ in CHECK_ORDER:
            score.add(name, False, _POINTS_PER_CHECK, "ответ не содержит кода файла")
        score.notes.append("не удалось извлечь файл из ответа")
        return score

    sandbox = Path(tempfile.mkdtemp(prefix="dailybench-c1-"))
    repo = sandbox / "repo"
    repo.mkdir(parents=True)
    for rel, content in task.workspace.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo / TARGET_FILE).write_text(code, encoding="utf-8")

    try:
        results = _run_hidden(sandbox, repo)
        for name, _ in CHECK_ORDER:
            passed, detail = results[name]
            score.add(name, passed, _POINTS_PER_CHECK, detail)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    kept_api = "class AsyncTTLCache" in code and "get_or_load" in code
    if not kept_api:
        score.notes.append("в ответе не найден публичный класс/метод AsyncTTLCache.get_or_load")
    return score


# --------------------------------------------------------------------------- #
# Scenario C-4 (pilot) — generational async cache
# --------------------------------------------------------------------------- #

C4_TARGET_FILE = "gencache/cache.py"

CHECK_ORDER_C4 = (
    ("first_load", "test_first_load"),
    ("ttl_hit", "test_hit_within_ttl"),
    ("ttl_expiry", "test_reload_after_ttl"),
    ("invalidate_entry", "test_invalidate_drops_entry"),
    ("invalidate_inflight", "test_invalidate_starts_fresh_load"),
    ("late_result", "test_late_result_does_not_win"),
    ("cancelled_waiter", "test_cancelled_waiter_keeps_load"),
    ("cancelled_sole_waiter", "test_cancelled_sole_waiter_caches"),
    ("exception_not_cached", "test_exception_not_cached"),
    ("no_background_tasks", "test_no_background_tasks"),
)

_STUB_C4 = '''"""Async TTL cache with invalidation."""

from __future__ import annotations

import asyncio
import time


class AsyncTTLCache:
    """Cache values produced by an async loader for a fixed TTL.

    Public API (must not change):

      * ``__init__(self, ttl, *, clock=time.monotonic, sleep=asyncio.sleep)``
        ``ttl`` is in the same units as ``clock()``.
      * ``async def get_or_load(self, key, loader)``
        Returns the cached value or calls ``loader()`` once for the key.
      * ``async def invalidate(self, key)``
        Drops the cached value and detaches the load that is currently running.
    """

    def __init__(self, ttl: float, *, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.ttl = ttl
        self._clock = clock
        self._sleep = sleep

    async def get_or_load(self, key, loader):
        raise NotImplementedError

    async def invalidate(self, key) -> None:
        raise NotImplementedError
'''

_REFERENCE_C4 = '''"""Async TTL cache with invalidation."""

from __future__ import annotations

import asyncio
import time

#: Keeps an exception of a future that every waiter abandoned from being
#: reported as "never retrieved" when the loop is closed.
def _consume(future: "asyncio.Future[object]") -> None:
    if not future.cancelled():
        future.exception()


class AsyncTTLCache:
    """Cache values produced by an async loader for a fixed TTL.

    Public API (must not change):

      * ``__init__(self, ttl, *, clock=time.monotonic, sleep=asyncio.sleep)``
        ``ttl`` is in the same units as ``clock()``.
      * ``async def get_or_load(self, key, loader)``
        Returns the cached value or calls ``loader()`` once for the key.
      * ``async def invalidate(self, key)``
        Drops the cached value and detaches the load that is currently running.
    """

    def __init__(self, ttl: float, *, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.ttl = ttl
        self._clock = clock
        self._sleep = sleep
        self._entries: dict[object, tuple[object, float]] = {}
        self._inflight: dict[object, "asyncio.Future[object]"] = {}
        self._generation: dict[object, int] = {}

    async def get_or_load(self, key, loader):
        entry = self._entries.get(key)
        if entry is not None:
            value, expires_at = entry
            if self._clock() < expires_at:
                return value
            self._entries.pop(key, None)

        pending = self._inflight.get(key)
        if pending is not None:
            return await asyncio.shield(pending)

        generation = self._generation.get(key, 0) + 1
        self._generation[key] = generation
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[object]" = loop.create_future()
        future.add_done_callback(_consume)
        self._inflight[key] = future
        # The loader runs in its own task, not in the first waiter: cancelling
        # that waiter must not cancel the shared load.
        task = loop.create_task(loader())
        task.add_done_callback(lambda done: self._finish(key, generation, future, done))
        return await asyncio.shield(future)

    def _finish(self, key, generation, future, task) -> None:
        if self._inflight.get(key) is future:
            self._inflight.pop(key, None)
        if task.cancelled():
            if not future.done():
                future.cancel()
            return
        error = task.exception()
        if error is not None:
            if not future.done():
                future.set_exception(error)
            return
        value = task.result()
        # Only the generation that is still current may write the entry: a
        # result finishing after invalidate must not resurrect the key.
        if self._generation.get(key) == generation:
            self._entries[key] = (value, self._clock() + self.ttl)
        if not future.done():
            future.set_result(value)

    async def invalidate(self, key) -> None:
        self._entries.pop(key, None)
        self._inflight.pop(key, None)
        self._generation[key] = self._generation.get(key, 0) + 1
'''

_PUBLIC_TEST_C4 = '''"""Public test showing the intended behaviour (a subset of the requirements)."""

import asyncio
import unittest

from gencache.cache import AsyncTTLCache


class PublicTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalidate_forces_a_reload(self) -> None:
        cache = AsyncTTLCache(30.0, clock=lambda: 0.0)
        calls = []

        async def loader():
            calls.append(1)
            return "value"

        self.assertEqual(await cache.get_or_load("k", loader), "value")
        await cache.invalidate("k")
        self.assertEqual(await cache.get_or_load("k", loader), "value")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
'''

_HIDDEN_TESTS_C4 = '''"""Hidden acceptance tests for the generational AsyncTTLCache."""

import asyncio
import unittest

from gencache.cache import AsyncTTLCache


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += float(delta)


def make_cache(ttl: float, clock: FakeClock) -> AsyncTTLCache:
    return AsyncTTLCache(ttl, clock=clock)


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_load(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return "value"

        self.assertEqual(await cache.get_or_load("k", loader), "value")
        self.assertEqual(len(calls), 1)

    async def test_hit_within_ttl(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return len(calls)

        first = await cache.get_or_load("k", loader)
        clock.advance(9.0)
        self.assertEqual(await cache.get_or_load("k", loader), first)
        self.assertEqual(len(calls), 1)

    async def test_reload_after_ttl(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return len(calls)

        await cache.get_or_load("k", loader)
        clock.advance(10.0)
        self.assertEqual(await cache.get_or_load("k", loader), 2)
        self.assertEqual(len(calls), 2)

    async def test_invalidate_drops_entry(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        calls = []

        async def loader():
            calls.append(1)
            return len(calls)

        await cache.invalidate("never-loaded")
        self.assertEqual(await cache.get_or_load("k", loader), 1)
        await cache.invalidate("k")
        await cache.invalidate("k")
        self.assertEqual(await cache.get_or_load("k", loader), 2)
        self.assertEqual(len(calls), 2)

    async def test_invalidate_starts_fresh_load(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def old():
            calls.append("old")
            started.set()
            await release.wait()
            return "old"

        old_task = asyncio.create_task(cache.get_or_load("k", old))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await cache.invalidate("k")

        async def fresh():
            calls.append("fresh")
            return "fresh"

        self.assertEqual(
            await asyncio.wait_for(cache.get_or_load("k", fresh), timeout=2.0), "fresh"
        )
        release.set()
        self.assertEqual(await asyncio.wait_for(old_task, timeout=2.0), "old")
        self.assertEqual(calls, ["old", "fresh"])

    async def test_late_result_does_not_win(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def old():
            calls.append("old")
            started.set()
            await release.wait()
            return "old"

        old_task = asyncio.create_task(cache.get_or_load("k", old))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await cache.invalidate("k")

        async def fresh():
            calls.append("fresh")
            return "fresh"

        self.assertEqual(
            await asyncio.wait_for(cache.get_or_load("k", fresh), timeout=2.0), "fresh"
        )
        release.set()
        await asyncio.wait_for(old_task, timeout=2.0)

        async def hit_only():
            calls.append("hit")
            return "hit"

        self.assertEqual(await cache.get_or_load("k", hit_only), "fresh")
        self.assertEqual(calls, ["old", "fresh"])

    async def test_cancelled_waiter_keeps_load(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def loader():
            calls.append(1)
            started.set()
            await release.wait()
            return "shared"

        cancelled_waiter = asyncio.create_task(cache.get_or_load("k", loader))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        other_waiter = asyncio.create_task(cache.get_or_load("k", loader))
        await asyncio.sleep(0)
        cancelled_waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_waiter
        release.set()
        self.assertEqual(await asyncio.wait_for(other_waiter, timeout=2.0), "shared")
        self.assertEqual(await cache.get_or_load("k", loader), "shared")
        self.assertEqual(len(calls), 1)

    async def test_cancelled_sole_waiter_caches(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = asyncio.Event()
        release = asyncio.Event()
        returned = asyncio.Event()
        calls = []

        async def loader():
            calls.append(1)
            started.set()
            await release.wait()
            returned.set()
            return "value"

        waiter = asyncio.create_task(cache.get_or_load("k", loader))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        release.set()
        await asyncio.wait_for(returned.wait(), timeout=2.0)
        for _ in range(100):
            await asyncio.sleep(0)

        async def unused():
            calls.append("unused")
            return "unused"

        self.assertEqual(await cache.get_or_load("k", unused), "value")
        self.assertEqual(calls, [1])

    async def test_exception_not_cached(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = asyncio.Event()

        async def failing():
            started.set()
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            await cache.get_or_load("k", failing)

        async def ok():
            return "good"

        self.assertEqual(await asyncio.wait_for(cache.get_or_load("k", ok), timeout=2.0), "good")

    async def test_no_background_tasks(self) -> None:
        clock = FakeClock()
        cache = make_cache(10.0, clock)
        started = asyncio.Event()
        release = asyncio.Event()
        before = set(asyncio.all_tasks())

        async def old():
            started.set()
            await release.wait()
            return "old"

        task = asyncio.create_task(cache.get_or_load("k", old))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await cache.invalidate("k")
        release.set()
        await asyncio.wait_for(task, timeout=2.0)
        leftover = [t for t in asyncio.all_tasks() if t not in before and not t.done()]
        for _ in range(100):
            if not leftover:
                break
            await asyncio.sleep(0)
            leftover = [t for t in asyncio.all_tasks() if t not in before and not t.done()]
        self.assertEqual(leftover, [], f"background tasks still running: {leftover}")


if __name__ == "__main__":
    unittest.main()
'''

_PROMPT_C4 = """В проекте есть модуль gencache с классом AsyncTTLCache.

Публичный API менять запрещено.

Класс обслуживает одновременные чтения одного key одним общим loader-ом,
умеет сбрасывать запись через invalidate и не должен воскрешать удалённые
записи.

Требования:

1. Первое обращение к отсутствующему key вызывает loader().
2. Результат хранится в кеше в течение TTL.
3. После окончания TTL loader вызывается снова.
4. invalidate удаляет закэшированное значение: следующий вызов снова
   вызывает loader.
5. Если invalidate пришёл, пока loader ещё не завершился, следующий
   get_or_load запускает новый loader, не дожидаясь старого.
6. Поздний результат старого loader не записывается в кеш и не
   перетирает значение нового поколения.
7. Отмена ожидающего waiter не отменяет общий loader и не блокирует key.
8. Если все waiters отменены, loader всё равно доводится до конца, а его
   результат кэшируется.
9. Исключение loader не кэшируется, а key освобождается для новой попытки.
10. Реализация не создаёт background tasks, живущих после завершения
    операций.

Порядок событий в тестах задаётся fake clock и Event-барьерами; sleep для
порядка не используется. Одновременно выполняется не более 8 операций и
используется не более 4 keys.

Изменяй только реализацию библиотеки.
Тесты изменять запрещено.

Верни ТОЛЬКО полный текст файла {target} целиком, в одном блоке кода.

Файл gencache/cache.py в рабочей копии:

```python
{stub}```"""


def _workspace_c4() -> dict[str, str]:
    return {
        "gencache/__init__.py": (
            '"""Async TTL cache with invalidation."""\n\n'
            "from .cache import AsyncTTLCache\n\n"
            '__all__ = ["AsyncTTLCache"]\n'
        ),
        C4_TARGET_FILE: _STUB_C4,
        "tests/test_cache.py": _PUBLIC_TEST_C4,
        "README.md": "# gencache\n\nAsync TTL cache with generation-safe invalidation.\n",
    }


def generate_c4(rng: random.Random, cfg: Any) -> core.TaskInstance:
    return core.TaskInstance(
        id="C-4",
        category="C",
        title="Python cache invalidation",
        prompt=_PROMPT_C4.format(target=C4_TARGET_FILE, stub=_STUB_C4),
        max_points=8.0,
        workspace=_workspace_c4(),
        oracle={
            "reference": _REFERENCE_C4,
            "solution": _REFERENCE_C4,
            "hidden_tests": _HIDDEN_TESTS_C4,
            "runner": _RUNNER,
            "target_file": C4_TARGET_FILE,
            "answer_file": C4_TARGET_FILE,
            "checks": [name for name, _ in CHECK_ORDER_C4],
        },
        meta={"scenario": "C-4"},
    )


def evaluate_c4(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    code = code_of(task, tr)
    if not code:
        for name, _ in CHECK_ORDER_C4:
            score.add(name, False, _POINTS_PER_CHECK, "ответ не содержит кода файла")
        score.notes.append("не удалось извлечь файл из ответа")
        return score

    sandbox = Path(tempfile.mkdtemp(prefix="dailybench-c4-"))
    repo = sandbox / "repo"
    repo.mkdir(parents=True)
    for rel, content in task.workspace.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo / C4_TARGET_FILE).write_text(code, encoding="utf-8")

    try:
        results = _run_hidden(
            sandbox,
            repo,
            tests_name="test_gencache_hidden.py",
            hidden=_HIDDEN_TESTS_C4,
            order=CHECK_ORDER_C4,
        )
        for name, _ in CHECK_ORDER_C4:
            passed, detail = results[name]
            score.add(name, passed, _POINTS_PER_CHECK, detail)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    if "class AsyncTTLCache" not in code or "def invalidate" not in code:
        score.notes.append("в ответе не найден класс AsyncTTLCache или метод invalidate")
    return score


# --------------------------------------------------------------------------- #
# Scenario C-7 (stretch) — library composition: base64 ZIP → CSV → SQLite
#
# Not another pure function: the candidate composes five pinned stdlib APIs
# (``base64``, ``zipfile``, ``csv``/``io.TextIOWrapper``, ``hashlib``,
# ``sqlite3`` plus ``os.replace``) and the hidden checks watch the observable
# result: rows and report on a valid archive, unchanged state on four declared
# failure classes, no leaked handles and the declared usage code. The oracle of
# every expected value is recomputed in the hidden test from the fixture CSVs.
# --------------------------------------------------------------------------- #

C7_TARGET_FILE = "bin/import.py"

CHECK_ORDER_C7 = (
    ("happy_rows", "test_imports_every_row"),
    ("idempotent_reimport", "test_reimport_is_idempotent"),
    ("report_content", "test_report_is_exact"),
    ("invalid_sha_atomic", "test_sha_mismatch_is_atomic"),
    ("path_traversal", "test_traversal_member_rejected"),
    ("header_mismatch", "test_header_mismatch_rejected"),
    ("bad_base64", "test_bad_base64_is_a_clean_failure"),
    ("resources_closed", "test_no_resource_leaks"),
    ("usage_code", "test_usage_exit_code"),
    ("pinned_apis", "test_pinned_apis_are_used"),
)

_C7_HEADER = "order_id,region,product,cents,status"

_C7_STUB = '''#!/usr/bin/env python3
"""CLI: import a checksummed export archive into the warehouse."""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''

_C7_REFERENCE = r'''#!/usr/bin/env python3
"""Import the checksummed export archive into the warehouse database."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import json
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
BUILD = ROOT / "build"
REPORTS = ROOT / "reports"
DB = BUILD / "warehouse.db"
MANIFEST = ROOT / "data" / "manifest.json"
SUMMARY = REPORTS / "summary.json"
STATUSES = ("paid", "refunded")

TABLE = """CREATE TABLE IF NOT EXISTS events (
    order_id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    cents INTEGER NOT NULL,
    status TEXT NOT NULL,
    batch TEXT NOT NULL
)"""


class Invalid(Exception):
    """The archive violates the declared format."""


def fail(message: str) -> int:
    print(f"invalid archive: {message}", file=sys.stderr)
    return 2


def usage(message: str) -> int:
    print(f"usage: {message}", file=sys.stderr)
    return 64


def columns() -> list[str]:
    text = (CONFIG / "columns.txt").read_text(encoding="utf-8")
    return [value for value in text.strip().split(",")]


def regions() -> set[str]:
    return set((CONFIG / "regions.txt").read_text(encoding="utf-8").split())


def entries() -> list[dict[str, object]]:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    batches = document.get("batches") if isinstance(document, dict) else None
    if not isinstance(batches, list) or not batches:
        raise Invalid("manifest")
    for batch in batches:
        if not isinstance(batch, dict):
            raise Invalid("manifest")
        if not isinstance(batch.get("member"), str) or not isinstance(batch.get("rows"), int):
            raise Invalid("manifest")
        if not isinstance(batch.get("sha256"), str):
            raise Invalid("manifest")
    return batches


def decode(path: Path) -> bytes:
    try:
        text = path.read_text(encoding="ascii")
        return base64.b64decode(text.strip(), validate=True)
    except (OSError, UnicodeDecodeError, binascii.Error, ValueError):
        raise Invalid("not base64") from None


def safe(name: str) -> bool:
    if not name or name != name.strip():
        return False
    piece = Path(name)
    return not piece.is_absolute() and ".." not in piece.parts


def plan(path: Path) -> list[tuple[str, list[tuple[int, str, str, int, str]]]]:
    """Read-only pass: the whole archive is validated before a row is written."""
    declared = entries()
    header = columns()
    allowed = regions()
    result: list[tuple[str, list[tuple[int, str, str, int, str]]]] = []
    with zipfile.ZipFile(io.BytesIO(decode(path))) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise Invalid(f"crc {corrupt}")
        names = [info.filename for info in archive.infolist()]
        if len(set(names)) != len(names):
            raise Invalid("duplicate member")
        for name in names:
            if not safe(name):
                raise Invalid(f"unsafe member {name}")
        listed = [str(batch["member"]) for batch in declared]
        for name in listed:
            if name not in names:
                raise Invalid(f"missing member {name}")
        for name in names:
            if name not in listed:
                raise Invalid(f"extra member {name}")
        for batch in declared:
            name = str(batch["member"])
            with archive.open(name) as member:
                with io.TextIOWrapper(member, encoding="utf-8", newline="") as stream:
                    text = stream.read()
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != batch["sha256"]:
                raise Invalid(f"sha256 {name}")
            rows: list[tuple[int, str, str, int, str]] = []
            reader = csv.reader(io.StringIO(text, newline=""))
            if next(reader, None) != header:
                raise Invalid(f"header {name}")
            for line, fields in enumerate(reader, start=2):
                if len(fields) != len(header):
                    raise Invalid(f"row {line} {name}")
                raw_id, region, product, raw_cents, status = fields
                if not raw_id.isdigit() or int(raw_id) <= 0:
                    raise Invalid(f"row {line} {name}")
                if region not in allowed or not product:
                    raise Invalid(f"row {line} {name}")
                if not raw_cents.isdigit() or int(raw_cents) <= 0:
                    raise Invalid(f"row {line} {name}")
                if status not in STATUSES:
                    raise Invalid(f"row {line} {name}")
                rows.append((int(raw_id), region, product, int(raw_cents), status))
            if len(rows) != batch["rows"]:
                raise Invalid(f"rows {name}")
            result.append((name, rows))
    return result


def summary(connection: sqlite3.Connection) -> dict[str, object]:
    count, total = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(cents), 0) FROM events"
    ).fetchone()
    by_region = [
        {"region": region, "rows": rows, "cents": cents}
        for region, rows, cents in connection.execute(
            "SELECT region, COUNT(*), SUM(cents) FROM events GROUP BY region ORDER BY region"
        )
    ]
    best = connection.execute(
        "SELECT product FROM events GROUP BY product"
        " ORDER BY SUM(cents) DESC, product ASC LIMIT 1"
    ).fetchone()
    return {
        "by_region": by_region,
        "rows": count,
        "top_product": best[0] if best else "",
        "total_cents": total,
    }


def publish(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    REPORTS.mkdir(exist_ok=True)
    temporary = SUMMARY.with_name(SUMMARY.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, SUMMARY)
    return text


def command_import(path: Path) -> int:
    try:
        batches = plan(path)
    except Invalid as exc:
        return fail(str(exc))
    BUILD.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB)
    try:
        with connection:
            connection.execute(TABLE)
            for name, rows in batches:
                connection.executemany(
                    "INSERT INTO events (order_id, region, product, cents, status, batch)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(order_id) DO UPDATE SET"
                    " region = excluded.region, product = excluded.product,"
                    " cents = excluded.cents, status = excluded.status, batch = excluded.batch",
                    [(order_id, region, product, cents, status, name)
                     for order_id, region, product, cents, status in rows],
                )
        payload = summary(connection)
    finally:
        connection.close()
    publish(payload)
    print(f"ok: rows={sum(len(rows) for _, rows in batches)}")
    return 0


def command_report() -> int:
    if not DB.exists():
        return fail("no database")
    connection = sqlite3.connect(DB)
    try:
        text = publish(summary(connection))
    finally:
        connection.close()
    print(text, end="")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        return usage("import <archive> | report")
    command, *rest = argv
    if command == "import":
        if len(rest) != 1:
            return usage("import <archive>")
        return command_import(Path(rest[0]))
    if command == "report":
        if rest:
            return usage("report")
        return command_report()
    return usage(f"unknown command {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''

_C7_HIDDEN = r'''"""Hidden acceptance tests for the archive import pipeline (C-7).

Every expectation about content is recomputed here from the fixture CSVs with
``csv`` alone; the failure variants are built by rewriting the archive in the
test. The candidate is loaded by path from a fresh copy of the working tree, so
the tests never trust the module's own helpers.
"""

import base64
import contextlib
import csv
import gc
import importlib.util
import io
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
HEADER = (REPO / "config" / "columns.txt").read_text(encoding="utf-8").strip()


def load(repo: Path):
    spec = importlib.util.spec_from_file_location("candidate_import", repo / "bin" / "import.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="c7-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.repo = self.root / "repo"
        shutil.copytree(REPO, self.repo, ignore=shutil.ignore_patterns("__pycache__"))
        self.archive = self.repo / "data" / "exports.zip.b64"
        self.manifest_path = self.repo / "data" / "manifest.json"
        self.db = self.repo / "build" / "warehouse.db"
        self.summary = self.repo / "reports" / "summary.json"
        self.module = load(self.repo)

    # -- fixture helpers ---------------------------------------------------- #

    def unpack(self):
        raw = base64.b64decode(self.archive.read_text(encoding="ascii"))
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return {name: archive.read(name) for name in archive.namelist()}

    def repack(self, members):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 1, 0, 0, 0))
                archive.writestr(info, data)
        self.archive.write_text(base64.b64encode(buffer.getvalue()).decode("ascii"), encoding="ascii")

    def manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, payload):
        self.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def rows(self):
        if not self.db.exists():
            return None
        connection = sqlite3.connect(self.db)
        try:
            return [
                tuple(row)
                for row in connection.execute(
                    "SELECT order_id, region, product, cents, status FROM events ORDER BY order_id"
                )
            ]
        finally:
            connection.close()

    def expected(self):
        members = self.unpack()
        result = {}
        for batch in self.manifest()["batches"]:
            reader = csv.reader(io.StringIO(members[batch["member"]].decode("utf-8"), newline=""))
            next(reader)
            for order_id, region, product, cents, status in reader:
                result[int(order_id)] = (int(order_id), region, product, int(cents), status)
        return result

    def expected_report(self):
        by_region = {}
        by_product = {}
        rows = list(self.expected().values())
        for _order_id, region, product, cents, _status in rows:
            bucket = by_region.setdefault(region, [0, 0])
            bucket[0] += 1
            bucket[1] += cents
            by_product[product] = by_product.get(product, 0) + cents
        return {
            "by_region": [
                {"region": region, "rows": by_region[region][0], "cents": by_region[region][1]}
                for region in sorted(by_region)
            ],
            "rows": len(rows),
            "top_product": min(by_product, key=lambda name: (-by_product[name], name)),
            "total_cents": sum(row[3] for row in rows),
        }

    def run_import(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = self.module.main(["import", str(self.archive)])
        return code, errors.getvalue()

    def snapshot(self):
        return {
            str(path.relative_to(self.repo)): path.read_bytes()
            for folder in ("config", "data", "docs")
            for path in sorted((self.repo / folder).rglob("*"))
            if path.is_file()
        }

    def mutate(self, mutate):
        """Rewrite the archive and the manifest from their own current content."""
        members = self.unpack()
        manifest = self.manifest()
        mutate(members, manifest)
        self.repack(members)
        self.write_manifest(manifest)

    # -- tests -------------------------------------------------------------- #

    def test_imports_every_row(self):
        before = self.snapshot()
        self.assertEqual(self.run_import()[0], 0)
        self.assertEqual(self.rows(), sorted(self.expected().values()))
        self.assertEqual(self.snapshot(), before)

    def test_reimport_is_idempotent(self):
        self.assertEqual(self.run_import()[0], 0)
        first = self.rows()
        self.assertEqual(self.run_import()[0], 0)
        self.assertEqual(self.rows(), first)

    def test_report_is_exact(self):
        self.assertEqual(self.run_import()[0], 0)
        expected = self.expected_report()
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8")), expected)
        canonical = json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
        self.assertEqual(self.summary.read_text(encoding="utf-8"), canonical)

    def test_sha_mismatch_is_atomic(self):
        self.assertEqual(self.run_import()[0], 0)
        before_rows, before_report = self.rows(), self.summary.read_text(encoding="utf-8")

        def flip(members, manifest):
            name = manifest["batches"][-1]["member"]
            text = members[name].decode("utf-8")
            head, _, last = text.rstrip("\n").rpartition("\n")
            status = "refunded" if last.endswith(",paid") else "paid"
            members[name] = (head + "\n" + last.rsplit(",", 1)[0] + "," + status + "\n").encode("utf-8")

        self.mutate(flip)
        code, errors = self.run_import()
        self.assertEqual(code, 2)
        self.assertTrue(errors.startswith("invalid archive:"), errors)
        self.assertEqual(self.rows(), before_rows)
        self.assertEqual(self.summary.read_text(encoding="utf-8"), before_report)

    def test_traversal_member_rejected(self):
        def escape(members, manifest):
            text = members[manifest["batches"][0]["member"]]
            import hashlib

            name = "../escape.csv"
            members[name] = text
            manifest["batches"].append(
                {"member": name, "rows": 1, "sha256": hashlib.sha256(text).hexdigest()}
            )

        self.mutate(escape)
        code, errors = self.run_import()
        self.assertEqual(code, 2)
        self.assertIn("unsafe", errors)

    def test_header_mismatch_rejected(self):
        def rename(members, manifest):
            import hashlib

            batch = manifest["batches"][0]
            name = batch["member"]
            data = members[name].replace(b"order_id,", b"id,", 1)
            members[name] = data
            batch["sha256"] = hashlib.sha256(data).hexdigest()

        self.mutate(rename)
        code, errors = self.run_import()
        self.assertEqual(code, 2)
        self.assertIn("header", errors)

    def test_bad_base64_is_a_clean_failure(self):
        self.archive.write_text("not base64!!\n", encoding="ascii")
        code, errors = self.run_import()
        self.assertEqual(code, 2)
        self.assertTrue(errors.startswith("invalid archive:"), errors)
        self.assertNotIn("Traceback", errors)
        self.assertIsNone(self.rows())

    def test_no_resource_leaks(self):
        def leaks():
            gc.collect()
            found = []
            for obj in list(gc.get_objects()):
                if isinstance(obj, zipfile.ZipFile):
                    if obj.fp is not None:
                        found.append("ZipFile")
                elif isinstance(obj, sqlite3.Connection):
                    try:
                        obj.execute("PRAGMA schema_version").fetchone()
                    except sqlite3.ProgrammingError:
                        continue
                    found.append("Connection")
            return found

        self.assertEqual(self.run_import()[0], 0)
        self.assertEqual(self.module.main(["report"]), 0)
        self.assertEqual(leaks(), [])
        self.archive.write_text("broken\n", encoding="ascii")
        self.assertEqual(self.run_import()[0], 2)
        self.assertEqual(leaks(), [])

    def test_usage_exit_code(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(self.module.main([]), 64)
            self.assertEqual(self.module.main(["frobnicate"]), 64)
            self.assertEqual(self.module.main(["import"]), 64)
        text = errors.getvalue()
        self.assertIn("usage:", text)
        self.assertNotIn("Traceback", text)

    def test_pinned_apis_are_used(self):
        source = (self.repo / "bin" / "import.py").read_text(encoding="utf-8")
        for needle in (
            "import zipfile",
            "import csv",
            "import sqlite3",
            "import base64",
            "import hashlib",
            "os.replace",
            'newline=""',
            "validate=True",
            "ZipFile",
        ):
            self.assertIn(needle, source)


if __name__ == "__main__":
    unittest.main()
'''

_C7_PROMPT = """Рабочая копия — конвейер приёма выгрузок. Нужен исполняемый файл {target}
(Python 3.12, только стандартная библиотека). У него две команды:

  import <archive>   принять архив в build/warehouse.db и обновить reports/summary.json
  report             пересчитать reports/summary.json по текущей БД и напечатать его

<archive> — путь к текстовому файлу с base64-представлением ZIP-архива;
data/exports.zip.b64 — предъявленный пример. data/manifest.json перечисляет
обязательные члены архива в порядке применения:
{{"batches": [{{"member": "<имя в архиве>", "rows": <число>, "sha256": "<hex>"}}]}}.

Каждый член — UTF-8 CSV с заголовком ровно из config/columns.txt
({header}), разделитель «,», концы строк LF; после заголовка ровно rows строк.
Допустимая запись: order_id — десятичное > 0, region из config/regions.txt,
product непустой, cents — десятичное > 0, status — paid или refunded.

Архив обязан содержать ровно перечисленные члены: отсутствующий или лишний
член, дубль имени, абсолютный путь или «..» в имени — ошибка. CRC каждого
члена и sha256 его содержимого должны сойтись с манифестом.

Порядок:

1. Сначала полностью проверить архив и все батчи, ничего не записывая.
2. Затем одной транзакцией применить все батчи по порядку манифеста;
   повторный order_id перезаписывается последней записью.
3. После закрытия транзакции атомарно (os.replace) заменить
   reports/summary.json.

На ошибочном входе build/warehouse.db не создаётся, а существующие БД и
reports/summary.json остаются побайтово прежними: код 2, в stderr одна строка
`invalid archive: <причина>`, без traceback. Неизвестная команда или неверное
число аргументов: код 64, в stderr строка `usage: ...`. Успешный import: код 0
и stdout `ok: rows=<применено>`.

Таблица: events(order_id INTEGER PRIMARY KEY, region TEXT NOT NULL,
product TEXT NOT NULL, cents INTEGER NOT NULL, status TEXT NOT NULL,
batch TEXT NOT NULL); создаётся при необходимости. Отчёт — компактный
канонический JSON (json.dumps(..., sort_keys=True, separators=(",", ":")) плюс
завершающий LF) с полями rows, total_cents, by_region (список
{{region, rows, cents}} по возрастанию region) и top_product (product с
наибольшей суммой cents, при равенстве — лексикографически меньший).

Открытые файлы и соединения обязаны закрываться до возврата из команды;
пути считать от расположения самого файла, не от cwd. Обязательные API
stdlib: zipfile.ZipFile/.open, csv.reader поверх io.TextIOWrapper(..., newline=""),
sqlite3, base64.b64decode(..., validate=True), hashlib.sha256, os.replace.
Файлы config/, data/ и docs/ менять запрещено.

Верни ТОЛЬКО полный текст файла {target} целиком, в одном блоке кода.

Файл {target} в рабочей копии:

```python
{stub}```"""


def _c7_rows(rng: random.Random, regions: list[str]) -> tuple[list, list]:
    ids = rng.sample(range(1000, 9999), 14)
    plan = [
        (ids[0], 700, "alpha"),
        (ids[1], 500, "alpha"),
        (ids[2], 300, "alpha"),
        (ids[3], 800, "beta"),
        (ids[4], 400, "beta"),
        (ids[5], 300, "beta"),
        (ids[6], 150, "gamma"),
        (ids[7], 90, "delta"),
        (ids[6], 200, "gamma"),
        (ids[7], 95, "delta"),
        (ids[8], 40, "epsilon"),
        (ids[9], 30, "gamma"),
        (ids[10], 20, "delta"),
        (ids[11], 10, "epsilon"),
    ]
    first, second = [], []
    for index, (order_id, cents, product) in enumerate(plan):
        row = (order_id, rng.choice(regions), product, cents, rng.choice(("paid", "refunded")))
        (first if index < 8 else second).append(row)
    return first, second


def _c7_csv(rows: list[tuple[int, str, str, int, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_C7_HEADER.split(","))
    writer.writerows(rows)
    return stream.getvalue()


def _c7_archive(batches: list[list[tuple[int, str, str, int, str]]]) -> tuple[str, str]:
    buffer = io.BytesIO()
    manifest: dict[str, list[dict[str, Any]]] = {"batches": []}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, rows in enumerate(batches, start=1):
            name = f"orders/batch_2026_10_{index:02d}.csv"
            data = _c7_csv(rows)
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 10, 1, 0, 0, 0)), data)
            manifest["batches"].append(
                {
                    "member": name,
                    "rows": len(rows),
                    "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
                }
            )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii") + "\n"
    return encoded, json.dumps(manifest, indent=2) + "\n"


def _c7_report(rows: list[tuple[int, str, str, int, str]]) -> dict[str, Any]:
    by_region: dict[str, list[int]] = {}
    by_product: dict[str, int] = {}
    for _, region, product, cents, _status in rows:
        bucket = by_region.setdefault(region, [0, 0])
        bucket[0] += 1
        bucket[1] += cents
        by_product[product] = by_product.get(product, 0) + cents
    return {
        "by_region": [
            {"region": region, "rows": by_region[region][0], "cents": by_region[region][1]}
            for region in sorted(by_region)
        ],
        "rows": len(rows),
        "top_product": min(by_product, key=lambda name: (-by_product[name], name)),
        "total_cents": sum(row[3] for row in rows),
    }


_DOCS = """# Формат выгрузки

Архив приходит base64-файлом, члены перечислены в `data/manifest.json`
вместе с числом строк и sha256. Заголовок CSV обязан совпадать с
`config/columns.txt`; регионы берутся из `config/regions.txt`.

Валидация выполняется целиком до первой записи, затем батчи применяются
одной транзакцией в порядке манифеста. Повторный `order_id` перезаписывается
последней записью. Отчёт публикуется `os.replace` только после коммита.
"""


def generate_c7(rng: random.Random, cfg: Any) -> core.TaskInstance:
    regions = ["north", "south", "east", "west"]
    first, second = _c7_rows(rng, regions)
    encoded, manifest = _c7_archive([first, second])
    final: dict[int, tuple[int, str, str, int, str]] = {}
    for row in first + second:
        final[row[0]] = row
    rows = [final[key] for key in sorted(final)]
    workspace = {
        C7_TARGET_FILE: _C7_STUB,
        "config/columns.txt": _C7_HEADER + "\n",
        "config/regions.txt": "\n".join(regions) + "\n",
        "data/manifest.json": manifest,
        "data/exports.zip.b64": encoded,
        "docs/format.md": _DOCS,
        "README.md": "# warehouse import\n\nПриём выгрузок в SQLite.\n",
    }
    return core.TaskInstance(
        id="C-7",
        category="C",
        title="Python archive import",
        prompt=_C7_PROMPT.format(target=C7_TARGET_FILE, header=_C7_HEADER, stub=_C7_STUB),
        max_points=8.0,
        workspace=workspace,
        oracle={
            "reference": _C7_REFERENCE,
            "solution": _C7_REFERENCE,
            "hidden_tests": _C7_HIDDEN,
            "runner": _RUNNER,
            "target_file": C7_TARGET_FILE,
            "answer_file": C7_TARGET_FILE,
            "checks": [name for name, _ in CHECK_ORDER_C7],
            "expected_rows": [list(row) for row in rows],
            "expected_report": _c7_report(rows),
        },
        meta={"scenario": "C-7"},
    )


def evaluate_c7(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    code = code_of(task, tr)
    if not code:
        for name, _ in CHECK_ORDER_C7:
            score.add(name, False, _POINTS_PER_CHECK, "ответ не содержит кода файла")
        score.notes.append("не удалось извлечь файл из ответа")
        return score

    sandbox = Path(tempfile.mkdtemp(prefix="dailybench-c7-"))
    repo = sandbox / "repo"
    repo.mkdir(parents=True)
    for rel, content in task.workspace.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo / C7_TARGET_FILE).write_text(code, encoding="utf-8")

    try:
        results = _run_hidden(
            sandbox,
            repo,
            tests_name="test_import_hidden.py",
            hidden=_C7_HIDDEN,
            order=CHECK_ORDER_C7,
        )
        for name, _ in CHECK_ORDER_C7:
            passed, detail = results[name]
            score.add(name, passed, _POINTS_PER_CHECK, detail)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    if "def main" not in code:
        score.notes.append("в ответе не найдена точка входа main(argv)")
    return score
