"""Scenario C-1 — Python ``AsyncTTLCache`` (10 hidden checks × 1 point).

The model receives a stub package and must return the complete corrected file.
Ten hidden checks cover the requirements from the specification: load-on-miss,
TTL hit, reload after expiry, single-flight for one key, parallel loading of
different keys, exceptions not cached, cancellation not wedging a key, ``None``
as a real value, hits not extending the TTL and no surviving background tasks.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .. import core, text

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

_RUNNER = '''"""Runs the hidden test module and reports one JSON line per test."""

import importlib.util
import json
import sys
import unittest


class Recorder(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.rows = []

    def addSuccess(self, test) -> None:
        self.rows.append((test.id(), True, ""))

    def addFailure(self, test, err) -> None:
        self.rows.append((test.id(), False, str(err[1])))

    def addError(self, test, err) -> None:
        self.rows.append((test.id(), False, str(err[1])))

    def addSkip(self, test, reason) -> None:
        self.rows.append((test.id(), False, f"skipped: {reason}"))


def main() -> int:
    repo, test_file = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location("hidden_ttl_tests", test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    recorder = Recorder()
    suite.run(recorder)
    for name, ok, detail in recorder.rows:
        print(json.dumps({"test": name, "ok": ok, "detail": detail[:400]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

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
        max_points=10.0,
        workspace=_workspace(),
        oracle={
            "reference": _REFERENCE,
            "solution": _REFERENCE,
            "hidden_tests": _HIDDEN_TESTS,
            "runner": _RUNNER,
            "target_file": TARGET_FILE,
            "checks": [name for name, _ in CHECK_ORDER],
        },
        meta={"scenario": "C-1"},
    )


def extract_code(reply: str) -> str | None:
    """Full-file answer: a fenced block when present, else the whole reply."""
    block = text.first_fenced_block(reply)
    if block:
        return block
    candidate = text.strip_fence(reply)[0]
    return candidate or None


def _run_hidden(sandbox: Path, repo: Path) -> dict[str, tuple[bool, str]]:
    runner = sandbox / "run_hidden.py"
    tests = sandbox / "test_ttlcache_hidden.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    tests.write_text(_HIDDEN_TESTS, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(runner), str(repo), str(tests)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            cwd=str(sandbox),
        )
    except subprocess.TimeoutExpired:
        return {name: (False, "hidden tests timed out (180s)") for name, _ in CHECK_ORDER}
    except OSError as exc:
        return {name: (False, f"cannot run hidden tests: {exc}") for name, _ in CHECK_ORDER}

    results: dict[str, tuple[bool, str]] = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        results[str(row.get("test", ""))] = (bool(row.get("ok")), str(row.get("detail", "")))

    mapped: dict[str, tuple[bool, str]] = {}
    for name, test_name in CHECK_ORDER:
        match = next((v for k, v in results.items() if k.endswith(test_name)), None)
        if match is None:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = detail[-1] if detail else "нет результата"
            mapped[name] = (False, f"тест {test_name} не выполнился: {tail[:200]}")
        else:
            mapped[name] = match
    return mapped


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    code = extract_code(tr.final_text)
    if not code:
        for name, _ in CHECK_ORDER:
            score.add(name, False, 1.0, "ответ не содержит кода файла")
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

    results = _run_hidden(sandbox, repo)
    for name, _ in CHECK_ORDER:
        passed, detail = results[name]
        score.add(name, passed, 1.0, detail)

    kept_api = "class AsyncTTLCache" in code and "get_or_load" in code
    if not kept_api:
        score.notes.append("в ответе не найден публичный класс/метод AsyncTTLCache.get_or_load")
    return score
