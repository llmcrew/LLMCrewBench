"""Category J — code review, 5 points over two baseline scenarios.

J-1 (3.0) — an off-by-one boundary defect in a pure window helper: it returns
one element more than asked whenever ``count`` is smaller than the input, and
the public test never exercises that case.
J-2 (2.0) — a defect that only shows under repetition: the module cache is keyed
on half of the input, so a second call with another root silently returns the
first result. A single call looks perfect; the state is what is wrong.

Two profile scenarios reuse the same contract: J-3 (pilot) is a tenant filter
applied after the page limit, so a page silently misses allowed rows, reports
foreign ones and can stop pagination early; J-4 (stretch) is deduplication on
the payload instead of the idempotency key, so a new operation is swallowed and
a retry from another user is charged twice.

Both scenarios ship a small repository whose public test file is green, and a
one-line defect. The model must return strict JSON with the file, the defect
line and the complete corrected file; scoring is behavioural — the returned
``fix`` is written into a temporary copy of the repository and the hidden tests
run there, so a plausible-sounding answer that does not actually repair the
behaviour earns nothing. The reported line is derived from the fixture itself,
never hard-coded.

The one-line contract is scored on its own, before the location: the returned
file must differ from the buggy one and must replace exactly one substantive
line, without adding or dropping lines. Reordering unchanged lines is allowed
(only the multiset of lines is compared), so a pure move is not a rewrite.
Appending a helper function keeps the hidden tests green but breaks the shape,
and a shape failure also costs the location check.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .. import core, text

IDS = ("J-1", "J-2")
PILOT_IDS = ("J-3",)
"""Scenario of the pilot profile: the multi-tenant pagination review."""
ALL_IDS = IDS + ("J-3", "J-4")
"""Scenarios of the stretch profile: the baseline plus both extensions."""
_POINTS = {"J-1": 3.0, "J-2": 2.0, "J-3": 2.0, "J-4": 2.0}

#: Per-task shares of the budget for the shape and the location; the repair gets
#: the remainder, so the three checks always sum to ``max_points`` even when a
#: profile redistributes the category total (J-3/J-4 average 1.25 in stretch).
#: Finding the defect is worth the same in both baseline scenarios; a review
#: that mislocates the line still gets paid for a fix that really works.
_SHARES = {
    "J-1": (1 / 3, 1 / 3),
    "J-2": (0.25, 0.5),
    "J-3": (1 / 3, 1 / 3),
    "J-4": (1 / 3, 1 / 3),
}

#: A wedged interpreter must not stall the whole run.
_HIDDEN_TIMEOUT_S = 60


# --------------------------------------------------------------------------- #
# Fixture templates
# --------------------------------------------------------------------------- #
# Every template is filled with ``_fill`` (plain ``str.replace``), so the braces
# of the Python code inside stay literal and no template needs escaping.

_J1_HELPER = '''"""Human-readable labels shared by the pipeline steps."""

from __future__ import annotations

_PREFIX = "@PREFIX@"


def label(name: str, value: object) -> str:
    """Render ``name=value`` with the service prefix."""
    return f"{_PREFIX}{name}={value}"
'''

#: ``@RETURN@`` is the slice expression: the buggy and the fixed file differ in
#: that one line only.
_J1_WINDOW = '''"""Sliding-window helpers for the metrics pipeline."""

from __future__ import annotations


def @FUNC@(values: list, count: int) -> list:
    """Return the last ``count`` values of ``values``, oldest first.

    ``count`` is clamped to the length of ``values``; a non-positive ``count``
    yields an empty list.
    """
    if count <= 0:
        return []
    return values[@RETURN@]
'''

_J1_PUBLIC_TEST = '''"""Public test: the documented behaviour of @FUNC@."""

import unittest

from @DOTTED@ import @FUNC@

SAMPLE = [@SAMPLE@]


class WindowTests(unittest.TestCase):
    def test_count_equal_to_length_keeps_everything(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, len(SAMPLE)), SAMPLE)

    def test_count_above_length_keeps_everything(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, 99), SAMPLE)

    def test_zero_count_is_empty(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, 0), [])

    def test_negative_count_is_empty(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, -3), [])

    def test_empty_input_is_empty(self) -> None:
        self.assertEqual(@FUNC@([], 4), [])


if __name__ == "__main__":
    unittest.main()
'''

_J1_HIDDEN_TESTS = '''"""Hidden acceptance tests for @DOTTED@.@FUNC@ (not shown to the model)."""

import unittest

from @DOTTED@ import @FUNC@


class BoundaryTests(unittest.TestCase):
    def test_returns_exactly_count_values(self) -> None:
        self.assertEqual(@FUNC@([1, 2, 3, 4, 5], 2), [4, 5])
        self.assertEqual(@FUNC@([1, 2, 3, 4, 5], 3), [3, 4, 5])
        self.assertEqual(@FUNC@([1, 2, 3, 4, 5], 4), [2, 3, 4, 5])

    def test_count_one_returns_the_last_value(self) -> None:
        self.assertEqual(@FUNC@([7, 8, 9], 1), [9])

    def test_count_above_length_keeps_everything(self) -> None:
        self.assertEqual(@FUNC@([1, 2, 3], 10), [1, 2, 3])

    def test_non_positive_count_is_empty(self) -> None:
        self.assertEqual(@FUNC@([1, 2, 3], 0), [])
        self.assertEqual(@FUNC@([1, 2, 3], -1), [])

    def test_empty_input_is_empty(self) -> None:
        self.assertEqual(@FUNC@([], 5), [])

    def test_duplicates_are_kept(self) -> None:
        self.assertEqual(@FUNC@([5, 5, 5, 5], 2), [5, 5])

    def test_input_is_not_mutated(self) -> None:
        values = [1, 2, 3]
        @FUNC@(values, 2)
        self.assertEqual(values, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
'''

_J2_DECOY = '''"""Static ingest configuration."""

from __future__ import annotations

DEFAULTS = {
    "shard_size": @SHARD@,
    "retries": 2,
}


def shard_size() -> int:
    """Number of records read per shard."""
    return DEFAULTS["shard_size"]
'''

#: ``@KEY@`` is the cache key: the buggy and the fixed file differ in that one
#: line only.
_J2_PATHS = '''"""Path joining helpers for the ingest service."""

from __future__ import annotations

_CACHE: dict = {}


def @FUNC@(root: str, name: str) -> str:
    """Join ``root`` and ``name`` with exactly one slash between them.

    Repeated calls with the same arguments are served from a small cache.
    """
    key = @KEY@
    if key in _CACHE:
        return _CACHE[key]
    joined = root.rstrip("/") + "/" + name.lstrip("/")
    _CACHE[key] = joined
    return joined
'''

_J2_PUBLIC_TEST = '''"""Public test: the documented behaviour of @FUNC@."""

import unittest

from @DOTTED@ import @FUNC@


class PathTests(unittest.TestCase):
    def test_join_adds_one_slash(self) -> None:
        self.assertEqual(@FUNC@("/srv/data", "report.csv"), "/srv/data/report.csv")

    def test_trailing_slashes_are_collapsed(self) -> None:
        self.assertEqual(@FUNC@("/srv/", "/tmp/out.txt"), "/srv/tmp/out.txt")

    def test_empty_name_keeps_the_root(self) -> None:
        self.assertEqual(@FUNC@("/srv/data", ""), "/srv/data/")

    def test_repeated_call_is_stable(self) -> None:
        first = @FUNC@("/srv/data", "report.csv")
        second = @FUNC@("/srv/data", "report.csv")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
'''

#: The names here are deliberately disjoint from the public test's names: with
#: the shared module cache, reusing a name would let the public run (or the
#: hidden run) change the other one's verdict on the buggy file, and the two
#: facts the fixture asserts — public green, hidden red — must both hold.
_J2_HIDDEN_TESTS = '''"""Hidden acceptance tests for @DOTTED@.@FUNC@ (not shown to the model)."""

import unittest

from @DOTTED@ import @FUNC@


class CacheTests(unittest.TestCase):
    def test_same_name_under_different_roots(self) -> None:
        self.assertEqual(@FUNC@("/srv/a", "part-1.csv"), "/srv/a/part-1.csv")
        self.assertEqual(@FUNC@("/srv/b", "part-1.csv"), "/srv/b/part-1.csv")

    def test_cache_does_not_cross_roots(self) -> None:
        self.assertEqual(@FUNC@("/mnt/one", "shard.bin"), "/mnt/one/shard.bin")
        self.assertEqual(@FUNC@("/mnt/two", "shard.bin"), "/mnt/two/shard.bin")

    def test_trailing_slashes_per_root(self) -> None:
        self.assertEqual(@FUNC@("/srv/x/", "tail.log"), "/srv/x/tail.log")
        self.assertEqual(@FUNC@("/srv/y/", "tail.log"), "/srv/y/tail.log")

    def test_repeated_call_is_stable(self) -> None:
        first = @FUNC@("/opt/app", "conf.ini")
        second = @FUNC@("/opt/app", "conf.ini")
        self.assertEqual(first, "/opt/app/conf.ini")
        self.assertEqual(second, "/opt/app/conf.ini")

    def test_name_with_leading_slash(self) -> None:
        self.assertEqual(@FUNC@("/opt/app", "/deep/file"), "/opt/app/deep/file")
        self.assertEqual(@FUNC@("/other/app", "/deep/file"), "/other/app/deep/file")


if __name__ == "__main__":
    unittest.main()
'''

_J3_PAGING = '''"""Tenant-scoped cursor pagination over the shared audit log."""

from __future__ import annotations

_TABLE = (
@ROWS@)

DEFAULT_LIMIT = @LIMIT@


def @FUNC@(tenant: str, cursor: int = 0, limit: int = DEFAULT_LIMIT) -> dict:
    """Return the next page of ``tenant`` rows after ``cursor``.

    Rows are ordered by ``id``; ``cursor`` is the id of the last row a caller
    saw, and ``0`` starts at the beginning. At most ``limit`` rows are
    returned, ``next_cursor`` is the id to pass next (``None`` on the last
    page), and rows of other tenants are never returned.
    """
    if limit <= 0:
        return {"items": [], "next_cursor": None}
    rows = [row for row in _TABLE if row["id"] > cursor]
    window = rows[:limit]
    visible = [row for row in window if row["tenant"] == tenant]
    nxt = visible[-1]["id"] if len(visible) == limit else None
    return {"items": [dict(row) for row in window], "next_cursor": nxt}
'''

_J3_PUBLIC_TEST = '''"""Public test: the documented behaviour of @FUNC@."""

import unittest

from @DOTTED@ import DEFAULT_LIMIT, @FUNC@

TENANT = "@US@"


class PageTests(unittest.TestCase):
    def test_first_page_returns_rows_of_the_tenant(self) -> None:
        result = @FUNC@(TENANT)
        self.assertEqual([row["id"] for row in result["items"]], [@FIRST_IDS@])
        self.assertEqual(result["next_cursor"], @FIRST_CURSOR@)

    def test_first_page_stays_within_the_default_limit(self) -> None:
        self.assertLessEqual(len(@FUNC@(TENANT)["items"]), DEFAULT_LIMIT)

    def test_non_positive_limit_is_empty(self) -> None:
        self.assertEqual(@FUNC@(TENANT, 0, 0), {"items": [], "next_cursor": None})

    def test_unknown_tenant_and_zero_limit(self) -> None:
        self.assertEqual(@FUNC@("nobody", 0, 0)["items"], [])


if __name__ == "__main__":
    unittest.main()
'''

_J3_HIDDEN_TESTS = '''"""Hidden acceptance tests for @DOTTED@.@FUNC@ (not shown to the model).

The oracle is the exact set of rows ``@US@`` may read: no foreign row may
appear, no allowed row may be skipped, and the cursor must move past exactly
the returned rows.
"""

import unittest

from @DOTTED@ import @FUNC@

TENANT = "@US@"
ALLOWED = [@ALL_IDS@]


def _ids(result: dict) -> list:
    return [row["id"] for row in result["items"]]


class AccessTests(unittest.TestCase):
    def test_second_page_skips_the_interleaved_foreign_row(self) -> None:
        result = @FUNC@(TENANT, 2, 2)
        self.assertEqual(_ids(result), [@SECOND_IDS@])
        self.assertEqual(result["next_cursor"], @SECOND_CURSOR@)

    def test_whole_tenant_fits_one_large_page(self) -> None:
        result = @FUNC@(TENANT, 0, 100)
        self.assertEqual(_ids(result), ALLOWED)
        self.assertIsNone(result["next_cursor"])

    def test_no_foreign_row_leaks(self) -> None:
        cursor: object = 0
        seen = 0
        while cursor is not None and seen < 20:
            result = @FUNC@(TENANT, cursor, 2)
            for row in result["items"]:
                self.assertEqual(row["tenant"], TENANT, row)
            seen += 1
            cursor = result["next_cursor"]
        self.assertLess(seen, 20)

    def test_pages_walk_the_allowed_rows_once(self) -> None:
        cursor: object = 0
        collected = []
        while cursor is not None:
            result = @FUNC@(TENANT, cursor, 2)
            collected.extend(_ids(result))
            cursor = result["next_cursor"]
        self.assertEqual(collected, ALLOWED)

    def test_last_page_has_no_next_cursor(self) -> None:
        last = ALLOWED[-1]
        result = @FUNC@(TENANT, last - 1, 5)
        self.assertEqual(_ids(result), [last])
        self.assertIsNone(result["next_cursor"])

    def test_cursor_past_the_end_is_empty(self) -> None:
        self.assertEqual(@FUNC@(TENANT, 999, 3), {"items": [], "next_cursor": None})

    def test_unknown_tenant_leaks_nothing(self) -> None:
        self.assertEqual(@FUNC@("nobody", 0, 2), {"items": [], "next_cursor": None})

    def test_returned_rows_are_copies(self) -> None:
        first = @FUNC@(TENANT, 0, 1)
        first["items"][0]["tenant"] = "mutated"
        self.assertEqual(@FUNC@(TENANT, 0, 1)["items"][0]["tenant"], TENANT)


if __name__ == "__main__":
    unittest.main()
'''

_J4_DELIVERY = '''"""Exactly-once delivery for the billing webhook queue."""

from __future__ import annotations


class Timeout(Exception):
    """The transport gave up before the receiver acknowledged the delivery."""


_SEEN: dict = {}
_EFFECTS: list = []


def reset() -> None:
    """Forget every delivered key and every recorded effect."""
    _SEEN.clear()
    _EFFECTS.clear()


def effects() -> list:
    """Copy of the effects that really happened, oldest first."""
    return [dict(effect) for effect in _EFFECTS]


def @FUNC@(key: str, user: str, amount: int) -> dict:
    """Deliver ``amount`` to ``user`` under the idempotency ``key``.

    A retry of an already acknowledged key returns the very same receipt and
    must not charge again. A new key charges once, even for the same user and
    amount.
    """
    token = (user, amount)
    if token in _SEEN:
        return dict(_SEEN[token])
    _EFFECTS.append({"user": user, "amount": amount})
    receipt = {"status": "charged", "receipt": len(_EFFECTS)}
    _SEEN[token] = receipt
    return dict(receipt)


def send(key: str, user: str, amount: int, transport=@FUNC@) -> dict:
    """Deliver through ``transport``, retrying once after a transport timeout.

    The retry reuses the key: a lost acknowledgement must not double the
    effect.
    """
    try:
        return transport(key, user, amount)
    except Timeout:
        return transport(key, user, amount)
'''

_J4_PUBLIC_TEST = '''"""Public test: the documented behaviour of @FUNC@."""

import unittest

from @DOTTED@ import Timeout, effects, reset, @FUNC@, send

KEY = "@KEY@"


class DeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset()

    def test_ordinary_delivery_charges_once(self) -> None:
        receipt = @FUNC@(KEY, "alice", @AMOUNT@)
        self.assertEqual(receipt["status"], "charged")
        self.assertEqual(len(effects()), 1)
        self.assertEqual(effects()[0]["amount"], @AMOUNT@)

    def test_same_key_and_payload_returns_the_same_receipt(self) -> None:
        first = @FUNC@(KEY, "alice", @AMOUNT@)
        second = @FUNC@(KEY, "alice", @AMOUNT@)
        self.assertEqual(first, second)
        self.assertEqual(len(effects()), 1)

    def test_timeout_retry_is_delivered_once(self) -> None:
        calls = []

        def flaky(key: str, user: str, amount: int) -> dict:
            calls.append(key)
            if len(calls) == 1:
                raise Timeout()
            return @FUNC@(key, user, amount)

        receipt = send(KEY, "alice", @AMOUNT@, flaky)
        self.assertEqual(receipt["status"], "charged")
        self.assertEqual(calls, [KEY, KEY])
        self.assertEqual(len(effects()), 1)

    def test_reset_clears_the_counter(self) -> None:
        @FUNC@(KEY, "alice", @AMOUNT@)
        reset()
        self.assertEqual(effects(), [])


if __name__ == "__main__":
    unittest.main()
'''

_J4_HIDDEN_TESTS = '''"""Hidden acceptance tests for @DOTTED@.@FUNC@ (not shown to the model).

The key identifies the operation: the same key is one effect no matter how
often or by whom it is retried, a new key is one new effect, and receipts stay
stable so a retry can return the first answer.
"""

import unittest

from @DOTTED@ import Timeout, effects, reset, @FUNC@, send

KEY = "@KEY@"
OTHER = "@OTHER_KEY@"


class IdempotencyTests(unittest.TestCase):
    def setUp(self) -> None:
        reset()

    def test_same_key_is_charged_once(self) -> None:
        first = @FUNC@(KEY, "alice", @AMOUNT@)
        second = @FUNC@(KEY, "alice", @AMOUNT@)
        self.assertEqual(len(effects()), 1)
        self.assertEqual(first, second)

    def test_timeout_retry_with_the_same_key_charges_once(self) -> None:
        calls = []

        def flaky(key: str, user: str, amount: int) -> dict:
            calls.append(key)
            if len(calls) == 1:
                raise Timeout()
            return @FUNC@(key, user, amount)

        receipt = send(KEY, "alice", @AMOUNT@, flaky)
        self.assertEqual(receipt["status"], "charged")
        self.assertEqual(len(effects()), 1)

    def test_new_key_charges_again_for_the_same_payload(self) -> None:
        first = @FUNC@(KEY, "alice", @AMOUNT@)
        second = @FUNC@(OTHER, "alice", @AMOUNT@)
        self.assertEqual(len(effects()), 2)
        self.assertEqual([effect["user"] for effect in effects()], ["alice", "alice"])
        self.assertNotEqual(first["receipt"], second["receipt"])

    def test_same_key_from_two_users_is_one_operation(self) -> None:
        first = @FUNC@(KEY, "alice", @AMOUNT@)
        second = @FUNC@(KEY, "bob", @AMOUNT@)
        self.assertEqual(len(effects()), 1)
        self.assertEqual(first, second)

    def test_two_users_with_different_keys_are_two_charges(self) -> None:
        @FUNC@(KEY, "alice", @AMOUNT@)
        @FUNC@(OTHER, "bob", @AMOUNT@)
        self.assertEqual([effect["user"] for effect in effects()], ["alice", "bob"])

    def test_retry_after_success_returns_the_stored_receipt(self) -> None:
        first = @FUNC@(KEY, "alice", @AMOUNT@)
        for _ in range(3):
            self.assertEqual(@FUNC@(KEY, "alice", @AMOUNT@), first)
        self.assertEqual(len(effects()), 1)

    def test_receipt_and_effect_copies_are_independent(self) -> None:
        receipt = @FUNC@(KEY, "alice", @AMOUNT@)
        receipt["status"] = "mutated"
        effects()[0]["user"] = "mutated"
        again = @FUNC@(KEY, "alice", @AMOUNT@)
        self.assertEqual(again["status"], "charged")
        self.assertEqual(effects()[0]["user"], "alice")

    def test_new_key_on_the_first_attempt_charges_once(self) -> None:
        @FUNC@(OTHER, "bob", @AMOUNT@ + 1)
        self.assertEqual(len(effects()), 1)
        self.assertEqual(effects()[0]["amount"], @AMOUNT@ + 1)


if __name__ == "__main__":
    unittest.main()
'''

_PROMPT = """В рабочей копии репозитория ровно один файл содержит дефект: поведение
расходится с тем, что обещает docstring, и с намерением публичных тестов.
Дефект укладывается ровно в одну строку — исправлять нужно только её.

Файлы рабочей копии:

{files}
Публичный тест {public_test} описывает ожидаемое поведение и сейчас проходит:
дефектный случай он не покрывает.

Верни ТОЛЬКО JSON, без markdown-обёртки и текста вокруг:

{{"file": "<путь файла с дефектом>", "line": <номер строки с дефектом>, "fix": "<полный исправленный текст файла>"}}

* "file" — путь относительно корня рабочей копии, ровно как в списке выше.
* "line" — номер дефектной строки в этом файле, нумерация с 1.
* "fix" — полный текст файла после исправления (не diff, без многоточий и сокращений).
"""

_CHILD_RUNNER = '''"""Child: imports the untrusted candidate and runs the given test modules.

The child picks its own report file with ``tempfile`` (its path never exists
before the import) and announces it in a control file that the supervisor reads
first-line-only. After every module ran, an ``eof`` row carrying the nonce that
travelled over stdin is appended: rows alone are not proof that the run
finished, because the candidate may write the file itself and call
``os._exit(0)``.

The grader state (``Recorder`` methods, ``TestCase.assert*``) is snapshotted
*before* any candidate import and restored before each module runs, so a fix
that patched ``assert*`` cannot silence the hidden failures (C4).
"""

import builtins
import importlib.util
import json
import os
import sys
import tempfile
import unittest

_open = builtins.open
_dumps = json.dumps


def _snapshot() -> dict:
    case = unittest.TestCase
    state = {"result": unittest.TestResult.__dict__.copy()}
    state["case"] = {name: getattr(case, name) for name in vars(case)
                     if callable(getattr(case, name, None))}
    return state


def _restore(state: dict) -> None:
    for name, value in state["result"].items():
        try:
            setattr(unittest.TestResult, name, value)
        except (AttributeError, TypeError):
            pass
    for name, value in state["case"].items():
        try:
            setattr(unittest.TestCase, name, value)
        except (AttributeError, TypeError):
            pass


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


def run_module(label: str, path: str, state: dict, rows: list) -> None:
    _restore(state)
    spec = importlib.util.spec_from_file_location(f"dailybench_review_{label}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # the untrusted candidate runs here
    _restore(state)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    recorder = Recorder()
    recorder.rows = []  # candidate import may have re-bound the class; fresh run
    suite.run(recorder)
    rows.extend((label, name, ok, detail) for name, ok, detail in recorder.rows)


def main() -> None:
    repo, control, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
    nonce = sys.stdin.read().strip()
    descriptor, report_path = tempfile.mkstemp(suffix=".jsonl", prefix="dailybench-review-")
    os.close(descriptor)
    with _open(control, "w", encoding="utf-8") as handshake:
        handshake.write(report_path + "\\n")
        handshake.flush()
        os.fsync(handshake.fileno())
    rows: "list[tuple[str, str, bool, str]]" = []
    try:
        sys.path.insert(0, repo)
        state = _snapshot()
        for label, path in zip(pairs[0::2], pairs[1::2]):
            try:
                run_module(label, path, state, rows)
            except BaseException as exc:  # a broken import is a failed test, not a crash
                _restore(state)
                rows.append((label, f"{label}.<import>", False, f"{type(exc).__name__}: {exc}"))
    except BaseException as exc:
        rows.append(("*", "*", False, f"{type(exc).__name__}: {exc}"))
    payload = "".join(
        _dumps({"module": label, "test": name, "ok": ok, "detail": str(detail)[:400]}, ensure_ascii=False) + "\\n"
        for label, name, ok, detail in rows
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
#: from ``_run_modules``, so a wedged interpreter is still reported as a failure.
_CHILD_TIMEOUT_S = 50

_RUNNER_TEMPLATE = '''"""Supervisor: the only process that writes the final verdict.

The candidate is never imported here. It runs in a child that is free to print,
forge a ``REPORT`` line from ``atexit`` and call ``os._exit`` — none of that
reaches the final report. The supervisor

* creates the final report path itself and passes it in argv (the child is only
  told a throwaway path of its own choosing, announced through a control file);
* requires a zero exit status and the nonce-marked ``eof`` row the child appends
  after the whole suite;
* derives the expected ``label.Class.test`` names from the supplied modules and
  requires every one of them, so a missing result is a failure rather than a
  silently shorter report; duplicates are last-wins;
* writes the final report from rows it vetted. stdout is never parsed.
"""

import ast
import json
import os
import secrets
import subprocess
import sys

_CHILD = __CHILD__

CHILD_TIMEOUT = __TIMEOUT__
MODULE_PREFIX = "dailybench_review_"


def _write(path: str, payload: str) -> None:
    with open(path, "w", encoding="utf-8") as report:
        report.write(payload)
        report.flush()
        os.fsync(report.fileno())


def _tail(*streams: str) -> str:
    for stream in streams:
        lines = (stream or "").strip().splitlines()
        if lines:
            return ": " + lines[-1][-300:]
    return ""


def _expected(label: str, path: str) -> "list[str]":
    """Test ids the module declares; the supervisor, not the child, names them."""
    try:
        with open(path, encoding="utf-8", errors="replace") as source:
            tree = ast.parse(source.read())
    except (OSError, SyntaxError, ValueError):
        return []
    names = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name.startswith("test"):
                names.append(f"{MODULE_PREFIX}{label}.{node.name}.{child.name}")
    return names


def _verdict(final_report: str, expected: "list[tuple[str, str]]", detail: str) -> int:
    _write(
        final_report,
        "".join(
            json.dumps({"module": label, "test": test, "ok": False, "detail": detail[:400]},
                       ensure_ascii=False) + "\\n"
            for label, test in expected
        ) + (json.dumps({"eof": ""}) + "\\n"),
    )
    return 0


def main() -> int:
    repo, final_report, control = sys.argv[1:4]
    pairs = sys.argv[4:]
    expected: "list[tuple[str, str]]" = []
    for label, path in zip(pairs[0::2], pairs[1::2]):
        expected += [(label, name) for name in _expected(label, path)]

    child = os.path.join(os.path.dirname(os.path.abspath(final_report)), "_review_child.py")
    _write(child, _CHILD)
    # The final report exists before anything untrusted starts and is only ever
    # written by this process, so nothing the candidate does can preempt it.
    _write(final_report, "")
    nonce = secrets.token_hex(16)
    try:
        proc = subprocess.run(
            [sys.executable, child, repo, control] + pairs,
            input=nonce,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHILD_TIMEOUT,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return _verdict(final_report, expected, f"тесты не уложились в {CHILD_TIMEOUT} с")
    except OSError as exc:
        return _verdict(final_report, expected, f"раннер не запустился: {exc}")

    failure = ""
    if proc.returncode != 0:
        failure = f"раннер завершился с кодом {proc.returncode}{_tail(proc.stderr, proc.stdout)}"

    try:
        with open(control, encoding="utf-8", errors="replace") as handshake:
            child_report = handshake.readline().strip()
    except OSError as exc:
        return _verdict(final_report, expected, failure or f"путь отчёта не объявлен: {exc}")

    rows = []
    marker = ""
    try:
        with open(child_report, encoding="utf-8", errors="replace") as report:
            blob = report.read()
    except OSError as exc:
        return _verdict(final_report, expected, failure or f"отчёт тестов не создан: {exc}")
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

    if not failure and marker != nonce:
        # Rows exist but the child never reached its own write: the candidate
        # wrote the file itself and called ``os._exit(0)`` from the import hook.
        failure = "раннер не дошёл до конца: маркера завершения нет"
    if failure:
        return _verdict(final_report, expected, failure)

    vetted: "dict[tuple[str, str], dict]" = {}
    for row in rows:
        vetted[(str(row.get("module", "")), str(row.get("test", "")))] = row
    imported = {
        label: str(row.get("detail", ""))
        for (label, test), row in vetted.items()
        if test == f"{label}.<import>"
    }
    payload = ""
    for label, test in expected:
        row = vetted.get((label, test))
        if row is None:
            detail = imported.get(label) or f"тест {test} не выполнился"
            row = {"module": label, "test": test, "ok": False, "detail": detail}
        payload += json.dumps(row, ensure_ascii=False) + "\\n"
    payload += json.dumps({"eof": nonce}) + "\\n"
    _write(final_report, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_RUNNER = _RUNNER_TEMPLATE.replace("__CHILD__", repr(_CHILD_RUNNER)).replace(
    "__TIMEOUT__", str(_CHILD_TIMEOUT_S)
)


# --------------------------------------------------------------------------- #
# Template filling and fixture arithmetic
# --------------------------------------------------------------------------- #


def _fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace(f"@{key}@", value)
    return template


def _render_files(files: dict[str, str]) -> str:
    blocks = []
    for rel, content in files.items():
        lang = "markdown" if rel.endswith(".md") else "python"
        blocks.append(f"### {rel}\n\n```{lang}\n{content}```\n")
    return "\n".join(blocks)


def _defect_line(source: str, marker: str) -> int:
    """Line number of the defect, taken from the fixture, never hard-coded."""
    hits = [i for i, line in enumerate(source.splitlines(), 1) if line == marker]
    assert len(hits) == 1, f"маркер дефекта {marker!r} встречается {len(hits)} раз"
    return hits[0]


def _assert_single_line_fix(task_id: str, buggy: str, reference: str, line: int) -> None:
    """The reported line really is the defect line: the only one that changes."""
    before, after = buggy.splitlines(), reference.splitlines()
    assert len(before) == len(after), f"{task_id}: правка меняет число строк файла"
    changed = [i for i, (a, b) in enumerate(zip(before, after), 1) if a != b]
    assert changed == [line], f"{task_id}: правка затрагивает строки {changed}, ожидалась только {line}"


# --------------------------------------------------------------------------- #
# Hidden tests: run them in a throwaway copy of the repository
# --------------------------------------------------------------------------- #


def _run_modules(
    repo: Path, sandbox: Path, modules: list[tuple[str, Path]]
) -> tuple[dict[str, dict[str, tuple[bool, str]]], str]:
    """Run the modules and return ``({label: {test: (ok, detail)}}, note)``."""
    runner = sandbox / "run_review_tests.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    # The trusted directory stays outside the sandbox: the final report path is
    # handed to the supervisor and only the supervisor writes it, after the
    # child is gone. The control file is how the child announces the throwaway
    # report path it picked itself.
    trusted = Path(tempfile.mkdtemp(prefix="dailybench-review-trusted-"))
    report_path = trusted / "report.jsonl"
    control = trusted / "child-path.txt"
    argv = [sys.executable, str(runner), str(repo), str(report_path), str(control)]
    for label, path in modules:
        argv += [label, str(path)]
    empty = {label: {} for label, _ in modules}
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_HIDDEN_TIMEOUT_S,
            cwd=str(sandbox),
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(trusted, ignore_errors=True)
        return empty, f"тесты не уложились в {_HIDDEN_TIMEOUT_S} с"
    except OSError as exc:
        shutil.rmtree(trusted, ignore_errors=True)
        return empty, f"не удалось запустить python: {exc}"

    try:
        blob = report_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        note = f"тесты не выполнились: {tail[-1][:200] if tail else 'нет вывода'} ({exc})"
        return empty, note
    finally:
        shutil.rmtree(trusted, ignore_errors=True)

    rows: dict[str, dict[str, tuple[bool, str]]] = empty
    for line in blob.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "eof" in row:
            continue
        label = str(row.get("module", ""))
        if label in rows:
            rows[label][str(row.get("test", ""))] = (bool(row.get("ok")), str(row.get("detail", "")))
    note = ""
    if any(not group for group in rows.values()):
        note = "тесты не выполнились"
    return rows, note


def _run_test_modules(
    task: core.TaskInstance, target_source: str, labels: tuple[str, ...]
) -> tuple[dict[str, dict[str, tuple[bool, str]]], str]:
    """Materialise the workspace, overwrite the reviewed file, run the tests."""
    sandbox = Path(tempfile.mkdtemp(prefix=f"dailybench-{task.id.lower()}-review-"))
    repo = sandbox / "repo"
    repo.mkdir(parents=True)
    for rel, content in (task.workspace or {}).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (repo / task.oracle["target_file"]).write_text(target_source, encoding="utf-8")

    modules: list[tuple[str, Path]] = []
    for label in labels:
        if label == "hidden":
            path = sandbox / "test_review_hidden.py"
            path.write_text(task.oracle["hidden_tests"], encoding="utf-8")
        else:
            path = repo / task.oracle["public_test_file"]
        modules.append((label, path))
    # The hidden module runs first: that is exactly what ``evaluate`` sees, so a
    # fixture claim proved here holds during scoring too.
    return _run_modules(repo, sandbox, modules)


def _hidden_summary(results: dict[str, tuple[bool, str]]) -> tuple[bool, str]:
    if not results:
        return False, "скрытые тесты не выполнились"
    failures = sorted((name, detail) for name, (ok, detail) in results.items() if not ok)
    if failures:
        name, detail = failures[0]
        return False, (
            f"{len(results) - len(failures)}/{len(results)} скрытых тестов прошли; "
            f"первый провал {name}: {detail[:200]}"
        )
    return True, f"все {len(results)} скрытых тестов прошли"


def _verify_fixture(task: core.TaskInstance, source: str, *, expect_hidden_pass: bool) -> None:
    """Assert a claim the scenario makes about its own fixture."""
    rows, note = _run_test_modules(task, source, ("hidden", "public"))
    assert not note, f"{task.id}: {note}"
    public = rows.get("public", {})
    assert public, f"{task.id}: публичные тесты не выполнились"
    assert all(ok for ok, _ in public.values()), f"{task.id}: публичные тесты упали: {public}"
    hidden = rows.get("hidden", {})
    assert hidden, f"{task.id}: скрытые тесты не выполнились"
    failed = sorted(name for name, (ok, _) in hidden.items() if not ok)
    if expect_hidden_pass:
        assert not failed, f"{task.id}: эталон провалил скрытые тесты {failed}"
    else:
        assert failed, f"{task.id}: дефектный файл не провалил ни одного скрытого теста"


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

_J1_MODULES = ("pipeline/window.py", "metrics/window.py", "pipeline/sliding.py")
_J1_FUNCS = ("tail", "last_values", "window")
_J1_PREFIXES = ("svc.", "metrics.", "pipe.")
_J2_MODULES = ("ingest/paths.py", "ingest/locations.py", "sink/paths.py")
_J2_FUNCS = ("resolve", "join_path", "full_path")
_J3_MODULES = ("feed/paging.py", "audit/paging.py", "stream/cursor.py")
_J3_FUNCS = ("page", "fetch_page", "cursor_page")
_J3_TENANTS = ("acme", "globex", "initech", "umbrella")
_J3_EVENTS = ("login", "upload", "export", "billing", "invite", "purge")
_J4_MODULES = ("billing/delivery.py", "payments/delivery.py", "ledger/charge.py")
_J4_FUNCS = ("charge", "deliver", "bill")


def _scenario_j1(rng: random.Random) -> core.TaskInstance:
    module_path = rng.choice(_J1_MODULES)
    package, _, filename = module_path.partition("/")
    dotted = f"{package}.{filename[:-3]}"
    func = rng.choice(_J1_FUNCS)
    prefix = rng.choice(_J1_PREFIXES)
    sample = [rng.randrange(1, 90) for _ in range(3)]

    buggy = _fill(_J1_WINDOW, FUNC=func, RETURN="-count - 1 :")
    reference = _fill(_J1_WINDOW, FUNC=func, RETURN="-count :")
    marker = "    return values[-count - 1 :]"
    line = _defect_line(buggy, marker)
    _assert_single_line_fix("J-1", buggy, reference, line)

    helper = _fill(_J1_HELPER, PREFIX=prefix)
    public_test = f"tests/test_{filename[:-3]}.py"
    public = _fill(
        _J1_PUBLIC_TEST, DOTTED=dotted, FUNC=func, SAMPLE=", ".join(str(value) for value in sample)
    )
    hidden = _fill(_J1_HIDDEN_TESTS, DOTTED=dotted, FUNC=func)

    workspace = {
        "README.md": f"# {package}\n\nSmall helpers shared by the reporting service: labels and window slices.\n",
        f"{package}/__init__.py": f'"""{package} package."""\n',
        f"{package}/labels.py": helper,
        module_path: buggy,
        public_test: public,
    }
    task = core.TaskInstance(
        id="J-1",
        category="J",
        title="Code review",
        prompt=_PROMPT.format(files=_render_files(workspace), public_test=public_test),
        max_points=_POINTS["J-1"],
        workspace=workspace,
        oracle={
            "reference": _reference_reply(module_path, line, reference),
            "reference_source": reference,
            "target_file": module_path,
            "line": line,
            "hidden_tests": hidden,
            "public_test_file": public_test,
            "defect": "граница окна: срез начинается на элемент раньше",
            "checks": ["one_line_fix", "defect_located", "fix_passes_hidden_tests"],
        },
        meta={"scenario": "J-1"},
    )
    _verify_fixture(task, buggy, expect_hidden_pass=False)
    _verify_fixture(task, reference, expect_hidden_pass=True)
    return task


def _scenario_j2(rng: random.Random) -> core.TaskInstance:
    module_path = rng.choice(_J2_MODULES)
    package, _, filename = module_path.partition("/")
    dotted = f"{package}.{filename[:-3]}"
    func = rng.choice(_J2_FUNCS)
    shard = rng.randrange(64, 4096)

    buggy = _fill(_J2_PATHS, FUNC=func, KEY="name")
    reference = _fill(_J2_PATHS, FUNC=func, KEY="(root, name)")
    marker = "    key = name"
    line = _defect_line(buggy, marker)
    _assert_single_line_fix("J-2", buggy, reference, line)

    decoy = _fill(_J2_DECOY, SHARD=str(shard))
    public_test = f"tests/test_{filename[:-3]}.py"
    public = _fill(_J2_PUBLIC_TEST, DOTTED=dotted, FUNC=func)
    hidden = _fill(_J2_HIDDEN_TESTS, DOTTED=dotted, FUNC=func)

    workspace = {
        "README.md": f"# {package}\n\nRecord ingest service: shard configuration and path joining.\n",
        f"{package}/__init__.py": f'"""{package} package."""\n',
        f"{package}/config.py": decoy,
        module_path: buggy,
        public_test: public,
    }
    task = core.TaskInstance(
        id="J-2",
        category="J",
        title="Code review",
        prompt=_PROMPT.format(files=_render_files(workspace), public_test=public_test),
        max_points=_POINTS["J-2"],
        workspace=workspace,
        oracle={
            "reference": _reference_reply(module_path, line, reference),
            "reference_source": reference,
            "target_file": module_path,
            "line": line,
            "hidden_tests": hidden,
            "public_test_file": public_test,
            "defect": "ключ кэша собран из части аргументов: повторный вызов с другим root отдаёт чужой результат",
            "checks": ["one_line_fix", "defect_located", "fix_passes_hidden_tests"],
        },
        meta={"scenario": "J-2"},
    )
    _verify_fixture(task, buggy, expect_hidden_pass=False)
    _verify_fixture(task, reference, expect_hidden_pass=True)
    return task


def _oracle_page(rows: list[dict], tenant: str, cursor: int, limit: int) -> tuple[list[int], int | None]:
    """Independent page oracle: exact allowed ids and the next cursor.

    Deliberately not the page function itself: a plain scan that knows nothing
    about slicing, so a wrong filter order cannot confirm its own bug.
    """
    allowed: list[int] = []
    for row in rows:
        if row["id"] <= cursor or row["tenant"] != tenant:
            continue
        allowed.append(row["id"])
        if len(allowed) >= limit:
            break
    nxt = allowed[-1] if allowed and len(allowed) == limit else None
    return allowed, nxt


def _walk_oracle(rows: list[dict], tenant: str, limit: int) -> list[list[int]]:
    pages: list[list[int]] = []
    cursor = 0
    while True:
        ids, nxt = _oracle_page(rows, tenant, cursor, limit)
        if ids or not pages:
            pages.append(ids)
        if nxt is None:
            return pages
        cursor = nxt


def _render_rows(rows: list[dict]) -> str:
    return "".join(
        f'    {{"id": {row["id"]}, "tenant": {row["tenant"]!r}, "event": {row["event"]!r}}},\n'
        for row in rows
    )


def _scenario_j3(rng: random.Random) -> core.TaskInstance:
    module_path = rng.choice(_J3_MODULES)
    package, _, filename = module_path.partition("/")
    dotted = f"{package}.{filename[:-3]}"
    func = rng.choice(_J3_FUNCS)
    us, them = rng.sample(_J3_TENANTS, 2)
    events = [_J3_EVENTS[rng.randrange(len(_J3_EVENTS))] for _ in range(10)]

    # The first two rows are ours (so the public first page is green on the
    # buggy file), then one of theirs interleaves ours. Positions 3-6 are fixed
    # so the hidden second page has to skip a foreign row; the tail is random
    # but must keep both tenants present.
    pattern = [us, us, them, us, them, us]
    pattern += [rng.choice((us, them)) for _ in range(4)]
    if them not in pattern[6:]:
        pattern[-1] = them
    if us not in pattern[6:]:
        pattern[-2] = us
    rows = [
        {"id": index, "tenant": tenant, "event": events[index - 1]}
        for index, tenant in enumerate(pattern, 1)
    ]
    allowed = [row["id"] for row in rows if row["tenant"] == us]
    second, second_cursor = _oracle_page(rows, us, 2, 2)
    first_ids, first_cursor = _oracle_page(rows, us, 0, 2)
    assert second and second_cursor is not None
    assert second[0] != 3 and any(row["tenant"] == them for row in rows[2:4])

    limit = 2
    buggy = _fill(_J3_PAGING, FUNC=func, ROWS=_render_rows(rows), LIMIT=str(limit))
    reference = _fill(
        _J3_PAGING,
        FUNC=func,
        ROWS=_render_rows(rows),
        LIMIT=str(limit),
    ).replace("    window = rows[:limit]", "    window = [row for row in rows if row[\"tenant\"] == tenant][:limit]")
    marker = "    window = rows[:limit]"
    line = _defect_line(buggy, marker)
    _assert_single_line_fix("J-3", buggy, reference, line)

    public_test = f"tests/test_{filename[:-3]}.py"
    public = _fill(
        _J3_PUBLIC_TEST,
        DOTTED=dotted,
        FUNC=func,
        US=us,
        FIRST_IDS=", ".join(str(value) for value in first_ids),
        FIRST_CURSOR=str(first_cursor),
    )
    hidden = _fill(
        _J3_HIDDEN_TESTS,
        DOTTED=dotted,
        FUNC=func,
        US=us,
        ALL_IDS=", ".join(str(value) for value in allowed),
        SECOND_IDS=", ".join(str(value) for value in second),
        SECOND_CURSOR=str(second_cursor),
    )

    workspace = {
        "README.md": f"# {package}\n\nTenant-scoped audit feed shared by the console and the export worker.\n",
        f"{package}/__init__.py": f'"""{package} package."""\n',
        module_path: buggy,
        public_test: public,
    }
    task = core.TaskInstance(
        id="J-3",
        category="J",
        title="Code review",
        prompt=_PROMPT.format(files=_render_files(workspace), public_test=public_test),
        max_points=_POINTS["J-3"],
        workspace=workspace,
        oracle={
            "reference": _reference_reply(module_path, line, reference),
            "reference_source": reference,
            "target_file": module_path,
            "line": line,
            "hidden_tests": hidden,
            "public_test_file": public_test,
            "tenant": us,
            "foreign_tenant": them,
            "allowed_ids": allowed,
            "pages": _walk_oracle(rows, us, 2),
            "defect": "фильтр доступа применяется после limit: страница теряет разрешённые записи, отдаёт чужие и обрывает курсор",
            "checks": ["one_line_fix", "defect_located", "fix_passes_hidden_tests"],
        },
        meta={"scenario": "J-3"},
    )
    _verify_fixture(task, buggy, expect_hidden_pass=False)
    _verify_fixture(task, reference, expect_hidden_pass=True)
    return task


def _scenario_j4(rng: random.Random) -> core.TaskInstance:
    module_path = rng.choice(_J4_MODULES)
    package, _, filename = module_path.partition("/")
    dotted = f"{package}.{filename[:-3]}"
    func = rng.choice(_J4_FUNCS)
    key = f"op-{rng.randrange(1000, 99999)}"
    other_key = f"op-{rng.randrange(100000, 999999)}"
    amount = rng.randrange(50, 500)

    buggy = _fill(_J4_DELIVERY, FUNC=func)
    reference = buggy.replace("    token = (user, amount)", "    token = key")
    marker = "    token = (user, amount)"
    line = _defect_line(buggy, marker)
    _assert_single_line_fix("J-4", buggy, reference, line)

    public_test = f"tests/test_{filename[:-3]}.py"
    public = _fill(_J4_PUBLIC_TEST, DOTTED=dotted, FUNC=func, KEY=key, AMOUNT=str(amount))
    hidden = _fill(
        _J4_HIDDEN_TESTS,
        DOTTED=dotted,
        FUNC=func,
        KEY=key,
        OTHER_KEY=other_key,
        AMOUNT=str(amount),
    )

    workspace = {
        "README.md": f"# {package}\n\nBilling delivery worker: at-least-once transport, exactly-once effect.\n",
        f"{package}/__init__.py": f'"""{package} package."""\n',
        module_path: buggy,
        public_test: public,
    }
    task = core.TaskInstance(
        id="J-4",
        category="J",
        title="Code review",
        prompt=_PROMPT.format(files=_render_files(workspace), public_test=public_test),
        max_points=_POINTS["J-4"],
        workspace=workspace,
        oracle={
            "reference": _reference_reply(module_path, line, reference),
            "reference_source": reference,
            "target_file": module_path,
            "line": line,
            "hidden_tests": hidden,
            "public_test_file": public_test,
            "key": key,
            "defect": "дедупликация по payload вместо idempotency key: новый ключ теряется, повтор от другого пользователя списывает второй раз",
            "checks": ["one_line_fix", "defect_located", "fix_passes_hidden_tests"],
        },
        meta={"scenario": "J-4"},
    )
    _verify_fixture(task, buggy, expect_hidden_pass=False)
    _verify_fixture(task, reference, expect_hidden_pass=True)
    return task


def _reference_reply(target_file: str, line: int, source: str) -> str:
    return json.dumps({"file": target_file, "line": line, "fix": source}, ensure_ascii=False)


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    profile = core.profile_of(cfg)
    if profile == "pilot":
        return [_scenario_j3(rng)]
    tasks = [_scenario_j1(rng), _scenario_j2(rng)]
    if profile == "stretch-v1":
        tasks += [_scenario_j3(rng), _scenario_j4(rng)]
    return tasks


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _same_path(value: Any, target: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().replace("\\", "/").lstrip("./") == target


def _as_line(value: Any) -> int | None:
    """The reported line as an integer; a quoted number is still the same line."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    return None


def extract_fix(payload: Any) -> str | None:
    """The corrected file from the reply: a fenced block when present, else the string."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("fix")
    if not isinstance(raw, str) or not raw.strip():
        return None
    block = text.first_fenced_block(raw)
    if block:
        return block
    candidate = text.strip_fence(raw)[0]
    return candidate or None


def _fix_shape(buggy: str, fix: str) -> tuple[bool, str]:
    """Is ``fix`` a one-line edit of ``buggy``?

    The file must differ, keep the same lines and replace exactly one
    substantive line. Only the multiset of lines is compared, so moving untouched
    lines around is still the same edit; adding a function or a block is not.
    """
    before = buggy.splitlines()
    after = fix.splitlines()
    if len(before) != len(after):
        return False, f"правка меняет число строк файла: {len(before)} → {len(after)}"
    removed = list((Counter(before) - Counter(after)).elements())
    added = list((Counter(after) - Counter(before)).elements())
    if not removed and not added:
        return False, "файл не изменён"
    if len(removed) != 1 or len(added) != 1:
        return False, (
            f"правка не укладывается в одну строку: удалено {len(removed)}, "
            f"добавлено {len(added)} строк"
        )
    if not removed[0].strip() or not added[0].strip():
        return False, "изменённая строка пустая"
    return True, "изменена ровно одна содержательная строка"


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="J")
    target = str(task.oracle["target_file"])
    defect_line = int(task.oracle["line"])
    shape_share, location_share = _SHARES[task.id]
    shape_points = round(task.max_points * shape_share, 4)
    location_points = round(task.max_points * location_share, 4)
    fix_points = round(task.max_points - shape_points - location_points, 4)

    payload, strict = text.strict_json(tr.final_text)
    if not strict:
        payload = text.loose_json(tr.final_text)
    reported_file = payload.get("file") if isinstance(payload, dict) else None
    reported_line = _as_line(payload["line"]) if isinstance(payload, dict) and "line" in payload else None

    # The shape comes first: a rewrite that happens to keep the behaviour still
    # violates the contract stated in the prompt (C5), and the reported location
    # is only worth something in a one-line diff.
    fix = extract_fix(payload)
    if fix is None:
        shape_ok, shape_detail = False, "в ответе нет полного исправленного файла"
    else:
        shape_ok, shape_detail = _fix_shape(str((task.workspace or {}).get(target, "")), fix)
    score.add("one_line_fix", shape_ok, shape_points, shape_detail)

    located = shape_ok and _same_path(reported_file, target) and reported_line == defect_line
    prefix = "" if shape_ok else "форма не соблюдена; "
    score.add(
        "defect_located",
        located,
        location_points,
        f"{prefix}получено file={reported_file!r} line={reported_line!r}, "
        f"ожидалось file={target!r} line={defect_line}",
    )

    if fix is None:
        score.add("fix_passes_hidden_tests", False, fix_points, "в ответе нет полного исправленного файла")
        return score

    rows, note = _run_test_modules(task, fix, ("hidden", "public"))
    if note:
        score.add("fix_passes_hidden_tests", False, fix_points, note)
        return score
    passed, detail = _hidden_summary(rows.get("hidden", {}))
    # The public tests ship in the prompt: a fix that keeps the hidden cases but
    # breaks them is not a fix (C5).
    public = rows.get("public", {})
    public_failed = [name for name, (ok, _) in public.items() if not ok]
    if public_failed:
        passed = False
        detail = f"публичные тесты упали: {public_failed[0]}"
    score.add("fix_passes_hidden_tests", passed, fix_points, detail)
    if not strict:
        score.notes.append("ответ не является строгим JSON целиком")
    return score


core.register(
    key="J",
    title="Code review",
    points=5.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
    expected_ids_by_profile={"pilot": PILOT_IDS, "stretch-v1": ALL_IDS},
)
