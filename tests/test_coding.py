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

import json
import random
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

from dailybench import core
from dailybench.tasks import coding, coding_go, coding_sql

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
