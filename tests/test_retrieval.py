"""Tests for category I — retrieval over workspace files.

Covers the frozen contract (ids, points, registry), determinism, the reference
and garbage score bands, the exact trap split for each scenario, and the safety
of the read-only workspace driver.
"""

from __future__ import annotations

from dailybench import ModelReply, RunConfig, TaskInstance, Transcript, Turn, core, text
from dailybench.rng import task_rng
from dailybench.runner import Runner, materialise
from dailybench.selfcheck import GarbageProvider, OracleProvider
from dailybench.tasks import retrieval

_REFERENCE_TOOLS = ["list_files", "read_file", "search_code"]


def _tasks(seed: int = 20260915) -> list[TaskInstance]:
    return retrieval.generate(task_rng(seed, "I"), RunConfig(seed=seed))


def _task(task_id: str, seed: int = 20260915) -> TaskInstance:
    return next(task for task in _tasks(seed) if task.id == task_id)


def _transcript(task: TaskInstance, reply: str) -> Transcript:
    return Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[], reply=ModelReply(text=reply))],
    )


def _failed(score: core.Score) -> list[str]:
    return [check.name for check in score.checks if not check.passed]


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_ids_points_and_registry() -> None:
    tasks = _tasks()
    assert [task.id for task in tasks] == list(retrieval.IDS)
    assert [task.category for task in tasks] == ["I", "I"]
    assert sum(task.max_points for task in tasks) == 4.0
    assert all(task.max_points == 2.0 for task in tasks)

    module = core.REGISTRY["I"]
    assert module.title == "Retrieval over files"
    assert module.points == 4.0
    assert module.expected_ids == retrieval.IDS


def test_every_task_carries_two_checks_of_one_point() -> None:
    for task in _tasks():
        score = retrieval.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert len(score.checks) == 2
        assert [check.points for check in score.checks] == [1.0, 1.0]
        assert score.possible == task.max_points


def test_only_read_only_tools_are_exposed() -> None:
    names = [schema["function"]["name"] for schema in retrieval.schemas()]
    assert names == _REFERENCE_TOOLS
    for task in _tasks():
        assert [tool["function"]["name"] for tool in (task.tools or [])] == names


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_generation_is_deterministic_but_seeded() -> None:
    first = _tasks(20260915)
    second = _tasks(20260915)
    other = _tasks(20260916)

    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.workspace for task in first] == [task.workspace for task in second]
    assert [task.oracle for task in first] == [task.oracle for task in second]
    assert [task.id for task in first] == [task.id for task in other]
    assert any(a.prompt != b.prompt for a, b in zip(first, other))
    assert any(a.workspace != b.workspace for a, b in zip(first, other))
    assert any(a.oracle["reference"] != b.oracle["reference"] for a, b in zip(first, other))


def test_prompts_do_not_leak_the_raw_records() -> None:
    for task in _tasks():
        for line in "\n".join(task.workspace.values()).splitlines():
            if line:
                assert line not in task.prompt


# --------------------------------------------------------------------------- #
# Score bands
# --------------------------------------------------------------------------- #


def test_reference_scores_the_full_two_points() -> None:
    for task in _tasks():
        score = retrieval.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert score.earned == 2.0, _failed(score)


def test_garbage_scores_zero() -> None:
    cfg = RunConfig()
    for task in _tasks():
        run = Runner(cfg, GarbageProvider()).run_task(task)
        assert run.score.earned == 0.0
        assert _failed(run.score) == ["json_shape", "aggregate"] or _failed(run.score) == [
            "json_shape",
            "conflict",
        ]


def test_a_fenced_answer_keeps_content_but_loses_shape() -> None:
    task = _task("I-1")
    fenced = f"```json\n{task.oracle['reference']}\n```"
    score = retrieval.evaluate(task, _transcript(task, fenced))
    assert score.earned == 1.0
    assert score.checks[0].passed is False and score.checks[1].passed is True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def test_reference_transcript_drives_the_read_only_driver() -> None:
    cfg = RunConfig()
    for task in _tasks():
        run = Runner(cfg, OracleProvider(task)).run_task(task)
        assert run.transcript.error is None
        assert run.transcript.extra.get("driver") == "I"
        reads = [call for call in run.transcript.tool_log if call["name"] == "read_file"]
        assert len(reads) == len(task.oracle["golden_calls"])
        assert not any(call["error"] for call in run.transcript.tool_log)
        # Read-only means the workspace is byte-identical before and after.
        assert run.transcript.workspace_after == run.transcript.workspace_before
        assert run.score.earned == 2.0


def test_driver_reads_the_workspace_but_refuses_to_escape_it() -> None:
    task = _task("I-1")
    work_dir = materialise(task)
    (work_dir.parent / "outside.txt").write_text("СЕКРЕТ-ВНЕ-КАТАЛОГА", encoding="utf-8")

    driver = core.TOOL_DRIVERS["I"](task, work_dir)

    # A legitimate read still works.
    first = sorted(task.workspace)[0]
    good = driver.execute("read_file", {"path": first})
    assert good.error is False and good.text == task.workspace[first]

    # ``../outside.txt`` is rejected and nothing outside the workspace is read.
    escape = driver.execute("read_file", {"path": "../outside.txt"})
    assert escape.error is True
    assert "СЕКРЕТ-ВНЕ-КАТАЛОГА" not in escape.text
    assert "вне каталога" in escape.text

    listing = driver.execute("list_files", {"path": ".."})
    assert listing.error is True
    assert "СЕКРЕТ-ВНЕ-КАТАЛОГА" not in listing.text


# --------------------------------------------------------------------------- #
# Exact trap split: right shape, wrong content
# --------------------------------------------------------------------------- #


def test_i1_one_file_aggregate_earns_shape_only() -> None:
    task = _task("I-1")
    _, strict = text.strict_json(task.oracle["naive_answer"])
    assert strict is True, "ловушка должна быть строгим JSON"
    score = retrieval.evaluate(task, _transcript(task, task.oracle["naive_answer"]))
    assert score.earned == 1.0
    assert score.checks[0].passed is True, "форма верна"
    assert score.checks[1].passed is False, "агрегат по одному файлу неверен"


def test_i2_first_and_last_file_win_traps_earn_shape_only() -> None:
    task = _task("I-2")
    assert task.oracle["naive_first"] != task.oracle["value"]
    assert task.oracle["naive_last"] != task.oracle["value"]
    for key in ("naive_answer", "naive_last_answer"):
        score = retrieval.evaluate(task, _transcript(task, task.oracle[key]))
        assert score.earned == 1.0, (key, _failed(score))
        assert score.checks[0].passed is True
        assert score.checks[1].passed is False
