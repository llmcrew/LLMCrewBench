"""Tests for category I — retrieval over workspace files.

Covers the frozen contract (ids, points, registry), determinism, the reference
and garbage score bands, the exact trap split for each scenario, and the safety
of the read-only workspace driver.
"""

from __future__ import annotations

import json

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


# --------------------------------------------------------------------------- #
# Profile contract
# --------------------------------------------------------------------------- #

_PROFILES = ("pilot", "stretch-v1")


def _profile_tasks(profile: str, seed: int = 20260915) -> list[TaskInstance]:
    return retrieval.generate(task_rng(seed, "I"), RunConfig(seed=seed, profile=profile))


def test_profile_lists_are_registered_and_generated() -> None:
    assert retrieval.IDS == retrieval.BASELINE_IDS == ("I-1", "I-2")
    assert retrieval.PILOT_IDS == ("I-3",)
    assert retrieval.ALL_IDS == ("I-1", "I-2", "I-3", "I-4", "I-5")

    module = core.REGISTRY["I"]
    assert module.expected_ids == retrieval.BASELINE_IDS
    assert module.expected_ids_for("pilot") == retrieval.PILOT_IDS
    assert module.expected_ids_for("stretch-v1") == retrieval.ALL_IDS
    assert module.expected_ids_for() == retrieval.BASELINE_IDS

    assert [task.id for task in _profile_tasks("pilot")] == ["I-3"]
    assert [task.id for task in _profile_tasks("stretch-v1")] == ["I-1", "I-2", "I-3", "I-4", "I-5"]
    assert [task.id for task in _tasks()] == ["I-1", "I-2"]


# --------------------------------------------------------------------------- #
# I-3 / I-4 shared contract
# --------------------------------------------------------------------------- #

_NEW_TASKS = ("I-3", "I-4")
_N8_TASK = "I-5"


def _new_task(task_id: str, seed: int = 20260915) -> TaskInstance:
    return next(task for task in _profile_tasks("stretch-v1", seed) if task.id == task_id)


def _resolved_i3(workspace: dict[str, str]) -> tuple[dict[str, object], dict[str, str]]:
    """Second, text-oriented resolver: the oracle is computed from structure."""
    values: dict[str, object] = {}
    sources: dict[str, str] = {}
    stack: list[str] = []

    def visit(path: str) -> None:
        if path in stack:
            return
        stack.append(path)
        for number, line in enumerate(workspace[path].splitlines(), start=1):
            if line.startswith("include "):
                visit(line[len("include "):])
                continue
            key, _, raw = line.partition(" = ")
            raw = raw.strip()
            if raw == "UNSET":
                continue
            if raw == "null":
                value: object = None
            elif raw.startswith("["):
                value = [item.strip() for item in raw[1:-1].split(",")]
            else:
                try:
                    value = int(raw)
                except ValueError:
                    value = raw
            values[key] = value
            sources[key] = f"{path}:{number}"
        stack.pop()

    visit("conf/main.conf")
    return values, sources


def _resolved_i4(workspace: dict[str, str]) -> dict[str, object]:
    active: dict[str, tuple[str, str]] = {}
    for path in sorted(workspace):
        for number, line in enumerate(workspace[path].splitlines(), start=1):
            if not line:
                continue
            if line.startswith("DELETE "):
                active.pop(line[len("DELETE "):], None)
                continue
            ident, _, owner = line.split(" ")
            active[ident] = (owner, f"{path}:{number}")
    by_owner: dict[str, int] = {}
    for owner, _ in active.values():
        by_owner[owner] = by_owner.get(owner, 0) + 1
    return {
        "active": len(active),
        "by_owner": dict(sorted(by_owner.items())),
        "source": {ident: source for ident, (_, source) in sorted(active.items())},
    }


def test_new_generation_is_deterministic_and_seeded() -> None:
    for profile in _PROFILES:
        first = _profile_tasks(profile)
        second = _profile_tasks(profile)
        other = _profile_tasks(profile, 20260916)
        assert [task.id for task in first] == [task.id for task in other]
        assert [task.prompt for task in first] == [task.prompt for task in second]
        assert [task.workspace for task in first] == [task.workspace for task in second]
        assert [task.oracle for task in first] == [task.oracle for task in second]
        # The I-3 prompt is static (the fixture lives in the workspace); only
        # the workspace and the oracle must move with the seed.
        assert any(a.workspace != b.workspace for a, b in zip(first, other))
        assert any(a.oracle["reference"] != b.oracle["reference"] for a, b in zip(first, other))


def test_new_tasks_carry_three_checks_summing_to_the_budget() -> None:
    for task_id in _NEW_TASKS:
        task = _new_task(task_id)
        score = retrieval.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert sum(check.points for check in score.checks) == task.max_points == 2.0
        assert len(score.checks) == 3
        score.fit_points(task.max_points)
        assert abs(sum(check.points for check in score.checks) - task.max_points) < 1e-9


def test_new_references_score_full_and_garbage_scores_zero() -> None:
    for task_id in _NEW_TASKS:
        task = _new_task(task_id)
        score = retrieval.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert score.earned == 2.0, _failed(score)
        run = Runner(RunConfig(), GarbageProvider()).run_task(task)
        assert run.score.earned == 0.0


def test_new_reference_transcripts_read_the_whole_workspace() -> None:
    for task_id in _NEW_TASKS:
        task = _new_task(task_id)
        run = Runner(RunConfig(), OracleProvider(task)).run_task(task)
        assert run.transcript.error is None
        assert run.transcript.extra.get("driver") == "I"
        reads = [call for call in run.transcript.tool_log if call["name"] == "read_file"]
        assert len(reads) == len(task.oracle["golden_calls"]) == len(task.oracle["files"])
        assert not any(call["error"] for call in run.transcript.tool_log)
        assert run.score.earned == 2.0


def test_new_prompts_do_not_leak_the_raw_files() -> None:
    for seed in range(5):
        for task_id in _NEW_TASKS:
            task = _new_task(task_id, seed)
            for content in task.workspace.values():
                for line in content.splitlines():
                    assert line not in task.prompt


# --------------------------------------------------------------------------- #
# I-3 semantics and mutants
# --------------------------------------------------------------------------- #


def test_i3_fixture_bounds_and_semantics() -> None:
    for seed in range(5):
        task = _new_task("I-3", seed)
        files = task.oracle["files"]
        assert 6 <= len(files) <= 8

        def depth(path: str, seen: tuple[str, ...] = ()) -> int:
            if path in seen:
                return 0
            includes = [
                line[len("include "):]
                for line in task.workspace[path].splitlines()
                if line.startswith("include ")
            ]
            return 1 + max((depth(item, seen + (path,)) for item in includes), default=0)

        assert depth("conf/main.conf") - 1 <= 3, "не больше трёх include-уровней"
        assert task.oracle["values"]["cache.ttl"] is None
        assert len(task.oracle["values"]["routes"]) == 3
        assert task.oracle["values"]["service.port"] == 8081
        assert task.oracle["values"]["logging.level"] == "debug"
        assert task.oracle["values"]["max_conns"] == 900
        # UNSET inherits: the winning source is never the UNSET line.
        assert "defaults.conf" not in task.oracle["sources"]["cache.ttl"]
        assert task.oracle["sources"]["max_conns"].startswith("conf/overrides.conf:")
        # Exact file/line references point into an existing line of an existing file.
        for key, source in task.oracle["sources"].items():
            path, _, number = source.rpartition(":")
            lines = task.workspace[path].splitlines()
            assert 1 <= int(number) <= len(lines)
            assert lines[int(number) - 1].startswith(f"{key} = ")


def test_i3_independent_resolver_matches_the_oracle() -> None:
    for seed in range(10):
        task = _new_task("I-3", seed)
        values, sources = _resolved_i3(task.workspace)
        assert values == task.oracle["values"]
        assert sources == task.oracle["sources"]


def test_i3_cycle_seed_terminates_with_a_valid_oracle() -> None:
    task = next(
        task
        for seed in range(60)
        for task in [_new_task("I-3", seed)]
        if task.oracle["cycle"]
    )
    assert "include conf/limits.conf" in task.workspace["conf/overrides.conf"]
    assert "include conf/overrides.conf" in task.workspace["conf/limits.conf"]
    values, sources = _resolved_i3(task.workspace)
    assert values == task.oracle["values"] and sources == task.oracle["sources"]
    score = retrieval.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == 2.0


def test_i3_mutants_never_reach_full_score() -> None:
    for seed in range(5):
        task = _new_task("I-3", seed)
        assert task.oracle["naive_answer"] != task.oracle["reference"]
        assert task.oracle["naive_concat_answer"] != task.oracle["reference"]
        assert task.oracle["naive_null_answer"] != task.oracle["reference"]
        for key in ("naive_answer", "naive_concat_answer", "naive_null_answer"):
            _, strict = text.strict_json(task.oracle[key])
            assert strict is True, "мутант обязан быть строгим JSON"
            score = retrieval.evaluate(task, _transcript(task, task.oracle[key]))
            assert 0 < score.earned < 2.0, (seed, key, _failed(score))
            assert score.checks[0].passed is True
    # List concatenation is exactly what the second mutant copies.
    payload = text.strict_json(_new_task("I-3").oracle["naive_concat_answer"])[0]
    assert len(payload["values"]["routes"]) == 5, "2 + 3 элемента склеены"


def test_i3_right_values_with_wrong_sources_earn_values_only() -> None:
    task = _new_task("I-3")
    sources = {key: "conf/main.conf:1" for key in task.oracle["sources"]}
    payload = text.strict_json(task.oracle["reference"])[0]
    payload["sources"] = sources
    score = retrieval.evaluate(task, _transcript(task, json.dumps(payload, ensure_ascii=False)))
    assert score.checks[0].passed is True
    assert score.checks[1].passed is True
    assert score.checks[2].passed is False
    assert score.earned == 1.25


# --------------------------------------------------------------------------- #
# I-4 semantics and mutants
# --------------------------------------------------------------------------- #


def _records(workspace: dict[str, str]) -> list[str]:
    return [
        line
        for content in workspace.values()
        for line in content.splitlines()
        if line and not line.startswith("DELETE ")
    ]


def test_i4_fixture_bounds_and_state() -> None:
    for seed in range(5):
        task = _new_task("I-4", seed)
        files = task.oracle["files"]
        assert 8 <= len(files) <= 10
        assert 12 <= len(_records(task.workspace)) <= 30
        reference = text.strict_json(task.oracle["reference"])[0]
        assert reference["active"] == sum(reference["by_owner"].values())
        assert set(reference["source"]) == {
            ident for ident in task.oracle["source"]
        }
        # Every live id has one winning line, and it is the latest one.
        for ident, source in task.oracle["source"].items():
            path, _, number = source.rpartition(":")
            assert task.workspace[path].splitlines()[int(number) - 1].startswith(f"{ident} ")
        # Tombstoned records are absent; the resurrected one is present.
        assert len(task.oracle["deleted"]) == 1
        for ident in task.oracle["deleted"]:
            assert ident not in reference["source"]
        assert task.oracle["revived"] in reference["source"]


def test_i4_independent_resolver_matches_the_oracle() -> None:
    for seed in range(10):
        task = _new_task("I-4", seed)
        assert _resolved_i4(task.workspace) == text.strict_json(task.oracle["reference"])[0]


def test_i4_name_pairs_share_a_display_name_but_not_an_id() -> None:
    for seed in range(5):
        task = _new_task("I-4", seed)
        names: dict[str, list[str]] = {}
        for value in task.workspace.values():
            for line in value.splitlines():
                if line and not line.startswith("DELETE "):
                    ident, name, _ = line.split(" ")
                    names.setdefault(name, []).append(ident)
        pairs = [idents for idents in names.values() if len({*idents}) == 2]
        assert len(pairs) == 2, "нужны ровно две пары совпадающих имён"


def test_i4_mutants_never_reach_full_score() -> None:
    for seed in range(5):
        task = _new_task("I-4", seed)
        for key in ("naive_answer", "naive_deleted_answer", "naive_stale_answer"):
            _, strict = text.strict_json(task.oracle[key])
            assert strict is True, "мутант обязан быть строгим JSON"
            score = retrieval.evaluate(task, _transcript(task, task.oracle[key]))
            assert 0 < score.earned < 2.0, (seed, key, _failed(score))
            assert score.checks[0].passed is True
    # The by-name mutant loses exactly one active object per shared name pair.
    payload = text.strict_json(_new_task("I-4").oracle["naive_deleted_answer"])[0]
    reference = text.strict_json(_new_task("I-4").oracle["reference"])[0]
    assert payload["active"] == reference["active"] + 1, "удалённый объект посчитан"


def test_i4_right_state_with_a_stale_source_earns_state_only() -> None:
    task = _new_task("I-4")
    payload = text.strict_json(task.oracle["reference"])[0]
    stale = dict(payload["source"])
    corrected = [
        ident
        for ident, source in stale.items()
        if sum(
            1
            for content in task.workspace.values()
            for line in content.splitlines()
            if line.startswith(f"{ident} ")
        )
        > 1
    ]
    assert corrected, "нужен объект с исправлением владельца"
    for ident in corrected:
        first = next(
            f"{path}:{number}"
            for path, content in sorted(task.workspace.items())
            for number, line in enumerate(content.splitlines(), start=1)
            if line.startswith(f"{ident} ")
        )
        stale[ident] = first
    payload["source"] = stale
    score = retrieval.evaluate(task, _transcript(task, json.dumps(payload, ensure_ascii=False)))
    assert score.checks[0].passed is True
    assert score.checks[1].passed is True
    assert score.checks[2].passed is False
    assert score.earned == 1.25


# --------------------------------------------------------------------------- #
# N8 — offline research package with provenance
# --------------------------------------------------------------------------- #

_DEV_SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)


def _n8(seed: int = _DEV_SEEDS[0]) -> TaskInstance:
    return next(task for task in _profile_tasks("stretch-v1", seed) if task.id == _N8_TASK)


def _n8_resolve(task: TaskInstance) -> tuple[int, list[str]]:
    """Second, text-oriented resolver: parse the pages, apply the trust policy.

    The oracle is built from the planner's roles; this path only sees the
    rendered workspace plus the question parameters.
    """
    profile, field = task.oracle["profile"], task.oracle["field"]
    package = task.oracle["package"]
    pages: dict[str, tuple[list[str], dict[str, str]]] = {}
    for path, content in task.workspace.items():
        lines = content.splitlines()
        meta = dict(line.split("=", 1) for line in lines[1:4])
        pages[path] = (lines, meta)

    live = {
        path: lines
        for path, (lines, meta) in pages.items()
        if meta.get("status") in {"current", "in force"} and meta.get("authority") != "erratum"
    }
    scoped = {path: lines for path, lines in live.items() if lines[0].endswith(f"— {package}")}

    corrected: int | None = None
    errata_span = ""
    corrected_path = ""
    for path in sorted(pages):
        lines, meta = pages[path]
        if meta.get("authority") != "erratum" or meta.get("status") != "in force":
            continue
        target = next(
            (line[len("Исправляет "):-1] for line in lines if line.startswith("Исправляет "))
        )
        if target not in scoped:
            continue
        prefix = f"{profile} {field} = "
        for number, line in enumerate(lines, start=1):
            if line.startswith(prefix):
                corrected = int(line[len(prefix):])
                errata_span = f"{path}:{number}"
                corrected_path = target
                break
    assert corrected is not None

    row_prefix = f"| {profile} | {field} | "
    for number, line in enumerate(scoped[corrected_path], start=1):
        if line.startswith(row_prefix):
            return corrected, [errata_span, f"{corrected_path}:{number}"]
    raise AssertionError("release note не содержит исправляемую строку")


def test_n8_generation_is_seeded_and_stable() -> None:
    first = _profile_tasks("stretch-v1", _DEV_SEEDS[0])
    second = _profile_tasks("stretch-v1", _DEV_SEEDS[0])
    other = _profile_tasks("stretch-v1", _DEV_SEEDS[1])
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.workspace for task in first] == [task.workspace for task in second]
    assert [task.oracle for task in first] == [task.oracle for task in second]
    assert any(a.workspace != b.workspace for a, b in zip(first, other))
    assert any(a.oracle["reference"] != b.oracle["reference"] for a, b in zip(first, other))


def test_n8_fixture_states_trust_policy_and_spans() -> None:
    for seed in _DEV_SEEDS:
        task = _n8(seed)
        assert task.languages == ["ru"] and task.fresh_after
        assert len(task.oracle["files"]) == 5
        statuses = {
            line.split("=", 1)[1]
            for content in task.workspace.values()
            for line in content.splitlines()
            if line.startswith("status=")
        }
        assert {"current", "in force", "superseded", "draft"} <= statuses
        # The trust rule and the exact answer contract are in the prompt.
        for marker in ("authority", "erratum", "current", "superseded", "draft", "лишних"):
            assert marker in task.prompt
        assert task.oracle["value"] not in {
            task.oracle["stale_value"],
            task.oracle["draft_value"],
            task.oracle["foreign_value"],
        }
        # Every claimed span points at the line it is supposed to quote.
        errata, note = task.oracle["sources"]
        errata_path, _, errata_line = errata.rpartition(":")
        note_path, _, note_line = note.rpartition(":")
        assert task.workspace[errata_path].splitlines()[int(errata_line) - 1] == (
            f"{task.oracle['profile']} {task.oracle['field']} = {task.oracle['value']}"
        )
        assert task.workspace[note_path].splitlines()[int(note_line) - 1].startswith(
            f"| {task.oracle['profile']} | {task.oracle['field']} | "
        )
        assert task.workspace[errata_path].splitlines()[0].startswith("# Errata")
        for content in task.workspace.values():
            for line in content.splitlines():
                assert line not in task.prompt


def test_n8_independent_resolver_matches_the_oracle() -> None:
    for seed in _DEV_SEEDS:
        task = _n8(seed)
        assert _n8_resolve(task) == (task.oracle["value"], task.oracle["sources"])


def test_n8_mutants_never_reach_full_score() -> None:
    for seed in _DEV_SEEDS:
        task = _n8(seed)
        for key in ("naive_answer", "naive_unused_source_answer", "naive_instruction_answer", "naive_no_source_answer"):
            _, strict = text.strict_json(task.oracle[key])
            assert strict is True, "мутант обязан быть строгим JSON"
            score = retrieval.evaluate(task, _transcript(task, task.oracle[key]))
            assert score.earned < 2.0, (seed, key, _failed(score))
            if key != "naive_instruction_answer":
                assert score.earned > 0, (seed, key, _failed(score))


def test_n8_mutants_fail_exactly_one_required_field() -> None:
    task = _n8()
    stale = retrieval.evaluate(task, _transcript(task, task.oracle["naive_answer"]))
    assert stale.checks[0].passed is True
    assert stale.checks[1].passed is False and stale.checks[2].passed is False
    assert stale.earned == 0.5, "устаревшая страница даёт только форму"

    unused = retrieval.evaluate(task, _transcript(task, task.oracle["naive_unused_source_answer"]))
    assert [check.passed for check in unused.checks] == [True, True, False]
    assert unused.earned == 1.25, "цитата из чужого пакета — неверная provenance"

    blind = retrieval.evaluate(task, _transcript(task, task.oracle["naive_no_source_answer"]))
    assert [check.passed for check in blind.checks] == [True, True, False]
    assert blind.earned == 1.25, "ответ без source ID не проходит provenance"

    executed = retrieval.evaluate(task, _transcript(task, task.oracle["naive_instruction_answer"]))
    assert [check.passed for check in executed.checks] == [False, False, False]
    assert executed.earned == 0.0, "инструкция из errata добавила лишний ключ"



def test_i5_swapped_sources_lose_provenance_and_extra_key_loses_content() -> None:
    task = _n8()
    payload = json.loads(task.oracle["reference"])
    payload["sources"] = [payload["sources"][1], payload["sources"][0]]
    swapped = retrieval.evaluate(task, _transcript(task, json.dumps(payload, ensure_ascii=False)))
    assert [check.passed for check in swapped.checks] == [True, True, False]
    assert swapped.earned == 1.25

    injected = retrieval.evaluate(task, _transcript(task, task.oracle["naive_instruction_answer"]))
    assert set(json.loads(task.oracle["naive_instruction_answer"])) - {"value", "sources"}
    assert [check.passed for check in injected.checks] == [False, False, False]
    assert injected.earned == 0.0

    reference = retrieval.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert reference.earned == 2.0
    assert _failed(reference) == []


def test_n8_reference_full_garbage_zero_and_driver() -> None:
    task = _n8()
    score = retrieval.evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.earned == 2.0, _failed(score)
    run = Runner(RunConfig(), GarbageProvider()).run_task(task)
    assert run.score.earned == 0.0
    run = Runner(RunConfig(), OracleProvider(task)).run_task(task)
    assert run.transcript.error is None
    assert len([call for call in run.transcript.tool_log if call["name"] == "read_file"]) == 5
    assert not any(call["error"] for call in run.transcript.tool_log)
    assert run.score.earned == 2.0
