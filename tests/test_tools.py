"""Evaluators for T-11 and T-12 must grade the facts the prompts ask for."""

from __future__ import annotations

import json
import random
import re

from dailybench import core
from dailybench.config import RunConfig
from dailybench.providers import ScriptedProvider
from dailybench.runner import Runner
from dailybench.tasks import tools


def _task(task_id: str, seed: int, profile: str) -> core.TaskInstance:
    return next(task for task in tools.generate(random.Random(seed), RunConfig(profile=profile)) if task.id == task_id)


def _call(index: int, name: str, arguments: dict) -> core.ModelReply:
    return core.ModelReply(
        text="",
        tool_calls=[{"id": f"c{index}", "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}],
    )


def _finish(task: core.TaskInstance, text: str):
    replies = [
        _call(index, call["name"], call["arguments"])
        for index, call in enumerate(task.oracle["golden_calls"], 1)
    ]
    replies.append(core.ModelReply(text=text))
    return Runner(RunConfig(agent=False), ScriptedProvider(replies)).run_task(task)


def _issued_reservation(run) -> str:
    for call in run.transcript.tool_log:
        match = re.search(r"создана резервация (r-\d+)", str(call.get("result") or ""))
        if match:
            return match.group(1)
    raise AssertionError("эталонный create_booking не выдал id резервации")


def _checks(run) -> dict[str, bool]:
    return {check.name: check.passed for check in run.score.checks}


def test_t11_version_token_anywhere_and_reservation_id_required() -> None:
    task = _task("T-11", 1, "stretch-v1")
    golden = _finish(task, task.oracle["final_answer"])
    reservation_id = _issued_reservation(golden)
    assert reservation_id in task.oracle["reference"]
    assert reservation_id in task.oracle["final_answer"]
    assert golden.score.earned == task.max_points

    spoken = _finish(task, f"версия слота теперь 2, {reservation_id}")
    assert spoken.score.earned == task.max_points, [
        (check.name, check.detail) for check in spoken.score.checks if not check.passed
    ]

    missing = _finish(task, "версия слота теперь 2")
    assert _checks(missing)["state_before_write"] is True
    assert _checks(missing)["single_effect_consistent_remaining"] is True
    assert _checks(missing)["fresh_version_no_stray"] is False
    assert missing.score.earned == 1.5

    # The old proximity regex accepts this spelling; the missing id must still cost the point.
    legacy = _finish(task, "остаток 3, версия слота 2")
    assert _checks(legacy)["fresh_version_no_stray"] is False
    assert legacy.score.earned == 1.5


def test_t12_revision_build_and_effect_count_need_no_phrase_list() -> None:
    task = _task("T-12", 1, "research-v1")
    oracle = task.oracle
    revision, build = 10, "b-1"
    oracle["release_revision"] = revision
    oracle["build"] = build
    oracle["golden_calls"] = [
        {"name": "get_release_state", "arguments": {}},
        {"name": "publish_release", "arguments": {"build_id": build, "revision": revision}},
        {"name": "get_release_state", "arguments": {"refresh": True}},
        {"name": "publish_release", "arguments": {"build_id": build, "revision": revision + 1}},
        {"name": "get_release_state", "arguments": {"refresh": True}},
    ]
    spoken = _finish(task, "ревизия 12, сборка b-1, эффектов 1")
    assert spoken.score.earned == task.max_points, [
        (check.name, check.detail) for check in spoken.score.checks if not check.passed
    ]

    plain = _task("T-12", 1, "research-v1")
    reference = _finish(plain, plain.oracle["final_answer"])
    assert reference.score.earned == plain.max_points, [
        (check.name, check.detail) for check in reference.score.checks if not check.passed
    ]


def test_two_effects_one_repeat_is_not_a_single_effect_and_the_oracle_stays_full() -> None:
    """The count is tied to «эффект»; the oracle still reports one and scores full."""
    task = _task("T-12", 1, "research-v1")
    oracle = task.oracle
    revision, build = int(oracle["release_revision"]), str(oracle["build"])
    spoken = _finish(
        task, f"ревизия {revision + 2}, сборка {build}, два эффекта, один повтор"
    )
    assert _checks(spoken)["single_effect_consistent_state"] is False

    plain = _task("T-12", 1, "research-v1")
    reference = _finish(plain, plain.oracle["final_answer"])
    assert reference.score.earned == plain.max_points


def test_two_places_does_not_satisfy_the_version_token() -> None:
    """«2 места» is not a version; the token has to sit with «верс»."""
    task = _task("T-11", 1, "stretch-v1")
    golden = _finish(task, task.oracle["final_answer"])
    reservation_id = _issued_reservation(golden)
    places = _finish(task, f"2 места, {reservation_id}")
    assert _checks(places)["fresh_version_no_stray"] is False

