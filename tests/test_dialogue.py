"""Category H — multi-turn dialogue: registry, determinism, and the exact split.

The point of the category is that a model which answers turn 1 correctly but
ignores the follow-up scores nothing, while a model that applies the follow-up
but breaks the untouched format keeps exactly one of the two points. These
tests pin both halves of every scenario and prove the runner really replays the
follow-up messages.
"""

from __future__ import annotations

import json
import random

from dailybench import core
from dailybench.config import RunConfig
from dailybench.providers import ScriptedProvider
from dailybench.runner import Runner
from dailybench.tasks import dialogue

SEED = 20260915


def _tasks(seed: int = SEED) -> list[core.TaskInstance]:
    return dialogue.generate(random.Random(seed), RunConfig(seed=seed))


def _profile_tasks(profile: str, seed: int = SEED) -> list[core.TaskInstance]:
    return dialogue.generate(random.Random(seed), RunConfig(seed=seed, profile=profile))


def _by_id(tasks: list[core.TaskInstance]) -> dict[str, core.TaskInstance]:
    return {task.id: task for task in tasks}


def _transcript(task: core.TaskInstance, replies: list[str]) -> core.Transcript:
    """A transcript with one turn per reply, mirroring the runner's history."""
    turns = []
    for index, reply in enumerate(replies):
        user = task.prompt if index == 0 else task.dialogue[index - 1]
        turns.append(
            core.Turn(
                messages=[{"role": "user", "content": user}],
                reply=core.ModelReply(text=reply),
            )
        )
    return core.Transcript(task_id=task.id, category="H", turns=turns)


def _score(task: core.TaskInstance, final: str) -> core.Score:
    return dialogue.evaluate(task, _transcript(task, [final]))


def _score_seq(task: core.TaskInstance, replies: list[str]) -> core.Score:
    """Score a full dialogue, one reply per turn (prompt + follow-ups)."""
    return dialogue.evaluate(task, _transcript(task, replies))


def _checks(score: core.Score) -> dict[str, bool]:
    return {check.name: check.passed for check in score.checks}


# --------------------------------------------------------------------------- #
# Registry and generator contract
# --------------------------------------------------------------------------- #


def test_ids_points_and_registry() -> None:
    tasks = _tasks()
    assert [task.id for task in tasks] == list(dialogue.IDS)
    assert len({task.id for task in tasks}) == 3
    assert all(task.category == "H" for task in tasks)
    assert all(task.max_points == 2.0 for task in tasks)
    assert sum(task.max_points for task in tasks) == 6.0
    assert all(len(task.dialogue or []) >= 1 for task in tasks), "фоллоу-апы обязательны"

    module = core.REGISTRY["H"]
    assert module.expected_ids == dialogue.IDS
    assert module.points == 6.0
    assert module.title == "Multi-turn dialogue"


def test_two_checks_of_one_point_each() -> None:
    for task in _tasks():
        score = _score(task, task.oracle["reference"])
        assert score.possible == task.max_points == 2.0
        assert len(score.checks) == 2
        assert [check.points for check in score.checks] == [1.0, 1.0]


def test_generation_is_deterministic_but_seeded() -> None:
    first = _tasks(SEED)
    second = _tasks(SEED)
    other = _tasks(SEED + 1)
    assert [(t.prompt, t.dialogue, t.oracle, t.meta) for t in first] == [
        (t.prompt, t.dialogue, t.oracle, t.meta) for t in second
    ]
    assert [t.id for t in other] == list(dialogue.IDS)
    assert any(
        (left.prompt, left.oracle, left.meta) != (right.prompt, right.oracle, right.meta)
        for left, right in zip(first, other)
    )


# --------------------------------------------------------------------------- #
# Oracle and garbage
# --------------------------------------------------------------------------- #


def test_reference_conversation_scores_full_points() -> None:
    for task in _tasks():
        replies = [str(text) for text in task.oracle["turn_answers"]]
        assert len(replies) == 1 + len(task.dialogue or [])
        score = dialogue.evaluate(task, _transcript(task, replies))
        assert score.earned == 2.0, task.id
        assert score.possible == 2.0


def test_garbage_scores_zero_everywhere() -> None:
    for task in _tasks():
        score = _score(task, "не знаю")
        assert score.earned == 0.0, task.id
        assert all(not check.passed for check in score.checks), task.id


# --------------------------------------------------------------------------- #
# Per-scenario traps: the split is exact
# --------------------------------------------------------------------------- #


def test_h1_correction_and_format_split() -> None:
    task = _by_id(_tasks())["H-1"]
    naive = f"итог: {task.oracle['cheap']}"
    assert naive != task.oracle["reference"]

    score = _score(task, naive)
    assert score.earned == 0.0
    assert _checks(score) == {"corrected_tariff": False, "format_kept": False}

    corrected_but_verbose = f"итог: {task.oracle['dear']}\nПояснение: этот тариф самый дорогой."
    score = _score(task, corrected_but_verbose)
    assert _checks(score) == {"corrected_tariff": True, "format_kept": False}
    assert score.earned == 1.0

    assert _score(task, task.oracle["reference"]).earned == 2.0


def test_h2_rule_survives_pressure_split() -> None:
    task = _by_id(_tasks())["H-2"]
    naive = task.oracle["naive"]
    assert naive != task.oracle["reference"]

    score = _score(task, naive)
    assert score.earned == 0.0
    assert _checks(score) == {"status_sequence": False, "final_status": False}

    # A single allowed word that is the wrong status keeps the final-form rule
    # but loses the sequence point — the checks are separable.
    score = _score(task, "degraded")
    assert _checks(score) == {"status_sequence": False, "final_status": False}

    # The full correct sequence ok -> degraded -> down -> down passes.
    first, second, third, final = task.oracle["turn_answers"]
    score = _score_seq(task, [first, second, third, final])
    assert _checks(score) == {"status_sequence": True, "final_status": True}
    assert score.earned == 2.0

    # A constant answer ("down" to everything) must not pass: the first
    # statuses are wrong even though the final one is right.
    score = _score_seq(task, [final, final, final, final])
    assert score.earned == 1.0

    # Wrong first move with a correct final keeps only the final point.
    score = _score_seq(task, [second, second, third, final])
    assert _checks(score) == {"status_sequence": False, "final_status": True}
    assert score.earned == 1.0

def test_h3_contradiction_split() -> None:
    task = _by_id(_tasks())["H-3"]
    naive = str(task.oracle["threshold1"])
    assert naive != task.oracle["reference"]

    score = _score(task, naive)
    assert score.earned == 0.0
    assert _checks(score) == {"new_threshold": False, "superseded_values_gone": False}

    # The new value is right but the superseded one still shows up: half credit.
    carries_old = f"{task.oracle['threshold2']} (было {task.oracle['limit1']})"
    score = _score(task, carries_old)
    assert _checks(score) == {"new_threshold": True, "superseded_values_gone": False}
    assert score.earned == 1.0

    reference = task.oracle["reference"]
    score = _score(task, reference)
    assert score.earned == 2.0
    assert str(task.oracle["limit1"]) not in reference
    assert str(task.oracle["threshold1"]) not in reference


# --------------------------------------------------------------------------- #
# The runner really replays the follow-ups
# --------------------------------------------------------------------------- #


def test_runner_sends_every_follow_up() -> None:
    cfg = RunConfig(seed=SEED)
    for task in _tasks():
        replies = [str(text) for text in task.oracle["turn_answers"]]
        provider = ScriptedProvider(replies=list(replies))
        run = Runner(cfg, provider).run_task(task)

        assert run.transcript.error is None, task.id
        assert len(run.transcript.turns) == 1 + len(task.dialogue or []), task.id
        # Every follow-up reached the model as the newest user message.
        last_messages = run.transcript.turns[-1].messages
        assert last_messages[-1] == {"role": "user", "content": task.dialogue[-1]}
        assert len(provider.calls) == 1 + len(task.dialogue or [])
        assert run.score.earned == 2.0, task.id


# --------------------------------------------------------------------------- #
# H-4 / H-5 — pilot and stretch profiles
# --------------------------------------------------------------------------- #


def test_profile_ids_and_registry() -> None:
    assert dialogue.IDS == dialogue.BASELINE_IDS == ("H-1", "H-2", "H-3")
    assert dialogue.PILOT_IDS == ("H-4",)
    assert dialogue.ALL_IDS == dialogue.BASELINE_IDS + ("H-4", "H-5")

    module = core.REGISTRY["H"]
    assert module.expected_ids_for() == dialogue.BASELINE_IDS
    assert module.expected_ids_for("pilot") == dialogue.PILOT_IDS
    assert module.expected_ids_for("stretch-v1") == dialogue.ALL_IDS

    assert [task.id for task in _profile_tasks("pilot")] == ["H-4"]
    assert [task.id for task in _profile_tasks("stretch-v1")] == list(dialogue.ALL_IDS)
    assert [task.id for task in _profile_tasks("baseline-20261020")] == list(dialogue.BASELINE_IDS)
    assert _by_id(_profile_tasks("pilot"))["H-4"].meta["turns"] == 1 + 3
    assert _by_id(_profile_tasks("stretch-v1"))["H-5"].meta["turns"] == 1 + 4


def test_new_scenarios_have_four_unit_checks() -> None:
    for profile, task_id in (("pilot", "H-4"), ("stretch-v1", "H-5")):
        task = _by_id(_profile_tasks(profile))[task_id]
        assert task.max_points == 4.0
        score = _score_seq(task, task.oracle["turn_answers"])
        assert score.possible == task.max_points == 4.0
        assert [check.points for check in score.checks] == [1.0] * 4
        assert len(score.checks) == 4


def test_h4_reference_scores_full_and_naive_zero() -> None:
    task = _by_id(_profile_tasks("pilot"))["H-4"]
    assert task.oracle["naive"] != task.oracle["reference"]
    assert len(task.meta["constraints"]) == 3

    score = _score_seq(task, task.oracle["turn_answers"])
    assert score.earned == 4.0
    assert _checks(score) == {
        "per_turn_answers": True,
        "format_all_turns": True,
        "exclusion_after_amendment": True,
        "final_form_and_constraints": True,
    }

    garbage = _score(task, "не знаю")
    assert garbage.earned == 0.0
    assert all(not check.passed for check in garbage.checks)


def test_h4_only_latest_amendment_forgets_exclusion_and_format() -> None:
    task = _by_id(_profile_tasks("pilot"))["H-4"]
    naive, _, amended, _ = task.oracle["turn_answers"]

    # The rule change is applied, the exclusion and the field order are gone.
    only_amendment = [naive, naive, amended, amended]
    score = _score_seq(task, only_amendment)
    assert _checks(score) == {
        "per_turn_answers": False,
        "format_all_turns": True,
        "exclusion_after_amendment": False,
        "final_form_and_constraints": True,
    }
    assert score.earned == 2.0


def test_h4_right_final_after_wrong_intermediates_is_partial() -> None:
    task = _by_id(_profile_tasks("pilot"))["H-4"]
    first, second, amended, final = task.oracle["turn_answers"]

    score = _score_seq(task, [second, second, first, final])
    assert _checks(score) == {
        "per_turn_answers": False,
        "format_all_turns": True,
        "exclusion_after_amendment": False,
        "final_form_and_constraints": True,
    }
    assert score.earned == 2.0


def test_h4_explanation_must_not_add_lines() -> None:
    task = _by_id(_profile_tasks("pilot"))["H-4"]
    first, second, amended, final = task.oracle["turn_answers"]

    verbose = final + "\nПояснение: слот дольше всех работает без перезапуска."
    score = _score_seq(task, [first, second, amended, verbose])
    assert _checks(score) == {
        "per_turn_answers": True,
        "format_all_turns": False,
        "exclusion_after_amendment": False,
        "final_form_and_constraints": False,
    }
    assert score.earned == 1.0


def test_h4_garbage_on_every_turn_scores_zero() -> None:
    task = _by_id(_profile_tasks("pilot"))["H-4"]
    score = _score_seq(task, ["не знаю"] * (1 + len(task.dialogue or [])))
    assert score.earned == 0.0
    assert _checks(score) == {
        "per_turn_answers": False,
        "format_all_turns": False,
        "exclusion_after_amendment": False,
        "final_form_and_constraints": False,
    }


def test_h4_leftover_exclusion_still_fails_the_exclusion_check() -> None:
    task = _by_id(_profile_tasks("pilot"))["H-4"]
    first, second, amended, final = task.oracle["turn_answers"]
    excluded = task.oracle["excluded"]

    stale = [first, second, amended, final.replace(
        f"итог: {task.oracle['turn_hosts'][-1]}", f"итог: {excluded}"
    )]
    score = _score_seq(task, stale)
    assert _checks(score)["exclusion_after_amendment"] is False
    assert _checks(score)["format_all_turns"] is True


def _states_json(task: core.TaskInstance) -> list[str]:
    return [json.dumps(state, ensure_ascii=False) for state in task.oracle["states"]]


def test_h5_reference_scores_full_and_naive_zero() -> None:
    task = _by_id(_profile_tasks("stretch-v1"))["H-5"]
    assert task.oracle["naive"] != task.oracle["reference"]
    assert task.meta["fields"] == list(dialogue._STATE_FIELDS)

    score = _score_seq(task, task.oracle["turn_answers"])
    assert score.earned == 4.0
    assert _checks(score) == {
        "state_per_turn": True,
        "exception_lifecycle": True,
        "independent_fields_intact": True,
        "final_json_strict": True,
    }

    garbage = _score(task, "не знаю")
    assert garbage.earned == 0.0
    assert all(not check.passed for check in garbage.checks)


def test_h5_leftover_exception_is_partial() -> None:
    task = _by_id(_profile_tasks("stretch-v1"))["H-5"]
    replies = _states_json(task)
    replies[-1] = json.dumps(
        {**task.oracle["states"][-1], "exclude_tag": task.oracle["tag"]}, ensure_ascii=False
    )
    score = _score_seq(task, replies)
    assert _checks(score) == {
        "state_per_turn": False,
        "exception_lifecycle": False,
        "independent_fields_intact": True,
        "final_json_strict": False,
    }
    assert score.earned == 1.0


def test_h5_rollback_of_unrelated_field_is_partial() -> None:
    task = _by_id(_profile_tasks("stretch-v1"))["H-5"]
    replies = _states_json(task)
    replies[-1] = json.dumps(
        {**task.oracle["states"][-1], "min_cpu": task.oracle["quoted_min_cpu"]},
        ensure_ascii=False,
    )
    score = _score_seq(task, replies)
    assert _checks(score) == {
        "state_per_turn": False,
        "exception_lifecycle": True,
        "independent_fields_intact": False,
        "final_json_strict": False,
    }
    assert score.earned == 1.0


def test_h5_quote_taken_as_rule_is_partial() -> None:
    task = _by_id(_profile_tasks("stretch-v1"))["H-5"]
    replies = _states_json(task)
    quoted = {
        **task.oracle["states"][-1],
        "min_cpu": task.oracle["quoted_min_cpu"],
        "limit": task.oracle["old_limit"],
    }
    replies[-1] = json.dumps(quoted, ensure_ascii=False)
    score = _score_seq(task, replies)
    assert _checks(score) == {
        "state_per_turn": False,
        "exception_lifecycle": True,
        "independent_fields_intact": False,
        "final_json_strict": False,
    }
    assert score.earned == 1.0


def test_h5_correct_final_after_wrong_intermediate_keeps_three_points() -> None:
    task = _by_id(_profile_tasks("stretch-v1"))["H-5"]
    replies = _states_json(task)
    replies[2] = replies[1]  # the replacement was not applied on its own turn
    score = _score_seq(task, replies)
    assert _checks(score) == {
        "state_per_turn": False,
        "exception_lifecycle": True,
        "independent_fields_intact": True,
        "final_json_strict": True,
    }
    assert score.earned == 3.0


def test_h5_fenced_final_loses_strict_json_point() -> None:
    task = _by_id(_profile_tasks("stretch-v1"))["H-5"]
    replies = _states_json(task)
    replies[-1] = "```json\n" + replies[-1] + "\n```"
    score = _score_seq(task, replies)
    assert _checks(score) == {
        "state_per_turn": True,
        "exception_lifecycle": True,
        "independent_fields_intact": True,
        "final_json_strict": False,
    }
    assert score.earned == 3.0


def test_new_scenarios_are_seed_deterministic_and_valid() -> None:
    signatures: set[tuple[str, str]] = set()
    for seed in range(SEED, SEED + 5):
        pilot = _profile_tasks("pilot", seed)
        stretch = _profile_tasks("stretch-v1", seed)
        h4 = pilot[0]
        h5 = _by_id(stretch)["H-5"]

        signatures.add((json.dumps(h4.oracle, sort_keys=True), json.dumps(h5.oracle, sort_keys=True)))
        assert 4 <= 1 + len(h4.dialogue or []) <= 5
        assert 4 <= 1 + len(h5.dialogue or []) <= 5
        assert len(h4.meta["constraints"]) <= 3
        assert len(h5.meta["fields"]) == 4

        for task, profile in ((h4, "pilot"), (h5, "stretch-v1")):
            assert task.oracle["turn_answers"][-1] == task.oracle["reference"]
            assert len(task.oracle["turn_answers"]) == 1 + len(task.dialogue or [])
            assert _score_seq(task, task.oracle["turn_answers"]).earned == 4.0
            assert _profile_tasks(profile, seed)[-1].oracle == task.oracle if profile == "pilot" else True

        # Every seed has at least one mutation with genuine partial credit.
        naive_h4, _, amended_h4, _ = h4.oracle["turn_answers"]
        assert 0.0 < _score_seq(h4, [naive_h4, naive_h4, amended_h4, amended_h4]).earned < 4.0

        replies = _states_json(h5)
        replies[-1] = json.dumps(
            {**h5.oracle["states"][-1], "exclude_tag": h5.oracle["tag"]}, ensure_ascii=False
        )
        assert 0.0 < _score_seq(h5, replies).earned < 4.0

    assert len(signatures) > 1, "fixtures обязаны зависеть от seed"


def test_runner_replays_new_dialogues() -> None:
    for profile, task_id in (("pilot", "H-4"), ("stretch-v1", "H-5")):
        task = _by_id(_profile_tasks(profile))[task_id]
        replies = [str(reply) for reply in task.oracle["turn_answers"]]
        provider = ScriptedProvider(replies=list(replies))
        run = Runner(RunConfig(seed=SEED, profile=profile), provider).run_task(task)

        assert run.transcript.error is None, task.id
        assert len(run.transcript.turns) == 1 + len(task.dialogue or []), task.id
        assert provider.calls and provider.calls[-1][-1]["content"] == task.dialogue[-1]
        assert run.score.earned == task.max_points, task.id
