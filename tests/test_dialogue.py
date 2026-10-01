"""Category H — multi-turn dialogue: registry, determinism, and the exact split.

The point of the category is that a model which answers turn 1 correctly but
ignores the follow-up scores nothing, while a model that applies the follow-up
but breaks the untouched format keeps exactly one of the two points. These
tests pin both halves of every scenario and prove the runner really replays the
follow-up messages.
"""

from __future__ import annotations

import random

from dailybench import core
from dailybench.config import RunConfig
from dailybench.providers import ScriptedProvider
from dailybench.runner import Runner
from dailybench.tasks import dialogue

SEED = 20260915


def _tasks(seed: int = SEED) -> list[core.TaskInstance]:
    return dialogue.generate(random.Random(seed), RunConfig(seed=seed))


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
    assert _checks(score) == {"final_status": False, "single_word_rule": False}

    # A single allowed word that is the wrong status keeps the rule but loses
    # the artefact point — the two checks really are separable.
    score = _score(task, "degraded")
    assert _checks(score) == {"final_status": False, "single_word_rule": True}
    assert score.earned == 1.0

    # A digit smuggled into the final word breaks the turn-1 rule under pressure.
    score = _score(task, f"{task.oracle['final_status']}1")
    assert _checks(score) == {"final_status": False, "single_word_rule": False}

    assert _score(task, task.oracle["reference"]).earned == 2.0


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
