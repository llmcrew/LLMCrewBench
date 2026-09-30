from __future__ import annotations

import random

from dailybench import core
from dailybench.config import RunConfig
from dailybench.tasks import reasoning


def _tasks(seed: int) -> list[core.TaskInstance]:
    return reasoning.generate(random.Random(seed), RunConfig(seed=seed))


def _transcript(task: core.TaskInstance, reply: str) -> core.Transcript:
    return core.Transcript(
        task_id=task.id,
        category="B",
        turns=[
            core.Turn(
                messages=[{"role": "user", "content": task.prompt}],
                reply=core.ModelReply(text=reply),
            )
        ],
    )


def test_ids_points_and_registry() -> None:
    tasks = _tasks(20260915)
    assert [task.id for task in tasks] == list(reasoning.IDS)
    assert sum(task.max_points for task in tasks) == 20.0
    module = core.REGISTRY["B"]
    assert module.expected_ids == reasoning.IDS
    assert module.points == 20.0


def test_generation_is_deterministic_but_seeded() -> None:
    first = _tasks(20260915)
    second = _tasks(20260915)
    other = _tasks(20260916)
    assert [(task.prompt, task.oracle, task.meta) for task in first] == [
        (task.prompt, task.oracle, task.meta) for task in second
    ]
    assert [task.id for task in other] == list(reasoning.IDS)
    assert any(
        (left.prompt, left.oracle, left.meta) != (right.prompt, right.oracle, right.meta)
        for left, right in zip(first, other)
    )


def test_every_canonical_reference_scores_two() -> None:
    for task in _tasks(20260915):
        score = reasoning.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert score.earned == 2.0
        assert score.possible == 2.0
        assert len(score.checks) == 1
        assert score.checks[0].points == 2.0


def test_empty_unknown_multiline_and_extra_text_score_zero() -> None:
    for task in _tasks(20260915):
        assert reasoning.evaluate(task, _transcript(task, "")).earned == 0.0
        assert reasoning.evaluate(task, _transcript(task, "не знаю")).earned == 0.0
        assert reasoning.evaluate(task, _transcript(task, task.oracle["reference"] + "\nобъяснение")).earned == 0.0
        assert reasoning.evaluate(task, _transcript(task, task.oracle["reference"] + " объяснение")).earned == 0.0


def test_reasoning_specific_traps() -> None:
    r1 = _tasks(20260915)[0]
    assert reasoning.evaluate(r1, _transcript(r1, str(r1.oracle["answer"] + 1))).earned == 0.0

    # The unreduced hypergeometric fraction is intentionally still valid,
    # whereas changing its numerator changes the probability.
    r7 = next(
        task
        for seed in range(100)
        for task in _tasks(seed)
        if task.id == "R-7" and task.meta["unreduced"] != task.oracle["reference"]
    )
    assert reasoning.evaluate(r7, _transcript(r7, r7.meta["unreduced"])).earned == 2.0
    numerator, denominator = r7.meta["unreduced"].split("/")
    assert reasoning.evaluate(r7, _transcript(r7, f"{int(numerator) + 1}/{denominator}")).earned == 0.0

    # LRU requires both fields: a right miss count without order and an order
    # with the wrong MRU direction are both rejected.
    r8 = next(
        task
        for seed in range(100)
        for task in _tasks(seed)
        if task.id == "R-8" and len(task.meta["final_order"]) > 1
    )
    misses, order = r8.oracle["reference"].split(";", 1)
    assert reasoning.evaluate(r8, _transcript(r8, misses)).earned == 0.0
    assert reasoning.evaluate(r8, _transcript(r8, f"{misses};{','.join(reversed(order.split(',')))}")).earned == 0.0

    # A non-divisible request count exposes the omitted-final-batch trap.
    r6 = next(
        task
        for seed in range(100)
        for task in _tasks(seed)
        if task.id == "R-6" and task.meta["requests"] % task.meta["batch_size"]
    )
    floor_answer = r6.meta["naive_floor_distractor"]
    assert floor_answer != r6.oracle["answer"]
    assert reasoning.evaluate(r6, _transcript(r6, str(floor_answer))).earned == 0.0


def test_packing_oracle_proves_minimum() -> None:
    for task in _tasks(20260915):
        if task.id == "R-10":
            assert task.meta["minimum_nodes"] == task.oracle["answer"]
            assert task.meta["one_less_feasible"] is False
            assert task.meta["expected"] == task.oracle["answer"]
