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


def _by_id(seed: int, task_id: str) -> core.TaskInstance:
    return next(task for task in _tasks(seed) if task.id == task_id)


def test_ids_points_and_registry() -> None:
    tasks = _tasks(20260915)
    assert [task.id for task in tasks] == list(reasoning.IDS)
    assert len(tasks) == 15
    assert sum(task.max_points for task in tasks) == 15.0
    module = core.REGISTRY["B"]
    assert module.expected_ids == reasoning.IDS
    assert module.points == 15.0


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


def test_every_canonical_reference_scores_one() -> None:
    for task in _tasks(20260915):
        score = reasoning.evaluate(task, _transcript(task, task.oracle["reference"]))
        assert score.earned == 1.0
        assert score.possible == 1.0
        assert len(score.checks) == 1
        assert score.checks[0].points == 1.0


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
    assert reasoning.evaluate(r7, _transcript(r7, r7.meta["unreduced"])).earned == 1.0
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


def test_new_scenario_traps() -> None:
    # R-11: the naive answer reports the window's end, not its start.
    r11 = _by_id(20260915, "R-11")
    assert r11.meta["naive_window_end"] != r11.oracle["answer"]
    assert reasoning.evaluate(r11, _transcript(r11, r11.oracle["reference"])).earned == 1.0
    assert reasoning.evaluate(r11, _transcript(r11, str(r11.meta["naive_window_end"]))).earned == 0.0

    # R-12: applying B's standard UTC+1 instead of the post-switch UTC+2.
    r12 = _by_id(20260915, "R-12")
    assert r12.meta["naive_standard_offset"] != r12.oracle["reference"]
    assert reasoning.evaluate(r12, _transcript(r12, r12.oracle["reference"])).earned == 1.0
    assert reasoning.evaluate(r12, _transcript(r12, r12.meta["naive_standard_offset"])).earned == 0.0

    # R-13: both the unreduced and the reduced fraction are valid, while the
    # marginal probability or a changed numerator are rejected.
    r13 = next(
        task
        for seed in range(200)
        for task in _tasks(seed)
        if task.id == "R-13" and task.meta["unreduced"] != task.meta["reduced"]
    )
    assert reasoning.evaluate(r13, _transcript(r13, r13.meta["unreduced"])).earned == 1.0
    assert reasoning.evaluate(r13, _transcript(r13, r13.meta["reduced"])).earned == 1.0
    assert reasoning.evaluate(r13, _transcript(r13, r13.meta["naive_marginal"])).earned == 0.0
    numerator, denominator = r13.meta["unreduced"].split("/")
    assert reasoning.evaluate(r13, _transcript(r13, f"{int(numerator) + 1}/{denominator}")).earned == 0.0

    # R-14: the plausible greedy list schedule is strictly worse than the exact
    # minimum, which takes the release times into account.
    r14 = _by_id(20260915, "R-14")
    assert r14.meta["naive_greedy"] > r14.oracle["answer"]
    assert reasoning.evaluate(r14, _transcript(r14, r14.oracle["reference"])).earned == 1.0
    assert reasoning.evaluate(r14, _transcript(r14, str(r14.meta["naive_greedy"]))).earned == 0.0

    # R-15: a different but still valid topological order must not pass.
    r15 = _by_id(20260915, "R-15")
    assert r15.meta["naive_fifo"] != r15.oracle["reference"]
    assert reasoning.evaluate(r15, _transcript(r15, r15.oracle["reference"])).earned == 1.0
    assert reasoning.evaluate(r15, _transcript(r15, r15.meta["naive_fifo"])).earned == 0.0


def test_packing_oracle_proves_minimum() -> None:
    for task in _tasks(20260915):
        if task.id == "R-10":
            assert task.meta["minimum_nodes"] == task.oracle["answer"]
            assert task.meta["one_less_feasible"] is False
            assert task.meta["expected"] == task.oracle["answer"]


def test_prompt_states_the_answer_contract() -> None:
    """Every B prompt must pin the reply format the evaluator enforces.

    Two run-time failures were formulation defects, not model errors: R-8 never
    said the keys are comma-separated (models answered ``9;C A``), and the
    single-value tasks never demanded a reply without explanations (a model
    answered ``**19/31**`` plus reasoning and failed the one-line contract).
    """
    for task in _tasks(20260915):
        assert "одной строкой без пояснений" in task.prompt, task.id

    r8 = _by_id(20260915, "R-8")
    assert "через запятую" in r8.prompt
    # The format spec must itself parse: the oracle answer is spelled exactly
    # the way the prompt describes it (misses first, keys comma-separated).
    misses, order = r8.oracle["reference"].split(";", 1)
    assert "," in order or len(r8.meta["final_order"]) == 1
