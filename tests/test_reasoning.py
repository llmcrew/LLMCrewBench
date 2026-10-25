from __future__ import annotations

import itertools
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


# --------------------------------------------------------------------------- #
# R-16 — pilot scenario, run on the pilot profile only
# --------------------------------------------------------------------------- #

DEV_SEEDS = (20260915, 20260916, 20260917, 20260918, 20260919)


def _pilot_tasks(seed: int) -> list[core.TaskInstance]:
    return reasoning.generate(random.Random(seed), RunConfig(seed=seed, profile="pilot"))


def _pilot(seed: int) -> core.TaskInstance:
    return _pilot_tasks(seed)[0]


def _r16_nodes(task: core.TaskInstance) -> list[dict]:
    """Rebuild the generator's node graph from the published oracle.

    A retry is a separate object with the same payload; its dependencies and
    the tombstone targets point at the first delivery of that id.
    """
    nodes = [
        {
            "id": event["id"],
            "ts": event["ts"],
            "outcome": event["outcome"],
            "revokes": None,
            "deps": [],
        }
        for event in task.meta["events"]
    ]
    first: dict[str, dict] = {}
    for node in nodes:
        first.setdefault(node["id"], node)
    for node, event in zip(nodes, task.meta["events"]):
        node["deps"] = [first[dep] for dep in event["depends_on"]]
        if event["revokes"]:
            node["revokes"] = first[event["revokes"]]
    return nodes


def test_r16_is_pilot_only_and_registered_by_profile() -> None:
    assert "R-16" not in reasoning.IDS
    assert [task.id for task in _tasks(20260915)] == list(reasoning.IDS)
    assert [task.id for task in _pilot_tasks(20260915)] == ["R-16"]
    stretch = reasoning.generate(
        random.Random(20260915), RunConfig(seed=20260915, profile="stretch-v1")
    )
    assert [task.id for task in stretch] == list(reasoning.STRETCH_IDS)

    module = core.REGISTRY["B"]
    assert module.expected_ids == reasoning.IDS
    assert module.expected_ids_for() == reasoning.IDS
    assert module.expected_ids_for("pilot") == ("R-16",)
    assert module.expected_ids_for("stretch-v1") == reasoning.STRETCH_IDS == reasoning.IDS + (
        "R-16", "R-17", "R-17-KK", "R-18")


def test_r16_reference_scores_full_and_every_mutant_fails() -> None:
    for seed in DEV_SEEDS:
        task = _pilot(seed)
        reference = task.oracle["reference"]
        assert task.max_points == 1.0
        score = reasoning.evaluate(task, _transcript(task, reference))
        assert score.earned == 1.0
        assert score.possible == 1.0
        assert len(score.checks) == 1 and score.checks[0].name == "value"

        for name in ("mutant_double_retry", "mutant_kept_revoked", "mutant_text_order"):
            mutant = task.meta[name]
            assert mutant != reference, name
            assert reasoning.evaluate(task, _transcript(task, mutant)).earned == 0.0, name

        for junk in ("", "не знаю", "active;2", reference + "\nобъяснение", reference + " объяснение"):
            assert reasoning.evaluate(task, _transcript(task, junk)).earned == 0.0, junk


def test_r16_oracle_survives_an_independent_replay() -> None:
    """The published trace is re-proved by the verifier, not by the loop."""
    for seed in DEV_SEEDS:
        task = _pilot(seed)
        nodes = _r16_nodes(task)
        result = reasoning._r16_run(nodes, collapse=True)
        reasoning._r16_verify(nodes, result)
        assert reasoning._r16_signature(result) == task.oracle["reference"]
        # Rerunning the same graph in the mutant modes must move the answer.
        for mutant in (
            reasoning._r16_run(nodes),
            reasoning._r16_run(nodes, collapse=True, keep_revoked=True),
            reasoning._r16_run(nodes, collapse=True, order_key="id"),
        ):
            assert reasoning._r16_signature(mutant) != task.oracle["reference"]


def test_r16_fixture_edge_cases_and_seed_determinism() -> None:
    for seed in DEV_SEEDS:
        first = _pilot(seed)
        second = _pilot(seed)
        assert (first.prompt, first.oracle, first.meta) == (second.prompt, second.oracle, second.meta)

        meta = first.meta
        events = meta["events"]
        unique = meta["unique_events"]
        assert 12 <= unique <= 20
        assert meta["deliveries"] >= 1
        assert meta["revoked"] and meta["blocked"] >= 1 and meta["illegal"] >= 2
        assert len(events) == unique + meta["deliveries"]

        # Retries are byte-identical redeliveries of one event id.
        rows: dict[str, set[tuple]] = {}
        for event in events:
            rows.setdefault(event["id"], set()).add(
                (event["ts"], event["outcome"], event["revokes"], tuple(event["depends_on"]))
            )
        assert len(rows) == unique
        assert any(len(payloads) == 1 and sum(e["id"] == key for e in events) > 1 for key, payloads in rows.items())

        # Every dependency is causally earlier, and tombstones name a real event.
        by_id = {event["id"]: event for event in events}
        assert all(
            by_id[dep]["ts"] < event["ts"] for event in events for dep in event["depends_on"]
        )
        tombstones = [event for event in events if event["outcome"] == "tombstone"]
        assert tombstones
        assert all(tombstone["revokes"] in by_id for tombstone in tombstones)

        # The listing order is neither the id order nor the timestamp order.
        id_order = [event["id"] for event in sorted(events, key=lambda event: event["id"])]
        ts_order = [
            event["id"] for event in sorted(events, key=lambda event: (event["ts"], event["id"]))
        ]
        assert id_order != ts_order

        status, version, count = first.oracle["reference"].split(";")
        assert status in reasoning._R16_STATUSES
        assert (version, count) == (str(int(version)), str(int(count)))
        assert len(meta["applied"]) == int(count)
        assert not set(meta["revoked"]) & set(meta["applied"])
        assert all(event_id in by_id for event_id in meta["applied"])

        # Retry is idempotent: only the applied count moves, never backwards.
        doubled = meta["mutant_double_retry"].split(";")
        assert int(doubled[2]) > int(count)
        assert doubled[0] == status or doubled[1] != version

        assert "tombstone" in first.prompt and "один раз" in first.prompt
        assert "СТАТУС;ВЕРСИЯ;ПРИМЕНЕНО" in first.prompt
        assert "одной строкой без пояснений" in first.prompt


# --------------------------------------------------------------------------- #
# R-18 — bounded optimisation, stretch and private
# --------------------------------------------------------------------------- #

OPTIM_SEEDS = (20260915, 20260916, 20260917)
_OPTIM_LABELS = "ABCDEF"


def _stretch_tasks(seed: int) -> list[core.TaskInstance]:
    return reasoning.generate(
        random.Random(seed), RunConfig(seed=seed, profile="stretch-v1")
    )


def _optim(seed: int) -> core.TaskInstance:
    return next(task for task in _stretch_tasks(seed) if task.id == "R-18")


def _sim(
    seq: list[tuple[int, int]], duration: list[int], release: list[int], edges: list[list[int]]
) -> tuple[int, bool]:
    """Test-side replay, written separately from the module's simulator."""
    free = [0, 0]
    start = [0] * len(duration)
    end = [0] * len(duration)
    for task, machine in seq:
        start[task] = max(free[machine], release[task])
        end[task] = start[task] + duration[task]
        free[machine] = end[task]
    return max(end), all(start[b] >= end[a] for a, b in edges)


def _optimal_schedules(meta: dict) -> set[str]:
    """Enumerate every feasible optimum of the published fixture from scratch."""
    count = len(meta["duration"])
    found: set[str] = set()
    for order in itertools.permutations(range(count)):
        for machines in itertools.product((0, 1), repeat=count):
            seq = list(zip(order, machines))
            makespan, feasible = _sim(seq, meta["duration"], meta["release"], meta["edges"])
            if feasible and makespan == meta["optimal_makespan"]:
                found.add(",".join(f"{_OPTIM_LABELS[task]}@{machine + 1}" for task, machine in seq))
    return found


def _score(task: core.TaskInstance, reply: str) -> float:
    return reasoning.evaluate(task, _transcript(task, reply)).earned


def test_bounded_optimization_is_stretch_only_and_private() -> None:
    assert "R-18" not in reasoning.IDS
    assert reasoning.STRETCH_IDS[-1] == "R-18"
    assert "R-18" not in [task.id for task in _tasks(20260915)]
    assert "R-18" in [task.id for task in _stretch_tasks(20260915)]

    for seed in OPTIM_SEEDS:
        task = _optim(seed)
        assert task.private is True
        assert task.max_points == 1.0
        first, second = _optim(seed), _optim(seed)
        assert (first.prompt, first.oracle, first.meta) == (second.prompt, second.oracle, second.meta)
    assert _optim(OPTIM_SEEDS[0]).prompt != _optim(OPTIM_SEEDS[1]).prompt


def test_bounded_optimization_accepts_every_optimal_schedule() -> None:
    for seed in OPTIM_SEEDS:
        task = _optim(seed)
        reference = task.oracle["reference"]
        assert _score(task, reference) == 1.0
        optimal = _optimal_schedules(task.meta)
        assert len(optimal) >= 3
        assert reference in optimal
        for schedule in optimal:
            assert _score(task, schedule) == 1.0, schedule
        # Case and spacing are the same answer.
        assert _score(task, reference.lower().replace(",", ", ")) == 1.0


def test_bounded_optimization_rejects_every_mutant() -> None:
    for seed in OPTIM_SEEDS:
        task = _optim(seed)
        meta = task.meta
        optimum = meta["optimal_makespan"]

        # Feasible, but leaves time on the table.
        slow = meta["mutant_feasible_slow"]
        slow_seq = reasoning._r18_parse(slow, len(meta["duration"]))
        assert slow_seq is not None
        slow_makespan, slow_ok = _sim(slow_seq, meta["duration"], meta["release"], meta["edges"])
        assert slow_ok is True and slow_makespan > optimum
        assert _score(task, slow) == 0.0

        # Optimal objective, but a dependency starts before its predecessor ends.
        broken = meta["mutant_optimal_infeasible"]
        broken_seq = reasoning._r18_parse(broken, len(meta["duration"]))
        assert broken_seq is not None
        broken_makespan, broken_ok = _sim(broken_seq, meta["duration"], meta["release"], meta["edges"])
        assert broken_ok is False and broken_makespan == optimum
        assert _score(task, broken) == 0.0

        # Structurally incomplete or malformed answers stay rejected even when
        # the makespan they describe would be optimal.
        parts = task.oracle["reference"].split(",")
        assert _score(task, ",".join(parts[:-1])) == 0.0
        assert _score(task, ",".join([parts[0], *parts])) == 0.0
        assert _score(task, task.oracle["reference"].replace("@1", "@3")) == 0.0
        assert _score(task, "не знаю") == 0.0


def test_bounded_optimization_oracle_is_independently_proven() -> None:
    for seed in OPTIM_SEEDS:
        task = _optim(seed)
        meta = task.meta
        count = len(meta["duration"])
        assert count in (5, 6)
        assert 1 <= len(meta["edges"]) <= 3
        assert all(0 <= source < target < count for source, target in meta["edges"])

        brute = min(
            makespan
            for order in itertools.permutations(range(count))
            for machines in itertools.product((0, 1), repeat=count)
            for seq in [list(zip(order, machines))]
            for makespan, feasible in [_sim(seq, meta["duration"], meta["release"], meta["edges"])]
            if feasible
        )
        assert brute == meta["optimal_makespan"] == task.oracle["answer"]["optimum"]
        assert meta["optimal_schedule"] == task.oracle["reference"]

        # The prompt states every constraint and the acceptance rule the
        # evaluator applies.
        prompt = task.prompt
        assert "2 одинаковые машины" in prompt
        assert "ровно один раз" in prompt
        assert "подряд" in prompt and "раньше своего момента выпуска" in prompt
        assert "любое оптимальное расписание" in prompt
        assert "ЗАДАЧА@МАШИНА" in prompt
        for index, label in enumerate(_OPTIM_LABELS[:count]):
            assert f"Задача {label}: выпущена в {meta['release'][index]} мин" in prompt
        for source, target in meta["edges"]:
            assert (
                f"Задача {_OPTIM_LABELS[source]} должна завершиться "
                f"до старта задачи {_OPTIM_LABELS[target]}"
            ) in prompt
