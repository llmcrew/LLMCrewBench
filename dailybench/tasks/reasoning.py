"""Category B — deterministic numeric reasoning scenarios.

Each scenario has one all-or-nothing check worth one point.  The values in a
fixture are generated from the supplied random stream, while the answer is
computed here before the prompt is handed to a model.

The fifteen scenarios cover queueing (R-1), critical paths (R-2), subnets
(R-3), token cost (R-4), bit operations (R-5), batching (R-6), probability
(R-7), LRU state (R-8), retry timelines (R-9), bin packing (R-10), free-slot
scheduling (R-11), timezone/DST arithmetic (R-12), conditional probability
(R-13), constrained makespan (R-14) and topological tie-breaking (R-15).
"""

from __future__ import annotations

import heapq
import ipaddress
import random
import re
from collections import deque
from decimal import Decimal
from fractions import Fraction
from itertools import permutations, product
from math import comb
from typing import Any, Callable

from .. import core, text
from ..config import RunConfig

IDS = (
    "R-1",
    "R-2",
    "R-3",
    "R-4",
    "R-5",
    "R-6",
    "R-7",
    "R-8",
    "R-9",
    "R-10",
    "R-11",
    "R-12",
    "R-13",
    "R-14",
    "R-15",
)
_POINTS = 1.0


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _minutes_word(n: int) -> str:
    if 11 <= n % 100 <= 14:
        return "минут"
    return {1: "минута", 2: "минуты", 3: "минуты", 4: "минуты"}.get(n % 10, "минут")


def _hhmm(minutes: int) -> str:
    """Format minutes from midnight as a fixed ``HH:MM`` clock value."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _task(
    scenario: str,
    title: str,
    prompt: str,
    answer: Any,
    *,
    kind: str,
    reference: str | None = None,
    unit_whitelist: tuple[str, ...] = (),
    meta: dict[str, Any] | None = None,
) -> core.TaskInstance:
    """Build a single-check reasoning task and retain the oracle's native value."""
    canonical = str(answer) if reference is None else reference
    oracle = {
        "reference": canonical,
        "answer": answer,
        "kind": kind,
        "unit_whitelist": unit_whitelist,
    }
    return core.TaskInstance(
        id=scenario,
        category="B",
        title=title,
        prompt=prompt.rstrip("\n") + "\n\nВерни ответ одной строкой без пояснений.",
        max_points=_POINTS,
        oracle=oracle,
        meta={"scenario": scenario, "kind": kind, "expected": answer, **(meta or {})},
    )


def _int_task(
    scenario: str,
    title: str,
    prompt: str,
    answer: int,
    meta: dict[str, Any],
    *,
    units: tuple[str, ...] = (),
) -> core.TaskInstance:
    return _task(scenario, title, prompt, answer, kind="int", unit_whitelist=units, meta=meta)


# --------------------------------------------------------------------------- #
# R-1 — GPU worker queue
# --------------------------------------------------------------------------- #


def r1(rng: random.Random) -> core.TaskInstance:
    total = rng.randrange(120, 320)
    rate_a = rng.randrange(5, 9)
    rate_b = rng.randrange(3, 7)
    rate_c = rng.randrange(3, 6)
    a_down = rng.randrange(4, 9)
    b_window = rng.randrange(a_down + 4, a_down + 14)

    # Minute zero is the initial state.  A worker available in minute m does
    # work during (m - 1, m], so "the first n full minutes" is represented by
    # m <= n.  Stop as soon as the queue reaches zero at a minute boundary.
    remaining = total
    minute = 0
    timeline: list[dict[str, int]] = []
    while remaining:
        minute += 1
        capacity = 0
        if minute > a_down:
            capacity += rate_a
        if minute <= b_window:
            capacity += rate_b
        capacity += rate_c
        done = min(remaining, capacity)
        remaining -= done
        timeline.append({"minute": minute, "capacity": capacity, "done": done, "remaining": remaining})
        if minute > 1000:  # defensive bound; generated rates make this unreachable
            raise AssertionError("queue did not empty")

    assert remaining == 0
    assert timeline[-1]["remaining"] == 0
    assert all(item["remaining"] > 0 for item in timeline[:-1])
    answer = minute
    prompt = (
        f"В момент t=0 в очереди находится {total} задач.\n\n"
        f"Worker A обрабатывает {rate_a} задач в минуту,\n"
        f"но первые {a_down} полных минут недоступен.\n\n"
        f"Worker B обрабатывает {rate_b} задач в минуту,\n"
        f"но работает только первые {b_window} минут.\n\n"
        f"Worker C обрабатывает {rate_c} задач в минуту\n"
        "и работает постоянно.\n\n"
        "Все workers начинают и заканчивают задачи\n"
        "на границах минут.\n\n"
        "Через сколько полных минут очередь впервые станет пустой?\n\n"
        "Ответь только целым числом."
    )
    return _int_task(
        "R-1",
        "Очередь GPU workers",
        prompt,
        answer,
        {
            "total": total,
            "rates": {"a": rate_a, "b": rate_b, "c": rate_c},
            "a_down": a_down,
            "b_window": b_window,
            "timeline": timeline,
            "previous_remaining": timeline[-2]["remaining"] if len(timeline) > 1 else total,
            "distractors": [answer - 1, answer + 1],
        },
        units=("минут", "минуты", "минута", "min", "minutes", "minute"),
    )


# --------------------------------------------------------------------------- #
# R-2 — DAG critical path
# --------------------------------------------------------------------------- #


_SHAPES: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = (
    (("A", ()), ("B", ()), ("C", ("A",)), ("D", ("A", "B")), ("E", ("C",)), ("F", ("D", "E"))),
    (("A", ()), ("B", ("A",)), ("C", ("A",)), ("D", ("B",)), ("E", ("C", "D")), ("F", ("E",))),
    (("A", ()), ("B", ()), ("C", ("A", "B")), ("D", ("C",)), ("E", ("C",)), ("F", ("D", "E"))),
    (("A", ()), ("B", ()), ("C", ()), ("D", ("A", "B")), ("E", ("B", "C")), ("F", ("D", "E"))),
)


def r2(rng: random.Random) -> core.TaskInstance:
    shape = rng.choice(_SHAPES)
    durations = {name: rng.randrange(2, 10) for name, _ in shape}

    # Longest path in the DAG, calculated recursively from each node rather
    # than by summing all durations.  Shapes are topologically ordered, but
    # recursion also makes the oracle independent of that presentation order.
    dependencies = {name: deps for name, deps in shape}
    finish: dict[str, int] = {}

    def finish_time(name: str) -> int:
        if name not in finish:
            finish[name] = durations[name] + max(
                (finish_time(dep) for dep in dependencies[name]), default=0
            )
        return finish[name]

    for name, _ in shape:
        finish_time(name)
    answer = max(finish.values())

    lines = []
    for name, deps in shape:
        suffix = f", начинается после {', '.join(deps)}" if deps else ""
        lines.append(f"{name} = {durations[name]} {_minutes_word(durations[name])}{suffix}")
    prompt = (
        "Есть задачи:\n\n"
        + "\n".join(lines)
        + "\n\nКоличество исполнителей не ограничено.\n"
        "Задача начинается сразу после выполнения всех зависимостей.\n\n"
        "За сколько минут выполнится весь DAG?\n\n"
        "Ответь только числом."
    )
    return _int_task(
        "R-2",
        "Критический путь DAG",
        prompt,
        answer,
        {
            "shape": [{"name": name, "dependencies": list(deps)} for name, deps in shape],
            "durations": durations,
            "per_task_finish": finish,
            "critical_path": answer,
            "naive_sum_distractor": sum(durations.values()),
        },
        units=("минут", "минуты", "минута", "min", "minutes", "minute"),
    )


# --------------------------------------------------------------------------- #
# R-3 — /26 subnet of a /20
# --------------------------------------------------------------------------- #


def r3(rng: random.Random) -> core.TaskInstance:
    # The prefixes are deliberately fixed by the scenario: a /26 inside a /20.
    first_octet = rng.choice((10, 172, 192))
    second_octet = rng.randrange(0, 256)
    third_octet = rng.randrange(0, 16) * 16
    outer = ipaddress.ip_network(f"{first_octet}.{second_octet}.{third_octet}.0/20")

    subnet_size = 1 << (32 - 26)
    subnet_index = rng.randrange(outer.num_addresses // subnet_size)
    inside_offset = subnet_index * subnet_size + rng.randrange(1, subnet_size - 1)
    inside = ipaddress.ip_address(int(outer.network_address) + inside_offset)
    subnet = ipaddress.ip_network(f"{inside}/26", strict=False)
    answer = [
        str(subnet.network_address),
        str(subnet.broadcast_address),
        str(subnet.network_address + 1),
        str(subnet.broadcast_address - 1),
    ]
    assert subnet.subnet_of(outer)
    assert inside in subnet
    prompt = (
        f"Дана сеть {outer.network_address}/20.\n\n"
        "Определи /26-подсеть, содержащую IP:\n"
        f"{inside}\n\n"
        "Верни ТОЛЬКО в точном порядке:\n\n"
        "NETWORK,BROADCAST,FIRST_HOST,LAST_HOST"
    )
    return _task(
        "R-3",
        "Подсеть",
        prompt,
        answer,
        kind="csv4",
        reference=",".join(answer),
        meta={"outer": str(outer), "subnet": str(subnet), "ip": str(inside)},
    )


# --------------------------------------------------------------------------- #
# R-4 — token cost
# --------------------------------------------------------------------------- #


def r4(rng: random.Random) -> core.TaskInstance:
    input_m = Decimal(rng.randrange(5, 100)) / Decimal(10)
    output_m = Decimal(rng.randrange(2, 40)) / Decimal(10)
    price_in = Decimal(rng.randrange(10, 60)) / Decimal(100)
    price_out = Decimal(rng.randrange(40, 140)) / Decimal(100)
    exact_total = input_m * price_in + output_m * price_out
    total = exact_total.quantize(Decimal("0.001"))
    assert total == exact_total  # one decimal million × two decimal price is exact to 3 places
    reference = f"{total:.3f}"
    prompt = (
        "За один день было использовано:\n\n"
        f"{input_m} млн input tokens\n"
        f"{output_m} млн output tokens\n\n"
        "Цена:\n"
        f"input = ${price_in} за 1 млн\n"
        f"output = ${price_out} за 1 млн\n\n"
        "Какова итоговая цена?\n\n"
        "Ответь только числом в долларах с тремя знаками после точки."
    )
    return _task(
        "R-4",
        "Цена токенов",
        prompt,
        total,
        kind="decimal3",
        reference=reference,
        meta={
            "input_m": str(input_m),
            "output_m": str(output_m),
            "price_in": str(price_in),
            "price_out": str(price_out),
            "exact_total": str(exact_total),
        },
        unit_whitelist=("$", "доллар", "доллара", "долларов", "USD"),
    )


# --------------------------------------------------------------------------- #
# R-5 — bit operations
# --------------------------------------------------------------------------- #


def r5(rng: random.Random) -> core.TaskInstance:
    x = rng.randrange(0x80, 0x100)
    xor_mask = rng.randrange(0x21, 0x100)
    and_mask = rng.choice((0xF0, 0xE0, 0xC0, 0x3C))
    shift = rng.randrange(1, 5)

    after_xor = x ^ xor_mask
    after_and = after_xor & and_mask
    answer = after_and >> shift
    prompt = (
        f"x = {hex(x).upper().replace('0X', '0x')}\n\n"
        "Последовательно выполнить:\n\n"
        f"x = x XOR {hex(xor_mask).upper().replace('0X', '0x')}\n"
        f"x = x AND {hex(and_mask).upper().replace('0X', '0x')}\n"
        f"x = x >> {shift}\n\n"
        "Каково итоговое значение x в десятичной системе?\n\n"
        "Ответь только числом."
    )
    return _int_task(
        "R-5",
        "Битовые операции",
        prompt,
        answer,
        {"x": x, "xor": xor_mask, "and": and_mask, "shift": shift, "steps": [after_xor, after_and, answer]},
    )


# --------------------------------------------------------------------------- #
# R-6 — batching
# --------------------------------------------------------------------------- #


def r6(rng: random.Random) -> core.TaskInstance:
    requests = rng.randrange(200, 2000)
    batch_size = rng.choice((16, 32, 48, 64, 100, 128))
    overhead = rng.randrange(50, 200)
    per_request = rng.randrange(3, 15)
    batches = (requests + batch_size - 1) // batch_size
    answer = batches * overhead + requests * per_request

    prompt = (
        f"Нужно последовательно обработать {requests} запросов.\n\n"
        f"Максимальный batch = {batch_size} запроса.\n\n"
        "Каждый batch занимает:\n\n"
        f"{overhead} мс фиксированного overhead\n+\n"
        f"{per_request} мс за каждый запрос внутри batch.\n\n"
        "Следующий batch начинается только после окончания предыдущего.\n\n"
        "Сколько миллисекунд потребуется для обработки всех запросов?\n\n"
        "Ответь только целым числом."
    )
    return _int_task(
        "R-6",
        "Batching",
        prompt,
        answer,
        {
            "requests": requests,
            "batch_size": batch_size,
            "batches": batches,
            "overhead_ms": overhead,
            "per_request_ms": per_request,
            "naive_floor_distractor": (requests // batch_size) * overhead + requests * per_request,
        },
        units=("ms", "мс", "миллисекунд", "миллисекунды", "миллисекунда"),
    )


# --------------------------------------------------------------------------- #
# R-7 — hypergeometric probability
# --------------------------------------------------------------------------- #


def r7(rng: random.Random) -> core.TaskInstance:
    total = rng.randrange(14, 42)
    damaged = rng.randrange(3, max(4, total // 2))
    picks = rng.randrange(2, 6)
    numerator = comb(damaged, 1) * comb(total - damaged, picks - 1)
    denominator = comb(total, picks)
    exact = Fraction(numerator, denominator)
    reference = f"{exact.numerator}/{exact.denominator}"

    prompt = (
        f"В наборе {total} файлов:\n"
        f"{damaged} повреждённых и {total - damaged} исправных.\n\n"
        f"Случайно без возвращения выбираются {picks} файла.\n\n"
        "Какова вероятность выбрать ровно один повреждённый файл?\n\n"
        "Ответ дай только несократимой дробью вида a/b."
    )
    return _task(
        "R-7",
        "Вероятность",
        prompt,
        exact,
        kind="fraction",
        reference=reference,
        meta={
            "total": total,
            "damaged": damaged,
            "healthy": total - damaged,
            "picks": picks,
            "unreduced": f"{numerator}/{denominator}",
        },
    )


# --------------------------------------------------------------------------- #
# R-8 — LRU cache
# --------------------------------------------------------------------------- #


def r8(rng: random.Random) -> core.TaskInstance:
    capacity = rng.randrange(2, 5)
    alphabet = [chr(ord("A") + i) for i in range(rng.randrange(4, 7))]
    sequence = [rng.choice(alphabet) for _ in range(rng.randrange(8, 15))]

    order: list[str] = []  # most recently used first
    misses = 0
    for key in sequence:
        if key in order:
            order.remove(key)
        else:
            misses += 1
        order.insert(0, key)
        if len(order) > capacity:
            order.pop()
    answer = f"{misses};{','.join(order)}"

    prompt = (
        f"LRU-кэш вмещает {capacity} ключа.\n\n"
        "Изначально он пуст.\n\n"
        "Последовательность обращений:\n\n"
        + " ".join(sequence)
        + "\n\nПри miss ключ добавляется.\n"
        "При попадании ключ становится самым недавно использованным.\n"
        "При переполнении удаляется Least Recently Used.\n\n"
        "Верни только одну строку формата:\n\n"
        "MISSES;K1,K2,... — сначала число промахов, затем ключи через запятую "
        "от самого свежего к самому старому"
    )
    return _task(
        "R-8",
        "LRU",
        prompt,
        answer,
        kind="lru",
        reference=answer,
        meta={"capacity": capacity, "sequence": sequence, "misses": misses, "final_order": list(order)},
    )


# --------------------------------------------------------------------------- #
# R-9 — retry / backoff timeline
# --------------------------------------------------------------------------- #


def r9(rng: random.Random) -> core.TaskInstance:
    attempts = 4  # exactly the demo's maximum: three failures and one success
    timeout_s = rng.randrange(2, 6)
    backoff_base = rng.choice((1, 2, 3))
    backoffs = [backoff_base, 2 * backoff_base, 4 * backoff_base]
    success_s = rng.randrange(1, 5)

    # Explicit events prevent accidentally omitting the wait between a failed
    # attempt and its retry.
    events: list[tuple[str, int]] = []
    for wait in backoffs:
        events.append(("timeout", timeout_s))
        events.append(("backoff", wait))
    events.append(("success", success_s))
    elapsed = 0
    for _, duration in events:
        elapsed += duration
    assert events[-1][0] == "success"
    answer = elapsed

    waits = "\n".join(f"после {i}-й неудачи ожидание {wait} секунд;" for i, wait in enumerate(backoffs, 1))
    prompt = (
        "Не более 4 попыток.\n\n"
        f"Первые 3 попытки завершаются timeout ровно через {timeout_s} секунд каждая.\n\n"
        f"После них:\n{waits}\n\n"
        f"4-я попытка успешна и занимает {success_s} секунды.\n\n"
        "Первая попытка начинается в t=0.\n\n"
        "В какой момент времени запрос успешно завершится?\n\n"
        "Ответь только числом секунд."
    )
    return _int_task(
        "R-9",
        "Retry/backoff",
        prompt,
        answer,
        {
            "attempts": attempts,
            "timeout_s": timeout_s,
            "backoff_base_s": backoff_base,
            "backoffs": backoffs,
            "success_s": success_s,
            "events": events,
            "without_backoff_distractor": 3 * timeout_s + success_s,
        },
        units=("секунд", "секунды", "секунда", "сек", "s", "seconds", "second"),
    )


# --------------------------------------------------------------------------- #
# R-10 — Kubernetes packing
# --------------------------------------------------------------------------- #


_POD_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((3, 5), (3, 5), (3, 5), (3, 5), (2, 7), (2, 7), (2, 7)),
    ((4, 6), (4, 6), (2, 3), (2, 3), (1, 4), (1, 4), (1, 4)),
    ((2, 4), (2, 4), (2, 4), (3, 6), (3, 6), (1, 2), (1, 2)),
    ((5, 8), (3, 5), (3, 5), (2, 4), (2, 4), (2, 4), (1, 1)),
)


def _can_pack(pods: list[tuple[int, int]], node_cpu: int, node_ram: int, nodes: int) -> bool:
    """Exhaustively place a small pod set, skipping equivalent node states."""
    if nodes < 1:
        return not pods
    ordered = sorted(pods, key=lambda pod: (-(pod[0] * node_ram + pod[1] * node_cpu), -pod[0], -pod[1]))
    used = [[0, 0] for _ in range(nodes)]

    def place(index: int) -> bool:
        if index == len(ordered):
            return True
        pod_cpu, pod_ram = ordered[index]
        tried_states: set[tuple[int, int]] = set()
        for node in range(nodes):
            state = (used[node][0], used[node][1])
            if state in tried_states:
                continue
            tried_states.add(state)
            if used[node][0] + pod_cpu > node_cpu or used[node][1] + pod_ram > node_ram:
                continue
            used[node][0] += pod_cpu
            used[node][1] += pod_ram
            if place(index + 1):
                return True
            used[node][0] -= pod_cpu
            used[node][1] -= pod_ram
        return False

    return place(0)


def _min_nodes(pods: list[tuple[int, int]], node_cpu: int, node_ram: int) -> int:
    lower = max(
        1,
        (sum(cpu for cpu, _ in pods) + node_cpu - 1) // node_cpu,
        (sum(ram for _, ram in pods) + node_ram - 1) // node_ram,
    )
    for nodes in range(lower, len(pods) + 1):
        if _can_pack(pods, node_cpu, node_ram, nodes):
            return nodes
    raise AssertionError("generated pods cannot be packed")


def r10(rng: random.Random) -> core.TaskInstance:
    # Retry only degenerate one-node draws; all generated fixtures remain tiny
    # enough that the exact search below is effectively instantaneous.
    for _ in range(100):
        node_cpu = rng.choice((6, 8, 10, 12, 16))
        node_ram = rng.choice((12, 16, 24, 32))
        pods = list(rng.choice(_POD_SHAPES))
        if any(cpu > node_cpu or ram > node_ram for cpu, ram in pods):
            continue
        answer = _min_nodes(pods, node_cpu, node_ram)
        if answer > 1:
            break
    else:
        raise AssertionError("could not generate a non-degenerate packing")

    # These are part of fixture validation, not assumptions about the model's
    # answer: prove both feasibility and minimality for every generated task.
    feasible = _can_pack(pods, node_cpu, node_ram, answer)
    one_less_feasible = _can_pack(pods, node_cpu, node_ram, answer - 1)
    assert feasible
    assert not one_less_feasible

    counts: dict[tuple[int, int], int] = {}
    for pod in pods:
        counts[pod] = counts.get(pod, 0) + 1
    lines = "\n".join(
        f"{count} pod типа {chr(ord('P') + index)}:\n{cpu} CPU, {ram} GiB каждый\n"
        for index, ((cpu, ram), count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[0][0], -item[0][1]))
        )
    )
    prompt = (
        f"Каждый node имеет:\n\n{node_cpu} CPU\n{node_ram} GiB RAM\n\n"
        f"Нужно разместить:\n\n{lines}\n"
        "Pod нельзя делить между nodes.\n"
        "Requests нельзя превышать.\n\n"
        "Какое минимальное количество nodes необходимо?\n\n"
        "Ответь только целым числом."
    )
    return _int_task(
        "R-10",
        "Kubernetes packing",
        prompt,
        answer,
        {
            "node": {"cpu": node_cpu, "ram": node_ram},
            "pods": [{"cpu": cpu, "ram": ram} for cpu, ram in pods],
            "sum_cpu": sum(cpu for cpu, _ in pods),
            "sum_ram": sum(ram for _, ram in pods),
            "minimum_nodes": answer,
            "one_less_feasible": one_less_feasible,
        },
        units=("nodes", "node", "нод", "нода", "ноды"),
    )


# --------------------------------------------------------------------------- #
# R-11 — longest free interval
# --------------------------------------------------------------------------- #

_DAY_START = 8 * 60
_DAY_END = 20 * 60


def _free_slots(busy: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return ``(start, duration)`` for every gap between merged busy ranges."""
    slots: list[tuple[int, int]] = []
    cursor = _DAY_START
    for start, end in busy:
        slots.append((cursor, start - cursor))
        cursor = end
    slots.append((cursor, _DAY_END - cursor))
    return slots


def r11(rng: random.Random) -> core.TaskInstance:
    minimum = rng.choice((40, 45, 50, 60))
    # Retry until the longest qualifying window is unique and strictly longer
    # than the runner-up, so the answer is a single unambiguous minute.
    for _ in range(300):
        busy: list[tuple[int, int]] = []
        cursor = _DAY_START + rng.randrange(0, 30)
        for _ in range(rng.randrange(3, 6)):
            start = cursor + rng.randrange(15, 120)
            duration = rng.randrange(30, 100)
            end = start + duration
            if end > _DAY_END - 90:
                break
            busy.append((start, end))
            cursor = end
        if len(busy) < 3:
            continue
        slots = _free_slots(busy)
        qualifying = [slot for slot in slots if slot[1] >= minimum]
        if not qualifying:
            continue
        best = max(qualifying, key=lambda slot: slot[1])
        if sum(1 for slot in qualifying if slot[1] == best[1]) != 1:
            continue
        answer = best[0]
        # A model that reports the end of the window instead of its start
        # misses the mark; the exact end is the plausible distractor.
        naive = best[0] + best[1]
        assert naive != answer
        break
    else:
        raise AssertionError("could not generate a unique longest free slot")
    assert sum(1 for slot in qualifying if slot[1] == best[1]) == 1, "longest slot must be unique"

    ranges = "\n".join(f"{_hhmm(start)}–{_hhmm(end)}" for start, end in busy)
    prompt = (
        "Рабочий день длится с 08:00 до 20:00.\n\n"
        f"Зал занят в следующие интервалы:\n{ranges}\n\n"
        "Интервалы не пересекаются и уже объединены.\n\n"
        f"Найди самое длинное свободное окно длительностью не менее {minimum} минут.\n\n"
        "Если такого окна нет, ответь ровно словом: нет\n\n"
        "Верни время начала этого окна в минутах от 00:00.\n\n"
        "Ответь только целым числом."
    )
    return _int_task(
        "R-11",
        "Свободное окно",
        prompt,
        answer,
        {
            "day": {"start": _DAY_START, "end": _DAY_END},
            "busy": [{"start": start, "end": end} for start, end in busy],
            "minimum_slot": minimum,
            "slots": [{"start": start, "duration": duration} for start, duration in slots],
            "best_window": {"start": best[0], "duration": best[1]},
            "naive_window_end": naive,
        },
        units=("минут", "минуты", "минута", "min", "minutes", "minute"),
    )


# --------------------------------------------------------------------------- #
# R-12 — timezone / DST arrival
# --------------------------------------------------------------------------- #

_A_OFFSET_H = 3  # origin zone, UTC+3 all year
_B_STANDARD_H = 1  # arrival zone standard offset
_B_DST_H = 2  # arrival zone offset after the spring-forward switch
_SWITCH_LOCAL = 2 * 60  # 02:00 local becomes 03:00 on the arrival night


def r12(rng: random.Random) -> core.TaskInstance:
    # The arrival zone has already switched to DST, so the local arrival time
    # must be converted with UTC+2.  Using the standard UTC+1 is the trap.
    for _ in range(300):
        departure_local = rng.randrange(5 * 60, 12 * 60, 5)
        duration = rng.randrange(90, 400, 5)
        departure_utc = departure_local - _A_OFFSET_H * 60
        arrival_utc = departure_utc + duration
        arrival_local = arrival_utc + _B_DST_H * 60
        if departure_utc < 0 or arrival_utc < 0 or arrival_utc >= 24 * 60:
            continue
        if arrival_local >= 24 * 60 or arrival_local < _SWITCH_LOCAL + 60:
            continue
        break
    else:
        raise AssertionError("could not generate a same-day DST arrival")

    answer = _hhmm(arrival_utc)
    naive = _hhmm(arrival_local - _B_STANDARD_H * 60)
    assert naive != answer
    assert arrival_local - _B_DST_H * 60 == arrival_utc
    assert arrival_utc - departure_utc == duration
    prompt = (
        f"Самолёт вылетел из города A в {_hhmm(departure_local)} по местному времени.\n"
        f"A всегда живёт по UTC+{_A_OFFSET_H}.\n\n"
        f"Он приземлился в городе B в {_hhmm(arrival_local)} по местному времени.\n\n"
        f"B зимой живёт по UTC+{_B_STANDARD_H}, но в ночь перед прилётом\n"
        "в 02:00 местного времени часы перевели на 1 час вперёд (02:00 → 03:00).\n\n"
        "Во сколько по UTC самолёт приземлился?\n\n"
        "Ответь только временем в формате HH:MM."
    )
    return _task(
        "R-12",
        "Часовые пояса и DST",
        prompt,
        answer,
        kind="hm",
        reference=answer,
        meta={
            "departure_local": _hhmm(departure_local),
            "arrival_local": _hhmm(arrival_local),
            "duration_min": duration,
            "a_offset_h": _A_OFFSET_H,
            "b_standard_h": _B_STANDARD_H,
            "b_dst_h": _B_DST_H,
            "arrival_utc": answer,
            "naive_standard_offset": naive,
        },
    )


# --------------------------------------------------------------------------- #
# R-13 — conditional probability
# --------------------------------------------------------------------------- #


def r13(rng: random.Random) -> core.TaskInstance:
    # Retry until the marginal probability differs from the conditional one,
    # so the naive distractor is genuinely rejected.
    for _ in range(300):
        total = rng.randrange(40, 120)
        both = rng.randrange(5, 20)
        python_only = rng.randrange(5, 20)
        sql_only = rng.randrange(5, 20)
        python = both + python_only
        sql = both + sql_only
        if python + sql - both > total:
            continue
        answer = Fraction(both, sql)
        naive = f"{python}/{total}"
        if Fraction(naive) == answer:
            continue
        break
    else:
        raise AssertionError("could not generate a distinct marginal distractor")
    assert python + sql - both <= total, "counts must fit inside the company"
    assert Fraction(naive) != answer, "the marginal distractor must differ"

    # P(Python | SQL) = both / sql, kept as the unreduced fraction on purpose.
    reference = f"{both}/{sql}"
    reduced = f"{answer.numerator}/{answer.denominator}"
    prompt = (
        f"В компании {total} сотрудников.\n\n"
        f"{python} знают Python,\n"
        f"{sql} знают SQL,\n"
        f"{both} знают и Python, и SQL.\n\n"
        "Случайно выбирают сотрудника, про которого известно,\n"
        "что он знает SQL.\n\n"
        "Какова вероятность, что он знает Python?\n\n"
        "Ответ дай точной дробью вида a/b (сокращать не обязательно)."
    )
    return _task(
        "R-13",
        "Условная вероятность",
        prompt,
        answer,
        kind="fraction",
        reference=reference,
        meta={
            "total": total,
            "python": python,
            "sql": sql,
            "both": both,
            "reduced": reduced,
            "unreduced": reference,
            "naive_marginal": naive,
        },
    )


# --------------------------------------------------------------------------- #
# R-14 — constrained makespan on identical machines
# --------------------------------------------------------------------------- #

_R14_MACHINES = 3


def _r14_completion(subset: tuple[int, ...], release: list[int], duration: list[int], setup: int) -> int:
    """Minimal finish of one machine's front setup plus this exact job subset."""
    best: int | None = None
    for perm in permutations(subset):
        available = setup
        for job in perm:
            available = max(available, release[job]) + duration[job]
        if best is None or available < best:
            best = available
    return 0 if best is None else best


def _r14_makespan(release: list[int], duration: list[int], setup: int) -> int:
    n = len(release)
    cache: dict[tuple[int, ...], int] = {}

    def completion(subset: tuple[int, ...]) -> int:
        if subset not in cache:
            cache[subset] = _r14_completion(subset, release, duration, setup)
        return cache[subset]

    best: int | None = None
    # Each machine is independent, so its subset's minimal finish can be
    # minimised separately; the makespan is the maximum over used machines.
    for colors in product(range(_R14_MACHINES), repeat=n):
        per_machine: list[list[int]] = [[] for _ in range(_R14_MACHINES)]
        for job, machine in enumerate(colors):
            per_machine[machine].append(job)
        makespan = max((completion(tuple(subset)) for subset in per_machine if subset), default=0)
        if best is None or makespan < best:
            best = makespan
    assert best is not None
    return best


def _r14_greedy(release: list[int], duration: list[int], setup: int) -> int:
    """List scheduling in release order: plausible, but not always optimal."""
    available = [setup] * _R14_MACHINES
    used = [False] * _R14_MACHINES
    for job in sorted(range(len(release)), key=lambda index: release[index]):
        starts = [
            max(available[machine] if used[machine] else setup, release[job])
            for machine in range(_R14_MACHINES)
        ]
        machine = min(range(_R14_MACHINES), key=lambda index: (starts[index], index))
        available[machine] = starts[machine] + duration[job]
        used[machine] = True
    return max(available[machine] for machine in range(_R14_MACHINES) if used[machine])


def r14(rng: random.Random) -> core.TaskInstance:
    # Retry until the greedy list schedule is strictly suboptimal, which is
    # what makes the release times bite and the exact search necessary.
    for _ in range(400):
        count = rng.randrange(5, 7)
        setup = rng.randrange(3, 9)
        release = [rng.randrange(0, 45, 5) for _ in range(count)]
        duration = [rng.randrange(5, 26, 5) for _ in range(count)]
        optimal = _r14_makespan(release, duration, setup)
        naive = _r14_greedy(release, duration, setup)
        if naive > optimal:
            break
    else:
        raise AssertionError("could not generate a case where greedy is suboptimal")

    assert _R14_MACHINES == 3
    assert naive > optimal, "the greedy schedule must be strictly worse"
    lines = "\n".join(
        f"Задача {chr(ord('A') + index)}: выпущена в {release[index]} мин, длится {duration[index]} мин"
        for index in range(len(release))
    )
    prompt = (
        f"Есть {_R14_MACHINES} идентичные машины.\n\n"
        f"Перед первой задачей на каждой машине нужна настройка {setup} мин.\n"
        "Если машина не используется, настройка не выполняется.\n\n"
        f"{lines}\n\n"
        "Задачи не прерываются, машина выполняет их по одной.\n"
        "Задача не может начаться раньше момента выпуска.\n\n"
        "Каков минимальный makespan (время завершения всех задач)?\n\n"
        "Ответь только целым числом минут."
    )
    return _int_task(
        "R-14",
        "Расписание с ограничениями",
        prompt,
        optimal,
        {
            "machines": _R14_MACHINES,
            "setup": setup,
            "release": release,
            "duration": duration,
            "optimal_makespan": optimal,
            "naive_greedy": naive,
        },
        units=("минут", "минуты", "минута", "min", "minutes", "minute"),
    )


# --------------------------------------------------------------------------- #
# R-15 — lexicographically smallest topological order
# --------------------------------------------------------------------------- #


def _topo_lex_smallest(nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    indegree = [0] * nodes
    outgoing: list[list[int]] = [[] for _ in range(nodes)]
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    available = [node for node in range(nodes) if indegree[node] == 0]
    heapq.heapify(available)
    order: list[int] = []
    while available:
        node = heapq.heappop(available)
        order.append(node)
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(available, target)
    return order


def _topo_fifo(nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    """Textbook Kahn with a plain queue: a valid order, not the smallest one."""
    indegree = [0] * nodes
    outgoing: list[list[int]] = [[] for _ in range(nodes)]
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    available = deque(node for node in range(nodes) if indegree[node] == 0)
    order: list[int] = []
    while available:
        node = available.popleft()
        order.append(node)
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                available.append(target)
    return order


def _is_topological(nodes: int, edges: list[tuple[int, int]], order: list[int]) -> bool:
    if sorted(order) != list(range(nodes)):
        return False
    position = {node: index for index, node in enumerate(order)}
    return all(position[source] < position[target] for source, target in edges)


def r15(rng: random.Random) -> core.TaskInstance:
    # Retry until the tie-break actually matters: a random layout makes edges
    # point in either letter direction, the alphabetical order is not valid,
    # and the FIFO queue disagrees with the lexicographically smallest order.
    for _ in range(300):
        nodes = rng.randrange(6, 8)
        layout = list(range(nodes))
        rng.shuffle(layout)
        edges = [
            (layout[early], layout[late])
            for early in range(nodes)
            for late in range(early + 1, nodes)
            if rng.random() < 0.25
        ]
        if not edges:
            continue
        if _is_topological(nodes, edges, list(range(nodes))):
            continue  # alphabetical would be valid: no real tie-break to test
        smallest = _topo_lex_smallest(nodes, edges)
        naive_order = _topo_fifo(nodes, edges)
        if naive_order == smallest:
            continue
        assert _is_topological(nodes, edges, smallest)
        assert _is_topological(nodes, edges, naive_order)
        break
    else:
        raise AssertionError("could not generate a DAG with a meaningful tie-break")

    assert not _is_topological(nodes, edges, list(range(nodes))), "alphabetical must be invalid"
    assert naive_order != smallest, "FIFO must not match the lexicographically smallest order"

    answer = "".join(chr(ord("A") + node) for node in smallest)
    naive = "".join(chr(ord("A") + node) for node in naive_order)
    assert naive != answer
    lines = "\n".join(
        f"{chr(ord('A') + source)} → {chr(ord('A') + target)}" for source, target in edges
    )
    universe = ", ".join(chr(ord("A") + node) for node in range(nodes))
    prompt = (
        "Даны зависимости между задачами:\n\n"
        f"{lines}\n\n"
        f"Полный список задач: {universe}. У задачи без зависимостей нет"
        " ограничений; изолированные задачи входят в порядок на общих"
        " основаниях.\n\n"
        "Задача выполняется только после всех своих зависимостей.\n\n"
        "Среди всех допустимых порядков выбери лексикографически\n"
        "наименьший: на каждом шаге бери доступную задачу\n"
        "с наименьшей буквой.\n\n"
        "Верни порядок одним словом из букв без пробелов (например, EBCADFG).\n\n"
        "Ответь только этим словом."
    )
    return _task(
        "R-15",
        "Топологический порядок",
        prompt,
        answer,
        kind="order",
        reference=answer,
        meta={
            "nodes": nodes,
            "edges": [{"from": source, "to": target} for source, target in edges],
            "lex_smallest": answer,
            "naive_fifo": naive,
        },
    )


# --------------------------------------------------------------------------- #
# Generation and evaluation
# --------------------------------------------------------------------------- #


_BUILDERS: dict[str, Callable[[random.Random], core.TaskInstance]] = {
    "R-1": r1,
    "R-2": r2,
    "R-3": r3,
    "R-4": r4,
    "R-5": r5,
    "R-6": r6,
    "R-7": r7,
    "R-8": r8,
    "R-9": r9,
    "R-10": r10,
    "R-11": r11,
    "R-12": r12,
    "R-13": r13,
    "R-14": r14,
    "R-15": r15,
}


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    del cfg
    return [_BUILDERS[scenario](rng) for scenario in IDS]


def _compare(kind: str, token: str | None, answer: Any) -> tuple[bool, str]:
    if token is None:
        return False, "ответ не является единственным значением"
    if kind == "int":
        if not re.fullmatch(r"[+-]?\d+", token):
            return False, f"ожидалось целое число, получено {token!r}"
        passed = int(token) == int(answer)
        return passed, f"получено {token}, ожидалось {answer}"
    if kind == "decimal3":
        # parse_decimal is intentionally permissive for other categories; this
        # single-value task must reject trailing prose.
        if not re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)", token):
            return False, f"ожидалось десятичное число, получено {token!r}"
        try:
            value = Decimal(token.replace(",", "."))
        except Exception:
            return False, f"ожидалось десятичное число, получено {token!r}"
        expected = Decimal(str(answer))
        passed = text.decimals_equal(value, expected, places=3)
        return passed, f"получено {token}, ожидалось {expected:.3f}"
    if kind == "csv4":
        parts = [part.strip() for part in token.split(",")]
        passed = len(parts) == 4 and parts == list(answer)
        return passed, f"получено {parts}, ожидалось {list(answer)}"
    if kind == "fraction":
        value = text.parse_fraction(token)
        if value is None:
            return False, f"ожидалась дробь a/b, получено {token!r}"
        expected = answer if isinstance(answer, Fraction) else Fraction(str(answer))
        passed = value == expected
        return passed, f"получено {token} ({value}), ожидалось {expected}"
    if kind == "hm":
        match = re.fullmatch(r"(\d{1,2}):([0-5]\d)", token)
        if not match:
            return False, f"ожидалось время HH:MM, получено {token!r}"
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23:
            return False, f"ожидалось время HH:MM, получено {token!r}"
        passed = f"{hour:02d}:{minute:02d}" == str(answer)
        return passed, f"получено {token}, ожидалось {answer}"
    if kind == "order":
        # Exact single token: any different (even valid) ordering is rejected.
        passed = token.upper() == str(answer).upper()
        return passed, f"получено {token}, ожидалось {answer}"
    if kind == "lru":
        if token.count(";") != 1:
            return False, f"ожидался формат MISSES;MRU_TO_LRU, получено {token!r}"
        misses_raw, order_raw = token.split(";")
        if not re.fullmatch(r"\d+", misses_raw.strip()):
            return False, f"число промахов не является целым: {misses_raw!r}"
        order = [key.strip() for key in order_raw.split(",")]
        if any(not key for key in order):
            return False, f"пустой ключ в порядке: {order_raw!r}"
        expected_misses, expected_order = str(answer).split(";", 1)
        expected_keys = expected_order.split(",")
        passed = int(misses_raw.strip()) == int(expected_misses) and order == expected_keys
        return passed, f"получено {token}, ожидалось {answer}"
    raise AssertionError(f"unknown comparison kind {kind!r}")


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="B")
    oracle = task.oracle
    token = text.answer_token(tr.final_text, tuple(oracle["unit_whitelist"]))
    passed, detail = _compare(oracle["kind"], token, oracle["answer"])
    score.add("value", passed, _POINTS, detail)
    if token is None and tr.final_text.strip():
        score.notes.append("ответ содержит более одной строки или лишний текст")
    return score


core.register(
    key="B",
    title="Reasoning",
    points=15.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
)
