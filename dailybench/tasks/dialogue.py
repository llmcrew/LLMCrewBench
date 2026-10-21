"""Category H — multi-turn dialogue, 3 scenarios × 2 checks × 1 point.

The runner delivers ``TaskInstance.dialogue`` after every assistant reply that
carries no tool calls, so the model sees the whole conversation and the *last*
answer has to satisfy everything that was said before it. Each scenario scores
exactly two separable things:

* H-1 — *correction mid-dialogue*: turn 1 fixes a task and its format; turn 2
  changes one requirement. One point for the corrected choice, one for the
  format of turn 1 still holding (and the rejected value not resurfacing).
* H-2 — *constraint retention under pressure*: turn 1 imposes a hard rule
  (exactly one word of a fixed set, no digits, no explanation); later turns
  pile on data and finally ask for prose. One point for the right final status,
  one for the rule surviving the pressure.
* H-3 — *contradiction resolution*: turn 2 explicitly overrides turn 1. One
  point for the recomputed value, one for the superseded values never appearing
  in the final answer.

* H-4 (pilot) — *accumulated constraints*: the format and a selection rule
  stand from turn 1, a slot is excluded on turn 2 and exactly one rule is
  changed on turn 3; the closing "explain" request must not dissolve the
  format. The evaluator walks the transcript turn by turn, so a right final
  answer after wrong intermediate ones is partial credit only.
* H-5 (stretch) — *parameter replacement and exception rollback*: four state
  fields, an explicit replacement, a separate revocation of a temporary
  exclusion and a quoted README that is not a correction. Every turn is
  checked against an expected state and the final reply has to be strict JSON.

Everything is checkable without a judge model: a labelled token, a member of an
allow-set, an integer, and a fixed set of forbidden substrings.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from .. import core, text

BASELINE_IDS = ("H-1", "H-2", "H-3")
PILOT_IDS = ("H-4",)
ALL_IDS = BASELINE_IDS + PILOT_IDS + ("H-5",)

IDS = BASELINE_IDS
_CHECK = 1.0
_POINTS = 2.0
#: H-4/H-5 issue four unit checks each; the budget is the issue itself.
_WIDE_POINTS = 4.0

#: H-2 — status vocabulary the whole conversation must stay inside.
STATUS_SET = ("ok", "degraded", "down")

_TARIFFS = ("alpha", "beta", "delta", "kappa", "omega", "sigma", "tau", "zeta")
_SERVICES = ("payments", "billing", "auth", "search", "notify", "ledger")

#: H-3 — limit pairs and percentages chosen so every threshold is an integer.
_LIMITS_OLD = (120, 140, 160, 180, 200)
_LIMITS_NEW = (260, 300, 340, 400, 440)
_PERCENTS = (75, 80, 90)

#: H-4 — slot names that never nest inside each other, so a text search for a
#: forbidden slot cannot hit a different one.
_HOSTS = (
    "node-alfa",
    "node-bravo",
    "node-charlie",
    "node-delta",
    "node-echo",
    "node-foxtrot",
    "node-golf",
    "node-hotel",
    "node-india",
    "node-juliet",
    "node-kilo",
    "node-lima",
    "node-mike",
    "node-november",
    "node-oscar",
    "node-papa",
)

_SLOT_RE = re.compile(
    r"итог\s*:\s*(?P<host>[a-z][a-z0-9-]*)\s*\|\s*ram=(?P<ram>\d+)\s*\|"
    r"\s*uptime=(?P<uptime>\d+)\s*\|\s*причина=(?P<reason>[^\s|]+)",
    re.IGNORECASE,
)

#: H-5 — four state fields, one replacement, one temporary exclusion.
_STATE_FIELDS = ("region", "min_cpu", "limit", "exclude_tag")
_REGIONS = ("eu-west", "us-east", "ap-south")
_MIN_CPU = (6, 8, 12, 16)
_LIMIT_BASE = (20, 24, 28)
_TAGS = ("legacy", "canary", "beta")
#: The quoted README value: never a valid active ``min_cpu``.
_QUOTED_MIN_CPU = 4


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _transcript(task: core.TaskInstance, final: str) -> core.Transcript:
    """A one-turn transcript carrying ``final`` — for fixture-time assertions."""
    return core.Transcript(
        task_id=task.id,
        category="H",
        turns=[
            core.Turn(
                messages=[{"role": "user", "content": task.prompt}],
                reply=core.ModelReply(text=final),
            )
        ],
    )


def _assert_naive_trap(task: core.TaskInstance, naive: str, reference: str) -> None:
    """The pre-correction answer must differ from the reference and score zero.

    This is the guarantee the category rests on: a model that reads turn 2 but
    keeps answering turn 1 cannot collect a single point, so the score measures
    the dialogue and not the topic.
    """
    assert naive != reference, f"{task.id}: наивный ответ совпал с эталоном"
    score = evaluate(task, _transcript(task, naive))
    assert score.possible == task.max_points, f"{task.id}: возможный балл {score.possible}"
    assert score.earned == 0.0, (
        f"{task.id}: наивный ответ набрал {score.earned} из {score.possible} "
        f"(провалены {[c.name for c in score.checks if not c.passed]})"
    )


# --------------------------------------------------------------------------- #
# H-1 — the correction applies, the untouched format survives
# --------------------------------------------------------------------------- #


def _h1(rng: random.Random) -> core.TaskInstance:
    names = rng.sample(_TARIFFS, 5)
    prices = rng.sample(range(120, 9900, 30), 5)
    pairs = list(zip(names, prices))
    cheap_name = min(pairs, key=lambda p: p[1])[0]
    dear_name = max(pairs, key=lambda p: p[1])[0]
    assert cheap_name != dear_name, "минимальная и максимальная цены должны различаться"
    # The forbidden-content check searches the final text for the rejected
    # tariff: a name nested inside the other would make the check meaningless.
    assert cheap_name not in dear_name and dear_name not in cheap_name

    rows = [f"{name} — {price}₽" for name, price in pairs]
    rng.shuffle(rows)
    reference = f"итог: {dear_name}"
    naive = f"итог: {cheap_name}"
    return core.TaskInstance(
        id="H-1",
        category="H",
        title="Правка посреди диалога",
        prompt=(
            "Список тарифов:\n\n"
            + "\n".join(rows)
            + "\n\nВыбери тариф с МИНИМАЛЬНОЙ ценой.\n"
            "Ответ — ровно одна строка вида `итог: <имя>`: имя строчными буквами, "
            "без цены и без пояснений."
        ),
        max_points=_POINTS,
        dialogue=[
            "Меняю одно условие: нужен тариф с МАКСИМАЛЬНОЙ ценой. "
            "Формат прежний — ровно одна строка `итог: <имя>`."
        ],
        oracle={
            "reference": reference,
            "turn_answers": [naive, reference],
            "cheap": cheap_name,
            "dear": dear_name,
            "naive": naive,
        },
        meta={"scenario": "H-1", "turns": 2, "kind": "mid_dialogue_correction"},
    )


def _eval_h1(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="H")
    dear = task.oracle["dear"]
    cheap = task.oracle["cheap"]
    final = tr.final_text

    # Find the labelled value anywhere, so a correct choice wrapped in extra
    # prose loses only the format point and not both.
    match = re.search(r"итог\s*:\s*([^\s,.;:]+)", final, re.IGNORECASE)
    chosen = match.group(1).casefold() if match else None
    score.add(
        "corrected_tariff",
        chosen == dear,
        _CHECK,
        f"выбран {chosen!r}, ожидался {dear!r} — условие изменено во втором сообщении",
    )

    lines = text.answer_lines(final)
    shape_ok = len(lines) == 1 and bool(re.fullmatch(r"итог\s*:\s*\S+", lines[0], re.IGNORECASE))
    stale = cheap in final.casefold()
    score.add(
        "format_kept",
        shape_ok and not stale,
        _CHECK,
        "одна строка `итог: <имя>` из первого сообщения и отвергнутый тариф не упомянут",
    )
    return score


# --------------------------------------------------------------------------- #
# H-2 — the hard rule survives the growing pressure
# --------------------------------------------------------------------------- #


def _status(error_pct: int, latency_ms: int, norm_ms: int) -> str:
    """The status rule the prompt states, applied to the fixture numbers."""
    if error_pct >= 25 or latency_ms >= 3 * norm_ms:
        return "down"
    if error_pct >= 5 or latency_ms >= 2 * norm_ms:
        return "degraded"
    return "ok"


def _h2(rng: random.Random) -> core.TaskInstance:
    service = rng.choice(_SERVICES)
    norm_ms = rng.choice((50, 80, 120, 200))
    error_pct = rng.choice((8, 12, 18))
    peak_ms = 5 * norm_ms

    first = _status(0, norm_ms, norm_ms)
    second = _status(error_pct, norm_ms, norm_ms)
    third = _status(error_pct, peak_ms, norm_ms)
    assert (first, second, third) == ("ok", "degraded", "down"), "траектория статуса искажена"

    reference = third
    naive = (
        "Статус: down.\n"
        f"Причина: error_pct = {error_pct}% и latency = {peak_ms} мс против нормы {norm_ms} мс."
    )
    assert first in STATUS_SET and third in STATUS_SET, "статусы обязаны быть словами набора"
    return core.TaskInstance(
        id="H-2",
        category="H",
        title="Правило держится под давлением",
        prompt=(
            "Правило на весь диалог: на любой мой вопрос отвечай РОВНО ОДНИМ словом "
            "из набора: ok, degraded, down. Никаких цифр, пояснений и дополнительных строк.\n"
            "Как считается статус: down — если error_pct >= 25 или latency >= 3 норм; "
            "degraded — если error_pct >= 5 или latency >= 2 норм; иначе ok.\n\n"
            f"Сервис {service}: error_pct = 0, latency = {norm_ms} мс, норма = {norm_ms} мс.\n"
            "Какой статус?"
        ),
        max_points=_POINTS,
        dialogue=[
            f"Новые данные по {service}: error_pct = {error_pct}, "
            f"latency = {norm_ms} мс, норма = {norm_ms} мс. Какой статус?",
            f"И ещё: latency выросла до {peak_ms} мс при прежней норме {norm_ms} мс. Какой статус?",
            "И всё-таки объясни, пожалуйста, почему именно такой статус.",
        ],
        oracle={
            "reference": reference,
            "turn_answers": [first, second, third, reference],
            "allowed": list(STATUS_SET),
            "final_status": third,
            "naive": naive,
        },
        meta={"scenario": "H-2", "turns": 4, "kind": "constraint_retention"},
    )


def _eval_h2(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="H")
    allowed = tuple(task.oracle["allowed"])
    expected_turns = task.oracle["turn_answers"]
    final = tr.final_text

    # The last follow-up is a free-form "why": only the status turns are
    # answerable with one word. A constant answer ("down" to everything) used
    # to pass both checks on the final text alone — the whole sequence is
    # the actual task.
    status_replies = [
        (text.answer_token(turn.reply.text or "") or "").casefold()
        for turn in tr.turns[: len(expected_turns) - 1]
    ]
    expected_status = expected_turns[: len(status_replies)]
    sequence_ok = status_replies == expected_status and len(status_replies) == len(expected_turns) - 1
    score.add(
        "status_sequence",
        sequence_ok,
        _CHECK,
        f"последовательность статусов {'совпала' if sequence_ok else 'не совпала'}: "
        f"получено {status_replies or '—'}, ожидалось {expected_turns[:-1]} (последний ход — свободное объяснение)",
    )

    final_token = text.answer_token(final)
    given = final_token.casefold() if final_token else None
    no_digits = not any(ch.isdigit() for ch in final)
    score.add(
        "final_status",
        given == task.oracle["final_status"] and no_digits and given in allowed,
        _CHECK,
        f"получено {given!r}, ожидался {task.oracle['final_status']!r}; ровно одно слово из набора и ни одной цифры",
    )
    return score


# --------------------------------------------------------------------------- #
# H-3 — the new value wins, the old one is gone
# --------------------------------------------------------------------------- #


def _h3(rng: random.Random) -> core.TaskInstance:
    # Build the valid combinations first so the choice never lands on a pair
    # whose numbers nest inside each other and would defeat the substring check.
    combos: list[tuple[int, int, int, int, int]] = []
    for old in _LIMITS_OLD:
        for new in _LIMITS_NEW:
            for percent in _PERCENTS:
                first = old * percent // 100
                second = new * percent // 100
                if (old * percent) % 100 or (new * percent) % 100:
                    continue
                if first == second:
                    continue
                strings = (str(old), str(first), str(new), str(second))
                if any(a != b and a in b for a in strings for b in strings):
                    continue
                combos.append((old, new, percent, first, second))
    assert combos, "не нашлось непересекающихся пары лимитов"
    old, new, percent, first, second = rng.choice(combos)

    naive = str(first)
    reference = str(second)
    return core.TaskInstance(
        id="H-3",
        category="H",
        title="Разрешение противоречия",
        prompt=(
            f"Лимит сервиса — {old} запросов в секунду. Порог тревоги — {percent}% от лимита.\n"
            "Вычисли порог и верни ТОЛЬКО число (запросов в секунду), без пояснений."
        ),
        max_points=_POINTS,
        dialogue=[
            f"Поправка: лимит изменился — теперь {new} запросов в секунду, не {old}. "
            f"Процент порога прежний — {percent}%. Пересчитай.",
            "Итог: назови только окончательное значение — одно число, "
            "без пояснений и без старых цифр.",
        ],
        oracle={
            "reference": reference,
            "turn_answers": [naive, reference, reference],
            "limit1": old,
            "threshold1": first,
            "limit2": new,
            "threshold2": second,
            "naive": naive,
        },
        meta={"scenario": "H-3", "turns": 3, "kind": "contradiction_resolution"},
    )


def _eval_h3(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="H")
    expected = task.oracle["threshold2"]
    stale = (str(task.oracle["limit1"]), str(task.oracle["threshold1"]))
    final = tr.final_text

    got = text.parse_integer(final)
    score.add("new_threshold", got == expected, _CHECK, f"получено {got!r}, ожидалось {expected}")

    token = text.answer_token(final)
    single_number = token is not None and bool(re.fullmatch(r"\d+", token))
    score.add(
        "superseded_values_gone",
        single_number and not any(s in final for s in stale),
        _CHECK,
        f"в финале одно число и нет ни старого лимита {stale[0]}, ни старого порога {stale[1]}",
    )
    return score


# --------------------------------------------------------------------------- #
# H-4 — the format and the exclusion survive the rule change
# --------------------------------------------------------------------------- #


def _slot_line(values: dict[str, tuple[int, int]], host: str, reason: str) -> str:
    ram, uptime = values[host]
    return f"итог: {host} | ram={ram} | uptime={uptime} | причина={reason}"


def _parse_slot(reply: str) -> dict[str, str] | None:
    """The single formatted line of a reply, or ``None`` if the shape is off."""
    lines = text.answer_lines(reply)
    if len(lines) != 1:
        return None
    match = _SLOT_RE.fullmatch(lines[0])
    return match.groupdict() if match else None


def _chosen_host(reply: str) -> str:
    """The slot named in the reply, independent of the rest of the format."""
    match = re.search(r"итог\s*:\s*([a-z][a-z0-9-]*)", reply or "", re.IGNORECASE)
    return match.group(1).casefold() if match else ""


def _h4(rng: random.Random) -> core.TaskInstance:
    hosts = rng.sample(_HOSTS, 8)
    edge, core_hosts = hosts[:4], hosts[4:]
    # Two core slots outrank every edge slot by RAM, so dropping the zone rule
    # changes the choice; the excluded slot leads both rankings, so dropping
    # the exclusion changes it as well once the rule is switched to uptime.
    rams = sorted(rng.sample(range(24, 120, 8), 8), reverse=True)
    uptimes = sorted(rng.sample(range(16, 216, 8), 8), reverse=True)
    values: dict[str, tuple[int, int]] = {
        core_hosts[0]: (rams[0], uptimes[4]),
        core_hosts[1]: (rams[1], uptimes[5]),
        edge[0]: (rams[2], uptimes[0]),  # excluded on turn 2, leads both metrics
        edge[1]: (rams[3], uptimes[2]),
        edge[2]: (rams[4], uptimes[1]),  # uptime leader once turn 3 applies
        edge[3]: (rams[5], uptimes[3]),
        core_hosts[2]: (rams[6], uptimes[6]),
        core_hosts[3]: (rams[7], uptimes[7]),
    }
    assert max(values[host][0] for host in hosts) == values[core_hosts[0]][0] > values[edge[0]][0]
    assert max(values[host][1] for host in hosts) == values[edge[0]][1]
    assert min(rams) < values[edge[0]][0] and min(uptimes) < values[edge[3]][1]

    rows = [
        f"{host} — зона={'edge' if host in edge else 'core'}, "
        f"ram={values[host][0]}, uptime={values[host][1]}"
        for host in hosts
    ]
    rng.shuffle(rows)

    excluded = edge[0]
    answers = [
        _slot_line(values, edge[0], "максимум"),
        _slot_line(values, edge[1], "резерв"),
        _slot_line(values, edge[2], "старейший"),
        _slot_line(values, edge[2], "старейший"),
    ]
    return core.TaskInstance(
        id="H-4",
        category="H",
        title="Накопленные ограничения и одна поправка",
        prompt=(
            "Слоты кластера (зона, свободная RAM, uptime в часах):\n\n"
            + "\n".join(rows)
            + "\n\nВыбери слот зоны edge с МАКСИМАЛЬНОЙ свободной RAM.\n"
            "Ответ — ровно одна строка и ровно такой формы:\n"
            "`итог: <слот> | ram=<ram> | uptime=<uptime> | причина=<одно слово>`\n"
            "Без пояснений и лишних строк."
        ),
        max_points=_WIDE_POINTS,
        dialogue=[
            f"Из списка ещё исключи слот {excluded} — он в резерве. "
            "Остальные правила и формат прежние.",
            "Меняю одно правило: нужен самый большой uptime среди оставшихся, "
            "а не максимальная RAM. Зона и исключение прежние, формат прежний.",
            "И объясни причину выбора — одним словом в поле «причина». "
            "Ответ по-прежнему ровно одна строка в прежнем формате.",
        ],
        oracle={
            "reference": answers[-1],
            "turn_answers": answers,
            "turn_hosts": [edge[0], edge[1], edge[2], edge[2]],
            "turn_values": [values[host] for host in (edge[0], edge[1], edge[2], edge[2])],
            "excluded": excluded,
            "zone": "edge",
            "naive": "не знаю",
        },
        meta={
            "scenario": "H-4",
            "turns": 4,
            "kind": "accumulated_constraints",
            "constraints": ["format", "zone=edge", f"exclude {excluded}"],
        },
    )


def _eval_h4(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="H")
    expected_hosts = task.oracle["turn_hosts"]
    expected_values = task.oracle["turn_values"]
    replies = [turn.reply.text or "" for turn in tr.turns]
    parsed = [_parse_slot(reply) for reply in replies]

    # Every turn is answered, not only the last one: a model that gets the
    # first choices wrong and only the final one right loses this check.
    hosts = [_chosen_host(reply) for reply in replies]
    answers_ok = len(replies) == len(expected_hosts) and hosts == [
        host.casefold() for host in expected_hosts
    ]
    score.add(
        "per_turn_answers",
        answers_ok,
        _CHECK,
        f"выбор по ходам {hosts or '—'}, ожидался {list(expected_hosts)}; "
        "формат и исключение тоже действуют на каждом ходе",
    )

    format_ok = len(parsed) == len(expected_hosts) and all(parsed)
    score.add(
        "format_all_turns",
        format_ok,
        _CHECK,
        "каждый ответ — ровно одна строка `итог: … | ram=… | uptime=… | причина=…`; "
        "просьба объяснить не добавляет строк",
    )

    excluded = task.oracle["excluded"].casefold()
    after = replies[1:]
    after_slots = parsed[1:]
    # An unparseable reply names no slot at all, so it cannot prove that the
    # exclusion is still respected; every post-amendment turn must parse.
    exclusion_ok = (
        len(after_slots) == len(expected_hosts) - 1
        and all(slots for slots in after_slots)
        and all((slots or {}).get("host", "").casefold() != excluded for slots in after_slots)
        and all(excluded not in reply.casefold() for reply in after)
    )
    score.add(
        "exclusion_after_amendment",
        exclusion_ok,
        _CHECK,
        f"исключённый слот {task.oracle['excluded']!r} не появляется после второго хода, "
        "включая финальное объяснение",
    )

    final = parsed[-1] if parsed else None
    final_host = expected_hosts[-1]
    ram, uptime = expected_values[-1]
    final_ok = bool(
        final
        and len(parsed) == len(expected_hosts)
        and final["host"].casefold() == final_host.casefold()
        and int(final["ram"]) == ram
        and int(final["uptime"]) == uptime
        and final["reason"].strip()
    )
    score.add(
        "final_form_and_constraints",
        final_ok,
        _CHECK,
        f"финал: слот {final_host!r} с ram={ram} и uptime={uptime} в той же строке, "
        "причина заполнена; все действующие ограничения сохранены",
    )
    return score


# --------------------------------------------------------------------------- #
# H-5 — replacement, revocation, and a quote that is not a correction
# --------------------------------------------------------------------------- #


def _state(region: str, min_cpu: int, limit: int, tag: str | None) -> dict[str, Any]:
    return {"region": region, "min_cpu": min_cpu, "limit": limit, "exclude_tag": tag}


def _state_json(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False)


def _parse_state(reply: str) -> dict[str, Any] | None:
    value, _ = text.strict_json(reply)
    return value if isinstance(value, dict) else None


def _h5(rng: random.Random) -> core.TaskInstance:
    region = rng.choice(_REGIONS)
    min_cpu = rng.choice(_MIN_CPU)
    base_limit = rng.choice(_LIMIT_BASE)
    new_limit = base_limit + 10
    tag = rng.choice(_TAGS)
    assert _QUOTED_MIN_CPU not in _MIN_CPU
    assert new_limit != base_limit and tag not in region and tag not in str(base_limit)

    start = _state(region, min_cpu, base_limit, None)
    with_exception = _state(region, min_cpu, base_limit, tag)
    replaced = _state(region, min_cpu, new_limit, tag)
    revoked = _state(region, min_cpu, new_limit, None)
    states = [start, with_exception, replaced, revoked, revoked]
    answers = [_state_json(state) for state in states]
    return core.TaskInstance(
        id="H-5",
        category="H",
        title="Замена параметра и отзыв исключения",
        prompt=(
            "Конфигурация задачи:\n\n"
            + _state_json(start)
            + "\n\nМеняй её только по явной команде. Значение, процитированное из "
            "документации или приведённое как чужое мнение, действующую "
            "конфигурацию не меняет. Ключи всегда одни и те же: "
            "region, min_cpu, limit, exclude_tag (null — исключений нет).\n\n"
            "Пришли текущую конфигурацию: только JSON, ровно эти четыре поля."
        ),
        max_points=_WIDE_POINTS,
        dialogue=[
            f"Включаю временное исключение: exclude_tag = {tag!r}. "
            "Остальные поля не трогай. Верни обновлённый JSON.",
            f"Замена параметра: limit = {new_limit} вместо {base_limit}. "
            "Остальное прежнее. Верни обновлённый JSON.",
            f"Отзываю исключение {tag!r}: снова без исключений (null). "
            "Остальное прежнее. Верни обновлённый JSON.",
            f"Коллега настаивает: «в README стоит min_cpu: {_QUOTED_MIN_CPU}, "
            f"limit: {base_limit}» — но это цитата, а не поправка. "
            "Пришли актуальную конфигурацию: только JSON, ровно четыре поля.",
        ],
        oracle={
            "reference": answers[-1],
            "turn_answers": answers,
            "states": states,
            "base": start,
            "tag": tag,
            "old_limit": base_limit,
            "new_limit": new_limit,
            "quoted_min_cpu": _QUOTED_MIN_CPU,
            "naive": "не знаю",
        },
        meta={
            "scenario": "H-5",
            "turns": 5,
            "kind": "replacement_and_rollback",
            "fields": list(_STATE_FIELDS),
        },
    )


def _eval_h5(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="H")
    expected = task.oracle["states"]
    replies = [turn.reply.text or "" for turn in tr.turns]
    parsed = [_parse_state(reply) for reply in replies]
    base = task.oracle["base"]

    def _exact(value: dict[str, Any] | None, want: dict[str, Any]) -> bool:
        return value is not None and set(value) == set(_STATE_FIELDS) and value == want

    per_turn = len(parsed) == len(expected) and all(
        _exact(value, want) for value, want in zip(parsed, expected)
    )
    score.add(
        "state_per_turn",
        per_turn,
        _CHECK,
        f"состояние по ходам {parsed or '—'}, ожидалось {expected}; "
        "у каждого ответа ровно четыре поля",
    )

    tag = task.oracle["tag"]
    tag_at = [value.get("exclude_tag") if value else "—" for value in parsed]
    exception_ok = (
        len(parsed) == len(expected)
        and tag_at[1:3] == [tag, tag]
        and tag_at[3:] == [None, None]
    )
    score.add(
        "exception_lifecycle",
        exception_ok,
        _CHECK,
        f"exclude_tag по ходам {tag_at or '—'}: {tag!r} включён на 2–3-м и снят с 4-го хода",
    )

    independent_ok = len(parsed) == len(expected) and all(
        value is not None
        and value.get("region") == base["region"]
        and value.get("min_cpu") == base["min_cpu"]
        for value in parsed
    )
    score.add(
        "independent_fields_intact",
        independent_ok,
        _CHECK,
        f"region={base['region']!r} и min_cpu={base['min_cpu']} не меняются ни на одном ходе",
    )

    final = parsed[-1] if parsed else None
    final_text = replies[-1] if replies else ""
    _, strict_ok = text.strict_json(final_text)
    quote_ignored = _exact(final, expected[-1])
    score.add(
        "final_json_strict",
        bool(strict_ok and quote_ignored and len(parsed) == len(expected)),
        _CHECK,
        f"финал — только строгий JSON с четырьмя полями и состоянием {expected[-1]}; "
        f"цитата min_cpu={task.oracle['quoted_min_cpu']} и limit={task.oracle['old_limit']} не применена",
    )
    return score


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

_BUILDERS: dict[str, Any] = {
    "H-1": _h1,
    "H-2": _h2,
    "H-3": _h3,
    "H-4": _h4,
    "H-5": _h5,
}
_EVALUATORS: dict[str, Any] = {
    "H-1": _eval_h1,
    "H-2": _eval_h2,
    "H-3": _eval_h3,
    "H-4": _eval_h4,
    "H-5": _eval_h5,
}


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    """Baseline scenarios, the pilot scenario, or the baseline plus stretch."""
    profile = core.profile_of(cfg)
    if profile == "pilot":
        ids = PILOT_IDS
    elif profile == "stretch-v1":
        ids = ALL_IDS
    else:
        ids = BASELINE_IDS
    tasks = [_BUILDERS[task_id](rng) for task_id in ids]
    for task in tasks:
        _assert_naive_trap(task, task.oracle["naive"], task.oracle["reference"])
    return tasks


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    return _EVALUATORS[task.id](task, tr)


core.register(
    key="H",
    title="Multi-turn dialogue",
    points=6.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=BASELINE_IDS,
    expected_ids_by_profile={"pilot": PILOT_IDS, "stretch-v1": ALL_IDS},
)
