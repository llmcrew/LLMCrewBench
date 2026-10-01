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

Everything is checkable without a judge model: a labelled token, a member of an
allow-set, an integer, and a fixed set of forbidden substrings.
"""

from __future__ import annotations

import random
import re
from typing import Any

from .. import core, text

IDS = ("H-1", "H-2", "H-3")
_CHECK = 1.0
_POINTS = 2.0

#: H-2 — status vocabulary the whole conversation must stay inside.
STATUS_SET = ("ok", "degraded", "down")

_TARIFFS = ("alpha", "beta", "delta", "kappa", "omega", "sigma", "tau", "zeta")
_SERVICES = ("payments", "billing", "auth", "search", "notify", "ledger")

#: H-3 — limit pairs and percentages chosen so every threshold is an integer.
_LIMITS_OLD = (120, 140, 160, 180, 200)
_LIMITS_NEW = (260, 300, 340, 400, 440)
_PERCENTS = (75, 80, 90)


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
    expected = task.oracle["final_status"]
    final = tr.final_text

    token = text.answer_token(final)
    given = token.casefold() if token else None
    score.add(
        "final_status",
        given == expected,
        _CHECK,
        f"получено {given!r}, ожидался {expected!r}",
    )

    no_digits = not any(ch.isdigit() for ch in final)
    held = given in allowed and no_digits
    score.add(
        "single_word_rule",
        held,
        _CHECK,
        "ровно одно слово из набора и ни одной цифры — правило первого сообщения",
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
# Registration
# --------------------------------------------------------------------------- #

_BUILDERS: dict[str, Any] = {"H-1": _h1, "H-2": _h2, "H-3": _h3}
_EVALUATORS: dict[str, Any] = {"H-1": _eval_h1, "H-2": _eval_h2, "H-3": _eval_h3}


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    tasks = [_BUILDERS[task_id](rng) for task_id in IDS]
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
    expected_ids=IDS,
)
