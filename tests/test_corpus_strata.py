"""Corpus strata: ru/kk twins, the RTL/NFC edge case and the private gate.

The wave adds culturally adapted tasks rather than translations: the Kazakh
sibling of every Russian brief has its own fixture, prompt and oracle, and the
evaluator never compares the two languages with each other. The private layer
is asserted per category, because the stretch gate is a per-category share, not
an overall average.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import date, timedelta
from decimal import Decimal

import pytest

from dailybench import core
from dailybench.config import RunConfig
from dailybench.decontam import private_gate
from dailybench.rng import task_rng
from dailybench.suite import build_suite, public_prompts
from dailybench.tasks import instruction, reasoning

DEV_SEEDS = (11, 12, 13, 14, 15)
NEW_A = ("IF-7", "IF-7-KK", "IF-7-EN", "IF-8", "IF-9", "IF-10")
NEW_B = ("R-17", "R-17-KK", "R-18")


def _transcript(task: core.TaskInstance, answer: str) -> core.Transcript:
    return core.Transcript(
        task_id=task.id,
        category=task.category,
        turns=[core.Turn(messages=[{"role": "user", "content": task.prompt}], reply=core.ModelReply(text=answer))],
    )


def _checks(score: core.Score) -> dict[str, bool]:
    return {check.name: check.passed for check in score.checks}


def _if_tasks(seed: int) -> list[core.TaskInstance]:
    return instruction.generate(task_rng(seed, "A"), RunConfig(profile="stretch-v1"))


def _r_tasks(seed: int) -> list[core.TaskInstance]:
    return reasoning.generate(task_rng(seed, "B"), RunConfig(profile="stretch-v1"))


def _by_id(tasks: list[core.TaskInstance], task_id: str) -> core.TaskInstance:
    return next(task for task in tasks if task.id == task_id)


def _if7(seed: int, lang: str) -> core.TaskInstance:
    return _by_id(_if_tasks(seed), {"ru": "IF-7", "kk": "IF-7-KK", "en": "IF-7-EN"}[lang])


def _r17(seed: int, lang: str) -> core.TaskInstance:
    return _by_id(_r_tasks(seed), "R-17" if lang == "ru" else "R-17-KK")


def _answer(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Private gate
# --------------------------------------------------------------------------- #


def test_stretch_private_gate_passes_in_every_category():
    tasks = build_suite(RunConfig(profile="stretch-v1"))
    report = private_gate(tasks, "stretch-v1")
    assert report.ok is True
    assert report.failed == []
    assert [row.category for row in report.rows] == list("ABCDEFGHIJ")
    for row in report.rows:
        assert row.total > 0
        assert row.share >= 0.2, (row.category, row.private, row.total)


def test_the_new_tasks_are_private_and_fresh():
    tasks = build_suite(RunConfig(profile="stretch-v1"))
    for task_id in NEW_A + NEW_B:
        task = next(task for task in tasks if task.id == task_id)
        assert task.private is True
        assert task.fresh_after == instruction.FRESH_AFTER
    hidden = {entry["id"]: entry for entry in public_prompts(tasks)}
    for task_id in NEW_A + NEW_B:
        assert "prompt" not in hidden[task_id]
        assert len(hidden[task_id]["prompt_sha256"]) == 64


def test_old_private_marks_are_exactly_the_declared_ones():
    a_private = [task.id for task in _if_tasks(1) if task.private]
    b_private = [task.id for task in _r_tasks(1) if task.private]
    assert a_private == list(NEW_A)
    assert b_private == ["R-3", "R-9", *NEW_B]
    assert all(task.id not in instruction.PRIVATE_IDS for task in _if_tasks(1) if task.category != "A")


# --------------------------------------------------------------------------- #
# Language and culture stratum
# --------------------------------------------------------------------------- #


def test_if7_twins_speak_their_own_language_and_calendar():
    for seed in DEV_SEEDS:
        ru, kk, en = _if7(seed, "ru"), _if7(seed, "kk"), _if7(seed, "en")
        assert ru.languages == ["ru"] and ru.cultures == ["ru"]
        assert kk.languages == ["kk"] and kk.cultures == ["kz"]
        assert en.languages == ["en"] and en.cultures == ["en-GB"]
        assert ru.meta["languages"] == ["ru"] and ru.meta["cultures"] == ["ru"]
        assert kk.meta["languages"] == ["kk"] and kk.meta["cultures"] == ["kz"]
        assert en.meta["languages"] == ["en"] and en.meta["cultures"] == ["en-GB"]
        assert "ГГГГ-ММ-ДД" in ru.prompt and any(month in ru.prompt for month in instruction.RU_MONTHS)
        assert "ЖЖЖЖ-АА-КК" in kk.prompt and any(month in kk.prompt for month in instruction.KK_MONTHS)
        assert "YYYY-MM-DD" in en.prompt and any(month in en.prompt for month in instruction.EN_MONTHS)
        assert "1 200,0 кг" in ru.prompt and "1 200,0 кг" in kk.prompt and "1,200.0 kg" in en.prompt
        assert len({ru.prompt, kk.prompt, en.prompt}) == 3
        assert len({ru.oracle["reference"], kk.oracle["reference"], en.oracle["reference"]}) == 3


def test_r17_twins_name_kazakh_holidays_in_their_own_language():
    for seed in DEV_SEEDS:
        ru, kk = _r17(seed, "ru"), _r17(seed, "kk")
        assert ru.languages == ["ru"] and ru.cultures == ["kz"]
        assert kk.languages == ["kk"] and kk.cultures == ["kz"]
        assert ru.meta["languages"] == ["ru"] and kk.meta["languages"] == ["kk"]
        assert "праздники Казахстана" in ru.prompt and "Наурыз" in ru.prompt
        assert "Қазақстанның мемлекеттік мерекелері" in kk.prompt and "Наурыз" in kk.prompt
        assert ru.prompt != kk.prompt
        assert ru.oracle["reference"] != kk.oracle["reference"]


# --------------------------------------------------------------------------- #
# Unicode edge case
# --------------------------------------------------------------------------- #


def test_if8_prompt_mixes_scripts_and_writes_decomposed_keys():
    for seed in DEV_SEEDS:
        task = _by_id(_if_tasks(seed), "IF-8")
        assert task.languages == ["ru"]
        assert task.meta["cultures"] == ["il", "kz"]
        assert any("\u0590" <= char <= "\u05ff" for char in task.prompt)  # Hebrew
        assert any("\u0600" <= char <= "\u06ff" for char in task.prompt)  # Arabic
        assert task.meta["decomposed_rows"] >= 2
        assert unicodedata.normalize("NFC", task.prompt) != task.prompt


def test_if8_accepts_a_fully_decomposed_answer():
    for seed in DEV_SEEDS:
        task = _by_id(_if_tasks(seed), "IF-8")
        payload = json.loads(task.oracle["reference"])
        decomposed = {
            "keys": [
                {"key": unicodedata.normalize("NFD", item["key"]), "value": item["value"]}
                for item in payload["keys"]
            ]
        }
        score = instruction.evaluate(task, _transcript(task, _answer(decomposed)))
        assert score.earned == 2.0
        assert _checks(score) == {"json_format": True, "keys": True, "order": True}


def test_if8_oracle_is_nfc_and_sorted_by_code_point():
    task = _by_id(_if_tasks(1), "IF-8")
    keys = [item["key"] for item in task.oracle["keys"]]
    assert all(unicodedata.normalize("NFC", key) == key for key in keys)
    assert keys == sorted(keys)
    assert len(keys) == len({unicodedata.normalize("NFC", key) for key in keys}) == task.meta["distinct_keys"]


# --------------------------------------------------------------------------- #
# Determinism and holdout behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task_id", NEW_A + NEW_B)
def test_each_new_task_is_seed_deterministic(task_id: str):
    def build(seed: int) -> core.TaskInstance:
        tasks = _if_tasks(seed) if task_id in NEW_A else _r_tasks(seed)
        return _by_id(tasks, task_id)

    for seed in DEV_SEEDS:
        first, second = build(seed), build(seed)
        assert (first.prompt, first.oracle, first.meta) == (second.prompt, second.oracle, second.meta)

    variants = {
        (build(seed).prompt, build(seed).oracle["reference"])
        for seed in DEV_SEEDS
    }
    assert len(variants) == len(DEV_SEEDS)


@pytest.mark.parametrize("task_id", NEW_A + NEW_B)
def test_every_new_task_scores_its_reference_and_rejects_garbage(task_id: str):
    for seed in DEV_SEEDS:
        tasks = _if_tasks(seed) if task_id in NEW_A else _r_tasks(seed)
        task = _by_id(tasks, task_id)
        good = (instruction if task.category == "A" else reasoning).evaluate(
            task, _transcript(task, task.oracle["reference"])
        )
        assert good.earned == good.possible == task.max_points
        bad = (instruction if task.category == "A" else reasoning).evaluate(task, _transcript(task, "не знаю"))
        assert bad.earned == 0.0


def test_r17_schedule_is_recomputable_from_the_prompt():
    """The oracle is a schedule, not a copied string: recompute it independently."""
    def finish(start: date, days: int, holidays: set[date]) -> date:
        day = start
        while day.weekday() >= 5 or day in holidays:
            day += timedelta(days=1)
        left = days
        while True:
            left -= 1
            if left == 0:
                return day
            day += timedelta(days=1)
            while day.weekday() >= 5 or day in holidays:
                day += timedelta(days=1)

    for seed in DEV_SEEDS:
        for lang in ("ru", "kk"):
            task = _r17(seed, lang)
            year = int(task.meta["start"][:4])
            holidays = {
                day
                for day, _, _ in reasoning._r17_calendar(year)[0]
            }
            expected = finish(
                date.fromisoformat(task.meta["start"]), task.meta["working_days"], holidays
            )
            assert task.oracle["reference"] == expected.isoformat() == task.meta["finish"]
            assert date.fromisoformat(expected.isoformat()).weekday() < 5


# --------------------------------------------------------------------------- #
# Resistant mutants
# --------------------------------------------------------------------------- #


def _if7_payload(task: core.TaskInstance) -> dict:
    return json.loads(task.oracle["reference"])


def test_if7_keeping_the_native_date_loses_the_rows():
    task = _if7(1, "ru")
    payload = _if7_payload(task)
    payload["rows"][0]["date"] = "15 января 2027 г."
    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert score.earned < 2.0
    assert _checks(score)["rows"] is False


def test_if7_dropping_the_inclusive_boundary_loses_the_rows():
    task = _if7(1, "ru")
    payload = _if7_payload(task)
    boundary = Decimal("12.0")
    payload["rows"] = [row for row in payload["rows"] if Decimal(str(row["tonnes"])) != boundary]
    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert score.earned < 2.0
    assert _checks(score)["rows"] is False


def test_if7_forgetting_from_kilograms_to_tonnes_loses_both_values():
    task = _if7(2, "ru")
    payload = _if7_payload(task)
    for row in payload["rows"]:
        row["tonnes"] = round(row["tonnes"] * 1000, 3)
    payload["total_tonnes"] = round(payload["total_tonnes"] * 1000, 3)
    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert _checks(score) == {"json_format": True, "rows": False, "total": False}


def test_if7_kazakh_fixture_has_its_own_boundary_mutant():
    task = _if7(3, "kk")
    payload = _if7_payload(task)
    boundary = Decimal("12.0")
    payload["rows"] = [row for row in payload["rows"] if Decimal(str(row["tonnes"])) != boundary]
    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert _checks(score)["rows"] is False


def test_if8_skipping_nfc_leaves_duplicate_keys():
    task = _by_id(_if_tasks(1), "IF-8")
    payload = json.loads(task.oracle["reference"])
    doubled = {"keys": payload["keys"] + [
        {"key": unicodedata.normalize("NFD", payload["keys"][0]["key"]),
         "value": payload["keys"][0]["value"] + "-old"}
    ]}
    score = instruction.evaluate(task, _transcript(task, _answer(doubled)))
    assert score.earned < 2.0
    assert _checks(score)["keys"] is False


def test_if8_unsorted_answer_loses_only_the_order_point():
    task = _by_id(_if_tasks(1), "IF-8")
    payload = json.loads(task.oracle["reference"])
    payload["keys"].reverse()
    score = instruction.evaluate(task, _transcript(task, _answer(payload)))
    assert _checks(score) == {"json_format": True, "keys": True, "order": False}
    assert score.earned == 1.25


def _r17_wrong(task: core.TaskInstance, *, weekends: bool = False) -> str:
    """Recompute with holidays ignored (or weekends counted) for the seed."""
    day = date.fromisoformat(task.meta["start"])
    left = task.meta["working_days"]
    while day.weekday() >= 5:
        day += timedelta(days=1)
    while True:
        left -= 1
        if left == 0:
            return day.isoformat()
        day += timedelta(days=1)
        if not weekends:
            while day.weekday() >= 5:
                day += timedelta(days=1)


def _r17_case(lang: str, *, weekends: bool = False) -> tuple[core.TaskInstance, str]:
    """The first seed whose span really crosses a holiday or a weekend."""
    for seed in range(1, 400):
        task = _r17(seed, lang)
        wrong = _r17_wrong(task, weekends=weekends)
        if wrong != task.oracle["reference"]:
            return task, wrong
    raise AssertionError("no seed crosses the expected non-working days")


def test_r17_ignoring_holidays_misses_the_date():
    task, wrong = _r17_case("ru")
    holidays = {day for day, _, _ in reasoning._r17_calendar(int(task.meta["start"][:4]))[0]}
    assert any(day.isoformat() in task.prompt for day in holidays)
    assert reasoning.evaluate(task, _transcript(task, wrong)).earned == 0.0
    assert reasoning.evaluate(task, _transcript(task, task.oracle["reference"])).earned == 1.0


def test_r17_counting_weekends_misses_the_date():
    task, wrong = _r17_case("kk", weekends=True)
    assert reasoning.evaluate(task, _transcript(task, wrong)).earned == 0.0


def test_r17_rejects_a_non_iso_or_impossible_date():
    task = _r17(1, "ru")
    for answer in ("4 мая 2027", "2027-02-30", task.oracle["reference"] + " г."):
        score = reasoning.evaluate(task, _transcript(task, answer))
        assert score.earned == 0.0
    assert reasoning.evaluate(task, _transcript(task, task.oracle["reference"])).earned == 1.0
