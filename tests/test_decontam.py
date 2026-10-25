"""Contamination filter and private-layer contract."""

from __future__ import annotations

import unicodedata

import pytest

from dailybench import core
from dailybench.decontam import (
    CONTAMINATED,
    check_corpus,
    is_private,
    private_gate,
    publishable,
    shingles,
)
from dailybench.suite import public_prompts

PHRASE = (
    "Ниже журнал смен и три версии политики построй расписание на дату среза "
    "учитывая отзыв последней версии и tombstone удалённых записей"
)
#: Long untouched head of PHRASE: a paraphrase keeps these words in order.
KEPT = " ".join(PHRASE.split()[:20])


def make_task(task_id: str, prompt: str, category: str = "A", private: bool = False) -> core.TaskInstance:
    task = core.TaskInstance(
        id=task_id, category=category, title=task_id, prompt=prompt, max_points=1.0
    )
    task.private = private
    return task


def test_exact_duplicate_is_fully_contaminated():
    report = check_corpus([make_task("A-1", PHRASE)], [PHRASE])
    row = report.rows[0]
    assert row.share == 1.0
    assert row.contaminated is True
    assert report.contaminated is True
    assert report.to_dict()["verdict"] == "contaminated"
    assert report.contaminated_ids == ["A-1"]
    assert "A-1" in report.affected_ids


def test_paraphrase_with_long_kept_phrases_is_partial():
    borrowed = KEPT + (
        " затем переписанный хвост совсем другими словами про кухню погоду "
        "море горы музыку книги и прочее"
    )
    report = check_corpus([make_task("A-2", borrowed)], [PHRASE])
    share = report.rows[0].share
    assert 0.0 < share < CONTAMINATED
    assert report.rows[0].contaminated is False
    assert report.contaminated is False


def test_independent_text_has_zero_overlap():
    other = "Совершенно другой набор терминов про кулинарию и погоду за окном сегодня"
    report = check_corpus([make_task("A-3", other)], [PHRASE])
    assert report.rows[0].share == 0.0
    assert report.affected_ids == []
    assert report.contaminated is False


def test_nfc_and_case_do_not_hide_a_duplicate():
    text = "Съешь ещё этих мягких французских булок да выпей чаю с йодом и майонезом"
    corpus = unicodedata.normalize("NFD", text).upper()
    report = check_corpus([make_task("A-4", unicodedata.normalize("NFD", text.lower()))], [corpus])
    assert report.rows[0].matched == report.rows[0].shingles
    assert report.rows[0].share == 1.0


def test_weak_duplicate_below_threshold_does_not_flag():
    unrelated = [make_task("A-5", "Первое задание про таблицы"), make_task("A-6", PHRASE)]
    report = check_corpus(unrelated, [PHRASE])
    assert report.affected_ids == ["A-6"]
    assert report.contaminated_ids == ["A-6"]


def test_short_exact_text_still_matches_itself():
    report = check_corpus([make_task("A-7", "Короткий текст")], ["Короткий текст"])
    assert report.rows[0].share == 1.0


def test_ngram_size_is_respected():
    assert len(shingles("a b c d e", n=2)) == 4
    report = check_corpus([make_task("A-8", "a b c d e")], ["a b c"], n=2)
    assert report.rows[0].share == 0.5


# --------------------------------------------------------------------------- #
# Private layer
# --------------------------------------------------------------------------- #


def test_private_gate_passes_at_a_fifth():
    tasks = [
        make_task(f"{cat}-{index}", "текст", cat, private=index < 2)
        for cat in "ABCDEFGHIJ"
        for index in range(10)
    ]
    report = private_gate(tasks, "stretch-v1")
    assert report.gated is True
    assert report.share == 0.2
    assert report.ok is True
    assert report.failed == []


def test_private_gate_fails_when_a_category_is_empty():
    """A fifth overall is not enough: every category must clear the mark."""
    tasks = [make_task(f"A-{index}", "текст", "A", private=index < 2) for index in range(10)]
    report = private_gate(tasks, "stretch-v1")
    assert report.share >= report.required
    assert report.ok is False
    assert report.failed == list("BCDEFGHIJ")
    assert "B" in report.render()


def test_private_gate_fails_below_a_fifth():
    tasks = [make_task(f"{cat}-1", "текст", cat, private=cat == "A") for cat in "ABCDEFGHIJ"]
    report = private_gate(tasks, "stretch-v1")
    assert report.ok is False
    assert "B" in report.failed
    assert report.to_dict()["verdict"] == "failed"


def test_private_gate_is_inactive_on_other_profiles():
    tasks = [make_task("A-1", "текст", "A")]
    report = private_gate(tasks, core.DEFAULT_PROFILE)
    assert report.gated is False
    assert report.ok is True
    assert report.failed == []


def test_private_prompt_never_reaches_public_prompts():
    tasks = [make_task("A-1", "открытый промпт"), make_task("A-2", "секретный промпт", private=True)]
    published = public_prompts(tasks)
    assert [entry["id"] for entry in published] == ["A-1", "A-2"]
    private_entry = published[1]
    assert private_entry["private"] is True
    assert "prompt" not in private_entry
    assert "секретный промпт" not in str(private_entry)
    assert len(private_entry["prompt_sha256"]) == 64
    with_private = public_prompts(tasks, include_private=True)
    assert with_private[1]["prompt"] == "секретный промпт"


def test_publishable_and_is_private():
    tasks = [make_task("A-1", "x"), make_task("A-2", "y", private=True)]
    assert [task.id for task in publishable(tasks)] == ["A-1"]
    assert is_private(tasks[1]) is True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_decontam_flags_a_duplicate(tmp_path, capsys):
    from dailybench import cli

    candidate = tmp_path / "candidate.txt"
    candidate.write_text(PHRASE, encoding="utf-8")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(PHRASE, encoding="utf-8")
    code = cli.main(["decontam", "--corpus", str(corpus), "--input", str(candidate), "--json"])
    payload = __import__("json").loads(capsys.readouterr().out)
    assert code == 1
    assert payload["verdict"] == "contaminated"
    assert payload["contaminated_ids"] == [str(candidate)]


def test_cli_decontam_clean_text_exits_zero(tmp_path, capsys):
    from dailybench import cli

    candidate = tmp_path / "candidate.txt"
    candidate.write_text("Совершенно независимый текст про горы и море", encoding="utf-8")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(PHRASE, encoding="utf-8")
    code = cli.main(["decontam", "--corpus", str(corpus), "--input", str(candidate)])
    assert code == 0
    assert "чисто" in capsys.readouterr().out


def test_cli_decontam_needs_candidates(capsys):
    from dailybench import cli

    assert cli.main(["decontam", "--corpus", "docs/FRAMEWORK_SPEC.md"]) == 2
    assert "нужен --suite или --input" in capsys.readouterr().err


def test_cli_private_check_baseline_is_not_gated(capsys):
    from dailybench import cli

    code = cli.main(["private-check", "--only", "A"])
    assert code == 0
    assert "гейт не применяется" in capsys.readouterr().out


def test_threshold_applies_per_row() -> None:
    task = make_task("A-9", "a b c d")
    corpus = ["a b x y"]
    loose = check_corpus([task], corpus, n=2, threshold=0.25)
    assert loose.rows[0].share == pytest.approx(1 / 3)
    assert loose.rows[0].contaminated is True
    assert loose.contaminated is True
    assert loose.to_dict()["threshold"] == 0.25
    strict = check_corpus([task], corpus, n=2, threshold=0.5)
    assert strict.rows[0].contaminated is False
    assert strict.contaminated is False


def test_cli_decontam_honours_threshold(tmp_path, capsys):
    from dailybench import cli

    candidate = tmp_path / "candidate.txt"
    candidate.write_text("a b c d", encoding="utf-8")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b x y", encoding="utf-8")
    args = ["decontam", "--corpus", str(corpus), "--input", str(candidate), "--ngram", "2"]
    assert cli.main([*args, "--threshold", "0.5", "--json"]) == 0
    assert __import__("json").loads(capsys.readouterr().out)["verdict"] == "clean"
    assert cli.main([*args, "--threshold", "0.25", "--json"]) == 1
    assert __import__("json").loads(capsys.readouterr().out)["verdict"] == "contaminated"
