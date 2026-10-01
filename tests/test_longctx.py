from __future__ import annotations

from dailybench import ModelReply, RunConfig, Score, TaskInstance, Transcript, Turn
from dailybench.rng import task_rng
from dailybench.tasks.longctx import evaluate, generate


def _transcript(task: TaskInstance, answer: str, *, prompt_tokens: int | None = None, error: str | None = None) -> Transcript:
    return Transcript(
        task_id=task.id,
        category=task.category,
        turns=[Turn(messages=[{"role": "user", "content": task.prompt}], reply=ModelReply(text=answer, prompt_tokens=prompt_tokens))],
        error=error,
    )


def test_longctx_ids_points_and_self_checks() -> None:
    cfg = RunConfig()
    tasks = generate(task_rng(20260915, "E"), cfg)
    assert [task.id for task in tasks] == ["LC-1", "LC-2", "LC-3", "LC-4", "LC-5"]
    assert sum(task.max_points for task in tasks) == 10.0
    for task in tasks:
        good = evaluate(task, _transcript(task, task.oracle["reference"]))
        assert good.earned == 2.0
        assert good.possible == 2.0
        assert evaluate(task, _transcript(task, "")).earned == 0.0
        assert evaluate(task, _transcript(task, "не знаю")).earned == 0.0
        if task.id == "LC-5":
            assert evaluate(task, _transcript(task, task.oracle["distractor_answer"])).earned == 0.0
        if task.id == "LC-4":
            assert evaluate(task, _transcript(task, task.oracle["naive_last_line_answer"])).earned == 0.0


def test_longctx_is_deterministic_and_seeded() -> None:
    cfg = RunConfig()
    first = generate(task_rng(17, "E"), cfg)
    second = generate(task_rng(17, "E"), cfg)
    other = generate(task_rng(18, "E"), cfg)
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle for task in first] == [task.oracle for task in second]
    assert [task.id for task in first] == [task.id for task in other]
    assert any(a.prompt != b.prompt for a, b in zip(first, other))
    assert any(a.oracle["reference"] != b.oracle["reference"] for a, b in zip(first, other))


def test_longctx_sizes_and_unique_chain_facts() -> None:
    tasks = generate(task_rng(20260915, "E"), RunConfig())
    for task in tasks:
        estimated = task.meta["estimated_tokens"]
        target = task.meta["target_tokens"]
        assert target * 0.75 <= estimated <= target * 1.25
        assert task.meta["prompt_chars"] == len(task.prompt)
        for fact in task.oracle.get("target_facts", []):
            assert task.prompt.split("\n\nQUESTION:", 1)[0].count(fact) == 1
    assert tasks[3].meta["estimated_tokens"] >= 20_000
    assert tasks[4].meta["estimated_tokens"] >= 20_000
    assert tasks[3].oracle["naive_last_line_answer"] != tasks[3].oracle["answer"]
    assert tasks[4].oracle["distractor_answer"] != tasks[4].oracle["answer"]


def test_longctx_non_admitted_variant() -> None:
    cfg = RunConfig(context_window=1024, n_ctx=1024, max_new_tokens=256)
    task = generate(task_rng(3, "E"), cfg)[-1]
    assert task.meta["admission_required"] == task.meta["target_tokens"]
    score = evaluate(task, _transcript(task, task.oracle["reference"]))
    assert score.admission is False
    assert score.notes
    assert score.possible == 2.0


def test_longctx_reported_overflow_and_measured_admission() -> None:
    task = generate(task_rng(4, "E"), RunConfig())[0]
    overflow = evaluate(task, _transcript(task, task.oracle["reference"], error="context length exceeded"))
    assert overflow.admission is False

    # A measured input that really fills the target band is admitted.
    target = task.meta["target_tokens"]
    admitted = evaluate(task, _transcript(task, task.oracle["reference"], prompt_tokens=int(target * 0.95)))
    assert admitted.admission is True

    # An input far below the target is not the same task: it must be reported as
    # an admission failure, never scored as if the context had been exercised.
    undersized = evaluate(task, _transcript(task, task.oracle["reference"], prompt_tokens=int(target * 0.2)))
    assert undersized.admission is False
    assert any("не дотянул" in note for note in undersized.notes)


def test_longctx_accepts_the_documented_spellings_only() -> None:
    """The entity must be right; repeating the document's field label is allowed."""
    tasks = generate(task_rng(11, "E"), RunConfig())
    lc1, lc5 = tasks[0], tasks[4]

    plain = evaluate(lc1, _transcript(lc1, lc1.oracle["answer"], prompt_tokens=8000))
    assert plain.admission is True and plain.earned == 2.0

    prefixed = evaluate(lc1, _transcript(lc1, f"sha256:{lc1.oracle['answer']}", prompt_tokens=8000))
    assert prefixed.earned == 2.0, "префикс поля не несёт информации о качестве поиска"

    wrong = evaluate(lc1, _transcript(lc1, "sha256:deadbeef", prompt_tokens=8000))
    assert wrong.earned == 0.0

    # The sibling service is a deliberate distractor and must never be accepted.
    distractor = evaluate(lc5, _transcript(lc5, str(lc5.oracle["distractor_answer"]), prompt_tokens=27000))
    assert distractor.earned == 0.0
    labeled = evaluate(lc5, _transcript(lc5, f"service {lc5.oracle['answer']}", prompt_tokens=27000))
    assert labeled.earned == 2.0
