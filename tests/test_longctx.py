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

    plain = evaluate(lc1, _transcript(lc1, lc1.oracle["answer"], prompt_tokens=lc1.meta["target_tokens"]))
    assert plain.admission is True and plain.earned == 2.0

    prefixed = evaluate(lc1, _transcript(lc1, f"sha256:{lc1.oracle['answer']}", prompt_tokens=lc1.meta["target_tokens"]))
    assert prefixed.earned == 2.0, "префикс поля не несёт информации о качестве поиска"

    wrong = evaluate(lc1, _transcript(lc1, "sha256:deadbeef", prompt_tokens=lc1.meta["target_tokens"]))
    assert wrong.earned == 0.0

    # The sibling service is a deliberate distractor and must never be accepted.
    distractor = evaluate(lc5, _transcript(lc5, str(lc5.oracle["distractor_answer"]), prompt_tokens=lc5.meta["target_tokens"]))
    assert distractor.earned == 0.0
    labeled = evaluate(lc5, _transcript(lc5, f"service {lc5.oracle['answer']}", prompt_tokens=lc5.meta["target_tokens"]))
    assert labeled.earned == 2.0


def _task(cfg: RunConfig, seed: int, task_id: str) -> TaskInstance:
    from dailybench.tasks.longctx import generate as _generate

    return next(task for task in _generate(task_rng(seed, "E"), cfg) if task.id == task_id)


def _score(task: TaskInstance, answer: str) -> Score:
    return evaluate(task, _transcript(task, answer, prompt_tokens=int(task.meta["target_tokens"])))


def test_longctx_profiles_select_their_scenarios() -> None:
    """Pilot holds only LC-6, stretch keeps the baseline and adds both new ones."""
    assert [task.id for task in generate(task_rng(5, "E"), RunConfig(profile="pilot"))] == ["LC-6"]
    assert [task.id for task in generate(task_rng(5, "E"), RunConfig())] == ["LC-1", "LC-2", "LC-3", "LC-4", "LC-5"]
    assert [task.id for task in generate(task_rng(5, "E"), RunConfig(profile="stretch-v1"))] == [
        "LC-1", "LC-2", "LC-3", "LC-4", "LC-5", "LC-6", "LC-7", "LC-8",
    ]


def test_lc6_full_credit_and_every_mutant_fails() -> None:
    """The reference answer scores; positional, tombstone, cutoff and ID-less answers do not."""
    task = _task(RunConfig(profile="pilot"), 3, "LC-6")  # seed 3 is UNRESOLVED
    assert task.oracle["status"] == "UNRESOLVED"
    assert _score(task, task.oracle["reference"]).earned == 2.0
    assert evaluate(task, _transcript(task, "")).earned == 0.0
    assert evaluate(task, _transcript(task, "не знаю")).earned == 0.0
    for key in ("naive_last_position_answer", "naive_ignore_tombstone_answer", "naive_ignore_cutoff_answer"):
        answer = task.oracle[key]
        assert answer != task.oracle["reference"], key
        assert _score(task, answer).earned == 0.0, key
    assert _score(task, task.oracle["answer_without_ids"]).earned == 0.0
    # Both halves are required in one line: status alone or IDs alone are not enough.
    ids_only = "СОБЫТИЯ " + ",".join(task.oracle["evidence"])
    assert _score(task, ids_only).earned == 0.0
    # A wrapped two-line reply is not the single-line answer format.
    assert _score(task, task.oracle["reference"].replace("; ", ";\n")).earned == 0.0


def test_lc6_events_are_unique_and_both_statuses_occur() -> None:
    cfg = RunConfig(profile="pilot")
    statuses = set()
    for seed in (101, 102, 103, 104, 105):
        task = _task(cfg, seed, "LC-6")
        statuses.add(task.oracle["status"])
        assert len(task.oracle["events"]) == 10
        assert len({event["id"] for event in task.oracle["events"]}) == 10
        assert _score(task, task.oracle["reference"]).earned == 2.0
        document = task.prompt.split("\n\nQUESTION:", 1)[0]
        for fact in task.oracle["target_facts"]:
            assert document.count(fact) == 1
        assert task.oracle["cutoff"] in task.prompt
        # The evidence is exactly the deciding events of the reference machine.
        assert task.oracle["evidence"]
        assert set(task.oracle["evidence"]) <= {event["id"] for event in task.oracle["events"]}
        assert 0.75 <= task.meta["estimated_tokens"] / task.meta["target_tokens"] <= 1.25
    assert statuses == {"ASSIGNED", "UNRESOLVED"}


def test_lc7_chain_full_credit_and_mutants() -> None:
    task = _task(RunConfig(profile="stretch-v1"), 7, "LC-7")
    assert len(task.oracle["evidence"]) == 4
    assert task.oracle["answer"].startswith(f"OWNER {task.oracle['answer_entity']}; СОБЫТИЯ ")
    assert _score(task, task.oracle["reference"]).earned == 2.0
    assert evaluate(task, _transcript(task, "не знаю")).earned == 0.0
    for key in ("naive_last_position_answer", "naive_mixed_chain_answer", "naive_last_version_answer"):
        assert task.oracle[key] != task.oracle["reference"], key
        assert _score(task, task.oracle[key]).earned == 0.0, key
    assert _score(task, task.oracle["answer_without_ids"]).earned == 0.0
    # Right owner with the wrong (distractor) evidence is not a pass.
    wrong_ids = task.oracle["evidence"][::-1]
    assert _score(task, f"OWNER {task.oracle['answer_entity']}; СОБЫТИЯ {','.join(wrong_ids)}").earned == 0.0


def test_lc7_graph_is_unique_sized_and_seeded() -> None:
    cfg = RunConfig(profile="stretch-v1")
    for seed in (201, 202, 203, 204, 205):
        first = _task(cfg, seed, "LC-7")
        second = _task(cfg, seed, "LC-7")
        assert first.prompt == second.prompt and first.oracle == second.oracle
        target = first.meta["target_tokens"]
        assert round(cfg.context_window * 0.73) <= target <= round(cfg.context_window * 0.86)
        assert target * 0.75 <= first.meta["estimated_tokens"] <= target * 1.25
        document = first.prompt.split("\n\nQUESTION:", 1)[0]
        for fact in first.oracle["target_facts"]:
            assert document.count(fact) == 1
        edges = first.oracle["edges"]
        assert len(edges) == 12
        assert len({edge["id"] for edge in edges}) == 12
        # One versioned edge: two versions share a key and the target node.
        versions = {}
        for edge in edges:
            versions.setdefault(edge["key"], []).append(edge)
        shared = [group for group in versions.values() if len(group) == 2]
        assert len(shared) == 2
        assert all(len({edge["version"] for edge in group}) == 2 for group in shared)
        assert first.oracle["distractor_answer"] != first.oracle["reference"]
        assert _score(first, first.oracle["reference"]).earned == 2.0


def test_pilot_sizing_calibrates_the_pilot_long_context_task() -> None:
    """Calibration must not depend on the baseline id LC-1 being present."""
    from dailybench import core
    from dailybench.providers import CallableProvider
    from dailybench.suite import build_suite as _build
    from dailybench.suite import fit_suite_sizing

    cfg = RunConfig(profile="pilot")
    assert [task.id for task in _build(cfg) if task.category == "E"] == ["LC-6"]

    def fn(messages, tools):
        text = messages[-1]["content"]
        return core.ModelReply(text="ok", prompt_tokens=max(1, len(text) // 2))

    result = fit_suite_sizing(cfg, CallableProvider(fn=fn))
    assert result["calibrated"] is True
    assert "нет long-context" not in result["note"]
    assert set(cfg.task_ratios) == {"LC-6"}


def test_long_context_answers_do_not_depend_on_the_sizing_ratio() -> None:
    """A ratio change resizes the documents but must not move the facts.

    Calibration measures a provisional suite and then rebuilds it at the
    measured ratio; the tasks the model is scored on must still be the ones
    that were measured.
    """
    seed = 5150
    low = {task.id: task for task in generate(task_rng(seed, "E"), RunConfig(chars_per_token=2.0))}
    high = {task.id: task for task in generate(task_rng(seed, "E"), RunConfig(chars_per_token=4.0))}
    for task_id in ("LC-2", "LC-3", "LC-4", "LC-5"):
        assert low[task_id].oracle["answer"] == high[task_id].oracle["answer"], task_id
        assert low[task_id].oracle["reference"] == high[task_id].oracle["reference"], task_id
        assert low[task_id].meta["prompt_chars"] != high[task_id].meta["prompt_chars"], task_id
        assert low[task_id].prompt != high[task_id].prompt, task_id


def test_a_per_task_ratio_touches_only_its_own_document() -> None:
    """Calibration coefficients are per document and never a global reshuffle."""
    seed = 4242
    plain = {task.id: task for task in generate(task_rng(seed, "E"), RunConfig())}
    tuned = {
        task.id: task
        for task in generate(task_rng(seed, "E"), RunConfig(task_ratios={"LC-3": 5.5}))
    }
    # Offsets move with any resize, so semantics is compared without them.
    for task_id, task in plain.items():
        for key in ("answer", "reference", "target_facts"):
            assert tuned[task_id].oracle.get(key) == task.oracle.get(key), (task_id, key)
    assert tuned["LC-3"].oracle["target_facts"] == plain["LC-3"].oracle["target_facts"]
    assert tuned["LC-3"].meta["prompt_chars"] != plain["LC-3"].meta["prompt_chars"]
    for task_id in ("LC-1", "LC-2", "LC-4", "LC-5"):
        assert tuned[task_id].prompt == plain[task_id].prompt, task_id


def test_a_scenario_keeps_its_facts_in_every_profile() -> None:
    """LC-6 is the same task whether or not the baseline scenarios precede it."""
    seed = 77
    pilot = next(iter(generate(task_rng(seed, "E"), RunConfig(profile="pilot"))))
    stretch = _task(RunConfig(profile="stretch-v1"), seed, "LC-6")
    assert pilot.prompt == stretch.prompt
    assert pilot.oracle == stretch.oracle
    assert pilot.meta["fact_positions"] == stretch.meta["fact_positions"]


def _stretch_task(seed: int, task_id: str = "LC-8") -> TaskInstance:
    return _task(RunConfig(profile="stretch-v1"), seed, task_id)


def _walk_records(records: list[dict], start: tuple[str, str], cutoff: str) -> tuple[str, list[str]]:
    """Independent traversal of the oracle graph, written from the trust rule."""
    rank = {"primary": 2, "secondary": 1}
    live = [record for record in records if record["ts"] <= cutoff and record["source"] in rank]
    winners = {
        key: max((record for record in live if record["key"] == key), key=lambda r: (rank[r["source"]], r["version"]))
        for key in {record["key"] for record in live}
    }
    current = start
    ids: list[str] = []
    while current[0] != "artifact":
        match = [record for record in winners.values() if record["subject"] == current]
        assert len(match) == 1, current
        ids.append(match[0]["id"])
        current = match[0]["object"]
    return current[1], ids


def test_n5_authority_full_credit_and_every_mutant_fails() -> None:
    """Only the value reached by the trust rule with its own witnesses scores."""
    task = _stretch_task(301)
    assert task.oracle["format"] == "authority"
    assert _score(task, task.oracle["reference"]).earned == 2.0
    assert evaluate(task, _transcript(task, "")).earned == 0.0
    assert evaluate(task, _transcript(task, "не знаю")).earned == 0.0
    for key in (
        "naive_last_position_answer",
        "naive_unverified_answer",
        "naive_low_priority_answer",
        "naive_ignore_cutoff_answer",
        "distractor_answer",
        "answer_without_evidence",
        "answer_by_question_id",
    ):
        assert task.oracle[key] != task.oracle["reference"], key
        assert _score(task, task.oracle[key]).earned == 0.0, key
    # The right value with evidence borrowed from the neighbouring chain is not a pass.
    assert task.oracle["right_value_foreign_evidence"].startswith(task.oracle["reference"].split(";")[0])
    assert _score(task, task.oracle["right_value_foreign_evidence"]).earned == 0.0
    # Witnesses without the value, and a wrapped two-line reply, are not the format.
    assert _score(task, "СВИДЕТЕЛИ " + ",".join(task.oracle["evidence"])).earned == 0.0
    assert _score(task, task.oracle["reference"].replace("; ", ";\n")).earned == 0.0


def test_n5_authority_accepts_the_documented_spellings_only() -> None:
    """The hex value is exact; the field label's case is not information."""
    task = _stretch_task(302)
    upper = task.oracle["reference"].replace("sha256:", "SHA256: ")
    assert _score(task, upper).earned == 2.0
    other = next(value for value in ("01234567", "89abcdef") if value not in task.oracle["answer_artifact"])
    assert _score(task, f"ARTIFACT sha256:{other}; СВИДЕТЕЛИ {','.join(task.oracle['evidence'])}").earned == 0.0
    # A different witness order is a different citation.
    reversed_ids = task.oracle["evidence"][::-1]
    assert _score(task, f"ARTIFACT sha256:{task.oracle['answer_artifact']}; СВИДЕТЕЛИ {','.join(reversed_ids)}").earned == 0.0


def test_n5_authority_graph_is_unique_sized_and_seeded() -> None:
    """Five dev seeds: deterministic, in the declared window band, one path only."""
    from dailybench.tasks import longctx

    cfg = RunConfig(profile="stretch-v1")
    for seed in (301, 302, 303, 304, 305):
        first = _stretch_task(seed)
        second = _stretch_task(seed)
        assert first.prompt == second.prompt and first.oracle == second.oracle
        target = first.meta["target_tokens"]
        assert round(cfg.context_window * 0.76) <= target <= round(cfg.context_window * 0.89)
        assert target * 0.75 <= first.meta["estimated_tokens"] <= target * 1.25
        assert first.fresh_after == longctx.FRESH_AFTER
        assert first.languages == ["ru"]
        assert first.private is True

        document = first.prompt.split("\n\nQUESTION:", 1)[0]
        assert first.oracle["cutoff"] in document
        for fact in first.oracle["target_facts"]:
            assert document.count(fact) == 1
        records = first.oracle["records"]
        assert len(records) == 12
        assert len({record["id"] for record in records}) == 12
        assert {record["source"] for record in records} == {"primary", "secondary", "chat"}

        # Both traversals agree, the evidence is exactly the path, and the
        # superseded mention is the last record of the document.
        value, ids = _walk_records(records, ("incident", first.oracle["target_incident"]), first.oracle["cutoff"])
        assert value == first.oracle["answer_artifact"]
        assert ids == first.oracle["evidence"]
        assert len(ids) == 3
        positions = {record["id"]: document.find(f"record {record['id']}") for record in records}
        assert min(positions.values()) >= 0
        assert max(positions, key=positions.get) == next(
            record["id"] for record in records
            if record["object"] == ("artifact", first.oracle["naive_last_position_answer"].split("sha256:")[1].split(";")[0])
        )
        assert _score(first, first.oracle["reference"]).earned == 2.0


def test_n5_authority_ranks_sources_before_versions_and_cutoff() -> None:
    """Each trap is a real record: chat is highest-version, the copy is fresher."""
    task = _stretch_task(303)
    records = task.oracle["records"]
    cutoff = task.oracle["cutoff"]
    target = task.oracle["target_incident"]
    evidence = task.oracle["evidence"]

    by_id = {record["id"]: record for record in records}
    path = [by_id[edge_id] for edge_id in evidence]
    # A contiguous incident -> deployment -> release -> artifact path.
    assert path[0]["subject"] == ("incident", target)
    for left, right in zip(path, path[1:]):
        assert left["object"] == right["subject"]
    assert path[-1]["object"] == ("artifact", task.oracle["answer_artifact"])
    assert all(record["source"] == "primary" for record in path)
    assert all(record["ts"] <= cutoff for record in path)

    # Every losing record is a genuine, printed record.
    artifact_key = path[-1]["key"]
    rivals = [record for record in records if record["key"] == artifact_key and record["id"] != path[-1]["id"]]
    chat = [record for record in rivals if record["source"] == "chat"]
    assert chat and chat[0]["version"] > max(
        record["version"] for record in records
        if record["key"] == artifact_key and record["source"] != "chat"
    )
    assert chat[0]["ts"] <= cutoff
    late = [record for record in records if record["ts"] > cutoff and record["source"] == "primary"]
    assert late and late[0]["object"][0] == "artifact"
    assert all(record["id"] in by_id for record in rivals + late)
    # The chat deployment claim is not a second path, and the copy's tail is complete.
    assert any(record["source"] == "chat" and record["subject"] == ("incident", target) for record in records)
    copy_tail = next(
        record for record in records
        if record["source"] == "secondary" and record["object"][0] == "release"
    )
    assert copy_tail["subject"] == path[1]["subject"] and copy_tail["object"] != path[1]["object"]
    assert any(record["subject"] == copy_tail["object"] for record in records)
    # The neighbouring chain is complete on its own and ends elsewhere.
    other_value, other_ids = _walk_records(records, ("incident", task.oracle["distractor_incident"]), cutoff)
    assert other_value == task.oracle["distractor_artifact"] != task.oracle["answer_artifact"]
    assert other_ids == task.oracle["distractor_evidence"]
    assert not set(other_ids) & set(evidence)
