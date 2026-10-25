"""Category V: typed local assets, the multimodal wire and the VQA oracle.

The block exists to prove one thing a screenshot cannot: the endpoint really
receives the picture. Every test therefore goes through the real ``Runner`` and
reads the captured request, not a hand-built message. A text-only provider must
come out ``skipped`` instead of scoring a fake zero accuracy.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest

from dailybench import core, schema
from dailybench.config import RunConfig
from dailybench.providers import CallableProvider, ScriptedProvider
from dailybench.runner import Runner
from dailybench.tasks import load_all, visual_input

load_all()

CFG = RunConfig(profile="research-v1")


def _tasks(seed: int = 41) -> dict[str, core.TaskInstance]:
    return {task.id: task for task in visual_input.generate(random.Random(seed), CFG)}


def _runner(provider) -> Runner:
    return Runner(CFG, provider)


def _image_parts(messages) -> list[dict]:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts.extend(part for part in content if part.get("type") == "image_url")
    return parts


# --------------------------------------------------------------------------- #
# Fixtures and schema
# --------------------------------------------------------------------------- #


def test_three_scenarios_are_generated_with_typed_png_assets():
    tasks = _tasks()
    assert list(tasks) == list(visual_input.IDS)
    for task in tasks.values():
        assert task.category == "V"
        assert (task.block, task.family) == ("open", "vis/vqa")
        assert task.max_points == 1.0
        assert len(task.assets or []) == 1
        asset = task.assets[0]
        assert asset.media_type == "image/png"
        assert asset.data.startswith(b"\x89PNG\r\n\x1a\n")
        assert (int.from_bytes(asset.data[16:20], "big"), int.from_bytes(asset.data[20:24], "big")) == core.ASSET_SIZE
        assert schema.validate_task(task) == []
        # The filename alone cannot answer: it names the kind, not the value.
        assert task.oracle["answer"].lower() not in Path(asset.path).stem.lower()
        assert task.oracle["mutant"] != task.oracle["answer"]


def test_every_task_has_a_mutant_that_scores_zero():
    for task in _tasks().values():
        runner = _runner(ScriptedProvider([str(task.oracle["mutant"])], images=True))
        run = runner.run_task(task)
        assert run.score.earned == 0.0, task.id


def test_generation_is_seed_deterministic_and_the_answers_move():
    first = {task_id: task.assets[0].sha256 for task_id, task in _tasks(7).items()}
    again = {task_id: task.assets[0].sha256 for task_id, task in _tasks(7).items()}
    assert first == again
    answers = {seed: tuple(task.oracle["answer"] for task in _tasks(seed).values()) for seed in range(6)}
    assert len(set(answers.values())) > 1


# --------------------------------------------------------------------------- #
# The wire
# --------------------------------------------------------------------------- #


def test_the_request_carries_an_image_url_and_no_external_link():
    task = _tasks()["V-1"]
    provider = ScriptedProvider([str(task.oracle["answer"])], images=True)
    run = _runner(provider).run_task(task)
    assert run.score.earned == task.max_points

    payload = provider.calls[0]
    parts = _image_parts(payload)
    assert len(parts) == 1
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert "http" not in url
    # No alt text: the model gets the bytes, not a hint.
    assert set(parts[0]["image_url"]) == {"url"}
    body = url.split(",", 1)[1]
    assert hashlib.sha256(__import__("base64").b64decode(body)).hexdigest() == task.assets[0].sha256
    # The text part is still the prompt verbatim.
    text_part = next(part for part in payload[-1]["content"] if part["type"] == "text")
    assert text_part["text"] == task.prompt


def test_a_text_only_model_is_skipped_not_scored_zero():
    task = _tasks()["V-2"]
    provider = ScriptedProvider([str(task.oracle["answer"])], images=False)
    run = _runner(provider).run_task(task)
    assert run.score.status == "skipped"
    assert run.score.earned == 0.0
    assert run.score.possible == 0.0
    assert "capability probe" in (run.score.skipped or "")
    assert provider.calls == []  # no task request was spent on the picture
    probe = run.transcript.extra["vision_probe"]
    assert probe["supported"] is False


def test_removing_the_image_url_makes_the_answer_unavailable(monkeypatch):
    task = _tasks()["V-3"]

    def answer_only_with_image(messages, tools=None):
        return str(task.oracle["answer"]) if _image_parts(messages) else "не знаю"

    provider = CallableProvider(answer_only_with_image, images=True)
    runner = _runner(provider)
    full = runner.run_task(task)
    assert full.score.earned == task.max_points

    from dailybench import runner as runner_module

    monkeypatch.setattr(runner_module, "user_content", lambda task, work_dir: task.prompt)
    text_only = _runner(CallableProvider(answer_only_with_image, images=True)).run_task(task)
    assert text_only.score.earned == 0.0


def test_swapping_the_picture_changes_the_answer():
    chart = _tasks()["V-1"]
    other = visual_input._chart(random.Random(100))
    assert other.oracle["answer"] != chart.oracle["answer"]
    # Same question, different picture: the old answer is now wrong.
    swapped = core.TaskInstance(
        id=chart.id,
        category=chart.category,
        title=chart.title,
        prompt=chart.prompt,
        max_points=chart.max_points,
        workspace=dict(chart.workspace or {}),
        assets=list(other.assets),
        oracle=dict(other.oracle),
    )
    run = _runner(ScriptedProvider([str(chart.oracle["answer"])], images=True)).run_task(swapped)
    assert run.score.earned == 0.0

    # The answer that belongs to the presented picture does score.
    right = _runner(ScriptedProvider([str(other.oracle["answer"])], images=True)).run_task(swapped)
    assert right.score.earned == swapped.max_points


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reply,earned",
    [
        ("APR", 1.0),
        ("  apr.\n", 1.0),
        ("Самый высокий столбик — APR.", 0.5),
        ("apr, май", 0.5),
        ("MAY", 0.0),
        ("не знаю", 0.0),
    ],
)
def test_exact_answer_and_answer_region_are_scored_separately(reply, earned):
    task = _tasks(7)["V-1"]
    transcript = core.Transcript(task_id=task.id, category="V")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=reply)))
    score = visual_input.evaluate(task, transcript)
    assert score.earned == earned


def test_the_category_stays_outside_the_scored_denominator():
    assert core.points_for("V", "research-v1") == 3.0
    assert "V" not in core.scored_categories("research-v1")
    assert "V" in core.SHOWCASE_CATEGORIES
    assert core.scored_total("research-v1") == 35.0 + 36.0
