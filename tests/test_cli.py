"""CLI behaviour: every subcommand driven through ``dailybench.cli.main``.

These tests exercise the wiring a user actually types, against the stub
endpoint from :mod:`httpstub` — the only way to prove that ``--only``/``--limit``
/``--seed`` reach the provider, since those values are otherwise invisible in
the returned exit code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import httpstub
from dailybench import cli
from dailybench.config import RunConfig
from dailybench.rng import derived_seed
from dailybench.runner import Runner
from dailybench.suite import build_suite

SEED = 20260915


@pytest.fixture()
def endpoint():
    base_url, server = httpstub.start()
    yield base_url, server
    httpstub.stop(server)


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# doctor / list / generate
# --------------------------------------------------------------------------- #


def test_doctor_reports_a_reachable_endpoint(endpoint, capsys):
    base_url, _ = endpoint
    code = cli.main(["doctor", "--base-url", base_url, "--model", "test-model"])
    assert code == 0
    info = json.loads(capsys.readouterr().out)
    assert info["endpoint"]["reachable"] is True
    assert info["endpoint"]["model"] == "test-model"
    assert "test-model" in info["endpoint"]["models"]
    assert info["endpoint"]["tokenize_endpoint"] is True
    assert "python" in info and "gpus" in info


def test_doctor_unreachable_endpoint_exits_one(capsys):
    code = cli.main(["doctor", "--base-url", "http://127.0.0.1:9/v1", "--model", "test-model"])
    assert code == 1
    info = json.loads(capsys.readouterr().out)
    assert info["endpoint"]["reachable"] is False
    assert info["endpoint"]["error"]


def test_list_prints_one_row_per_task(capsys):
    assert cli.main(["list", "--only", "A"]) == 0
    out = capsys.readouterr().out
    for task_id in ("IF-1", "IF-2", "IF-3", "IF-4", "IF-5"):
        assert task_id in out
    assert "R-1" not in out, "фильтр --only не должен пускать чужие категории"
    assert "Заданий: 5" in out


def test_generate_writes_prompts_oracles_and_digest(tmp_path: Path):
    out_dir = tmp_path / "suite"
    assert cli.main(["generate", "--only", "A", "--out", str(out_dir)]) == 0
    prompts = _read(out_dir / "prompts.json")
    oracles = _read(out_dir / "oracles.json")
    digest = _read(out_dir / "suite.json")
    assert [entry["id"] for entry in prompts] == ["IF-1", "IF-2", "IF-3", "IF-4", "IF-5"]
    assert [entry["id"] for entry in oracles] == ["IF-1", "IF-2", "IF-3", "IF-4", "IF-5"]
    assert digest["tasks"] == 5
    assert "oracle" not in json.dumps(prompts), "публикуемый файл промптов не содержит ответов"
    assert all(entry.get("oracle") for entry in oracles)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def _run(base_url: str, out_dir: Path, *extra: str) -> int:
    return cli.main(
        [
            "run",
            "--base-url",
            base_url,
            "--model",
            "test-model",
            "--no-calibrate",
            "--out-dir",
            str(out_dir),
            *extra,
        ]
    )


def test_run_writes_the_artefacts_and_returns_zero(endpoint, tmp_path: Path):
    base_url, server = endpoint
    assert _run(base_url, tmp_path, "--only", "IF-2", "--limit", "1", "--name", "smoke") == 0
    out = tmp_path / "smoke"
    for name in ("config.json", "scores.json", "tasks.json", "suite.json", "report.md"):
        assert (out / name).is_file(), name
    assert "test-model" in (out / "report.md").read_text(encoding="utf-8")
    assert len(httpstub.chat_payloads(server)) == 1


def test_run_checkpoints_after_every_task(endpoint, tmp_path: Path, monkeypatch):
    """A crash at task 2 of 2 must not lose task 1.

    A real run is hours of GPU time; the process died once at task 92 of 96 under
    a debugger and lost everything because results were only written at the end.
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    real = Runner.run_task
    seen = {"n": 0}

    def explode(self, task):
        seen["n"] += 1
        if seen["n"] == 2:
            raise SystemExit(0)  # exactly how lldb's PyRun_SimpleString killed it
        return real(self, task)

    monkeypatch.setattr(Runner, "run_task", explode)
    with pytest.raises(SystemExit):
        _run(base_url, out, "--only", "IF-1,IF-2", "--name", "crash")
    saved = _read(out / "crash" / "scores.json")
    assert len(saved) == 1, "первая задача обязана лежать на диске до конца прогона"
    assert (out / "crash" / "tasks.json").is_file()


def test_resume_finishes_an_interrupted_run_without_replaying_it(endpoint, tmp_path: Path, monkeypatch):
    """A crashed run must be completable in place, and the kept tasks must not be re-asked.

    The reboot left 67 of 96 tasks on disk; re-running the whole suite spends GPU
    hours twice, while re-running *nothing* ships an incomplete artefact. The
    suite digest guard is what makes the merge legal at all.
    """
    base_url, server = endpoint
    out = tmp_path / "runs"
    real = Runner.run_task
    seen = {"n": 0}

    def explode(self, task):
        seen["n"] += 1
        if seen["n"] == 3:
            raise SystemExit(0)
        return real(self, task)

    monkeypatch.setattr(Runner, "run_task", explode)
    with pytest.raises(SystemExit):
        _run(base_url, out, "--only", "IF-1,IF-2,IF-3", "--name", "half")
    first = len(httpstub.chat_payloads(server))
    assert first == 2, "два задания обязаны быть записаны до обрыва"

    assert _run(base_url, out, "--only", "IF-1,IF-2,IF-3", "--name", "half", "--resume", "--no-html") == 0
    payloads = httpstub.chat_payloads(server)
    assert len(payloads) == 3, "догрузка обязана спросить ровно недостающее задание"
    assert [p["messages"][-1]["content"] for p in payloads[2:]] == [_expected_prompt("IF-3")]
    scores = _read(out / "half" / "scores.json")
    assert [row["task_id"] for row in scores] == ["IF-1", "IF-2", "IF-3"], "порядок сьюта обязан сохраниться"
    assert all(not row.get("error") for row in scores)


def _expected_prompt(task_id: str) -> str:
    cfg = RunConfig(seed=SEED, only=(task_id,))
    return next(task.prompt for task in build_suite(cfg) if task.id == task_id)


def test_resume_with_a_narrower_only_slice_compares_the_full_suite(endpoint, tmp_path: Path):
    """``--only IF-2`` must not look like a different suite than the saved run.

    The suite digest identifies the profile; the filter only picks tasks. A
    narrowed resume is exactly how a handful of interrupted tasks is replayed
    (see the collector flow), so a digest of the slice would refuse every one.
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "narrow") == 0

    assert _run(base_url, out, "--only", "IF-1", "--name", "narrow", "--resume") == 0
    saved = _read(out / "narrow" / "tasks.json")
    assert [entry["task"]["id"] for entry in saved] == ["IF-1", "IF-2"]


def test_resume_replays_a_task_whose_agent_died(endpoint, tmp_path: Path):
    """A truncated agent record is re-run, not shipped half-empty."""
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "cut") == 0
    saved = _read(out / "cut" / "tasks.json")
    assert len(saved) == 2
    saved[1]["transcript"]["error"] = "агент завершился с кодом 143"
    (out / "cut" / "tasks.json").write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    from dailybench.cli import _resume_tasks

    cfg = RunConfig(seed=SEED, only=("IF-1", "IF-2"), out_dir=out / "cut",
                    model="test-model", base_url=base_url, answer_cap=4096)
    missing, kept = _resume_tasks(cfg, build_suite(cfg))
    assert [task.id for task in missing] == ["IF-2"], "оборванное задание обязано уйти в повтор"
    assert [run.task.id for run in kept] == ["IF-1"]


def test_resume_replays_a_task_killed_mid_generation(endpoint, tmp_path: Path):
    """A last turn with no finish_reason is a kill, not an answer (T4).

    A transport drop can kill the turn after the agent's exit marker was
    recorded: no ``error``, ``exit_ok`` is True, only the unfinished reply
    betrays it. Resume must replay the task instead of keeping a half-answer
    with a near-zero latency record.
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "torn") == 0
    saved = _read(out / "torn" / "tasks.json")
    last_reply = saved[1]["transcript"]["turns"][-1]["reply"]
    last_reply["finish_reason"] = None
    (out / "torn" / "tasks.json").write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    from dailybench.cli import _resume_tasks

    cfg = RunConfig(seed=SEED, only=("IF-1", "IF-2"), out_dir=out / "torn",
                    model="test-model", base_url=base_url, answer_cap=4096)
    missing, kept = _resume_tasks(cfg, build_suite(cfg))
    assert [task.id for task in missing] == ["IF-2"], "убитая посреди генерации задача обязана уйти в повтор"
    assert [run.task.id for run in kept] == ["IF-1"]


def test_resume_rebuilds_from_transcripts_when_tasks_json_is_corrupt(endpoint, tmp_path: Path):
    """A reboot while tasks.json was being written must not cost the run.

    tasks.json is rewritten whole at every checkpoint; the per-task mirror under
    transcripts/ is what makes a truncated master recoverable.
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "torn") == 0
    run_dir = out / "torn"
    (run_dir / "tasks.json").write_text('{"broken": ', encoding="utf-8")

    from dailybench.cli import _resume_tasks
    cfg = RunConfig(seed=SEED, only=("IF-1", "IF-2"), out_dir=out / "torn",
                    model="test-model", base_url=base_url, answer_cap=4096)
    missing, kept = _resume_tasks(cfg, build_suite(cfg))
    assert [run.task.id for run in kept] == ["IF-1", "IF-2"], "записи обязаны подняться из transcripts/"
    assert missing == []


def test_resume_refuses_to_mix_two_models(endpoint, tmp_path: Path):
    """A saved run of model-a must not be continued by model-b.

    The merged directory would carry answers of two different models under one
    config, and the final report would silently attribute the new tasks to the
    old identity (or vice versa).
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1", "--name", "mix") == 0

    from dailybench.cli import _resume_tasks

    cfg = RunConfig(seed=SEED, only=("IF-1",), out_dir=out / "mix", model="model-b")
    with pytest.raises(SystemExit, match="resume смешал бы"):
        _resume_tasks(cfg, build_suite(cfg))

    same = RunConfig(seed=SEED, only=("IF-1",), out_dir=out / "mix",
                     model="test-model", base_url=base_url, answer_cap=4096)
    missing, kept = _resume_tasks(same, build_suite(same))
    assert missing == [] and [run.task.id for run in kept] == ["IF-1"]


def test_resume_ignores_a_mirror_the_master_no_longer_lists(endpoint, tmp_path: Path):
    """R09: mirrors recover a corrupt master, they do not extend a valid one.

    After C-1/C-2 a checkpoint containing only C-1 was written; the C-2 mirror
    stayed on disk and resume used to resurrect it next to the new master.
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "mirrors") == 0
    run_dir = out / "mirrors"

    from dailybench.core import Score
    from dailybench.runner import RunResult, TaskRun, transcript_from_dict

    saved = _read(run_dir / "tasks.json")
    cfg = RunConfig.from_dict(_read(run_dir / "config.json"))
    task = next(t for t in build_suite(cfg) if t.id == "IF-1")
    first = TaskRun(
        task=task,
        transcript=transcript_from_dict(saved[0]["transcript"], task),
        score=Score.from_dict(saved[0]["score"]),
    )
    RunResult(cfg=cfg, runs=[first]).save(run_dir)
    assert (run_dir / "transcripts" / "IF-2.json").is_file(), "старое зеркало обязано остаться на диске"

    from dailybench.cli import _resume_tasks

    missing, kept = _resume_tasks(cfg, build_suite(cfg))
    assert [run.task.id for run in kept] == ["IF-1"], "старое зеркало не должно воскреснуть"
    assert [item.id for item in missing] == ["IF-2"]


def test_resume_ignores_mirrors_without_a_master(endpoint, tmp_path: Path):
    """R09: without tasks.json there is no checkpoint to trust the mirrors for."""
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "orphan") == 0
    run_dir = out / "orphan"
    (run_dir / "tasks.json").unlink()
    assert list((run_dir / "transcripts").glob("*.json")), "зеркала на месте, но master отсутствует"

    from dailybench.cli import _resume_tasks

    cfg = RunConfig.from_dict(_read(run_dir / "config.json"))
    missing, kept = _resume_tasks(cfg, build_suite(cfg))
    assert kept == []
    assert [item.id for item in missing] == ["IF-1", "IF-2"]


def test_resume_rejects_an_agent_switch(endpoint, tmp_path: Path):
    """R09: agent=True and agent=False are different harness contracts."""
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1", "--name", "agent-switch") == 0

    from dailybench.cli import _resume_tasks

    cfg = RunConfig.from_dict(_read(out / "agent-switch" / "config.json"))
    cfg.agent = False
    with pytest.raises(SystemExit, match="resume смешал бы"):
        _resume_tasks(cfg, build_suite(cfg))


def test_run_exits_two_when_every_chat_request_fails_infrastructure(endpoint, tmp_path: Path, monkeypatch):
    """R14: a dead transport is not a wrong model answer."""
    from dailybench.providers import OpenAICompatibleProvider, ProviderError

    base_url, _server = endpoint

    def dead(self, *args, **kwargs):
        raise ProviderError("HTTP 500: upstream is down")

    monkeypatch.setattr(OpenAICompatibleProvider, "chat", dead)
    assert _run(base_url, tmp_path, "--only", "IF-1,IF-2", "--name", "infra", "--no-html") == 2


def test_run_keeps_the_normal_exit_code_on_a_partial_infrastructure_failure(
    endpoint, tmp_path: Path, monkeypatch
):
    """R14: one surviving answer means the run still describes the model."""
    from dailybench.providers import OpenAICompatibleProvider, ProviderError

    base_url, _server = endpoint
    real = OpenAICompatibleProvider.chat
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return real(self, *args, **kwargs)
        raise ProviderError("HTTP 500: upstream is down")

    monkeypatch.setattr(OpenAICompatibleProvider, "chat", flaky)
    assert _run(base_url, tmp_path, "--only", "IF-1,IF-2", "--name", "partial", "--no-html") == 0
    scores = _read(tmp_path / "partial" / "scores.json")
    assert any(row.get("error") for row in scores) and any(not row.get("error") for row in scores)


def test_resume_rejects_a_changed_weight(endpoint, tmp_path: Path, monkeypatch):
    """The same fixtures scored out of another denominator are another measurement.

    ``suite_digest`` used to hash prompts, dialogue, workspace and tools only, so
    a weight change left the identity intact and resume merged 2-point tasks
    with 5-point ones under one score file.
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "weights") == 0

    from dailybench import suite as suite_module
    from dailybench.cli import _resume_tasks

    real = suite_module.apply_profile_weights

    def heavier(tasks, profile):
        real(tasks, profile)
        for task in tasks:
            task.max_points *= 2

    monkeypatch.setattr(suite_module, "apply_profile_weights", heavier)
    cfg = RunConfig(seed=SEED, only=("IF-1", "IF-2"), out_dir=out / "weights",
                    model="test-model", base_url=base_url)
    with pytest.raises(SystemExit, match="разошёлся"):
        _resume_tasks(cfg, build_suite(cfg))


def test_resume_rejects_a_saved_denominator(tmp_path: Path):
    """A suite.json recording another possible_total cannot be resumed into."""
    from dailybench.cli import _resume_tasks, build_suite_digest

    cfg = RunConfig(seed=SEED, out_dir=tmp_path / "full")
    cfg.out_dir.mkdir()
    digest = build_suite_digest(cfg)
    digest["possible_total"] = float(digest.get("possible_total") or 0.0) + 1
    (cfg.out_dir / "suite.json").write_text(json.dumps(digest, ensure_ascii=False), encoding="utf-8")
    (cfg.out_dir / "tasks.json").write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match="знаменател"):
        _resume_tasks(cfg, build_suite(cfg))


def test_resume_rejects_a_suite_off_the_profile_total(tmp_path: Path, monkeypatch):
    """A generated suite must weigh exactly what its profile declares."""
    from dailybench import core
    from dailybench.cli import _resume_tasks, build_suite_digest

    cfg = RunConfig(seed=SEED, out_dir=tmp_path / "full")
    cfg.out_dir.mkdir()
    (cfg.out_dir / "suite.json").write_text(
        json.dumps(build_suite_digest(cfg), ensure_ascii=False), encoding="utf-8"
    )
    (cfg.out_dir / "tasks.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(core, "profile_total", lambda profile=core.DEFAULT_PROFILE: 999.0)

    with pytest.raises(SystemExit, match="знаменател"):
        _resume_tasks(cfg, build_suite(cfg))


def test_resume_without_speed_keeps_the_saved_profile(endpoint, tmp_path: Path):
    """`--resume` must not wipe speed.json just because the flag was omitted.

    The profile is measured once (minutes of GPU time); a resumed tail pass
    finishing the suite would otherwise blank the columns the report reads.
    """
    base_url, server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1", "--limit", "1", "--name", "sp") == 0
    rows = _read(out / "sp" / "speed.json") if (out / "sp" / "speed.json").is_file() else None
    assert rows is None, "без --speed профиля быть не должно"
    (out / "sp" / "speed.json").write_text(
        json.dumps([{"label": "short", "runs": 2, "ttft_s": 0.1, "generation_tok_s": 42.0}]),
        encoding="utf-8",
    )
    assert _run(base_url, out, "--only", "IF-1", "--name", "sp", "--resume", "--no-html") == 0
    kept = _read(out / "sp" / "speed.json")
    assert kept and kept[0]["generation_tok_s"] == 42.0, "сохранённый профиль обязан пережить resume"


def test_bench_forwards_the_profile_to_run():
    from dailybench.cli import _bench_run_argv, build_parser

    argv = _bench_run_argv(build_parser().parse_args(
        ["bench", "--base-url", "http://127.0.0.1:1/v1", "--model", "test-model",
         "--profile", "pilot"]
    ))
    assert argv[argv.index("--profile") + 1] == "pilot"


def test_bench_forwards_resume_to_run():
    """`bench --resume` must reach `run --resume`, or bench would start a second run over a live one."""
    from dailybench.cli import _bench_run_argv, build_parser

    args = build_parser().parse_args(
        [
            "bench",
            "--base-url", "http://127.0.0.1:1/v1",
            "--model", "test-model",
            "--config", "generic-gguf",
            "--resume",
        ]
    )
    assert "--resume" in _bench_run_argv(args)
    assert "--resume" not in _bench_run_argv(build_parser().parse_args(
        ["bench", "--base-url", "http://127.0.0.1:1/v1", "--model", "test-model"]
    ))


def test_run_maps_only_limit_and_seed_onto_the_config(endpoint, tmp_path: Path):
    """The parser's selection must reach the provider, not just the exit code."""
    base_url, server = endpoint
    assert (
        _run(
            base_url,
            tmp_path,
            "--only",
            "IF-1,IF-3",
            "--limit",
            "1",
            "--seed",
            str(SEED),
            "--name",
            "mapped",
        )
        == 0
    )
    payloads = httpstub.chat_payloads(server)
    assert len(payloads) == 1, "limit=1 должен отправить ровно один запрос"
    payload = payloads[0]
    assert payload["model"] == "test-model"
    assert payload["seed"] == derived_seed(SEED, "IF-1", "0"), "seed задания выводится из --seed"

    expected = next(t for t in build_suite(RunConfig(seed=SEED, only=("IF-1", "IF-3"))) if t.id == "IF-1")
    assert payload["messages"][-1]["content"] == expected.prompt, "--only выбрал IF-1, а не IF-2"


def test_run_writes_the_html_report(endpoint, tmp_path: Path):
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1", "--limit", "1", "--name", "with-html") == 0
    page = out / "with-html" / "report.html"
    assert page.is_file(), "run обязан собрать HTML-отчёт прогона"
    assert "Провенанс" in page.read_text(encoding="utf-8")
    assert (out / "with-html" / "_vendor" / "plotly.min.js").is_file()


def test_run_can_skip_the_html_report(endpoint, tmp_path: Path):
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1", "--limit", "1",
                "--name", "no-html", "--no-html") == 0
    assert not (out / "no-html" / "report.html").exists()
    assert (out / "no-html" / "report.md").is_file()


def test_run_maps_unsupported_onto_the_config(endpoint, tmp_path: Path):
    """A category declared unsupported is skipped before the model is called."""
    base_url, server = endpoint
    assert _run(base_url, tmp_path, "--only", "D", "--unsupported", "D", "--name", "skipped") == 0
    assert httpstub.chat_payloads(server) == [], "непроверяемая категория не должна вызывать модель"
    scores = _read(tmp_path / "skipped" / "scores.json")
    assert scores and all(row.get("skipped") for row in scores)


def test_run_persists_the_profile_and_hides_the_key(endpoint, tmp_path: Path, monkeypatch):
    base_url, _server = endpoint
    monkeypatch.setenv("DAILYBENCH_API_KEY", "sk-canary")
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1", "--limit", "1", "--name", "prof",
                "--profile", "stretch-v1") == 0
    config = _read(out / "prof" / "config.json")
    assert config["profile"] == "stretch-v1"
    assert "api_key" not in config
    assert config["api_key_set"] is True
    assert "sk-canary" not in (out / "prof" / "config.json").read_text(encoding="utf-8")
    suite = _read(out / "prof" / "suite.json")
    assert suite["profile"] == "stretch-v1"


def test_resume_rejects_another_profile(endpoint, tmp_path: Path):
    base_url, _ = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1", "--limit", "1", "--name", "same") == 0
    same = _run(base_url, out, "--only", "IF-1", "--limit", "1", "--name", "same", "--resume")
    assert same == 0
    with pytest.raises(SystemExit) as exc:
        _run(base_url, out, "--only", "IF-1", "--limit", "1", "--name", "same",
             "--profile", "stretch-v1", "--resume")
    assert "профил" in str(exc.value)


def test_run_unreachable_endpoint_returns_two(tmp_path: Path, capsys):
    code = cli.main(
        ["run", "--base-url", "http://127.0.0.1:9/v1", "--model", "test-model", "--out-dir", str(tmp_path)]
    )
    assert code == 2
    assert "недоступен" in capsys.readouterr().err


def test_run_missing_model_returns_two(endpoint, tmp_path: Path, capsys):
    base_url, _ = endpoint
    code = cli.main(
        ["run", "--base-url", base_url, "--model", "ghost-model", "--out-dir", str(tmp_path)]
    )
    assert code == 2
    assert "отсутствует" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# report / rescore
# --------------------------------------------------------------------------- #


def test_report_rewrites_report_md(endpoint, tmp_path: Path, capsys):
    base_url, _ = endpoint
    assert _run(base_url, tmp_path, "--only", "IF-2", "--limit", "1", "--name", "smoke") == 0
    out = tmp_path / "smoke"
    report = out / "report.md"
    report.write_text("stale", encoding="utf-8")

    assert cli.main(["report", str(out)]) == 0
    assert report.is_file() and report.read_text(encoding="utf-8") != "stale"
    assert str(report) in capsys.readouterr().out


def test_rescore_replays_a_saved_run_without_changes(endpoint, tmp_path: Path, capsys):
    base_url, _ = endpoint
    assert _run(base_url, tmp_path, "--only", "IF-2", "--limit", "1", "--name", "smoke") == 0
    out = tmp_path / "smoke"

    assert cli.main(["rescore", str(out)]) == 0
    marker = _read(out / "rescored.json")
    assert marker["tasks"] == 1
    assert marker["changed"] == [], "повторный расчёт по тем же ответам должен совпасть"
    assert "изменений 0" in capsys.readouterr().out


def test_rescore_legacy_rows_without_extra_hashes_still_replay(endpoint, tmp_path: Path, capsys):
    """An old suite.json recorded the prompt hash only.

    Absent dialogue/workspace/tools keys mean "not recorded then"; they must not
    make every task look drifted and turn the re-score into a no-op.
    """
    base_url, _ = endpoint
    assert _run(base_url, tmp_path, "--only", "IF-2", "--limit", "1", "--name", "smoke") == 0
    out = tmp_path / "smoke"
    suite = _read(out / "suite.json")
    suite["per_task"] = [
        {key: row[key] for key in ("id", "category", "prompt_sha256")} for row in suite["per_task"]
    ]
    (out / "suite.json").write_text(json.dumps(suite), encoding="utf-8")

    assert cli.main(["rescore", str(out)]) == 0
    marker = _read(out / "rescored.json")
    assert marker["changed"] == [], "отсутствие новых ключей — это не дрейф фикстуры"


def test_rescore_keeps_the_score_when_the_recorded_prompt_drifted(endpoint, tmp_path: Path, capsys):
    base_url, _ = endpoint
    assert _run(base_url, tmp_path, "--only", "IF-2", "--limit", "1", "--name", "smoke") == 0
    out = tmp_path / "smoke"
    suite = _read(out / "suite.json")
    # The digest covers the whole profile; damage the row of the saved task.
    row = next(item for item in suite["per_task"] if item["id"] == "IF-2")
    row["prompt_sha256"] = "0" * 64
    (out / "suite.json").write_text(json.dumps(suite), encoding="utf-8")

    assert cli.main(["rescore", str(out)]) == 0
    marker = _read(out / "rescored.json")
    assert marker["changed"] and "prompt_sha256" in marker["changed"][0]["status"]


def test_rescore_records_both_grading_contracts_and_recomputes_the_identity(
    endpoint, tmp_path: Path, monkeypatch
):
    """A re-score is a new grading contract, not the old run under a marker.

    Keeping the old identity hid that the evaluator moved; only a recollected
    pair of contracts lets a reader tell which numbers belong to which build.
    """
    base_url, _ = endpoint
    assert _run(base_url, tmp_path, "--only", "IF-2", "--limit", "1", "--name", "smoke") == 0
    out = tmp_path / "smoke"
    old_config = _read(out / "config.json")
    old_identity = old_config["identity"]
    old_suite = _read(out / "suite.json")

    from dailybench import suite as suite_module

    real = suite_module.evaluator_sha
    monkeypatch.setattr(suite_module, "evaluator_sha", lambda category: real(category)[::-1])

    assert cli.main(["rescore", str(out)]) == 0
    marker = _read(out / "rescored.json")
    seal = _read(out / "manifest.json")

    assert marker["source_identity"] == old_identity
    assert marker["identity"] != old_identity
    assert marker["source_generation"]
    assert marker["grading_contract"]["source"]["sha256"] == old_suite["sha256"]
    assert marker["grading_contract"]["current"]["sha256"] != old_suite["sha256"]
    source_eval = marker["grading_contract"]["source"]["evaluator"]["IF-2"]
    assert source_eval == old_suite["per_task"][0]["evaluator_sha256"]
    assert source_eval == real("A")
    assert marker["grading_contract"]["current"]["evaluator"]["A"] == real("A")[::-1]
    assert marker["grading_contract"]["current"]["evaluator"]["A"] != old_suite["per_task"][0]["evaluator_sha256"]
    assert seal["identity"] == marker["identity"]
    assert _read(out / "config.json")["identity"] == marker["identity"]


# --------------------------------------------------------------------------- #
# selfcheck / configs / speed / preflight
# --------------------------------------------------------------------------- #


def test_selfcheck_scores_only_the_sampled_tasks(capsys):
    # Only the first three instruction-following tasks: they need no toolchain,
    # so the exit code is a property of the harness and not of the machine.
    assert cli.main(["selfcheck", "--only", "A", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    for task_id in ("IF-1", "IF-2", "IF-3"):
        assert task_id in out
    assert "IF-4" not in out


def test_configs_lists_entries_and_writes_the_file(tmp_path: Path, capsys):
    assert cli.main(["configs"]) == 0
    out = capsys.readouterr().out
    assert "qwen3.8-27b-gsq-rco-iq3xxs" in out
    assert "Политика" in out

    target = tmp_path / "reference.json"
    assert cli.main(["configs", "--write", str(target)]) == 0
    payload = _read(target)
    assert payload["schema"] == "dailybench/reference/1"
    assert payload["models"] and payload["policy"]


def test_speed_prints_both_profiles(endpoint, capsys):
    base_url, server = endpoint
    code = cli.main(
        [
            "speed",
            "--base-url",
            base_url,
            "--model",
            "test-model",
            "--speed-long",
            "64",
            "--speed-repeats",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert '"label": "short"' in out
    assert '"label": "long_64"' in out
    # Warm-up (1) + repeats (2) per profile, times two profiles.
    assert len(httpstub.chat_payloads(server)) == 6


def test_preflight_command_uses_a_written_reference_file(endpoint, tmp_path: Path, capsys):
    base_url, _ = endpoint
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "schema": "dailybench/reference/1",
                "policy": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "context_window": 32768,
                    "max_total_tokens": 16384,
                    "max_tool_calls": 12,
                    "answer_budget_default": 4096,
                    "answer_budget_artifact": 8192,
                    "request_wall_clock": 600,
                    "reasoning_budget": 2048,
                },
                "models": [
                    {
                        "id": "stub-entry",
                        "label": "Stub entry",
                        "runtime_id": "llama.cpp",
                        "base_url": base_url,
                        "model": "test-model",
                        "quant": "test",
                        "context_window": 32768,
                        "kv_type": "f16",
                        "capabilities": {"tools": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "preflight.md"
    code = cli.main(
        ["preflight", "--config", "stub-entry", "--configs-path", str(reference_path), "--report", str(report_path)]
    )
    assert code == 0
    assert report_path.is_file()
    assert "stub-entry" in report_path.read_text(encoding="utf-8")


def _patched_list(monkeypatch, capsys, *extra: str) -> str:
    tasks = build_suite(RunConfig(only=("R-1", "K-1")))
    tasks[0].private = True
    tasks[0].prompt = "ПРИВАТНЫЙ ПРОМПТ"
    monkeypatch.setattr(cli, "build_suite", lambda _cfg: tasks)
    assert cli.main(["list", "--prompts", *extra]) == 0
    return capsys.readouterr().out


def test_list_prompts_hides_private_text_without_the_flag(monkeypatch, capsys):
    out = _patched_list(monkeypatch, capsys)
    assert "ПРИВАТНЫЙ ПРОМПТ" not in out
    assert "Приватных заданий скрыто: 1" in out
    assert "приватное" not in out


def test_list_prompts_shows_and_marks_private_with_the_flag(monkeypatch, capsys):
    out = _patched_list(monkeypatch, capsys, "--include-private")
    assert "ПРИВАТНЫЙ ПРОМПТ" in out
    assert "· приватное" in out
    assert "скрыто" not in out
