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

    cfg = RunConfig(seed=SEED, only=("IF-1", "IF-2"), out_dir=out / "cut")
    missing, kept = _resume_tasks(cfg, build_suite(cfg))
    assert [task.id for task in missing] == ["IF-2"], "оборванное задание обязано уйти в повтор"
    assert [run.task.id for run in kept] == ["IF-1"]


def test_resume_rebuilds_from_transcripts_when_tasks_json_is_corrupt(endpoint, tmp_path: Path):
    """A reboot while tasks.json was being written must not cost the run.

    tasks.json is rewritten whole at every checkpoint; the per-task mirror under
    transcripts/ is what makes a truncated master recoverable.
    """
    base_url, _server = endpoint
    out = tmp_path / "runs"
    assert _run(base_url, out, "--only", "IF-1,IF-2", "--name", "torn") == 0
    (out / "torn" / "tasks.json").write_text('{"broken": ', encoding="utf-8")

    from dailybench.cli import _resume_tasks

    cfg = RunConfig(seed=SEED, only=("IF-1", "IF-2"), out_dir=out / "torn")
    missing, kept = _resume_tasks(cfg, build_suite(cfg))
    assert [run.task.id for run in kept] == ["IF-1", "IF-2"], "записи обязаны подняться из transcripts/"
    assert missing == []


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
