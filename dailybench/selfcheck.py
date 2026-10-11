"""Self-check: prove that generators, oracles and evaluators agree — no GPU.

Two synthetic providers are run through the identical runner:

* :class:`OracleProvider` replays the canonical answer (including the golden
  tool sequence for category D and the reference file for category C). It must
  score the full points of every task; anything less is a broken evaluator.
* :class:`GarbageProvider` returns "не знаю" and must score zero everywhere.

The suite is also regenerated to prove determinism and re-run with another seed
to prove the fixtures actually move.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import core
from .config import RunConfig
from .providers import Provider
from .runner import RunResult, Runner
from .suite import build_suite, suite_digest, validate_suite


class OracleProvider:
    """Replays the canonical answer for one task: reference or golden calls."""

    name = "oracle"

    def __init__(self, task: core.TaskInstance) -> None:
        self.task = task
        self.pending = [dict(call) for call in task.oracle.get("golden_calls", [])]
        self.answers = [str(text) for text in (task.oracle.get("turn_answers") or [])]
        self.last = ""
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs: Any) -> core.ModelReply:
        self.calls += 1
        if self.pending:
            call = self.pending.pop(0)
            return core.ModelReply(
                text="",
                tool_calls=[
                    {
                        "id": f"oracle_{self.calls}",
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                    }
                ],
                prompt_tokens=None,
                completion_tokens=0,
            )
        if self.answers:
            self.last = self.answers.pop(0)
        text = self.last or self.task.oracle.get("final_answer") or self.task.oracle.get("reference", "")
        return core.ModelReply(text=str(text), prompt_tokens=None, completion_tokens=0)

    def tokenize(self, text: str) -> int | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "task": self.task.id}


@dataclass
class GarbageProvider:
    """Always answers with an explicit non-answer."""

    text: str = "не знаю"
    name: str = "garbage"
    calls: int = 0

    def chat(self, messages, tools=None, **kwargs: Any) -> core.ModelReply:
        self.calls += 1
        return core.ModelReply(text=self.text, prompt_tokens=None, completion_tokens=0)

    def tokenize(self, text: str) -> int | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name}


@dataclass
class SelfCheckReport:
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [
            "| Задание | Балл эталона | Балл мусора | Статус |",
            "| --- | ---: | ---: | --- |",
        ]
        for row in self.rows:
            if row.get("skipped"):
                lines.append(f"| {row['id']} | н/п | н/п | не проверялось |")
                continue
            status = "ok" if row["oracle"] == row["max"] and row["garbage"] == 0 else "ПРОБЛЕМА"
            lines.append(
                f"| {row['id']} | {row['oracle']:.0f}/{row['max']:.0f} | {row['garbage']:.0f} | {status} |"
            )
        if self.warnings:
            lines.append("")
            lines.append("Предупреждения:")
            lines.extend(f"- {w}" for w in self.warnings)
        if self.errors:
            lines.append("")
            lines.append("Ошибки:")
            lines.extend(f"- {e}" for e in self.errors)
        return "\n".join(lines)


def selfcheck(cfg: RunConfig, tasks: list[core.TaskInstance] | None = None) -> SelfCheckReport:
    report = SelfCheckReport()
    # Self-check drives the runner with Python fakes, not a live endpoint, so the
    # agent cannot be reached from here: it needs a real base_url to point pi at.
    # Agents own only their categories, so everything else is proven by exactly
    # the same code path a run uses, and C/K/L are proven end to end by the
    # ``agent`` tests instead.
    cfg = replace(cfg, agent=False)
    tasks = tasks if tasks is not None else build_suite(cfg)
    report.warnings.extend(validate_suite(tasks))

    total = sum(task.max_points for task in tasks)
    scored = sum(t.max_points for t in tasks if t.category in core.SCORED_CATEGORIES)
    if tasks and abs(scored - core.FULL_TOTAL) > 1e-9 and not cfg.only:
        report.errors.append(f"сумма баллов сьюта {scored} != {core.FULL_TOTAL}")

    digest_a = suite_digest(build_suite(cfg))
    digest_b = suite_digest(build_suite(cfg))
    if digest_a["sha256"] != digest_b["sha256"]:
        report.errors.append("сьют не детерминирован: два прогона с одним seed дают разные fixtures")

    other_seed = RunConfig.from_dict({**cfg.__dict__, "seed": cfg.seed + 1})
    digest_other = suite_digest(build_suite(other_seed))
    if digest_other["sha256"] == digest_a["sha256"] and any(task.meta.get("seed_sensitive", True) for task in tasks):
        report.errors.append("смена seed не меняет fixtures")

    for task in tasks:
        oracle_run = Runner(cfg, OracleProvider(task)).run_task(task)
        garbage_run = Runner(cfg, GarbageProvider()).run_task(task)
        row = {
            "id": task.id,
            "category": task.category,
            "max": task.max_points,
            "oracle": oracle_run.score.earned,
            "garbage": garbage_run.score.earned,
            "oracle_checks": [(c.name, c.passed, c.detail) for c in oracle_run.score.checks],
            "garbage_checks": [(c.name, c.passed, c.detail) for c in garbage_run.score.checks],
        }
        report.rows.append(row)

        if oracle_run.transcript.error:
            report.errors.append(f"{task.id}: прогон эталона упал: {oracle_run.transcript.error}")
            continue
        if oracle_run.score.skipped:
            # Not verified is not a failed evaluator: an unavailable toolchain
            # or a declared-unsupported capability leaves the row unscored, and
            # the self-check reports it as a warning rather than an error.
            row["skipped"] = oracle_run.score.skipped
            report.warnings.append(f"{task.id}: не проверялось — {oracle_run.score.skipped}")
            continue
        expected_checks = 0
        if abs(oracle_run.score.possible - task.max_points) > 1e-9:
            report.errors.append(
                f"{task.id}: возможный балл {oracle_run.score.possible} != max_points {task.max_points}"
            )
            expected_checks += 1
        if oracle_run.score.earned != task.max_points:
            failed = [c.name for c in oracle_run.score.checks if not c.passed]
            report.errors.append(
                f"{task.id}: эталон набрал {oracle_run.score.earned} из {task.max_points}; провалены {failed}"
            )
        if garbage_run.score.earned != 0:
            passed = [c.name for c in garbage_run.score.checks if c.passed]
            report.errors.append(f"{task.id}: мусорный ответ набрал {garbage_run.score.earned} ({passed})")
    return report


def write_selfcheck_report(path: Path | str, report: SelfCheckReport) -> Path:
    target = Path(path)
    target.write_text(report.render() + "\n", encoding="utf-8")
    return target


def save_suite(cfg: RunConfig, out_dir: Path | str) -> dict[str, Path]:
    """Write the generated suite: public prompts, grading material, digest."""
    from .suite import grading_material, public_prompts

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_suite(cfg)
    written = {
        "prompts": out / "prompts.json",
        "oracles": out / "oracles.json",
        "digest": out / "suite.json",
    }
    written["prompts"].write_text(
        json.dumps(public_prompts(tasks), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["oracles"].write_text(
        json.dumps(grading_material(tasks), ensure_ascii=False, indent=2, default=core.json_default),
        encoding="utf-8",
    )
    written["digest"].write_text(
        json.dumps(suite_digest(tasks), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = RunConfig()
    for index, item in enumerate(argv):
        if item == "--seed" and index + 1 < len(argv):
            cfg.seed = int(argv[index + 1])
        if item == "--only" and index + 1 < len(argv):
            cfg.only = tuple(argv[index + 1].split(","))
    report = selfcheck(cfg)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
