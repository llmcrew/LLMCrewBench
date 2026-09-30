"""The runner: materialise a task, drive the model, score it.

One code path for single-turn categories and for the multi-turn tool loop, so
every model faces identical budgets: the same prompt, the same system text, the
same ``max_new_tokens``, the same total generated-token ceiling and the same
tool-call ceiling. The runner owns the sandbox and the transcript; the category
module owns the fixture and the scoring.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import core
from .config import RunConfig
from .measure import VramSampler
from .providers import Provider, ProviderError
from .rng import derived_seed

SYSTEM_SENTINEL = "<<system>>"


@dataclass
class TaskRun:
    task: core.TaskInstance
    transcript: core.Transcript
    score: core.Score

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": {
                "id": self.task.id,
                "category": self.task.category,
                "title": self.task.title,
                "max_points": self.task.max_points,
                "meta": self.task.meta,
            },
            "transcript": transcript_to_dict(self.transcript),
            "score": self.score.as_dict(),
        }


def transcript_to_dict(transcript: core.Transcript) -> dict[str, Any]:
    return {
        "task_id": transcript.task_id,
        "category": transcript.category,
        "error": transcript.error,
        "duration_s": transcript.duration_s,
        "prompt_tokens": transcript.prompt_tokens(),
        "generated_tokens": transcript.generated_tokens(),
        "tool_log": transcript.tool_log,
        "workspace_before": transcript.workspace_before,
        "workspace_after": transcript.workspace_after,
        "extra": {k: v for k, v in transcript.extra.items() if k != "driver"},
        "turns": [
            {
                "messages": turn.messages,
                "reply": asdict(turn.reply),
                "tool_results": turn.tool_results,
            }
            for turn in transcript.turns
        ],
    }


def transcript_from_dict(data: dict[str, Any], task: core.TaskInstance) -> core.Transcript:
    """Rebuild a transcript from a saved run, for re-scoring without the model."""
    transcript = core.Transcript(
        task_id=data.get("task_id", task.id),
        category=data.get("category", task.category),
        error=data.get("error"),
        duration_s=data.get("duration_s"),
        tool_log=list(data.get("tool_log") or []),
        workspace_before=dict(data.get("workspace_before") or {}),
        workspace_after=dict(data.get("workspace_after") or {}),
        extra=dict(data.get("extra") or {}),
    )
    for turn in data.get("turns") or []:
        reply_data = dict(turn.get("reply") or {})
        reply = core.ModelReply(
            **{k: v for k, v in reply_data.items() if k in core.ModelReply.__dataclass_fields__}
        )
        transcript.turns.append(
            core.Turn(
                messages=list(turn.get("messages") or []),
                reply=reply,
                tool_results=list(turn.get("tool_results") or []),
            )
        )
    return transcript


def materialise(task: core.TaskInstance) -> Path:
    """Write ``task.workspace`` into a fresh temp directory."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"dailybench-{task.id.lower()}-"))
    for rel, content in (task.workspace or {}).items():
        target = work_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return work_dir


def snapshot(root: Path) -> dict[str, str]:
    """Relative path -> content for every file under ``root``."""
    result: dict[str, str] = {}
    if not root.is_dir():
        return result
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return result


class Runner:
    """Runs tasks against one provider and scores them with their category."""

    def __init__(self, cfg: RunConfig, provider: Provider, sampler: VramSampler | None = None) -> None:
        self.cfg = cfg
        self.provider = provider
        self.sampler = sampler or VramSampler()

    # -- one task ---------------------------------------------------------- #

    def run_task(self, task: core.TaskInstance) -> TaskRun:
        if task.category in self.cfg.unsupported:
            reason = (
                f"категория {task.category} объявлена неприменимой для этой конфигурации "
                "(конфигурация не поддерживает её — результат не проверялся)"
            )
            transcript = core.Transcript(task_id=task.id, category=task.category, extra={"skipped": reason})
            return TaskRun(
                task=task,
                transcript=transcript,
                score=core.Score(task_id=task.id, category=task.category, skipped=reason, notes=[reason]),
            )

        work_dir = materialise(task)
        before = snapshot(work_dir)
        driver_factory = core.TOOL_DRIVERS.get(task.category)
        driver = driver_factory(task, work_dir) if driver_factory else None

        transcript = core.Transcript(
            task_id=task.id,
            category=task.category,
            workspace_before=before,
            extra={"work_dir": str(work_dir)},
        )
        if driver is not None:
            transcript.extra["driver"] = task.category

        messages: list[dict[str, Any]] = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": task.prompt})

        started = time.perf_counter()
        sampler_before = self.sampler.peak_mib
        tool_calls = 0
        generated = 0
        try:
            while True:
                reply = self.provider.chat(
                    messages,
                    task.tools,
                    max_new_tokens=self.cfg.budget_for(task.category),
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    seed=derived_seed(self.cfg.seed, task.id, str(len(transcript.turns))),
                    wall_clock=self.cfg.request_wall_clock or None,
                )
                transcript.turns.append(core.Turn(messages=_copy_messages(messages), reply=reply))
                generated += reply.generated_tokens or 0

                if driver is None or not reply.tool_calls:
                    break
                if tool_calls + len(reply.tool_calls) > self.cfg.max_tool_calls:
                    transcript.extra.setdefault("notes", []).append(
                        f"бюджет вызовов инструментов исчерпан ({self.cfg.max_tool_calls})"
                    )
                    break

                messages.append(_assistant_tool_message(reply))
                results: list[dict[str, Any]] = []
                for call in reply.tool_calls:
                    arguments, parse_error = _parse_arguments(call.get("arguments"))
                    if parse_error:
                        outcome = core.ToolOutcome(text=f"invalid tool arguments: {parse_error}", error=True)
                    else:
                        outcome = driver.execute(str(call.get("name", "")), arguments)
                    transcript.tool_log.append(
                        {
                            "turn": len(transcript.turns) - 1,
                            "name": str(call.get("name", "")),
                            "arguments": arguments,
                            "result": outcome.text,
                            "error": outcome.error,
                            "illegal": outcome.illegal,
                        }
                    )
                    results.append({"tool_call_id": call.get("id", ""), "text": outcome.text})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "name": call.get("name", ""),
                            "content": outcome.text,
                        }
                    )
                transcript.turns[-1].tool_results = results
                tool_calls += len(reply.tool_calls)

                if generated > self.cfg.max_total_tokens:
                    transcript.extra.setdefault("notes", []).append(
                        f"бюджет генерации исчерпан ({self.cfg.max_total_tokens} токенов)"
                    )
                    break
        except ProviderError as exc:
            transcript.error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            transcript.error = f"{type(exc).__name__}: {exc}"

        transcript.duration_s = time.perf_counter() - started
        transcript.workspace_after = snapshot(work_dir)
        transcript.extra["work_dir_size"] = len(transcript.workspace_after)
        if self.sampler.peak_mib is not None:
            transcript.extra["vram_peak_mib"] = self.sampler.peak_mib
            if sampler_before is not None:
                transcript.extra["vram_before_task_mib"] = sampler_before

        score = self._score(task, transcript)
        return TaskRun(task=task, transcript=transcript, score=score)

    def _score(self, task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
        module = core.REGISTRY.get(task.category)
        if module is None:
            return core.Score(task_id=task.id, category=task.category, error="category not registered")
        try:
            score = module.evaluate(task, transcript)
        except Exception as exc:  # pragma: no cover - defensive
            return core.Score(
                task_id=task.id,
                category=task.category,
                error=f"evaluator failed: {type(exc).__name__}: {exc}",
            )
        if transcript.error:
            score.error = transcript.error
            score.notes.append(f"прогон прерван: {transcript.error}")
        return score

    # -- many tasks -------------------------------------------------------- #

    def run_suite(self, tasks: list[core.TaskInstance], on_result=None) -> list[TaskRun]:
        runs: list[TaskRun] = []
        for task in tasks:
            run = self.run_task(task)
            runs.append(run)
            if on_result is not None:
                on_result(run)
        return runs


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if "tool_calls" in item:
            item["tool_calls"] = json.loads(json.dumps(item["tool_calls"]))
        copied.append(item)
    return copied


def _assistant_tool_message(reply: core.ModelReply) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": reply.text or None,
        "tool_calls": [
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments")
                    if isinstance(call.get("arguments"), str)
                    else json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for index, call in enumerate(reply.tool_calls)
        ],
    }


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str]:
    if isinstance(raw, dict):
        return raw, ""
    text = (raw or "").strip()
    if not text:
        return {}, ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    if not isinstance(parsed, dict):
        return {}, f"arguments must be a JSON object, got {type(parsed).__name__}"
    return parsed, ""


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    cfg: RunConfig
    runs: list[TaskRun] = field(default_factory=list)
    provider_info: dict[str, Any] = field(default_factory=dict)
    suite_digest: dict[str, Any] = field(default_factory=dict)
    speed: list[dict[str, Any]] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)

    @property
    def scores(self) -> list[core.Score]:
        return [run.score for run in self.runs]

    def save(self, directory: Path | None = None) -> Path:
        out = Path(directory or self.cfg.out_dir)
        (out / "transcripts").mkdir(parents=True, exist_ok=True)

        def dump(payload: Any) -> str:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=core.json_default)

        (out / "config.json").write_text(self.cfg.to_json(), encoding="utf-8")
        (out / "suite.json").write_text(dump(self.suite_digest), encoding="utf-8")
        (out / "provider.json").write_text(dump(self.provider_info), encoding="utf-8")
        (out / "hardware.json").write_text(dump(self.hardware), encoding="utf-8")
        if self.speed:
            (out / "speed.json").write_text(dump(self.speed), encoding="utf-8")
        (out / "scores.json").write_text(dump([score.as_dict() for score in self.scores]), encoding="utf-8")
        (out / "tasks.json").write_text(dump([run.as_dict() for run in self.runs]), encoding="utf-8")
        for run in self.runs:
            path = out / "transcripts" / f"{run.task.id.replace('/', '_')}.json"
            path.write_text(dump(run.as_dict()), encoding="utf-8")
        return out
