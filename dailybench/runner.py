"""The runner: materialise a task, drive the model, score it.

One code path for single-turn categories and for the multi-turn tool loop, so
every model faces identical budgets: the same prompt, the same system text, the
same ``max_new_tokens``, the same total generated-token ceiling and the same
tool-call ceiling. The runner owns the sandbox and the transcript; the category
module owns the fixture and the scoring.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from . import core
from . import agent
from . import generation
from .config import RunConfig
from .measure import VramSampler
from .providers import Provider, ProviderError
from .rng import derived_seed

SYSTEM_SENTINEL = "<<system>>"

#: Bumped when the meaning of a checkpoint changes; a resume must not mix
#: records written by a different harness contract.
HARNESS_CONTRACT = 2

#: Config fields that belong to the identity of a run. ``only`` is a selection
#: over the same suite, ``out_dir`` is the location, and the key is local.
_IDENTITY_EXCLUDED = {"api_key", "out_dir", "only"}


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    return value


def config_identity(cfg: RunConfig) -> str:
    """Hash of everything that makes two runs incomparable when it differs.

    Agent mode is part of it on purpose: a ``agent=True`` answer is produced by
    a different harness than a chat answer, so continuing one with the other
    would present two regimes as one run.
    """
    payload = {
        item.name: _canonical(getattr(cfg, item.name))
        for item in fields(RunConfig)
        if item.name not in _IDENTITY_EXCLUDED
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def run_identity(cfg: RunConfig, suite_sha256: str) -> str:
    """Full checkpoint identity: config contract, suite digest, harness version."""
    blob = json.dumps(
        {
            "schema": HARNESS_CONTRACT,
            "config": config_identity(cfg),
            "suite": suite_sha256,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _prompt_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Rough size of the request in tokens, for the window-aware budget.

    Deliberately cheap and conservative: it decides how much room the answer
    gets, and the provider's own count is recorded afterwards. The last real
    prompt_tokens is used when the runtime reported one, so the estimate only
    matters for the first call.

    ``tools`` MUST be counted too: the runner sends their JSON schemas on every
    request, and on a tool scenario they outweigh the user prompt several times
    over. Without them the budget overshoots the window and the runtime rejects
    the request with HTTP 400.
    """
    from .providers import estimate_tokens

    total = 0
    for message in messages:
        content = message.get("content") or ""
        total += estimate_tokens(content if isinstance(content, str) else str(content))
        for call in message.get("tool_calls") or []:
            total += estimate_tokens(str(call.get("function", {}).get("arguments") or ""))
    if tools:
        total += estimate_tokens(json.dumps(tools, ensure_ascii=False))
    return total


def _request_tokens(provider: Provider, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    """Request size in tokens for the window-aware answer budget.

    Exact when the runtime can tokenize its own input: a character estimate on a
    700 KB document overshoots the window by tens of thousands of tokens and
    collapses the answer budget to its 256-token floor, which reads as an empty
    answer rather than a measured one. The estimate stays as the fallback and
    covers the chat template's framing (a few hundred tokens), while the 2048 +
    2 % reserve in ``budget_for`` covers the estimate error on the way back.
    """
    text = "\n".join(str(message.get("content") or "") for message in messages)
    if tools:
        text += "\n" + json.dumps(tools, ensure_ascii=False)
    estimate = _prompt_tokens(messages, tools)
    try:
        exact = provider.tokenize(text)
    except Exception:  # pragma: no cover - transport dependent
        exact = None
    return exact if exact is not None else estimate


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
        "spend": transcript.spend(),
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
        # A task built outside the suite (tests, the visual debug loop, a single
        # reused fixture) must still find its evaluator.
        from .tasks import load_all

        load_all()
        if task.category in self.cfg.unsupported:
            reason = (
                f"категория {task.category} объявлена неприменимой для этой конфигурации "
                "(конфигурация не поддерживает её — результат не проверялся)"
            )
            transcript = core.Transcript(task_id=task.id, category=task.category, extra={"skipped": reason})
            score = core.Score(task_id=task.id, category=task.category, skipped=reason, notes=[reason])
            # The budget is a property of the task even when nothing is scored;
            # ``Score.possible`` still reports 0 for a skip.
            score.max_points = task.max_points
            return TaskRun(task=task, transcript=transcript, score=score)

        agent_mode = self._uses_agent(task)
        work_dir = agent.prepare_work_dir(task) if agent_mode else materialise(task)
        before = snapshot(work_dir)
        driver_factory = None if agent_mode else core.TOOL_DRIVERS.get(task.category)
        driver = driver_factory(task, work_dir) if driver_factory else None

        transcript = core.Transcript(
            task_id=task.id,
            category=task.category,
            workspace_before=before,
            extra={"work_dir": str(work_dir), "harness": "agent" if agent_mode else "chat"},
        )
        if driver is not None:
            transcript.extra["driver"] = task.category

        if agent_mode:
            self._run_agent(task, transcript, work_dir)
            # The working copy is temporary: record what the agent produced while
            # it still exists, so a report rendered later shows the graded file.
            agent.record_artefact(task, transcript)
            transcript.workspace_after = snapshot(work_dir)
            transcript.extra["work_dir_size"] = len(transcript.workspace_after)
            score = self._score(task, transcript)
            return TaskRun(task=task, transcript=transcript, score=score)

        messages: list[dict[str, Any]] = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        started = time.perf_counter()
        # One absolute deadline for the whole task, not one per request: the
        # old per-request ceiling was multiplied by every tool turn and doubled
        # again by the ``stream_options`` retry. Each request below gets the
        # smaller of its own ceiling and what is left on this deadline.
        total_wall_clock = float(self.cfg.task_wall_clock or self.cfg.request_wall_clock or 1800)
        task_deadline = started + total_wall_clock
        sampler_before = self.sampler.peak_mib
        tool_calls = 0
        generated = 0
        # A task may need headroom above the shared cap (I-1 reads a dozen
        # files after listing the directory); the extra is explicit in the
        # fixture, so lowering the cap for a smoke run still applies.
        call_budget = self.cfg.max_tool_calls + int((task.oracle or {}).get("call_headroom") or 0)
        budget_spent = False
        followups = list(task.dialogue or [])
        try:
            while True:
                # One stop rule, checked before every new request: a spent
                # generation budget must never turn into another call just
                # because the previous reply happened to carry a tool call.
                if generated >= self.cfg.max_total_tokens:
                    transcript.extra.setdefault("notes", []).append(
                        f"бюджет генерации исчерпан ({self.cfg.max_total_tokens})"
                    )
                    break
                # A request started after the deadline would only be killed by
                # its own socket timeout, so the task stops here instead and is
                # reported unverified rather than scored on a partial answer.
                left = task_deadline - time.monotonic()
                if left <= 0:
                    note = f"предел времени на задание {total_wall_clock:g} с исчерпан"
                    transcript.extra.setdefault("notes", []).append(note)
                    transcript.error = note
                    transcript.extra["infra_error"] = True
                    break
                request_budget = min(float(self.cfg.request_wall_clock or left), left)
                reply = self.provider.chat(
                    messages,
                    # Once the call budget is spent, send no tools: a tool-less
                    # request releases the model's final answer instead of one
                    # more callable batch. Without this every budget-capped task
                    # ended on an empty tool_calls reply (I-1, T-2, T-4, T-6).
                    None if budget_spent else task.tools,
                    max_new_tokens=self.cfg.budget_for(task.category, _request_tokens(self.provider, messages, task.tools)),
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    seed=derived_seed(self.cfg.seed, task.id, str(len(transcript.turns))),
                    wall_clock=request_budget,
                )
                transcript.turns.append(core.Turn(messages=_copy_messages(messages), reply=reply))
                generated += reply.generated_tokens or 0

                if reply.tool_calls and driver is None:
                    # Nothing can execute the calls: stop on the reply as it is.
                    break
                if not reply.tool_calls:
                    if followups:
                        # Scripted dialogue: hand the next user message to the
                        # model and keep every earlier constraint in the history.
                        messages.append({"role": "assistant", "content": reply.text or ""})
                        messages.append({"role": "user", "content": followups.pop(0)})
                        continue
                    break
                room = call_budget - tool_calls
                if room <= 0:
                    if budget_spent:
                        # A tool-less request already went out and still came
                        # back with calls: one more would loop forever.
                        break
                    transcript.extra.setdefault("notes", []).append(
                        f"бюджет вызовов инструментов исчерпан ({call_budget})"
                    )
                    budget_spent = True
                    # Without tools the model may still *print* a call (the
                    # stand returns the request markup as content). Say plainly
                    # that the budget is gone, or the turn is spent re-asking.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Бюджет вызовов инструментов исчерпан. "
                                "Ответь на задание по уже собранным данным, без инструментов."
                            ),
                        }
                    )
                    continue
                calls = list(reply.tool_calls)
                if len(calls) > room:
                    # A batch that overshoots by one used to be dropped whole:
                    # the model had already planned those reads, and losing them
                    # cost I-1 the answer (11 reads of 11 files discarded).
                    # Execute what fits, then stop.
                    calls = calls[:room]
                    transcript.extra.setdefault("notes", []).append(
                        f"батч урезан бюджетом вызовов: {len(reply.tool_calls)} → {room} "
                        f"({call_budget} на задание)"
                    )
                reply.tool_calls = calls

                messages.append(_assistant_tool_message(reply))
                results: list[dict[str, Any]] = []
                for call in calls:
                    arguments, parse_error = _parse_arguments(call.get("arguments"))
                    if parse_error:
                        outcome = core.ToolOutcome(text=f"invalid tool arguments: {parse_error}", error=True)
                    else:
                        outcome = driver.execute(str(call.get("name", "")), arguments)
                    # The endpoint key reaches the model's own tools through the
                    # environment (``echo $DAILYBENCH_API_KEY``, ``cat
                    # $PI_CODING_AGENT_DIR/models.json``). The transcript is a
                    # run artefact, so it is scrubbed here before both the log
                    # and the next request carry the result.
                    result_text = agent.redact(outcome.text, self.cfg.api_key)
                    transcript.tool_log.append(
                        {
                            "turn": len(transcript.turns) - 1,
                            "name": str(call.get("name", "")),
                            "arguments": arguments,
                            "result": result_text,
                            "error": outcome.error,
                            "illegal": outcome.illegal,
                        }
                    )
                    results.append({"tool_call_id": call.get("id", ""), "text": result_text})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "name": call.get("name", ""),
                            "content": result_text,
                        }
                    )
                transcript.turns[-1].tool_results = results
                tool_calls += len(reply.tool_calls)
                if tool_calls >= call_budget:
                    transcript.extra.setdefault("notes", []).append(
                        f"бюджет вызовов инструментов исчерпан ({call_budget})"
                    )
                    budget_spent = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Бюджет вызовов инструментов исчерпан. "
                                "Ответь на задание по уже собранным данным, без инструментов."
                            ),
                        }
                    )

                if generated >= self.cfg.max_total_tokens:
                    transcript.extra.setdefault("notes", []).append(
                        f"бюджет генерации исчерпан ({self.cfg.max_total_tokens})"
                    )
                    break
        except ProviderError as exc:
            transcript.error = str(exc)
            if getattr(exc, "infra", True):
                transcript.extra["infra_error"] = True
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

    # -- the coding agent ---------------------------------------------- #

    def _uses_agent(self, task: core.TaskInstance) -> bool:
        return bool(self.cfg.agent) and task.category in tuple(self.cfg.agent_categories)

    def _agent_budget(self, task: core.TaskInstance) -> int:
        """Per-response ceiling handed to the agent.

        The artefact budgets (8192) are enough for a *chat* answer, but an agent
        spends them on thinking *and* a tool call in one response: a real C-2 run
        produced an 8012-token response out of 8192 and was cut off before it
        wrote anything. The agent therefore gets the room a window-aware chat run
        already gets: the same ``budget_for`` the chat path uses, floored at the
        category table so a tiny window cannot starve a whole-file answer.
        """
        table = int(self.cfg.max_new_tokens_by_category.get(task.category, self.cfg.max_new_tokens))
        if not self.cfg.unbounded_budget:
            return table
        return max(table, int(self.cfg.budget_for(task.category, 0)))

    def _run_agent(self, task: core.TaskInstance, transcript: core.Transcript, work_dir: Path) -> None:
        """Hand the task to ``pi`` in its working copy and record what happened.

        The agent owns its own loop, so the transcript carries the agent's own
        report: one synthetic turn with the final words, one row per executed
        tool call, and the usage the agent measured. The answer itself is read
        from disk afterwards (see :func:`agent.answer_text`) — a chat reply is a
        fallback, not the deliverable.
        """
        started = time.perf_counter()
        try:
            agent_dir = agent.config_dir(
                work_dir.parent,
                self.cfg,
                budget=self._agent_budget(task),
            )
            prompt = agent.prompt_for(task)
            run = agent.run(prompt, work_dir, self.cfg, agent_dir=agent_dir)
        except Exception as exc:  # pragma: no cover - environment dependent
            transcript.error = f"{type(exc).__name__}: {exc}"
            transcript.duration_s = time.perf_counter() - started
            return

        transcript.extra.update(run.as_extra(self.cfg.api_key))
        transcript.extra["agent"]["exit_ok"] = run.ok
        # The suite digest hashes the *scenario* prompt; the agent gets that plus
        # the grading contract, so the bytes actually sent are recorded here.
        transcript.extra["agent"]["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        transcript.extra["agent"]["prompt_chars"] = len(prompt)
        transcript.duration_s = run.duration_s
        transcript.tool_log.extend(run.tool_calls)
        reply = core.ModelReply(
            text=run.text,
            prompt_tokens=run.prompt_tokens_last,
            completion_tokens=run.answer_tokens,
            reasoning_tokens=run.reasoning_tokens,
            ttft_s=run.ttft_s,
            total_s=run.duration_s,
            finish_reason=(run.finish_reasons or [None])[-1],
        )
        transcript.turns.append(core.Turn(messages=[], reply=reply))
        transcript.extra["agent"]["turns_reported"] = run.turns
        if run.error:
            transcript.error = run.error

    def _score(self, task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
        module = core.REGISTRY.get(task.category)
        if module is None:
            return core.Score(
                task_id=task.id, category=task.category, error="category not registered", max_points=task.max_points
            )
        try:
            score = module.evaluate(task, transcript)
        except Exception as exc:
            # An evaluator crash must not remove the task from the
            # denominator: the model still failed to earn these points.
            score = core.Score(
                task_id=task.id,
                category=task.category,
                error=f"evaluator failed: {type(exc).__name__}: {exc}",
            )
            score.add("evaluator", False, task.max_points,
                      f"оценка не выполнена: {type(exc).__name__}: {exc}")
        # Single point where the harness owns the denominator: an evaluator
        # cannot widen ``possible`` by adding a penalty check, and a profile's
        # re-split budget is enforced here rather than in every module.
        score.max_points = task.max_points
        score.fit_points()
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
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def scores(self) -> list[core.Score]:
        return [run.score for run in self.runs]

    def save(self, directory: Path | None = None) -> Path:
        out = Path(directory or self.cfg.out_dir)
        (out / "transcripts").mkdir(parents=True, exist_ok=True)

        def dump(payload: Any) -> str:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=core.json_default)

        if not self.suite_digest:
            from .suite import build_suite, suite_digest

            self.suite_digest = suite_digest(build_suite(self.cfg), self.cfg.profile)
        identity = run_identity(self.cfg, str(self.suite_digest.get("sha256") or ""))

        config = json.loads(self.cfg.to_json())
        config["run_id"] = identity
        config["identity"] = identity
        config["environment"] = self.environment
        suite = dict(self.suite_digest)
        suite["identity"] = identity
        entries = []
        for run in self.runs:
            entry = run.as_dict()
            entry["identity"] = identity
            entries.append(entry)

        files: dict[str, str] = {
            "config.json": dump(config),
            "suite.json": dump(suite),
            "provider.json": dump(self.provider_info),
            "hardware.json": dump(self.hardware),
            "environment.json": dump(self.environment),
        }
        if self.speed:
            files["speed.json"] = dump(self.speed)
        files["scores.json"] = dump([score.as_dict() for score in self.scores])
        # tasks.json goes after scores.json: an interruption between them is
        # exactly the mixed checkpoint the generation manifest has to survive.
        files["tasks.json"] = dump(entries)
        for entry, run in zip(entries, self.runs):
            files[f"transcripts/{run.task.id.replace('/', '_')}.json"] = dump(entry)
        generation.commit(
            out,
            files,
            run_id=identity,
            identity=identity,
            environment=self.environment,
        )
        return out
