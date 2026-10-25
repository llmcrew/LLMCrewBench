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
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from . import core
from . import agent
from . import generation
from . import isolation
from .config import RunConfig
from .measure import VramSampler
from .providers import Provider, ProviderError
from .rng import derived_seed

SYSTEM_SENTINEL = "<<system>>"

#: Bumped when the meaning of a checkpoint changes; a resume must not mix
#: records written by a different harness contract. Contract 4: the agentic
#: categories execute in a container (host execution is a marked fallback), so a
#: boundary change is a different measurement. Contract 3 covered the executed
#: evaluator bundle, closed oracle, toolchain and agent binary, and the agent
#: harness pins its release.
HARNESS_CONTRACT = 4

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
    """Full checkpoint identity: config contract, suite digest, harness version.

    When the run drives the coding agent, the agent binary's version and digest
    join the identity: two checkpoints produced by different ``pi`` releases are
    not one observation. ``agent.identity()`` never installs and never raises, so
    an absent binary is recorded as ``found: null`` instead of blocking resume.
    """
    blob = json.dumps(
        {
            "schema": HARNESS_CONTRACT,
            "config": config_identity(cfg),
            "suite": suite_sha256,
            "agent": agent.identity() if cfg.agent else None,
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


def _request_size(
    provider: Provider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cfg: RunConfig,
    task_id: str,
    meta_ratio: float | None = None,
) -> tuple[int, str]:
    """``(tokens, source)`` — how much room the request leaves, and on what basis.

    Exact when the runtime can tokenize its own input: a character estimate on a
    700 KB document overshoots the window by tens of thousands of tokens and
    collapses the answer budget to its 256-token floor, which reads as an empty
    answer rather than a measured one.

    Without ``/tokenize`` the *saved per-document coefficient* is used before
    the generic 2.5 divider. The calibration that produced it ran against the
    real tokenizer, so it is the measured answer to the same question; the
    divider is a last resort that once turned a 31K prompt into 37K estimated
    tokens and left every LC request at the floor. The source travels with the
    number: an approximated budget must not look measured.
    """
    text = "\n".join(str(message.get("content") or "") for message in messages)
    if tools:
        text += "\n" + json.dumps(tools, ensure_ascii=False)
    try:
        exact = provider.tokenize(text)
    except Exception:  # pragma: no cover - transport dependent
        exact = None
    if exact is not None:
        return int(exact), "tokenizer"
    ratio = float(meta_ratio or cfg.task_ratios.get(task_id) or 0)
    if ratio > 0:
        return max(1, round(len(text) / ratio)), "task-coefficient"
    return _prompt_tokens(messages, tools), "estimate"


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


def materialise(task: core.TaskInstance, parent: Path | None = None) -> Path:
    """Write ``task.workspace`` into a fresh temp directory."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"dailybench-{task.id.lower()}-", dir=parent))
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

    def __init__(
        self,
        cfg: RunConfig,
        provider: Provider,
        sampler: VramSampler | None = None,
        environment: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.provider = provider
        self.sampler = sampler or VramSampler()
        self.environment = environment if environment is not None else {}
        self._template: dict[str, Any] | None = None
        self._template_probed = False
        self._sandbox_state: tuple[bool, str] | None = None

    def _sandbox_usable(self) -> tuple[bool, str]:
        """Whether a container can run at all, checked once per run.

        A failed probe is a property of the host, not of the task, so it is
        cached: a Docker-less machine must not pay a daemon timeout per task.
        """
        if self._sandbox_state is None:
            self._sandbox_state = isolation.ready()
        return self._sandbox_state

    def _sandbox(self, work_dir: Path) -> isolation.Sandbox:
        """The boundary for one agent task, marked as host when it cannot run.

        The object is returned either way: an unusable sandbox carries the
        reason, and the transcript then says ``host_run`` with it instead of
        silently inheriting the machine.
        """
        usable, reason = self._sandbox_usable()
        box = isolation.Sandbox(
            work_dir=work_dir,
            image=isolation.default_image(),
            runtime=isolation.default_runtime(),
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            deadline=float(self.cfg.task_wall_clock or self.cfg.request_wall_clock or 1800),
            ready_state=(usable, reason),
        )
        if not usable:
            box.fallback(reason or "песочница недоступна")
        return box

    def _note_container_run(self) -> None:
        """The run really executed in a container: say so in the environment.

        Written into the shared environment record of the ``RunResult``, which
        the manifest and the report read. A single container is enough: the
        claim is about the executor actually used.
        """
        if self.environment and self.environment.get("executor_status") != "container_run":
            self.environment["executor_status"] = "container_run"
            self.environment["executor_reason"] = ""

    # -- the generation contract ------------------------------------------- #

    def _template_identity(self) -> tuple[str, str | None]:
        """Declared template version plus the digest the server reports.

        The chat template is half of the prompt, so a run without it pinned is
        not comparable to the next one. Only llama.cpp exposes it (``/props``);
        an endpoint without it stays ``None`` and the gap is visible.
        """
        if not self._template_probed:
            self._template_probed = True
            probe = getattr(self.provider, "props", None)
            self._template = probe() if callable(probe) else None
        template = (self._template or {}).get("chat_template")
        digest = (
            hashlib.sha256(template.encode("utf-8")).hexdigest()
            if isinstance(template, str) and template
            else None
        )
        return self.cfg.template_version, digest

    def _wire_contract(self, output_limit: int, seed: int) -> WireContract:
        """The request contract for one answer, from the selected mode.

        ``native`` deliberately carries no sampling and no reasoning budget: the
        server decides. ``fixed`` carries all of it, and the request verifies it
        on the final payload — the two are different measurements and the
        transcript says which one ran.
        """
        from .providers import WireContract

        version, digest = self._template_identity()
        fixed = self.cfg.generation_mode == "fixed"
        return WireContract(
            mode=self.cfg.generation_mode,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            seed=seed,
            output_limit=output_limit,
            reasoning_budget=int(self.cfg.reasoning_budget) if fixed else None,
            stop=tuple(self.cfg.stop_sequences),
            template_version=version,
            template_sha256=digest,
        )

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
        # One root per task, removed in the ``finally`` below: the working copy,
        # the agent's config and its logs live inside it. A shared parent let two
        # tasks collide on ``pi-agent/models.json`` and left every copy behind for
        # the next model to find.
        task_root = Path(tempfile.mkdtemp(prefix=f"dailybench-{task.id.lower()}-"))
        work_dir = (
            agent.prepare_work_dir(task, task_root) if agent_mode else materialise(task, task_root)
        )
        before = snapshot(work_dir)
        driver_factory = None if agent_mode else core.TOOL_DRIVERS.get(task.category)
        driver = driver_factory(task, work_dir) if driver_factory else None

        transcript = core.Transcript(
            task_id=task.id,
            category=task.category,
            workspace_before=before,
            extra={
                "work_dir": str(work_dir),
                "harness": "agent" if agent_mode else "chat",
            },
        )
        sandbox = self._sandbox(work_dir) if agent_mode and self.cfg.sandbox else None
        if agent_mode:
            transcript.extra["sandbox"] = (
                sandbox.record()
                if sandbox is not None
                else {
                    "executor_status": "host_run",
                    "executor_reason": "sandbox выключен в конфигурации (sandbox=false)",
                }
            )
        try:
            return self._run_task_inner(
                task, transcript, work_dir, agent_mode, driver, sandbox
            )
        finally:
            if sandbox is not None:
                # Rewritten after execution: only a container that really ran may
                # call itself ``container_run``.
                transcript.extra["sandbox"] = sandbox.record()
                if sandbox.runs:
                    self._note_container_run()
            if driver is not None:
                closer = getattr(driver, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # pragma: no cover - cleanup must not abort
                        pass
            shutil.rmtree(task_root, ignore_errors=True)
            transcript.extra["work_dir_removed"] = not task_root.exists()

    def _run_task_inner(
        self,
        task: core.TaskInstance,
        transcript: core.Transcript,
        work_dir: Path,
        agent_mode: bool,
        driver: Any,
        sandbox: isolation.Sandbox | None,
    ) -> TaskRun:
        """The task itself, on a working copy the caller already made and cleans."""
        if driver is not None:
            transcript.extra["driver"] = task.category

        if agent_mode:
            self._run_agent(task, transcript, work_dir, sandbox)
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
        started = time.perf_counter()  # elapsed metric only
        # One absolute deadline for the whole task, not one per request: the
        # old per-request ceiling was multiplied by every tool turn and doubled
        # again by the ``stream_options`` retry. Each request below gets the
        # smaller of its own ceiling and what is left on this deadline. The
        # deadline lives in the ``monotonic`` domain — the same clock the stop
        # rule below compares against; ``perf_counter`` only feeds durations.
        total_wall_clock = float(self.cfg.task_wall_clock or self.cfg.request_wall_clock or 1800)
        task_deadline = time.monotonic() + total_wall_clock
        sampler_before = self.sampler.peak_mib
        tool_calls = 0
        generated = 0
        # A task may need headroom above the shared cap (I-1 reads a dozen
        # files after listing the directory); the extra is explicit in the
        # fixture, so lowering the cap for a smoke run still applies.
        call_budget = self.cfg.max_tool_calls + int((task.oracle or {}).get("call_headroom") or 0)
        budget_spent = False
        followups = list(task.dialogue or [])
        # One ceiling for the whole task, and the same ceiling the answer budget
        # is clamped to: an unbounded window allowed a 17K completion under a
        # 16K total, and the scenario was then graded on its first half.
        total_limit = int(self.cfg.max_total_tokens)
        usage_missing = False
        tools_answered = False
        awaiting = False
        """A message the model still owes an answer to; a budget stop here is unverified."""
        overrun = False
        # Declared first: if the very first request dies, the report still names
        # the mode that was intended instead of showing nothing at all.
        transcript.extra["wire"] = {"mode": self.cfg.generation_mode, "verified": "declared"}
        try:
            while True:
                # One stop rule, checked before every new request: a spent
                # generation budget must never turn into another call just
                # because the previous reply happened to carry a tool call.
                # A scenario that was not presented to the end, or a tool result
                # that never reached the model, is unverified — not ``ok``.
                if generated >= total_limit:
                    if "бюджет генерации" not in " ".join(transcript.extra.get("notes", [])):
                        transcript.extra.setdefault("notes", []).append(
                            f"бюджет генерации исчерпан ({total_limit})"
                        )
                    unfinished = awaiting
                    if not unfinished:
                        if not transcript.turns or transcript.turns[-1].reply.tool_calls:
                            # The last reply only asked for tools the budget no
                            # longer allows: no answer was produced.
                            transcript.error = (
                                f"бюджет генерации исчерпан ({total_limit}) до ответа"
                            )
                            transcript.extra["failure"] = "incomplete_scenario"
                        break
                    # The dialogue turn or the release turn is still owed:
                    # stopping here would grade a half-presented scenario as ok.
                    if not overrun:
                        overrun = True
                        transcript.extra["total_budget_overrun"] = True
                        transcript.extra.setdefault("notes", []).append(
                            "бюджет превышен ради предъявления сценария до конца"
                        )
                    tools_answered = False
                # A request started after the deadline would only be killed by
                # its own socket timeout, so the task stops here instead and is
                # reported unverified rather than scored on a partial answer.
                left = task_deadline - time.monotonic()
                if left <= 0:
                    note = f"предел времени на задание {total_wall_clock:g} с исчерпан"
                    transcript.extra.setdefault("notes", []).append(note)
                    transcript.error = note
                    # A declared per-task ceiling is a property of the test, not
                    # of the endpoint: the scenario is exhausted, not
                    # unavailable, and stays in the denominator.
                    transcript.extra["failure"] = "candidate_resource_exhausted"
                    break
                request_budget = min(float(self.cfg.request_wall_clock or left), left)
                size, size_source = _request_size(
                    self.provider,
                    messages,
                    task.tools,
                    self.cfg,
                    task.id,
                    task.meta.get("chars_per_token"),
                )
                plan = self.cfg.plan_answer(task.category, size, source=size_source)
                if not overrun:
                    # The per-answer limit is bounded by what is left of the
                    # task total: two ceilings that disagreed were the bug.
                    remaining = total_limit - generated
                    if remaining < plan.tokens:
                        plan = replace(
                            plan,
                            tokens=max(1, remaining),
                            reason=f"{plan.reason}; остаток общего бюджета {remaining}",
                            capped_by="total_budget",
                        )
                transcript.extra.setdefault("answer_budgets", []).append(
                    {**plan.as_dict(), "turn": len(transcript.turns)}
                )
                if plan.floor:
                    transcript.extra.setdefault("notes", []).append(
                        f"ответ у floor {plan.tokens}: {plan.reason}"
                    )
                seed = derived_seed(self.cfg.seed, task.id, str(len(transcript.turns)))
                reply = self.provider.chat(
                    messages,
                    # Once the call budget is spent, send no tools: a tool-less
                    # request releases the model's final answer instead of one
                    # more callable batch. Without this every budget-capped task
                    # ended on an empty tool_calls reply (I-1, T-2, T-4, T-6).
                    None if budget_spent else task.tools,
                    max_new_tokens=plan.tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    seed=seed,
                    wall_clock=request_budget,
                    contract=self._wire_contract(plan.tokens, seed),
                )
                transcript.turns.append(core.Turn(messages=_copy_messages(messages), reply=reply))
                if isinstance(reply.raw.get("wire"), dict):
                    transcript.extra.setdefault("wire", {}).update(reply.raw["wire"])
                if reply.generated_tokens is None:
                    usage_missing = True
                generated += reply.generated_tokens or 0
                tools_answered = bool(reply.tool_calls)
                awaiting = tools_answered

                if reply.tool_calls and driver is None:
                    # Nothing can execute the calls: stop on the reply as it is.
                    break
                if not reply.tool_calls:
                    if followups:
                        # Scripted dialogue: hand the next user message to the
                        # model and keep every earlier constraint in the history.
                        messages.append(_assistant_message(reply, tool_calls=bool(reply.tool_calls)))
                        messages.append({"role": "user", "content": followups.pop(0)})
                        awaiting = True
                        continue
                    break
                room = call_budget - tool_calls
                if room <= 0:
                    if budget_spent:
                        # A tool-less request already went out and still came
                        # back with calls: one more would loop forever. No final
                        # answer was produced, so the row is unverified.
                        transcript.error = transcript.error or (
                            f"бюджет вызовов инструментов исчерпан ({call_budget}) до ответа"
                        )
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
                    awaiting = True
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
                    awaiting = True

                if generated >= total_limit:
                    if "бюджет генерации" not in " ".join(transcript.extra.get("notes", [])):
                        transcript.extra.setdefault("notes", []).append(
                            f"бюджет генерации исчерпан ({total_limit})"
                        )
                    break
        except ProviderError as exc:
            transcript.error = str(exc)
            if getattr(exc, "infra", True):
                transcript.extra["infra_error"] = True
                text = str(exc)
                transport = any(
                    marker in text
                    for marker in ("обрыв соединения", "unreachable", "сокет", "ConnectionReset", "RemoteDisconnected")
                )
                transcript.extra["failure"] = (
                    "transport_unavailable" if transport else "provider_unavailable"
                )
            else:
                transcript.extra["failure"] = "model_protocol_failure"
        except Exception as exc:  # pragma: no cover - defensive
            transcript.error = f"{type(exc).__name__}: {exc}"

        if usage_missing:
            # Without ``completion_tokens`` the stop rule counted every reply as
            # free: the total budget was bypassed silently and the row still
            # read ``ok``. Unknown spend is unverified spend.
            transcript.extra["usage_unconfirmed"] = True
            if transcript.error is None:
                transcript.extra["usage_unconfirmed_only"] = True
                transcript.error = (
                    "endpoint не сообщает расход токенов: бюджет генерации не подтверждён"
                )
            transcript.extra.setdefault("notes", []).append(
                "расход токенов не подтверждён endpoint'ом (usage отсутствует)"
            )

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
        wrote anything. The agent therefore gets the room the same
        ``plan_answer`` policy gives the chat path, floored at the category
        table so a tiny window cannot starve a whole-file answer.
        """
        table = int(self.cfg.max_new_tokens_by_category.get(task.category, self.cfg.max_new_tokens))
        if not self.cfg.unbounded_budget:
            return table
        return max(table, self.cfg.plan_answer(task.category, 0).tokens)

    def _run_agent(
        self,
        task: core.TaskInstance,
        transcript: core.Transcript,
        work_dir: Path,
        sandbox: isolation.Sandbox | None = None,
    ) -> None:
        """Hand the task to ``pi`` in its working copy and record what happened.

        The agent owns its own loop, so the transcript carries the agent's own
        report: one synthetic turn with the final words, one row per executed
        tool call, and the usage the agent measured. The answer itself is read
        from disk afterwards (see :func:`agent.answer_text`) — a chat reply is a
        fallback, not the deliverable.

        With *sandbox* the loop runs in a container that mounts the working copy
        and nothing else; the endpoint it dials is rewritten to the host gateway.
        """
        started = time.perf_counter()
        budget = self._agent_budget(task)
        # The agent loop runs inside ``pi``, so its per-answer ceiling, seed and
        # reasoning budget travel as provider model fields. There is no wire log
        # to inspect here; the contract is declared instead, and ``config_dir``
        # refuses to write an incomplete one.
        contract = self._wire_contract(budget, derived_seed(self.cfg.seed, task.id, "0"))
        transcript.extra["wire"] = contract.as_dict()
        endpoint, rewritten = isolation.endpoint_url(self.cfg.base_url)
        try:
            agent_dir = agent.config_dir(
                work_dir.parent,
                self.cfg,
                budget=budget,
                seed=contract.seed,
                base_url=endpoint if sandbox is not None else None,
                name=work_dir.name,
            )
            if sandbox is not None:
                sandbox.agent_dir = agent_dir
            # Read the pinned settings back off disk before the agent runs: the
            # transcript must name the document that was actually mounted, and
            # the digest is checkable against the file afterwards.
            settings = agent.settings_pin(agent_dir)
            prompt = agent.prompt_for(task)
            run = agent.run(prompt, work_dir, self.cfg, agent_dir=agent_dir, sandbox=sandbox)
        except Exception as exc:  # pragma: no cover - environment dependent
            transcript.error = f"{type(exc).__name__}: {exc}"
            transcript.extra["failure"] = "evaluator_unavailable"
            transcript.duration_s = time.perf_counter() - started
            return
        if sandbox is not None:
            transcript.extra["sandbox"] = sandbox.record()
            transcript.extra["sandbox"]["endpoint_rewritten_by_harness"] = rewritten

        transcript.extra.update(run.as_extra(self.cfg.api_key))
        transcript.extra["agent"]["settings"] = settings
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
            transcript.extra["failure"] = run.failure or "model_protocol_failure"

    def _score(self, task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
        module = core.REGISTRY.get(task.category)
        if module is None:
            return core.Score(
                task_id=task.id,
                category=task.category,
                error="category not registered",
                failure="evaluator_unavailable",
                max_points=task.max_points,
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
                failure="evaluator_unavailable",
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
            score.failure = transcript.extra.get("failure")
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


def _assistant_message(reply: core.ModelReply, *, tool_calls: bool = True) -> dict[str, Any]:
    """An assistant message that keeps the protocol fields of the reply.

    The history used to be rebuilt as a trimmed ``role/content/tool_calls``
    triple. That is enough for most stands, but DeepSeek's thinking mode
    requires the ``reasoning_content`` back on the assistant turn that carries
    the calls, and answers HTTP 400 without it. Whatever the provider recorded
    in ``raw["protocol"]`` — the reasoning under the key it was delivered with —
    is echoed verbatim; nothing is invented for a stand that sent none.
    """
    message: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
    if tool_calls:
        message["tool_calls"] = [
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
        ]
    protocol = reply.raw.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        # Offline providers carry the reasoning without a wire key; the
        # documented DeepSeek field is the safe default.
        protocol = {"reasoning_content": reply.reasoning} if reply.reasoning else {}
    for key, value in protocol.items():
        if key not in ("role", "content", "tool_calls") and value:
            message[key] = value
    return message


def _assistant_tool_message(reply: core.ModelReply) -> dict[str, Any]:
    return _assistant_message(reply)


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

            # The digest is the whole profile, independent of the --only slice.
            cfg = replace(self.cfg, only=()) if self.cfg.only else self.cfg
            self.suite_digest = suite_digest(build_suite(cfg), self.cfg.profile)
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
