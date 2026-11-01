"""The coding-agent runtime: a task is solved by ``pi`` inside a working copy.

Categories C, K and L hand the model a *repository*, not a single prompt: the
answer is whatever the agent leaves on disk after reading files, editing them and
running the project's own tests. One agent runtime serves all three, so a
scenario cannot accidentally measure a *different* harness than its neighbour.

Why a separate path from ``runner``'s chat loop:

* the chat loop owns the tool protocol — it sends tool schemas and executes the
  calls itself. That is right for a bare endpoint, and wrong for a coding agent:
  ``pi`` already has read/bash/edit/write, a system prompt that teaches them, and
  its own loop. Sending ``pi``'s prompt *and* the benchmark's schemas would
  measure neither.
* the agent owns its own budgets (turns, compaction, retries), so the transcript
  records what the agent reported (usage events) instead of a benchmark ceiling.

The isolation rules are the point of the module:

``sandbox``
    the run itself happens in a container (``dailybench/isolation.py``) that
    mounts the presented fixture and the config only — no repository, no oracle,
    an allowlisted environment and no network beyond the endpoint. Without a
    container the run stays on the host and the sandbox record says so;
``--no-session``
    no state leaks between tasks: every task is a fresh agent run in its own
    working copy;
``--no-context-files``
    the benchmark's own ``AGENTS.md`` must not write itself into the prompt;
``--no-approve`` / ``--no-extensions`` / ``--no-skills`` / ``--no-prompt-templates``
    a ``.pi`` directory a model happens to create cannot change the harness
    between two tasks;
``--offline``
    no catalog refresh and no telemetry, so two runs of two models differ only by
    the weights;
``PI_CODING_AGENT_DIR``
    a per-run config directory, unique to the task, holding exactly one provider
    (this endpoint), the seeds for the decoding policy and a pinned
    ``settings.json``: compaction, retries and thinking level are the harness's
    numbers, not whatever release defaults happen to be current;
``--tools read,bash,edit,write``
    the smallest tool set a coding task needs, spelled out so a future default
    cannot change the measurement.

The binary itself is pinned (``PI_VERSION``) and probed with ``--version`` at
every start: the CLI is not stable across releases — 0.80.10 rejects the
``--`` separator and stops before the first request — so a foreign release is an
explicit error naming the install command, never a quietly accepted guess. The
arrived version and the binary's sha256 are written into the transcript and the
run identity.

Anything the agent does *not* do is a result: a page it never wrote scores zero,
and the transcript keeps the events that show it.

Two operational details are load-bearing and easy to get wrong:

* **stdin must be empty** (``DEVNULL``). ``pi`` reads stdin when it is a pipe and
  waits for EOF, so a run started from a shell pipeline hung for the whole wall
  clock before emitting a single event.
* **the per-response ceiling must be the window-aware one**. An agent spends the
  budget on thinking *and* a tool call in the same response; a chat-sized 8192 was
  reached (8012 tokens) with nothing written yet, and the run ended on
  ``stopReason: length`` over an untouched file.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import core
from . import isolation
from .config import DEFAULT_CONTEXT_WINDOW, RunConfig
from .providers import WireContract

#: Environment variable with the ``pi`` binary, for a machine where it is not on
#: ``PATH`` (a user-local install, a CI image).
AGENT_BIN_ENV = "DAILYBENCH_AGENT_BIN"
#: Override the command that installs ``pi`` (``make agent`` uses the same one).
AGENT_INSTALL_ENV = "DAILYBENCH_AGENT_INSTALL"

#: The one ``pi`` release the harness is verified against, and the install
#: command that puts exactly that release on the machine. Pinning is
#: load-bearing: the CLI is not stable across releases. 0.80.10 rejects the
#: end-of-options separator outright (``Error: Unknown option: --``) and the run
#: dies before the first model request, which used to be indistinguishable from
#: "the model produced nothing".
PI_VERSION = "1.0.0"
#: Oldest release whose CLI the harness knows how to drive. Kept separate from
#: :data:`PI_VERSION` because the check (`== PI_VERSION`) is a policy decision
#: while the argv (:func:`_command`) only needs the capability.
PI_MIN_VERSION = "1.0.0"
#: First release that accepts ``--`` before positional messages.
PI_SEPARATOR_VERSION = "1.0.0"


#: Version of the pinned ``settings.json`` document. Bumped whenever the
#: document changes shape or numbers, so a transcript names the exact policy
#: that ran instead of inheriting the release's defaults.
PI_SETTINGS_VERSION = "1"

#: Pinned agent-level retries. A transient 529 must not turn a task into a
#: model failure, but an unbounded retry loop must not hide a dead endpoint.
PI_RETRIES = 3
#: Compaction reserves, clamped to the window: the trigger leaves at least this
#: much room below the ceiling and keeps at most this much history verbatim.
PI_RESERVE_MIN, PI_RESERVE_MAX = 1024, 16384
PI_KEEP_RECENT_MIN, PI_KEEP_RECENT_MAX = 2048, 20000

DEFAULT_INSTALL = f"npm install -g @earendil-works/pi-coding-agent@{PI_VERSION}"

#: Where the npm global prefix usually puts the binary when it is not on PATH.
PI_CANDIDATES = (
    Path.home() / ".npm-global" / "bin" / "pi",
    Path("/usr/local/bin/pi"),
    Path("/opt/homebrew/bin/pi"),
    Path.home() / ".local" / "bin" / "pi",
)

#: The tool set a coding task needs. Spelled out, never inherited: a future
#: default change must not silently change what the benchmark measures.
AGENT_TOOLS = ("read", "bash", "edit", "write")

#: Appended to every agent prompt. The scenario prompt says *what* to build; this
#: says how the result is graded, so a model cannot "answer" instead of writing.
AGENT_CONTRACT = """## Как это оценивается

Работай в текущем каталоге как обычно: читай файлы, правь их, запускай команды.
Финальный ответ в чате ничего не решает — проверяется то, что осталось на диске.

{target}

Заверши работу, когда результат записан в файл и команда проверки проходит."""

AGENT_TARGET_FILE = (
    "Результат — файл `{path}`: он обязан существовать и содержать готовый текст целиком.\n"
    "Тесты в рабочей копии запускать можно и нужно: они показывают, что реализация верна;"
    " сами тесты менять запрещено."
)
AGENT_TARGET_PAGE = (
    "Результат — файл `{path}`: он обязан существовать и быть ровно тем документом,"
    " который описан выше (цельный `<html>`, без внешних ресурсов)."
)


def binary() -> str | None:
    """The ``pi`` binary, from the environment, ``PATH`` or the usual prefixes."""
    from_env = os.environ.get(AGENT_BIN_ENV)
    if from_env and Path(from_env).is_file():
        return from_env
    found = shutil.which("pi")
    if found:
        return found
    for candidate in PI_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def install_command() -> str:
    return os.environ.get(AGENT_INSTALL_ENV) or DEFAULT_INSTALL


class AgentVersionError(RuntimeError):
    """The found ``pi`` is not the pinned, verified release.

    ``probed`` carries what was learnt before the refusal (when anything was),
    so the failed run can still record the version and digest it met.
    """

    def __init__(self, message: str, probed: PiBinary | None = None) -> None:
        super().__init__(message)
        self.probed = probed


_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")


@dataclass(frozen=True)
class PiBinary:
    """A probed binary: where it is, what it claims to be, and its bytes.

    ``sha256`` is the digest of the executable as found (a symlink is followed,
    so the digest covers the code that actually runs). Version and digest both
    travel into the transcript and the run identity: "pinned agent" is only a
    claim until the arrived binary is named and hashed.
    """

    path: str
    version: str
    sha256: str

    @property
    def parsed(self) -> tuple[int, int, int]:
        return parse_version(self.version) or (0, 0, 0)


def parse_version(text: str) -> tuple[int, int, int] | None:
    """The first ``X.Y.Z`` in ``--version`` output (``1.0.0``, ``pi/1.0.0``)."""
    match = _VERSION_RE.search(text or "")
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def version_hint(found: str | None = None) -> str:
    """The one actionable sentence for a wrong or unreadable binary."""
    which = f"pi {found}" if found else "pi"
    return (
        f"{which} не подходит: каркас проверен ровно на pi {PI_VERSION}. "
        f"Поставьте её: `{install_command()}` или укажите проверенный бинарник "
        f"в {AGENT_BIN_ENV}."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Probe cache keyed by (real path, mtime, size): a run probes once per binary,
#: and a replaced binary at the same path is probed again.
_PROBED: dict[tuple[str, int, int], PiBinary] = {}


def inspect(executable: str) -> PiBinary:
    """The binary's version and digest, from the binary itself.

    ``--version`` is executed, never inferred from the path or a package file: a
    shim can point at any release. Unreadable output or a non-zero exit is an
    explicit error, not an accepted unknown — an unverified agent is exactly
    what produced silent zeros before.
    """
    path = Path(executable)
    try:
        stat = path.stat()
    except OSError as exc:
        raise AgentVersionError(f"{executable}: {exc}. {version_hint()}") from exc
    key = (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)
    cached = _PROBED.get(key)
    if cached is not None:
        return cached
    try:
        digest = _sha256(path)
    except OSError as exc:
        raise AgentVersionError(f"{executable}: не читается ({exc}). {version_hint()}") from exc
    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AgentVersionError(f"{executable}: {exc.__class__.__name__}: {exc}. {version_hint()}") from exc
    if proc.returncode != 0:
        raise AgentVersionError(f"{executable}: `--version` завершился с кодом {proc.returncode}. {version_hint()}")
    parsed = parse_version((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if parsed is None:
        raise AgentVersionError(f"{executable}: не удалось разобрать версию из `--version`. {version_hint()}")
    found = "%d.%d.%d" % parsed
    probed = PiBinary(path=executable, version=found, sha256=digest)
    _PROBED[key] = probed
    return probed


def verify(executable: str) -> PiBinary:
    """Probe the binary and refuse anything but the pinned release.

    Older releases are the dangerous case (0.80.10 parses no ``--``), but a
    newer one is refused too: nothing about it has been verified against the
    recorded results, so accepting it would silently change the harness.
    """
    probed = inspect(executable)
    expected = parse_version(PI_VERSION)
    if probed.parsed != expected:
        below = ""
        if probed.parsed < (parse_version(PI_MIN_VERSION) or expected):
            below = f" Она старше минимальной поддерживаемой {PI_MIN_VERSION}."
        raise AgentVersionError(f"{executable}: {version_hint(probed.version)}{below}", probed)
    return probed


def identity() -> dict[str, Any]:
    """The agent half of the run identity: pinned requirement, arrived version,
    digest and settings pin.

    Never installs and never raises: ``doctor`` and ``resume`` read it on
    machines where the binary may be absent or wrong, and the identity must
    still be computable (with ``found: null``) so a mismatch is *reported*
    rather than crashing before the comparison.
    """
    found = binary()
    if not found:
        return {"required": PI_VERSION, "found": None, "sha256": None, "settings": PI_SETTINGS_VERSION}
    try:
        probed = inspect(found)
    except AgentVersionError as exc:
        return {
            "required": PI_VERSION,
            "found": None,
            "sha256": None,
            "settings": PI_SETTINGS_VERSION,
            "error": str(exc),
        }
    return {
        "required": PI_VERSION,
        "found": probed.version,
        "sha256": probed.sha256,
        "settings": PI_SETTINGS_VERSION,
    }


def install(log: Any = None) -> tuple[bool, str]:
    """Install ``pi`` when it is missing. Returns ``(ok, note)``.

    The benchmark ships no dependency on npm, but a missing agent runtime is a
    configuration error rather than an unverified result: unlike a missing Go
    toolchain, the fix is one command and the run is otherwise complete.
    """
    command = install_command()
    if log:
        log(f"pi не найден; ставлю: {command}")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()
        return False, tail[-1][:300] if tail else f"код {proc.returncode}"
    return bool(binary()), "установлен"


def ensure(log: Any = None) -> str:
    """The binary, installed first when missing and verified either way.

    Verification lives here rather than in :func:`run` so ``make agent`` fails
    at install time on a wrong release instead of at task 1 of a long run.
    """
    found = binary()
    if not found:
        ok, note = install(log)
        if not ok or not binary():
            raise RuntimeError(f"pi недоступен и не установился ({note}); выставьте {AGENT_BIN_ENV}")
        found = str(binary())
    verify(found)
    return found


#: Event kinds worth persisting: turns, tool traffic and terminations. Token
#: deltas and stream bookkeeping would only bloat the transcript.
_DIAGNOSTIC_KINDS = {
    "message_start",
    "message_end",
    "tool_execution_start",
    "tool_execution_end",
    "compaction_start",
    "compaction_end",
    "auto_compaction_start",
    "auto_compaction_end",
    "auto_retry_start",
    "auto_retry_end",
    "error",
    "agent_end",
}
_DIAGNOSTIC_EVENTS = 400
_DIAGNOSTIC_FIELD = 2000


#: Secret shapes scrubbed from anything a tool can print: the endpoint is handed
#: to the agent as ``apiKey`` and as ``DAILYBENCH_API_KEY``, so an ordinary
#: ``cat models.json`` or ``echo`` would otherwise persist it into ``runs/``.
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{8,}")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")


def redact(text: str, secret: str | None = None) -> str:
    """Scrub the endpoint key from ``text`` before it reaches a transcript.

    Two rules: the configured key by exact value (the agent's ``models.json``
    holds it verbatim, so ``***`` must not depend on its shape), and the generic
    ``Bearer <token>`` / ``sk-…`` shapes for keys the harness never saw (an
    ``Authorization`` header printed by a self-check, a nested provider).
    """
    if secret and secret != "not-needed" and len(secret) >= 6:
        text = text.replace(secret, "***")
    text = _BEARER_RE.sub("Bearer ***", text)
    return _SK_RE.sub("***", text)


def _redacted(value: Any, secret: str | None) -> Any:
    if isinstance(value, str):
        return redact(value, secret)
    if isinstance(value, list):
        return [_redacted(item, secret) for item in value]
    if isinstance(value, dict):
        return {str(key): _redacted(item, secret) for key, item in value.items()}
    return value


def _clipped(value: Any, limit: int = _DIAGNOSTIC_FIELD) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [_clipped(item, limit) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _clipped(item, limit) for key, item in value.items()}
    return value


def _diagnostic_events(events: list[dict[str, Any]], secret: str | None = None) -> list[dict[str, Any]]:
    """A bounded, still-readable slice of the stream for the transcript."""
    kept = [event for event in events if event.get("type") in _DIAGNOSTIC_KINDS]
    return [_clipped(_redacted(event, secret)) for event in kept[-_DIAGNOSTIC_EVENTS:]]


@dataclass
class AgentRun:
    """What one agent invocation produced."""

    ok: bool
    text: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    prompt_tokens: int | None = None
    answer_tokens: int | None = None
    reasoning_tokens: int | None = None
    prompt_tokens_last: int | None = None
    ttft_s: float | None = None
    duration_s: float = 0.0
    finish_reasons: list[str] = field(default_factory=list)
    error: str | None = None
    failure: str | None = None
    """Taxonomy kind (see :data:`core.FAILURE_KINDS`) when the invocation broke."""
    stderr_tail: str = ""
    agent_version: str | None = None
    agent_sha256: str | None = None
    agent_path: str | None = None
    #: Output tokens spent on compaction summaries; part of ``answer_tokens``,
    #: kept apart so the report can show the measurement overhead.
    summary_tokens: int | None = None
    #: One row per compaction the agent performed: reason and before/after size.
    compactions: list[dict[str, Any]] = field(default_factory=list)

    def replay_telemetry(self, journal: Any) -> None:
        """Replay the event stream into the task journal.

        The agent owns its loop, so the stream is the only trace of when a
        request started, when the first token came and how long a tool ran.
        ``pi`` timestamps nothing, so the gaps between preparation and send are
        marked unknown; order, TTFT (``ttftMs``) and usage are exact.
        """
        pending: dict[str, float] = {}
        for event in self.events:
            kind = event.get("type")
            if kind == "message_start":
                journal.attempt_start(reason="agent", queue_known=False)
                journal.request_sent(reason="agent")
            elif kind == "message_end":
                message = event.get("message") or {}
                if message.get("role") != "assistant":
                    continue
                usage = message.get("usage") or {}
                ttft = message.get("ttftMs")
                journal.first_token(
                    ttft_s=(float(ttft) / 1000.0) if isinstance(ttft, (int, float)) else None
                )
                journal.usage(
                    prompt_tokens=usage.get("input") if isinstance(usage.get("input"), int) else None,
                    completion_tokens=(
                        usage.get("output") if isinstance(usage.get("output"), int) else None
                    ),
                    reasoning_tokens=(
                        usage.get("reasoning") if isinstance(usage.get("reasoning"), int) else None
                    ),
                    exact=isinstance(usage.get("output"), int),
                )
            elif kind == "tool_execution_start":
                pending[str(event.get("toolCallId"))] = time.perf_counter()
                journal.tool_call_start(name=str(event.get("toolName") or ""))
            elif kind == "tool_execution_end":
                name = str(event.get("toolName") or "")
                started = pending.pop(str(event.get("toolCallId")), None)
                journal.tool_call_end(
                    name=name,
                    ok=not bool(event.get("isError")),
                    duration_s=(time.perf_counter() - started) if started else None,
                    error=bool(event.get("isError")),
                )

    def as_extra(self, secret: str | None = None) -> dict[str, Any]:
        """The provenance blob a report reads: how the agent was driven.

        The raw stream is kept, not just its length: a timeout's transcript used
        to say only how many events there were, so a run that had already
        measured usage and called tools looked empty. Long fields are clipped and
        the log is capped; the counts stay exact. ``secret`` is the endpoint key,
        scrubbed so a tool that printed it leaves no plaintext in the artefact.

        The binary is named by version and digest: the README's "pinned agent"
        is only true if the transcript shows which release actually ran. The
        compactions the agent performed are kept too: a summary is a model
        request the score depends on, and its output tokens are part of the
        spend.
        """
        log = _diagnostic_events(self.events, secret)
        return {
            "agent": {
                "tools": list(AGENT_TOOLS),
                "required_version": PI_VERSION,
                "version": self.agent_version,
                "sha256": self.agent_sha256,
                "binary": self.agent_path,
                "events": len(self.events),
                "event_log": log,
                "events_logged": len(log),
                "tool_calls": len(self.tool_calls),
                "tool_names": [str(c.get("name")) for c in self.tool_calls],
                "compactions": self.compactions,
                "summary_tokens": self.summary_tokens,
                "stderr": redact(self.stderr_tail, secret)[-600:],
            }
        }


def _write_atomic(path: Path, text: str) -> None:
    """Write ``path`` through a temp file in the same directory, then replace.

    ``os.replace`` is atomic on every platform the harness runs on, so a reader
    (another task's process, a human inspecting the run) sees the whole document
    or none, never a half-written one.
    """
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def pinned_settings(cfg: RunConfig) -> dict[str, Any]:
    """The ``settings.json`` document the harness pins for every agent task.

    The release's own defaults are a moving target and were invisible in the
    transcript (R04): context was compacted at an internal reserve, retries
    happened silently, thinking level was inherited. Every knob that can change
    what the model sees is therefore written out. The numbers are derived from
    the window so a small window keeps a usable working set; for the standard
    window they reproduce the release's own defaults.
    """
    window = int(cfg.context_window or DEFAULT_CONTEXT_WINDOW)
    reserve = min(max(window // 8, PI_RESERVE_MIN), PI_RESERVE_MAX, max(window // 4, 1))
    keep_recent = min(max(window // 2, PI_KEEP_RECENT_MIN), PI_KEEP_RECENT_MAX, window - reserve)
    return {
        "defaultThinkingLevel": "medium",
        # The argv already spells the tool set out; this keeps a future release
        # default from adding a tool to a session started without ``--tools``.
        "defaultTools": list(AGENT_TOOLS),
        "compaction": {
            "enabled": True,
            "reserveTokens": reserve,
            "keepRecentTokens": keep_recent,
        },
        "retry": {
            "enabled": True,
            "maxRetries": PI_RETRIES,
            "baseDelayMs": 2000,
            "maxAgentDelayMs": 60000,
        },
        "httpIdleTimeoutMs": 300000,
        "showCacheMissNotices": False,
    }


def settings_pin(folder: Path) -> dict[str, Any]:
    """The settings document actually on disk, with the pin version and digest.

    Read back rather than taken from the builder: the point of the pin is that
    the transcript's claim is checkable against the file the agent read.
    """
    path = folder / "settings.json"
    text = path.read_text(encoding="utf-8")
    return {
        "version": PI_SETTINGS_VERSION,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "document": json.loads(text),
        "path": str(path),
    }


def config_dir(
    root: Path,
    cfg: RunConfig,
    *,
    budget: int | None = None,
    seed: int | None = None,
    base_url: str | None = None,
    name: str | None = None,
) -> Path:
    """A throwaway per-run ``PI_CODING_AGENT_DIR`` holding exactly this endpoint.

    ``root`` is the shared temp parent every working copy is created under, so
    the directory itself is made with :func:`tempfile.mkdtemp`: two tasks (or
    two parallel benchmark processes) each get their own folder and cannot read
    each other's endpoint, model or key. ``name`` only makes the leftover folder
    recognisable in a temp directory listing.

    ``models.json``, ``wire.json`` and the pinned ``settings.json`` go through
    :func:`_write_atomic`, so a reader never sees half a document. The settings
    document pins compaction, retries and the thinking level instead of
    inheriting the release's defaults (R04); its version and digest are recorded
    per task (see :func:`settings_pin`). No ``auth.json`` is written, so the key
    never lands on disk next to the run.

    ``budget`` is the per-answer ceiling for the task's category. It matters: the
    agent thinks *and* emits a tool call in one response, so a chat-sized 4096
    can be spent on thinking with nothing written — the run then ends on
    ``stopReason: length`` and the working copy keeps the stub.

    The whole decoding contract goes into ``samplingParams``, because ``pi``
    copies that record onto the request body verbatim. A model-level
    ``temperature`` is inert — the adapter never forwarded it — which is how the
    declared sampling silently stopped being the measured one (H08). In
    ``native`` mode the record is left out on purpose: the stand decides.

    ``base_url`` overrides the config's endpoint for one caller: inside the
    sandbox a loopback endpoint is reached through ``host.docker.internal``, and
    ``models.json`` is what the agent dials.
    """
    prefix = f"{name}." if name else ""
    folder = Path(tempfile.mkdtemp(prefix=f"{prefix}pi-agent-", dir=root))
    limit = int(budget or cfg.max_new_tokens)
    contract = wire_contract(cfg, limit, seed)
    model: dict[str, Any] = {
        "id": cfg.model,
        "name": cfg.model,
        "reasoning": True,
        "contextWindow": int(cfg.context_window or DEFAULT_CONTEXT_WINDOW),
        "maxTokens": limit,
        # ``reasoning_effort`` is not forced: with it, pi sends its own
        # "medium" and a stand may reject it (DeepSeek V4.1 accepts only
        # low/high/xhigh/max). The thinking cap is pinned instead, in
        # ``samplingParams``, so the budget is the same for every model.
        "compat": {"supportsReasoningEffort": False},
    }
    fields = contract.payload_fields(include_limit=False)
    if fields:
        model["samplingParams"] = fields
    payload = {
        "providers": {
            "bench": {
                "api": "openai-completions",
                "baseUrl": base_url or cfg.base_url,
                "apiKey": cfg.api_key or "not-needed",
                "models": [model],
            }
        }
    }
    _check_contract(model, contract)
    _write_atomic(folder / "models.json", json.dumps(payload, ensure_ascii=False, indent=2))
    _write_atomic(folder / "wire.json", json.dumps(contract.as_dict(), ensure_ascii=False, indent=2))
    _write_atomic(folder / "settings.json", json.dumps(pinned_settings(cfg), ensure_ascii=False, indent=2))
    return folder


def wire_contract(cfg: RunConfig, output_limit: int, seed: int | None) -> WireContract:
    """The declared generation contract for one agent task.

    ``pi`` writes no request log, so the contract cannot be read off a payload
    here; it is declared, written next to ``models.json`` and checked against
    the model definition before the agent starts.
    """
    fixed = cfg.generation_mode == "fixed"
    return WireContract(
        mode=cfg.generation_mode,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=seed,
        output_limit=int(output_limit),
        reasoning_budget=int(cfg.reasoning_budget) if fixed else None,
        stop=tuple(cfg.stop_sequences),
        template_version=cfg.template_version,
    )


def _check_contract(model: dict[str, Any], contract: WireContract) -> None:
    """Refuse a model definition that lost part of the contract.

    The same rule the chat transport applies to its final payload: a field that
    did not reach the definition is a different condition, not a detail. The
    answer limit travels as pi's ``maxTokens`` alias, which becomes
    ``max_tokens`` on the wire.
    """
    probe = {
        **model,
        "max_tokens": model.get("maxTokens"),
        **(model.get("samplingParams") or {}),
    }
    missing = [key for key in contract.payload_fields() if key not in probe]
    if missing:
        raise RuntimeError(f"pi payload без полей контракта: {', '.join(missing)}")


def _tool_path(current: str) -> str:
    """``PATH`` plus the toolchains the benchmark itself locates by hand.

    Only directories that really hold a tool are added, and only in front of the
    existing value, so nothing on the machine is shadowed silently.
    """
    from .tasks.coding_go import _go_binary

    extra: list[str] = []
    go = _go_binary()
    if go:
        extra.append(str(Path(go).parent))
    for candidate in (Path("/usr/local/bin"), Path("/opt/homebrew/bin")):
        if candidate.is_dir():
            extra.append(str(candidate))
    parts = [p for p in ([*extra, current] if current else extra) if p]
    seen: list[str] = []
    for part in parts:
        for piece in part.split(os.pathsep):
            if piece and piece not in seen:
                seen.append(piece)
    return os.pathsep.join(seen)


def _usage_of(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals from the agent's own usage events, never from the reply text.

    ``pi`` reports cumulative provider usage per assistant message, so the sum
    over messages is the honest spend. A compaction is one more model request
    whose output steers the rest of the run: its usage is added to the answer
    total and also reported apart, so summarization cannot look free. Missing
    usage stays ``None`` instead of being guessed.
    """
    prompts: list[int] = []
    answers = 0
    reasoning = 0
    measured_output = False
    measured_reasoning = False
    turns = 0
    ttft: float | None = None
    finishes: list[str] = []
    compactions: list[dict[str, Any]] = []
    summary_output = 0
    measured_summary = False
    for event in events:
        kind = event.get("type")
        if kind in ("compaction_end", "auto_compaction_end"):
            result = event.get("result") or {}
            usage = result.get("usage") or {}
            output = usage.get("output")
            if isinstance(output, int):
                summary_output += int(output)
                measured_summary = True
            compactions.append(
                {
                    "reason": event.get("reason"),
                    "tokens_before": result.get("tokensBefore"),
                    "tokens_after": result.get("estimatedTokensAfter"),
                    "output_tokens": output if isinstance(output, int) else None,
                    "aborted": bool(event.get("aborted")),
                    "error": event.get("errorMessage"),
                }
            )
            continue
        if kind != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        turns += 1
        usage = message.get("usage") or {}
        if isinstance(usage.get("input"), int):
            prompts.append(int(usage["input"]))
        if isinstance(usage.get("output"), int):
            answers += int(usage["output"])
            measured_output = True
        if isinstance(usage.get("reasoning"), int):
            reasoning += int(usage["reasoning"])
            measured_reasoning = True
        if ttft is None and isinstance(message.get("ttftMs"), (int, float)):
            ttft = float(message["ttftMs"]) / 1000.0
        reason = message.get("stopReason")
        if reason and reason != "pending":
            finishes.append(str(reason))
    if measured_summary:
        answers += summary_output
        measured_output = True
    return {
        "turns": turns,
        "prompt_tokens": prompts[-1] if prompts else None,
        # Unknown spend stays ``None``: a zero would read as a free run in the
        # report, and a timeout still records what the provider measured.
        "answer_tokens": answers if measured_output else None,
        "reasoning_tokens": reasoning if measured_reasoning else None,
        "summary_tokens": summary_output if measured_summary else None,
        "compactions": compactions,
        "ttft_s": ttft,
        "finish_reasons": finishes,
    }


def _tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per executed tool call: name, arguments, result, error flag."""
    started: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("type")
        if kind == "tool_execution_start":
            started[str(event.get("toolCallId"))] = event
        elif kind == "tool_execution_end":
            call = started.pop(str(event.get("toolCallId")), {})
            result = event.get("result") or {}
            text = " ".join(
                str(block.get("text") or "")
                for block in (result.get("content") or [])
                if isinstance(block, dict)
            )
            rows.append(
                {
                    "turn": 0,
                    "name": str(event.get("toolName") or ""),
                    "arguments": call.get("args") or {},
                    "result": text[:4000],
                    "error": bool(event.get("isError")),
                    "illegal": False,
                }
            )
    return rows


def _final_text(events: list[dict[str, Any]]) -> str:
    """The agent's last visible text — what it said after finishing the work."""
    text = ""
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        blocks = [
            str(block.get("text") or "")
            for block in (message.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "".join(blocks)
        if joined.strip():
            text = joined
    return text


def _is_windows() -> bool:
    """Read at call time: the tests drive the Windows branch from any host."""
    return sys.platform.startswith("win")


def _spawn_kwargs() -> dict[str, Any]:
    """Keep the agent in a tree of its own on every platform.

    POSIX gets a session (so ``killpg`` reaches the shells ``pi`` starts);
    Windows has no groups to signal and gets a console group instead, while the
    Job Object attached afterwards is what really covers descendants.
    """
    if _is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) or 0}
    return {"start_new_session": True}


def _win32() -> Any:
    """``kernel32`` on Windows, ``None`` on any other host (or no ctypes)."""
    if not _is_windows():
        return None
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Handles are pointer-sized; the default ``c_int`` restype truncates.
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        return kernel32
    except (AttributeError, OSError):  # pragma: no cover - Windows only
        return None


class _WindowsJob:
    """A kill-on-demand Job Object: reaches descendants whatever their cwd.

    Windows has neither process groups worth signalling nor ``lsof`` to find a
    survivor by working directory. A job captures the whole tree at creation,
    including a ``start_new_session`` equivalent that escapes a ``killpg`` on
    POSIX, so one handle is both the timeout kill and the post-exit sweep.
    """

    def __init__(self, kernel32: Any, handle: Any, pid: int) -> None:
        self._kernel32 = kernel32
        self.handle = handle
        self.pid = pid
        self.closed = False

    @classmethod
    def attach(cls, proc: subprocess.Popen) -> _WindowsJob | None:
        kernel32 = _win32()
        if kernel32 is None:
            return None
        _JOB_ATTACH_ERRORS.pop(proc.pid, None)
        try:
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return None
            job = cls(kernel32, handle, proc.pid)
            if not kernel32.AssignProcessToJobObject(handle, int(proc._handle)):
                import ctypes

                _JOB_ATTACH_ERRORS[proc.pid] = f"не прикреплён: WinError {ctypes.get_last_error()}"
                job.close()
                return None
            return job
        except (AttributeError, OSError, ValueError):  # pragma: no cover - Windows only
            return None

    def terminate(self) -> None:
        if self.closed:
            return
        try:
            self._kernel32.TerminateJobObject(self.handle, 1)
        except (AttributeError, OSError):  # pragma: no cover - Windows only
            pass

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._kernel32.CloseHandle(self.handle)
        except (AttributeError, OSError):  # pragma: no cover - Windows only
            pass


# Work dirs of agent runs that have not yet been swept: the atexit hook keyed on
# them is the only cover for an exit that never unwinds ``run``'s stack.
_OPEN_WORK_DIRS: dict[str, Path] = {}
# The Windows job of an open run, by working directory and by agent pid.
_OPEN_JOBS: dict[str, _WindowsJob] = {}
_PROC_JOBS: dict[int, _WindowsJob] = {}
# Last ``AssignProcessToJobObject`` failure by agent pid, for diagnostics.
_JOB_ATTACH_ERRORS: dict[int, str] = {}


def _guard_process(proc: subprocess.Popen, work_dir: Path) -> None:
    """Attach a job on Windows; a no-op where ``killpg`` already works."""
    if not _is_windows():
        return
    job = _WindowsJob.attach(proc)
    if job is None:
        return  # no ctypes/job on this host: ``taskkill``/``proc.kill`` remain
    _OPEN_JOBS[str(work_dir)] = job
    _PROC_JOBS[proc.pid] = job


def _proc_ppid(pid: int) -> int | None:
    """Parent of ``pid`` from ``/proc``; ``None`` when there is no ``/proc``."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # ``comm`` may contain spaces and parentheses: split after the last ')'.
    fields = stat.rsplit(")", 1)[-1].split()
    try:
        return int(fields[1])  # state, then ppid
    except (IndexError, ValueError):
        return None


def _descendants_proc(root: int) -> list[int]:
    """A parent-chain walk over ``/proc``, used when ``ps`` is absent."""
    base = Path("/proc")
    if not base.is_dir():
        return []
    children: dict[int, list[int]] = {}
    for entry in base.iterdir():
        if not entry.name.isdigit():
            continue
        ppid = _proc_ppid(int(entry.name))
        if ppid is not None:
            children.setdefault(ppid, []).append(int(entry.name))
    return _walk_children(root, children)


def _walk_children(root: int, children: dict[int, list[int]]) -> list[int]:
    found: list[int] = []
    queue = [root]
    while queue:
        pid = queue.pop()
        found.append(pid)
        queue.extend(children.get(pid, []))
    return found


def _descendants(root: int) -> list[int]:
    """Every live descendant of ``root`` (BFS), root included.

    Needed because the process *group* is not enough here: ``pi`` starts its
    shell tool with ``detached: true`` (its own session), so a Chrome a
    self-check script launched has a pgid of its own and survives a ``killpg``
    on the agent's group. The parent chain is the only link that always holds.

    ``ps`` is the portable POSIX source; where it is missing the kernel's
    ``/proc`` is read instead. Windows relies on ``taskkill /T`` or the job, so
    the walk is just the root there.
    """
    if _is_windows():
        return [root]
    ps = shutil.which("ps")
    if ps is None:
        return _descendants_proc(root) or [root]
    try:
        out = subprocess.run(
            [ps, "-Ao", "pid=,ppid="], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - platform dependent
        return _descendants_proc(root) or [root]
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    return _walk_children(root, children)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, just not ours any more
        return True
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        return False
    return True


def _signal(pids: list[int], pgid: int | None, sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            # ProcessLookupError: it exited between the snapshot and the
            # signal. PermissionError: macOS returns EPERM for a pid that
            # was already reaped and handed to another process.
            continue
        except AttributeError:  # pragma: no cover - os.kill exists on Windows too
            break
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        except AttributeError:  # pragma: no cover - Windows without a group
            pass


def _wait_gone(proc: subprocess.Popen, pids: list[int], timeout: float) -> bool:
    """Wait until the agent and every snapshotted pid are gone."""
    deadline = time.monotonic() + timeout
    while True:
        others = [pid for pid in pids if pid != proc.pid]
        if proc.poll() is not None and not any(_alive(pid) for pid in others):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


#: Seconds a runaway script gets to flush after SIGTERM before SIGKILL.
_TERM_GRACE = 3.0


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the agent and everything it spawned, then reap it.

    POSIX signals the group *and* a snapshotted parent chain, because either
    alone leaks: the group catches the shells ``pi`` starts directly, the walk
    catches the Chrome a self-check script started in its own session. SIGTERM
    first, so a runaway script gets a chance to flush; the tree is polled until
    it is empty, and only then is SIGKILL skipped — a descendant that ignores
    TERM still gets the hammer.

    Windows has no signals: ``taskkill /T /F`` reaches the tree, the job (when
    one could be attached) reaches it even when the process died early, and
    ``proc.kill`` is the last resort when neither exists.
    """
    if _is_windows():
        _kill_tree_windows(proc)
        return
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (AttributeError, ProcessLookupError):
        pgid = None
    # Snapshot the tree *before* signalling: once the agent dies its children are
    # reparented to PID 1 and the parent chain no longer leads to them.
    targets = _descendants(proc.pid)
    _signal(targets, pgid, signal.SIGTERM)
    if _wait_gone(proc, targets, _TERM_GRACE):
        return
    # Re-scan: the tree may have grown while TERM was being delivered.
    targets = sorted(set(targets) | set(_descendants(proc.pid)))
    _signal(targets, pgid, signal.SIGKILL)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _kill_tree_windows(proc: subprocess.Popen) -> None:
    """``taskkill`` first, job and ``proc.kill`` as fallbacks."""
    job = _PROC_JOBS.pop(proc.pid, None)
    if job is not None:
        job.terminate()
        job.close()
    if proc.poll() is None:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
            )
            if result.returncode == 0:
                try:
                    proc.wait(timeout=10)
                    return
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
        except (OSError, subprocess.TimeoutExpired):
            pass  # no taskkill on this Windows (or it hung): kill the root
    try:
        proc.kill()
    except OSError:  # pragma: no cover - already gone
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _strays_in(work_dir: Path) -> list[int]:
    """Pids, other than ours, whose cwd is inside ``work_dir``.

    ``lsof`` prints the *resolved* cwd (``/private/var`` where ``TMPDIR`` says
    ``/var``), so the comparison goes through ``realpath`` on both sides. It is
    an optional external tool: without it, Linux ``/proc`` is enough, and a
    host with neither simply yields no cwd-based strays (Windows uses a job).
    """
    if _is_windows():
        return []
    real = os.path.realpath(work_dir)
    lsof = shutil.which("lsof")
    if lsof is None and Path("/usr/sbin/lsof").is_file():
        lsof = "/usr/sbin/lsof"
    if lsof is None:
        return _strays_by_proc(real)
    try:
        out = subprocess.run(
            [lsof, "-w", "-d", "cwd", "-Fpn"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - platform dependent
        return _strays_by_proc(real)
    found: list[int] = []
    pid: int | None = None
    for line in out.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid is not None:
            cwd = line[1:]
            if pid != os.getpid() and (cwd == real or cwd.startswith(real + os.sep)):
                found.append(pid)
    return found


def _strays_by_proc(real: str) -> list[int]:
    """Fallback cwd discovery by ``/proc/<pid>/cwd`` symlinks."""
    base = Path("/proc")
    if not base.is_dir():
        return []
    found: list[int] = []
    for entry in base.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cwd = os.path.realpath(os.readlink(entry / "cwd"))
        except OSError:
            continue
        if cwd == real or cwd.startswith(real + os.sep):
            found.append(int(entry.name))
    return found


def _kill_strays(work_dir: Path) -> None:
    """Kill every survivor of a finished task, whatever its parent is now.

    ``_kill_tree`` cannot cover this: a model's own check script *detaches*
    (``python -m http.server`` in its own session, ``node check.js`` left in the
    background), the intermediate shell exits long before the agent does, and on
    a *clean* exit the tree walk never runs at all. The survivor keeps its
    hard-coded port bound, and the next task's identical script then hangs on
    the occupied port for the whole wall clock — one leak poisons every
    following task.

    The working copy is per-task and unique, so "cwd inside it" is the one
    marker that survives every reparenting. Called after the agent process is
    gone: a live agent's own shells also live there. On Windows the job attached
    at spawn covers descendants even after they change cwd.

    The sweep is also idempotent: after a timeout the job was already
    terminated, so a second call finds nothing.
    """
    job = _OPEN_JOBS.pop(str(work_dir), None)
    if job is not None:
        _PROC_JOBS.pop(job.pid, None)
        job.terminate()
        job.close()
    pids = _strays_in(work_dir)
    if not pids:
        return
    targets: list[int] = []
    for pid in pids:
        targets.extend(_descendants(pid))
    _signal(targets, None, signal.SIGTERM)
    if _wait_gone_pids(targets, 0.5):
        return
    _signal(targets, None, signal.SIGKILL)


def _wait_gone_pids(pids: list[int], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not any(_alive(pid) for pid in pids):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _sweep_on_exit() -> None:
    """Last-resort sweep for the exit path that never unwinds our stack.

    ``finally: _kill_strays`` covers an exception, but a debugger's
    ``PyRun_SimpleString`` raising ``SystemExit`` ends the interpreter through
    ``Py_Exit`` *without* unwinding the caller — the killed run left a
    ``python -m http.server`` bound to its port while the process was already
    gone. The sweep is idempotent, so running it here as well costs nothing.
    """
    for suffix, work_dir in list(_OPEN_WORK_DIRS.items()):
        try:
            _kill_strays(work_dir)
        except Exception:  # pragma: no cover - a dead process must not raise again
            pass
        _OPEN_WORK_DIRS.pop(suffix, None)


atexit.register(_sweep_on_exit)


def _stderr_tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


#: Text markers that make a broken agent invocation an endpoint fault rather
#: than a protocol one. ``pi`` reports HTTP 400 only as ``stopReason: error``,
#: so the message is the only place the status survives.
_ENDPOINT_MARKERS = ("HTTP 5", "HTTP 429", "HTTP 408", "Connection", "ECONN", "unreachable", "таймаут")


def _failure_of(text: str) -> str:
    """Taxonomy kind of an agent failure: endpoint down or protocol broken."""
    if any(marker in text for marker in _ENDPOINT_MARKERS):
        return "infra_failure"
    return "model_failure"


def _stop_error(events: list[dict[str, Any]]) -> str | None:
    """The message behind a terminal ``stopReason: error``, if there is one."""
    for event in reversed(events):
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        reason = message.get("stopReason") or message.get("stop_reason")
        if reason and reason != "error":
            return None
        detail = (
            message.get("errorMessage")
            or message.get("error")
            or (message.get("usage") or {}).get("error")
        )
        return str(detail) if detail else "агент остановился со stopReason=error"
    return None


def _finish(
    events: list[dict[str, Any]],
    *,
    started: float,
    stderr_path: Path,
    ok: bool,
    error: str | None = None,
    failure: str | None = None,
    secret: str | None = None,
    pin: PiBinary | None = None,
) -> AgentRun:
    """The one normalisation: every exit path goes through here.

    A timeout used to return bare events with zeroed usage and an empty tool
    log, so a run that had already measured spend looked free. Success, non-zero
    exit and timeout now agree on what an event stream means; events stay on the
    result for the report, and unknown usage stays ``None``.
    """
    usage = _usage_of(events)
    return AgentRun(
        ok=ok,
        text=_final_text(events),
        events=events,
        tool_calls=_tool_calls(events),
        duration_s=time.perf_counter() - started,
        stderr_tail=redact(_stderr_tail(stderr_path), secret),
        prompt_tokens_last=usage["prompt_tokens"],
        answer_tokens=usage["answer_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
        summary_tokens=usage["summary_tokens"],
        compactions=usage["compactions"],
        ttft_s=usage["ttft_s"],
        turns=usage["turns"],
        finish_reasons=usage["finish_reasons"],
        error=error,
        failure=failure,
        agent_version=pin.version if pin else None,
        agent_sha256=pin.sha256 if pin else None,
        agent_path=pin.path if pin else None,
    )


def _command(pin: PiBinary, model: str, prompt: str) -> list[str]:
    """The exact argv for the pinned CLI.

    The end-of-options separator is version-gated rather than unconditional:
    ``pi`` 0.80.10 has no ``--`` and dies on it with ``Unknown option: --``
    before the first model request, while 1.0.0 needs it so a prompt beginning
    with ``-`` is not parsed as a flag. Everything else in the list is the
    isolation contract and identical for every version the check accepts.
    """
    command = [
        pin.path,
        "--print",
        "--mode",
        "json",
        "--model",
        f"bench/{model}",
        "--tools",
        ",".join(AGENT_TOOLS),
        "--no-session",
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--offline",
        "--no-approve",
    ]
    if pin.parsed >= (parse_version(PI_SEPARATOR_VERSION) or (0, 0, 0)):
        command.append("--")
    command.append(prompt)
    return command


def _probe_in_sandbox(sandbox: isolation.Sandbox, host_env: dict[str, str]) -> None:
    """Refuse an image whose ``pi`` is not the pinned release.

    The host binary was verified already; the agent runs the *image's* copy, so
    the image must be verified too or the recorded pin would describe a binary
    that never executed. A missing or foreign binary is an infrastructure fault
    and the caller falls back to the host.
    """
    try:
        report = sandbox.run_command(["pi", "--version"], host_env=host_env, deadline=90, network="none")
    except isolation.InfraError:
        raise
    text = str(report.get("output") or "").strip()
    if report.get("status") != "ok":
        raise isolation.InfraError(
            f"в образе {sandbox.image} нет рабочего pi: {text.splitlines()[-1] if text else report.get('status')}"
        )
    found = parse_version(text)
    if found is None:
        raise isolation.InfraError(f"версия pi в образе не разобрана: {text!r}")
    want = parse_version(PI_VERSION) or (0, 0, 0)
    if found != want:
        raise isolation.InfraError(
            f"pi в образе {'.'.join(map(str, found))}, а закреплён {PI_VERSION}; "
            f"пересоберите образ ({install_command()})"
        )


def _run_isolated(
    sandbox: isolation.Sandbox,
    command: list[str],
    work_dir: Path,
    env: dict[str, str],
    deadline: float,
    errors: Any,
) -> tuple[str, dict[str, Any]]:
    """Run the agent in a container; return its stdout and the supervisor report.

    The streams go to files because the report keeps only a tail: the event log
    of a long run is far larger than the report's truncation.
    """
    stdout_path = work_dir.parent / f"{work_dir.name}.stdout.log"
    report = sandbox.run_command(
        command,
        host_env=env,
        deadline=deadline,
        stdout_path=stdout_path,
        stderr_path=Path(errors.name),
    )
    stdout = (
        stdout_path.read_text(encoding="utf-8", errors="replace")
        if stdout_path.is_file()
        else str(report.get("output") or "")
    )
    return stdout, report


def run(
    prompt: str,
    work_dir: Path,
    cfg: RunConfig,
    *,
    agent_dir: Path,
    timeout: float | None = None,
    sandbox: isolation.Sandbox | None = None,
) -> AgentRun:
    """Solve one task in ``work_dir`` with a fresh, isolated agent process.

    The process is *not* interactive: ``--print`` makes it run once and exit, and
    the JSON event stream is the whole record. The exit code decides whether the
    run is trusted at all; a non-zero exit keeps the events and the stderr tail
    so the report can show *why* rather than scoring an empty directory.

    A wrong binary is a returned failed run, not an exception: ``Runner`` turns
    ``error`` into the task's error while the report keeps the reason, so a
    mismatched pi cannot look like a model that wrote nothing.

    With ``sandbox`` the agent runs in a container that mounts ``work_dir`` and
    nothing else; ``agent_dir`` also reaches it, because it carries
    the endpoint. When the container cannot be established the run continues on
    the host and the sandbox object says so (``host_run`` + reason).
    """
    try:
        executable = ensure()
        pin = inspect(executable)
    except AgentVersionError as exc:
        probed = exc.probed
        return AgentRun(
            ok=False,
            error=str(exc),
            agent_version=probed.version if probed else None,
            agent_sha256=probed.sha256 if probed else None,
            agent_path=probed.path if probed else None,
        )
    started = time.perf_counter()
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["PI_OFFLINE"] = "1"
    env["PI_SKIP_VERSION_CHECK"] = "1"
    # A toolchain the benchmark can find but the agent cannot is a rigged task:
    # `go` usually lives outside PATH (see coding_go.GO_CANDIDATES), and an agent
    # that cannot run the project's own tests is measured on worse information
    # than the evaluator uses.
    env["PATH"] = _tool_path(env.get("PATH", ""))
    # The run must not depend on the machine's git/npm state.
    env.pop("PI_CODING_AGENT_SESSION_DIR", None)

    command = _command(pin, cfg.model, prompt)
    # One file per task: the parent is the task's own root directory, so a
    # fixed name would let two tasks overwrite each other's stderr.
    stderr_path = work_dir.parent / f"{work_dir.name}.stderr.log"
    deadline = timeout or float(cfg.task_wall_clock or cfg.request_wall_clock or 1800)
    _OPEN_WORK_DIRS[str(work_dir)] = work_dir
    isolated = sandbox
    try:
        if isolated is not None:
            try:
                isolated.ensure()
                _probe_in_sandbox(isolated, env)
            except isolation.InfraError as exc:
                isolated.fallback(f"агент не запущен в контейнере: {exc}")
                isolated = None
        with open(stderr_path, "wb") as errors:
            if isolated is not None:
                stdout, report = _run_isolated(isolated, command, work_dir, env, deadline, errors)
                events = _parse_events(stdout, cfg.api_key)
                if report.get("timed_out"):
                    return _finish(
                        events,
                        started=started,
                        stderr_path=stderr_path,
                        ok=False,
                        error=f"агент не уложился в {deadline:.0f} с",
                        failure="resource_exhausted",
                        secret=cfg.api_key,
                        pin=pin,
                    )
                if report.get("status") == "oom":
                    return _finish(
                        events,
                        started=started,
                        stderr_path=stderr_path,
                        ok=False,
                        error=f"контейнер убит по памяти ({sandbox.memory if sandbox else ''})",
                        failure="resource_exhausted",
                        secret=cfg.api_key,
                        pin=pin,
                    )
                if report.get("status") == "infra_error":
                    return _finish(
                        events,
                        started=started,
                        stderr_path=stderr_path,
                        ok=False,
                        error=f"песочница: {report.get('error')}",
                        failure="evaluator_failure",
                        secret=cfg.api_key,
                        pin=pin,
                    )
                returncode = int(report.get("container_exit_code") or 0)
            else:
                # ``Popen`` + own process group, not ``subprocess.run``: pi spawns a
                # shell per tool call, and a timeout must reach the whole tree.
                # ``run(timeout=)`` kills only pi itself, and the orphaned
                # ``bash → node → Chrome`` keeps running for hours — worse, it keeps
                # its hard-coded debugging port (a model's own check script binds one),
                # so the *next* task's identical script hangs on the occupied port and
                # also burns the full wall clock. One leak therefore poisons every
                # following task, not just its own.
                proc = subprocess.Popen(
                    command,
                    cwd=str(work_dir),
                    env=env,
                    # Never inherit stdin: pi treats a *pipe* on stdin as "more prompt
                    # to read" and blocks until EOF, so a run started from a shell
                    # pipeline hung for the whole wall clock without emitting a single
                    # event. The prompt is already in argv, so stdin must be empty.
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    **_spawn_kwargs(),
                )
                try:
                    _guard_process(proc, work_dir)
                except Exception:  # pragma: no cover - a job must never abort a run
                    pass
                try:
                    stdout_bytes, _ = proc.communicate(timeout=deadline)
                except subprocess.TimeoutExpired:
                    # Kill first, drain second: the children inherited the stdout
                    # pipe, so a second ``communicate`` without the kill just times
                    # out again on a pipe the survivors still hold open.
                    _kill_tree(proc)
                    try:
                        stdout_bytes, _ = proc.communicate(timeout=15)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        stdout_bytes = b""
                    stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
                    return _finish(
                        _parse_events(stdout, cfg.api_key),
                        started=started,
                        stderr_path=stderr_path,
                        ok=False,
                        error=f"агент не уложился в {deadline:.0f} с",
                        failure="resource_exhausted",
                        secret=cfg.api_key,
                        pin=pin,
                    )
                except BaseException:
                    _kill_tree(proc)
                    raise
                stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
                returncode = proc.returncode
    except OSError as exc:
        return _finish(
            [],
            started=started,
            stderr_path=stderr_path,
            ok=False,
            error=f"не удалось запустить агента: {exc}",
            failure="evaluator_failure",
            secret=cfg.api_key,
            pin=pin,
        )
    finally:
        # Whatever the outcome, the working copy must not keep serving: a
        # detached self-check server would hold its hard-coded port and hang the
        # next task's identical script for the full wall clock. A container run
        # already took its whole process tree down with it.
        if isolated is None:
            _kill_strays(work_dir)
        _OPEN_WORK_DIRS.pop(str(work_dir), None)

    events = _parse_events(stdout, cfg.api_key)
    result = _finish(
        events,
        started=started,
        stderr_path=stderr_path,
        ok=returncode == 0 and bool(events),
        secret=cfg.api_key,
        pin=pin,
    )
    if returncode != 0:
        tail = result.stderr_tail.strip().splitlines()
        if returncode < 0:
            # A negative code is a signal, not the agent's own verdict: the
            # kernel (OOM killer, watchdog, a human) ended it. Reading that as
            # ``model_failure`` blamed the model for a resource limit.
            result.error = f"агент убит сигналом {-returncode}" + (
                f": {tail[-1][:200]}" if tail else ""
            )
            result.failure = "candidate_resource_exhausted"
        else:
            result.error = f"агент завершился с кодом {returncode}" + (f": {tail[-1][:200]}" if tail else "")
            result.failure = _failure_of(result.error + "\n" + result.stderr_tail)
        result.ok = False
    elif not events:
        result.error = "агент не выдал ни одного события"
        result.failure = "infra_failure"
    else:
        # ``pi`` returns process exit 0 on an HTTP 400 and reports the fault only
        # as ``stopReason: error``: the process state alone used to read ``ok``.
        stop_error = _stop_error(events)
        if stop_error is not None:
            result.ok = False
            result.error = result.error or stop_error
            result.failure = _failure_of(stop_error + "\n" + result.stderr_tail)
    return result


def _parse_events(stdout: str, secret: str | None = None) -> list[dict[str, Any]]:
    """The event stream as records, with the endpoint key already scrubbed.

    Everything persisted (the diagnostic log, the tool rows, the final text) is
    derived from these records, so redacting once here is what keeps a tool that
    echoed ``$DAILYBENCH_API_KEY`` or ``models.json`` out of a run artefact.
    """
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(_redacted(json.loads(line), secret))
        except json.JSONDecodeError:
            continue
    return events


# --------------------------------------------------------------------------- #
# Working copies
# --------------------------------------------------------------------------- #


def prepare_work_dir(task: core.TaskInstance, parent: Path | None = None) -> Path:
    """A fresh working copy of the task's fixture, agent-ready.

    ``.pi/`` gets the pinned Three.js build (via a symlink, so a 2 MiB library is
    not copied per task) and a short ``AGENTS.md`` telling the agent where the
    library is and that the page is graded on disk. Both exist only when the task
    ships visual assets, so a Python task sees exactly its own repository.
    """
    root = Path(tempfile.mkdtemp(prefix=f"dailybench-agent-{task.id.lower()}-", dir=parent))
    for rel, content in (task.workspace or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    if task.oracle.get("assets"):
        _stage_assets(root, task)
    return root


def _stage_assets(root: Path, task: core.TaskInstance) -> None:
    from . import browser

    vendor = root / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    for name in ("three.module.js", "three.core.js"):
        source = browser.VENDOR / name
        target = vendor / name
        try:
            target.symlink_to(source)
        except OSError:  # pragma: no cover - filesystem without symlinks
            shutil.copyfile(source, target)
    (root / "AGENTS.md").write_text(_assets_readme(task), encoding="utf-8")


def _assets_readme(task: core.TaskInstance) -> str:
    """The one file the agent reads first: what is here and what is graded."""
    target = str(task.oracle.get("answer_file") or "")
    lines = [
        "# Как устроена эта задача",
        "",
        "Библиотека Three.js (r186) лежит рядом с рабочей копией:",
        "`vendor/three.module.js` — доступна, если нужна; подключать её не",
        "обязательно.",
        "Другие внешние ресурсы недоступны: сеть у агента есть (только до",
        "endpoint модели), у страницы — нет.",
        "",
        f"Результат проверяется по файлу `{target}` на диске.",
    ]

    return "\n".join(lines) + "\n"


def record_artefact(task: core.TaskInstance, transcript: core.Transcript) -> None:
    """Copy the produced file into the transcript, once, at the end of the task.

    The working copy lives in a temp directory: grading happens immediately, but
    a report is rendered later and the directory may be gone. The recorded text is
    also *exactly* what the evaluator saw, so a later reader cannot disagree with
    the score.
    """
    rel = task.oracle.get("answer_file")
    work_dir = transcript.extra.get("work_dir")
    if not rel or not work_dir:
        return
    path = Path(str(work_dir)) / str(rel)
    if not path.is_file():
        transcript.extra["artefact"] = {"file": str(rel), "text": "", "missing": True}
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    transcript.extra["artefact"] = {"file": str(rel), "text": text, "missing": not text.strip()}


def answer_text(task: core.TaskInstance, transcript: core.Transcript, fallback: str) -> str:
    """The artefact the run produced: the recorded copy, the file, else the reply.

    The disk is authoritative *only* for an agent run. A chat run also has a
    working copy, but it is the untouched fixture: the stub file is right there,
    and reading it would grade the *stub* instead of the printed answer. So the
    harness decides, not the file's existence.

    For an agent, an empty or missing file is the honest zero: neither the
    untouched stub nor a chat reference can stand in for the deliverable. A
    printed-but-unsaved file is not accepted either — the agent contract names
    the file, and the file is what gets graded.
    """
    rel = task.oracle.get("answer_file")
    recorded = transcript.extra.get("artefact")
    if rel and isinstance(recorded, dict) and recorded.get("text"):
        return str(recorded["text"])
    work_dir = transcript.extra.get("work_dir")
    if rel and work_dir and transcript.extra.get("harness") == "agent":
        path = Path(str(work_dir)) / str(rel)
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if text.strip():
                return text
        return ""
    return fallback


def prompt_for(task: core.TaskInstance) -> str:
    """The scenario prompt plus the grading contract, addressed to an agent."""
    from . import art, visual

    if task.category == "L":
        tail = AGENT_TARGET_PAGE.format(path=art.PAGE_FILE)
    elif task.category == "K":
        tail = AGENT_TARGET_PAGE.format(path=visual.PAGE_FILE)
    else:
        tail = AGENT_TARGET_FILE.format(path=str(task.oracle.get("answer_file") or ""))
    return f"{task.prompt.rstrip()}\n\n{AGENT_CONTRACT.format(target=tail)}"

