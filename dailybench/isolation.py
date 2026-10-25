"""Task-scoped container execution for the agentic categories (H01).

The runner used to start ``pi`` on the host: the agent inherited the machine's
environment, its network and — through nothing but the filesystem — the whole
benchmark repository, neighbouring working copies and hidden tests. The agent
is a candidate, so what it can *see* is part of the measurement.

This module puts the boundary where the trust boundary is. One task gets one
throwaway container:

* only the presented fixture is mounted, as ``/workspace``. The repository,
  ``task.oracle`` and the neighbouring tasks' directories are not mounted, so
  an ``ls /`` cannot find a hidden test that was not handed over;
* the container environment is an allowlist. Docker inherits nothing, so a host
  variable (``DAILYBENCH_API_KEY`` included) leaks only when it is named;
* network is the endpoint alone: the container is put on the default bridge and
  the only name handed to it is ``host.docker.internal``, mapped to the host
  gateway. A loopback ``base_url`` is rewritten to that name;
* cgroups bound cpu, memory (swap off) and pids, the rootfs is read-only and
  everything the command writes lands on tmpfs or the workspace mount;
* a deadline ends in ``SIGKILL`` of the container, and the container is removed
  in a ``finally`` whatever happened.

The user is the agent path of the artefact categories (C/K/L — whatever
``RunConfig.agent_categories`` names), because those are the tasks whose answer is
produced by executing a toolchain, not by talking.

The supervisor is ``scripts/sandbox_run.py``; it is loaded by path because it is
a script, not an installed module. When Docker is missing the caller falls back
to the host — and must mark it: an *image being present* is not evidence that a
container ran, only the report of a container that did.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import generation

#: Host docker alias the endpoint is reached by from inside the container.
ENDPOINT_HOST = "host.docker.internal"

#: The only host variables copied into the container. ``DAILYBENCH_API_KEY`` is
#: deliberately absent: the key already reaches the agent through its config
#: directory, and an allowlisted secret is one an ``env`` dump can leak.
ENV_ALLOW = frozenset(
    {
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TZ",
        "TERM",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "TMPDIR",
        "PATH",
        "PI_OFFLINE",
        "PI_SKIP_VERSION_CHECK",
        "PI_CODING_AGENT_DIR",
    }
)

#: Container PATH: the image's own, never the host's (host paths do not exist).
CONTAINER_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/go/bin"

_DEFAULT_DEADLINE = 1800.0

_SANDBOX_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sandbox_run.py"
_SUPERVISOR: Any = None


def supervisor() -> Any:
    """``scripts/sandbox_run`` as a module, loaded once by path."""
    global _SUPERVISOR
    if _SUPERVISOR is None:
        spec = importlib.util.spec_from_file_location("dailybench_sandbox_run", _SANDBOX_SCRIPT)
        if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
            raise RuntimeError(f"cannot load sandbox supervisor from {_SANDBOX_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SUPERVISOR = module
    return _SUPERVISOR


class InfraError(RuntimeError):
    """The sandbox could not be established; the caller may fall back to host."""


def default_image() -> str:
    return os.environ.get("DAILYBENCH_SANDBOX_IMAGE") or generation.DEFAULT_IMAGE


def default_runtime() -> str:
    return os.environ.get("DAILYBENCH_SANDBOX_RUNTIME") or "runc"


def ready(*, image: str | None = None, runtime: str | None = None) -> tuple[bool, str]:
    """``(usable, reason)`` — can a container be started at all right now?

    Checked before a run so the record can say *why* the host was used, and so
    that a missing image surfaces as an environment fact instead of task 1.
    """
    try:
        sandbox_run = supervisor()
    except (OSError, RuntimeError) as exc:  # pragma: no cover - broken checkout
        return False, f"супервизор песочницы не загрузился: {exc}"
    image = image or default_image()
    runtime = runtime or default_runtime()
    if not sandbox_run.shutil.which("docker"):
        return False, "docker CLI не найден"
    if not sandbox_run.docker_available():
        return False, "демон docker недоступен"
    try:
        sandbox_run.resolve_runtime(runtime)
    except sandbox_run.InfraError as exc:
        return False, str(exc)
    if not sandbox_run.SECCOMP_PROFILE.is_file():
        return False, f"профиль seccomp отсутствует: {sandbox_run.SECCOMP_PROFILE}"
    try:
        sandbox_run.image_digest(image)
    except sandbox_run.InfraError as exc:
        return False, str(exc).strip()
    return True, ""


def endpoint_url(base_url: str, *, host: str = ENDPOINT_HOST) -> tuple[str, bool]:
    """``(url, rewritten)`` — the URL the container can use.

    Only a loopback endpoint is rewritten: that is the case where the host name
    the model server listens on does not exist inside the container.
    """
    parts = urlsplit(base_url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return base_url, False
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    userinfo = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
    return urlunsplit((parts.scheme, userinfo + netloc, parts.path, parts.query, parts.fragment)), True


@dataclass
class Sandbox:
    """One task’s container boundary: fixture mount, allowlist, limits, cleanup.

    ``work_dir`` is the presented fixture and becomes ``/workspace`` writable;
    ``agent_dir`` is the harness config at ``/agent``. Both are the *only* host
    paths the container sees. The config mount is writable on purpose: ``pi``
    maintains ``auth.json`` and lock files there, and a read-only directory makes
    it fail before the first request.
    """

    work_dir: Path
    agent_dir: Path | None = None
    image: str = ""
    runtime: str = "runc"
    base_url: str = ""
    api_key: str | None = None
    include_api_key: bool = False
    cpus: str = "2"
    memory: str = "2g"
    pids_limit: int = 256
    deadline: float = _DEFAULT_DEADLINE
    ready_state: tuple[bool, str] | None = None
    fallback_reason: str = ""
    runs: list[dict[str, Any]] = field(default_factory=list)

    # -- the boundary ------------------------------------------------------ #

    def ensure(self) -> None:
        """Raise :class:`InfraError` unless a container can really be started."""
        if not self.image:
            self.image = default_image()
        if self.ready_state is None:
            self.ready_state = ready(image=self.image, runtime=self.runtime)
        usable, reason = self.ready_state
        if not usable:
            raise InfraError(reason or "песочница недоступна")

    def env(self, host_env: Mapping[str, str]) -> dict[str, str]:
        """The complete container environment: allowlist, plus the key if asked.

        Host values are copied only for names whose meaning survives the border;
        the path-bearing ones are rewritten, because a macOS ``HOME`` or
        ``TMPDIR`` does not exist inside the image.
        """
        env = {name: host_env[name] for name in ENV_ALLOW if name in host_env}
        env["PATH"] = CONTAINER_PATH
        env["HOME"] = "/home/sandbox"
        env["TMPDIR"] = "/tmp"
        if self.agent_dir is not None:
            env["PI_CODING_AGENT_DIR"] = "/agent"
        if self.include_api_key and self.api_key:
            env["DAILYBENCH_API_KEY"] = self.api_key
        return env

    def command(self, argv: list[str]) -> list[str]:
        """The argv with host paths replaced by the in-container names."""
        if not argv:
            return argv
        head = argv[0]
        if Path(head).name == "pi":
            return ["pi", *argv[1:]]
        if head == "python":
            return ["python3", *argv[1:]]
        return argv

    def fallback(self, reason: str) -> None:
        """Record that the host ran instead, and why."""
        self.fallback_reason = reason

    def record(self) -> dict[str, Any]:
        """What was *actually* applied, for the transcript and the report.

        ``container_run`` only when a container really started and reported;
        otherwise ``host_run`` with the reason. Naming an image is not evidence.
        """
        payload: dict[str, Any] = {
            "executor_status": "container_run" if self.runs else "host_run",
            "executor_reason": "" if self.runs else (
                self.fallback_reason or "контейнер не запускался"
            ),
            "image": self.image,
            "runtime": self.runtime,
        }
        if not self.runs:
            return payload
        last = self.runs[-1]
        payload.update(
            {
                "mounts": last.get("mounts") or [],
                "env": last.get("env") or [],
                "network": last.get("network") or "none",
                "endpoint": last.get("endpoint") or self.base_url,
                "endpoint_rewritten": bool(last.get("endpoint_rewritten")),
                "cgroups": last.get("cgroups") or {},
                "rootfs": last.get("rootfs") or "read_only",
                "capabilities": last.get("capabilities") or "dropped",
                "deadline_s": last.get("deadline_s"),
                "runs": len(self.runs),
                "statuses": [str(item.get("status")) for item in self.runs],
            }
        )
        return payload

    # -- one execution ----------------------------------------------------- #

    def run_command(
        self,
        argv: list[str],
        *,
        host_env: Mapping[str, str],
        deadline: float | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        network: str = "endpoint",
    ) -> dict[str, Any]:
        """Run *argv* in a fresh container and return the supervisor report.

        The container is created and removed by the supervisor inside one call,
        so a crash here cannot leave one behind. ``InfraError`` from the
        supervisor is re-raised as this module's own type.
        """
        self.ensure()
        sandbox_run = supervisor()
        url, rewritten = endpoint_url(self.base_url)
        docker_network = "bridge" if network == "endpoint" else network
        mounts = [(self.agent_dir, "/agent", False)] if self.agent_dir else []
        try:
            report = sandbox_run.run_sandbox(
                self.command(argv),
                image=self.image,
                runtime=self.runtime,
                network=docker_network,
                deadline=float(deadline or self.deadline),
                cpus=self.cpus,
                memory=self.memory,
                pids_limit=self.pids_limit,
                task_dir=self.work_dir,
                workspace_writable=True,
                extra_mounts=mounts,
                env=self.env(host_env),
                host_gateway=docker_network != "none",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except sandbox_run.InfraError as exc:
            raise InfraError(str(exc)) from exc
        record = dict(report)
        record["endpoint"] = url
        record["endpoint_rewritten"] = rewritten
        self.runs.append(record)
        return record
