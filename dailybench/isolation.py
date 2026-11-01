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
* network is the endpoint alone: a per-run ``docker network create --internal``
  plus a forwarder that dials only that endpoint. The agent joins the internal
  network and nothing else; HTTP is rewritten to ``http://dailybench-egress:8080``,
  HTTPS keeps the hostname (SNI) mapped to the forwarder. The host gateway is
  not handed to the agent;
* an https endpoint behind the corporate interception proxy needs the host CA.
  When the host has one — named by ``SSL_CERT_FILE``/``CURL_CA_BUNDLE`` or at
  the workstation default path — it is mounted read-only at
  :data:`CONTAINER_CA` and the container's TLS variables point at *that*
  container path. Without a bundle the run is not broken, only unverified: the
  record says so;
* cgroups bound cpu, memory (swap off) and pids, the rootfs is read-only and
  everything the command writes lands on tmpfs or the workspace mount;
* a deadline ends in ``SIGKILL`` of the container, and the container is removed
  in a ``finally`` whatever happened.

The user is the agent path of the artefact categories (C/K/L — whatever
``RunConfig.agent_categories`` names), because those are the tasks whose answer is
produced by executing a toolchain, not by talking.

The supervisor is ``scripts/sandbox_run.py``; it is loaded by path because it is
a script, not an installed module. An image that is not present is built once
per process (:func:`prepare`) instead of silently dropping the whole run onto
the host; when the build or the daemon fails the caller still falls back — and
must mark it: ``executor_unverified`` with the reason. An *image being present*
is not evidence that a container ran, only the report of a container that did.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import generation


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

#: Where a host CA bundle is exposed inside the container. A fixed path, so the
#: container environment never names a host directory.
CONTAINER_CA = "/etc/ssl/certs/dailybench-ca.pem"

#: Variables that make a TLS client trust :data:`CONTAINER_CA`: OpenSSL (and so
#: curl, python-ssl) honours ``SSL_CERT_FILE``, node — which ``pi`` runs on —
#: ``NODE_EXTRA_CA_CERTS``, and ``CURL_CA_BUNDLE`` covers the curl CLI.
CA_ENV = ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")

#: Host variables naming a bundle explicitly, in priority order.
CA_ENV_NAMES = ("SSL_CERT_FILE", "CURL_CA_BUNDLE")

#: Reason recorded when no bundle exists: the run continues, verification is
#: only weakened, so this is never an :class:`InfraError`.
CA_MISSING_REASON = (
    "корпоративный CA не найден: https-endpoint не проверяется из контейнера"
)

_DEFAULT_DEADLINE = 1800.0

#: Wall-clock ceiling for the one automatic image build of a run. A rebuild is
#: not retried afterwards: a failed build costs the same minutes on every task
#: and the run must not pay it once per task.
_DEFAULT_BUILD_TIMEOUT = 900.0

#: Marks the ``ready`` reason that a build can repair; the caller must be able to
#: tell "the image is absent" from "Docker cannot start anything at all".
MISSING_IMAGE_MARKER = "образ не найден локально"

#: ``(usable, reason)`` per image, remembered for the life of the process: the
#: first task builds, every later task reads the answer.
_BUILD_STATE: dict[str, tuple[bool, str]] = {}
_BUILD_LOCK = threading.Lock()

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
        # dataclasses look the class up in sys.modules while the body runs.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SUPERVISOR = module
    return _SUPERVISOR


class InfraError(RuntimeError):
    """The sandbox could not be established; the caller may fall back to host."""


def default_image() -> str:
    return os.environ.get("DAILYBENCH_SANDBOX_IMAGE") or generation.DEFAULT_IMAGE


def default_runtime() -> str:
    return os.environ.get("DAILYBENCH_SANDBOX_RUNTIME") or "runc"


def default_ca_bundle() -> Path:
    """The workstation bundle used when nothing names one.

    Resolved lazily (not at import) so a test or a run with a different
    ``HOME`` sees its own path.
    """
    return Path(os.path.expanduser("~/.config/omp/certs/sberdevices-ca-bundle.pem"))


def ca_bundle(host_env: Mapping[str, str] | None = None) -> Path | None:
    """The host CA bundle to hand into the container, or ``None``.

    A bundle named by ``SSL_CERT_FILE``/``CURL_CA_BUNDLE`` wins; otherwise the
    workstation default is used, because the interception proxy is a property
    of the host and not of a single run. A named file that does not exist is
    skipped rather than fatal: the container then trusts only the image's own
    store and the record carries :data:`CA_MISSING_REASON`.
    """
    source = os.environ if host_env is None else host_env
    for name in CA_ENV_NAMES:
        raw = source.get(name)
        if raw and Path(raw).is_file():
            return Path(raw).resolve()
    fallback = default_ca_bundle()
    return fallback.resolve() if fallback.is_file() else None


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
    if not image_present(image):
        return False, f"{MISSING_IMAGE_MARKER}: {image}"
    return True, ""


def image_present(image: str | None = None) -> bool:
    """Does *image* resolve locally to an immutable id?"""
    try:
        _, image_id = supervisor().image_digest(image or default_image())
    except Exception:  # pragma: no cover - daemon anomaly
        return False
    return bool(image_id)


def build_timeout() -> float:
    raw = os.environ.get("DAILYBENCH_SANDBOX_BUILD_TIMEOUT")
    try:
        return float(raw) if raw else _DEFAULT_BUILD_TIMEOUT
    except ValueError:
        return _DEFAULT_BUILD_TIMEOUT


def forget_build_state() -> None:
    """Drop the remembered answer (tests, or a manual rebuild in-process)."""
    with _BUILD_LOCK:
        _BUILD_STATE.clear()


def prepare(
    *, image: str | None = None, runtime: str | None = None
) -> tuple[bool, str, str]:
    """``(usable, reason, executor_status)`` — the run's one sandbox decision.

    A missing image is built here, once: the previous behaviour left a missing
    image to a single log line and silently ran the whole suite on the host,
    where the executor record then claimed ``host_run`` without saying that the
    sandbox was merely unbuilt. Now an absent image is built under a deadline
    and the *same* answer is reused by every following task — a failed build
    must not cost its minutes once per task.

    ``executor_status`` is ``""`` when a container can run, ``host_run`` for a
    host-level limitation (no Docker, dead daemon, unknown runtime) and
    ``executor_unverified`` when the sandbox was expected but could not be
    established — the difference between a deliberate host run and a lost
    boundary.
    """
    image = image or default_image()
    runtime = runtime or default_runtime()
    usable, reason = ready(image=image, runtime=runtime)
    if usable:
        return True, "", ""
    if not reason.startswith(MISSING_IMAGE_MARKER):
        return False, reason, "host_run"
    with _BUILD_LOCK:
        cached = _BUILD_STATE.get(image)
        if cached is not None:
            answer, cached_reason = cached
            return answer, cached_reason, "" if answer else "executor_unverified"
        try:
            supervisor().build_image(tag=image, timeout=build_timeout())
        except Exception as exc:  # any build fault (InfraError, OSError, timeout) is a reason
            state = (False, f"{MISSING_IMAGE_MARKER}: {image}; сборка не удалась: {exc}")
        else:
            state = ready(image=image, runtime=runtime)
            if not state[0]:
                state = (
                    False,
                    f"образ {image} собран, но песочница не поднялась: {state[1]}",
                )
        _BUILD_STATE[image] = state
        answer, cached_reason = state
        return answer, cached_reason, "" if answer else "executor_unverified"



def endpoint_url(base_url: str, *, host: str = "") -> tuple[str, bool]:
    """``(url, rewritten)`` — the URL the container dials through the forwarder.

    HTTP always becomes ``http://dailybench-egress:8080/...``: the agent sits on
    an internal network and must not learn the real host. HTTPS keeps the
    hostname so the certificate and SNI match; a loopback HTTPS endpoint uses
    ``host.docker.internal`` because an IP cannot be pinned with ``--add-host``.
    *host* overrides the HTTP alias (tests); the default is the egress name.
    """
    alias = host or supervisor().EGRESS_ALIAS
    return supervisor().rewrite_endpoint(base_url, alias=alias)


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
    #: What the record calls the executor when no container ran: ``host_run``
    #: for a deliberate or host-level fallback, ``executor_unverified`` when the
    #: sandbox was expected but could not be established.
    fallback_status: str = "host_run"
    runs: list[dict[str, Any]] = field(default_factory=list)
    #: Host CA bundle presented at :data:`CONTAINER_CA`; ``None`` (once
    #: resolved) means the host has none and TLS stays unverified in-container.
    _ca: Path | None = None
    _ca_resolved: bool = False

    # -- the boundary ------------------------------------------------------ #

    def ca(self, host_env: Mapping[str, str] | None = None) -> Path | None:
        """The bundle applied to the container, resolved once and remembered."""
        if not self._ca_resolved:
            self._ca = ca_bundle(host_env)
            self._ca_resolved = True
        return self._ca

    def ensure(self) -> None:
        """Raise :class:`InfraError` unless a container can really be started."""
        if not self.image:
            self.image = default_image()
        self.ca()
        if self.ready_state is None:
            usable, reason, status = prepare(image=self.image, runtime=self.runtime)
            self.ready_state = (usable, reason)
            if not usable and status:
                self.fallback_status = status
        usable, reason = self.ready_state
        if not usable:
            raise InfraError(reason or "песочница недоступна")

    def env(self, host_env: Mapping[str, str]) -> dict[str, str]:
        """The complete container environment: allowlist, plus the key if asked.

        Host values are copied only for names whose meaning survives the border;
        the path-bearing ones are rewritten, because a macOS ``HOME`` or
        ``TMPDIR`` does not exist inside the image. The CA variables point at
        the container mount, never at the host bundle.
        """
        env = {name: host_env[name] for name in ENV_ALLOW if name in host_env}
        env["PATH"] = CONTAINER_PATH
        env["HOME"] = "/home/sandbox"
        env["TMPDIR"] = "/tmp"
        if self.agent_dir is not None:
            env["PI_CODING_AGENT_DIR"] = "/agent"
        if self.include_api_key and self.api_key:
            env["DAILYBENCH_API_KEY"] = self.api_key
        if self.ca(host_env) is not None:
            env.update({name: CONTAINER_CA for name in CA_ENV})
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

    def fallback(self, reason: str, *, status: str | None = None) -> None:
        """Record that the host ran instead, and why."""
        self.fallback_reason = reason
        if status:
            self.fallback_status = status

    def record(self) -> dict[str, Any]:
        """What was *actually* applied, for the transcript and the report.

        ``container_run`` only when a container really started and reported;
        otherwise the recorded fallback status — never a bare claim that the
        image named in the record is what executed.
        """
        payload: dict[str, Any] = {
            "executor_status": "container_run" if self.runs else self.fallback_status,
            "executor_reason": "" if self.runs else (
                self.fallback_reason or "контейнер не запускался"
            ),
            "image": self.image,
            "runtime": self.runtime,
            # Absent CA is not a failure: the boundary still holds, only TLS
            # verification is weaker. It is stated instead of being assumed.
            "ca_bundle": str(self.ca() or ""),
            "ca_container_path": CONTAINER_CA if self._ca else "",
            "ca_mounted": self._ca is not None,
            "ca_reason": "" if self._ca else CA_MISSING_REASON,
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

        The host CA bundle is resolved from *host_env* and mounted read-only at
        :data:`CONTAINER_CA`; the environment then names that container path.
        """
        ca = self.ca(host_env)
        self.ensure()
        sandbox_run = supervisor()
        if network == "endpoint" and self.base_url:
            url, rewritten = endpoint_url(self.base_url)
        else:
            url, rewritten = self.base_url, False
        mounts = [(self.agent_dir, "/agent", False)] if self.agent_dir else []
        if ca is not None:
            mounts.append((ca, CONTAINER_CA, True))
        # ``endpoint`` is an internal network plus a forwarder. An explicit
        # bridge still gets the host gateway; the agent on endpoint does not.
        gateway = bool(network) and network not in ("none", "endpoint")
        try:
            report = sandbox_run.run_sandbox(
                self.command(argv),
                image=self.image,
                runtime=self.runtime,
                network=network,
                deadline=float(deadline or self.deadline),
                cpus=self.cpus,
                memory=self.memory,
                pids_limit=self.pids_limit,
                task_dir=self.work_dir,
                workspace_writable=True,
                extra_mounts=mounts,
                env=self.env(host_env),
                host_gateway=gateway,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                endpoint_url=self.base_url,
            )
        except sandbox_run.InfraError as exc:
            raise InfraError(str(exc)) from exc
        record = dict(report)
        record["endpoint"] = url
        record["endpoint_rewritten"] = rewritten
        self.runs.append(record)
        return record
