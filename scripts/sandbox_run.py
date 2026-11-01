#!/usr/bin/env python3
"""Supervisor for one sandboxed command (spec §6).

Builds the runner image from a digest-pinned base, executes the command under
the isolation contract (read-only rootfs, tmpfs-only /tmp, no capabilities,
no new privileges, seccomp allow-list, pids limit, cgroup v2 limits) and an
absolute deadline, then reports a machine-readable status.

The status vocabulary is deliberate: ``timeout`` and ``oom`` exhaust a
declared ceiling and map to ``resource_exhausted`` (same as ``agent.py``),
not ``evaluator_failure`` and not a zero that looks like the model.
``infra_error`` stays an evaluator fault (``evaluator_failure``). Docker
reserves exit code 125 for its own failures (daemon gone, image pulled away,
name collision) and 126/127 for exec failures *inside* the container; the
first is infra, the latter two are the candidate's.  The container must be
observed in a terminal state before any exit code is classified, because the
CLI's code is meaningless if the daemon died mid-attach.

Usage::

    python scripts/sandbox_run.py build [--digest sha256:...] [--tag TAG]
    python scripts/sandbox_run.py run --deadline 60 -- python solution.py

Exit codes: 0 ok, 1 candidate failure, 2 infrastructure, 3 timeout, 4 oom.
``125`` from ``docker run`` is reported as ``infra_error``/2, never as ``failed``.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent.parent
DOCKER_DIR = ROOT / "docker"
SECCOMP_PROFILE = DOCKER_DIR / "seccomp.json"
DEFAULT_DOCKERFILE = DOCKER_DIR / "Dockerfile.agent"
DEFAULT_BASE_REPO = "python:3.12-slim"
DEFAULT_BASE_DIGEST = (
    "sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47"
)
DEFAULT_TAG = "dailybench/runner-agent:local"
DEFAULT_SOURCE_DATE_EPOCH = 1700000000

#: In-memory tail kept for the report. The files get the full stream.
TAIL_LIMIT = 64 * 1024
#: Name the agent dials on the internal network. Not a host alias.
EGRESS_ALIAS = "dailybench-egress"
EGRESS_LISTEN = "0.0.0.0:8080"
EGRESS_PORT = 8080
EGRESS_SCRIPT = DOCKER_DIR / "egress_forward.py"
#: Loopback as seen from the forwarder is the forwarder itself. The host stub
#: is reached as this name, mapped to the daemon gateway on the forwarder only.
ENDPOINT_HOST = "host.docker.internal"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
DIGEST_REF_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INFRA = 2
EXIT_TIMEOUT = 3
EXIT_OOM = 4

#: Tag → image id for the life of the process. A rebuild mid-run cannot move it.
_IMAGE_PINS: dict[str, str] = {}
_PIN_LOCK = threading.Lock()
#: SBOM lines per image id. One ``dpkg-query`` container per id per process.
_SBOM_CACHE: dict[str, list[str]] = {}
_SBOM_LOCK = threading.Lock()


class InfraError(RuntimeError):
    """The sandbox itself could not be established."""


# --------------------------------------------------------------------------
# docker plumbing


def _docker() -> str:
    exe = shutil.which("docker")
    if not exe:
        raise InfraError("docker CLI not found in PATH")
    return exe


def _docker_env() -> dict[str, str]:
    env = dict(os.environ)
    # Deterministic docker CLI behaviour; the daemon socket comes from env.
    env.setdefault("DOCKER_CLI_HINTS", "false")
    return env


def _call(argv: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=_docker_env(),
    )


def docker_available() -> bool:
    """True when a Docker daemon answers (used by tests for skipping)."""
    if not shutil.which("docker"):
        return False
    try:
        return _call(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def registered_runtimes() -> set[str]:
    proc = _call(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=20)
    if proc.returncode != 0:
        raise InfraError(f"docker info failed: {proc.stderr.strip()}")
    try:
        return set(json.loads(proc.stdout.strip() or "{}"))
    except json.JSONDecodeError as exc:  # pragma: no cover - daemon anomaly
        raise InfraError(f"cannot parse docker runtimes: {exc}") from exc


def resolve_runtime(requested: str) -> None:
    if requested == "runc":
        return
    if requested == "runsc":
        available = registered_runtimes()
        if "runsc" not in available:
            raise InfraError(
                "runtime 'runsc' is not registered with the Docker daemon; "
                "install gVisor and run `sudo runsc install`, then restart docker "
                "(see docker/README.md). Registered runtimes: "
                + ", ".join(sorted(available) or ["<none>"])
            )
        return
    available = registered_runtimes()
    if requested not in available:
        raise InfraError(
            f"runtime {requested!r} is not registered; available: "
            + ", ".join(sorted(available) or ["<none>"])
        )


def image_digest(image: str) -> tuple[str, str]:
    """Return ``(repo_digest, image_id)``; either may be empty."""
    # ``{{json ...}}`` is the only template form that survives every Docker CLI:
    # a buildkit image with attestations hands ``.RepoDigests`` over as
    # ``[]interface{}``, which ``{{join ...}}`` refuses to type-assert.
    proc = _call(
        [
            "docker",
            "image",
            "inspect",
            image,
            "--format",
            "{{json .RepoDigests}}||{{.Id}}",
        ],
        timeout=30,
    )
    if proc.returncode != 0:
        raise InfraError(f"image {image!r} not available: {proc.stderr.strip()}")
    raw, _, image_id = proc.stdout.strip().partition("||")
    try:
        digests = json.loads(raw) if raw else []
    except ValueError:
        digests = [part for part in raw.split(",") if part]
    return (digests[0] if digests else ""), image_id


def forget_image_pins() -> None:
    """Drop the per-process image-id pin (tests, or a deliberate rebuild)."""
    with _PIN_LOCK:
        _IMAGE_PINS.clear()


def forget_sbom_cache() -> None:
    """Drop the per-process SBOM cache."""
    with _SBOM_LOCK:
        _SBOM_CACHE.clear()


def _require_pinned(image: str, allow_latest: bool) -> None:
    if "@sha256:" in image:
        return
    tag = image.rsplit("/", 1)[-1]
    tag = tag.split(":", 1)[1] if ":" in tag else ""
    # ``latest`` is refused even with ``--allow-latest``: a floating tag must
    # not become the id this process pins. The flag only admits a bare name.
    if tag == "latest" or (not tag and not allow_latest):
        raise InfraError(
            f"image {image!r} uses the forbidden 'latest' tag; pass a digest "
            "reference or a named tag"
        )


def resolve_image(image: str, *, allow_latest: bool = False) -> str:
    """The immutable reference the container is started by.

    A ``name@sha256:<64 hex>`` reference and a bare image id are used as is.
    A tag is resolved once per process to ``docker image inspect --format
    {{.Id}}``; a later rebuild cannot change the id this process already
    pinned. An unresolvable tag is refused. ``latest`` stays refused.
    """
    if DIGEST_REF_RE.match(image) or IMAGE_ID_RE.match(image):
        return image
    _require_pinned(image, allow_latest)
    with _PIN_LOCK:
        cached = _IMAGE_PINS.get(image)
        if cached:
            return cached
    _digest, image_id = image_digest(image)
    if not IMAGE_ID_RE.match(image_id):
        raise InfraError(
            f"image {image!r} has no local id; pass name@sha256:<64 hex> or build the tag"
        )
    with _PIN_LOCK:
        return _IMAGE_PINS.setdefault(image, image_id)


def source_date_epoch() -> int:
    """``SOURCE_DATE_EPOCH`` as an int, or *InfraError* when malformed.

    Called before anything is executed so a typo cannot surface as a candidate
    failure after the command already ran.
    """
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None or not raw.strip():
        return DEFAULT_SOURCE_DATE_EPOCH
    try:
        return int(raw)
    except ValueError:
        raise InfraError(
            f"SOURCE_DATE_EPOCH must be an integer epoch, got {raw!r}"
        ) from None


def _mount_spec(host: Path, target: str, *, readonly: bool) -> str:
    """Return a ``--mount`` bind spec for *host*.

    ``--mount`` is used instead of ``-v`` because the CSV form keeps ``:``
    legal inside a POSIX path (``-v`` would read it as a field separator).  A
    comma is still a separator, so it is refused explicitly rather than
    silently mounting a different directory.
    """
    text = str(host)
    if os.name == "nt":
        # Docker's ``--mount`` takes a container-side style path; a drive letter
        # with backslashes is read as one opaque token otherwise.
        text = text.replace("\\", "/")
    if "," in text:
        raise InfraError(
            f"path {text!r} contains ',' which --mount uses as a field "
            "separator; move the directory or bind it manually"
        )
    spec = f"type=bind,src={text},dst={target}"
    return f"{spec},readonly" if readonly else spec


def _image_sbom_uncached(image: str, *, timeout: float) -> list[str]:
    try:
        proc = _call(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "",
                image,
                "dpkg-query",
                "-W",
                "-f=${Package}\t${Version}\n",
            ],
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())


def image_sbom(image: str, *, timeout: float = 120) -> list[str]:
    """Installed packages of ``image`` as ``package<TAB>version`` lines.

    Cached per image id for the life of the process: a command must not pay
    for another ``dpkg-query`` container when the id has not changed.
    """
    key = image if IMAGE_ID_RE.match(image) else ""
    if not key:
        try:
            pinned = resolve_image(image, allow_latest=True)
        except InfraError:
            pinned = image
        key = pinned if IMAGE_ID_RE.match(pinned) else image
    with _SBOM_LOCK:
        cached = _SBOM_CACHE.get(key)
        if cached is not None:
            return list(cached)
    lines = _image_sbom_uncached(key if IMAGE_ID_RE.match(key) else image, timeout=timeout)
    with _SBOM_LOCK:
        stored = _SBOM_CACHE.setdefault(key, lines)
        return list(stored)


# --------------------------------------------------------------------------
# endpoint egress


def rewrite_endpoint(
    base_url: str, *, alias: str = EGRESS_ALIAS, port: int = EGRESS_PORT
) -> tuple[str, bool]:
    """``(url, rewritten)`` — what the agent dials through the forwarder.

    HTTP always becomes ``http://<alias>:<port>/...``: the agent is on an
    internal network and must not learn the real host. HTTPS keeps the
    hostname so the certificate and SNI match; the supervisor maps that name
    to the forwarder. A loopback HTTPS endpoint uses ``host.docker.internal``
    because an IP cannot be pinned with ``--add-host``.
    """
    parts = urlsplit(base_url)
    if not parts.hostname or not parts.scheme:
        return base_url, False
    userinfo = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
    loopback = parts.hostname in LOOPBACK_HOSTS
    if parts.scheme == "https":
        host = ENDPOINT_HOST if loopback else parts.hostname
        netloc = f"{userinfo}{host}:{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)), True
    netloc = f"{userinfo}{alias}:{port}"
    return urlunsplit(("http", netloc, parts.path, parts.query, parts.fragment)), True


def _forward_target(base_url: str) -> tuple[str, str]:
    """``(host:port the forwarder dials, SNI hostname or "")``."""
    parts = urlsplit(base_url)
    if not parts.hostname or not parts.scheme:
        return "", ""
    loopback = parts.hostname in LOOPBACK_HOSTS
    if parts.scheme == "https":
        port = parts.port or 443
        sni = ENDPOINT_HOST if loopback else parts.hostname
    else:
        port = parts.port or 80
        sni = ""
    host = ENDPOINT_HOST if loopback else parts.hostname
    return f"{host}:{port}", sni


@dataclass
class EndpointPlan:
    """Argv for one endpoint-only run. Inspected by tests; executed by the supervisor."""

    network: str
    forwarder_name: str
    rewritten_url: str
    target: str
    listen: str
    sni_host: str
    network_create: list[str]
    forwarder_argv: list[str]
    bridge_connect: list[str]
    agent_network_argv: list[str]


def endpoint_plan(
    base_url: str,
    *,
    run_id: str,
    docker: str = "docker",
    image: str = "",
    script: Path | None = None,
) -> EndpointPlan:
    """Internal network, a forwarder on that network and the default bridge, agent on the internal network only."""
    script = Path(script) if script is not None else EGRESS_SCRIPT
    network = f"dailybench-{run_id}"
    forwarder_name = f"dailybench-egress-{run_id}"
    rewritten, _changed = rewrite_endpoint(base_url) if base_url else ("", False)
    target, sni = _forward_target(base_url) if base_url else ("", "")
    network_create = [docker, "network", "create", "--internal", network]
    forwarder_argv: list[str] = []
    bridge_connect: list[str] = []
    if target:
        forwarder_argv = [
            docker,
            "run",
            "-d",
            "--name",
            forwarder_name,
            "--network",
            network,
            "--network-alias",
            EGRESS_ALIAS,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--security-opt",
            f"seccomp={SECCOMP_PROFILE}",
            "--pids-limit",
            "64",
            "--cpus",
            "0.5",
            "--memory",
            "128m",
            "--memory-swap",
            "128m",
            "--user",
            "1000:1000",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m",
            "--mount",
            _mount_spec(script, "/egress_forward.py", readonly=True),
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
        ]
        if target.startswith(f"{ENDPOINT_HOST}:"):
            # Only the forwarder may reach the host. The agent never gets this.
            forwarder_argv += ["--add-host", f"{ENDPOINT_HOST}:host-gateway"]
        forwarder_argv += [
            "--entrypoint",
            "",
            image,
            "python3",
            "/egress_forward.py",
            "--listen",
            EGRESS_LISTEN,
            "--target",
            target,
        ]
        bridge_connect = [docker, "network", "connect", "bridge", forwarder_name]
    return EndpointPlan(
        network=network,
        forwarder_name=forwarder_name,
        rewritten_url=rewritten,
        target=target,
        listen=EGRESS_LISTEN,
        sni_host=sni,
        network_create=network_create,
        forwarder_argv=forwarder_argv,
        bridge_connect=bridge_connect,
        agent_network_argv=["--network", network],
    )


def agent_endpoint_argv(plan: EndpointPlan, forwarder_ip: str) -> list[str]:
    """Agent joins only the internal network. HTTPS pins the SNI name to the forwarder."""
    argv = ["--network", plan.network]
    if plan.sni_host:
        if not forwarder_ip:
            raise InfraError("forwarder has no address for SNI pinning")
        argv += ["--add-host", f"{plan.sni_host}:{forwarder_ip}"]
    return argv


@dataclass
class _EndpointSession:
    network: str
    forwarder_name: str
    extra_hosts: list[str]
    target: str


def _container_ip(container: str, network: str) -> str:
    for _ in range(25):
        proc = _call(
            ["docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", container],
            timeout=20,
        )
        if proc.returncode == 0:
            try:
                nets = json.loads(proc.stdout.strip() or "{}")
            except json.JSONDecodeError:
                nets = {}
            info = nets.get(network) if isinstance(nets, dict) else None
            if isinstance(info, dict) and info.get("IPAddress"):
                return str(info["IPAddress"])
        time.sleep(0.1)
    raise InfraError(f"forwarder {container} has no address on {network}")


def _open_endpoint(docker: str, image: str, base_url: str) -> _EndpointSession:
    """Create the internal network and the forwarder. Cleans up if setup fails."""
    plan = endpoint_plan(base_url, run_id=secrets.token_hex(5), docker=docker, image=image)
    created = False
    started = False
    try:
        proc = _call(plan.network_create, timeout=30)
        if proc.returncode != 0:
            raise InfraError(
                "docker network create failed: " + (proc.stderr or proc.stdout).strip()
            )
        created = True
        extra_hosts: list[str] = []
        if plan.forwarder_argv:
            proc = _call(plan.forwarder_argv, timeout=60)
            if proc.returncode != 0:
                raise InfraError(
                    "egress forwarder failed: " + (proc.stderr or proc.stdout).strip()
                )
            started = True
            if plan.bridge_connect:
                proc = _call(plan.bridge_connect, timeout=30)
                if proc.returncode != 0:
                    raise InfraError(
                        "cannot attach forwarder to the bridge: "
                        + (proc.stderr or proc.stdout).strip()
                    )
            if plan.sni_host:
                ip = _container_ip(plan.forwarder_name, plan.network)
                joined = agent_endpoint_argv(plan, ip)
                extra_hosts = [joined[joined.index("--add-host") + 1]]
        return _EndpointSession(
            network=plan.network,
            forwarder_name=plan.forwarder_name if started else "",
            extra_hosts=extra_hosts,
            target=plan.target,
        )
    except Exception:
        _cleanup(
            plan.forwarder_name if started else None,
            network=plan.network if created else None,
        )
        raise


# --------------------------------------------------------------------------
# build


def build_image(
    digest: str = DEFAULT_BASE_DIGEST,
    *,
    tag: str = DEFAULT_TAG,
    dockerfile: Path = DEFAULT_DOCKERFILE,
    base_repo: str = DEFAULT_BASE_REPO,
    timeout: float = 1800,
) -> dict[str, object]:
    """Build a runner image from ``base_repo@digest`` and return its identity."""
    if DIGEST_RE.match(digest):
        base = f"{base_repo}@{digest}"
    elif "@sha256:" in digest:
        base = digest
    else:
        raise InfraError(f"base digest must look like sha256:<64 hex>: {digest!r}")
    docker = _docker()
    epoch = source_date_epoch()
    proc = _call(
        [
            docker,
            "build",
            "-f",
            str(dockerfile),
            "--build-arg",
            f"PYTHON_IMAGE={base}",
            "--build-arg",
            f"SOURCE_DATE_EPOCH={epoch}",
            "-t",
            tag,
            str(dockerfile.parent),
        ],
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise InfraError("docker build failed: " + " | ".join(tail))
    repo_digest, image_id = image_digest(tag)
    return {
        "tag": tag,
        "base": base,
        "image_digest": repo_digest,
        "image_id": image_id,
        "source_date_epoch": epoch,
        "sbom": image_sbom(image_id or tag),
    }


# --------------------------------------------------------------------------
# run


def _mount_pairs(
    task_dir: Path,
    *,
    workspace_writable: bool,
    extra_mounts: Sequence[tuple[Path, str, bool]],
    write_dir: Path | None,
) -> list[tuple[Path, str, bool]]:
    """``(host, target, readonly)`` for every bind, with duplicate targets refused.

    A duplicate target silently makes the *last* mount win, which turns a
    read-only fixture into a writable one without the argv showing it.
    """
    pairs: list[tuple[Path, str, bool]] = [(task_dir, "/workspace", not workspace_writable)]
    for host, target, readonly in extra_mounts:
        if not target.startswith("/"):
            raise InfraError(f"mount target must be absolute: {target!r}")
        pairs.append((host, target, readonly))
    if write_dir is not None:
        pairs.append((write_dir, "/out", False))
    seen: set[str] = set()
    for _host, target, _readonly in pairs:
        if target in seen:
            raise InfraError(f"mount target {target!r} listed twice")
        seen.add(target)
    return pairs


def _env_argv(env: Mapping[str, str] | None) -> list[str]:
    """``--env`` pairs for an explicit allowlist.

    Docker inherits nothing from the caller, so this list *is* the container
    environment: a host variable that is not named here cannot leak in.
    """
    argv: list[str] = []
    for name in sorted(env or {}):
        if not ENV_NAME_RE.match(name):
            raise InfraError(f"invalid environment name: {name!r}")
        argv += ["--env", f"{name}={env[name]}"]
    return argv


def _run_argv(
    *,
    docker: str,
    container: str,
    image: str,
    command: list[str],
    runtime: str,
    network: str,
    task_dir: Path,
    write_dir: Path | None,
    cpus: str,
    memory: str,
    pids_limit: int,
    seccomp: Path,
    workspace_writable: bool = False,
    extra_mounts: Sequence[tuple[Path, str, bool]] = (),
    env: Mapping[str, str] | None = None,
    host_gateway: bool = False,
    extra_hosts: Sequence[str] = (),
) -> list[str]:
    argv = [
        docker,
        "run",
        "--name",
        container,
        "--runtime",
        runtime,
        # PID 1 reaper: without it, orphans of a killed child stay zombies.
        "--init",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
        # Docker's implicit /dev/shm is already a writable 64 MiB tmpfs; mount
        # it explicitly so the write surface is in the argv and not a daemon
        # default that can drift.
        "--tmpfs",
        "/dev/shm:rw,nosuid,nodev,noexec,size=64m,mode=1777",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--security-opt",
        f"seccomp={seccomp}",
        "--pids-limit",
        str(pids_limit),
        "--cpus",
        cpus,
        "--memory",
        memory,
        # memory-swap == memory disables swap (cgroup v2 memory.swap.max=0).
        "--memory-swap",
        memory,
        "--user",
        "1000:1000",
        "--workdir",
        "/workspace",
    ]
    for host, target, readonly in _mount_pairs(
        task_dir,
        workspace_writable=workspace_writable,
        extra_mounts=extra_mounts,
        write_dir=write_dir,
    ):
        argv += ["--mount", _mount_spec(host, target, readonly=readonly)]
    argv += ["--network", network or "none"]
    if host_gateway:
        # Explicit bridge only. Endpoint mode must not hand the agent the gateway:
        # that alias reaches every host port.
        argv += ["--add-host", f"{ENDPOINT_HOST}:host-gateway"]
    for item in extra_hosts:
        argv += ["--add-host", item]
    argv += _env_argv(env)
    if write_dir is not None:
        argv += ["--env", "SANDBOX_OUT=/out"]
    argv += ["--entrypoint", "", image, *command]
    return argv


def _inspect_state(container: str) -> dict[str, object] | None:
    """``.State`` of *container*, or ``None`` when it cannot be read.

    ``None`` means the daemon answered neither the run nor the follow-up
    inspect: the CLI's exit code alone must not be trusted then.
    """
    proc = _call(
        ["docker", "inspect", "--format", "{{json .State}}", container],
        timeout=20,
    )
    if proc.returncode != 0:
        return None
    try:
        state = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:  # pragma: no cover - daemon anomaly
        return None
    return state if isinstance(state, dict) else None


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the docker CLI and its children. ``taskkill /T /F`` on Windows."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _cleanup(*names: str | None, network: str | None = None) -> None:
    """``docker rm`` even when ``docker kill`` fails. Never raises."""
    for name in names:
        if not name:
            continue
        try:
            _call(["docker", "kill", "--signal", "KILL", name], timeout=20)
        except Exception:
            pass
        try:
            _call(["docker", "rm", "--force", "--volumes", name], timeout=30)
        except Exception:
            pass
    if network:
        try:
            _call(["docker", "network", "rm", network], timeout=30)
        except Exception:
            pass


class _BoundedTail:
    """A byte tail that never grows past ``limit``. ``peak`` is the high-water mark."""

    def __init__(self, limit: int = TAIL_LIMIT) -> None:
        self.limit = limit
        self._buf = bytearray()
        self.peak = 0

    def extend(self, data: bytes) -> None:
        if not data:
            return
        if len(data) >= self.limit:
            self._buf[:] = data[-self.limit :]
        else:
            self._buf.extend(data)
            if len(self._buf) > self.limit:
                del self._buf[: -self.limit]
        self.peak = max(self.peak, len(self._buf))

    def text(self) -> str:
        return bytes(self._buf).decode("utf-8", errors="replace")


def _pump(stream: object, path: Path | None, tail: _BoundedTail) -> None:
    if stream is None or not hasattr(stream, "read"):
        return
    handle = None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", encoding="utf-8", newline="")
        while True:
            chunk = stream.read(16 * 1024)  # type: ignore[union-attr]
            if not chunk:
                break
            if isinstance(chunk, str):  # pragma: no cover - pipes are binary
                chunk = chunk.encode("utf-8", errors="replace")
            tail.extend(chunk)
            if handle is not None:
                handle.write(decoder.decode(chunk))
        if handle is not None:
            handle.write(decoder.decode(b"", final=True))
    except OSError:
        pass
    finally:
        if handle is not None:
            handle.close()


def capture_streams(
    proc: subprocess.Popen,
    stdout_path: Path | None,
    stderr_path: Path | None,
    *,
    readers: list[threading.Thread] | None = None,
) -> tuple[_BoundedTail, _BoundedTail]:
    """Drain both pipes to *paths* (full) and bounded tails.

    Joins the pump threads unless *readers* is given: the caller then owns
    the deadline and joins after ``wait``. Does not wait on *proc* itself.
    """
    out = _BoundedTail()
    err = _BoundedTail()
    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, stdout_path, out), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, stderr_path, err), daemon=True),
    ]
    for thread in threads:
        thread.start()
    if readers is None:
        for thread in threads:
            thread.join()
    else:
        readers.extend(threads)
    return out, err


def run_sandbox(
    command: list[str],
    *,
    image: str = DEFAULT_TAG,
    runtime: str = "runc",
    network: str = "none",
    deadline: float = 600.0,
    cpus: str = "2",
    memory: str = "2g",
    pids_limit: int = 256,
    task_dir: Path = ROOT,
    write_dir: Path | None = None,
    allow_latest: bool = False,
    seccomp: Path = SECCOMP_PROFILE,
    workspace_writable: bool = False,
    extra_mounts: Sequence[tuple[Path, str, bool]] = (),
    env: Mapping[str, str] | None = None,
    host_gateway: bool = False,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    endpoint_url: str = "",
) -> dict[str, object]:
    """Run one command inside the sandbox and return the status report.

    The caller controls the whole exposure surface: ``task_dir`` is the fixture
    and is read-only unless the command must write into it, ``extra_mounts``
    adds the few other host paths allowed (as ``(host, container target,
    read-only)``), and ``env`` is the complete container environment — Docker
    inherits nothing, so a host secret that is not listed cannot leak in.

    ``network="endpoint"`` builds a per-run internal network and a forwarder
    that dials only ``endpoint_url``. The agent joins that network and nothing
    else. ``stdout_path``/``stderr_path`` receive the full streams; the report
    keeps a 64 KiB tail of both.
    """
    if not command:
        raise InfraError("empty command")
    if deadline <= 0:
        raise InfraError("deadline must be positive")
    docker = _docker()
    if not docker_available():
        raise InfraError("docker daemon is not reachable")
    epoch = source_date_epoch()
    pinned = resolve_image(image, allow_latest=allow_latest)
    resolve_runtime(runtime)
    if not seccomp.is_file():
        raise InfraError(f"seccomp profile missing: {seccomp}")
    task_dir = task_dir.resolve()
    if not task_dir.is_dir():
        raise InfraError(f"task directory missing: {task_dir}")
    if write_dir is not None:
        write_dir = write_dir.resolve()
        write_dir.mkdir(parents=True, exist_ok=True)
    extras: list[tuple[Path, str, bool]] = []
    for host, target, readonly in extra_mounts:
        host = Path(host).resolve()
        # A file mount is legitimate here (the host CA bundle); only absence is
        # an error, and it is still caught before the image lookup.
        if not host.exists():
            raise InfraError(f"mounted path missing: {host}")
        extras.append((host, target, readonly))
    mounts = _mount_pairs(
        task_dir,
        workspace_writable=workspace_writable,
        extra_mounts=extras,
        write_dir=write_dir,
    )

    if IMAGE_ID_RE.match(pinned):
        repo_digest, image_id = "", pinned
    else:
        repo_digest, image_id = image_digest(pinned)
        image_id = image_id or ""

    endpoint: _EndpointSession | None = None
    docker_network = network or "none"
    extra_hosts: list[str] = []
    agent_gateway = host_gateway
    report_network = network
    if network == "endpoint":
        endpoint = _open_endpoint(docker, pinned, endpoint_url)
        docker_network = endpoint.network
        extra_hosts = list(endpoint.extra_hosts)
        agent_gateway = False
        report_network = "endpoint"

    container = f"dailybench-sandbox-{secrets.token_hex(5)}"
    started = time.monotonic()
    timed_out = False
    oom_killed = False
    exit_code: int | None = None
    output = ""
    errors = ""
    infra_reason = ""
    proc: subprocess.Popen | None = None
    try:
        argv = _run_argv(
            docker=docker,
            container=container,
            image=pinned,
            command=command,
            runtime=runtime,
            network=docker_network,
            task_dir=task_dir,
            write_dir=write_dir,
            cpus=cpus,
            memory=memory,
            pids_limit=pids_limit,
            seccomp=seccomp.resolve(),
            workspace_writable=workspace_writable,
            extra_mounts=extras,
            env=env,
            host_gateway=agent_gateway,
            extra_hosts=extra_hosts,
        )
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_docker_env(),
            start_new_session=True,
        )
        readers: list[threading.Thread] = []
        out_tail, err_tail = capture_streams(
            proc, stdout_path, stderr_path, readers=readers
        )
        try:
            proc.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                _call(["docker", "kill", "--signal", "KILL", container], timeout=30)
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        for reader in readers:
            reader.join(timeout=15)
        output = out_tail.text()
        errors = err_tail.text()
        exit_code = proc.returncode
        state = _inspect_state(container)
        if state is not None and state.get("Running") is True:
            # No terminal state means the CLI code cannot be classified yet.
            try:
                _call(["docker", "kill", "--signal", "KILL", container], timeout=20)
            except Exception:
                pass
            _kill_tree(proc)
            state = _inspect_state(container)
        oom_killed = bool(state and state.get("OOMKilled") is True)
        if state is None:
            infra_reason = "docker inspect failed after docker run; no container state"
            if exit_code == 125:
                infra_reason += "; docker run itself exited 125"
        elif state.get("Running") is True:
            infra_reason = "container never reached a terminal state"
        elif exit_code == 125:
            infra_reason = (
                "docker run failed before the command started (exit 125: CLI or daemon)"
            )
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
        _cleanup(container)
        if endpoint is not None:
            _cleanup(endpoint.forwarder_name or None, network=endpoint.network)

    duration = time.monotonic() - started
    if timed_out:
        status, code = "timeout", EXIT_TIMEOUT
    elif oom_killed:
        status, code = "oom", EXIT_OOM
    elif infra_reason:
        status, code = "infra_error", EXIT_INFRA
    elif exit_code == 0:
        status, code = "ok", EXIT_OK
    else:
        status, code = "failed", EXIT_FAILED

    combined = (output or "") + ("\n" + errors if errors else "")
    report: dict[str, object] = {
        "status": status,
        "exit_code": code,
        "error": infra_reason or None,
        "container_exit_code": exit_code,
        "runtime": runtime,
        "runtime_verified": runtime == "runc" or runtime in registered_runtimes(),
        "image": image,
        "image_digest": repo_digest,
        "image_id": image_id,
        "image_pinned": pinned,
        "source_date_epoch": epoch,
        "sbom": image_sbom(image_id or pinned),
        "command": command,
        "network": report_network,
        "deadline_s": deadline,
        "duration_s": round(duration, 3),
        "timed_out": timed_out,
        "oom_killed": oom_killed,
        "task_dir": str(task_dir),
        "write_dir": str(write_dir) if write_dir else None,
        "container": container,
        "mounts": [
            {"src": str(host), "dst": target, "readonly": readonly}
            for host, target, readonly in mounts
        ],
        "workspace_writable": workspace_writable,
        "env": sorted(env or {}),
        "host_gateway": agent_gateway,
        "cgroups": {"cpus": cpus, "memory": memory, "pids_limit": pids_limit},
        "rootfs": "read_only",
        "capabilities": "dropped",
        "cpus": cpus,
        "memory": memory,
        "pids_limit": pids_limit,
        "output": combined[-TAIL_LIMIT:],
    }
    if endpoint is not None:
        report["network_name"] = endpoint.network
        report["endpoint_target"] = endpoint.target
        report["forwarder"] = endpoint.forwarder_name
    return report


# --------------------------------------------------------------------------
# CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run the LLMCrewBench sandbox image.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    build = sub.add_parser("build", help="build the runner image")
    build.add_argument("--digest", default=DEFAULT_BASE_DIGEST, help="sha256 digest of the base")
    build.add_argument("--base-repo", default=DEFAULT_BASE_REPO)
    build.add_argument("--tag", default=DEFAULT_TAG)
    build.add_argument("--dockerfile", type=Path, default=DEFAULT_DOCKERFILE)
    build.add_argument("--timeout", type=float, default=1800.0)

    run = sub.add_parser("run", help="execute one command under the sandbox")
    run.add_argument("--image", default=DEFAULT_TAG)
    run.add_argument("--runtime", default="runc", help="runsc (gVisor) or runc")
    run.add_argument(
        "--network",
        default="none",
        help="none, bridge, endpoint (internal + forwarder) or a network name",
    )
    run.add_argument(
        "--endpoint",
        default="",
        help="endpoint URL when --network endpoint; the forwarder dials only this host:port",
    )
    run.add_argument("--deadline", type=float, default=600.0, help="wall-clock seconds")
    run.add_argument("--cpus", default="2")
    run.add_argument("--memory", default="2g")
    run.add_argument("--pids-limit", type=int, default=256)
    run.add_argument("--task-dir", type=Path, default=ROOT)
    run.add_argument("--workspace-rw", action="store_true", help="let the command write the task dir")
    run.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                     help="one allowlisted container variable; repeats")
    run.add_argument("--host-gateway", action="store_true",
                     help="add host.docker.internal -> host gateway (explicit bridge only)")
    run.add_argument("--write-dir", type=Path, default=None)
    run.add_argument("--seccomp", type=Path, default=SECCOMP_PROFILE)
    run.add_argument("--allow-latest", action="store_true")
    run.add_argument("--report", type=Path, default=None)
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "build":
            result = build_image(
                args.digest,
                tag=args.tag,
                dockerfile=args.dockerfile.resolve(),
                base_repo=args.base_repo,
                timeout=args.timeout,
            )
            json.dump(result, sys.stdout, indent=2)
            sys.stdout.write("\n")
            return EXIT_OK

        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise InfraError("no command after '--'")
        pairs: dict[str, str] = {}
        for item in args.env:
            name, sep, value = item.partition("=")
            if not sep:
                raise InfraError(f"--env expects NAME=VALUE, got {item!r}")
            pairs[name] = value
        report = run_sandbox(
            command,
            image=args.image,
            runtime=args.runtime,
            network=args.network,
            deadline=args.deadline,
            cpus=args.cpus,
            memory=args.memory,
            pids_limit=args.pids_limit,
            task_dir=args.task_dir,
            workspace_writable=args.workspace_rw,
            env=pairs,
            host_gateway=args.host_gateway,
            write_dir=args.write_dir,
            allow_latest=args.allow_latest,
            seccomp=args.seccomp,
            endpoint_url=args.endpoint,
        )
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return int(report["exit_code"])  # type: ignore[arg-type]
    except (InfraError, subprocess.TimeoutExpired) as exc:
        report = {"status": "infra_error", "error": str(exc)}
        if args.action == "run" and args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return EXIT_INFRA
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
