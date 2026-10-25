#!/usr/bin/env python3
"""Supervisor for one sandboxed command (spec §6).

Builds the runner image from a digest-pinned base, executes the command under
the isolation contract (read-only rootfs, tmpfs-only /tmp, no capabilities,
no new privileges, seccomp allow-list, pids limit, cgroup v2 limits) and an
absolute deadline, then reports a machine-readable status.

The status vocabulary is deliberate: ``timeout``, ``oom`` and ``infra_error``
are infrastructure outcomes and must map to ``evaluator_failure``, never to
a zero score for the model.  Docker reserves exit code 125 for its own
failures (daemon gone, image pulled away, name collision) and 126/127 for exec
failures *inside* the container; the first is infra, the latter two are the
candidate's.  The container must be observed in a terminal state before any
exit code is classified, because the CLI's code is meaningless if the daemon
died mid-attach.

Usage::

    python scripts/sandbox_run.py build [--digest sha256:...] [--tag TAG]
    python scripts/sandbox_run.py run --deadline 60 -- python solution.py

Exit codes: 0 ok, 1 candidate failure, 2 infrastructure, 3 timeout, 4 oom.
``125`` from ``docker run`` is reported as ``infra_error``/2, never as ``failed``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

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

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INFRA = 2
EXIT_TIMEOUT = 3
EXIT_OOM = 4


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
    proc = _call(
        [
            "docker",
            "image",
            "inspect",
            image,
            "--format",
            "{{join .RepoDigests \",\"}}|{{.Id}}",
        ],
        timeout=30,
    )
    if proc.returncode != 0:
        raise InfraError(f"image {image!r} not available: {proc.stderr.strip()}")
    digests, _, image_id = proc.stdout.strip().partition("|")
    return (digests.split(",")[0] if digests else ""), image_id


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


def image_sbom(image: str, *, timeout: float = 120) -> list[str]:
    """Installed packages of ``image`` as ``package<TAB>version`` lines.

    Runs the image itself because the package list is part of the provenance:
    a rebuild from the same base must produce the same record, or the change
    is visible in the manifest before any score is attributed to the model.
    """
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


def _require_pinned(image: str, allow_latest: bool) -> None:
    if "@sha256:" in image:
        return
    tag = image.rsplit("/", 1)[-1]
    tag = tag.split(":", 1)[1] if ":" in tag else ""
    if (not tag or tag == "latest") and not allow_latest:
        raise InfraError(
            f"image {image!r} uses the forbidden 'latest' tag; pass a digest "
            "reference or --allow-latest"
        )


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
        "sbom": image_sbom(tag),
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
        # The endpoint usually listens on the host's loopback. The alias is
        # the only name the container needs; no other host path is reachable.
        argv += ["--add-host", "host.docker.internal:host-gateway"]
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
) -> dict[str, object]:
    """Run one command inside the sandbox and return the status report.

    The caller controls the whole exposure surface: ``task_dir`` is the fixture
    and is read-only unless the command must write into it, ``extra_mounts``
    adds the few other host paths allowed (as ``(host, container target,
    read-only)``), and ``env`` is the complete container environment — Docker
    inherits nothing, so a host secret that is not listed cannot leak in.

    ``stdout_path``/``stderr_path`` receive the full streams; the report always
    keeps a tail of both.
    """
    if not command:
        raise InfraError("empty command")
    if deadline <= 0:
        raise InfraError("deadline must be positive")
    docker = _docker()
    if not docker_available():
        raise InfraError("docker daemon is not reachable")
    epoch = source_date_epoch()
    _require_pinned(image, allow_latest)
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

    container = f"dailybench-sandbox-{secrets.token_hex(5)}"
    argv = _run_argv(
        docker=docker,
        container=container,
        image=image,
        command=command,
        runtime=runtime,
        network=network,
        task_dir=task_dir,
        write_dir=write_dir,
        cpus=cpus,
        memory=memory,
        pids_limit=pids_limit,
        seccomp=seccomp.resolve(),
        workspace_writable=workspace_writable,
        extra_mounts=extras,
        env=env,
        host_gateway=host_gateway,
    )
    # Mount specs are validated above so a bad path fails before the image is
    # looked up.
    repo_digest, image_id = image_digest(image)

    started = time.monotonic()
    timed_out = False
    oom_killed = False
    exit_code: int | None = None
    output = ""
    errors = ""
    infra_reason = ""
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_docker_env(),
            start_new_session=True,
        )
        try:
            output, errors = proc.communicate(timeout=deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill the container tree first, then the CLI's process group.
            _call(["docker", "kill", "--signal", "KILL", container], timeout=30)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc.wait(timeout=30)
            try:
                output, errors = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - stuck pipe
                output, errors = "", ""
        exit_code = proc.returncode
        if stdout_path is not None:
            stdout_path.write_text(output or "", encoding="utf-8")
        if stderr_path is not None:
            stderr_path.write_text(errors or "", encoding="utf-8")
        state = _inspect_state(container)
        if state is not None and state.get("Running") is True:
            # No terminal state means the CLI code cannot be classified yet.
            _call(["docker", "kill", "--signal", "KILL", container], timeout=20)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
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
        if proc is None or proc.poll() is None:
            _call(["docker", "kill", "--signal", "KILL", container], timeout=20)
        _call(["docker", "rm", "--force", "--volumes", container], timeout=30)

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

    return {
        "status": status,
        "exit_code": code,
        "error": infra_reason or None,
        "container_exit_code": exit_code,
        "runtime": runtime,
        "runtime_verified": runtime == "runc" or runtime in registered_runtimes(),
        "image": image,
        "image_digest": repo_digest,
        "image_id": image_id,
        "source_date_epoch": epoch,
        "sbom": image_sbom(image),
        "command": command,
        "network": network,
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
        "host_gateway": host_gateway,
        "cgroups": {"cpus": cpus, "memory": memory, "pids_limit": pids_limit},
        "rootfs": "read_only",
        "capabilities": "dropped",
        "cpus": cpus,
        "memory": memory,
        "pids_limit": pids_limit,
        "output": ((output or "") + ("\n" + errors if errors else ""))[-8000:],
    }


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
    run.add_argument("--network", default="none", help="none, bridge or a network name")
    run.add_argument("--deadline", type=float, default=600.0, help="wall-clock seconds")
    run.add_argument("--cpus", default="2")
    run.add_argument("--memory", default="2g")
    run.add_argument("--pids-limit", type=int, default=256)
    run.add_argument("--task-dir", type=Path, default=ROOT)
    run.add_argument("--workspace-rw", action="store_true", help="let the command write the task dir")
    run.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                     help="one allowlisted container variable; repeats")
    run.add_argument("--host-gateway", action="store_true",
                     help="add host.docker.internal -> host gateway")
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
