"""Committed generations of a run directory.

A checkpoint is not one file: config, suite, scores, tasks and the per-task
mirrors must describe the same moment. ``os.replace`` per file leaves a mixed
directory when the process dies between two of them, and a reader then sees a
fresh score next to an old answer.

A generation is therefore staged whole, fsynced, and only then made reachable
by atomically replacing ``manifest.json``. The manifest names the generation,
the run identity and a SHA-256 per file. Readers resolve the manifest first and
read from the committed generation instead of the root copies, so an
interrupted commit always leaves the previous generation visible. The root
files stay in place as copies (clone where the filesystem supports it) for
tools that treat a run as plain files: an in-place root edit must not change
the bytes of the sealed generation.

Every file is written to a sibling temporary file, flushed through a writable
descriptor (``os.fsync`` rejects a read-mode one on Windows), and renamed in.
Directory barriers need a read-only handle: Windows cannot open a directory, so
it is skipped. A barrier the filesystem reports as unsupported (EINVAL/ENOTSUP)
degrades the checkpoint instead of aborting it, and the manifest records every
such fallback under ``durability``; any other ``OSError`` fails the commit.

A checkpoint of a long sweep is gigabytes — ``tasks.json`` alone carries every
answer — so the store does not grow without bound: unchanged files are hard
links on the previous generation, and after a successful publish only the
newest ``generation_keep`` generations survive. The current generation and the
one the manifest names are never removed, and a failed removal only warns.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import time
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA = "dailybench/generation/1"
STORE = ".generations"
MANIFEST = "manifest.json"
ENVIRONMENT = "environment.json"

#: Generations kept per run directory when the checkpoint's config says nothing.
DEFAULT_KEEP = 3
#: Prefix of the folder a generation is renamed to before it is deleted.
TRASH = ".trash-"

#: Provenance of the executor. Stored in the manifest next to the run identity
#: so a reader can tell a resumed run whether it is still on the same image.
PROVENANCE_SCHEMA = "dailybench/environment/1"
DEFAULT_IMAGE = "dailybench/runner-agent:local"
SOURCE_DATE_EPOCH_DEFAULT = 1700000000


class ProvenanceError(RuntimeError):
    """The run was continued on a different executor than it started on."""


class IntegrityError(RuntimeError):
    """A manifest exists but no longer describes an intact generation.

    The root copies are an export, not a second source of truth: falling back
    to them would serve whatever bytes are there as the newest state.
    """


#: ``path`` → (mtime_ns, size, inode, digest) of files that already hashed to
#: their manifest entry. Re-hashing every transcript on every ``read_json`` grew
#: quadratic with a run's length; a stat match is enough to trust a previous
#: verification, and an edit (in-place or replaced file) changes mtime/inode.
_CHECKED: dict[str, tuple[int, int, int, str]] = {}


#: errnos that mean "this filesystem cannot flush that kind of handle", not
#: "the flush failed". Such a case degrades durability and is recorded in the
#: manifest; any other ``OSError`` must abort the commit.
UNSUPPORTED_FSYNC = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP})


def _dir_fsync_supported() -> bool:
    """A directory is opened read-only: Windows cannot fsync such a handle."""
    return os.name != "nt"


def _fsync(fd: int, kind: str, degraded: list[str]) -> None:
    try:
        os.fsync(fd)
    except OSError as error:
        if error.errno not in UNSUPPORTED_FSYNC:
            raise
        marker = f"{kind}-fsync:unsupported:{errno.errorcode.get(error.errno, error.errno)}".lower()
        if marker not in degraded:
            degraded.append(marker)


def _fsync_file(path: Path, degraded: list[str]) -> None:
    """Flush a file that is already in place (a carried-over checkpoint file).

    ``os.fsync`` needs a writable descriptor: an ``rb`` handle is rejected on
    Windows, which is how the first checkpoint used to die unpublished.
    """
    with open(path, "r+b") as handle:
        _fsync(handle.fileno(), "file", degraded)


def _fsync_dir(path: Path, degraded: list[str]) -> None:
    if not _dir_fsync_supported():
        if "directory-fsync:unsupported:nt" not in degraded:
            degraded.append("directory-fsync:unsupported:nt")
        return
    handle = os.open(path, os.O_RDONLY)
    try:
        _fsync(handle, "directory", degraded)
    finally:
        os.close(handle)


def _manifest_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_durable(target: Path, text: str, degraded: list[str]) -> None:
    """Write ``text`` beside ``target``, flush it, and only then rename it in.

    Writing the target directly leaves a truncated file when the process dies
    mid-write; ``os.replace`` of a fully flushed sibling is atomic.
    """
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        _fsync(handle.fileno(), "file", degraded)
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# provenance


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def environment_digest(record: dict[str, Any]) -> str:
    """SHA-256 of one environment record (stable across key order)."""
    return sha256(_canonical(record).encode("utf-8")).hexdigest()


def _docker_call(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("DOCKER_CLI_HINTS", "false")
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def image_identity(image: str, *, docker: str | None = None) -> tuple[str, str]:
    """``(repo_digest, image_id)`` from ``docker image inspect``; either may be empty."""
    docker = docker or shutil.which("docker")
    if not docker:
        raise ProvenanceError("docker CLI not found in PATH")
    proc = _docker_call(
        [docker, "image", "inspect", image, "--format", '{{join .RepoDigests ","}}|{{.Id}}'],
        timeout=30,
    )
    if proc.returncode != 0:
        raise ProvenanceError(f"image {image!r} is not available: {proc.stderr.strip()}")
    digests, _, image_id = proc.stdout.strip().partition("|")
    return (digests.split(",")[0] if digests else ""), image_id


def sbom(image: str, *, docker: str | None = None, timeout: float = 120) -> list[str]:
    """Installed packages of the image as ``package\\tversion`` lines.

    Best effort: an image without ``dpkg`` yields an empty list instead of an
    error, and the caller records the environment as unverified.
    """
    docker = docker or shutil.which("docker")
    if not docker:
        return []
    try:
        proc = _docker_call(
            [
                docker,
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


def _runtimes(docker: str) -> set[str]:
    try:
        proc = _docker_call([docker, "info", "--format", "{{json .Runtimes}}"], timeout=20)
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    try:
        data = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return set()
    return set(data) if isinstance(data, dict) else set()


def environment(
    image: str | None = None,
    runtime: str | None = None,
    *,
    container_ran: bool = False,
) -> dict[str, Any]:
    """Executor provenance for the manifest and the report's «Среда» section.

    A missing Docker CLI is not a failure: the record says ``unverified`` and a
    resume on such a host continues instead of refusing. The image and runtime
    default to the supervisor's local tag and can be overridden with
    ``DAILYBENCH_SANDBOX_IMAGE`` / ``DAILYBENCH_SANDBOX_RUNTIME``.

    ``status`` is ``verified`` only once the runtime is registered in Docker and
    the image has an immutable identity. ``executor_status`` is separate: naming
    an image does not mean a container was used, so without ``container_ran``
    the record says ``host_run`` with a reason, and a missing Docker says
    ``unverified`` instead of pretending the host run matched the image.

    ``container_ran`` comes from the caller's pre-flight probe
    (``isolation.prepare()``, which also builds a missing image), the same probe
    the runner uses: the record and the execution therefore agree on whether a
    container will be started.
    """
    image = image or os.environ.get("DAILYBENCH_SANDBOX_IMAGE") or DEFAULT_IMAGE
    runtime = runtime or os.environ.get("DAILYBENCH_SANDBOX_RUNTIME") or "runc"
    raw_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    try:
        epoch = int(raw_epoch) if raw_epoch is not None else SOURCE_DATE_EPOCH_DEFAULT
    except ValueError:
        raise ProvenanceError(f"SOURCE_DATE_EPOCH={raw_epoch!r} не целое число") from None
    record: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "image": image,
        "runtime": runtime,
        "runtime_verified": False,
        "source_date_epoch": epoch,
        "image_digest": "",
        "image_id": "",
        "sbom": [],
        "status": "unverified",
        "reason": "",
        "executor_status": "container_run" if container_ran else "host_run",
        "executor_reason": ""
        if container_ran
        else "образ указан, но контейнер не запускался — прогон шёл на хосте",
    }
    docker = shutil.which("docker")
    if not docker:
        record["reason"] = "docker CLI не найден"
        record["executor_status"] = "unverified"
        record["executor_reason"] = "docker CLI не найден — исполнитель не подтверждён"
        return record
    problems: list[str] = []
    runtimes = _runtimes(docker)
    record["runtime_verified"] = runtime in runtimes
    if not record["runtime_verified"]:
        known = ", ".join(sorted(runtimes)) or "нет данных"
        problems.append(f"runtime {runtime!r} не зарегистрирован в docker ({known})")
    try:
        digest, image_id = image_identity(image, docker=docker)
    except (OSError, subprocess.SubprocessError, ProvenanceError) as exc:
        problems.append(f"docker недоступен: {exc}")
    else:
        record["image_id"] = image_id
        # A locally built tag has no repo digest; the image id is an equally
        # immutable reference and keeps the resume guard working.
        record["image_digest"] = digest or image_id
        record["sbom"] = sbom(image, docker=docker)
        if not (digest or image_id):
            problems.append("у образа нет ни digest, ни id")
    if problems:
        record["reason"] = "; ".join(problems)
    else:
        record["status"] = "verified"
    return record


_ENVIRONMENT_KEYS = (
    "schema",
    "image",
    "runtime",
    "runtime_verified",
    "source_date_epoch",
    "image_digest",
    "image_id",
    "sbom",
    "status",
    "reason",
    "executor_status",
    "executor_reason",
    # The CA handed into the container is part of the TLS conditions: a resume
    # on a host whose bundle appeared or vanished is another environment. Without
    # the keys here the *saved* side kept them (a nested record is used whole)
    # while the fresh side dropped them, and every resume was refused.
    "ca_bundle",
    "ca_container_path",
    "ca_mounted",
    "ca_reason",
)


def _environment_record(value: dict[str, Any] | None) -> dict[str, Any]:
    """The environment record out of a saved config, or the record itself.

    Resume reads the old ``config.json``, while the tests and the CLI also pass
    a bare environment record; the rest of a config (model, paths, calibration)
    is guarded by the identity, so it must not enter this digest.
    """
    if not value:
        return {}
    nested = value.get("environment")
    if isinstance(nested, dict):
        return nested
    return {key: value[key] for key in _ENVIRONMENT_KEYS if key in value}


def verify_environment(recorded: dict[str, Any] | None, current: dict[str, Any] | None) -> None:
    """Refuse a resume whose executor differs from the one the run started on.

    The whole record is compared by digest, so the image, the runtime, the
    build epoch and the recorded packages all count — not just the image.
    ``unverified`` on either side continues: the point is to catch a foreign
    executor, not to make Docker mandatory.
    """
    expected_record = _environment_record(recorded)
    current_record = _environment_record(current)
    if not expected_record:
        return
    if str(expected_record.get("status") or "") == "unverified":
        return
    if not current_record or str(current_record.get("status") or "") == "unverified":
        return
    expected = environment_digest(expected_record)
    actual = environment_digest(current_record)
    if expected == actual:
        return
    changed = sorted(
        key
        for key in set(expected_record) | set(current_record)
        if _canonical(expected_record.get(key)) != _canonical(current_record.get(key))
    )
    detail = f" ({', '.join(changed)})" if changed else ""
    raise ProvenanceError(
        f"окружение песочницы сменилось{detail}: прогон начат на {expected[:19]}…, "
        f"сейчас {actual[:19]}… — resume смешал бы два окружения "
        "(соберите тот же digest или запустите новый прогон с новым --name)"
    )


def _manifest(directory: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    generation = str(data.get("generation") or "")
    if not generation or "/" in generation or generation.startswith("."):
        return None
    return data


def _verified(directory: Path, name: str, manifest: dict[str, Any] | None = None) -> Path | None:
    """The generation directory if every file still matches its manifest hash."""
    folder = directory / STORE / name
    if not folder.is_dir():
        return None
    data = manifest if manifest is not None and manifest.get("generation") == name else _manifest(folder)
    if data is None:
        return None
    own = _manifest(folder)
    if own is None or _canonical(own) != _canonical(data):
        # A root manifest edited together with the hashes it lists would
        # otherwise verify itself; the generation keeps its own copy.
        return None
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        return None
    for file, digest in files.items():
        path = folder / file
        try:
            stat = path.stat()
            key = str(path)
            cached = _CHECKED.get(key)
            if cached and cached[:3] == (stat.st_mtime_ns, stat.st_size, stat.st_ino) and cached[3] == digest:
                continue
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
                _CHECKED.pop(key, None)
                return None
            _CHECKED[key] = (stat.st_mtime_ns, stat.st_size, stat.st_ino, digest)
        except OSError:
            return None
    # The manifest identity must be the identity the run recorded in its own
    # config: a manifest swapped from another run would otherwise pass hashes.
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    expected = str(data.get("identity") or "")
    if expected:
        if str(config.get("identity") or "") != expected:
            return None
    elif config.get("identity"):
        return None
    # The recorded executor must be the one the config claims: a manifest
    # swapped from another environment would otherwise pass every hash.
    recorded = data.get("environment") or {}
    if recorded:
        if _canonical(config.get("environment") or {}) != _canonical(recorded):
            return None
        if data.get("environment_sha256") and data["environment_sha256"] != environment_digest(recorded):
            return None
    elif config.get("environment"):
        return None
    return folder


def _root_problems(directory: Path, data: dict[str, Any]) -> list[str]:
    """Root copies that no longer match the manifest they were exported from."""
    problems: list[str] = []
    for name, digest in (data.get("files") or {}).items():
        path = directory / name
        try:
            stat = path.stat()
            key = str(path)
            cached = _CHECKED.get(key)
            if cached and cached[:3] == (stat.st_mtime_ns, stat.st_size, stat.st_ino) and cached[3] == digest:
                continue
            if sha256(path.read_bytes()).hexdigest() != digest:
                problems.append(name)
            else:
                _CHECKED[key] = (stat.st_mtime_ns, stat.st_size, stat.st_ino, digest)
        except OSError:
            problems.append(name)
    return problems


def resolve(directory: Path | str) -> tuple[Path, str, list[str]]:
    """``(base, integrity, problems)`` for a reader.

    ``integrity`` is ``sealed`` for a verified generation with matching root
    copies, ``root-drift`` when only the export was edited (the sealed bytes are
    what this base serves), and ``legacy`` for a directory that never had a
    manifest. A present-but-broken manifest raises: quarantine, no legacy
    fallback.
    """
    directory = Path(directory)
    generation = current(directory)
    if generation is not None:
        problems = _root_problems(directory, _manifest(directory) or {})
        return generation, ("root-drift" if problems else "sealed"), problems
    if (directory / MANIFEST).is_file():
        raise IntegrityError(
            f"{directory}: манифест generation не прошёл проверку — данные в карантине, "
            "root-копии не являются источником истины (пересоберите прогон или "
            "восстановите generation из .generations)"
        )
    return directory, "legacy", []


def current(directory: Path | str) -> Path | None:
    """The committed generation, or ``None`` for an uncommitted/legacy run.

    A generation only counts when every file still hashes to its manifest entry
    and its config carries the manifest identity. A damaged generation yields
    ``None``: :func:`committed` then falls back to the root working copy (which
    is flagged by :func:`resolve`), and a half-published commit never reaches
    here because the manifest is replaced last.
    """
    directory = Path(directory)
    data = _manifest(directory)
    if data is None:
        return None
    return _verified(directory, str(data["generation"]), data)


def committed(directory: Path | str) -> Path:
    """Base directory a working reader must use.

    The sealed generation wins while the root export still matches it. An
    edited root is the live working copy (the operator changed state on
    purpose), so readers follow it — :func:`resolve` is what tells a reader the
    export no longer matches the seal.
    """
    directory = Path(directory)
    generation = current(directory)
    if generation is None:
        return directory
    if _root_problems(directory, _manifest(directory) or {}):
        return directory
    return generation


def read_text(directory: Path | str, name: str) -> str:
    return (committed(directory) / name).read_text(encoding="utf-8")


def read_json(directory: Path | str, name: str) -> Any:
    return json.loads(read_text(directory, name))


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


#: Linux ``FICLONE``: reflink a file so the root export is a copy-on-write
#: clone, not a shared inode. Falls back to a plain copy elsewhere.
_FICLONE = 0x40049409


def _copy_on_write(source: Path, handle) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        fcntl = None
    if fcntl is not None:
        try:
            with open(source, "rb") as origin:
                fcntl.ioctl(handle.fileno(), _FICLONE, origin.fileno())
            return
        except OSError:
            pass
    with open(source, "rb") as origin:
        shutil.copyfileobj(origin, handle)


def _publish_root(directory: Path, name: str, source: Path, degraded: list[str]) -> None:
    """Expose a committed file under the old plain name as an independent copy."""
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.link")
    if tmp.exists():
        tmp.unlink()
    with open(tmp, "wb") as handle:
        _copy_on_write(source, handle)
        handle.flush()
        _fsync(handle.fileno(), "file", degraded)
    os.replace(tmp, target)


def _keep_of(checkpoint: Path, keep: int | None) -> int:
    """Generations to keep: the argument, else what the checkpoint itself says.

    The config is read from the staged generation, so a run keeps the number it
    recorded even when the caller does not pass it (a re-score, a script). An
    unreadable or bogus value falls back to :data:`DEFAULT_KEEP`; retention is
    housekeeping and must never be the reason a checkpoint fails.
    """
    if keep is not None:
        return max(1, int(keep))
    try:
        data = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return DEFAULT_KEEP
    try:
        return max(1, int(data.get("generation_keep", DEFAULT_KEEP)))
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_KEEP


def _drop(folder: Path, warnings_: list[str]) -> None:
    """Rename ``folder`` into the trash, then delete it.

    The rename is the atomic step: a reader iterating the store either still
    sees the generation or no longer does, never a half-emptied one. A failure
    to delete afterwards only leaves invisible bytes behind and is retried by
    the next commit.
    """
    trash = folder.with_name(f"{TRASH}{folder.name}")
    try:
        if trash.exists():
            shutil.rmtree(trash)
        os.replace(folder, trash)
    except OSError as error:
        warnings_.append(f"{folder}: поколение не удалено: {error}")
        return
    try:
        shutil.rmtree(trash)
    except OSError as error:
        warnings_.append(f"{trash}: удаление не завершено: {error}")


def _sweep(store: Path, warnings_: list[str]) -> None:
    """Finish removals and interrupted stagings a crash left in the store."""
    for entry in sorted(store.iterdir()):
        if not entry.name.startswith(".") or not entry.is_dir():
            continue
        if not (entry.name.startswith(TRASH) or entry.name.endswith(".tmp")):
            continue
        try:
            shutil.rmtree(entry)
        except OSError as error:
            warnings_.append(f"{entry}: мусор не убран: {error}")


def prune(directory: Path | str, *, keep: int = DEFAULT_KEEP, current: str = "") -> list[str]:
    """Drop all but the newest ``keep`` generations; return the warnings.

    ``current`` and the generation the manifest names are protected — a reader
    resolving the manifest must always find its bytes, so retention can never
    delete what is being served. Failures come back as strings and are warnings
    to the caller: the checkpoint is already published, and a full store is
    still readable.
    """
    directory = Path(directory)
    store = directory / STORE
    warnings_: list[str] = []
    try:
        entries = list(store.iterdir())
    except OSError as error:
        return [f"{store}: хранилище поколений не прочитано: {error}"]
    names = sorted(
        (entry.name for entry in entries if entry.is_dir() and not entry.name.startswith(".")),
        reverse=True,
    )
    # Names start with a zero-padded ``time.time_ns()``, so sorting the name is
    # sorting by age: the tail is the oldest.
    targeted = str((_manifest(directory) or {}).get("generation") or "")
    protected = {name for name in (current, targeted) if name}
    doomed = [name for name in names[max(1, int(keep)) :] if name not in protected]
    for name in doomed:
        _drop(store / name, warnings_)
    _sweep(store, warnings_)
    if not doomed:
        return warnings_
    if _dir_fsync_supported():
        try:
            _fsync_dir(store, [])
        except OSError as error:
            warnings_.append(f"{store}: удаления не сброшены на диск: {error}")
    return warnings_


def commit(
    directory: Path | str,
    files: dict[str, str],
    *,
    run_id: str,
    identity: str,
    environment: dict[str, Any] | None = None,
    carry: bool = True,
    keep: int | None = None,
) -> str:
    """Write ``files`` as one new generation and publish it atomically.

    Files that are unchanged since the previous generation — whether carried
    over or repeated in ``files`` with the same bytes — are hard linked instead
    of written, so a checkpoint costs mostly its own ``tasks.json``. ``files``
    are written through a temporary file and flushed, and the manifest — the
    commit point — is replaced only after every file is durable. Unsupported
    barriers are listed in the manifest ``durability`` field.

    After the publish, all but the newest ``keep`` generations (``None`` = the
    ``generation_keep`` of the checkpoint, :data:`DEFAULT_KEEP` when absent) are
    removed; the current generation and the manifest target are always kept.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    store = directory / STORE
    store.mkdir(exist_ok=True)
    generation = f"{time.time_ns():019d}-{os.getpid()}"
    stage = store / f".{generation}.tmp"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()

    previous = current(directory) if carry else None
    if environment is None:
        # A rescore rewrites scores but runs in the same executor: inherit the
        # provenance of the checkpoint it rewrites instead of dropping it.
        environment = (_manifest(directory) or {}).get("environment") or {}
    previous_files = (_manifest(directory) or {}).get("files") or {} if previous is not None else {}
    if previous is not None:
        for name in previous_files:
            if name in files:
                continue
            _link_or_copy(previous / name, stage / name)
    elif carry:
        # Legacy run without a manifest: its root files are still the current
        # state and must not disappear from the generation a reader will see.
        for source in sorted(directory.glob("*.json")) + sorted((directory / "transcripts").glob("*.json")):
            name = source.relative_to(directory).as_posix()
            if source.name == MANIFEST or name in files or not source.is_file():
                continue
            _link_or_copy(source, stage / name)

    degraded: list[str] = []
    if not _dir_fsync_supported():
        degraded.append("directory-fsync:unsupported:nt")
    for source in sorted(stage.rglob("*")):
        if source.is_file() and str(source.relative_to(stage)) not in files:
            _fsync_file(source, degraded)

    # Assumed digest of a file that is the same bytes as in the previous
    # generation: it is not read back, the manifest already says what it is.
    reused: dict[str, str] = {}
    for name, text in files.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(text.encode("utf-8")).hexdigest()
        if previous is not None and previous_files.get(name) == digest:
            # Identical bytes: one inode serves both generations. ``tasks.json``
            # is expected to differ on every checkpoint and is written anew.
            try:
                _link_or_copy(previous / name, target)
            except OSError:
                _write_durable(target, text, degraded)
            else:
                reused[name] = digest
            continue
        _write_durable(target, text, degraded)

    hashes = {
        str(path.relative_to(stage)): reused.get(str(path.relative_to(stage)))
        or sha256(path.read_bytes()).hexdigest()
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    # Flush the directory entries before the manifest names them, so a barrier
    # that turns out unsupported is already on record.
    _fsync_dir(stage, degraded)
    payload = {
        "schema": SCHEMA,
        "generation": generation,
        "run_id": run_id,
        "identity": identity,
        "environment": environment,
        "environment_sha256": environment_digest(environment),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": hashes,
        "durability": sorted(degraded),
    }
    _write_durable(stage / MANIFEST, _manifest_text(payload), degraded)
    _fsync_dir(stage, degraded)
    os.replace(stage, store / generation)
    _fsync_dir(store, degraded)

    for name in hashes:
        _publish_root(directory, name, stage.parent / generation / name, degraded)

    _write_durable(directory / MANIFEST, _manifest_text(payload), degraded)
    _fsync_dir(directory, degraded)

    if sorted(degraded) != payload["durability"]:
        # A barrier after the manifest was written went unsupported too: a
        # marker that only appears late still has to reach both copies.
        payload["durability"] = sorted(degraded)
        _write_durable(store / generation / MANIFEST, _manifest_text(payload), degraded)
        _fsync_dir(store / generation, degraded)
        _write_durable(directory / MANIFEST, _manifest_text(payload), degraded)
        _fsync_dir(directory, degraded)

    # The checkpoint is published and verified by the manifest; retention is
    # housekeeping, so a failure here warns instead of failing the run.
    if carry:
        for message in prune(directory, keep=_keep_of(store / generation, keep), current=generation):
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return generation
