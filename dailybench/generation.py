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
files stay in place as hard links for tools that treat a run as plain files.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA = "dailybench/generation/1"
STORE = ".generations"
MANIFEST = "manifest.json"

#: ``path`` → (mtime_ns, size, inode, digest) of files that already hashed to
#: their manifest entry. Re-hashing every transcript on every ``read_json`` grew
#: quadratic with a run's length; a stat match is enough to trust a previous
#: verification, and an edit (in-place or replaced file) changes mtime/inode.
_CHECKED: dict[str, tuple[int, int, int, str]] = {}


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


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
    return folder


def current(directory: Path | str) -> Path | None:
    """The committed generation, or ``None`` for an uncommitted/legacy run.

    A generation only counts when every file still hashes to its manifest entry
    and its config carries the manifest identity. A damaged generation yields
    ``None``: readers then use the root copies, which are hard links to the same
    bytes and therefore carry any deliberate hand edit; a half-published commit
    never reaches here because the manifest is replaced last.
    """
    directory = Path(directory)
    data = _manifest(directory)
    if data is None:
        return None
    return _verified(directory, str(data["generation"]), data)


def committed(directory: Path | str) -> Path:
    """Base directory a reader must use: the generation, else the root."""
    return current(directory) or Path(directory)


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


def _publish_root(directory: Path, name: str, source: Path) -> None:
    """Expose a committed file under the old plain name (hard link + replace)."""
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.link")
    if tmp.exists():
        tmp.unlink()
    _link_or_copy(source, tmp)
    os.replace(tmp, target)


def commit(
    directory: Path | str,
    files: dict[str, str],
    *,
    run_id: str,
    identity: str,
    carry: bool = True,
) -> str:
    """Write ``files`` as one new generation and publish it atomically.

    Unchanged files of the previous generation are carried over (hard linked,
    so the checkpoint stays cheap), ``files`` are written and fsynced, and the
    manifest — the commit point — is replaced only after every file is durable.
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
    if previous is not None:
        for name in (_manifest(directory) or {}).get("files") or {}:
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

    for name, text in files.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    hashes = {
        str(path.relative_to(stage)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    payload = {
        "schema": SCHEMA,
        "generation": generation,
        "run_id": run_id,
        "identity": identity,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": hashes,
    }
    manifest = stage / MANIFEST
    with open(manifest, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    for name in list(hashes) + [MANIFEST]:
        _fsync_file(stage / name)
    _fsync_dir(stage)
    os.replace(stage, store / generation)
    _fsync_dir(store)

    for name in hashes:
        _publish_root(directory, name, stage.parent / generation / name)

    tmp = directory / f"{MANIFEST}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, directory / MANIFEST)
    _fsync_dir(directory)
    return generation
