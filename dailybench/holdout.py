"""Holdout splits and rights records for the public layer.

The suite has three audiences, and only one of them may see task text:

``public-dev``
    Published in full: prompt, workspace, dialogue, tools, oracle.
``audit``
    Internal set used to hunt evaluator defects; the public export keeps hashes.
``closed-eval``
    Comparative holdout; nothing but hashes and the denominator leaves the
    machine. Seeds are withheld too, because ``generator + seed`` reconstructs
    the fixture.

Assignment is keyed by task ID only: a SHA-256 bucket, so adding a task never
moves its neighbours even when the suite grows. The manifest carries a hash of
every shown field and a digest of the whole composition; ``check`` compares
both against a fresh ``build_suite`` and against the public export. Dividing
seeds and oracles into different files is not enough — :func:`check_export`
proves the export really replaced them with hashes.

The second half of the module is the rights registry. One JSON record per
external material lists origin, revision, upstream ID, code and data licences,
attribution, redistribution and verification status. A missing or unverified
licence refuses the import; it is never assumed to be MIT.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import schema

MANIFEST_SCHEMA_VERSION = "dailybench/holdout/1"
LICENSE_SCHEMA_VERSION = "dailybench/license/1"

SPLITS: tuple[str, ...] = ("public-dev", "audit", "closed-eval")
"""Splits, from fully published to hash-only."""

HIDDEN_SPLITS: tuple[str, ...] = ("audit", "closed-eval")
"""Splits whose content never appears in a public export."""

CLOSED_SPLIT = "closed-eval"
PUBLIC_SPLIT = "public-dev"

CLOSED_SHARE = 0.40
"""Default closed-eval share. Matches the plan's six dev / four closed cases."""

AUDIT_SHARE = 0.20
"""Default internal audit share; the remainder stays public development."""

_SALT = "dailybench/holdout/1"

FINGERPRINT_FIELDS: tuple[str, ...] = (
    "prompt_sha256",
    "dialogue_sha256",
    "workspace_sha256",
    "tools_sha256",
    "system_sha256",
    "assets_sha256",
    "oracle_sha256",
)

LICENSE_FIELDS: tuple[str, ...] = (
    "schema",
    "id",
    "origin",
    "revision",
    "upstream_id",
    "code_license",
    "data_license",
    "attribution",
    "redistribution",
    "status",
)

LICENSE_STATUSES: tuple[str, ...] = ("verified", "unverified", "blocked")
REDISTRIBUTION: tuple[str, ...] = ("allowed", "restricted", "forbidden")

_REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_SCHEMA_PATH = schema.SCHEMA_DIR / "holdout.schema.json"
LICENSE_DIR = _REPO_ROOT / "docs" / "licenses"
"""Checked-in rights records; every ``*.json`` here must be importable."""


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def _canonical(value: Any) -> Any:
    """Replace objects whose ``repr`` is not stable.

    K oracle specs carry closures and code objects; ``str()`` of either embeds a
    memory address, so two builds of the same seed would hash differently.
    Qualname plus bytecode and constants identify the predicate without the
    address; sets are sorted because their iteration order depends on hashing.
    """
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, types.CodeType):
        return {
            "code": hashlib.sha256(value.co_code).hexdigest(),
            "consts": [_canonical(item) for item in value.co_consts],
        }
    if callable(value):
        code = getattr(value, "__code__", None)
        if code is None:
            return f"<callable {getattr(value, '__qualname__', type(value).__name__)}>"
        return {
            "callable": getattr(value, "__qualname__", type(value).__name__),
            **_canonical(code),
        }
    return value


def _digest(value: Any) -> str:
    """Canonical SHA-256 of a JSON value (sorted keys, no locale)."""
    text = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _document(task: Any) -> dict[str, Any]:
    return task if isinstance(task, Mapping) else schema.task_to_dict(task)


def fingerprint(task: Any) -> dict[str, str]:
    """Hashes of every field a candidate can observe, plus the oracle.

    ``prompt`` and ``workspace`` are hashed as the frozen bytes; the oracle is
    hashed but never published. Reusing :func:`decontam.prompt_hashes` keeps
    the wire format identical to the private-stub one.
    """
    from .decontam import prompt_hashes

    doc = _document(task)
    return {
        **prompt_hashes(doc),
        "system_sha256": _digest(doc.get("system")),
        "assets_sha256": _digest(doc.get("assets") or []),
        "oracle_sha256": _digest(doc.get("oracle")),
    }


def composition_hash(entries: Iterable[Mapping[str, Any]]) -> str:
    """Hash of one split's ordered composition: IDs and their fingerprints."""
    return _digest([{"id": entry["id"], "hashes": entry["hashes"]} for entry in entries])


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return _digest(manifest["composition"])


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def bucket(task_id: str) -> float:
    """Stable point in ``[0, 1)`` from the task ID alone."""
    raw = hashlib.sha256(f"{_SALT}:{task_id}".encode()).hexdigest()
    return int(raw[:8], 16) / 0x1_0000_0000


def split_for(
    task_id: str,
    *,
    closed_share: float = CLOSED_SHARE,
    audit_share: float = AUDIT_SHARE,
) -> str:
    """Deterministic split of one ID.

    The same ID keeps its split across runs and across task-count changes.
    """
    point = bucket(task_id)
    if point < closed_share:
        return CLOSED_SPLIT
    if point < closed_share + audit_share:
        return "audit"
    return PUBLIC_SPLIT


def build_manifest(
    tasks: Iterable[Any],
    *,
    profile: str,
    seed: int | None = None,
    closed_share: float = CLOSED_SHARE,
    audit_share: float = AUDIT_SHARE,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Split a generated suite and hash the composition.

    ``overrides`` pins individual IDs (a template that must stay closed, or a
    family that needs one public example) without disturbing the hash buckets
    of every other task.
    """
    if closed_share < 0 or audit_share < 0 or closed_share + audit_share > 1:
        raise ValueError("доли сплитов должны быть неотрицательны и в сумме не больше 1")
    pinned = dict(overrides or {})
    unknown = sorted(set(pinned) - {getattr(task, "id", "") for task in tasks})
    if unknown:
        raise ValueError(f"override для неизвестных заданий: {', '.join(unknown)}")
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for task in tasks:
        identifier = str(getattr(task, "id"))
        split = pinned.get(identifier) or split_for(
            identifier, closed_share=closed_share, audit_share=audit_share
        )
        if split not in SPLITS:
            raise ValueError(f"{identifier}: неизвестный сплит {split!r}")
        splits[split].append({"id": identifier, "hashes": fingerprint(task)})
    for entries in splits.values():
        entries.sort(key=lambda entry: entry["id"])
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "profile": profile,
        "policy": {"closed_share": closed_share, "audit_share": audit_share},
        "splits": splits,
        "composition": {name: composition_hash(splits[name]) for name in SPLITS},
    }
    if seed is not None:
        manifest["seed"] = seed
    manifest["digest"] = manifest_digest(manifest)
    return manifest


def split_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    """``id -> split`` for every entry of a manifest."""
    return {
        entry["id"]: name
        for name in SPLITS
        for entry in manifest.get("splits", {}).get(name, [])
    }


def hidden_ids(manifest: Mapping[str, Any]) -> dict[str, str]:
    """``id -> split`` for the hash-only splits (audit and closed-eval)."""
    mapping = split_map(manifest)
    return {key: value for key, value in mapping.items() if value in HIDDEN_SPLITS}


# --------------------------------------------------------------------------- #
# Manifest validation and checks
# --------------------------------------------------------------------------- #


def holdout_schema() -> dict[str, Any]:
    return schema._load(str(HOLDOUT_SCHEMA_PATH))


def _schema_errors(doc: Any, definition: str, prefix: str) -> list[str]:
    root = holdout_schema()
    return schema._validate(doc, {"$ref": f"#/$defs/{definition}"}, root, prefix)


def validate_manifest(doc: Any, prefix: str = "holdout") -> list[str]:
    """Every contract violation of a manifest, including its digest."""
    if not isinstance(doc, Mapping):
        return [f"{prefix}: ожидался объект, получено {schema._json_type(doc)}"]
    errors = _schema_errors(doc, "manifest", prefix)
    splits = doc.get("splits")
    if not isinstance(splits, Mapping):
        return errors

    seen: dict[str, str] = {}
    for name in SPLITS:
        entries = splits.get(name)
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                continue
            identifier = entry.get("id")
            if not isinstance(identifier, str):
                continue
            if identifier in seen:
                errors.append(
                    f"{prefix}/splits/{name}/{index}/id: {identifier!r} уже в сплите {seen[identifier]}"
                )
            seen[identifier] = name
        complete = all(
            isinstance(entry, Mapping)
            and isinstance(entry.get("id"), str)
            and isinstance(entry.get("hashes"), Mapping)
            for entry in entries
        )
        if complete:
            declared = (doc.get("composition") or {}).get(name)
            if isinstance(declared, str) and declared != composition_hash(entries):
                errors.append(f"{prefix}/composition/{name}: не совпадает с составом сплита")

    composition = doc.get("composition")
    digest = doc.get("digest")
    if isinstance(composition, Mapping) and isinstance(digest, str):
        if manifest_digest(doc) != digest:
            errors.append(f"{prefix}/digest: не совпадает с composition")
    values = set(seen.values())
    if CLOSED_SPLIT not in values:
        errors.append(f"{prefix}/splits/{CLOSED_SPLIT}: нет ни одного закрытого задания")
    if PUBLIC_SPLIT not in values:
        errors.append(f"{prefix}/splits/{PUBLIC_SPLIT}: нет ни одного публичного задания")
    return errors


def check_manifest(manifest: Mapping[str, Any], tasks: Iterable[Any]) -> list[str]:
    """Compare a manifest with the suite it was built from."""
    errors = validate_manifest(manifest)
    if errors:
        return errors
    by_id = {str(getattr(task, "id")): task for task in tasks}
    mapping = split_map(manifest)
    missing = sorted(set(mapping) - set(by_id))
    if missing:
        errors.append(f"holdout: в сьютe нет заданий из манифеста: {', '.join(missing)}")
    extra = sorted(set(by_id) - set(mapping))
    if extra:
        errors.append(f"holdout: манифест не покрывает задания: {', '.join(extra)}")
    for name in SPLITS:
        for entry in manifest["splits"][name]:
            task = by_id.get(entry["id"])
            if task is None:
                continue
            actual = fingerprint(task)
            for key in FINGERPRINT_FIELDS:
                declared = entry["hashes"].get(key)
                if declared != actual[key]:
                    errors.append(
                        f"holdout/{name}/{entry['id']}/{key}: hash изменился "
                        f"({declared} != {actual[key]})"
                    )
    return errors


def check_export(doc: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    """Prove that no hidden text, seed or oracle reached a public export.

    Only ``public-dev`` may carry prompt/workspace/dialogue/tools/oracle.
    Hidden entries must be stubs whose hashes equal the manifest.
    """
    errors = validate_manifest(manifest)
    if errors:
        return errors
    tasks = doc.get("tasks")
    if not isinstance(tasks, list):
        return [f"export/tasks: ожидался массив, получено {schema._json_type(tasks)}"]
    by_id = {
        entry.get("id"): entry for entry in tasks if isinstance(entry, Mapping)
    }
    for name in SPLITS:
        for entry in manifest["splits"][name]:
            identifier = entry["id"]
            task = by_id.get(identifier)
            if task is None:
                errors.append(f"export: закрытое задание {identifier} отсутствует в экспорте")
                continue
            if name == PUBLIC_SPLIT:
                if not task.get("prompt"):
                    errors.append(f"export/{identifier}: public-dev задание без prompt")
                continue
            if task.get("prompt") is not None:
                errors.append(f"export/{identifier}: {name} утекло текстом prompt")
            for field in ("workspace", "dialogue", "tools", "oracle", "assets", "system"):
                if task.get(field) is not None:
                    errors.append(f"export/{identifier}: {name} утекло полем {field}")
            if task.get("holdout_seeds") is not None:
                errors.append(f"export/{identifier}: {name} опубликовало holdout_seeds")
            if task.get("split") != name:
                errors.append(f"export/{identifier}/split: ожидался {name!r}, получено {task.get('split')!r}")
            for key in FINGERPRINT_FIELDS:
                if task.get(key) != entry["hashes"][key]:
                    errors.append(f"export/{identifier}/{key}: хеш не совпадает с манифестом")
    for identifier in by_id:
        if identifier is not None and identifier not in split_map(manifest):
            errors.append(f"export: задание {identifier} отсутствует в манифесте")
    return errors


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Rights registry
# --------------------------------------------------------------------------- #


def license_schema() -> dict[str, Any]:
    return holdout_schema()


def validate_record(record: Any, prefix: str = "license") -> list[str]:
    """Contract violations of one rights record, plus the import decision."""
    if not isinstance(record, Mapping):
        return [f"{prefix}: ожидался объект, получено {schema._json_type(record)}"]
    errors = _schema_errors(record, "material", prefix)
    status = record.get("status")
    if status in LICENSE_STATUSES and status != "verified":
        errors.append(f"{prefix}/status: {status} — импорт запрещён до проверки лицензии")
    if record.get("redistribution") == "forbidden":
        errors.append(f"{prefix}/redistribution: запрещено распространение")
    if record.get("redistribution") == "restricted" and not str(record.get("note") or "").strip():
        errors.append(f"{prefix}/note: ограниченное распространение требует назвать ограничение")
    return errors


def importable(record: Mapping[str, Any]) -> bool:
    return not validate_record(record)


def require_importable(record: Mapping[str, Any], name: str = "материал") -> None:
    errors = validate_record(record, name)
    if errors:
        raise ValueError(f"{name}: импорт отклонён: " + "; ".join(errors))


def load_record(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_record_by_id(record_id: str, directory: Path | None = None) -> dict[str, Any]:
    """Load ``<directory>/<id>.json`` and refuse it when it is not importable."""
    folder = Path(directory) if directory is not None else LICENSE_DIR
    path = folder / f"{record_id}.json"
    if not path.is_file():
        raise ValueError(f"{record_id}: нет записи о правах {path}")
    record = load_record(path)
    require_importable(record, str(path))
    return record


def check_directory(directory: Path | None = None) -> list[tuple[Path, list[str]]]:
    """Validate every record of a registry directory.

    The file name must repeat the record ``id``, so a copy-pasted record cannot
    hide behind a mismatched name. An empty registry is an error: a clone that
    lost its records would otherwise look clean.
    """
    folder = Path(directory) if directory is not None else LICENSE_DIR
    failures: list[tuple[Path, list[str]]] = []
    paths = sorted(folder.glob("*.json"))
    if not paths:
        return [(folder, [f"{folder}: нет ни одной записи о правах"])]
    for path in paths:
        try:
            record = load_record(path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append((path, [f"{path}: {exc}"]))
            continue
        errors = validate_record(record, str(path))
        if isinstance(record, Mapping) and record.get("id") != path.stem:
            errors.append(f"{path}/id: {record.get('id')!r} не совпадает с именем файла")
        if errors:
            failures.append((path, errors))
    return failures
