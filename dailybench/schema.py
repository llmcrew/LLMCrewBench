"""Versioned task/profile contract and its independent validator.

The JSON documents in ``schema/`` are the published contract (draft
2020-12); this module validates against them with the standard library only,
so a clone without ``jsonschema`` still rejects a broken fixture. The
semantic layer on top of the documents covers what JSON Schema cannot express
cheaply: the category fixes the block, the family prefix must agree with the
block, and a profile manifest must add up to its declared denominator.

Error strings always carry a JSON-pointer-like path and never raise, so the
CLI can print all problems of a document in one go.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "dailybench/task/2"
PROFILE_SCHEMA_VERSION = "dailybench/profile/2"
SUITE_SCHEMA_VERSION = "dailybench/suite/2"

BLOCKS: tuple[str, ...] = ("deterministic", "open")
"""The two directions of the corpus. ``category`` is kept for history."""

FAMILIES: tuple[str, ...] = (
    "det/math",
    "det/logic",
    "det/code",
    "det/tools",
    "det/data",
    "vis/sim",
    "vis/vqa",
    "vis/gen",
    "vis/art",
    "vis/aes",
)
"""Families of the taxonomy; the prefix is fixed by :data:`BLOCKS`."""

CATEGORIES: tuple[str, ...] = tuple("ABCDEFGHIJKL")

CATEGORY_BLOCK: dict[str, str] = {
    **{key: "deterministic" for key in "ABCDEFGHIJ"},
    "K": "open",
    "L": "open",
}
"""Direction is a property of the category, not of a hand-written field."""

CATEGORY_FAMILY: dict[str, str] = {
    "A": "det/logic",
    "B": "det/math",
    "C": "det/code",
    "D": "det/tools",
    "E": "det/data",
    "F": "det/logic",
    "G": "det/logic",
    "H": "det/logic",
    "I": "det/data",
    "J": "det/code",
    "K": "vis/sim",
    "L": "vis/art",
}
"""Default family of a category (a category may host more than one)."""

_REPO_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"
SCHEMA_DIR = _REPO_SCHEMA_DIR if _REPO_SCHEMA_DIR.is_dir() else Path(sys.prefix) / "schema"
"""Contract documents: next to the repository, or in a data-files install."""

TASK_SCHEMA_PATH = SCHEMA_DIR / "task.schema.json"
PROFILE_SCHEMA_PATH = SCHEMA_DIR / "profile.schema.json"

_LANGUAGE = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")

_TASK_FIELDS = (
    "id",
    "category",
    "block",
    "family",
    "languages",
    "cultures",
    "fresh_after",
    "private",
    "holdout_seeds",
    "title",
    "prompt",
    "system",
    "workspace",
    "tools",
    "dialogue",
    "meta",
    "oracle",
)


def block_for(category: str) -> str | None:
    """Default block of a category, ``None`` for an unknown category."""
    return CATEGORY_BLOCK.get(category)


def family_for(category: str) -> str | None:
    """Default family of a category, ``None`` for an unknown category."""
    return CATEGORY_FAMILY.get(category)


# --------------------------------------------------------------------------- #
# Reading the published documents
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_schema() -> dict[str, Any]:
    return _load(str(TASK_SCHEMA_PATH))


def profile_schema() -> dict[str, Any]:
    return _load(str(PROFILE_SCHEMA_PATH))


# --------------------------------------------------------------------------- #
# Small JSON Schema (2020-12) subset validator
# --------------------------------------------------------------------------- #


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _type_matches(value: Any, expected: str) -> bool:
    actual = _json_type(value)
    if expected == "number":
        return actual in ("number", "integer")
    return actual == expected


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _resolve(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        return {}
    node: Any = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or token not in node:
            return {}
        node = node[token]
    return node if isinstance(node, Mapping) else {}


def _one_of_errors(value: Any, options: Iterable[Any], root: Mapping[str, Any], path: str) -> list[str]:
    matches = 0
    for option in options:
        if isinstance(option, Mapping) and not _validate(value, option, root, path):
            matches += 1
    return [] if matches == 1 else [f"{path}: expected exactly one matching variant, matched {matches}"]


def _validate(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> list[str]:
    if "$ref" in schema:
        return _validate(value, _resolve(root, str(schema["$ref"])), root, path)

    errors: list[str] = []
    if "type" in schema:
        expected = schema["type"]
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_type_matches(value, name) for name in names):
            errors.append(f"{path}: expected {' or '.join(names)}, got {_json_type(value)}")
            return errors

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(item) for item in schema["enum"])
        errors.append(f"{path}: expected one of [{allowed}], got {value!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(str(schema["pattern"]), value):
            errors.append(f"{path}: does not match {schema['pattern']}")
        if schema.get("format") == "date":
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                errors.append(f"{path}: expected an ISO date (YYYY-MM-DD), got {value!r}")
            else:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    errors.append(f"{path}: invalid calendar date {value!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: less than {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: must be greater than {schema['exclusiveMinimum']}")

    if isinstance(value, (list, tuple)):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen: set[str] = set()
            for item in value:
                marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                if marker in seen:
                    errors.append(f"{path}: duplicate item {item!r}")
                seen.add(marker)
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(_validate(item, item_schema, root, f"{path}/{index}"))

    if isinstance(value, Mapping):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties") or {}
        patterns = schema.get("patternProperties") or {}
        for key, item in value.items():
            matched = key in properties
            if matched:
                errors.extend(_validate(item, properties[key], root, f"{path}/{_escape(str(key))}"))
            for pattern, subschema in patterns.items():
                if re.search(str(pattern), str(key)):
                    matched = True
                    errors.extend(_validate(item, subschema, root, f"{path}/{_escape(str(key))}"))
            if not matched and schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")

    for option in schema.get("allOf", ()):
        errors.extend(_validate(value, option, root, path))
    if "anyOf" in schema:
        if any(not _validate(value, option, root, path) for option in schema["anyOf"]):
            pass
        else:
            errors.append(f"{path}: matches no allowed variant")
    if "oneOf" in schema:
        errors.extend(_one_of_errors(value, schema["oneOf"], root, path))
    return errors


# --------------------------------------------------------------------------- #
# Task documents
# --------------------------------------------------------------------------- #


def task_to_dict(task: Any) -> dict[str, Any]:
    """Serialise a ``TaskInstance`` (or a mapping) to the contract document."""
    if isinstance(task, Mapping):
        doc = {str(key): value for key, value in task.items()}
        for key in ("block", "family"):
            if doc.get(key) is None:
                doc.pop(key, None)
        if "points" not in doc and "max_points" in doc:
            doc["points"] = doc.pop("max_points")
        if "schema" not in doc:
            doc["schema"] = SCHEMA_VERSION
        category = str(doc.get("category") or "")
        doc.setdefault("block", block_for(category))
        doc.setdefault("family", family_for(category))
        doc.setdefault("languages", ["ru"])
        doc.setdefault("cultures", [])
        doc.setdefault("fresh_after", None)
        doc.setdefault("private", False)
        doc.setdefault("holdout_seeds", [])
        return doc

    doc = {"schema": SCHEMA_VERSION}
    for name in _TASK_FIELDS:
        value = getattr(task, name, None)
        doc[name] = list(value) if isinstance(value, tuple) else value
    doc["points"] = float(getattr(task, "max_points"))
    return doc


def _task_dict(task: Any) -> dict[str, Any]:
    return task_to_dict(task)


def _is_task_document(task: Any) -> bool:
    """A mapping or a ``TaskInstance``-shaped object, not a scalar or a list."""
    return isinstance(task, Mapping) or hasattr(task, "max_points")


def _id_schema() -> re.Pattern[str]:
    """The one id contract: the same expression the gallery turns into a file name."""
    from . import gallery

    return gallery._ID_SCHEMA


def _semantic_task_errors(doc: Mapping[str, Any], prefix: str, *, cross_only: bool = False) -> list[str]:
    errors: list[str] = []
    category = doc.get("category")
    block = doc.get("block")
    family = doc.get("family")
    if category not in CATEGORY_BLOCK:
        errors.append(f"{prefix}/category: unknown category {category!r}")
    else:
        expected = CATEGORY_BLOCK[category]
        if block != expected:
            errors.append(f"{prefix}/block: category {category} is {expected}, got {block!r}")

    if block in BLOCKS and family in FAMILIES:
        expected_prefix = "det/" if block == "deterministic" else "vis/"
        if not str(family).startswith(expected_prefix):
            errors.append(f"{prefix}/family: block {block} requires a {expected_prefix}* family, got {family!r}")

    prompt = doc.get("prompt")
    if prompt is None:
        if doc.get("private") is not True:
            errors.append(f"{prefix}/prompt: a public task needs prompt text")
        elif not isinstance(doc.get("prompt_sha256"), str):
            errors.append(f"{prefix}/prompt_sha256: a private prompt stub needs its hash")
    elif isinstance(prompt, str) and not prompt.strip():
        errors.append(f"{prefix}/prompt: prompt is empty")

    identifier = doc.get("id")
    if isinstance(identifier, str) and not _id_schema().fullmatch(identifier):
        errors.append(
            f"{prefix}/id: invalid task id {identifier!r}: expected the file-name form K-1 "
            "(letters, digits and inner dashes; no '/')"
        )

    if cross_only:
        return errors

    languages = doc.get("languages")
    if not isinstance(languages, list) or not languages:
        errors.append(f"{prefix}/languages: at least one language is required")
    else:
        for index, language in enumerate(languages):
            if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
                errors.append(f"{prefix}/languages/{index}: invalid language tag {language!r}")

    cultures = doc.get("cultures")
    if isinstance(cultures, list):
        for index, culture in enumerate(cultures):
            if not isinstance(culture, str) or not culture.strip():
                errors.append(f"{prefix}/cultures/{index}: expected a non-empty string")

    fresh_after = doc.get("fresh_after")
    if fresh_after is not None and isinstance(fresh_after, str):
        try:
            date.fromisoformat(fresh_after)
        except ValueError:
            errors.append(f"{prefix}/fresh_after: invalid ISO date {fresh_after!r}")

    private = doc.get("private")
    if not isinstance(private, bool):
        errors.append(f"{prefix}/private: expected boolean")

    seeds = doc.get("holdout_seeds")
    if isinstance(seeds, list):
        for index, seed in enumerate(seeds):
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                errors.append(f"{prefix}/holdout_seeds/{index}: expected a non-negative integer")

    points = doc.get("points")
    if isinstance(points, bool) or not isinstance(points, (int, float)):
        errors.append(f"{prefix}/points: expected a number")
    elif points <= 0:
        errors.append(f"{prefix}/points: must be greater than 0, got {points}")

    return errors


def validate_task(task: Any, prefix: str = "task") -> list[str]:
    """All contract violations of one task document (never raises)."""
    if not _is_task_document(task):
        return [f"{prefix}: expected object, got {_json_type(task)}"]
    doc = _task_dict(task)
    errors = _validate(doc, task_schema(), task_schema(), prefix)
    return errors + _semantic_task_errors(doc, prefix, cross_only=True)


def require_valid_task(task: Any, prefix: str = "task") -> None:
    errors = validate_task(task, prefix)
    if errors:
        raise ValueError("; ".join(errors))


def validate_task_fields(task: Any, prefix: str = "task") -> list[str]:
    """Semantic-only check used by ``TaskInstance.__post_init__``."""
    if not _is_task_document(task):
        return [f"{prefix}: expected object, got {_json_type(task)}"]
    return _semantic_task_errors(_task_dict(task), prefix)


# --------------------------------------------------------------------------- #
# Profile and suite documents
# --------------------------------------------------------------------------- #


def profile_document(profile: str) -> dict[str, Any]:
    """A profile manifest for the exported suite."""
    from . import core

    return {
        "schema": PROFILE_SCHEMA_VERSION,
        "profile": profile,
        "frozen": True,
        "total": round(core.profile_total(profile), 6),
        "categories": {
            key: round(core.points_for(key, profile) or 0.0, 6) for key in CATEGORIES
        },
    }


def validate_profile(doc: Any, prefix: str = "profile") -> list[str]:
    from . import core

    if not isinstance(doc, Mapping):
        return [f"{prefix}: expected object, got {_json_type(doc)}"]
    errors = _validate(doc, profile_schema(), profile_schema(), prefix)
    profile = doc.get("profile")
    if isinstance(profile, str) and profile not in core.PROFILES:
        errors.append(f"{prefix}/profile: unknown profile {profile!r}")
    categories = doc.get("categories")
    total = doc.get("total")
    if isinstance(categories, Mapping) and isinstance(total, (int, float)) and not isinstance(total, bool):
        summed = sum(value for value in categories.values() if isinstance(value, (int, float)))
        if abs(summed - total) > 1e-6:
            errors.append(f"{prefix}/total: does not add up to the category sums ({total} != {summed})")
        if profile in core.PROFILES and abs(total - core.profile_total(profile)) > 1e-6:
            errors.append(
                f"{prefix}/total: profile {profile} declares {core.profile_total(profile)}, got {total}"
            )
    return errors


def _private_stub(task: Any) -> dict[str, Any]:
    """Published form of a private task: metadata, denominator and hashes.

    Fields are taken from the serialised document, so a mapping and a stub
    whose ``prompt`` is already ``None`` hash exactly what is published.
    """
    from .decontam import prompt_hashes

    doc = task_to_dict(task)
    return {
        "schema": doc["schema"],
        "id": doc["id"],
        "category": doc["category"],
        "block": doc["block"],
        "family": doc["family"],
        "languages": doc["languages"],
        "cultures": doc["cultures"],
        "fresh_after": doc["fresh_after"],
        "private": True,
        "holdout_seeds": doc["holdout_seeds"],
        "points": doc["points"],
        "prompt": None,
        **prompt_hashes(doc),
    }


def _document_is_private(task: Any) -> bool:
    if isinstance(task, Mapping):
        return task.get("private") is True
    return getattr(task, "private", False) is True


def export_document(
    profile: str,
    tasks: Iterable[Any],
    *,
    include_private: bool = False,
) -> dict[str, Any]:
    """Suite document ready to be written as JSON or YAML.

    A private task is exported as its hash stub: the denominator stays whole
    and the prompt text never leaves the machine unless ``include_private`` is
    set.
    """
    items = [
        task_to_dict(task)
        if include_private or not _document_is_private(task)
        else _private_stub(task)
        for task in tasks
    ]
    return {
        "schema": SUITE_SCHEMA_VERSION,
        "profile": profile_document(profile),
        "tasks": items,
    }


def validate_suite_doc(doc: Any, *, check_totals: bool = False, prefix: str = "suite") -> list[str]:
    from . import core

    if not isinstance(doc, Mapping):
        return [f"{prefix}: expected object, got {_json_type(doc)}"]
    errors: list[str] = []
    if doc.get("schema") != SUITE_SCHEMA_VERSION:
        errors.append(f"{prefix}/schema: expected {SUITE_SCHEMA_VERSION!r}, got {doc.get('schema')!r}")

    profile_doc = doc.get("profile")
    errors.extend(validate_profile(profile_doc, f"{prefix}/profile"))
    profile = profile_doc.get("profile") if isinstance(profile_doc, Mapping) else None

    tasks = doc.get("tasks")
    if not isinstance(tasks, list):
        errors.append(f"{prefix}/tasks: expected array, got {_json_type(tasks)}")
        return errors
    if not tasks:
        errors.append(f"{prefix}/tasks: the suite is empty")
    seen: set[str] = set()
    totals: dict[str, float] = {}
    for index, task in enumerate(tasks):
        where = f"{prefix}/tasks/{index}"
        errors.extend(validate_task(task, where))
        if isinstance(task, Mapping):
            identifier = task.get("id")
            if isinstance(identifier, str):
                if identifier in seen:
                    errors.append(f"{where}/id: duplicate task id {identifier!r}")
                seen.add(identifier)
            category = task.get("category")
            points = task.get("points")
            if isinstance(category, str) and isinstance(points, (int, float)) and not isinstance(points, bool):
                totals[category] = totals.get(category, 0.0) + float(points)

    if check_totals and profile in core.PROFILES:
        for category, total in totals.items():
            expected = core.points_for(category, profile)
            if expected is not None and abs(total - expected) > 1e-6:
                errors.append(
                    f"{prefix}/tasks: category {category} weighs {total}, profile declares {expected}"
                )
    return errors
