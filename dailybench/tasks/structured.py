"""Category G — structured output (JSON Schema conformance), 4 scenarios × 2 points.

Every scenario hands the model facts in prose and asks for one strict JSON
document. Scoring is therefore split the same way everywhere: one point for the
shape, one for the content.

* ``schema`` — the reply parses as strict JSON (no fence, no prose around it) and
  the parsed payload conforms to the JSON Schema subset declared in
  ``oracle["schema"]``. The subset is validated by ``validate`` in this module:
  ``type`` (object/array/string/integer/number/boolean/null), ``required``,
  ``properties``, ``additionalProperties=false``, ``enum``, ``items``,
  ``minItems``/``maxItems``, ``minimum``/``maximum``, ``pattern``.
* ``content`` — the parsed payload equals the oracle exactly. For the two array
  scenarios that includes the order: the prompt demands dedupe and a named sort
  key, and an array in the wrong order is a wrong answer even though it is a
  well-formed one.

Strict JSON is what makes the shape point separable from the content point: a
fenced answer still parses (``text.strict_json`` returns the payload with
``strict=False``), so it loses the shape point and can keep the content point.
Each fixture also carries a ``distractor`` — the answer a plausible wrong
solution produces — asserted at generation time to differ from the reference.

Scenarios:

* G-1 — nested incident card: required sub-object, enum, bounded integer,
  ``additionalProperties=false``.
* G-2 — extraction from a dirty event dump: ISO-8601 normalisation, severity
  enum mapping, dedupe, sorting by date then service.
* G-3 — array of service records: dedupe by id, bounded ``minItems``/``maxItems``,
  typed fields, id checked by ``pattern``.
* G-4 — repair of a draft config: string digits to integer, unknown keys dropped,
  missing optional fields stay missing, enum lower-cased; the repaired object
  must equal the oracle exactly.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import random
import re
from typing import Any

from .. import core, text

IDS = ("G-1", "G-2", "G-3", "G-4")

_POINT_SCHEMA = 1.0
_POINT_CONTENT = 1.0

SERVICES = ("api-01", "api-02", "web-01", "web-02", "db-01", "queue-03", "cache-07")
LEVELS = ("INFO", "WARN", "ERROR")
TEAMS = ("platform", "payments", "infra", "search", "growth")
OWNERS = ("nora", "ivan", "mei", "alex", "zoe", "oleg", "kira", "pavel", "lena", "timur")
MODES = ("fast", "safe")
SERVICE_NAMES = ("gateway", "ingest", "billing", "scheduler", "notifier", "router")

# Severity spellings the dirty dump actually contains. None of them is the
# canonical enum value, so the answer always has to normalise at least the case.
_SEVERITY_WORDS = {
    "INFO": ("info", "inf"),
    "WARN": ("warn", "warning"),
    "ERROR": ("error", "err"),
}

_BASE_TIME = dt.datetime(2026, 3, 4, 9, 15)


# --------------------------------------------------------------------------- #
# JSON Schema subset validator
# --------------------------------------------------------------------------- #
#
# Deliberately small: it covers exactly the keywords the scenarios use, so the
# oracle can be checked against its own schema without pulling in a dependency.
# ``validate`` returns human-readable errors (empty list == conforms) instead of
# a bare bool, because the evaluator prints the first error in the check detail.

_SUPPORTED_TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")
_SUPPORTED_KEYWORDS = (
    "type",
    "required",
    "properties",
    "additionalProperties",
    "enum",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "pattern",
)


def _json_type(value: Any) -> str:
    """JSON type name of a Python value, with ``bool`` kept out of ``integer``."""
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
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "number":
        # JSON Schema: an integer is a number. Exclude bool, which is an int in Python.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return _json_type(value) == expected


def validate(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Validate ``payload`` against the supported schema subset.

    Returns the list of violations, empty when the payload conforms. ``$`` is the
    root path; nested paths look like ``$.owner.team`` or ``$[2].ts``.
    """
    errors: list[str] = []
    _validate(payload, schema, "$", errors)
    return errors


def _validate(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if expected is not None and not _type_matches(value, expected):
        errors.append(f"{path}: ожидался тип {expected}, получен {_json_type(value)}")
        return  # the remaining keywords are meaningless for a mistyped value

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: значение {value!r} не входит в enum {schema['enum']}")

    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: отсутствует обязательное поле {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: лишнее поле {key!r} (additionalProperties=false)")
        for key, sub in properties.items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}", errors)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: элементов {len(value)}, нужно не меньше {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: элементов {len(value)}, нужно не больше {schema['maxItems']}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]", errors)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} меньше минимума {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} больше максимума {schema['maximum']}")

    if isinstance(value, str) and "pattern" in schema:
        if re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: {value!r} не соответствует шаблону {schema['pattern']}")


def check_schema(schema: dict[str, Any], path: str = "$") -> None:
    """Assert the schema document itself uses only the supported subset.

    Called from ``generate`` for every fixture: a schema the validator cannot
    fully interpret would silently weaken the shape point.
    """
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: схема должна быть объектом")
    unknown = [key for key in schema if key not in _SUPPORTED_KEYWORDS]
    if unknown:
        raise ValueError(f"{path}: неподдерживаемые ключи схемы {unknown}")
    kind = schema.get("type")
    if kind is not None and kind not in _SUPPORTED_TYPES:
        raise ValueError(f"{path}: неподдерживаемый тип {kind!r}")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}: enum должен быть списком")
    if "required" in schema and not isinstance(schema["required"], list):
        raise ValueError(f"{path}: required должен быть списком")
    if "pattern" in schema and not isinstance(schema["pattern"], str):
        raise ValueError(f"{path}: pattern должен быть строкой")
    for key, sub in (schema.get("properties") or {}).items():
        check_schema(sub, f"{path}.{key}")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def _dump(value: Any) -> str:
    """Canonical reference text: strict JSON, no fence, no trailing prose."""
    return json.dumps(value, ensure_ascii=False)


def _finish(
    task_id: str,
    prompt: str,
    schema: dict[str, Any],
    answer: Any,
    distractor: Any,
    *,
    reference: Any | None = None,
) -> core.TaskInstance:
    """Build a task and assert every claim its scenario makes about the fixture.

    ``reference`` is the canonical reply text when it is not the plain dump of
    ``answer`` (unused today, kept for symmetry with the other categories).
    """
    check_schema(schema)
    reference_text = reference if reference is not None else _dump(answer)

    # The fixture must be well-formed: a request whose own reference cannot pass
    # its schema could never score, and the self-check would report it as a bug.
    errors = validate(answer, schema)
    assert not errors, f"{task_id}: эталон не проходит собственную схему: {errors}"
    assert json.loads(reference_text) == answer, f"{task_id}: reference — не дамп эталона"

    # And the plausible wrong answer must actually be wrong: otherwise the
    # scenario cannot separate a correct model from a sloppy one.
    assert distractor != answer, f"{task_id}: дистрактор совпал с эталоном"

    return core.TaskInstance(
        id=task_id,
        category="G",
        title="Structured output",
        prompt=prompt,
        max_points=_POINT_SCHEMA + _POINT_CONTENT,
        oracle={
            "schema": schema,
            "answer": answer,
            "reference": reference_text,
            "distractor": distractor,
        },
        meta={"scenario": task_id},
    )


# --------------------------------------------------------------------------- #
# G-1 — nested incident card
# --------------------------------------------------------------------------- #


def _g1(rng: random.Random) -> core.TaskInstance:
    service = rng.choice(SERVICES)
    level = rng.choice(LEVELS)
    latency = rng.randrange(5, 4000)
    owner = {"name": rng.choice(OWNERS), "team": rng.choice(TEAMS)}

    schema = {
        "type": "object",
        "required": ["service", "level", "latency_ms", "owner"],
        "additionalProperties": False,
        "properties": {
            "service": {"type": "string"},
            "level": {"type": "string", "enum": list(LEVELS)},
            "latency_ms": {"type": "integer", "minimum": 1, "maximum": 9999},
            "owner": {
                "type": "object",
                "required": ["name", "team"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "team": {"type": "string", "enum": list(TEAMS)},
                },
            },
        },
    }

    answer = {"service": service, "level": level, "latency_ms": latency, "owner": owner}
    # A model that "helpfully" annotates the card adds a field the schema forbids.
    distractor = copy.deepcopy(answer)
    distractor["note"] = "заполнено автоматически"

    prompt = (
        "Заполни карточку инцидента по данным ниже.\n\n"
        f"Данные:\n- сервис: {service}\n- уровень: {level}\n- задержка: {latency} мс\n"
        f"- владелец: {owner['name']}, команда: {owner['team']}\n\n"
        "Верни ТОЛЬКО один JSON-объект без Markdown и без пояснений, ровно такой формы:\n\n"
        '{\n  "service": "<сервис>",\n  "level": "<одно из: INFO, WARN, ERROR>",\n'
        '  "latency_ms": <целое от 1 до 9999>,\n'
        '  "owner": {"name": "<владелец>", "team": "<одно из: '
        + ", ".join(TEAMS)
        + '>"}\n}\n\n'
        "Ровно эти ключи, других полей не добавляй. Бери значения из данных, ничего не выдумывай."
    )
    return _finish("G-1", prompt, schema, answer, distractor)


# --------------------------------------------------------------------------- #
# G-2 — extraction from a dirty event dump
# --------------------------------------------------------------------------- #


def _render_event(when: dt.datetime, service: str, severity: str, latency: int, style: int) -> str:
    """One messy source line; both styles carry the same facts, differently."""
    if style == 0:
        return f"{when:%d.%m.%Y %H:%M} | {service} | {severity} | latency={latency}ms"
    return f"{when:%d/%m/%Y %H:%M};{service};{severity};{latency}ms"


def _g2(rng: random.Random) -> core.TaskInstance:
    count = rng.randrange(4, 7)
    services = rng.sample(SERVICES, count)
    minutes = sorted(rng.sample(range(0, 600), count))
    minutes[1] = minutes[0]  # two events share a timestamp: the sort needs its tie-break

    events = []
    for index in range(count):
        events.append(
            {
                "when": _BASE_TIME + dt.timedelta(minutes=minutes[index]),
                "service": services[index],
                "severity": rng.choice(LEVELS),
                "latency": rng.randrange(50, 5000),
            }
        )

    # Every event is printed twice (two formats, two severity spellings), so the
    # dump contains exactly the duplicates the prompt tells the model to drop.
    lines: list[str] = []
    for event in events:
        for style in (0, 1):
            word = rng.choice(_SEVERITY_WORDS[event["severity"]])
            lines.append(_render_event(event["when"], event["service"], word, event["latency"], style))
    rng.shuffle(lines)

    dump = (
        "# формат строк смешан; повторы возможны\n"
        "# мусорная строка ниже не является событием\n"
        "collector: buffer flushed, 0 events lost\n"
        + "\n".join(lines)
    )

    items = [
        {
            "ts": event["when"].strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "service": event["service"],
            "severity": event["severity"],
        }
        for event in events
    ]
    answer = sorted(items, key=lambda item: (item["ts"], item["service"]))

    schema = {
        "type": "array",
        "minItems": count,
        "maxItems": count,
        "items": {
            "type": "object",
            "required": ["ts", "service", "severity"],
            "additionalProperties": False,
            "properties": {
                "ts": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"},
                "service": {"type": "string"},
                "severity": {"type": "string", "enum": list(LEVELS)},
            },
        },
    }

    # Echoing a source spelling instead of the canonical enum value is the
    # classic extraction mistake.
    distractor = copy.deepcopy(answer)
    distractor[0]["severity"] = distractor[0]["severity"].lower()

    prompt = (
        "Ниже — грязный выхлоп сборщика событий: строки в разных форматах, с повторами "
        "и мусорными строками в начале.\n\n"
        f"{dump}\n\n"
        "Собери из него массив событий и верни ТОЛЬКО JSON-массив без Markdown и без пояснений.\n"
        'Каждый элемент: {"ts": "<ISO-8601 UTC>", "service": "<сервис>", '
        '"severity": "<одно из: INFO, WARN, ERROR>"}.\n'
        "Правила:\n"
        "- даты привести к виду 2026-03-04T09:15:00Z (все входные даты уже в UTC);\n"
        "- уровни привести к каноническому виду: info → INFO, warn/warning → WARN, error/err → ERROR;\n"
        f"- дубликаты одного и того же события убрать: остаться должно столько событий, "
        f"сколько уникальных (в этом дампе — {count});\n"
        "- порядок: по ts по возрастанию, при равных ts — по service по возрастанию;\n"
        "- латентность в ответ не входит, лишних полей нет."
    )

    task = _finish("G-2", prompt, schema, answer, distractor)
    # The dump really does contain duplicates (every event twice) — the prompt's
    # dedupe instruction would be vacuous otherwise.
    assert len(lines) == 2 * count
    assert len({(item["ts"], item["service"], item["severity"]) for item in answer}) == count
    return task


# --------------------------------------------------------------------------- #
# G-3 — array conformance with a patterned id
# --------------------------------------------------------------------------- #


def _g3(rng: random.Random) -> core.TaskInstance:
    count = rng.randrange(3, 6)
    ids = rng.sample(range(100, 1000), count)

    records = [
        {
            "id": f"svc-{sid:03d}",
            "name": rng.choice(SERVICE_NAMES),
            "replicas": rng.randrange(1, 13),
            "healthy": rng.choice((True, False)),
        }
        for sid in ids
    ]

    # Two renderings of each record, same values. Rotating the sorted list puts
    # the smallest id last, so the input is deterministically unsorted instead of
    # relying on a lucky sample order.
    ordered = sorted(records, key=lambda record: record["id"])
    rotated = ordered[1:] + ordered[:1]
    lines: list[str] = []
    for record in rotated:
        lines.append(
            f'{record["id"]} name={record["name"]} replicas={record["replicas"]} '
            f'healthy={str(record["healthy"]).lower()}'
        )
        lines.append(
            f'{record["id"]},{record["name"]},{record["replicas"]},{str(record["healthy"]).lower()}'
        )

    answer = ordered

    schema = {
        "type": "array",
        "minItems": count,
        "maxItems": count,
        "items": {
            "type": "object",
            "required": ["id", "name", "replicas", "healthy"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "pattern": r"^svc-\d{3}$"},
                "name": {"type": "string"},
                "replicas": {"type": "integer", "minimum": 0, "maximum": 99},
                "healthy": {"type": "boolean"},
            },
        },
    }

    # A well-formed array in the wrong order: the shape point survives, the
    # content point does not.
    distractor = list(reversed(answer))

    prompt = (
        "Ниже черновой список реплик сервисов: есть повторы и беспорядок, "
        "форматы строк смешаны.\n\n"
        + "\n".join(lines)
        + "\n\n"
        "Верни ТОЛЬКО JSON-массив без Markdown и без пояснений, ровно столько записей, "
        f"сколько уникальных id (в этом наборе — {count}), вида:\n"
        '{"id": "svc-NNN", "name": "<строка>", "replicas": <целое 0..99>, "healthy": <true|false>}\n'
        "Правила:\n"
        "- id уникальны; строки с одинаковым id описывают одну запись (данные у них совпадают), "
        "повторы удалить;\n"
        "- порядок — по id по возрастанию;\n"
        "- id имеет вид svc- и ровно три цифры (например svc-042);\n"
        "- других полей не добавлять."
    )

    task = _finish("G-3", prompt, schema, answer, distractor)
    # The input is genuinely unsorted and duplicated: sorting and dedupe both bite.
    assert len(lines) == 2 * count
    input_ids = [line.split()[0].split(",")[0] for line in lines]
    assert input_ids != sorted(input_ids), "вход G-3 должен быть не отсортирован"
    return task


# --------------------------------------------------------------------------- #
# G-4 — repair and coercion
# --------------------------------------------------------------------------- #


def _g4(rng: random.Random) -> core.TaskInstance:
    name = rng.choice(SERVICE_NAMES)
    port = rng.randrange(1024, 65000)
    mode = rng.choice(MODES)
    retries = rng.randrange(0, 11)

    draft = {
        "name": name,
        "port": str(port),
        "mode": mode.upper(),
        "retries": str(retries),
        "comment": "черновик от стажёра",
        "legacy": True,
    }

    schema = {
        "type": "object",
        "required": ["name", "port", "mode", "retries"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "mode": {"type": "string", "enum": list(MODES)},
            "retries": {"type": "integer", "minimum": 0, "maximum": 10},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
        },
    }

    answer = {"name": name, "port": port, "mode": mode, "retries": retries}
    # Forgetting rule 1: the port stays a string and the schema rejects it.
    distractor = {"name": name, "port": str(port), "mode": mode, "retries": retries}

    prompt = (
        "Ниже черновик конфига, который не проходит валидацию схемы. Приведи его к схеме.\n\n"
        f"Черновик:\n{_dump(draft)}\n\n"
        "Правила приведения:\n"
        "1. строки из одних цифр превратить в целые числа;\n"
        "2. неизвестные схеме ключи удалить;\n"
        "3. отсутствующие необязательные поля не добавлять;\n"
        "4. значение mode привести к нижнему регистру.\n\n"
        "Схема: обязательные поля — name (строка), port (целое 1..65535), "
        "mode (одно из: fast, safe), retries (целое 0..10); "
        "необязательное поле timeout (целое 1..600).\n\n"
        "Верни ТОЛЬКО исправленный JSON-объект без Markdown и без пояснений."
    )

    task = _finish("G-4", prompt, schema, answer, distractor)
    # Rule 3 is only observable because the optional field is absent from the draft.
    assert "timeout" not in draft and "timeout" not in answer
    return task


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

_SCENARIOS = (
    ("G-1", _g1),
    ("G-2", _g2),
    ("G-3", _g3),
    ("G-4", _g4),
)


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    return [builder(rng) for _, builder in _SCENARIOS]


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="G")
    schema = task.oracle["schema"]

    # ``strict`` is False for a fenced or prose-wrapped reply: the payload may
    # still parse, but the shape point is about delivering bare JSON.
    payload, strict = text.strict_json(tr.final_text)
    errors = validate(payload, schema)
    if not strict:
        detail = "ответ не является строгим JSON (обвязка или пояснения вокруг)"
    elif errors:
        detail = f"схема нарушена: {errors[0]}"
    else:
        detail = "строгий JSON, схема соблюдена"
    score.add("schema", strict and not errors, _POINT_SCHEMA, detail)

    expected = task.oracle["answer"]
    matches = payload == expected
    score.add(
        "content",
        matches,
        _POINT_CONTENT,
        "payload совпадает с эталоном" if matches else f"получено {payload!r}, ожидалось {expected!r}",
    )
    return score


core.register(
    key="G",
    title="Structured output",
    points=8.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
)
