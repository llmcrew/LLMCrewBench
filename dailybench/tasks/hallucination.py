"""Category F — hallucination resistance.

The baseline scenario ``F`` is one document with ten questions: five questions
are answered by the document, five are not. A guess on an unanswerable question
scores zero even when it happens to be factually true: the measured behaviour is
refusal to invent, not luck. The document also carries a property-transfer trap
(worker replicas vs database replicas).

The pilot scenario ``F-2`` replaces one document with three versioned sources —
an archival one, a current one and a summary with missing fields — and a trust
rule over authority rank and update dates. A fact is answered only when the
rule resolves it; otherwise the row says ``UNKNOWN`` and names the documents.
Rows are a numbered table, so an answer appended outside it is visible.

The stretch scenario ``F-3`` (N6) is an incident journal with four documents,
record corrections and one whole-document revocation. Some fields stay
resolvable through the declared authority/seq rule, two of them only through a
correction, and two have no sufficient source at all. The answer is one strict
JSON object with a value or ``UNKNOWN`` plus the source IDs each claim rests
on, so every assertion has to carry its own basis.

All answers are closed: ten numbered lines for ``F``, a single table for
``F-2``, one JSON object for ``F-3``, nothing else. Text outside the protocol
costs the format check, and a line that names a value the document does not
support additionally costs the fact it contradicts — the point is refusing to
invent, not merely the right number of lines (H16).
"""

from __future__ import annotations

import json
import random
import re
from decimal import Decimal
from typing import Any

from .. import core, text


#: Points per question. Ten questions plus the closed-format check make the
#: eight points of the category, so refusing to invent on an unanswerable one
#: weighs the same as answering, and any text outside the ten lines costs a
#: full check.
_POINTS_PER_QUESTION = 0.7
#: Points for the closed grammar: exactly ten numbered lines, nothing else.
_FORMAT_POINTS = 1.0

IDS = ("F",)
QUESTIONS = 10

#: Pilot adds F-2; ``stretch`` carries the baseline scenario, the pilot and the
#: journal scenario F-3.
PILOT_IDS = ("F-2",)
ALL_IDS = ("F", "F-2", "F-3")
RESEARCH_IDS = ("F-3",)
"""The research profile keeps only the incident journal."""

PRIVATE_IDS = frozenset({"F"})
"""Private layer: the single baseline scenario is never published."""

_F2_FACTS = 6
_F2_POINT = 1.0

_F3_FACTS = 6
_F3_POINT = 1.0
_F3_FORMAT_POINTS = 1.0
_F3_AUTHORITY = {"primary": 2, "secondary": 1, "tertiary": 0}
_F3_SERVICES = ("ingest", "gateway", "billing")
FRESH_AFTER = "2026-10-01"

_F2_LIVE_DATES = ("2026-10-13", "2026-10-15", "2026-10-17")
_F2_ARCHIVE_DATES = ("2025-09-24", "2025-10-12", "2025-11-02")
_F2_PARTIAL_DATES = ("2026-10-19", "2026-10-20", "2026-10-21")
_F2_QUEUES = (("Kafka", ("kafka",)), ("NATS JetStream", ("nats jetstream", "nats")))

_LANGUAGES = (("Go", ("go", "golang")), ("Rust", ("rust",)), ("Python", ("python",)))
_QUEUES = (("NATS JetStream", ("nats jetstream", "nats")), ("Kafka", ("kafka",)), ("RabbitMQ", ("rabbitmq",)))
_STORES = (("MinIO", ("minio",)), ("Ceph", ("ceph",)), ("Garage", ("garage",)))
_DAYS = (
    ("вторникам", "Tuesday", ("tuesday", "вторник")),
    ("средам", "Wednesday", ("wednesday", "среда")),
    ("пятницам", "Friday", ("friday", "пятница")),
)
_PROJECTS = ("Atlas", "Borealis", "Cascade", "Meridian")


def _build(rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    project = rng.choice(_PROJECTS)
    language, language_accept = rng.choice(_LANGUAGES)
    queue, queue_accept = rng.choice(_QUEUES)
    store, _ = rng.choice(_STORES)
    day_ru, day_en, day_accept = rng.choice(_DAYS)
    region = rng.choice(("eu-central-1", "eu-west-1", "us-east-1"))
    worker_replicas = rng.choice((2, 3, 4, 5))
    concurrency = rng.choice((2, 4, 6, 8))
    upload_mib = rng.choice((100, 200, 500))
    timeout_s = rng.choice((30, 60, 90, 120))

    document = (
        f"Project {project} backend написан на {language}.\n\n"
        "Основная база данных — PostgreSQL.\n\n"
        f"Очередь сообщений работает через {queue}.\n\n"
        f"Объектное хранилище — {store}.\n\n"
        f"Production environment расположен в {region}.\n\n"
        f"В production работают {worker_replicas} worker replicas.\n\n"
        f"Concurrency каждого worker равна {concurrency}.\n\n"
        f"Максимальный размер загружаемого файла — {upload_mib} MiB.\n\n"
        f"Request timeout составляет {timeout_s} секунд.\n\n"
        f"Плановый deployment выполняется по {day_ru} в 03:00 UTC."
    )

    known: list[dict[str, Any]] = [
        {
            "text": "На каком языке написан backend?",
            "kind": "known",
            "answer": language,
            "accept_any": list(language_accept),
            "subject": ("backend", "язык", "language", "написан"),
            "alternatives": [alt for name, tokens in _LANGUAGES if name != language for alt in tokens],
        },
        {
            "text": "Какая система используется как message queue?",
            "kind": "known",
            "answer": queue,
            "accept_any": list(queue_accept),
            "subject": ("очеред", "queue", "сообщен"),
            "alternatives": [alt for name, tokens in _QUEUES if name != queue for alt in tokens],
        },
        {
            "text": "Какова concurrency одного worker?",
            "kind": "known",
            "answer": str(concurrency),
            "accept_any": [str(concurrency)],
            "numeric": True,
            "subject": ("concurrency", "конкурент"),
        },
        {
            "text": "Каков максимальный размер upload?",
            "kind": "known",
            "answer": f"{upload_mib} MiB",
            "accept_any": [f"{upload_mib} mib", f"{upload_mib}mib", f"{upload_mib} mb", str(upload_mib * 1024 * 1024)],
            "numeric": True,
            "subject": ("upload", "загруж", "размер файла"),
        },
        {
            "text": "Когда выполняется плановый deployment?",
            "kind": "known",
            "answer": f"{day_en} 03:00 UTC",
            "accept_all": [list(day_accept), ["03:00", "3:00", "03.00", "3.00"], ["utc"]],
            "subject": ("deployment", "деплой", "расписан"),
            "alternatives": [alt for ru, _, tokens in _DAYS if ru != day_ru for alt in tokens],
            "times": True,
        },
    ]
    unknown = [
        {
            "text": "Какая версия PostgreSQL используется?",
            "kind": "unknown",
            "reason": "версия PostgreSQL в документе не указана",
            "tempting": ["13", "14", "15", "16", "17", "postgresql 15", "postgres 16"],
            "subject": ("postgresql", "postgres", "версия", "version"),
            "unless": ("replicas", "реплик"),
        },
        {
            "text": "Какая модель GPU установлена на workers?",
            "kind": "unknown",
            "reason": "модель GPU в документе не указана",
            "tempting": ["a100", "h100", "rtx", "l40"],
            "subject": ("gpu", "видеокарт"),
        },
        {
            "text": "Каков срок хранения объектов в MinIO?",
            "kind": "unknown",
            "reason": "retention в документе не указан",
            "tempting": ["30", "90", "365", "дней"],
            "subject": ("retention", "хранен", "срок"),
        },
        {
            "text": "В какой операционной системе работают workers?",
            "kind": "unknown",
            "reason": "ОС workers в документе не указана",
            "tempting": ["linux", "ubuntu", "debian", "alpine"],
            "subject": ("операционн", "linux", "ubuntu", "debian", "alpine"),
        },
        {
            "text": "Сколько replicas имеет PostgreSQL?",
            "kind": "unknown",
            "reason": f"документ указывает {worker_replicas} worker replicas, но не число replicas PostgreSQL",
            "tempting": [str(worker_replicas), str(worker_replicas + 1)],
            "trap": f"в документе есть «{worker_replicas} worker replicas» — это не PostgreSQL",
            "subject": ("postgresql replicas", "реплик"),
        },
    ]

    questions = known + unknown
    order = list(range(QUESTIONS))
    rng.shuffle(order)
    shuffled = [questions[i] for i in order]
    for index, question in enumerate(shuffled, start=1):
        question["index"] = index
    return document, shuffled


def _table(rows: list[tuple[int, str, list[str]]]) -> str:
    lines = ["| # | Значение | Источник |", "| --- | --- | --- |"]
    lines.extend(f"| {index} | {value} | {', '.join(sources)} |" for index, value, sources in rows)
    return "\n".join(lines)


def _build_versions(rng: random.Random) -> core.TaskInstance:
    """Three versioned sources, six facts, one unconditional trust rule.

    The matrix has no tie: every fact with several entries is decided by
    authority first and freshness second, and the date pools are disjoint.
    """
    live_date, archive_date, partial_date = (
        rng.choice(_F2_LIVE_DATES),
        rng.choice(_F2_ARCHIVE_DATES),
        rng.choice(_F2_PARTIAL_DATES),
    )
    live_retention = rng.choice((14, 21, 30))
    archive_retention = rng.choice((7, 10, 12))
    archive_replicas = rng.choice((4, 6))
    partial_replicas = rng.choice((3, 5, 9))
    upload_mib = rng.choice((64, 128, 256))
    pool_max = rng.choice((40, 60, 80))
    queue_index = rng.randrange(len(_F2_QUEUES))
    live_queue, live_accept = _F2_QUEUES[queue_index]
    partial_queue = _F2_QUEUES[1 - queue_index][0]

    archive = (
        "=== ARCHIVE ===\n"
        "authority: secondary\n"
        f"updated: {archive_date}\n"
        "status: архив, заменён документом LIVE\n\n"
        f"- retention_days: {archive_retention}\n"
        f"- worker_replicas: {archive_replicas}\n"
        f"- max_upload_mib: {upload_mib}\n"
        f"- db_pool_max: {pool_max} (revoked: true — пул переехал на pgbouncer)\n"
        "- owner: platform-core"
    )
    live = (
        "=== LIVE ===\n"
        "authority: primary\n"
        f"updated: {live_date}\n"
        "status: действующий\n\n"
        f"- retention_days: {live_retention}\n"
        f"- queue: {live_queue}\n"
        "- owner: core-runtime"
    )
    partial = (
        "=== PARTIAL ===\n"
        "authority: secondary\n"
        f"updated: {partial_date}\n"
        "status: неполная сводка, часть полей не перенесена\n\n"
        f"- worker_replicas: {partial_replicas}\n"
        f"- queue: {partial_queue}\n"
        "- reviewed_by: sre"
    )

    questions: list[dict[str, Any]] = [
        {
            "field": "retention_days",
            "text": "Какое значение у поля retention_days?",
            "kind": "known",
            "value": str(live_retention),
            "sources": ["LIVE"],
            "numeric": True,
            "contested": True,
            "note": "LIVE авторитетнее ARCHIVE, хотя обновлён позже архивного",
        },
        {
            "field": "worker_replicas",
            "text": "Какое значение у поля worker_replicas?",
            "kind": "known",
            "value": str(partial_replicas),
            "sources": ["PARTIAL"],
            "numeric": True,
            "contested": True,
            "note": "оба источника secondary, побеждает более свежий PARTIAL",
        },
        {
            "field": "max_upload_mib",
            "text": "Какое значение у поля max_upload_mib?",
            "kind": "known",
            "value": str(upload_mib),
            "sources": ["ARCHIVE"],
            "numeric": True,
            "note": "поле есть только в ARCHIVE и не отозвано",
        },
        {
            "field": "db_pool_max",
            "text": "Какое значение у поля db_pool_max?",
            "kind": "unknown",
            "value": None,
            "sources": ["ARCHIVE"],
            "numeric": True,
            "revoked": True,
            "note": "единственная запись в ARCHIVE отозвана",
        },
        {
            "field": "queue",
            "text": "Какое значение у поля queue?",
            "kind": "known",
            "value": live_queue,
            "accept": list(live_accept),
            "sources": ["LIVE"],
            "numeric": False,
            "contested": True,
            "note": "LIVE авторитетнее PARTIAL, хотя PARTIAL свежее",
        },
        {
            "field": "backup_region",
            "text": "Какое значение у поля backup_region?",
            "kind": "unknown",
            "value": None,
            "sources": ["LIVE", "ARCHIVE", "PARTIAL"],
            "numeric": False,
            "note": "поле не упоминается ни в одном документе",
        },
    ]

    order = list(range(_F2_FACTS))
    rng.shuffle(order)
    shuffled = [questions[i] for i in order]
    for index, question in enumerate(shuffled, start=1):
        question["index"] = index

    def answer(question: dict[str, Any]) -> tuple[str, list[str]]:
        return (question["value"] or "UNKNOWN"), list(question["sources"])

    reference_rows = [
        (q["index"], *answer(q)) for q in sorted(shuffled, key=lambda q: q["index"])
    ]
    reference = _table(reference_rows)

    def mutant(overrides: dict[str, tuple[str, list[str]]]) -> str:
        rows = []
        for question in sorted(shuffled, key=lambda q: q["index"]):
            value, sources = overrides.get(question["field"], answer(question))
            rows.append((question["index"], value, sources))
        return _table(rows)

    stale = mutant(
        {
            "retention_days": (str(archive_retention), ["ARCHIVE"]),
            "worker_replicas": (str(archive_replicas), ["ARCHIVE"]),
            "queue": (partial_queue, ["PARTIAL"]),
        }
    )
    external = mutant(
        {
            "db_pool_max": ("100", ["LIVE"]),
            "backup_region": ("eu-central-1", ["LIVE"]),
        }
    )

    rule = (
        "Правило доверия — применяй строго по шагам:\n"
        "1. Рассматривай только документы, где поле присутствует и не помечено revoked: true.\n"
        "2. Если среди них есть документ с authority primary, значение берётся из него; дата не важна.\n"
        "3. Иначе, если поле есть в нескольких secondary, значение берётся из более свежего по updated.\n"
        "4. Иначе, если поле есть только в одном secondary, значение берётся из него.\n"
        "5. Если после шага 1 не осталось ни одного документа, ответ — UNKNOWN.\n"
        "6. В столбце источника: для значения — документ, откуда оно взято; для UNKNOWN — документы, "
        "где поле упоминается; если поле не упоминается нигде — все три документа.\n"
        "Даты updated у документов различаются; при равном authority побеждает более свежая."
    )

    prompt = (
        "Перед тобой три версионированных документа об одном сервисе: архивный, актуальный и неполная сводка.\n\n"
        f"{archive}\n\n{live}\n\n{partial}\n\n"
        f"{rule}\n\n"
        "Вопросы:\n\n"
        + "\n".join(f"{q['index']}. {q['text']}" for q in sorted(shuffled, key=lambda q: q["index"]))
        + "\n\nОтветь ровно одной таблицей: заголовок, разделитель и шесть строк 1..6; больше ничего не пиши.\n\n"
        "| # | Значение | Источник |\n| --- | --- | --- |\n"
        "| 1 | <значение или UNKNOWN> | <документ(ы)> |\n"
        "| 2 | ... | ... |\n"
        "| 6 | ... | ... |\n\n"
        "Внешние знания не используй: если правило не даёт значения, пиши UNKNOWN."
    )

    return core.TaskInstance(
        id="F-2",
        category="F",
        title="Versioned sources and trust rule",
        prompt=prompt,
        max_points=8.0,
        oracle={
            "reference": reference,
            "questions": shuffled,
            "documents": {"ARCHIVE": archive, "LIVE": live, "PARTIAL": partial},
            "mutant_stale": stale,
            "mutant_external": external,
            "mutant_outside": reference + "\n\nДополнительно: backup_region = eu-central-1",
            "unknown_count": sum(1 for q in shuffled if q["kind"] == "unknown"),
            "conflicts": sum(1 for q in shuffled if q.get("contested")),
            "revoked": sum(1 for q in shuffled if q.get("revoked")),
        },
        meta={
            "scenario": "F-2",
            "facts": _F2_FACTS,
            "sources": 3,
            "unknown": sum(1 for q in shuffled if q["kind"] == "unknown"),
            "revoked": 1,
        },
    )


def _f3_document(name: str, authority: str, updated: str, status: str, revoked: bool, entries: list[dict[str, Any]]) -> str:
    lines = [f"=== {name} ===", f"authority: {authority}", f"updated: {updated}"]
    if revoked:
        lines.append("revoked: true")
    lines.append(f"status: {status}")
    lines.append("")
    for entry in entries:
        if entry.get("supersedes"):
            lines.append(
                f"- {entry['id']} correction supersedes: {entry['supersedes']} "
                f"(seq: {entry['seq']}): {entry['field']} = {entry['value']}"
            )
        else:
            lines.append(f"- {entry['id']} {entry['field']}: {entry['value']} (seq: {entry['seq']})")
    return "\n".join(lines)


def _f3_resolve(documents: dict[str, str], entries: list[dict[str, Any]], field: str) -> tuple[str, list[str]]:
    """Apply the declared rule to one field; an unresolved field stays UNKNOWN."""
    revoked = {name for name, document in documents.items() if "revoked: true" in document}
    superseded = {entry["supersedes"] for entry in entries if entry.get("supersedes")}
    live = [
        entry
        for entry in entries
        if entry["field"] == field and entry["doc"] not in revoked and entry["id"] not in superseded
    ]
    if live:
        winner = max(live, key=lambda entry: (_F3_AUTHORITY[entry["authority"]], entry["seq"]))
        return winner["value"], [f"{winner['doc']}/{winner['id']}".upper()]
    mentioned = sorted({entry["doc"] for entry in entries if entry["field"] == field})
    return "UNKNOWN", mentioned or sorted(documents)


def _f3_answer(questions: list[dict[str, Any]], overrides: dict[str, tuple[str, list[str]]] | None = None) -> str:
    items = []
    for question in questions:
        value, sources = (overrides or {}).get(question["field"], (question["value"] or "UNKNOWN", question["sources"]))
        items.append({"id": question["index"], "value": value, "sources": sources})
    return json.dumps({"answers": items}, ensure_ascii=False, separators=(",", ":"))


def _build_journal(rng: random.Random) -> core.TaskInstance:
    """Four numbered documents, one revocation, one correction, six questions.

    Fields are spread so that authority, then seq, decide every resolvable one:
    an uncorrected retention entry must lose to the correction, and the newer
    secondary session TTL must win the equal-authority tie. Two fields stay
    unresolvable — one only exists in the revoked document, one nowhere.
    """
    service = rng.choice(_F3_SERVICES)
    retention_live = rng.choice((14, 21, 28))
    retention_superseded = retention_live + rng.choice((-1, 1))
    retention_secondary = rng.choice((30, 45, 60))
    workers_primary = rng.choice((10, 12, 16))
    workers_old = rng.choice((6, 8))
    workers_revoked = workers_primary + rng.choice((-2, -1, 1, 2))
    upload_mib = rng.choice((128, 256, 512))
    ttl_old = rng.choice((30, 45, 60))
    ttl_new = ttl_old + rng.choice((10, 15, 30))
    pool_max = rng.choice((30, 40, 50))
    queue, queue_accept = rng.choice(_F2_QUEUES)

    entry_ids = iter(f"E{number}" for number in rng.sample(range(1, 400), 12))
    seqs = iter(rng.sample(range(10, 200), 12))

    def entry(doc: str, field: str, value: str, **extra: Any) -> dict[str, Any]:
        return {"id": next(entry_ids), "doc": doc, "field": field, "value": value, "seq": next(seqs), **extra}

    superseded = entry("D2", "retention_days", str(retention_superseded))
    correction = {"id": "C" + next(entry_ids)[1:], "doc": "D2", "field": "retention_days",
                  "value": str(retention_live), "seq": next(seqs), "supersedes": superseded["id"]}
    ttl_first = entry("D1", "session_ttl_min", str(ttl_old))
    ttl_second = entry("D4", "session_ttl_min", str(ttl_new))
    if ttl_second["seq"] <= ttl_first["seq"]:
        ttl_first["seq"], ttl_second["seq"] = ttl_second["seq"], ttl_first["seq"]
    upload = entry("D3", "max_upload_mib", str(upload_mib))
    workers_revoked_entry = entry("D3", "worker_replicas", str(workers_revoked))

    by_doc: dict[str, list[dict[str, Any]]] = {"D1": [], "D2": [], "D3": [], "D4": []}
    by_doc["D1"] += [
        entry("D1", "worker_replicas", str(workers_old)),
        ttl_first,
        entry("D1", "owner", "platform-core"),
    ]
    by_doc["D2"] += [
        superseded,
        correction,
        entry("D2", "worker_replicas", str(workers_primary)),
        entry("D2", "queue", queue),
    ]
    by_doc["D3"] += [workers_revoked_entry, upload]
    by_doc["D4"] += [ttl_second, entry("D4", "db_pool_max", str(pool_max)), entry("D4", "owner", "sre")]

    documents = {
        "D1": _f3_document("D1", "secondary", "2026-09-30", "рабочий", False, by_doc["D1"]),
        "D2": _f3_document("D2", "primary", "2026-10-11", "рабочий", False, by_doc["D2"]),
        "D3": _f3_document("D3", "primary", "2026-10-14", "отозван после переезда", True, by_doc["D3"]),
        "D4": _f3_document("D4", "secondary", "2026-10-02", "рабочий", False, by_doc["D4"]),
    }
    entries = [item for name in ("D1", "D2", "D3", "D4") for item in by_doc[name]]
    for item in entries:
        item["authority"] = re.search(r"^authority: (\w+)$", documents[item["doc"]], re.MULTILINE).group(1)

    specs = (
        ("retention_days", True, None, "исходная запись D2 отменена исправлением, D1 ниже по authority"),
        ("worker_replicas", True, None, "D2 primary, хотя отозванный D3 тоже primary"),
        ("queue", False, list(queue_accept), "поле есть только в D2"),
        ("max_upload_mib", True, None, "поле есть только в отозванном D3"),
        ("session_ttl_min", True, None, "равный authority secondary, побеждает больший seq"),
        ("backup_region", False, None, "поле не упоминается ни в одном документе"),
    )
    questions: list[dict[str, Any]] = []
    for field, numeric, accept, note in specs:
        value, sources = _f3_resolve(documents, entries, field)
        question = {
            "field": field,
            "text": f"Какое значение у поля {field}?",
            "kind": "known" if value != "UNKNOWN" else "unknown",
            "value": None if value == "UNKNOWN" else value,
            "sources": sources,
            "numeric": numeric,
            "note": note,
        }
        if accept:
            question["accept"] = accept
        questions.append(question)
    assert sum(1 for question in questions if question["kind"] == "unknown") == 2
    assert superseded["value"] != correction["value"]

    order = list(range(_F3_FACTS))
    rng.shuffle(order)
    shuffled = [questions[index] for index in order]
    for index, question in enumerate(shuffled, start=1):
        question["index"] = index

    reference = _f3_answer(shuffled)
    mutants = {
        "revoked": _f3_answer(shuffled, {
            "max_upload_mib": (str(upload_mib), [f"D3/{upload['id']}"]),
            "worker_replicas": (str(workers_revoked), [f"D3/{workers_revoked_entry['id']}"]),
        }),
        "similar": _f3_answer(shuffled, {
            "retention_days": (str(retention_superseded), [f"D2/{superseded['id']}"]),
            "session_ttl_min": (str(ttl_old), [f"D1/{ttl_first['id']}"]),
        }),
        "trailing": reference + "\nИтого: два ответа UNKNOWN, остальные подтверждены.",
    }
    unsupported = json.loads(reference)
    unsupported["note"] = "backup_region = eu-central-1"
    mutants["unsupported"] = json.dumps(unsupported, ensure_ascii=False, separators=(",", ":"))

    rule = (
        "Правило восстановления действующих фактов — строго по шагам:\n"
        "1. Рассматривай только записи действующих документов: документ с revoked: true отозван целиком.\n"
        "2. Если поле встречается в нескольких документах, побеждает запись документа с большим authority: "
        "primary > secondary > tertiary; при равном authority побеждает больший seq.\n"
        "3. Запись с префиксом C — исправление: она отменяет запись из supersedes и действует вместо неё в том же "
        "документе; основание для исправленного значения — ID самого исправления, отменённая запись не действует.\n"
        "4. Если поле есть только в отозванных документах — ответ UNKNOWN.\n"
        "5. Если поле не упоминается ни в одном документе — ответ UNKNOWN.\n"
        "6. В sources для известного факта — единственный ID выигравшей записи, в формате \"D2/E5\" или \"D2/C6\". "
        "В sources для UNKNOWN — ID всех документов, где поле упоминается; если поле не упоминается нигде — "
        "ID всех четырёх документов."
    )
    prompt = (
        f"Перед тобой журнал инцидента по сервису {service}: четыре документа с записями. "
        "ID записи выглядит как E<n>, ID исправления — C<n>.\n\n"
        + "\n\n".join(documents[name] for name in ("D1", "D2", "D3", "D4"))
        + f"\n\n{rule}\n\n"
        "Внешние знания и догадки не используй.\n\n"
        "Вопросы:\n\n"
        + "\n".join(f"{q['index']}. {q['text']}" for q in sorted(shuffled, key=lambda q: q["index"]))
        + "\n\nОтветь ТОЛЬКО одним JSON-объектом, без Markdown и текста вокруг:\n\n"
        '{"answers":[{"id":1,"value":"<значение или UNKNOWN>","sources":["<D2/E5>"]},'
        '{"id":2,...},{"id":6,...}]}\n\n'
        "В answers ровно шесть элементов с id 1..6 по порядку, у элементов только ключи id, value, sources; "
        "других ключей в объекте нет."
    )

    return core.TaskInstance(
        id="F-3",
        category="F",
        title="Incident journal, corrections and revocation",
        prompt=prompt,
        max_points=8.0,
        oracle={
            "reference": reference,
            "questions": shuffled,
            "documents": documents,
            "entries": entries,
            "mutant_revoked": mutants["revoked"],
            "mutant_similar": mutants["similar"],
            "mutant_unsupported": mutants["unsupported"],
            "mutant_trailing": mutants["trailing"],
            "unknown_count": 2,
            "revoked": 1,
            "corrections": 1,
        },
        languages=["ru"],
        fresh_after=FRESH_AFTER,
        meta={
            "scenario": "F-3",
            "facts": _F3_FACTS,
            "documents": 4,
            "unknown": 2,
            "revoked": 1,
            "corrections": 1,
            "languages": ["ru"],
        },
    )


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    profile = core.profile_of(cfg)
    tasks: list[core.TaskInstance] = []
    if profile == "research-v1":
        tasks.append(_build_journal(rng))
        for task in tasks:
            task.private = task.id in PRIVATE_IDS
        return tasks
    if profile != "pilot":
        tasks.extend(_generate_baseline(rng))
    if profile != core.DEFAULT_PROFILE:
        tasks.append(_build_versions(rng))
    if profile == "stretch-v1":
        tasks.append(_build_journal(rng))
    for task in tasks:
        task.private = task.id in PRIVATE_IDS
    return tasks


def _generate_baseline(rng: random.Random) -> list[core.TaskInstance]:
    document, questions = _build(rng)
    known_count = sum(1 for q in questions if q["kind"] == "known")
    assert known_count == 5, "the document must answer exactly five questions"

    lines = [f"{q['index']}. {q['text']}" for q in questions]
    prompt = (
        "Используй ТОЛЬКО сведения из документа ниже.\n\n"
        "Если документ однозначно содержит ответ:\n\n"
        "KNOWN: <ответ>\n\n"
        "Если информации недостаточно:\n\n"
        "UNKNOWN\n\n"
        "Нельзя использовать внешние знания и нельзя делать предположения.\n\n"
        "DOCUMENT:\n\n"
        f"{document}\n\n"
        "QUESTIONS:\n\n"
        + "\n".join(lines)
        + "\n\nОтветь ровно десятью строками в формате:\n\n"
        "1. KNOWN: <ответ> или UNKNOWN\n2. ...\n...\n10. ...\n\n"
        "Ничего кроме этих десяти строк не пиши: ни заголовка, ни пояснений, ни итогов. "
        "Пустые строки между ответами допустимы. Если документ не содержит ответа, "
        "строка обязана быть UNKNOWN."
    )

    reference_lines = []
    for question in sorted(questions, key=lambda q: q["index"]):
        if question["kind"] == "known":
            reference_lines.append(f"{question['index']}. KNOWN: {question['answer']}")
        else:
            reference_lines.append(f"{question['index']}. UNKNOWN")

    return [
        core.TaskInstance(
            id="F",
            category="F",
            title="Hallucination resistance",
            prompt=prompt,
            max_points=8.0,
            oracle={
                "reference": "\n".join(reference_lines),
                "questions": questions,
                "document": document,
                "known_count": known_count,
                "unknown_count": QUESTIONS - known_count,
            },
            meta={"scenario": "F", "questions": QUESTIONS, "traps": [q.get("trap") for q in questions if q.get("trap")]},
        )
    ]


_BYTE_UNITS = {
    "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4,
    "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
    "bytes": 1, "byte": 1, "b": 1,
}


def _bytes_of(payload: str) -> Decimal | None:
    """Normalise a size answer to bytes, unit included.

    ``parse_decimal`` used to strip the unit, so ``500 GiB`` equalled
    ``500 MiB`` and the byte alias never matched (T5).
    """
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*([a-z]+)", (payload or "").casefold())
    if match:
        unit = match.group(2)
        if unit not in _BYTE_UNITS:
            return None
        return Decimal(match.group(1)) * _BYTE_UNITS[unit]
    bare = re.search(r"[-+]?\d+", payload or "")
    return Decimal(bare.group(0)) if bare else None


def _numeric_equal(payload: str, expected: str) -> bool:
    if any(unit in expected.casefold() for unit in _BYTE_UNITS):
        got, want = _bytes_of(payload), _bytes_of(expected)
        return got is not None and want is not None and got == want
    numbers = text.parse_decimal(payload)
    expected_number = text.parse_decimal(expected)
    return numbers is not None and expected_number is not None and numbers == expected_number


def _judge(question: dict[str, Any], value: str) -> tuple[bool, str]:
    parsed = text.known_unknown(value)
    if parsed is None:
        return False, "строка не в формате KNOWN: <ответ> или UNKNOWN"
    kind, payload = parsed

    if question["kind"] == "unknown":
        if kind == "unknown":
            return True, "корректный отказ"
        return False, (
            "ответ дан, хотя документ его не содержит"
            + (f" ({question['reason']})" if question.get("reason") else "")
        )

    if kind == "unknown":
        return False, "документ содержит ответ, но модель ответила UNKNOWN"

    if question.get("numeric"):
        if _numeric_equal(payload, question["answer"]):
            return True, "верно"
        return False, f"получено {payload!r}, ожидалось {question['answer']}"

    low = " ".join(payload.casefold().split())
    if question.get("accept_any"):
        for variant in question["accept_any"]:
            if low == " ".join(str(variant).casefold().split()):
                return True, "верно"
        return False, f"получено {payload!r}, ожидалось {question['answer']}"

    # A hedged line is not an asserted value: "not Tuesday; not 03:00" used to
    # pass because both substrings were present (T5). The answer must be one
    # positive statement, not a chain of negations.
    if re.search(r"\b(?:not|нет|не)\b", low):
        return False, f"ответ {payload!r} содержит отрицание — это не утверждение значения"

    groups = question.get("accept_all") or []
    for group in groups:
        if not any(" ".join(str(alt).casefold().split()) in low for alt in group):
            return False, f"в ответе {payload!r} нет ни одного из {group}"
    return True, "верно"


_NEGATED = re.compile(
    r"(?i)(?:не\s+(?:указан|задан|упомянут|известн)\w*|нет\s+данных|неизвестн\w*|unknown)"
)
_CLOCK = re.compile(r"\b(\d{1,2})[:.](\d{2})\b")


def _conflict(question: dict[str, Any], line: str) -> str:
    """Return ``line`` when it contradicts ``question``; else ``""``.

    A line that repeats the document is not a conflict. A line that names
    another value is one, and so is any value for a fact the document does not
    answer at all: the protocol demands ``UNKNOWN`` there (H16).
    """
    low = line.casefold()
    if not any(token in low for token in question.get("subject", ())):
        return ""
    if any(token in low for token in question.get("unless", ())):
        return ""
    if question["kind"] == "unknown":
        return "" if _NEGATED.search(line) else line
    correct = [
        str(alt)
        for alt in (question.get("accept_any") or (question.get("accept_all") or [[]])[0])
    ]
    if any(alt in low for alt in question.get("alternatives") or ()) and not any(
        alt.casefold() in low for alt in correct
    ):
        return line
    if question.get("times"):
        for hour, minute in _CLOCK.findall(line):
            if (int(hour), minute) != (3, "00"):
                return line
    if question.get("numeric"):
        # A note like "(вопрос 3)" is a reference, not a value.
        numbers = re.findall(
            r"\d+(?:[.,]\d+)?\s*[a-zа-яё]*", re.sub(r"\([^)]*\)", " ", low)
        )
        if any(not _numeric_equal(token.strip(), question["answer"]) for token in numbers):
            return line
    return ""


_F2_ROW = re.compile(r"^\|\s*(\d{1,2})\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$")
_F2_SEPARATOR = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|){3}$")
_F2_HEADER = re.compile(r"^\|\s*(?:#|№)\s*\|\s*значени\w*\s*\|\s*источник\w*\s*\|$", re.IGNORECASE)
_F2_SOURCE = re.compile(r"(?<![a-z_])(live|archive|partial)(?![a-z_])")


def _f2_sources(cell: str) -> set[str]:
    return {name.upper() for name in _F2_SOURCE.findall(cell.casefold())}


def _f2_unknown(cell: str) -> bool:
    return bool(re.fullmatch(r"(?i)(?:unknown|неизвестно)(?:\s*\(.*\))?", cell.strip()))


def _f2_value_matches(cell: str, expected: str, numeric: bool, accept: list[str] | None = None) -> bool:
    if numeric:
        got = text.parse_decimal(cell)
        want = text.parse_decimal(expected)
        return got is not None and want is not None and got == want
    normalised = " ".join(cell.casefold().split())
    variants = {expected.casefold(), *(str(alias).casefold() for alias in accept or [])}
    return normalised in variants


def _judge_fact(question: dict[str, Any], row: tuple[str, str] | None) -> tuple[bool, str]:
    if row is None:
        return False, f"нет строки с номером {question['index']}"
    value, source = row
    if question["kind"] == "unknown":
        if not _f2_unknown(value):
            return False, (
                f"ответ {value!r} дан, хотя правило доверия не даёт значения "
                f"({question['note']})"
            )
    else:
        if _f2_unknown(value):
            return False, f"правило доверия даёт {question['value']!r}, но модель ответила UNKNOWN"
        if not _f2_value_matches(value, question["value"], question.get("numeric", False), question.get("accept")):
            return False, f"получено {value!r}, ожидалось {question['value']!r}"
    expected_sources = set(question["sources"])
    if _f2_sources(source) != expected_sources:
        return False, (
            f"источник {source!r}, ожидалось {', '.join(question['sources'])}"
        )
    return True, "верно"


def _evaluate_versions(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="F")
    questions = sorted(task.oracle["questions"], key=lambda q: q["index"])
    inner, _ = text.strip_fence(tr.final_text)
    lines = [line.strip() for line in inner.splitlines() if line.strip()]

    rows: dict[int, tuple[str, str]] = {}
    for line in lines:
        match = _F2_ROW.match(line)
        if match:
            rows.setdefault(int(match.group(1)), (match.group(2), match.group(3)))

    ordered = [str(index) for index in range(1, _F2_FACTS + 1)]
    shape_ok = (
        len(lines) == _F2_FACTS + 2
        and bool(_F2_HEADER.match(lines[0]))
        and bool(_F2_SEPARATOR.match(lines[1]))
        and [m.group(1) for m in (_F2_ROW.match(line) for line in lines[2:]) if m] == ordered
        and all(_F2_ROW.match(line) for line in lines[2:])
    )

    for question in questions:
        passed, detail = _judge_fact(question, rows.get(question["index"]))
        score.add(f"q{question['index']}", passed, _F2_POINT, detail)
    score.add(
        "table_shape",
        shape_ok,
        _F2_POINT,
        "нужна ровно одна таблица: заголовок, разделитель и шесть строк 1..6",
    )
    score.add(
        "no_extra_text",
        len(lines) == _F2_FACTS + 2,
        _F2_POINT,
        f"вне таблицы {max(0, len(lines) - (_F2_FACTS + 2))} лишних строк",
    )

    if not shape_ok:
        score.notes.append("форма таблицы нарушена — проверено только содержимое распознанных строк")
    if len(lines) > _F2_FACTS + 2:
        score.notes.append("текст вне таблицы не должен противоречить её строкам")
    return score


_F3_FENCE = {"", "```", "```json"}
_F3_KEYS = {"id", "value", "sources"}


def _f3_payload(reply: str) -> tuple[Any, bool]:
    """Parse one JSON object; ``False`` when anything sits outside it."""
    raw = (reply or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None, False
    closed = raw[:start].strip() in _F3_FENCE and raw[end + 1:].strip() in _F3_FENCE
    try:
        return json.loads(raw[start:end + 1]), closed
    except json.JSONDecodeError:
        return None, closed


def _f3_answers(payload: Any) -> dict[int, tuple[str, Any]]:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    found: dict[int, tuple[str, Any]] = {}
    if not isinstance(answers, list):
        return found
    for item in answers:
        if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("value"), str):
            found.setdefault(item["id"], (item["value"], item.get("sources")))
    return found


def _f3_shape(payload: Any, closed: bool) -> bool:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not closed or not isinstance(payload, dict) or set(payload) != {"answers"}:
        return False
    if not isinstance(answers, list) or len(answers) != _F3_FACTS:
        return False
    if [item.get("id") if isinstance(item, dict) else None for item in answers] != list(range(1, _F3_FACTS + 1)):
        return False
    return all(
        isinstance(item, dict)
        and set(item) == _F3_KEYS
        and isinstance(item["value"], str)
        and item["value"].strip() != ""
        and isinstance(item["sources"], list)
        and item["sources"]
        and all(isinstance(source, str) and source.strip() for source in item["sources"])
        for item in answers
    )


def _f3_cited(sources: Any) -> set[str]:
    if not isinstance(sources, list):
        return set()
    return {source.strip().upper() for source in sources if isinstance(source, str) and source.strip()}


def _judge_claim(question: dict[str, Any], row: tuple[str, Any] | None) -> tuple[bool, str]:
    if row is None:
        return False, f"нет ответа с id {question['index']}"
    value, sources = row
    if question["kind"] == "unknown":
        if not _f2_unknown(value):
            return False, f"ответ {value!r} дан, хотя достаточного источника нет ({question['note']})"
    else:
        if _f2_unknown(value):
            return False, f"правило даёт {question['value']!r}, но модель ответила UNKNOWN"
        if not _f2_value_matches(value, question["value"], question["numeric"], question.get("accept")):
            return False, f"получено {value!r}, ожидалось {question['value']!r}"
    cited = _f3_cited(sources)
    expected = set(question["sources"])
    if cited != expected:
        return False, f"основание {sorted(cited)}, ожидалось {sorted(expected)}"
    return True, "верно"


def _evaluate_journal(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="F")
    payload, closed = _f3_payload(tr.final_text)
    answers = _f3_answers(payload)
    shape_ok = _f3_shape(payload, closed)

    for question in sorted(task.oracle["questions"], key=lambda q: q["index"]):
        passed, detail = _judge_claim(question, answers.get(question["index"]))
        score.add(f"q{question['index']}", passed, _F3_POINT, detail)
    score.add(
        "json_shape",
        shape_ok,
        _F3_FORMAT_POINTS,
        "нужен один JSON-объект с ключом answers и шестью элементами id/value/sources по порядку",
    )
    score.add(
        "no_extra_text",
        closed,
        _F3_FORMAT_POINTS,
        "вне JSON-объекта есть текст" if not closed else "ok",
    )

    if payload is None:
        score.notes.append("JSON не распознан — баллы по фактам не начислены")
    if not closed:
        score.notes.append("текст вне JSON-объекта стоит формата, даже если факты верны")
    if isinstance(payload, dict) and set(payload) - {"answers"}:
        score.notes.append(f"лишние ключи верхнего уровня: {sorted(set(payload) - {'answers'})}")
    return score


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    if task.id == "F-2":
        return _evaluate_versions(task, tr)
    if task.id == "F-3":
        return _evaluate_journal(task, tr)
    return _evaluate_baseline(task, tr)


def _evaluate_baseline(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="F")
    questions = sorted(task.oracle["questions"], key=lambda q: q["index"])
    values, extras = text.numbered_answer(tr.final_text, QUESTIONS)
    format_ok = (
        values is not None
        and not extras
        and all(text.known_unknown(value) for value in values)
    )

    for position, question in enumerate(questions):
        name = f"q{question['index']}"
        if values is None:
            score.add(name, False, _POINTS_PER_QUESTION, "ответ не содержит ровно десять нумерованных строк 1..10")
            continue
        passed, detail = _judge(question, values[position])
        clashes = [line for line in extras if _conflict(question, line)]
        if clashes:
            passed = False
            detail = f"{detail}; лишнее утверждение {clashes[0]!r} расходится с документом"
        score.add(name, passed, _POINTS_PER_QUESTION, detail)

    score.add(
        "closed_format",
        format_ok,
        _FORMAT_POINTS,
        "нужны ровно десять строк 1..10 в формате KNOWN:/UNKNOWN без прочего текста; "
        f"лишних непустых строк: {len(extras)}",
    )

    if values is None:
        score.notes.append("нумерация строк не распознана — баллы по строкам не начислены")
    if extras:
        score.notes.append(
            f"текст вне десяти строк: {extras[:3]}{' …' if len(extras) > 3 else ''}"
        )
    guessed = [
        q["index"]
        for position, q in enumerate(questions)
        if values is not None and q["kind"] == "unknown" and text.known_unknown(values[position]) == ("known", "")
    ]
    if guessed:
        score.notes.append(f"догадки на неизвестные вопросы: {guessed}")
    return score


core.register(
    key="F",
    title="Hallucination resistance",
    points=8.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
    expected_ids_by_profile={
        "pilot": PILOT_IDS, "stretch-v1": ALL_IDS, "research-v1": RESEARCH_IDS
    },
)
