"""Category A — instruction following, 5 scenarios × 3 checks × 1 point.

Each scenario scores three separable things: the requested output format, the
exact content, and the order or derived value the instruction demanded. A
model that gets the content right but ignores the format loses one point, not
the scenario — that separation is the point of the category.
"""

from __future__ import annotations

import json
import random
import unicodedata
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from .. import core, text

IDS = ("IF-1", "IF-2", "IF-3", "IF-4", "IF-5")
PILOT_IDS = ("IF-6",)
"""Scenarios that only the ``pilot`` and ``stretch-v1`` profiles measure."""
IF7_LANGS = ("ru", "kk", "en")
"""The ledger twins: one semantic case, three separately written locales."""
IF7_IDS = ("IF-7", "IF-7-KK", "IF-7-EN")
"""Scenario IDs of ``IF7_LANGS``, in the same order."""
STRETCH_IDS = IDS + PILOT_IDS + IF7_IDS + ("IF-8", "IF-9", "IF-10")
"""Stretch adds the ru/en/kk dated-ledger twins, the Unicode edge case and the
composite selection pair. The three ledger IDs sit together: they are one
semantic case in three locales, not three unrelated scenarios."""

RESEARCH_IDS = ("IF-9", "IF-10")
"""The research profile keeps only the composite selection pair."""

PRIVATE_IDS = frozenset({"IF-7", "IF-7-KK", "IF-7-EN", "IF-8", "IF-9", "IF-10"})
"""Private layer of the category: the six stretch-only scenarios."""
FRESH_AFTER = "2026-10-01"
"""The stretch scenarios post-date the model snapshot of the release."""

SERVICES = ("api-01", "api-02", "api-03", "web-01", "web-02", "db-01", "cache-07")
LEVELS = ("INFO", "WARN", "ERROR")
NAMES = ("nora", "ivan", "mei", "alex", "zoe", "oleg", "kira", "pavel", "lena", "timur")
FILES = ("qwen.gguf", "gemma.gguf", "tiny.gguf", "alpha.gguf", "delta.gguf", "omega.gguf", "readme.txt", "model.bin")


def _format_point(score: core.Score, name: str, passed: bool, detail: str = "") -> None:
    """Half a point: the requested shape of the answer, apart from its content."""
    score.add(name, passed, 0.5, detail)


def _content_point(score: core.Score, name: str, passed: bool, detail: str = "") -> None:
    """Three quarters: one separable content fact the scenario asked for."""
    score.add(name, passed, 0.75, detail)


# --------------------------------------------------------------------------- #
# IF-1 — log filtering, JSON with an integer mean
# --------------------------------------------------------------------------- #


def _if1(rng: random.Random) -> core.TaskInstance:
    target_service = rng.choice(SERVICES)
    threshold = rng.choice((500, 400, 600))
    selected_count = rng.randrange(3, 6)
    mean = rng.randrange(threshold + 100, 950)

    # Latencies of the selected records sum to selected_count * mean by
    # construction, so "целое среднее" is exact. Every deviation is bounded by
    # the headroom above the threshold, so no seed can push a selected value
    # out of the filter (free ±120 deltas aborted the build on ~5% of seeds
    # when the compensating value had to absorb a large positive sum).
    room = mean - threshold - 10
    assert room >= 80, "mean must keep headroom above the threshold"
    max_neg = (room // 10) * 10
    max_pos = max(10, max_neg // (selected_count - 1))
    pool = [d for d in range(-max_neg, max_pos + 1, 10) if d != 0]
    deltas = [rng.choice(pool) for _ in range(selected_count - 1)]
    deltas.append(-sum(deltas))              # deviations sum to exactly zero
    values = [mean + d for d in deltas]
    assert min(values) > threshold, "selected latency must satisfy the filter"
    rng.shuffle(values)

    rows: list[tuple[str, str, str, int, int]] = []
    minute = rng.randrange(10, 40)
    request_id = rng.randrange(100, 999)
    for latency in values:
        minute += rng.randrange(1, 4)
        rows.append((f"{minute:02d}:{rng.randrange(0, 60):02d}", target_service, "ERROR", request_id, latency))
        request_id += rng.randrange(1, 90)

    distractors = [
        (target_service, "ERROR", threshold),          # boundary: not > threshold
        (target_service, "ERROR", rng.randrange(20, 120)),
        (target_service, "WARN", mean + rng.randrange(1, 300)),
        (target_service, "INFO", mean + rng.randrange(1, 200)),
        (rng.choice([s for s in SERVICES if s != target_service]), "ERROR", mean + rng.randrange(50, 400)),
    ]
    for service, level, latency in distractors:
        minute += rng.randrange(1, 3)
        rows.append((f"{minute:02d}:{rng.randrange(0, 60):02d}", service, level, request_id, latency))
        request_id += rng.randrange(1, 90)
    rng.shuffle(rows)

    lines = [
        f"[{ts}] {service} {level:<5} request_id={rid} latency={latency}"
        for ts, service, level, rid, latency in rows
    ]
    expected_ids = sorted(
        rid for _, service, level, rid, latency in rows
        if service == target_service and level == "ERROR" and latency > threshold
    )
    prompt = (
        "Даны записи:\n\n"
        + "\n".join(lines)
        + "\n\nВерни ТОЛЬКО валидный JSON:\n\n"
        '{"ids":[],"mean_latency":0}\n\n'
        "В ids включи request_id только если одновременно:\n"
        f"1. service = {target_service}\n"
        "2. level = ERROR\n"
        f"3. latency > {threshold}\n\n"
        "ids отсортируй по возрастанию.\n"
        "mean_latency — целое среднее арифметическое latency выбранных записей.\n\n"
        "Никакого Markdown и дополнительного текста."
    )
    reference = f'{{"ids":[{",".join(str(i) for i in expected_ids)}],"mean_latency":{mean}}}'
    return core.TaskInstance(
        id="IF-1",
        category="A",
        title="Фильтрация логов",
        prompt=prompt,
        max_points=2.0,
        oracle={
            "reference": reference,
            "ids": expected_ids,
            "mean_latency": mean,
            "target_service": target_service,
            "threshold": threshold,
        },
        meta={"scenario": "IF-1", "rows": len(rows), "selected": selected_count, "integral_mean": mean},
    )


def _eval_if1(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    value, strict = text.strict_json(tr.final_text)
    payload = value if isinstance(value, dict) else text.loose_json(tr.final_text)

    fmt_ok = (
        strict
        and isinstance(payload, dict)
        and set(payload) == {"ids", "mean_latency"}
        and isinstance(payload.get("ids"), list)
        and all(isinstance(i, int) and not isinstance(i, bool) for i in payload.get("ids", []))
        and isinstance(payload.get("mean_latency"), int)
        and not isinstance(payload.get("mean_latency"), bool)
    )
    ids_ok = isinstance(payload, dict) and isinstance(payload.get("ids"), list) and payload["ids"] == oracle["ids"]
    mean_ok = (
        isinstance(payload, dict)
        and isinstance(payload.get("mean_latency"), int)
        and payload["mean_latency"] == oracle["mean_latency"]
    )
    _format_point(score, "json_format",
        fmt_ok,
        "строгий JSON с ключами ids и mean_latency (целое)" if not fmt_ok else "ok",
    )
    _content_point(score, "ids", ids_ok, f"получено {payload.get('ids') if isinstance(payload, dict) else None}, ожидалось {oracle['ids']}")
    _content_point(score, "mean_latency",
        mean_ok,
        f"получено {payload.get('mean_latency') if isinstance(payload, dict) else None}, ожидалось {oracle['mean_latency']}",
    )
    return score


# --------------------------------------------------------------------------- #
# IF-2 — CSV filtering and sorting
# --------------------------------------------------------------------------- #


def _if2(rng: random.Random) -> core.TaskInstance:
    names = rng.sample(NAMES, rng.randrange(6, 9))
    shared_usage = rng.randrange(120, 900)
    rows: list[dict[str, Any]] = []

    selected = [
        {"name": names[0], "plan": "pro", "status": "active", "usage": shared_usage},
        {"name": names[1], "plan": "pro", "status": "active", "usage": shared_usage},
        {"name": names[2], "plan": "pro", "status": "active", "usage": rng.randrange(101, 119)},
    ]
    if rng.random() < 0.5:
        selected.append({"name": names[3], "plan": "pro", "status": "active", "usage": 100})
    rows.extend(selected)
    rows.extend(
        [
            {"name": names[-1], "plan": "pro", "status": "suspended", "usage": rng.randrange(500, 950)},
            {"name": names[-2], "plan": "basic", "status": "active", "usage": rng.randrange(700, 990)},
            {"name": names[-3], "plan": "pro", "status": "active", "usage": 99},
            {"name": names[-4], "plan": "basic", "status": "suspended", "usage": 150},
        ]
    )
    rng.shuffle(rows)

    table = "\n".join(f"{r['name']},{r['plan']},{r['status']},{r['usage']}" for r in rows)
    ordered = sorted(selected, key=lambda r: (-r["usage"], r["name"]))
    expected_rows = [(r["name"], r["usage"]) for r in ordered]
    reference = "\n".join(f"{name},{usage}" for name, usage in expected_rows)

    prompt = (
        "Даны данные:\n\n"
        "name,plan,status,usage\n"
        + table
        + "\n\nОставь только записи, где:\n"
        "- plan = pro\n"
        "- status = active\n"
        "- usage >= 100\n\n"
        "Отсортируй:\n"
        "1. usage по убыванию;\n"
        "2. при одинаковом usage — name по алфавиту.\n\n"
        "Верни ТОЛЬКО CSV без заголовка.\n"
        "В каждой строке должны быть только name и usage."
    )
    return core.TaskInstance(
        id="IF-2",
        category="A",
        title="Сортировка записей",
        prompt=prompt,
        max_points=2.0,
        oracle={"reference": reference, "rows": expected_rows, "all_names": sorted(r["name"] for r in rows)},
        meta={"scenario": "IF-2", "tie_usage": shared_usage, "selected": len(selected)},
    )


def _parse_csv_rows(raw: str) -> list[tuple[str, str]] | None:
    inner, fenced = text.strip_fence(raw)
    if fenced or not inner:
        return None
    rows: list[tuple[str, str]] = []
    for line in inner.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("name,"):
            return None
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not all(parts):
            return None
        rows.append((parts[0], parts[1]))
    return rows or None


def _eval_if2(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    rows = _parse_csv_rows(tr.final_text)

    def _usage(value: str) -> int | None:
        # `--1` passes `lstrip('-').isdigit()` but kills `int()`: a malformed
        # number must be a format failure, not an evaluator crash.
        try:
            return int(value)
        except ValueError:
            return None

    fmt_ok = rows is not None and all(
        name in oracle["all_names"] and _usage(usage) is not None for name, usage in rows
    )
    parsed = [(name, _usage(usage)) for name, usage in rows] if fmt_ok else []
    expected = oracle["rows"]
    set_ok = sorted(parsed) == sorted(expected) if fmt_ok else False
    order_ok = parsed == expected if (fmt_ok and set_ok) else False

    _format_point(score, "csv_format", fmt_ok, "ожидался CSV name,usage без заголовка и без Markdown")
    _content_point(score, "rows", set_ok, f"получено {parsed}, ожидалось {sorted(expected)}")
    _content_point(score, "order", order_ok, f"получено {parsed}, ожидалось {expected}")
    return score


# --------------------------------------------------------------------------- #
# IF-3 — firewall rules
# --------------------------------------------------------------------------- #


def _if3(rng: random.Random) -> core.TaskInstance:
    allowed_tcp = rng.sample(range(20, 9000), rng.randrange(2, 4))
    denied_only = rng.choice([p for p in (9000, 9001, 3306, 5432, 8081) if p not in allowed_tcp])
    both = allowed_tcp[0]
    udp_only = rng.choice([p for p in (53, 123, 443, 514) if p not in allowed_tcp])

    rules: list[tuple[str, int, str]] = []
    for port in allowed_tcp:
        rules.append(("tcp", port, "allow"))
    rules.append(("tcp", both, "deny"))
    rules.append(("tcp", allowed_tcp[-1], "allow"))  # duplicate allow
    rules.append(("tcp", denied_only, "deny"))
    rules.append(("udp", udp_only, "allow"))
    rules.append(("udp", both, "allow"))
    rng.shuffle(rules)

    expected = sorted(allowed_tcp)
    reference = f'{{"open":[{",".join(str(p) for p in expected)}],"count":{len(expected)}}}'
    prompt = (
        "Даны правила:\n\n"
        + "\n".join(f"{proto} {port} {action}" for proto, port, action in rules)
        + "\n\nНужно выбрать только TCP-порты, для которых существует правило allow.\n"
        "Наличие отдельного deny не отменяет allow.\n\n"
        "Удалить дубликаты.\n"
        "Отсортировать порты по возрастанию.\n\n"
        "Верни ТОЛЬКО JSON без Markdown:\n\n"
        '{"open":[],"count":0}'
    )
    return core.TaskInstance(
        id="IF-3",
        category="A",
        title="Firewall rules",
        prompt=prompt,
        max_points=2.0,
        oracle={"reference": reference, "open": expected, "count": len(expected)},
        meta={"scenario": "IF-3", "deny_only": denied_only, "udp_only": udp_only, "allow_and_deny": both},
    )


def _eval_if3(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    value, strict = text.strict_json(tr.final_text)
    payload = value if isinstance(value, dict) else text.loose_json(tr.final_text)

    fmt_ok = (
        strict
        and isinstance(payload, dict)
        and set(payload) == {"open", "count"}
        and isinstance(payload.get("open"), list)
        and all(isinstance(p, int) and not isinstance(p, bool) for p in payload.get("open", []))
        and isinstance(payload.get("count"), int)
        and not isinstance(payload.get("count"), bool)
    )
    open_ok = isinstance(payload, dict) and payload.get("open") == oracle["open"]
    count_ok = (
        isinstance(payload, dict)
        and payload.get("count") == oracle["count"]
    )
    _format_point(score, "json_format", fmt_ok, "ожидался строгий JSON с ключами open и count")
    _content_point(score, "open",
        open_ok,
        f"получено {payload.get('open') if isinstance(payload, dict) else None}, ожидалось {oracle['open']}",
    )
    _content_point(score, "count",
        count_ok,
        f"получено {payload.get('count') if isinstance(payload, dict) else None}, ожидалось {oracle['count']}",
    )
    return score


# --------------------------------------------------------------------------- #
# IF-4 — file selection
# --------------------------------------------------------------------------- #


def _if4(rng: random.Random) -> core.TaskInstance:
    threshold = rng.choice((10, 12))
    tied_size = threshold + rng.randrange(1, 6)
    kept = [
        (FILES[0], tied_size),
        (FILES[3], tied_size),
    ]
    if rng.random() < 0.5:
        kept.append((FILES[1], threshold))
    entries = list(kept) + [
        (FILES[2], threshold - rng.randrange(1, 4)),
        (FILES[6], rng.randrange(1, 5)),
        (FILES[7], threshold + rng.randrange(3, 12)),
        (FILES[5], rng.randrange(1, 9)),
    ]
    rng.shuffle(entries)

    expected = [name for name, _ in sorted(kept, key=lambda e: (-e[1], e[0]))]
    reference = "[" + ",".join(f'"{name}"' for name in expected) + "]"
    prompt = (
        "Даны файлы:\n\n"
        + "\n".join(f"{name} {size:.1f}" for name, size in entries)
        + "\n\nЧисло после имени — размер в GiB.\n\n"
        "Оставь только файлы:\n"
        "- с расширением .gguf;\n"
        f"- размером не менее {threshold} GiB.\n\n"
        "Сортировка:\n"
        "1. размер по убыванию;\n"
        "2. при одинаковом размере — имя по алфавиту.\n\n"
        "Верни ТОЛЬКО JSON-массив имён файлов."
    )
    return core.TaskInstance(
        id="IF-4",
        category="A",
        title="Выбор файлов",
        prompt=prompt,
        max_points=2.0,
        oracle={"reference": reference, "files": expected, "threshold": threshold},
        meta={"scenario": "IF-4", "tied_size": tied_size, "entries": entries},
    )


def _eval_if4(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    value, strict = text.strict_json(tr.final_text)
    payload = value if isinstance(value, list) else text.loose_json(tr.final_text)

    comparable = isinstance(payload, list) and all(isinstance(n, str) for n in payload)
    fmt_ok = strict and comparable
    # A mixed-type payload ([null, "x"]) is a wrong answer, not a TypeError in
    # the evaluator: sort only what is actually sortable.
    set_ok = comparable and sorted(payload) == sorted(oracle["files"])
    order_ok = comparable and payload == oracle["files"]
    _format_point(score, "json_format", fmt_ok, "ожидался строгий JSON-массив строк без обвязки")
    _content_point(score, "files",
        set_ok,
        f"получено {payload if isinstance(payload, list) else None}, ожидалось {sorted(oracle['files'])}",
    )
    _content_point(score, "order", order_ok, f"получено {payload if isinstance(payload, list) else None}, ожидалось {oracle['files']}")
    return score


# --------------------------------------------------------------------------- #
# IF-5 — configuration precedence
# --------------------------------------------------------------------------- #


def _if5(rng: random.Random) -> core.TaskInstance:
    default = {
        "temperature": round(rng.uniform(0.5, 0.9), 1),
        "max_tokens": rng.choice((2048, 4096, 6144)),
        "tools": True,
        "mode": "auto",
    }
    profile = {
        "max_tokens": rng.choice((8192, 12288, 16384)),
        "mode": rng.choice(("fast", "balanced", "strict")),
    }
    request = {"temperature": 0, "tools": False}
    expected = {
        "temperature": request["temperature"],
        "max_tokens": profile["max_tokens"],
        "tools": request["tools"],
        "mode": profile["mode"],
    }
    reference = (
        "{"
        f'"temperature":{expected["temperature"]},'
        f'"max_tokens":{expected["max_tokens"]},'
        f'"tools":{str(expected["tools"]).lower()},'
        f'"mode":"{expected["mode"]}"'
        "}"
    )
    prompt = (
        "Есть три уровня конфигурации.\n\n"
        "DEFAULT:\n"
        f"temperature={default['temperature']}\n"
        f"max_tokens={default['max_tokens']}\n"
        f"tools={str(default['tools']).lower()}\n"
        f"mode={default['mode']}\n\n"
        "PROFILE:\n"
        f"max_tokens={profile['max_tokens']}\n"
        f"mode={profile['mode']}\n\n"
        "REQUEST:\n"
        f"temperature={request['temperature']}\n"
        f"tools={str(request['tools']).lower()}\n\n"
        "Приоритет:\n"
        "REQUEST > PROFILE > DEFAULT.\n\n"
        "Верни итоговую конфигурацию ТОЛЬКО как JSON:\n\n"
        '{\n  "temperature": ...,\n  "max_tokens": ...,\n  "tools": ...,\n  "mode": ...\n}\n\n'
        "Дополнительный текст запрещён."
    )
    return core.TaskInstance(
        id="IF-5",
        category="A",
        title="Приоритет конфигурации",
        prompt=prompt,
        max_points=2.0,
        oracle={
            "reference": reference,
            "expected": expected,
            "request_keys": ("temperature", "tools"),
            "profile_keys": ("max_tokens", "mode"),
            "default": default,
            "profile": profile,
        },
        meta={"scenario": "IF-5"},
    )


def _eval_if5(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    expected = oracle["expected"]
    value, strict = text.strict_json(tr.final_text)
    payload = value if isinstance(value, dict) else text.loose_json(tr.final_text)

    def numeric_ok(key: str) -> bool:
        if not isinstance(payload, dict):
            return False
        got = payload.get(key)
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            return False
        return float(got) == float(expected[key])

    fmt_ok = (
        strict
        and isinstance(payload, dict)
        and set(payload) == set(expected)
        and numeric_ok("temperature")
        and isinstance(payload.get("max_tokens"), int)
        and not isinstance(payload.get("max_tokens"), bool)
        and isinstance(payload.get("tools"), bool)
        and isinstance(payload.get("mode"), str)
    )
    request_ok = numeric_ok("temperature") and isinstance(payload, dict) and payload.get("tools") is False
    profile_ok = (
        isinstance(payload, dict)
        and payload.get("max_tokens") == expected["max_tokens"]
        and payload.get("mode") == expected["mode"]
    )
    _format_point(score, "json_format", fmt_ok, "ожидался строгий JSON ровно с четырьмя ключами и верными типами")
    _content_point(score, "request_layer",
        request_ok,
        f"temperature/tools: получено {payload if isinstance(payload, dict) else None}, ожидалось {expected['temperature']} и false",
    )
    _content_point(score, "profile_layer",
        profile_ok,
        f"max_tokens/mode: получено {payload if isinstance(payload, dict) else None}, ожидалось {expected['max_tokens']} и {expected['mode']!r}",
    )
    return score


# --------------------------------------------------------------------------- #
# IF-6 — CSV log summary with exception precedence (pilot)
# --------------------------------------------------------------------------- #

_IF6_FIELDS = ("id", "msg", "service", "level", "ms", "flag")

_IF6_MESSAGES = (
    "timeout, retry scheduled",
    'disk "full" on /var',
    "upstream 502, fallback opened",
    "queue backlog high",
    "tls handshake reset",
    "pool exhausted",
    "gc pause, long",
    "route table refresh",
    'header "retry-after" missing',
    "cache warmup done",
    "replica lag, growing",
    "socket closed, peer reset",
)


def _if6_field(value: str) -> str:
    """RFC 4180 field: quote when the value carries a comma, quote or newline."""
    if any(char in value for char in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def _if6(rng: random.Random) -> core.TaskInstance:
    """A CSV summary where the exception and the exclusion both override the rule.

    The oracle is computed from the structured rows, never from the rendered
    CSV: the prompt is the serialisation, so a quoting bug cannot agree with
    itself. Four rows carry a comma or an escaped quote in ``msg``, which is
    exactly where a naive ``split(",")`` shifts the semantic columns.
    """
    target_service = rng.choice(SERVICES)
    other_service = rng.choice([service for service in SERVICES if service != target_service])
    threshold = rng.choice((250, 300, 360))

    rows: list[dict[str, Any]] = []

    def add(msg: str, service: str, level: str, ms: int, flag: str = "no") -> dict[str, Any]:
        row = {"msg": msg, "service": service, "level": level, "ms": ms, "flag": flag}
        rows.append(row)
        return row

    # Rule 1: the general inclusion rule.
    for _ in range(3):
        add(rng.choice(_IF6_MESSAGES), target_service, "ERROR", threshold + rng.randrange(40, 500))
    boundary = add("timeout, retry scheduled", target_service, "ERROR", threshold)
    # Rule 2: CRITICAL from the target service is included regardless of ms.
    critical = add('disk "full" on /var', target_service, "CRITICAL", rng.randrange(20, threshold - 10))
    # Rule 3: ignore removes a record that both rules above would have taken.
    ignored_general = add("upstream 502, fallback opened", target_service, "ERROR", threshold + rng.randrange(30, 300), "ignore")
    ignored_critical = add("socket closed, peer reset", target_service, "CRITICAL", rng.randrange(10, threshold - 20), "ignore")
    # Near misses the rule may not swallow.
    add(rng.choice(_IF6_MESSAGES), other_service, "ERROR", threshold + rng.randrange(50, 600))
    add(rng.choice(_IF6_MESSAGES), target_service, "WARN", threshold + rng.randrange(50, 600))
    add(rng.choice(_IF6_MESSAGES), target_service, "ERROR", threshold - 1)
    add(rng.choice(_IF6_MESSAGES), target_service, "ERROR", rng.randrange(10, threshold - 40))
    for _ in range(rng.randrange(10, 21)):
        add(rng.choice(_IF6_MESSAGES), rng.choice(SERVICES), rng.choice(LEVELS), rng.randrange(20, 1000))

    assert 20 <= len(rows) <= 40, "fixture stays inside the announced 20–40 rows"
    pool = rng.sample(range(100, 1000), len(rows))
    for row_id, row in zip(pool, rows):
        row["id"] = row_id
    rng.shuffle(rows)

    selected = [
        row for row in rows
        if row["flag"] != "ignore"
        and row["service"] == target_service
        and (row["level"] == "CRITICAL" or (row["level"] == "ERROR" and row["ms"] >= threshold))
    ]
    expected_ids = sorted(row["id"] for row in selected)
    # Five constructed hits always qualify; a filler row may legitimately add
    # another, so only the lower bound is fixed.
    assert len(expected_ids) >= 5, "the constructed hits must all survive the rules"

    reference = json.dumps(
        {"ids": expected_ids, "count": len(expected_ids), "total_ms": sum(row["ms"] for row in selected)},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    csv_body = "\n".join(
        ",".join([
            str(row["id"]),
            _if6_field(row["msg"]),
            row["service"],
            row["level"],
            str(row["ms"]),
            row["flag"],
        ])
        for row in rows
    )
    prompt = (
        "Даны записи журнала в CSV (RFC 4180, разделитель — запятая);\n"
        "первая строка — заголовок, flag принимает значения yes или no:\n\n"
        f"{','.join(_IF6_FIELDS)}\n"
        f"{csv_body}\n\n"
        "Правила применяются сверху вниз:\n"
        f"1. include: service = {target_service} и level = ERROR и ms >= {threshold};\n"
        f"2. исключение: service = {target_service} и level = CRITICAL включаются всегда, независимо от ms;\n"
        "3. записи с flag = ignore исключаются, даже если подходят под п.1 или п.2.\n\n"
        "ids — числовые id отобранных записей по возрастанию.\n"
        "count — сколько записей отобрано, total_ms — сумма их ms.\n\n"
        "Верни ТОЛЬКО валидный JSON:\n\n"
        '{"ids":[],"count":0,"total_ms":0}\n\n'
        "Никакого Markdown и дополнительного текста."
    )
    return core.TaskInstance(
        id="IF-6",
        category="A",
        title="Сводка журнала с исключениями",
        prompt=prompt,
        max_points=2.0,
        oracle={
            "reference": reference,
            "ids": expected_ids,
            "count": len(expected_ids),
            "total_ms": sum(row["ms"] for row in selected),
            "target_service": target_service,
            "threshold": threshold,
            "boundary_id": boundary["id"],
            "critical_id": critical["id"],
            "ignored_ids": sorted((ignored_general["id"], ignored_critical["id"])),
        },
        meta={"scenario": "IF-6", "rows": len(rows), "service": target_service, "threshold": threshold},
    )


def _eval_if6(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    value, strict = text.strict_json(tr.final_text)
    payload = value if isinstance(value, dict) else text.loose_json(tr.final_text)

    def integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    fmt_ok = (
        strict
        and isinstance(payload, dict)
        and set(payload) == {"ids", "count", "total_ms"}
        and isinstance(payload.get("ids"), list)
        and all(integer(item) for item in payload.get("ids", []))
        and integer(payload.get("count"))
        and integer(payload.get("total_ms"))
    )
    ids_ok = isinstance(payload, dict) and payload.get("ids") == oracle["ids"]
    aggregate_ok = (
        isinstance(payload, dict)
        and payload.get("count") == oracle["count"]
        and payload.get("total_ms") == oracle["total_ms"]
    )
    _format_point(score, "json_format",
        fmt_ok,
        "ожидался строгий JSON с ключами ids, count, total_ms и целыми значениями"
        if not fmt_ok else "ok",
    )
    _content_point(score, "ids",
        ids_ok,
        f"получено {payload.get('ids') if isinstance(payload, dict) else None}, ожидалось {oracle['ids']}",
    )
    _content_point(score, "aggregate",
        aggregate_ok,
        f"получено {payload.get('count') if isinstance(payload, dict) else None}/"
        f"{payload.get('total_ms') if isinstance(payload, dict) else None}, "
        f"ожидалось {oracle['count']}/{oracle['total_ms']}",
    )
    return score


# --------------------------------------------------------------------------- #
# IF-7 — dated ledger: ru/en/kk date, number and Unicode forms into ISO/SI
#
# The three twins share one semantic fixture per seed — same days, masses and
# invoice numbers — but each language renders and states the rules on its own:
# its month names, its date order, its thousands and decimal separators. The
# invoice keys carry a combining-mark word, so a twin may differ only by NFC
# versus NFD spelling of one identity. All three are the same task, never a
# translation of one prompt.
# --------------------------------------------------------------------------- #

RU_MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
KK_MONTHS = (
    "қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым",
    "шілде", "тамыз", "қыркүйек", "қазан", "қараша", "желтоқсан",
)
EN_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

_IF7_PREFIX = {"ru": "НК", "kk": "НҚ", "en": "INV"}
_IF7_WORDS = {
    "ru": ("йод", "ёлка", "объём", "звёзды", "мёд", "пёрышко"),
    "kk": ("йод", "қайың", "жайлау", "мейірім", "сәйкес", "күй"),
    "en": ("café", "naïve", "Zoë", "résumé", "señor", "jalapeño"),
}
"""Counterparty words; every entry's NFC and NFD spellings differ."""

for _pool in _IF7_WORDS.values():
    assert all(unicodedata.normalize("NFD", word) != word for word in _pool), "the key must decompose"
_IF7_CULTURES = {"ru": ("ru",), "kk": ("kz",), "en": ("en-GB",)}

_IF7_THRESHOLD_TENTHS = 12_000  # 1200.0 kg, the inclusive boundary


def _if7_date(day: date, lang: str) -> str:
    """A date as documents of the language write it, not as ISO."""
    if lang == "kk":
        return f"{day.year} жылғы {day.day} {KK_MONTHS[day.month - 1]}"
    if lang == "en":
        return f"{day.day} {EN_MONTHS[day.month - 1]} {day.year}"
    return f"{day.day} {RU_MONTHS[day.month - 1]} {day.year} г."


def _if7_mass(tenths: int, lang: str) -> str:
    """Kilograms in the language's separators: ``1 845,5`` ru/kk, ``1,845.5`` en."""
    whole, fraction = divmod(tenths, 10)
    if lang == "en":
        return f"{whole:,}.{fraction}"
    return f"{whole:,}".replace(",", " ") + f",{fraction}"


def _if7_tonnes(tenths: int) -> Decimal:
    return (Decimal(tenths) / 1000).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def _if7_invoice(row: dict[str, Any], lang: str) -> str:
    """One identity in one language, in the row's declared Unicode form."""
    word = _IF7_WORDS[lang][row["word"]]
    raw = f"{_IF7_PREFIX[lang]}-{row['number']}/{row['slot']} {word}"
    return unicodedata.normalize("NFD" if row["decomposed"] else "NFC", raw)


def _if7_rows(rng: random.Random) -> tuple[date, list[dict[str, Any]]]:
    """Twelve rows over ten identities: three survive the window and the dedupe.

    Two identities appear twice, once per Unicode form and once on an earlier
    day; only the later row of each survives. Of the ten single rows, three are
    selected, three fall outside the window (one exactly on the boundary mass)
    and four are too light.
    """
    year = rng.choice((2026, 2027))
    base = date(year, 1, 5) + timedelta(days=rng.randrange(0, 300))
    numbers = rng.sample(range(1000, 9999), 10)
    words = [rng.randrange(len(_IF7_WORDS["ru"])) for _ in range(10)]
    forms = [rng.random() < 0.5 for _ in range(10)]
    heavy, light = lambda: rng.randrange(15_000, 28_000), lambda: rng.randrange(1_000, 11_999)
    plan = [
        (0, base - timedelta(days=2), heavy()),                     # before the window
        (1, base + timedelta(days=1), _IF7_THRESHOLD_TENTHS),       # exact boundary
        (2, base + timedelta(days=3), rng.randrange(16_000, 28_000)),
        (3, base + timedelta(days=5), rng.randrange(13_000, 20_000)),  # last day, inclusive
        (4, base + timedelta(days=7), heavy()),                     # after the window
        (5, base + timedelta(days=2), light()),                     # too light
        (6, base + timedelta(days=4), light()),
        (7, base + timedelta(days=6), rng.randrange(9_000, 11_999)),
        (8, base, _IF7_THRESHOLD_TENTHS),                           # boundary mass, early day
        (9, base + timedelta(days=4), light()),                     # too light
        (2, base + timedelta(days=2), rng.randrange(16_000, 28_000)),  # loses to the later row
        (3, base + timedelta(days=4), rng.randrange(13_000, 20_000)),  # loses to the later row
    ]
    rows = [
        {
            "identity": identity,
            "number": numbers[identity],
            "slot": identity + 1,
            "word": words[identity],
            "day": day,
            "tenths": tenths,
            "decomposed": forms[identity] if position < 10 else not forms[identity],
        }
        for position, (identity, day, tenths) in enumerate(plan)
    ]
    rng.shuffle(rows)
    return base, rows


def _if7_selection(rows: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = unicodedata.normalize("NFC", _if7_invoice(row, lang))
        current = latest.get(key)
        if current is None or row["day"] >= current["day"]:
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (row["day"], unicodedata.normalize("NFC", _if7_invoice(row, lang))),
    )


def _if7_window(rows: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["day"] >= start and row["day"] <= end and row["tenths"] >= _IF7_THRESHOLD_TENTHS
    ]


def _if7_prompt(lang: str, rows: list[dict[str, Any]], start: date, end: date) -> str:
    listing = "\n".join(
        f"{_if7_invoice(row, lang)}; {_if7_date(row['day'], lang)}; {_if7_mass(row['tenths'], lang)}"
        for row in rows
    )
    threshold = _if7_mass(_IF7_THRESHOLD_TENTHS, lang)
    if lang == "kk":
        return (
            "Төменде қойма ведомосінен үзінді (бөлгіш — нүктелі үтір):\n\n"
            f"{listing}\n\n"
            "Күндер қазақша, сандарда мыңдық разряд бос орынмен, ал бөлшек бөлігі үтірмен "
            "жазылған; масса килограммен берілген. Бір нөмір екі рет кездессе, "
            "кейінгі күнгі жазба ғана қалады.\n\n"
            f"Күні {start.isoformat()} мен {end.isoformat()} аралығына (қоса) кіретін және "
            f"массасы {threshold} кг кем емес жолдарды қалдыр.\n\n"
            "Әр қалған жол үшін:\n"
            "- күнді ISO (ЖЖЖЖ-АА-КК) түріне келтір;\n"
            "- массаны тоннаға ауыстырып, үтірден кейінгі 3 таңбаға дейін дөңгелекте "
            "(жартысы — жоғары);\n"
            "- нөмірді NFC түрінде жаз.\n\n"
            "Жолдарды күні бойынша, содан кейін нөмірі бойынша өсу ретімен сұрыпта.\n\n"
            "ТЕК жарамды JSON қайтар:\n\n"
            '{"rows":[{"date":"...","invoice":"...","tonnes":0.0}],"total_tonnes":0.0}\n\n'
            "Markdown және қосымша мәтін жоқ."
        )
    if lang == "en":
        return (
            "Below is an extract from a warehouse ledger (delimiter — semicolon):\n\n"
            f"{listing}\n\n"
            "Dates are written in English, numbers use a comma between thousands and a dot "
            "before the fractional part; the mass is in kilograms. When the same number "
            "appears twice, only the later-dated record stays.\n\n"
            f"Keep the rows whose date falls inside the window from {start.isoformat()} to "
            f"{end.isoformat()} inclusive and whose mass is at least {threshold} kg.\n\n"
            "For each remaining row:\n"
            "- convert the date to ISO (YYYY-MM-DD);\n"
            "- convert the mass to tonnes and round to 3 decimal places (half up);\n"
            "- write the number in NFC.\n\n"
            "Sort the rows by date, then by number ascending.\n\n"
            "Return ONLY valid JSON:\n\n"
            '{"rows":[{"date":"...","invoice":"...","tonnes":0.0}],"total_tonnes":0.0}\n\n'
            "No Markdown and no extra text."
        )
    return (
        "Ниже выписка из складской ведомости (разделитель — точка с запятой):\n\n"
        f"{listing}\n\n"
        "Даты записаны по-русски, числа — с пробелом между тысячами и запятой перед дробной "
        "частью; масса указана в килограммах. Если номер повторяется, остаётся только запись "
        "с более поздней датой.\n\n"
        f"Оставь строки, где дата попадает в окно с {start.isoformat()} по {end.isoformat()} "
        f"включительно, а масса не меньше {threshold} кг.\n\n"
        "Для каждой оставшейся строки:\n"
        "- переведи дату в ISO (ГГГГ-ММ-ДД);\n"
        "- переведи массу в тонны и округли до 3 знаков после запятой (половина — вверх);\n"
        "- запиши номер в NFC.\n\n"
        "Отсортируй строки по дате, затем по номеру по возрастанию.\n\n"
        "Верни ТОЛЬКО валидный JSON:\n\n"
        '{"rows":[{"date":"...","invoice":"...","tonnes":0.0}],"total_tonnes":0.0}\n\n'
        "Никакого Markdown и дополнительного текста."
    )


def _if7(rng: random.Random, lang: str, seed: int | None = None) -> core.TaskInstance:
    base, rows = _if7_rows(random.Random(seed) if seed is not None else rng)
    start, end = base + timedelta(days=1), base + timedelta(days=5)
    picked = _if7_selection(_if7_window(rows, start, end), lang)
    assert len(picked) == 3, "the fixture must select exactly three rows"
    expected_rows = [
        {
            "date": row["day"].isoformat(),
            "invoice": unicodedata.normalize("NFC", _if7_invoice(row, lang)),
            "tonnes": _if7_tonnes(row["tenths"]),
        }
        for row in picked
    ]
    for row in expected_rows:
        assert row["invoice"] == unicodedata.normalize("NFC", row["invoice"]), "the oracle keeps NFC keys"
    total = sum((row["tonnes"] for row in expected_rows), Decimal("0"))
    reference = json.dumps(
        {
            "rows": [
                {"date": row["date"], "invoice": row["invoice"], "tonnes": float(row["tonnes"])}
                for row in expected_rows
            ],
            "total_tonnes": float(total),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    scenario = IF7_IDS[IF7_LANGS.index(lang)]
    records = [
        {
            "invoice": _if7_invoice(row, lang),
            "identity": row["identity"],
            "day": row["day"].isoformat(),
            "tenths": row["tenths"],
            "decomposed": row["decomposed"],
        }
        for row in rows
    ]
    records.sort(key=lambda item: (item["day"], item["invoice"]))
    return core.TaskInstance(
        id=scenario,
        category="A",
        title={"ru": "Ведомость с датами ru", "kk": "Күндері бар ведомость kk",
               "en": "Dated ledger en"}[lang],
        prompt=_if7_prompt(lang, rows, start, end),
        max_points=2.0,
        oracle={
            "reference": reference,
            "rows": expected_rows,
            "records": records,
            "total_tonnes": total,
            "threshold_tenths": _IF7_THRESHOLD_TENTHS,
            "window": (start.isoformat(), end.isoformat()),
            "decomposed_rows": sum(1 for row in rows if row["decomposed"]),
        },
        languages=[lang],
        cultures=list(_IF7_CULTURES[lang]),
        fresh_after=FRESH_AFTER,
        meta={
            "scenario": scenario,
            "rows": len(rows),
            "selected": len(expected_rows),
            "window": [start.isoformat(), end.isoformat()],
            "languages": [lang],
            "cultures": list(_IF7_CULTURES[lang]),
        },
    )


def _if7_variants(rng: random.Random) -> list[core.TaskInstance]:
    """All three twins of one semantic case, in the declared id order."""
    seed = rng.randrange(1 << 32)
    return [_if7(rng, lang, seed=seed) for lang in IF7_LANGS]


def _if7_eval(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    value, strict = text.strict_json(tr.final_text)

    def number(raw: Any) -> Decimal | None:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        try:
            return Decimal(str(raw))
        except InvalidOperation:
            return None

    rows = value.get("rows") if isinstance(value, dict) else None
    iso = rows is not None and isinstance(rows, list) and all(
        isinstance(row, dict)
        and set(row) == {"date", "invoice", "tonnes"}
        and isinstance(row.get("date"), str)
        and isinstance(row.get("invoice"), str)
        and number(row.get("tonnes")) is not None
        for row in rows
    )
    try:
        parsed_dates = [date.fromisoformat(row["date"]) for row in rows] if iso else []
        iso = iso and len(parsed_dates) == len(rows)
    except ValueError:
        iso = False
    fmt_ok = (
        strict
        and isinstance(value, dict)
        and set(value) == {"rows", "total_tonnes"}
        and iso
        and number(value.get("total_tonnes")) is not None
    )

    rows_ok = False
    if fmt_ok:
        rows_ok = len(rows) == len(oracle["rows"]) and all(
            row["date"] == expected["date"]
            and row["invoice"] == expected["invoice"]
            and text.decimals_equal(number(row["tonnes"]), expected["tonnes"], places=3)
            for row, expected in zip(rows, oracle["rows"])
        )
    total_ok = False
    if fmt_ok:
        total_ok = text.decimals_equal(number(value["total_tonnes"]), oracle["total_tonnes"], places=3)

    _format_point(score, "json_format",
        fmt_ok,
        "ожидался строгий JSON с ключами rows и total_tonnes, ISO-датами, NFC-номерами и числовой массой"
        if not fmt_ok else "ok",
    )
    _content_point(score, "rows", rows_ok,
        f"получено {rows if isinstance(rows, list) else None}, ожидалось {oracle['rows']} "
        "(номер — в NFC, повтор с более поздней датой)",
    )
    _content_point(score, "total", total_ok,
        f"получено {value.get('total_tonnes') if isinstance(value, dict) else None}, "
        f"ожидалось {oracle['total_tonnes']}",
    )
    return score


# --------------------------------------------------------------------------- #
# IF-8 — RTL input, NFC/NFD key comparison
# --------------------------------------------------------------------------- #

_IF8_HEBREW = ("דבש", "תַּפּוּחַ", "שָׁלוֹם")
_IF8_ARABIC = ("مفتاح", "مرآة")
_IF8_COMPOSED = ("йод", "ёлка", "объём", "звёзды")
"""Keys whose NFC and NFD spellings really differ (composing Cyrillic letters)."""

_IF8_KEYS = _IF8_HEBREW + _IF8_ARABIC + _IF8_COMPOSED + ("подъезд",)


def _if8_rows(rng: random.Random) -> list[dict[str, str]]:
    """Ten records: five distinct keys, each written twice in two Unicode forms.

    Two composing Cyrillic keys prove the normalization, one Hebrew and one
    Arabic key keep both RTL scripts in the brief.
    """
    keys = rng.sample(_IF8_COMPOSED, 2) + [rng.choice(_IF8_HEBREW), rng.choice(_IF8_ARABIC)]
    rest = rng.sample([key for key in _IF8_KEYS if key not in keys], 1)
    keys += rest
    rng.shuffle(keys)
    rows: list[dict[str, str]] = []
    for index, key in enumerate(keys):
        first = unicodedata.normalize("NFC", key) if index % 2 == 0 else unicodedata.normalize("NFD", key)
        second = unicodedata.normalize("NFD", key) if index % 2 == 0 else unicodedata.normalize("NFC", key)
        # The decomposed spelling is the later, winning record.
        rows.append({"key": first, "day": f"2027-02-{10 + index:02d}", "value": f"v{rng.randrange(100, 999)}"})
        rows.append({"key": second, "day": f"2027-03-{10 + index:02d}", "value": f"v{rng.randrange(100, 999)}"})
    distractor = rng.choice([key for key in _IF8_KEYS if key not in keys])
    rows.append({"key": unicodedata.normalize("NFD", distractor), "day": "2027-01-05", "value": "v000"})
    extra = rng.choice([key for key in _IF8_KEYS if key not in keys and key != distractor])
    rows.append({"key": unicodedata.normalize("NFC", extra), "day": "2027-04-01", "value": "v999"})
    rng.shuffle(rows)
    return rows


def _if8_selection(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        key = unicodedata.normalize("NFC", row["key"])
        current = latest.get(key)
        if current is None or row["day"] >= current["day"]:
            latest[key] = {"key": key, "day": row["day"], "value": row["value"]}
    return [latest[key] for key in sorted(latest)]


def _if8(rng: random.Random) -> core.TaskInstance:
    rows = _if8_rows(rng)
    selected = _if8_selection(rows)
    listing = "\n".join(f"{row['key']}\t{row['day']}\t{row['value']}" for row in rows)
    reference = json.dumps(
        {"keys": [{"key": row["key"], "value": row["value"]} for row in selected]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = (
        "Даны записи (поля через табуляцию): ключ, дата, значение.\n\n"
        f"{listing}\n\n"
        "Ключи записаны на разных письменностях (иврит, арабский, кириллица), "
        "включая RTL. Один и тот же ключ может быть записан в разложенной форме Unicode "
        "(NFD) и в составленной (NFC) — это один и тот же ключ.\n\n"
        "Правила:\n"
        "1. сравнивай ключи после нормализации NFC;\n"
        "2. для каждого ключа оставь запись с более поздней датой;\n"
        "3. ключ в ответе выводи в NFC;\n"
        "4. отсортируй ключи по возрастанию кодовых точек Unicode.\n\n"
        "Верни ТОЛЬКО валидный JSON:\n\n"
        '{"keys":[{"key":"...","value":"..."}]}\n\n'
        "Никакого Markdown и дополнительного текста."
    )
    return core.TaskInstance(
        id="IF-8",
        category="A",
        title="RTL и NFC/NFD ключи",
        prompt=prompt,
        max_points=2.0,
        oracle={
            "reference": reference,
            "keys": [{"key": row["key"], "value": row["value"]} for row in selected],
            "distractor_key": unicodedata.normalize("NFC", selected[0]["key"]),
        },
        languages=["ru"],
        cultures=["il", "kz"],
        fresh_after=FRESH_AFTER,
        meta={
            "scenario": "IF-8",
            "rows": len(rows),
            "distinct_keys": len(selected),
            "decomposed_rows": sum(1 for row in rows if unicodedata.normalize("NFC", row["key"]) != row["key"]),
            "languages": ["ru"],
            "cultures": ["il", "kz"],
        },
    )


def _if8_eval(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    value, strict = text.strict_json(tr.final_text)
    payload = value.get("keys") if isinstance(value, dict) else None

    shaped = isinstance(payload, list) and all(
        isinstance(item, dict) and set(item) == {"key", "value"}
        and isinstance(item.get("key"), str) and isinstance(item.get("value"), str)
        for item in payload
    )
    fmt_ok = strict and isinstance(value, dict) and set(value) == {"keys"} and shaped

    # Промпт требует ключ в NFC: сравниваем выписанную строку с оракулом,
    # не нормализуя ответ. Иначе правило NFC не проверяется.
    emitted = payload if shaped else []
    pairs = sorted((item["key"], item["value"]) for item in emitted)
    expected_pairs = sorted((item["key"], item["value"]) for item in oracle["keys"])
    keys_ok = bool(shaped) and pairs == expected_pairs
    order_ok = bool(shaped) and [item["key"] for item in emitted] == [item["key"] for item in oracle["keys"]]

    _format_point(score, "json_format",
        fmt_ok,
        "ожидался строгий JSON с ключом keys и парами key/value"
        if not fmt_ok else "ok",
    )
    _content_point(score, "keys", keys_ok,
        f"получено {emitted or None}, ожидалось {oracle['keys']}",
    )
    _content_point(score, "order", order_ok,
        f"получено {[item['key'] for item in emitted]}, ожидалось {[item['key'] for item in oracle['keys']]}",
    )
    return score


# --------------------------------------------------------------------------- #
# IF-9 / IF-10 — composite selection: declared rule priority, NFC keys,
# a mid-dialogue bound change, and the empty selection
# --------------------------------------------------------------------------- #

_N7_HEADER = ("id", "key", "zone", "severity", "score", "ts", "rev", "muted")

_N7_GROUP_KEYS = ("йод", "ёлка", "объём", "звёзды")
"""Keys whose NFC and NFD spellings really differ (composing Cyrillic letters)."""
_N7_SPARE_KEYS = ("подъезд", "сеть", "мост", "свод", "смена", "пост", "канал", "шлюз")

_N7_FORMAT = 0.25
_N7_CONTENT = 0.375


def _n7_spell(key: str, form: str) -> str:
    return unicodedata.normalize(form, key)


def _n7_rows(rng: random.Random, variant: str) -> tuple[list[dict[str, Any]], int, int, str]:
    """Fixed skeleton of eleven to thirteen records around the two bounds.

    The roles are what the rules below are proven against: one key group loses
    its lower ``rev`` member, a muted record leads every score, and the two
    records the corrected bound keeps tie on ``score`` so the declared
    tie-break decides their order. ``variant == "empty"`` lifts the corrected
    bound above every score without touching the first bound.
    """
    t1 = rng.choice((55, 60, 65))
    t2 = t1 + (10 if variant == "select" else 25)
    target = rng.choice(("core", "edge"))
    other = "edge" if target == "core" else "core"
    first, second = rng.sample(_N7_GROUP_KEYS, 2)
    muted_key, *spare = rng.sample(_N7_SPARE_KEYS, len(_N7_SPARE_KEYS))

    if variant == "select":
        scores = {
            "a": t2 + 10, "b": t2, "c": t1, "d": t2, "e": t2 - 5,
            "f": t2 + 20, "g": t1 - 4, "h": t2 + 15, "k": t2 + 5,
        }
    else:
        scores = {
            "a": t1 + 10, "b": t1 + 2, "c": t1, "d": t1, "e": t1 - 5,
            "f": t1 + 24, "g": t1 - 9, "h": t1 + 20, "k": t1 + 12,
        }

    def record(role: str, key: str, zone: str, severity: str, rev: int, muted: str = "no") -> dict[str, Any]:
        row: dict[str, Any] = {
            "role": role, "key": key, "zone": zone, "severity": severity,
            "rev": rev, "muted": muted,
        }
        if role in scores:
            row["score"] = scores[role]
        return row

    rows = [
        record("a", _n7_spell(first, "NFC"), target, "critical", 1),
        record("b", _n7_spell(first, "NFD"), target, "critical", 3),
        record("c", _n7_spell(second, "NFD"), target, "critical", 2),
        record("d", _n7_spell(second, "NFC"), target, "critical", 4),
        record("e", spare[0], target, "critical", 1),
        record("f", _n7_spell(muted_key, "NFD"), target, "critical", 2, "yes"),
        record("g", _n7_spell(muted_key, "NFC"), target, "critical", 1),
        record("h", spare[1], other, "critical", 5),
        record("k", spare[2], target, "warn", 1),
    ]
    for index in range(rng.randrange(2, 5)):
        row = record(f"filler{index}", spare[3 + index], target, "critical", 1)
        row["score"] = t1 - rng.randrange(6, 40)
        rows.append(row)

    ids = dict(zip((row["role"] for row in rows), rng.sample(range(100, 1000), len(rows))))
    minutes = dict(zip((row["role"] for row in rows), rng.sample(range(480, 1150, 5), len(rows))))
    # The tie-break must be able to lose: the earlier record carries the larger id.
    if minutes["d"] > minutes["b"]:
        minutes["b"], minutes["d"] = minutes["d"], minutes["b"]
    if ids["b"] > ids["d"]:
        ids["b"], ids["d"] = ids["d"], ids["b"]
    for row in rows:
        row["id"] = ids[row["role"]]
        minute = minutes[row["role"]]
        row["ts"] = f"{minute // 60:02d}:{minute % 60:02d}"
    return rows, t1, t2, target


def _n7_selection(rows: list[dict[str, Any]], zone: str, threshold: int) -> list[dict[str, Any]]:
    """Rules in the declared order: include, exclusion, then one row per NFC key."""
    survivors = [
        row for row in rows
        if row["zone"] == zone and row["severity"] == "critical"
        and row["score"] >= threshold and row["muted"] != "yes"
    ]
    best: dict[str, dict[str, Any]] = {}
    for row in survivors:
        key = unicodedata.normalize("NFC", row["key"])
        if key not in best or row["rev"] > best[key]["rev"]:
            best[key] = row
    return sorted(best.values(), key=lambda row: (-row["score"], row["ts"], row["id"]))


def _n7(rng: random.Random, variant: str) -> core.TaskInstance:
    rows, t1, t2, target = _n7_rows(rng, variant)
    scenario = "IF-9" if variant == "select" else "IF-10"
    rng.shuffle(rows)

    by_role = {row["role"]: row for row in rows}
    pre = _n7_selection(rows, target, t1)
    picked = _n7_selection(rows, target, t2)
    assert pre, "the first bound must select something"
    assert by_role["f"]["muted"] == "yes"
    assert by_role["f"]["score"] == max(row["score"] for row in rows), "the muted row leads every score"
    if variant == "select":
        assert picked == [by_role["d"], by_role["b"]], "the corrected bound keeps exactly the tie pair"
        assert by_role["d"]["score"] == t2, "the second kept row sits exactly on the corrected bound"
        assert by_role["d"]["ts"] < by_role["b"]["ts"] and by_role["d"]["id"] > by_role["b"]["id"]
        assert [row["id"] for row in picked] != [row["id"] for row in pre]
    else:
        assert picked == [], "the corrected bound must empty the selection"
    for lower, higher in ((by_role["a"], by_role["b"]), (by_role["c"], by_role["d"])):
        assert lower["key"] != higher["key"]
        assert unicodedata.normalize("NFC", lower["key"]) == unicodedata.normalize("NFC", higher["key"])
        assert higher["rev"] > lower["rev"]

    listing = "\n".join(
        "\t".join([
            str(row["id"]), row["key"], row["zone"], row["severity"],
            str(row["score"]), row["ts"], str(row["rev"]), row["muted"],
        ])
        for row in rows
    )
    picks = [{"id": row["id"], "score": row["score"]} for row in picked]
    total = sum(item["score"] for item in picks)
    reference = json.dumps(
        {"picks": picks, "count": len(picks), "total": total},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = "\n\n".join([
        "Даны записи смены (поля через табуляцию): id, key, zone, severity, score, ts, rev, muted.",
        "{}\n{}".format("\t".join(_N7_HEADER), listing),
        "score, rev и id — целые числа; ts — время ЧЧ:ММ; muted — yes или no.\n"
        "Один и тот же key может быть записан в разложенной (NFD) и в составленной (NFC) "
        "форме Unicode — это один и тот же key.",
        "Правила применяются строго сверху вниз:\n"
        f"1. отбор: zone = {target} и severity = critical и score >= {t1};\n"
        "2. если muted = yes, запись исключается, даже когда подходит под п.1;\n"
        "3. один key — одна запись: key сравниваются после нормализации NFC, "
        "из группы остаётся запись с бо́льшим rev.",
        "picks — отобранные записи в виде {\"id\":…,\"score\":…}.\n"
        "Порядок picks: score по убыванию; при равном score — ts по возрастанию; "
        "при равенстве — id по возрастанию.\n"
        "count — длина picks, total — сумма score отобранных записей.",
        "Верни ТОЛЬКО валидный JSON:\n\n"
        '{"picks":[{"id":0,"score":0}],"count":1,"total":0}\n\n'
        "Никакого Markdown и дополнительного текста.",
    ])
    return core.TaskInstance(
        id=scenario,
        category="A",
        title="Составной отбор с приоритетом правил",
        prompt=prompt,
        max_points=1.0,
        dialogue=[
            f"Поправка: порог score теперь {t2}, а не {t1}. "
            "Остальные правила, порядок picks и формат прежние."
        ],
        oracle={
            "reference": reference,
            "picks": picks,
            "ids": [item["id"] for item in picks],
            "count": len(picks),
            "total": total,
            "zone": target,
            "threshold1": t1,
            "threshold2": t2,
            "tie_ids": [by_role["d"]["id"], by_role["b"]["id"]],
            "muted_id": by_role["f"]["id"],
            "low_rev_ids": sorted((by_role["a"]["id"], by_role["c"]["id"])),
            "superseded_ids": [row["id"] for row in pre],
        },
        languages=["ru"],
        fresh_after=FRESH_AFTER,
        meta={
            "scenario": scenario,
            "variant": variant,
            "rows": len(rows),
            "zone": target,
            "threshold1": t1,
            "threshold2": t2,
        },
    )


def _n7_eval(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="A")
    oracle = task.oracle
    value, strict = text.strict_json(tr.final_text)
    if not isinstance(value, dict):
        value = text.loose_json(tr.final_text)
    picks = value.get("picks") if isinstance(value, dict) else None

    def integer(candidate: Any) -> bool:
        return isinstance(candidate, int) and not isinstance(candidate, bool)

    shaped = isinstance(picks, list) and all(
        isinstance(item, dict)
        and set(item) == {"id", "score"}
        and integer(item.get("id"))
        and integer(item.get("score"))
        for item in picks
    )
    fmt_ok = (
        strict
        and isinstance(value, dict)
        and set(value) == {"picks", "count", "total"}
        and shaped
        and integer(value.get("count"))
        and integer(value.get("total"))
    )
    got = [(item["id"], item["score"]) for item in picks] if shaped else []
    expected = [(item["id"], item["score"]) for item in oracle["picks"]]
    ids_ok = bool(shaped) and sorted(got) == sorted(expected)
    order_ok = (
        bool(shaped)
        and got == expected
        and isinstance(value, dict)
        and value.get("count") == oracle["count"]
        and value.get("total") == oracle["total"]
    )
    score.add(
        "json_format", fmt_ok, _N7_FORMAT,
        "ожидался строгий JSON с ключами picks, count, total, парами id/score и целыми значениями"
        if not fmt_ok else "ok",
    )
    score.add(
        "records", ids_ok, _N7_CONTENT,
        f"получено {[item[0] for item in got]}, ожидалось {oracle['ids']}",
    )
    score.add(
        "ordering", order_ok, _N7_CONTENT,
        f"получено {got} / count={value.get('count') if isinstance(value, dict) else None} / "
        f"total={value.get('total') if isinstance(value, dict) else None}, "
        f"ожидалось {expected} / count={oracle['count']} / total={oracle['total']}",
    )
    return score


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

_SCENARIOS = (
    ("IF-1", _if1, _eval_if1),
    ("IF-2", _if2, _eval_if2),
    ("IF-3", _if3, _eval_if3),
    ("IF-4", _if4, _eval_if4),
    ("IF-5", _if5, _eval_if5),
)

_PILOT_SCENARIOS = (("IF-6", _if6, _eval_if6),)

_STRETCH_SCENARIOS = (
    ("IF-7", lambda rng: _if7(rng, "ru"), _if7_eval),
    ("IF-7-KK", lambda rng: _if7(rng, "kk"), _if7_eval),
    ("IF-7-EN", lambda rng: _if7(rng, "en"), _if7_eval),
    ("IF-8", _if8, _if8_eval),
    ("IF-9", lambda rng: _n7(rng, "select"), _n7_eval),
    ("IF-10", lambda rng: _n7(rng, "empty"), _n7_eval),
)

_EVALUATORS = {
    scenario: fn for scenario, _, fn in _SCENARIOS + _PILOT_SCENARIOS + _STRETCH_SCENARIOS
}


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    profile = core.profile_of(cfg)
    if profile == "pilot":
        return [builder(rng) for _, builder, _ in _PILOT_SCENARIOS]
    if profile == "research-v1":
        tasks = [
            builder(rng)
            for scenario, builder, _ in _STRETCH_SCENARIOS
            if scenario in RESEARCH_IDS
        ]
        for task in tasks:
            task.private = task.id in PRIVATE_IDS
        return tasks
    tasks = [builder(rng) for _, builder, _ in _SCENARIOS]
    if profile == "stretch-v1":
        tasks.extend(builder(rng) for _, builder, _ in _PILOT_SCENARIOS)
        # One shared fixture seed for the three locales, so the twins differ
        # only in presentation and never in the underlying case.
        tasks.extend(_if7_variants(rng))
        tasks.extend(
            builder(rng) for scenario, builder, _ in _STRETCH_SCENARIOS
            if scenario not in IF7_IDS
        )
    for task in tasks:
        task.private = task.id in PRIVATE_IDS
    return tasks


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    return _EVALUATORS[task.id](task, tr)


core.register(
    key="A",
    title="Instruction following",
    points=10.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
    expected_ids_by_profile={"pilot": PILOT_IDS, "stretch-v1": STRETCH_IDS, "research-v1": RESEARCH_IDS},
)
