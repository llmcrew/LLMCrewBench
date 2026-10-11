"""Category A — instruction following, 5 scenarios × 3 checks × 1 point.

Each scenario scores three separable things: the requested output format, the
exact content, and the order or derived value the instruction demanded. A
model that gets the content right but ignores the format loses one point, not
the scenario — that separation is the point of the category.
"""

from __future__ import annotations

import random
from typing import Any

from .. import core, text

IDS = ("IF-1", "IF-2", "IF-3", "IF-4", "IF-5")

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
# Registration
# --------------------------------------------------------------------------- #

_SCENARIOS = (
    ("IF-1", _if1, _eval_if1),
    ("IF-2", _if2, _eval_if2),
    ("IF-3", _if3, _eval_if3),
    ("IF-4", _if4, _eval_if4),
    ("IF-5", _if5, _eval_if5),
)

_EVALUATORS = {scenario: fn for scenario, _, fn in _SCENARIOS}


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    return [builder(rng) for _, builder, _ in _SCENARIOS]


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    return _EVALUATORS[task.id](task, tr)


core.register(
    key="A",
    title="Instruction following",
    points=10.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
)
