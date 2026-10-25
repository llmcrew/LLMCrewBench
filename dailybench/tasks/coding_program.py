"""Scenario N-9 — program from spec (stretch only): treasury ledger package.

The candidate delivers a small ``ledger`` package with three commands,
``post``, ``statement`` and ``unpost``. The trusted side only exercises the
public CLI in a throwaway copy of the worktree and compares observable state:

* the exact stdout contract and the exit codes,
* the strict JSON statement, recomputed here from the structured entries with
  ``Decimal`` (half-up ties are constructed so binary floats drift below the
  rounding boundary),
* idempotency — re-posting a batch applies nothing and leaves the state and the
  report byte-identical, a mixed batch applies exactly its new keys,
* atomicity — an invalid batch changes neither the state tree nor the report,
  and a clean copy stays empty,
* the protected fixture files and the write scope (``state/`` and the report).

A correct answer may split the modules, the state format and the arithmetic
differently; only the declared behaviour is graded. The candidate's own
``ok:`` line is a claim that still has to survive the checks above.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from .. import core

#: Ten checks of 0.1 each; the profile distributes the category total.
_POINTS_PER_CHECK = 0.1

PACKAGE = "ledger"
MAIN_FILE = "ledger/__main__.py"
INIT_FILE = "ledger/__init__.py"
STATE_DIR = "state/"
REPORT_PATH = "reports/statement.json"
ACCOUNTS_PATH = "config/accounts.json"
RATES_PATH = "config/rates.json"
README_PATH = "README.md"
VALID_BATCH = "data/batch_2026_10_01.jsonl"
INVALID_BATCH = "data/batch_2026_10_02.jsonl"
MIXED_BATCH = "data/batch_2026_10_03.jsonl"

PROTECTED = (README_PATH, ACCOUNTS_PATH, RATES_PATH, VALID_BATCH, INVALID_BATCH, MIXED_BATCH)

ALLOWED_PREFIX = STATE_DIR
ALLOWED_FILES = (REPORT_PATH,)

CENT = Decimal("0.01")
LIMIT = Decimal("1000000")
KEY_RE = re.compile(r"^[a-z0-9-]{3,32}$")
ID_RE = re.compile(r"^TX-[0-9]{4}$")
AMOUNT_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{4}$")
MONEY_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{2}$")


#: Two accounts settle in XDR at exactly ``1.0005``; a whole target of ten
#: makes the converted total end in a true half (``10.005``), and the
#: ``0.7 + 0.1 + 0.2`` split pushes a binary-float sum just below it.
RATES = {
    "EUR": "1.0000",
    "USD": "0.8255",
    "GBP": "0.7412",
    "CHF": "1.0512",
    "XDR": "1.0005",
    "JPY": "0.0059",
}

ACCOUNTS: tuple[tuple[str, str], ...] = (
    ("ACC-100", "USD"),
    ("ACC-210", "XDR"),
    ("ACC-305", "GBP"),
    ("ACC-411", "CHF"),
    ("ACC-502", "XDR"),
    ("ACC-640", "JPY"),
)

TIE_TARGETS = (10, 30, 50)

_STUB_INIT = '''"""Treasury ledger package. Implement the commands in ``__main__``."""
'''

_STUB_MAIN = '''#!/usr/bin/env python3
"""Treasury ledger: post | statement | unpost."""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    raise NotImplementedError("ledger is not implemented")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''

_REFERENCE_MAIN = r'''#!/usr/bin/env python3
"""Treasury ledger: post | statement | unpost."""

from __future__ import annotations

import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

STATE = Path("state/ledger.json")
REPORT = Path("reports/statement.json")
ACCOUNTS = Path("config/accounts.json")
RATES = Path("config/rates.json")
USAGE = "usage: python3 -m ledger post <batch.jsonl> | statement | unpost <key>"

CENT = Decimal("0.01")
LIMIT = Decimal("1000000")
KEY_RE = re.compile(r"^[a-z0-9-]{3,32}$")
ID_RE = re.compile(r"^TX-[0-9]{4}$")
AMOUNT_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{4}$")
FIELDS = {"id", "account", "amount", "key"}


def usage(message: str) -> int:
    print(USAGE, file=sys.stderr)
    if message:
        print(message, file=sys.stderr)
    return 64


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def accounts() -> dict[str, str]:
    payload = json.loads(ACCOUNTS.read_text(encoding="utf-8"))
    return {item["code"]: item["currency"] for item in payload["accounts"]}


def rates() -> dict[str, Decimal]:
    payload = json.loads(RATES.read_text(encoding="utf-8"))
    return {code: Decimal(value) for code, value in payload.items()}


def load() -> list[dict[str, str]]:
    if not STATE.is_file():
        return []
    payload = json.loads(STATE.read_text(encoding="utf-8"))
    return list(payload["entries"])


def save(entries: list[dict[str, str]]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_name(".ledger.json.tmp")
    payload = json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, separators=(",", ":"))
    tmp.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp, STATE)


def parse_line(line: str, known: dict[str, str], rate_table: dict[str, Decimal]) -> dict[str, str] | None:
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(item, dict) or set(item) != FIELDS:
        return None
    if not all(isinstance(item[field], str) for field in FIELDS):
        return None
    if not ID_RE.match(item["id"]) or not KEY_RE.match(item["key"]):
        return None
    account = item["account"]
    if account not in known or known[account] not in rate_table:
        return None
    amount = item["amount"]
    if not AMOUNT_RE.match(amount):
        return None
    value = Decimal(amount)
    if value <= 0 or value > LIMIT:
        return None
    return {
        "id": item["id"],
        "account": account,
        "amount": amount,
        "key": item["key"],
    }


def read_batch(path: str) -> tuple[list[dict[str, str]] | None, list[int]]:
    if not Path(path).is_file():
        return None, []
    known = accounts()
    rate_table = rates()
    entries: list[dict[str, str]] = []
    bad: list[int] = []
    seen_keys: set[str] = set()
    seen_ids: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            bad.append(number)
            continue
        entry = parse_line(line, known, rate_table)
        if entry is None or entry["key"] in seen_keys or entry["id"] in seen_ids:
            bad.append(number)
            continue
        seen_keys.add(entry["key"])
        seen_ids.add(entry["id"])
        entries.append(entry)
    return entries, bad


def money(value: Decimal) -> str:
    return str(value.quantize(CENT, rounding=ROUND_HALF_UP))


def command_post(path: str) -> int:
    entries, bad = read_batch(path)
    if entries is None:
        return fail(f"invalid batch: missing {path}")
    if bad:
        return fail("invalid batch: " + ",".join(str(number) for number in bad))
    current = load()
    have = {entry["key"] for entry in current}
    applied = 0
    skipped = 0
    for entry in entries:
        if entry["key"] in have:
            skipped += 1
            continue
        have.add(entry["key"])
        current.append(entry)
        applied += 1
    if applied:
        save(current)
    print(f"ok: applied={applied} skipped={skipped}")
    return 0


def command_unpost(key: str) -> int:
    current = load()
    kept = [entry for entry in current if entry["key"] != key]
    if len(kept) == len(current):
        return fail_missing(key)
    save(kept)
    print(f"ok: removed=1")
    return 0


def fail_missing(key: str) -> int:
    print(f"missing key: {key}", file=sys.stderr)
    return 3


def command_statement() -> int:
    current = load()
    known = accounts()
    rate_table = rates()
    totals: dict[str, list[Any]] = {}
    for entry in current:
        bucket = totals.setdefault(entry["account"], [0, Decimal("0")])
        bucket[0] += 1
        bucket[1] += Decimal(entry["amount"]) * rate_table[known[entry["account"]]]
    grand = sum((bucket[1] for bucket in totals.values()), Decimal("0"))
    payload = {
        "accounts": [
            {"account": account, "entries": totals[account][0], "eur_total": money(totals[account][1])}
            for account in sorted(totals)
        ],
        "rows": len(current),
        "grand_total": money(grand),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_name(".statement.json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, REPORT)
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        return usage("no command")
    command, *rest = argv
    if command == "post" and len(rest) == 1:
        return command_post(rest[0])
    if command == "unpost" and len(rest) == 1:
        return command_unpost(rest[0])
    if command == "statement" and not rest:
        return command_statement()
    return usage(f"unknown command: {' '.join(argv)}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''

_PROMPT = """Рабочая копия — касса с расчётными счетами. Нужен пакет `{package}` на Python 3.12 (только стандартная библиотека, без сети) с тремя командами.

Запуск: `python3 -m {package} <команда> [аргумент]` в корне рабочей копии.

Команды:

1. `python3 -m {package} post <batch>` — добавить проводки из файла batch. Повторная проводка с уже применённым `key` не должна учитываться снова.
2. `python3 -m {package} statement` — записать отчёт `{report}` по текущему состоянию.
3. `python3 -m {package} unpost <key>` — удалить проводку с этим `key`.

Коды возврата: успех — 0; неверные аргументы или неизвестная команда — 64 и строка `usage:` в stderr; невалидный batch или отсутствующий файл — 2 (см. ниже); `unpost` по отсутствующему ключу — 3, первая строка stderr ровно `missing key: <key>`.

Формат batch — UTF-8, по одному JSON-объекту в строке, без пустых строк. У объекта ровно четыре строковых поля: `id`, `account`, `amount`, `key`.

Валидация строки: `id` соответствует `TX-` и ровно четырём цифрам и уникален внутри файла; `account` есть в `{accounts}`; `amount` — строка из цифр и ровно четырёх знаков после точки, значение больше 0 и не больше 1000000.0000; `key` соответствует `[a-z0-9-]{{3,32}}` и уникален внутри файла.

Номер строки считается с 1. Любая невалидная строка, пустая строка, неверный набор полей, неизвестный счёт или валюта без курса — код 2, первая строка stderr: `invalid batch: <номера всех невалидных строк по возрастанию через запятую>`; для отсутствующего файла — `invalid batch: missing <путь>`.

Идемпотентность: `key` — ключ идемпотентности, он остаётся уникальным и после `unpost` (проводка с тем же `key` после удаления считается новым применением). Строка, чей `key` уже есть в состоянии, пропускается независимо от остальных полей, но всё равно валидируется. `post` печатает ровно одну строку `ok: applied=<применено> skipped=<пропущено>`.

Атомарность: определение валидности batch завершается до любой записи. Если batch невалиден, состояние, отчёт и остальные файлы остаются байт-в-байт прежними; на чистой копии не появляется ни состояния, ни отчёта. `statement` — только чтение состояния. Повторный `post` того же batch не меняет ни состояние, ни отчёт.

Отчёт — строгий JSON в одну строку, ключи ровно в этом порядке:
`{{"accounts":[{{"account":"<счёт>","entries":<число проводок>,"eur_total":"<сумма в EUR, 2 знака>"}}],"rows":<всего проводок>,"grand_total":"<общая сумма в EUR, 2 знака>"}}`
`accounts` содержит только счета с проводками и отсортирован по коду; `entries` — число проводок счёта; суммы и `grand_total` — ровно две цифры после точки. Пересчёт в EUR: сумма `amount` по счёту умножается на курс его валюты, затем округляется по правилу половина вверх до двух знаков; `grand_total` — сумма точных произведений по всем счетам, округлённая так же. Арифметика — десятичная (`decimal.Decimal`), не двоичная.

Курсы лежат в `{rates}`, счета — в `{accounts_path}`. Состояние пакета храните в `{state}` (формат свой). Вне `{state}` и `{report}` ничего не должно появиться или измениться. Защищённые файлы менять нельзя: {protected}.

Верни ТОЛЬКО файлы пакета, перед каждым блоком кода строка `### FILE: <путь>`. Обязательны `{main}` и `{init}`; путь — только внутри `{package}/`. Сейчас в копии лежат:

````
### FILE: {main}
```python
{stub_main}```

### FILE: {init}
```python
{stub_init}```
````
"""


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _amount(rng: random.Random) -> str:
    return f"{rng.randrange(1, 999_999)}.{rng.randrange(0, 10_000):04d}"


def _entries(rng: random.Random, chosen: list[str], tie_accounts: list[str]) -> list[dict[str, str]]:
    """Valid entries with constructed rounding ties and bounded filler amounts."""
    ids = rng.sample(range(1000, 9999), 40)
    keys = rng.sample(range(1000, 9999), 40)
    rows: list[dict[str, str]] = []
    cursor = 0

    def add(account: str, amount: str) -> None:
        nonlocal cursor
        rows.append({
            "id": f"TX-{ids[cursor]:04d}",
            "account": account,
            "amount": amount,
            "key": f"idem-{keys[cursor]:04d}",
        })
        cursor += 1

    for account in tie_accounts:
        target = rng.choice(TIE_TARGETS)
        # 0.7 + 0.1 + 0.2 is just below 1.0 in binary floating point.
        for amount in ("0.7000", "0.1000", "0.2000", f"{target - 1}.0000"):
            add(account, amount)
    for account in chosen:
        if account in tie_accounts:
            continue
        for _ in range(rng.randrange(2, 4)):
            add(account, _amount(rng))
    rng.shuffle(rows)
    return rows


def _jsonl(rows: list[dict[str, str]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def _invalid_batch(rng: random.Random, rows: list[dict[str, str]]) -> tuple[str, list[int]]:
    """Break eight lines in ways the contract names; returns JSONL and line numbers.

    Every broken line carries exactly one violation of the declared validator:
    unknown account, three decimals, a bad key shape, a bad id shape, a zero
    amount, an extra field, a duplicate id and a truncated JSON object.
    """
    marked: list[tuple[str, bool]] = []
    lines = [dict(row) for row in rows]
    lines[0]["account"] = "ACC-999"
    lines[1]["amount"] = "1.234"
    lines[2]["key"] = "Bad Key"
    lines[3]["id"] = "TX-12"
    lines[4]["amount"] = "0.0000"
    lines[5]["note"] = "extra"
    marked.extend((_json_line(row), index < 6) for index, row in enumerate(lines))
    marked.append((
        _json_line({"id": "TX-12", "account": lines[6]["account"], "amount": "1.0000", "key": "idem-0001"}),
        True,
    ))
    marked.append(('{"id": "TX-9999", "account": "' + lines[7]["account"] + '",', True))
    # Two clean rows that are not in the applied batch yet: a partial write of
    # the valid prefix must become visible in the state.
    used_ids = {row["id"] for row in rows} | {"TX-12"}
    used_keys = {row["key"] for row in rows} | {"idem-0001"}
    for index in range(2):
        fresh_id = next(f"TX-{value:04d}" for value in range(2000, 3000) if f"TX-{value:04d}" not in used_ids)
        fresh_key = next(f"idem-{value:04d}" for value in range(3000, 4000) if f"idem-{value:04d}" not in used_keys)
        used_ids.add(fresh_id)
        used_keys.add(fresh_key)
        marked.append((_json_line({
            "id": fresh_id,
            "account": lines[index + 8]["account"],
            "amount": "3.2500",
            "key": fresh_key,
        }), False))
    rng.shuffle(marked)
    bad = [number for number, (_line, broken) in enumerate(marked, start=1) if broken]
    return "\n".join(line for line, _broken in marked) + "\n", bad


def _json_line(row: dict[str, str]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def _mixed_batch(rng: random.Random, rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Two known keys plus two fresh ones; returns JSONL and the fresh entries."""
    known = rng.sample(rows, 2)
    # Disjoint ranges keep the four lines unique by id, and the fresh keys never
    # collide with an applied one, so the applied/skipped counts stay fixed.
    clone_ids = rng.sample(range(6000, 7000), 2)
    fresh_ids = rng.sample(range(7000, 8000), 2)
    taken = {row["key"] for row in rows}
    candidates = [f"idem-{value:04d}" for value in range(8000, 8500) if f"idem-{value:04d}" not in taken]
    fresh_keys = [int(candidate.split("-")[1]) for candidate in rng.sample(candidates, 2)]
    mixed = [
        {"id": f"TX-{clone_ids[index]:04d}", "account": known[index]["account"],
         "amount": "1.0000", "key": known[index]["key"]}
        for index in range(2)
    ]
    fresh = [
        {"id": f"TX-{fresh_ids[index]:04d}", "account": known[index]["account"],
         "amount": "2.5000", "key": f"idem-{fresh_keys[index]:04d}"}
        for index in range(2)
    ]
    mixed += fresh
    rng.shuffle(mixed)
    return _jsonl(mixed), fresh


def _accounts_payload(chosen: list[str]) -> str:
    payload = {
        "accounts": [
            {"code": account, "currency": dict(ACCOUNTS)[account]} for account in sorted(chosen)
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _currency(account: str) -> str:
    return dict(ACCOUNTS)[account]


def _report(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["account"]] = counts.get(row["account"], 0) + 1
    exact: dict[str, Decimal] = {}
    for row in rows:
        exact[row["account"]] = exact.get(row["account"], Decimal("0")) + Decimal(row["amount"]) * Decimal(
            RATES[_currency(row["account"])]
        )
    grand = sum(exact.values(), Decimal("0"))
    return {
        "accounts": [
            {"account": account, "entries": counts[account], "eur_total": str(exact[account].quantize(CENT, rounding=ROUND_HALF_UP))}
            for account in sorted(counts)
        ],
        "rows": len(rows),
        "grand_total": str(grand.quantize(CENT, rounding=ROUND_HALF_UP)),
    }


def generate(rng: random.Random, cfg: Any) -> core.TaskInstance:
    tie_pool = [code for code, currency in ACCOUNTS if currency == "XDR"]
    others = [code for code, _currency_name in ACCOUNTS if code not in tie_pool]
    tie_accounts = rng.sample(tie_pool, 2)
    chosen = tie_accounts + rng.sample(others, 2)

    rows = _entries(rng, chosen, tie_accounts)
    invalid_csv, bad_lines = _invalid_batch(rng, rows)
    mixed_csv, fresh_entries = _mixed_batch(rng, rows)

    unpost_key = rows[rng.randrange(len(rows))]["key"]
    applied_after_unpost = [row for row in rows + fresh_entries if row["key"] != unpost_key]

    prompt = _PROMPT.format(
        package=PACKAGE,
        main=MAIN_FILE,
        init=INIT_FILE,
        report=REPORT_PATH,
        state=STATE_DIR,
        accounts_path=ACCOUNTS_PATH,
        rates=RATES_PATH,
        accounts=", ".join(sorted(chosen)),
        protected=", ".join(f"`{path}`" for path in PROTECTED),
        stub_main=_STUB_MAIN,
        stub_init=_STUB_INIT,
    )
    return core.TaskInstance(
        id="N-9",
        category="C",
        title="Касса по ТЗ: пакет команд, Decimal, идемпотентность, атомарная ошибка",
        prompt=prompt,
        max_points=1.0,
        workspace={
            MAIN_FILE: _STUB_MAIN,
            INIT_FILE: _STUB_INIT,
            README_PATH: (
                "# Касса\n\n"
                "Проводки приходят выгрузкой; отчёт читает казначейство.\n"
                "Запуск: `python3 -m ledger post data/batch_2026_10_01.jsonl`, затем `statement`.\n"
            ),
            ACCOUNTS_PATH: _accounts_payload(chosen),
            RATES_PATH: json.dumps(RATES, ensure_ascii=False, indent=2) + "\n",
            VALID_BATCH: _jsonl(rows),
            INVALID_BATCH: invalid_csv,
            MIXED_BATCH: mixed_csv,
        },
        oracle={
            # ``reference`` stays a mapping for direct evaluator tests; the
            # selfcheck provider speaks text, so it gets the printed protocol.
            "reference": {"ledger/__init__.py": _STUB_INIT, "ledger/__main__.py": _REFERENCE_MAIN},
            "final_answer": _render_files(
                {"ledger/__init__.py": _STUB_INIT, "ledger/__main__.py": _REFERENCE_MAIN}
            ),
            "target_file": MAIN_FILE,
            "answer_file": MAIN_FILE,
            "package": PACKAGE,
            "valid_batch": VALID_BATCH,
            "invalid_batch": INVALID_BATCH,
            "mixed_batch": MIXED_BATCH,
            "report_path": REPORT_PATH,
            "state_dir": STATE_DIR,
            "entries": rows,
            "expected_report": _report(rows),
            "expected_mixed_report": _report(rows + fresh_entries),
            "expected_unpost_report": _report(applied_after_unpost),
            "unpost_key": unpost_key,
            "invalid_lines": bad_lines,
            "fresh_entries": len(fresh_entries),
            "protected": list(PROTECTED),
            "allowed_prefix": ALLOWED_PREFIX,
            "allowed_files": list(ALLOWED_FILES),
            "accounts": sorted(chosen),
            "checks": [
                "usage_exit",
                "post_apply",
                "statement_shape",
                "statement_values",
                "idempotent_repost",
                "mixed_batch",
                "unpost_contract",
                "invalid_contract_clean",
                "invalid_atomic_restore",
                "protected_and_scope",
            ],
        },
        meta={"scenario": "N-9", "entries": len(rows), "accounts": len(chosen)},
        fresh_after="2026-10-15",
        languages=["ru"],
    )


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #

_FILE_MARKER = re.compile(r"^### FILE:\s*(\S+)\s*$")
_FENCE = "```"


def _render_files(files: dict[str, str]) -> str:
    """The reference in the same ``### FILE:`` protocol the prompt asks for."""
    return "\n".join(f"### FILE: {rel}\n```python\n{body}```" for rel, body in files.items())


def _parse_answer(text: str) -> dict[str, str] | None:
    """Collect ``### FILE:`` fenced blocks; nothing outside ``ledger/`` is taken."""
    files: dict[str, str] = {}
    current: str | None = None
    collecting = False
    buffer: list[str] = []
    for line in text.splitlines():
        marker = _FILE_MARKER.match(line)
        if marker:
            if collecting and current:
                files[current] = "\n".join(buffer) + "\n"
            collecting = False
            buffer = []
            rel = marker.group(1)
            current = rel if rel.startswith(PACKAGE + "/") and ".." not in rel else None
            continue
        if current is None:
            continue
        if not collecting:
            if line.startswith(_FENCE):
                collecting = True
            continue
        if line.strip() == _FENCE:
            files[current] = "\n".join(buffer) + "\n"
            current = None
            collecting = False
            buffer = []
            continue
        buffer.append(line)
    if collecting and current:
        files[current] = "\n".join(buffer) + "\n"
    return files or None


def _answer_files(task: core.TaskInstance, tr: core.Transcript) -> dict[str, str] | None:
    """The files the agent left on disk, else the files it printed."""
    from .. import agent

    work_dir = tr.extra.get("work_dir")
    if tr.extra.get("harness") == "agent" and work_dir:
        root = Path(str(work_dir)) / PACKAGE
        if root.is_dir():
            found = {
                path.relative_to(Path(str(work_dir))).as_posix(): path.read_text(encoding="utf-8", errors="replace")
                for path in sorted(root.rglob("*.py"))
                if path.is_file()
            }
            if found:
                return found
        return None
    return _parse_answer(agent.answer_text(task, tr, tr.final_text))


def _snapshot(root: Path, rel: str) -> dict[str, bytes]:
    base = root / rel
    if not base.exists():
        return {}
    if base.is_file():
        return {rel: base.read_bytes()}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _diff_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, "-m", PACKAGE, *argv],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, -9, "", "timeout after 60s")


def _prepared(sandbox: Path, task: core.TaskInstance, files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="work-", dir=sandbox))
    for rel, content in task.workspace.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _read(root: Path, rel: str) -> str:
    path = root / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _statement_ok(report: Any) -> bool:
    if not isinstance(report, dict) or list(report) != ["accounts", "rows", "grand_total"]:
        return False
    if not isinstance(report["rows"], int) or isinstance(report["rows"], bool):
        return False
    if not isinstance(report["grand_total"], str) or not MONEY_RE.match(report["grand_total"]):
        return False
    if not isinstance(report["accounts"], list):
        return False
    for item in report["accounts"]:
        if not isinstance(item, dict) or list(item) != ["account", "entries", "eur_total"]:
            return False
        if not isinstance(item["entries"], int) or isinstance(item["entries"], bool):
            return False
        if not isinstance(item["eur_total"], str) or not MONEY_RE.match(item["eur_total"]):
            return False
    codes = [item["account"] for item in report["accounts"]]
    return codes == sorted(codes)


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    oracle = task.oracle
    names: list[str] = oracle["checks"]
    files = _answer_files(task, tr)

    def fail_all(detail: str) -> core.Score:
        for name in names:
            score.add(name, False, _POINTS_PER_CHECK, detail)
        score.notes.append(detail)
        return score

    if not files or INIT_FILE not in files or MAIN_FILE not in files or "def main" not in files[MAIN_FILE]:
        return fail_all("ответ не содержит пакета ledger с __init__.py и файлом __main__.py с main")

    sandbox = Path(tempfile.mkdtemp(prefix="dailybench-n9-"))
    try:
        root = _prepared(sandbox, task, files)
        before = _diff_files(root)
        valid = oracle["valid_batch"]
        applied_rows = oracle["entries"]

        # -- usage ---------------------------------------------------------- #
        empty = _run(root)
        unknown = _run(root, "frobnicate")
        arity = _run(root, "post")
        usage_ok = (
            empty.returncode == 64
            and unknown.returncode == 64
            and arity.returncode == 64
            and "usage:" in empty.stderr
            and "usage:" in unknown.stderr
            and "usage:" in arity.stderr
        )
        score.add("usage_exit", usage_ok, _POINTS_PER_CHECK, f"коды {empty.returncode}/{unknown.returncode}/{arity.returncode}")

        # -- apply ---------------------------------------------------------- #
        posted = _run(root, "post", valid)
        state_after_post = _snapshot(root, oracle["state_dir"])
        stdout_ok = posted.stdout.strip() == f"ok: applied={len(applied_rows)} skipped=0"
        apply_ok = posted.returncode == 0 and stdout_ok and bool(state_after_post)
        score.add(
            "post_apply",
            apply_ok,
            _POINTS_PER_CHECK,
            f"код {posted.returncode}, stdout {posted.stdout.strip()!r}, файлов состояния {len(state_after_post)}",
        )

        # -- statement ------------------------------------------------------ #
        state_before_statement = _snapshot(root, oracle["state_dir"])
        stated = _run(root, "statement")
        state_after_statement = _snapshot(root, oracle["state_dir"])
        raw_report = _read(root, REPORT_PATH)
        try:
            report = json.loads(raw_report)
        except json.JSONDecodeError:
            report = None
        shape_ok = (
            stated.returncode == 0
            and _statement_ok(report)
            and state_after_statement == state_before_statement
        )
        score.add(
            "statement_shape",
            shape_ok,
            _POINTS_PER_CHECK,
            f"код {stated.returncode}, состояние {'не тронуто' if state_after_statement == state_before_statement else 'изменено'}; "
            + ("строгая форма отчёта" if shape_ok else f"отчёт: {raw_report[:200]!r}"),
        )
        expected = oracle["expected_report"]
        values_ok = report == expected
        score.add(
            "statement_values",
            values_ok,
            _POINTS_PER_CHECK,
            "суммы и округление верны" if values_ok else f"получено {report}, ожидалось {expected}",
        )

        # -- idempotency ---------------------------------------------------- #
        report_before = _read(root, REPORT_PATH)
        state_before = _snapshot(root, oracle["state_dir"])
        again = _run(root, "post", valid)
        idempotent_ok = (
            again.returncode == 0
            and again.stdout.strip() == f"ok: applied=0 skipped={len(applied_rows)}"
            and _snapshot(root, oracle["state_dir"]) == state_before
            and _read(root, REPORT_PATH) == report_before
        )
        score.add(
            "idempotent_repost",
            idempotent_ok,
            _POINTS_PER_CHECK,
            f"код {again.returncode}, stdout {again.stdout.strip()!r}, состояние "
            f"{'то же' if _snapshot(root, oracle['state_dir']) == state_before else 'изменилось'}",
        )

        # -- mixed batch ---------------------------------------------------- #
        mixed = _run(root, "post", oracle["mixed_batch"])
        mixed_state = _snapshot(root, oracle["state_dir"])
        restated = _run(root, "statement")
        try:
            mixed_report = json.loads(_read(root, REPORT_PATH))
        except json.JSONDecodeError:
            mixed_report = None
        mixed_ok = (
            mixed.returncode == 0
            and mixed.stdout.strip() == f"ok: applied={oracle['fresh_entries']} skipped=2"
            and mixed_state != state_before
            and restated.returncode == 0
            and mixed_report == oracle["expected_mixed_report"]
        )
        score.add(
            "mixed_batch",
            mixed_ok,
            _POINTS_PER_CHECK,
            f"код {mixed.returncode}, stdout {mixed.stdout.strip()!r}",
        )

        # -- unpost --------------------------------------------------------- #
        removed = _run(root, "unpost", oracle["unpost_key"])
        _run(root, "statement")
        after_remove = _read(root, REPORT_PATH)
        try:
            removed_report = json.loads(after_remove)
        except json.JSONDecodeError:
            removed_report = None
        unpost_state = _snapshot(root, oracle["state_dir"])
        missing = _run(root, "unpost", oracle["unpost_key"])
        stable = _snapshot(root, oracle["state_dir"]) == unpost_state
        # The key leaves the state, so posting the batch again applies it afresh.
        reposted = _run(root, "post", valid)
        unpost_ok = (
            removed.returncode == 0
            and removed_report == oracle["expected_unpost_report"]
            and missing.returncode == 3
            and (missing.stderr or "").splitlines()[:1] == [f"missing key: {oracle['unpost_key']}"]
            and stable
            and reposted.returncode == 0
            and reposted.stdout.strip() == f"ok: applied=1 skipped={len(applied_rows) - 1}"
        )
        score.add(
            "unpost_contract",
            unpost_ok,
            _POINTS_PER_CHECK,
            f"коды {removed.returncode}/{missing.returncode}/{reposted.returncode}, "
            f"отчёт {'верен' if removed_report == oracle['expected_unpost_report'] else 'неверен'}, "
            f"повтор после удаления {reposted.stdout.strip()!r}",
        )

        # -- invalid batch on a clean copy ---------------------------------- #
        clean = _prepared(sandbox, task, files)
        clean_run = _run(clean, "post", oracle["invalid_batch"])
        expected_line = "invalid batch: " + ",".join(str(number) for number in oracle["invalid_lines"])
        absent = _run(clean, "post", "data/does-not-exist.jsonl")
        clean_ok = (
            clean_run.returncode == 2
            and (clean_run.stderr or "").splitlines()[:1] == [expected_line]
            and absent.returncode == 2
            and (absent.stderr or "").splitlines()[:1] == ["invalid batch: missing data/does-not-exist.jsonl"]
            and not _snapshot(clean, oracle["state_dir"])
            and not (clean / REPORT_PATH).exists()
        )
        score.add(
            "invalid_contract_clean",
            clean_ok,
            _POINTS_PER_CHECK,
            f"код {clean_run.returncode}/{absent.returncode}, stderr {(clean_run.stderr or '').splitlines()[:1]}, "
            f"состояние {bool(_snapshot(clean, oracle['state_dir']))}, отчёт {(clean / REPORT_PATH).exists()}",
        )

        # -- atomicity over the existing state ------------------------------ #
        state_before_bad = _snapshot(root, oracle["state_dir"])
        report_before_bad = _read(root, REPORT_PATH)
        broken = _run(root, "post", oracle["invalid_batch"])
        atomic_ok = (
            broken.returncode == 2
            and _snapshot(root, oracle["state_dir"]) == state_before_bad
            and _read(root, REPORT_PATH) == report_before_bad
        )
        score.add(
            "invalid_atomic_restore",
            atomic_ok,
            _POINTS_PER_CHECK,
            f"код {broken.returncode}; состояние {'цело' if _snapshot(root, oracle['state_dir']) == state_before_bad else 'изменено'}, "
            f"отчёт {'цел' if _read(root, REPORT_PATH) == report_before_bad else 'изменён'}",
        )

        # -- protected files and write scope -------------------------------- #
        after = _diff_files(root)
        changed = sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        )
        touched_protected = [path for path in changed if path in oracle["protected"]]
        outside = [
            path
            for path in changed
            if not path.startswith(oracle["allowed_prefix"])
            and path not in oracle["allowed_files"]
            and not path.startswith(PACKAGE + "/")
        ]
        scope_ok = not touched_protected and not outside
        score.add(
            "protected_and_scope",
            scope_ok,
            _POINTS_PER_CHECK,
            f"изменённые защищённые: {touched_protected}; вне разрешённых путей: {outside}",
        )
        return score
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
