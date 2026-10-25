"""Scenario N-2 — terminal/data workflow (stretch only).

The candidate returns ``pipeline.py``; the trusted side runs it in a throwaway
copy of the worktree and observes everything it did:

* the migrated SQLite schema and the imported rows,
* the JSON report built from the database,
* the complete filesystem diff (only ``build/`` and ``reports/summary.json``
  may change, the protected files must stay byte-identical),
* the exit codes and the declared error line numbers of an invalid batch,
* atomicity — an invalid batch leaves both the database and the report exactly
  as they were, and never materialises half a load on a clean copy.

The oracle is independent of the script: rows are generated structurally, the
expected report is recomputed in pure Python from those rows, and the schema
version list comes from the migration fixtures. A candidate's own ``PASS``
line is never consulted.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .. import core
from .coding_python import extract_code

#: Ten checks of 0.1 each; the profile distributes the category total.
_POINTS_PER_CHECK = 0.1

ANSWER_FILE = "pipeline.py"
DB_PATH = "build/warehouse.db"
REPORT_PATH = "reports/summary.json"
REGIONS_PATH = "config/regions.txt"
PRICES_PATH = "data/protected/prices.csv"
VALID_BATCH = "data/batch_2026_10_01.csv"
INVALID_BATCH = "data/batch_2026_10_02.csv"

MIGRATION_FILES = (
    "001_init.sql",
    "002_add_fields.sql",
    "003_batch_index.sql",
)

#: Byte-identical after every command, including the failing ones.
PROTECTED = (
    "README.md",
    "config/regions.txt",
    "data/protected/prices.csv",
    "migrations/001_init.sql",
    "migrations/002_add_fields.sql",
    "migrations/003_batch_index.sql",
    ANSWER_FILE,
)

#: The only paths a run may create or rewrite.
ALLOWED_PREFIX = "build/"
ALLOWED_FILES = (REPORT_PATH,)

#: Versions recorded by the shipped migrations, in application order.
EXPECTED_VERSIONS = [1, 2, 3]

STATUSES = ("paid", "refunded")
HEADER = "order_id,region,product,cents,status"

_MIGRATION_001 = """CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL UNIQUE,
    product TEXT NOT NULL,
    cents INTEGER NOT NULL,
    status TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '',
    batch TEXT NOT NULL DEFAULT ''
);
"""

_MIGRATION_002 = """ALTER TABLE orders ADD COLUMN currency TEXT NOT NULL DEFAULT 'EUR';
CREATE INDEX idx_orders_region ON orders(region);
"""

_MIGRATION_003 = """CREATE INDEX idx_orders_batch ON orders(batch);
"""

_STUB = '''#!/usr/bin/env python3
"""Warehouse pipeline: CSV -> SQLite -> report. Implement the commands below."""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    raise NotImplementedError("pipeline is not implemented")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''

_REFERENCE = r'''#!/usr/bin/env python3
"""Warehouse pipeline: CSV -> SQLite -> report."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

BUILD = Path("build")
DB = BUILD / "warehouse.db"
REPORTS = Path("reports")
SUMMARY = REPORTS / "summary.json"
REGIONS = Path("config/regions.txt")
MIGRATIONS = Path("migrations")
HEADER = "order_id,region,product,cents,status"
STATUSES = {"paid", "refunded"}
USAGE = "usage: pipeline.py migrate|import <csv>|report <csv>"
APPLIED_AT = "2026-10-01T00:00:00Z"


def usage(message: str) -> int:
    print(USAGE, file=sys.stderr)
    if message:
        print(message, file=sys.stderr)
    return 64


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def read_regions() -> set[str]:
    return {line.strip() for line in REGIONS.read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_batch(path: str) -> tuple[list[tuple[int, str, str, int, str]], list[int], bool]:
    """Return (rows, bad line numbers, header_ok). Line 1 is the header."""
    try:
        blob = Path(path).read_text(encoding="utf-8")
    except OSError:
        return [], [], False
    lines = blob.splitlines()
    if not lines or lines[0] != HEADER:
        return [], [1], False
    regions = read_regions()
    rows: list[tuple[int, str, str, int, str]] = []
    bad: list[int] = []
    seen: set[int] = set()
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        if len(fields) != 5:
            bad.append(number)
            continue
        raw_id, region, product, raw_cents, status = fields
        order_id = int(raw_id) if re.fullmatch(r"[0-9]+", raw_id) else 0
        cents = int(raw_cents) if re.fullmatch(r"[0-9]+", raw_cents) else 0
        row_bad = (
            order_id <= 0
            or region not in regions
            or not product
            or cents <= 0
            or status not in STATUSES
            or order_id in seen
        )
        if row_bad:
            bad.append(number)
            continue
        seen.add(order_id)
        rows.append((order_id, region, product, cents, status))
    return rows, bad, True


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def pending(connection: sqlite3.Connection) -> list[tuple[int, str]]:
    connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    have = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
    found: list[tuple[int, str]] = []
    for name in sorted(MIGRATIONS.glob("*.sql")):
        match = re.match(r"([0-9]+)_", name.name)
        if match and int(match.group(1)) not in have:
            found.append((int(match.group(1)), name.read_text(encoding="utf-8")))
    return found


def migrate(connection: sqlite3.Connection) -> None:
    for version, sql in pending(connection):
        for statement in sql.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, APPLIED_AT),
        )


def batch_name(path: str) -> str:
    return Path(path).name


def command_migrate() -> int:
    BUILD.mkdir(exist_ok=True)
    connection = connect(DB)
    try:
        with connection:
            migrate(connection)
    finally:
        connection.close()
    return 0


def command_import(path: str) -> int:
    if not Path(path).is_file():
        return fail(f"invalid batch: missing {path}")
    rows, bad, header_ok = parse_batch(path)
    if bad:
        if not header_ok:
            return fail("invalid batch: 1")
        return fail("invalid batch: " + ",".join(str(number) for number in bad))

    BUILD.mkdir(exist_ok=True)
    fresh = not DB.exists()
    target = DB
    if fresh:
        target = BUILD / ".warehouse.db.tmp"
    name = batch_name(path)
    connection = connect(target)
    try:
        with connection:
            migrate(connection)
            connection.executemany(
                "INSERT INTO orders (order_id, region, product, cents, status, batch)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(order_id) DO UPDATE SET"
                " region = excluded.region, product = excluded.product,"
                " cents = excluded.cents, status = excluded.status, batch = excluded.batch",
                [(order_id, region, product, cents, status, name) for order_id, region, product, cents, status in rows],
            )
    except BaseException:
        connection.close()
        if fresh and target.exists():
            target.unlink()
        raise
    connection.close()
    if fresh:
        os.replace(target, DB)
    print(f"ok: rows={len(rows)}")
    return 0


def command_report(path: str) -> int:
    if not DB.is_file():
        return fail("report: database is missing")
    name = batch_name(path)
    connection = connect(DB)
    try:
        rows = connection.execute(
            "SELECT region, product, cents FROM orders WHERE batch = ?", (name,)
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        return fail(f"report: no rows for batch {name}")

    by_region: dict[str, list[int]] = {}
    by_product: dict[str, int] = {}
    for region, product, cents in rows:
        bucket = by_region.setdefault(region, [0, 0])
        bucket[0] += 1
        bucket[1] += cents
        by_product[product] = by_product.get(product, 0) + cents

    payload = {
        "batch": name,
        "rows": len(rows),
        "total_cents": sum(cents for _, _, cents in rows),
        "by_region": [
            {"region": region, "rows": by_region[region][0], "cents": by_region[region][1]}
            for region in sorted(by_region)
        ],
        "top_product": min(by_product, key=lambda product: (-by_product[product], product)),
    }

    REPORTS.mkdir(exist_ok=True)
    tmp = REPORTS / ".summary.json.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, SUMMARY)
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        return usage("no command")
    command, *rest = argv
    if command == "migrate" and not rest:
        return command_migrate()
    if command in ("import", "report") and len(rest) == 1:
        return command_import(rest[0]) if command == "import" else command_report(rest[0])
    return usage(f"unknown command: {' '.join(argv)}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''

_PROMPT = """Рабочая копия — небольшой склад. Нужен исполняемый файл {answer} (Python 3.12, только стандартная библиотека), который реализует три команды.

Запуск: `python3 {answer} <команда> [аргумент]` в корне рабочей копии.

Команды и коды возврата:

1. `{answer} migrate` — применить все ещё не применённые файлы `migrations/*.sql` в лексикографическом порядке номеров. Версия миграции — целое из префикса имени. Каждая применённая версия записывается в `schema_migrations(version, applied_at)` со значением `applied_at = {applied}`. Повторный запуск ничего не применяет. Успех — код 0.
2. `{answer} import <csv>` — применить ожидающие миграции и импортировать batch в таблицу `orders` одной транзакцией. Повторный импорт того же batch не удваивает ни строки, ни суммы. Успех — код 0 и ровно одна строка `ok: rows=<N>` в stdout.
3. `{answer} report <csv>` — построить отчёт по строкам своего batch из БД и записать `{report}`. Успех — код 0.

База — `{db}`. Вне `build/` и `{report}` ни один путь не должен появиться, исчезнуть или измениться. Файлы из списка защищённых изменять запрещено: {protected}.

Формат batch (UTF-8, RFC 4180 без кавычек, разделитель — запятая): первая строка ровно `{header}`, далее строки `order_id,region,product,cents,status`.

Валидация строки: `order_id` — целое > 0, уникальное внутри файла; `region` — одна из строк `{regions}`; `product` — непустая строка; `cents` — целое > 0 без знака; `status` — `paid` или `refunded`. Номер строки считается с 1, заголовок — строка 1.

Любая невалидная строка (а также неверный заголовок, отсутствующий или нечитаемый файл) — код 2. Сообщение — первая строка stderr: `invalid batch: <номера всех невалидных строк по возрастанию через запятую>`; для неверного заголовка — `invalid batch: 1`; для отсутствующего файла — `invalid batch: missing <путь>`. Атомарность обязательна: при ошибке на чистой копии не должно появиться ни БД, ни отчёта; при ошибке поверх готового состояния файлы `{db}` и `{report}` должны остаться байт в байт прежними. Неизвестная команда или неверное число аргументов — код 64 и строка `usage: ...` в stderr.

Миграции (применяются ровно в этом порядке):

`{m1}` — {sql1}

`{m2}` — {sql2}

`{m3}` — {sql3}

Отчёт — строгий JSON без Markdown и без лишнего текста, ключи ровно в этом порядке:
`{{"batch":"<имя файла batch>","rows":<число строк batch>,"total_cents":<сумма cents>,"by_region":[{{"region":"<регион>","rows":<число>,"cents":<сумма>}}],"top_product":"<товар>"}}`
`by_region` отсортирован по `region` по возрастанию и содержит только регионы этого batch. `top_product` — товар с наибольшей суммой `cents` в этом batch, при равенстве — лексикографически меньший.

Верни ТОЛЬКО полный текст файла {answer} в одном блоке кода. Файл {answer} в рабочей копии:

```python
{stub}```"""


def _report(rows: list[tuple[int, str, str, int, str]], name: str) -> dict[str, Any]:
    by_region: dict[str, list[int]] = {}
    by_product: dict[str, int] = {}
    for _, region, product, cents, _status in rows:
        bucket = by_region.setdefault(region, [0, 0])
        bucket[0] += 1
        bucket[1] += cents
        by_product[product] = by_product.get(product, 0) + cents
    return {
        "batch": name,
        "rows": len(rows),
        "total_cents": sum(row[3] for row in rows),
        "by_region": [
            {"region": region, "rows": by_region[region][0], "cents": by_region[region][1]}
            for region in sorted(by_region)
        ],
        "top_product": (
            min(by_product, key=lambda product: (-by_product[product], product)) if by_product else ""
        ),
    }


def _rows(rng: random.Random, regions: list[str], products: list[str]) -> list[tuple[int, str, str, int, str]]:
    """A valid batch with a constructed top-product tie and bounded filler sums."""
    rows: list[tuple[int, str, str, int, str]] = []
    ids = rng.sample(range(1000, 9999), 18)
    first, second = products[0], products[1]
    tie_a = rng.randrange(500, 800)
    filler_products = products[2:]
    # Fillers never repeat a product more than twice, so their sums stay below
    # the tie pair and can never steal ``top_product`` on any seed.
    for index in range(10):
        product = filler_products[index % len(filler_products)]
        rows.append((
            ids[index],
            rng.choice(regions),
            product,
            rng.randrange(10, 100),
            "refunded" if index % 4 == 3 else "paid",
        ))
    rows.append((ids[10], regions[0], first, tie_a, "paid"))
    rows.append((ids[11], regions[-1], second, tie_a, "paid"))
    for index in range(12, 16):
        rows.append((ids[index], rng.choice(regions), rng.choice(filler_products), rng.randrange(10, 100), "paid"))
    rng.shuffle(rows)
    return rows


def _csv(rows: list[tuple[int, str, str, int, str]]) -> str:
    body = "\n".join(",".join(str(field) for field in row) for row in rows)
    return HEADER + "\n" + body + "\n"


def _invalid_batch(rows: list[tuple[int, str, str, int, str]]) -> tuple[str, list[int]]:
    """Corrupt four rows; returns the CSV and the expected 1-based line numbers."""
    lines = [[str(field) for field in row] for row in rows]
    bad: list[int] = []
    # Physical line = index + 2 (the header is line 1).
    lines[2][1] = "moon-base"
    bad.append(4)
    lines[5][3] = "12x"
    bad.append(7)
    lines[6][0] = lines[0][0]
    bad.append(8)
    lines[9][4] = "void"
    bad.append(11)
    body = "\n".join(",".join(line) for line in lines)
    return HEADER + "\n" + body + "\n", bad


def _workspace(valid: str, invalid: str, regions: list[str], products: list[str]) -> dict[str, str]:
    prices = "product,cents\n" + "\n".join(f"{product},500" for product in products) + "\n"
    return {
        ANSWER_FILE: _STUB,
        "README.md": (
            "# Склад\n\n"
            "Batch приходят из выгрузки; отчёты читает финансовая сводка.\n"
            "Запуск: `python3 pipeline.py migrate`, затем `import`, затем `report`.\n"
        ),
        REGIONS_PATH: "\n".join(regions) + "\n",
        PRICES_PATH: prices,
        VALID_BATCH: valid,
        INVALID_BATCH: invalid,
        f"migrations/{MIGRATION_FILES[0]}": _MIGRATION_001,
        f"migrations/{MIGRATION_FILES[1]}": _MIGRATION_002,
        f"migrations/{MIGRATION_FILES[2]}": _MIGRATION_003,
    }


def generate(rng: random.Random, cfg: Any) -> core.TaskInstance:
    regions = rng.sample(
        ["eu-west", "eu-central", "us-east", "ap-south", "sa-east"], rng.randrange(3, 5)
    )
    products = [
        "alpha",
        "beta",
        f"gamma-{rng.randrange(10, 99)}",
        f"delta-{rng.randrange(10, 99)}",
        f"omega-{rng.randrange(10, 99)}",
        f"sigma-{rng.randrange(10, 99)}",
    ]
    rows = _rows(rng, regions, products)
    invalid_csv, bad_lines = _invalid_batch(rows)
    rows.sort()
    report = _report(rows, Path(VALID_BATCH).name)

    prompt = _PROMPT.format(
        answer=ANSWER_FILE,
        db=DB_PATH,
        report=REPORT_PATH,
        protected=", ".join(PROTECTED),
        header=HEADER,
        regions=", ".join(regions),
        applied="2026-10-01T00:00:00Z",
        m1=MIGRATION_FILES[0],
        m2=MIGRATION_FILES[1],
        m3=MIGRATION_FILES[2],
        sql1=_MIGRATION_001.strip(),
        sql2=_MIGRATION_002.strip(),
        sql3=_MIGRATION_003.strip(),
        stub=_STUB,
    )
    return core.TaskInstance(
        id="N-2",
        category="C",
        title="Складской конвейер: CSV → SQLite → отчёт",
        prompt=prompt,
        max_points=1.0,
        workspace=_workspace(_csv(rows), invalid_csv, regions, products),
        oracle={
            "reference": _REFERENCE,
            "target_file": ANSWER_FILE,
            "answer_file": ANSWER_FILE,
            "valid_batch": VALID_BATCH,
            "invalid_batch": INVALID_BATCH,
            "db_path": DB_PATH,
            "report_path": REPORT_PATH,
            "expected_rows": [list(row) for row in rows],
            "expected_report": report,
            "invalid_lines": bad_lines,
            "versions": list(EXPECTED_VERSIONS),
            "protected": list(PROTECTED),
            "allowed_prefix": ALLOWED_PREFIX,
            "allowed_files": list(ALLOWED_FILES),
            "columns": {
                "order_id": "INTEGER",
                "product": "TEXT",
                "cents": "INTEGER",
                "status": "TEXT",
                "region": "TEXT",
                "batch": "TEXT",
            },
            "indexes": ["idx_orders_region", "idx_orders_batch"],
            "checks": [
                "usage_exit",
                "migrate_and_schema",
                "db_rows",
                "report_schema",
                "report_content",
                "report_from_db",
                "idempotent_reimport",
                "invalid_contract",
                "invalid_atomicity",
                "protected_and_scope",
            ],
        },
        meta={"scenario": "N-2", "rows": len(rows), "regions": len(regions)},
        fresh_after="2026-10-01",
        languages=["ru"],
    )


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, ANSWER_FILE, *argv],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, -9, "", "timeout after 60s")


def _db_rows(root: Path, batch: str) -> list[list[Any]] | None:
    path = root / DB_PATH
    if not path.is_file():
        return None
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT order_id, region, product, cents, status FROM orders WHERE batch = ? ORDER BY order_id",
            (batch,),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    return [list(row) for row in rows]


def _schema(root: Path) -> tuple[dict[str, str], set[str], list[int]] | None:
    path = root / DB_PATH
    if not path.is_file():
        return None
    connection = sqlite3.connect(path)
    try:
        columns = {
            row[1]: str(row[2]).upper()
            for row in connection.execute("PRAGMA table_info(orders)").fetchall()
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'orders'"
            ).fetchall()
        }
        versions = [row[0] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    return columns, indexes, versions


def _prepared(sandbox: Path, task: core.TaskInstance, code: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix="work-", dir=sandbox))
    for rel, content in task.workspace.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (root / ANSWER_FILE).write_text(code, encoding="utf-8")
    return root


def _protected_ok(root: Path, before: dict[str, str], listed: list[str]) -> tuple[bool, str]:
    changed = [rel for rel in listed if before.get(rel) != _read(root, rel)]
    return not changed, f"изменены защищённые файлы: {changed}" if changed else "защищённые файлы сохранены"


def _read(root: Path, rel: str) -> str | None:
    path = root / rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _blob(root: Path, rel: str) -> bytes | None:
    path = root / rel
    if not path.is_file():
        return None
    return path.read_bytes()


def _scope_ok(root: Path, before: dict[str, str], prefix: str, allowed: list[str]) -> tuple[bool, str]:
    after = _snapshot(root)
    touched = sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path) and path != ANSWER_FILE
    )
    outside = [
        path for path in touched if not path.startswith(prefix) and path not in allowed
    ]
    return not outside, f"изменено вне разрешённых путей: {outside}" if outside else "diff только в разрешённых путях"


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    from .. import agent

    score = core.Score(task_id=task.id, category="C")
    oracle = task.oracle
    names: list[str] = oracle["checks"]
    code = extract_code(agent.answer_text(task, tr, tr.final_text))

    def fail_all(detail: str) -> core.Score:
        for name in names:
            score.add(name, False, _POINTS_PER_CHECK, detail)
        score.notes.append(detail)
        return score

    if not code or "def main" not in code:
        return fail_all("ответ не содержит файла pipeline.py с функцией main")

    sandbox = Path(tempfile.mkdtemp(prefix="dailybench-n2-"))
    try:
        root = _prepared(sandbox, task, code)
        before = _snapshot(root)
        batch = Path(oracle["valid_batch"]).name

        # -- usage ---------------------------------------------------------- #
        empty = _run(root)
        unknown = _run(root, "frobnicate")
        usage_ok = (
            empty.returncode == 64
            and unknown.returncode == 64
            and "usage:" in empty.stderr
            and "usage:" in unknown.stderr
        )
        score.add("usage_exit", usage_ok, _POINTS_PER_CHECK, f"коды {empty.returncode}/{unknown.returncode}")

        # -- migrate + schema ---------------------------------------------- #
        migrated = _run(root, "migrate")
        schema = _schema(root)
        columns_ok = schema is not None and all(
            schema[0].get(name) == kind for name, kind in oracle["columns"].items()
        )
        indexes_ok = schema is not None and set(oracle["indexes"]) <= schema[1]
        versions_ok = schema is not None and schema[2] == oracle["versions"]
        schema_ok = migrated.returncode == 0 and columns_ok and indexes_ok and versions_ok
        score.add(
            "migrate_and_schema",
            schema_ok,
            _POINTS_PER_CHECK,
            f"код migrate {migrated.returncode}; колонки {schema[0] if schema else None}; "
            f"версии {schema[2] if schema else None}",
        )

        # -- happy import --------------------------------------------------- #
        imported = _run(root, "import", oracle["valid_batch"])
        rows = _db_rows(root, batch)
        want_rows = oracle["expected_rows"]
        rows_ok = rows == want_rows
        score.add(
            "db_rows",
            rows_ok,
            _POINTS_PER_CHECK,
            f"код {imported.returncode}, строк {len(rows) if rows is not None else 'нет'}, ожидалось {len(want_rows)}",
        )

        # -- report --------------------------------------------------------- #
        reported = _run(root, "report", oracle["valid_batch"])
        raw_report = _read(root, REPORT_PATH) or ""
        try:
            report = json.loads(raw_report)
        except json.JSONDecodeError:
            report = None

        expected = oracle["expected_report"]
        schema_shape = (
            isinstance(report, dict)
            and list(report) == list(expected)
            and isinstance(report.get("rows"), int)
            and not isinstance(report.get("rows"), bool)
            and isinstance(report.get("total_cents"), int)
            and isinstance(report.get("by_region"), list)
            and all(
                isinstance(item, dict)
                and list(item) == ["region", "rows", "cents"]
                and isinstance(item.get("rows"), int)
                and isinstance(item.get("cents"), int)
                for item in report.get("by_region", [])
            )
            and [item["region"] for item in report.get("by_region", [])]
            == sorted(item["region"] for item in report.get("by_region", []))
        )
        score.add("report_schema", schema_shape, _POINTS_PER_CHECK, f"код report {reported.returncode}; " + ("строгая форма отчёта" if schema_shape else f"отчёт: {raw_report[:160]!r}"))

        content_ok = isinstance(report, dict) and report == expected
        score.add("report_content", content_ok, _POINTS_PER_CHECK, "содержимое отчёта верно" if content_ok else f"получено {report}, ожидалось {expected}")

        from_db: dict[str, Any] | None = None
        if rows is not None:
            from_db = _report([tuple(row) for row in rows], batch)
        score.add(
            "report_from_db",
            from_db is not None and report == from_db,
            _POINTS_PER_CHECK,
            "отчёт совпадает с содержимым БД" if isinstance(report, dict) and report == from_db else "отчёт не выводится из БД",
        )

        # -- idempotency ---------------------------------------------------- #
        report_before_reimport = _read(root, REPORT_PATH)
        again = _run(root, "import", oracle["valid_batch"])
        rows_after = _db_rows(root, batch)
        idempotent_ok = (
            again.returncode == 0
            and again.stdout.strip() == f"ok: rows={len(want_rows)}"
            and rows_after == want_rows
            and _read(root, REPORT_PATH) == report_before_reimport
        )
        score.add(
            "idempotent_reimport",
            idempotent_ok,
            _POINTS_PER_CHECK,
            f"код {again.returncode}, строк {len(rows_after) if rows_after is not None else 'нет'}, ожидалось {len(want_rows)}",
        )

        # -- invalid batch on a clean copy ---------------------------------- #
        clean = _prepared(sandbox, task, code)
        clean_run = _run(clean, "import", oracle["invalid_batch"])
        bad_line = (clean_run.stderr or "").splitlines()[:1]
        expected_line = "invalid batch: " + ",".join(str(n) for n in oracle["invalid_lines"])
        invalid_ok = (
            clean_run.returncode == 2
            and bad_line == [expected_line]
            and not (clean / DB_PATH).exists()
            and not (clean / REPORT_PATH).exists()
        )
        score.add(
            "invalid_contract",
            invalid_ok,
            _POINTS_PER_CHECK,
            f"код {clean_run.returncode}, stderr {bad_line}, файлы БД/отчёта "
            f"{(clean / DB_PATH).exists()}/{(clean / REPORT_PATH).exists()}",
        )

        # -- atomicity over the existing state ------------------------------ #
        db_before = _blob(root, DB_PATH)
        report_before = _blob(root, REPORT_PATH)
        broken = _run(root, "import", oracle["invalid_batch"])
        atomic_ok = (
            broken.returncode == 2
            and _blob(root, DB_PATH) == db_before
            and _blob(root, REPORT_PATH) == report_before
        )
        score.add(
            "invalid_atomicity",
            atomic_ok,
            _POINTS_PER_CHECK,
            f"код {broken.returncode}; БД {'цела' if _blob(root, DB_PATH) == db_before else 'изменилась'}, "
            f"отчёт {'цел' if _blob(root, REPORT_PATH) == report_before else 'изменён'}",
        )

        protected_ok, protected_detail = _protected_ok(root, before, oracle["protected"])
        scope_ok, scope_detail = _scope_ok(root, before, oracle["allowed_prefix"], oracle["allowed_files"])
        score.add(
            "protected_and_scope",
            protected_ok and scope_ok,
            _POINTS_PER_CHECK,
            f"{protected_detail}; {scope_detail}",
        )
        return score
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
