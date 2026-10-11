"""Scenario C-3 — SQL aggregation (5 generated databases × 1 point).

Five databases are generated from the seed and the model's single SELECT is
executed against each of them; one point per exact result set. The expected
values come from a pure-Python oracle over the generated rows, never from a
second SQL query, so a query that merely agrees with itself cannot pass.

Backend: SQLite in-process by default (zero dependencies, fully reproducible).
PostgreSQL is supported through a Docker container when ``cfg.sql_backend`` is
``"postgres"``; if no backend is reachable the score is reported as an
infrastructure error instead of a silent zero.
"""

from __future__ import annotations

import random
import re
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .. import core, text


#: Points per hidden check; three scenarios share the constant so a check
#: weighs the same wherever it is earned.
_POINTS_PER_CHECK = 0.8
from .coding_python import code_of, extract_code

DATABASES = 5
YEAR = 2026

#: The file an agent leaves the statement in; the working copy also carries the
#: schema and a loadable seed so the query can be written and run for real.
ANSWER_FILE = "answer.sql"

FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "replace into",
)

SCHEMA_SQLITE = """
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    amount_cents INTEGER NOT NULL
);
"""

_REFERENCE_SQLITE = """SELECT strftime('%Y-%m-01', o.created_at) AS month,
       SUM(MAX(o.amount_cents - COALESCE(r.total, 0), 0)) AS net_revenue_cents
FROM orders AS o
LEFT JOIN (
    SELECT order_id, SUM(amount_cents) AS total
    FROM refunds
    GROUP BY order_id
) AS r ON r.order_id = o.id
WHERE o.status = 'paid'
  AND o.created_at >= '2026-01-01'
  AND o.created_at < '2027-01-01'
GROUP BY month
ORDER BY month;"""

_REFERENCE_POSTGRES = """SELECT date_trunc('month', o.created_at)::date AS month,
       SUM(GREATEST(o.amount_cents - COALESCE(r.total, 0), 0)) AS net_revenue_cents
FROM orders AS o
LEFT JOIN (
    SELECT order_id, SUM(amount_cents) AS total
    FROM refunds
    GROUP BY order_id
) AS r ON r.order_id = o.id
WHERE o.status = 'paid'
  AND o.created_at >= '2026-01-01'
  AND o.created_at < '2027-01-01'
GROUP BY 1
ORDER BY 1;"""

_PROMPT = """Напиши один __DIALECT__ SELECT.

Для каждого календарного месяца 2026 года, в котором был хотя бы один
успешно оплаченный заказ, вычисли net_revenue_cents.

Учитываются только orders.status = 'paid'.

Для каждого заказа:

net =
order.amount_cents
-
SUM(refunds.amount_cents для этого order)

Если refunds нет, вычитается 0.

Если сумма refund превышает amount заказа,
вклад такого заказа считается 0, а не отрицательным.

Неоплаченные orders полностью игнорируются.

Результат:

month | net_revenue_cents

month должен быть первым днём месяца типа date.

Сортировка по month по возрастанию.

Схема:

```sql
users(
    id INTEGER PRIMARY KEY,
    name TEXT
);

orders(
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    created_at TIMESTAMP,
    amount_cents INTEGER,
    status TEXT
);

refunds(
    id INTEGER PRIMARY KEY,
    order_id INTEGER,
    amount_cents INTEGER
);
```

Верни только SQL без Markdown.
"""

_NAMES = ("nora", "ivan", "mei", "alex", "zoe", "oleg", "kira", "pavel", "lena", "timur", "vera", "sasha")


# --------------------------------------------------------------------------- #
# Data generation and the independent oracle
# --------------------------------------------------------------------------- #


def _month_stamp(rng: random.Random, year: int, month: int) -> str:
    day = rng.randrange(1, 29)
    hour = rng.randrange(0, 24)
    minute = rng.randrange(0, 60)
    return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00"


def _database(rng: random.Random) -> dict[str, list[tuple]]:
    """One generated database with every boundary case from the specification."""
    users = [(index + 1, _NAMES[index % len(_NAMES)]) for index in range(rng.randrange(4, 8))]
    user_ids = [uid for uid, _ in users]
    orders: list[tuple] = []
    refunds: list[tuple] = []
    order_id = 0
    refund_id = 0

    busy_months = rng.sample(range(1, 13), rng.randrange(4, 8))
    empty_month = next(month for month in range(1, 13) if month not in busy_months)
    assert empty_month not in busy_months

    def add_order(year: int, month: int, amount: int, status: str) -> int:
        nonlocal order_id
        order_id += 1
        orders.append((order_id, rng.choice(user_ids), _month_stamp(rng, year, month), amount, status))
        return order_id

    for month in busy_months:
        for _ in range(rng.randrange(2, 4)):
            oid = add_order(YEAR, month, rng.randrange(500, 90_000), "paid")
            style = rng.random()
            if style < 0.35:
                refunds.append((refund_id := refund_id + 1, oid, rng.randrange(100, 400)))
                refunds.append((refund_id := refund_id + 1, oid, rng.randrange(100, 400)))
            elif style < 0.6:
                refunds.append((refund_id := refund_id + 1, oid, 10 ** 6))
            # else: no refunds at all
        # an unpaid order with a huge amount and its refunds must be ignored
        oid = add_order(YEAR, month, rng.randrange(500_000, 900_000), rng.choice(("pending", "cancelled", "refunded")))
        refunds.append((refund_id := refund_id + 1, oid, rng.randrange(1000, 5000)))

    for month in rng.sample(range(1, 13), 2):
        add_order(YEAR - 1, month, rng.randrange(1000, 90_000), "paid")
    for month in rng.sample(range(1, 13), 2):
        add_order(YEAR + 1, month, rng.randrange(1000, 90_000), "paid")

    # a paid order whose refunds exceed the amount (clamped to zero)
    clamped = add_order(YEAR, busy_months[0], 300, "paid")
    refunds.append((refund_id := refund_id + 1, clamped, 900))
    # a paid order with a large single refund still leaving a positive net
    partial = add_order(YEAR, busy_months[-1], 50_000, "paid")
    refunds.append((refund_id := refund_id + 1, partial, 20_000))

    return {"users": users, "orders": orders, "refunds": refunds}


def _oracle(database: dict[str, list[tuple]]) -> list[tuple[str, int]]:
    """Independent aggregation over the generated rows (no SQL involved)."""
    refund_totals: dict[int, int] = {}
    for _, order_id, amount in database["refunds"]:
        refund_totals[order_id] = refund_totals.get(order_id, 0) + amount

    monthly: dict[str, int] = {}
    for order_id, _, created_at, amount, status in database["orders"]:
        if status != "paid" or not created_at.startswith(str(YEAR)):
            continue
        month = f"{created_at[:7]}-01"
        monthly[month] = monthly.get(month, 0) + max(amount - refund_totals.get(order_id, 0), 0)
    return sorted(monthly.items())


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _sqlite_rows(database: dict[str, list[tuple]], statement: str, timeout_s: float = 20.0) -> list[tuple]:
    """Run one statement against a freshly built SQLite database."""
    with tempfile.TemporaryDirectory(prefix="dailybench-c3-") as folder:
        path = Path(folder) / "db.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.executescript(SCHEMA_SQLITE)
            connection.executemany("INSERT INTO users VALUES (?, ?)", database["users"])
            connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", database["orders"])
            connection.executemany("INSERT INTO refunds VALUES (?, ?, ?)", database["refunds"])
            connection.commit()
            connection.execute("PRAGMA query_only = ON")

            deadline = time.monotonic() + timeout_s
            connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
            cursor = connection.execute(statement)
            return list(cursor.fetchall())
        finally:
            connection.close()


class PostgresBackend:
    """A throwaway PostgreSQL container, driven through ``docker exec``."""

    def __init__(self) -> None:
        self.container: str | None = None

    def available(self) -> bool:
        try:
            proc = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def start(self) -> bool:
        if not self.available():
            return False
        try:
            proc = subprocess.run(
                [
                    "docker", "run", "-d", "--rm",
                    "-e", "POSTGRES_PASSWORD=postgres", "-e", "POSTGRES_DB=bench",
                    "postgres:16-alpine",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if proc.returncode != 0:
            return False
        self.container = proc.stdout.strip().splitlines()[-1]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self._psql("bench", "SELECT 1;", timeout=20).get("ok"):
                return True
            time.sleep(2)
        return False

    def _psql(self, database: str, sql: str, timeout: float = 60) -> dict[str, Any]:
        assert self.container
        try:
            proc = subprocess.run(
                [
                    "docker", "exec", "-i", self.container,
                    "psql", "-U", "postgres", "-d", database,
                    "-t", "-A", "-F", "\t", "--no-psqlrc", "-f", "-",
                ],
                input=sql,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}

    def load(self, database_name: str, database: dict[str, list[tuple]]) -> bool:
        statements = [
            "DROP TABLE IF EXISTS refunds; DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS users;",
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);",
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,"
            " created_at TIMESTAMP NOT NULL, amount_cents BIGINT NOT NULL, status TEXT NOT NULL);",
            "CREATE TABLE refunds (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL, amount_cents BIGINT NOT NULL);",
        ]
        for table in ("users", "orders", "refunds"):
            for row in database[table]:
                values = ", ".join("NULL" if value is None else f"'{value}'" for value in row)
                statements.append(f"INSERT INTO {table} VALUES ({values});")
        return self._psql(database_name, "\n".join(statements), timeout=120).get("ok", False)

    def query(self, database_name: str, statement: str, timeout: float = 30) -> dict[str, Any]:
        result = self._psql(database_name, statement, timeout=timeout)
        if not result.get("ok"):
            return result
        rows = []
        for line in (result.get("stdout") or "").splitlines():
            if not line.strip():
                continue
            rows.append(tuple(part.strip() for part in line.split("\t")))
        result["rows"] = rows
        return result

    def stop(self) -> None:
        if self.container:
            subprocess.run(
                ["docker", "rm", "-f", self.container],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.container = None


# --------------------------------------------------------------------------- #
# Statement validation and comparison
# --------------------------------------------------------------------------- #


def _statement_or_error(reply: str) -> tuple[str | None, str]:
    code = extract_code(reply)
    if not code:
        return None, "ответ пуст"
    statement = code.strip()
    if statement.endswith(";"):
        statement = statement[:-1].strip()
    if ";" in statement:
        return None, "несколько SQL-инструкций в одном ответе"
    if not re.match(r"(?is)^\s*(select|with)\b", statement):
        return None, "ответ не начинается с SELECT или WITH"
    lowered = statement.casefold()
    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            return None, f"запрещённая конструкция: {word}"
    return statement, ""


def _normalise_rows(rows: list[tuple]) -> list[tuple[str, int]]:
    normalised = []
    for row in rows:
        if len(row) < 2:
            continue
        month, revenue = row[0], row[1]
        month_text = str(month)[:10]
        match = re.match(r"(\d{4})-(\d{2})", month_text)
        if not match:
            normalised.append((month_text, int(float(revenue))))
            continue
        normalised.append((f"{match.group(1)}-{match.group(2)}-01", int(float(revenue))))
    return sorted(normalised)


# --------------------------------------------------------------------------- #
# Task generation
# --------------------------------------------------------------------------- #


def generate(rng: random.Random, cfg: Any) -> core.TaskInstance:
    backend = getattr(cfg, "sql_backend", "sqlite") or "sqlite"
    dialect = "PostgreSQL" if backend == "postgres" else "SQLite"
    databases = [_database(rng) for _ in range(DATABASES)]
    expected = [_oracle(database) for database in databases]
    reference = _REFERENCE_POSTGRES if backend == "postgres" else _REFERENCE_SQLITE

    task = core.TaskInstance(
        id="C-3",
        category="C",
        title="SQL",
        prompt=_PROMPT.replace("__DIALECT__", dialect),
        max_points=4.0,
        # An agent writes the query into the copy and can run it against the seed;
        # a chat run prints it and never touches the copy.
        workspace=_workspace(databases[0]),
        oracle={
            "reference": reference,
            "expected": [[[month, value] for month, value in rows] for rows in expected],
            "databases": databases,
            "backend": backend,
            "dialect": dialect,
            "checks": [f"db{i + 1}" for i in range(DATABASES)],
            "answer_file": ANSWER_FILE,
        },
        meta={"scenario": "C-3", "backend": backend, "databases": DATABASES},
    )
    return task


def _workspace(database: dict[str, list[tuple]]) -> dict[str, str]:
    """The agent's copy: schema, the first database as SQL, and the answer slot.

    Plain SQL rather than a binary file: it round-trips through text, so the
    fixture is byte-identical on every machine and the agent can load it with
    whatever database tool it prefers.
    """
    return {
        "schema.sql": SCHEMA_SQLITE,
        "seed.sql": _seed_sql(database),
        ANSWER_FILE: "SELECT 1;\n",
        "README.md": (
            "# C-3\n\nThe benchmark runs the statement in "
            f"`{ANSWER_FILE}` against generated data.\n"
            "`schema.sql` is the schema; `seed.sql` loads the first database,"
            " e.g. `sqlite3 db.sqlite < seed.sql`.\n"
        ),
    }


def _seed_sql(database: dict[str, list[tuple]]) -> str:
    """The first generated database as plain SQL, for the agent to load."""

    def quote(value: object) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    lines = [SCHEMA_SQLITE]
    for table, rows in (
        ("users", database["users"]),
        ("orders", database["orders"]),
        ("refunds", database["refunds"]),
    ):
        for row in rows:
            lines.append(f"INSERT INTO {table} VALUES ({', '.join(quote(v) for v in row)});")
    return "\n".join(lines) + "\n"


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    oracle = task.oracle
    names = oracle["checks"]
    # The agent writes the statement into the working copy; a chat run prints it.
    # Both end in the same validator: an `answer.sql` that came wrapped in a
    # fence or in prose must be judged exactly as a chat reply would be.
    from .. import agent as _agent

    raw = _agent.answer_text(task, tr, tr.final_text)
    statement, error = _statement_or_error(raw if raw.strip() else tr.final_text)
    if statement is None:
        for index, name in enumerate(names):
            score.add(name, False, _POINTS_PER_CHECK, error)
        score.notes.append(f"SQL не принят к исполнению: {error}")
        return score

    backend = oracle.get("backend", "sqlite")
    expected = [tuple(tuple(row) for row in rows) for rows in oracle["expected"]]

    if backend == "postgres":
        postgres = PostgresBackend()
        if not postgres.start():
            score.error = "postgres backend unavailable"
            score.notes.append("Docker/PostgreSQL недоступен — сценарий C-3 не мог быть выполнен")
            for name in names:
                score.add(name, False, _POINTS_PER_CHECK, "postgres backend unavailable")
            return score
        try:
            for index, (name, database, want) in enumerate(zip(names, oracle["databases"], expected)):
                database_name = f"bench{index + 1}"
                if not postgres.load(database_name, database):
                    score.add(name, False, _POINTS_PER_CHECK, "не удалось загрузить данные в PostgreSQL")
                    continue
                result = postgres.query(database_name, statement)
                if not result.get("ok"):
                    score.add(name, False, _POINTS_PER_CHECK, (result.get("stderr") or "ошибка выполнения").strip()[:300])
                    continue
                got = _normalise_rows(result.get("rows") or [])
                want_rows = [(str(month), int(value)) for month, value in want]
                score.add(name, got == want_rows, _POINTS_PER_CHECK, f"получено {got}, ожидалось {want_rows}")
        finally:
            postgres.stop()
        return score

    for name, database, want in zip(names, oracle["databases"], expected):
        try:
            rows = _sqlite_rows(database, statement)
        except sqlite3.Error as exc:
            score.add(name, False, _POINTS_PER_CHECK, f"ошибка SQLite: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - defensive
            score.add(name, False, _POINTS_PER_CHECK, f"ошибка выполнения: {exc}")
            continue
        got = _normalise_rows(rows)
        want_rows = [(str(month), int(value)) for month, value in want]
        score.add(name, got == want_rows, _POINTS_PER_CHECK, f"получено {got}, ожидалось {want_rows}")
    return score
