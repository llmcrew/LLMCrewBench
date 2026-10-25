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
from collections.abc import Callable
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

#: Table DDL per backend, parent first; the insert order and the drop order
#: (reversed) both follow it, so a load can never race the foreign keys.
_PG_SCHEMA_C3 = {
    "users": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);",
    "orders": (
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,"
        " created_at TIMESTAMP NOT NULL, amount_cents BIGINT NOT NULL, status TEXT NOT NULL);"
    ),
    "refunds": "CREATE TABLE refunds (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL, amount_cents BIGINT NOT NULL);",
}

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
    """A throwaway PostgreSQL container, driven through ``docker exec``.

    The container starts with a single ``bench`` database; every database a
    scenario actually queries is created here, before the first load, so a check
    can never fail merely because nobody issued ``CREATE DATABASE``. A psql
    connection failure is reported as ``unreachable`` — an evaluator defect —
    while a statement error stays a model answer.
    """

    def __init__(self) -> None:
        self.container: str | None = None
        self.databases: tuple[str, ...] = ()

    @staticmethod
    def database_name(index: int) -> str:
        return f"bench{index + 1}"

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

    def start(self, databases: int) -> bool:
        """Start the container and create ``databases`` scratch databases."""
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
                break
            time.sleep(2)
        else:
            return False

        names = []
        for index in range(databases):
            name = self.database_name(index)
            if not self._psql("bench", f'CREATE DATABASE "{name}";', timeout=60).get("ok"):
                return False
            if not self._psql(name, "SELECT 1;", timeout=20).get("ok"):
                return False
            names.append(name)
        self.databases = tuple(names)
        return True

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
            return {"ok": False, "unreachable": True, "error": str(exc)}
        stderr = proc.stderr or ""
        lowered = stderr.casefold()
        # psql exits 2 when the connection itself is bad; docker adds its own
        # words when the container vanished under the run.
        unreachable = proc.returncode == 2 or "no such container" in lowered or "is not running" in lowered
        return {
            "ok": proc.returncode == 0,
            "unreachable": unreachable,
            "stdout": proc.stdout,
            "stderr": stderr,
        }

    def load(self, database_name: str, tables: dict[str, tuple[str, list[tuple]]]) -> dict[str, Any]:
        """Rebuild ``tables`` (parent first) in the named database."""
        statements = ["DROP TABLE IF EXISTS " + ", ".join(reversed(list(tables))) + ";"]
        statements.extend(ddl for ddl, _ in tables.values())
        for table, (_, rows) in tables.items():
            for row in rows:
                values = ", ".join("NULL" if value is None else f"'{value}'" for value in row)
                statements.append(f"INSERT INTO {table} VALUES ({values});")
        return self._psql(database_name, "\n".join(statements), timeout=120)

    def query(self, database_name: str, statement: str, timeout: float = 30) -> dict[str, Any]:
        result = self._psql(database_name, statement, timeout=timeout)
        if not result.get("ok"):
            return result
        rows = []
        for line in (result.get("stdout") or "").splitlines():
            if not line.strip():
                continue
            rows.append(tuple(_field(part) for part in line.split("\t")))
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


_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def _field(raw: str) -> object:
    """psql prints every column as text; numbers must come back as numbers.

    A date (``2026-02-01``) keeps its spelling, so it is compared exactly; an
    integer or a numeric sum becomes a real number, so the cross-backend
    normalisation of whole values works for both SQLite and PostgreSQL.
    """
    value = raw.strip()
    if not _NUMBER.match(value):
        return value
    return float(value) if "." in value else int(value)


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


def _normalise_value(value: object) -> object:
    """A comparison key that keeps the numeric value intact.

    ``int(float(v))`` used to truncate ``+0.75`` away (C3); a real number now
    compares as itself, so a fractional sum is a wrong answer. Strings pass
    through unchanged: a date must match exactly, day included.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return str(value)


def _normalise_rows(rows: list[tuple]) -> list[tuple]:
    """Normalise only cross-backend spelling, never content or shape.

    The allowed differences between SQLite and Postgres outputs are: the column
    *names* (headers are not compared) and int/float spelling of whole numbers
    (``187168`` vs ``187168.0``). Everything else — column count, the exact
    date, fractional cents, row order — is part of the answer and is compared.
    """
    return [tuple(_normalise_value(value) for value in row) for row in rows]


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


def _c3_tables(database: dict[str, list[tuple]]) -> dict[str, tuple[str, list[tuple]]]:
    return {table: (ddl, database[table]) for table, ddl in _PG_SCHEMA_C3.items()}


def _c6_tables(database: dict[str, list[tuple]]) -> dict[str, tuple[str, list[tuple]]]:
    rows = {
        "customers": database["customers"],
        "orders": database["orders"],
        "order_events": database["events"],
    }
    return {table: (ddl, rows[table]) for table, ddl in _PG_SCHEMA_C6.items()}


def _postgres_unavailable(score: core.Score, names: list[str], reason: str) -> None:
    """The evaluator, not the answer, is what failed: mark it as such."""
    if score.error is None:
        score.error = f"evaluator unavailable: {reason}"
        score.notes.append(f"PostgreSQL недоступен — {reason}")
    for name in names:
        score.add(name, False, _POINTS_PER_CHECK, reason)


def _score_postgres(
    score: core.Score,
    names: list[str],
    databases: list[dict[str, list[tuple]]],
    expected: list[tuple[tuple, ...]],
    statement: str,
    tables_for: Callable[[dict[str, list[tuple]]], dict[str, tuple[str, list[tuple]]]],
) -> core.Score:
    """Load every database, run the statement once per database, compare.

    Every database named by the checks is created up front by ``start``; a lost
    connection afterwards stops the run and is reported as an evaluator error
    instead of silently becoming five zeroes.
    """
    postgres = PostgresBackend()
    if not postgres.start(len(names)):
        postgres.stop()
        _postgres_unavailable(score, names, "postgres backend unavailable (docker)")
        return score
    try:
        for index, (name, database, want) in enumerate(zip(names, databases, expected)):
            database_name = postgres.database_name(index)
            result = postgres.load(database_name, tables_for(database))
            if not result.get("ok"):
                if result.get("unreachable"):
                    _postgres_unavailable(score, names[index:], "соединение с PostgreSQL потеряно")
                    return score
                detail = (result.get("stderr") or "").strip()[:300]
                score.add(name, False, _POINTS_PER_CHECK, f"не удалось загрузить данные в PostgreSQL: {detail}")
                continue
            result = postgres.query(database_name, statement)
            if not result.get("ok"):
                if result.get("unreachable"):
                    _postgres_unavailable(score, names[index:], "соединение с PostgreSQL потеряно")
                    return score
                score.add(name, False, _POINTS_PER_CHECK, (result.get("stderr") or "ошибка выполнения").strip()[:300])
                continue
            got = _normalise_rows(result.get("rows") or [])
            want_rows = _normalise_rows([tuple(row) for row in want])
            score.add(name, got == want_rows, _POINTS_PER_CHECK, f"получено {got}, ожидалось {want_rows}")
    finally:
        postgres.stop()
    return score


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
        return _score_postgres(score, names, oracle["databases"], expected, statement, _c3_tables)

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
        want_rows = _normalise_rows([(month, value) for month, value in want])
        score.add(name, got == want_rows, _POINTS_PER_CHECK, f"получено {got}, ожидалось {want_rows}")
    return score


# --------------------------------------------------------------------------- #
# Scenario C-6 (stretch) — order journal at a cut-off
# --------------------------------------------------------------------------- #

C6_DATABASES = 5
C6_CUTOFF = "2026-05-31 23:59:59"
C6_ANSWER_FILE = "answer.sql"
C6_CATEGORIES = ("audio", "books", "garden", "kitchen", "outdoor", "tools")

C6_SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount_cents INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE TABLE order_events (
    event_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    kind TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    recorded_at TIMESTAMP NOT NULL
);
"""

_PG_SCHEMA_C6 = {
    "customers": "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL);",
    "orders": (
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL,"
        " amount_cents BIGINT NOT NULL, created_at TIMESTAMP NOT NULL);"
    ),
    "order_events": (
        "CREATE TABLE order_events (event_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL,"
        " revision INTEGER NOT NULL, kind TEXT NOT NULL, amount_cents BIGINT NOT NULL,"
        " recorded_at TIMESTAMP NOT NULL);"
    ),
}

_REFERENCE_C6_SQLITE = """WITH admissible AS (
    SELECT order_id, revision, kind, MAX(amount_cents) AS amount_cents
    FROM order_events
    WHERE recorded_at <= '2026-05-31 23:59:59'
    GROUP BY order_id, revision, kind
),
final AS (
    SELECT order_id, revision, kind,
           ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY revision DESC) AS rn
    FROM admissible
),
refunds AS (
    SELECT a.order_id, SUM(a.amount_cents) AS total
    FROM admissible AS a
    JOIN final AS f ON f.order_id = a.order_id AND f.rn = 1
    WHERE a.kind = 'refunded' AND a.revision <= f.revision
    GROUP BY a.order_id
),
state AS (
    SELECT f.order_id, f.kind, o.amount_cents AS amount, o.customer_id,
           COALESCE(r.total, 0) AS refunded
    FROM final AS f
    JOIN orders AS o ON o.id = f.order_id
    LEFT JOIN refunds AS r ON r.order_id = f.order_id
    WHERE f.rn = 1
)
SELECT c.category AS category,
       SUM(CASE WHEN s.kind = 'cancelled' THEN 0 ELSE 1 END) AS active_orders,
       SUM(CASE WHEN s.kind = 'cancelled' THEN 0 ELSE MAX(s.amount - s.refunded, 0) END) AS net_cents,
       SUM(CASE WHEN s.kind = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
FROM state AS s
JOIN customers AS c ON c.id = s.customer_id
GROUP BY c.category
ORDER BY net_cents DESC, c.category ASC
LIMIT 2;"""

_REFERENCE_C6_POSTGRES = _REFERENCE_C6_SQLITE.replace(
    "MAX(s.amount - s.refunded, 0)", "GREATEST(s.amount - s.refunded, 0)"
)

_PROMPT_C6 = """Напиши один __DIALECT__ SELECT по журналу изменений заказа.

Срез: cutoff = '2026-05-31 23:59:59'.

События order_events — ревизии заказа. Для каждого заказа допустимы только
события с recorded_at <= cutoff. Состояние заказа определяет ТОЛЬКО его
последняя допустимая ревизия (максимальный revision среди допустимых):

* kind = 'placed'    — заказ активен, net = orders.amount_cents;
* kind = 'refunded'  — заказ активен, net = max(orders.amount_cents -
                       сумма возвратов, 0);
* kind = 'cancelled' — заказ отменён, net = 0, возвраты не вычитаются.

Возвратами считаются допустимые события kind = 'refunded' с revision не
больше последней ревизии заказа. Повторная доставка события (та же пара
order_id + revision, возможно с другим event_id) учитывается ровно один раз:
ни состояние, ни возврат второй раз не меняются.

Заказ без допустимых событий в результат не попадает.

Сгруппируй по customers.category. Для каждой категории выведи:

category | active_orders | net_cents | cancelled_orders

active_orders — число активных заказов (placed или refunded),
net_cents — сумма net по ним, cancelled_orders — число отменённых.

Отсортируй по net_cents по убыванию, при равенстве — category по
возрастанию. Оставь первые 2 строки.

Схема:

```sql
customers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT
);

orders(
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount_cents INTEGER,
    created_at TIMESTAMP
);

order_events(
    event_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    revision INTEGER,
    kind TEXT,
    amount_cents INTEGER,
    recorded_at TIMESTAMP
);
```

Верни только SQL без Markdown.
"""


def _c6_stamp(rng: random.Random, after: bool = False) -> str:
    if after:
        day = rng.randrange(1, 21)
        return f"2026-06-{day:02d} {rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}:00"
    month = rng.randrange(1, 6)
    day = rng.randrange(1, 29)
    return f"2026-{month:02d}-{day:02d} {rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}:00"


def _c6_split(total: int, parts: int, rng: random.Random) -> list[int]:
    """``parts`` positive multiples of 100 that add up to ``total``."""
    chunks = [100] * parts
    left = total - 100 * parts
    for index in range(parts):
        take = rng.randrange(0, left // 100 + 1)
        chunks[index] += take * 100
        left -= take * 100
        if left == 0:
            break
    if left:
        chunks[-1] += left
    return chunks


def _database_c6(rng: random.Random) -> dict[str, list[tuple]]:
    """One generated base: controlled category nets, ties in the top two."""
    categories = list(rng.sample(C6_CATEGORIES, rng.randrange(3, 5)))
    assignment = list(categories) + [
        rng.choice(categories) for _ in range(rng.randrange(2, 5))
    ]
    rng.shuffle(assignment)
    customers = [
        (index + 1, _NAMES[index % len(_NAMES)], category)
        for index, category in enumerate(assignment)
    ]
    by_category: dict[str, list[int]] = {category: [] for category in categories}
    for customer_id, _, category in customers:
        by_category[category].append(customer_id)

    # The two leading categories are tied on purpose; the third is strictly
    # below them, so a top-2 without the tie-break is wrong.
    base = rng.randrange(200, 501) * 100
    targets = [base, base]
    for _ in range(len(categories) - 2):
        base = base - rng.randrange(10, 51) * 100
        targets.append(max(base, 1000))

    orders: list[tuple] = []
    events: list[tuple] = []
    order_id = 0
    event_id = 0

    def add_order(customer_id: int, amount: int, created_at: str) -> int:
        nonlocal order_id
        order_id += 1
        orders.append((order_id, customer_id, amount, created_at))
        return order_id

    def add_event(oid: int, revision: int, kind: str, amount: int, stamp: str, copies: int = 1) -> None:
        nonlocal event_id
        for _ in range(copies):
            event_id += 1
            events.append((event_id, oid, revision, kind, amount, stamp))

    for category, target in zip(categories, targets):
        customer_id = rng.choice(by_category[category])
        refunded = target >= 9000 and rng.random() < 0.6
        over = rng.random() < 0.25
        slots = rng.randrange(2, 5)
        active_total = target - (4000 if refunded else 0)
        for amount in _c6_split(max(active_total, slots * 100), slots, rng):
            oid = add_order(customer_id, amount, "2026-01-15 10:00:00")
            add_event(oid, 1, "placed", 0, _c6_stamp(rng), copies=2 if rng.random() < 0.3 else 1)
        if refunded:
            oid = add_order(customer_id, 5000, "2026-02-10 10:00:00")
            add_event(oid, 1, "placed", 0, _c6_stamp(rng))
            add_event(oid, 2, "refunded", 1000, _c6_stamp(rng), copies=2)
        # An order whose late cancellation arrives after the cut-off: the state
        # at the cut-off is still "placed".
        oid = add_order(customer_id, 2000, "2026-03-01 10:00:00")
        add_event(oid, 1, "placed", 0, _c6_stamp(rng))
        add_event(oid, 2, "cancelled", 0, _c6_stamp(rng, after=True))
        # A cancelled order counts once and adds no revenue.
        oid = add_order(customer_id, rng.randrange(10, 60) * 100, "2026-02-01 10:00:00")
        add_event(oid, 1, "placed", 0, _c6_stamp(rng))
        add_event(oid, 2, "cancelled", 0, _c6_stamp(rng), copies=2 if rng.random() < 0.5 else 1)
        if over:
            oid = add_order(customer_id, 800, "2026-04-01 10:00:00")
            add_event(oid, 1, "placed", 0, _c6_stamp(rng))
            add_event(oid, 2, "refunded", 1000, _c6_stamp(rng))

    # Orders that only exist after the cut-off never reach the result.
    for _ in range(rng.randrange(2, 5)):
        customer_id = rng.choice(customers)[0]
        oid = add_order(customer_id, rng.randrange(10, 90) * 100, "2026-06-05 10:00:00")
        add_event(oid, 1, "placed", 0, _c6_stamp(rng, after=True))
    # An order with no events at all is ignored as well.
    add_order(rng.choice(customers)[0], 12300, "2026-05-01 10:00:00")

    return {"customers": customers, "orders": orders, "events": events}


def _oracle_c6(database: dict[str, list[tuple]]) -> list[tuple[str, int, int, int]]:
    """Independent Python model: no SQL, no shared helper with the reference."""
    order_rows = {oid: (customer_id, amount) for oid, customer_id, amount, _ in database["orders"]}
    category_of = {customer_id: category for customer_id, _, category in database["customers"]}

    last: dict[int, tuple[int, str]] = {}
    refunds: dict[int, dict[int, int]] = {}
    for _, oid, revision, kind, amount, recorded_at in database["events"]:
        if recorded_at > C6_CUTOFF or oid not in order_rows:
            continue
        current = last.get(oid)
        if current is None or revision > current[0]:
            last[oid] = (revision, kind)
        if kind == "refunded":
            refunds.setdefault(oid, {})[revision] = amount

    grouped: dict[str, list[int]] = {}
    for oid, (revision, kind) in last.items():
        customer_id, order_amount = order_rows[oid]
        category = category_of[customer_id]
        counters = grouped.setdefault(category, [0, 0, 0])
        if kind == "cancelled":
            counters[2] += 1
            continue
        refunded = sum(
            amount for rev, amount in refunds.get(oid, {}).items() if rev <= revision
        )
        counters[0] += 1
        counters[1] += max(order_amount - refunded, 0)

    rows = [
        (category, counters[0], counters[1], counters[2])
        for category, counters in grouped.items()
    ]
    rows.sort(key=lambda row: (-row[2], row[0]))
    return rows[:2]


def _c6_seed_sql(database: dict[str, list[tuple]]) -> str:
    def quote(value: object) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    lines = [C6_SCHEMA]
    for table, rows in (
        ("customers", database["customers"]),
        ("orders", database["orders"]),
        ("order_events", database["events"]),
    ):
        for row in rows:
            lines.append(f"INSERT INTO {table} VALUES ({', '.join(quote(v) for v in row)});")
    return "\n".join(lines) + "\n"


def _workspace_c6(database: dict[str, list[tuple]]) -> dict[str, str]:
    return {
        "schema.sql": C6_SCHEMA,
        "seed.sql": _c6_seed_sql(database),
        C6_ANSWER_FILE: "SELECT 1;\n",
        "README.md": (
            "# C-6\n\nThe benchmark runs the statement in "
            f"`{C6_ANSWER_FILE}` against generated journals.\n"
            "`schema.sql` is the schema; `seed.sql` loads the first database."
        ),
    }


def _sqlite_rows_c6(database: dict[str, list[tuple]], statement: str, timeout_s: float = 20.0) -> list[tuple]:
    with tempfile.TemporaryDirectory(prefix="dailybench-c6-") as folder:
        path = Path(folder) / "db.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.executescript(C6_SCHEMA)
            connection.executemany("INSERT INTO customers VALUES (?, ?, ?)", database["customers"])
            connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", database["orders"])
            connection.executemany("INSERT INTO order_events VALUES (?, ?, ?, ?, ?, ?)", database["events"])
            connection.commit()
            connection.execute("PRAGMA query_only = ON")

            deadline = time.monotonic() + timeout_s
            connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
            return list(connection.execute(statement).fetchall())
        finally:
            connection.close()


def generate_c6(rng: random.Random, cfg: Any) -> core.TaskInstance:
    backend = getattr(cfg, "sql_backend", "sqlite") or "sqlite"
    dialect = "PostgreSQL" if backend == "postgres" else "SQLite"
    databases = [_database_c6(rng) for _ in range(C6_DATABASES)]
    expected = [_oracle_c6(database) for database in databases]
    reference = _REFERENCE_C6_POSTGRES if backend == "postgres" else _REFERENCE_C6_SQLITE

    return core.TaskInstance(
        id="C-6",
        category="C",
        title="SQL order journal",
        prompt=_PROMPT_C6.replace("__DIALECT__", dialect),
        max_points=4.0,
        workspace=_workspace_c6(databases[0]),
        oracle={
            "reference": reference,
            "expected": [[list(row) for row in rows] for rows in expected],
            "databases": databases,
            "backend": backend,
            "dialect": dialect,
            "checks": [f"db{i + 1}" for i in range(C6_DATABASES)],
            "target_file": C6_ANSWER_FILE,
            "answer_file": C6_ANSWER_FILE,
        },
        meta={"scenario": "C-6", "backend": backend, "databases": C6_DATABASES},
    )


def evaluate_c6(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    oracle = task.oracle
    names = oracle["checks"]
    from .. import agent as _agent

    raw = _agent.answer_text(task, tr, tr.final_text)
    statement, error = _statement_or_error(raw if raw.strip() else tr.final_text)
    if statement is None:
        for name in names:
            score.add(name, False, _POINTS_PER_CHECK, error)
        score.notes.append(f"SQL не принят к исполнению: {error}")
        return score

    backend = oracle.get("backend", "sqlite")
    expected = [tuple(tuple(row) for row in rows) for rows in oracle["expected"]]

    if backend == "postgres":
        return _score_postgres(score, names, oracle["databases"], expected, statement, _c6_tables)

    for name, database, want in zip(names, oracle["databases"], expected):
        try:
            rows = _sqlite_rows_c6(database, statement)
        except sqlite3.Error as exc:
            score.add(name, False, _POINTS_PER_CHECK, f"ошибка SQLite: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - defensive
            score.add(name, False, _POINTS_PER_CHECK, f"ошибка выполнения: {exc}")
            continue
        got = _normalise_rows(rows)
        want_rows = _normalise_rows([tuple(row) for row in want])
        score.add(name, got == want_rows, _POINTS_PER_CHECK, f"получено {got}, ожидалось {want_rows}")
    return score
