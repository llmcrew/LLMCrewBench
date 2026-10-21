"""Category I — retrieval over workspace files.

Baseline scenarios (2 × 2 checks × 1 point):

* I-1 aggregates records scattered over ``records/part-NN.txt`` (sum, mean,
  top-3 names, the file holding the maximum);
* I-2 resolves one entity recorded in several files with conflicting values
  under a stated rule (newest ``updated`` wins, ties broken by the larger file
  name) and asks how many records lost.

Pilot scenario:

* I-3 resolves the effective configuration of an include tree: layer order is
  the depth-first include order, ``UNSET`` inherits, ``null`` clears and a list
  replaces the previous list; every final value must carry a file/line source.

Stretch scenario:

* I-4 joins versioned inventory shards by stable id, applies an owner
  correction and tombstones, and aggregates only the live objects, with the
  winning line of every live id as provenance.

Scoring reads only the final JSON answer: calling a tool earns nothing by itself.
The reference transcript is nevertheless driven through the same read-only driver
the model sees (its ``read_file`` calls come from ``oracle["golden_calls"]``), so
every required fact is provably retrievable from the workspace.

The surface is deliberately read-only — ``list_files``, ``read_file``,
``search_code``. ``_resolve`` rejects any path that escapes the workspace, so a
model cannot read the host filesystem even if it asks to.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .. import core, text

IDS = ("I-1", "I-2")

#: Two checks per scenario; every scenario is worth 2.0 points (frozen table).
_CHECK = 1.0

WORKSPACE_TOOLS = ("list_files", "read_file", "search_code")

_DESCRIPTIONS = {
    "list_files": "Перечислить файлы каталога задачи (путь относительно корня задачи).",
    "read_file": "Прочитать содержимое файла каталога задачи.",
    "search_code": "Найти подстроку в файлах каталога задачи и вернуть строки с путями.",
}

_PARAMS: dict[str, dict[str, Any]] = {
    "list_files": {"type": "object", "properties": {"path": {"type": "string"}}},
    "read_file": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "search_code": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
        "required": ["query"],
    },
}


def schemas() -> list[dict[str, Any]]:
    """The read-only function surface: no tool can write, rename or delete."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": _DESCRIPTIONS[name],
                "parameters": _PARAMS[name],
            },
        }
        for name in WORKSPACE_TOOLS
    ]


# --------------------------------------------------------------------------- #
# Read-only driver
# --------------------------------------------------------------------------- #


def _normalise(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    while raw.startswith("./"):
        raw = raw[2:]
    return str(PurePosixPath(raw)) if raw else ""


class ReadOnlyWorkspace:
    """Read-only view of the task workspace, driven by the runner's tool loop."""

    def __init__(self, task: core.TaskInstance, work_dir: Path) -> None:
        self.task = task
        self.work_dir = Path(work_dir)

    def schemas(self) -> list[dict[str, Any]]:
        return schemas()

    # -- helpers ---------------------------------------------------------- #

    def _resolve(self, path: str) -> Path:
        """Map a model-supplied path into the workspace, rejecting escapes.

        Resolving first means ``..`` traversal, absolute paths and symlinks all
        leave a path that is checked against the workspace root; anything that
        does not stay inside raises before any read happens.
        """
        target = (self.work_dir / _normalise(path)).resolve()
        root = self.work_dir.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"путь вне каталога задачи: {path}")
        return target

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.work_dir).as_posix()

    def _files(self, root: Path) -> list[Path]:
        return sorted(p for p in root.rglob("*") if p.is_file())

    def _target(self, path: str) -> Path:
        return self.work_dir if path in (".", "./", "") else self._resolve(path)

    # -- tools ------------------------------------------------------------ #

    def execute(self, name: str, arguments: dict[str, Any]) -> core.ToolOutcome:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return core.ToolOutcome(text=f"unknown tool: {name}", error=True)
        try:
            return handler(**arguments)
        except TypeError as exc:
            return core.ToolOutcome(text=f"invalid arguments for {name}: {exc}", error=True)
        except ValueError as exc:
            return core.ToolOutcome(text=str(exc), error=True)

    def _tool_list_files(self, path: str = ".") -> core.ToolOutcome:
        target = self._target(path)
        if target.is_file():
            return core.ToolOutcome(text=self._rel(target))
        if not target.is_dir():
            return core.ToolOutcome(text=f"not a directory: {path}", error=True)
        files = self._files(target)
        if not files:
            return core.ToolOutcome(text="(пусто)")
        return core.ToolOutcome(text="\n".join(self._rel(file) for file in files))

    def _tool_read_file(self, path: str) -> core.ToolOutcome:
        target = self._resolve(path)
        if not target.is_file():
            return core.ToolOutcome(text=f"file not found: {path}", error=True)
        try:
            return core.ToolOutcome(text=target.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            return core.ToolOutcome(text=f"cannot read {path}: {exc}", error=True)

    def _tool_search_code(self, query: str, path: str = ".") -> core.ToolOutcome:
        needle = str(query)
        target = self._target(path)
        files = [target] if target.is_file() else self._files(target)
        matches: list[str] = []
        for file in files:
            try:
                content = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(content.splitlines(), start=1):
                if needle in line:
                    matches.append(f"{self._rel(file)}:{number}: {line.strip()}")
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
        if not matches:
            return core.ToolOutcome(text=f"no matches for {needle!r}")
        return core.ToolOutcome(text="\n".join(matches))


def make_driver(task: core.TaskInstance, work_dir: Path) -> ReadOnlyWorkspace:
    return ReadOnlyWorkspace(task, work_dir)


core.register_tool_driver("I", make_driver)


# --------------------------------------------------------------------------- #
# Answer helpers
# --------------------------------------------------------------------------- #


def _integral(value: Any) -> int | None:
    """The announced type is an integer; accept ``41`` and ``41.0`` alike."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_answer(raw: str) -> tuple[Any, bool]:
    """Strict JSON first; a fenced/embedded object still counts for content."""
    payload, strict = text.strict_json(raw)
    if payload is None:
        payload = text.loose_json(raw)
    return payload, strict


def _check_columns(payload: Any) -> bool:
    return isinstance(payload, dict)


# --------------------------------------------------------------------------- #
# I-1 — aggregation across many files
# --------------------------------------------------------------------------- #

_RECORDS_DIR = "records"
_PER_FILE = 3
_JSON_I1 = (
    '{"sum": <сумма всех значений>, "mean": <среднее, округли до 2 знаков>, '
    '"top3": ["<имя1>", "<имя2>", "<имя3>"], "max_file": "<файл с максимумом>"}'
)


def _i1(rng: random.Random) -> core.TaskInstance:
    # 10..11 files: 11 reads plus a ``list_files`` still fit the 12-call budget.
    n_files = rng.randrange(10, 12)
    count = n_files * _PER_FILE
    # Distinct values across all files: the top-3 order and the maximum are then
    # unambiguous, so the reference answer is a single exact JSON object.
    values = rng.sample(range(10, 1000), count)
    names = [f"item-{index:02d}" for index in range(1, count + 1)]
    records = list(zip(names, values))

    files: dict[str, str] = {}
    owner: dict[str, str] = {}
    for file_index in range(n_files):
        path = f"{_RECORDS_DIR}/part-{file_index + 1:02d}.txt"
        chunk = records[file_index * _PER_FILE : (file_index + 1) * _PER_FILE]
        files[path] = "".join(f"{name} {value}\n" for name, value in chunk)
        for name, _ in chunk:
            owner[name] = path

    total = sum(values)
    mean = round(total / count, 2)
    ranked = [name for _, name in sorted(zip(values, names), reverse=True)]
    top3 = ranked[:3]
    max_file = owner[names[values.index(max(values))]]

    payload = {"sum": total, "mean": mean, "top3": top3, "max_file": max_file}
    reference = json.dumps(payload, ensure_ascii=False)

    prompt = "\n".join(
        [
            f"В каталоге {_RECORDS_DIR} лежат {n_files} файлов part-01.txt … part-{n_files:02d}.txt, "
            f"по {_PER_FILE} строки в каждом.",
            "Каждая строка — это `<имя> <целое>`: имя записи и её значение; имена и значения "
            "не повторяются.",
            "",
            "Доступны только инструменты чтения: list_files, read_file, search_code. "
            f"Собери данные из всех файлов каталога {_RECORDS_DIR}.",
            "",
            "Верни ТОЛЬКО JSON, без пояснений и без markdown:",
            _JSON_I1,
            "",
            "top3 — три имени с наибольшими значениями по убыванию значения; "
            f"max_file — путь файла (например {_RECORDS_DIR}/part-03.txt), в котором лежит максимум.",
        ]
    )

    # A model that reads only the first file still produces a well-formed answer:
    # that is the trap the scenario must split exactly (shape ok, content wrong).
    first_chunk = records[:_PER_FILE]
    first_sum = sum(value for _, value in first_chunk)
    naive = {
        "sum": first_sum,
        "mean": round(first_sum / _PER_FILE, 2),
        "top3": [name for name, _ in sorted(first_chunk, key=lambda pair: pair[1], reverse=True)],
        "max_file": f"{_RECORDS_DIR}/part-01.txt",
    }
    naive_answer = json.dumps(naive, ensure_ascii=False)

    # --- claims the scenario makes about its own fixtures ------------------ #
    assert n_files >= 10, "сценарий требует не меньше 10 файлов"
    assert n_files <= 12, "эталон должен уложиться в бюджет вызовов инструментов"
    assert len(values) == count and len(set(values)) == count, "значения должны быть уникальны"
    assert len(set(names)) == count
    assert all(value >= 10 for value in values)
    assert naive["sum"] != total, "однофайловый агрегат обязан отличаться от эталона"
    assert naive != payload
    for line in "".join(files.values()).splitlines():
        assert line not in prompt, "prompt не должен содержать сырых данных"
    for name in names:
        assert name not in prompt, "prompt не должен содержать имён записей"

    return core.TaskInstance(
        id="I-1",
        category="I",
        title="Retrieval over files",
        prompt=prompt,
        max_points=2.0,
        workspace=files,
        tools=schemas(),
        oracle={
            "reference": reference,
            "final_answer": reference,
            "naive_answer": naive_answer,
            "sum": total,
            "mean": mean,
            "top3": top3,
            "max_file": max_file,
            "golden_calls": [
                {"name": "read_file", "arguments": {"path": path}} for path in sorted(files)
            ],
            # Two listing calls of headroom: the natural order is "list, then
            # read", and the shared cap otherwise always cut the last read
            # (I-1 sat at 1/2 whenever the model listed first).
            "call_headroom": 2,
            "files": sorted(files),
        },
        meta={"scenario": "I-1"},
    )


def _shape_i1(payload: Any) -> bool:
    if not _check_columns(payload) or set(payload) != {"sum", "mean", "top3", "max_file"}:
        return False
    if _integral(payload["sum"]) is None or _number(payload["mean"]) is None:
        return False
    top3 = payload["top3"]
    if not isinstance(top3, list) or len(top3) != 3 or not all(isinstance(x, str) for x in top3):
        return False
    return isinstance(payload["max_file"], str)


def _content_i1(payload: Any, task: core.TaskInstance) -> bool:
    if not isinstance(payload, dict):
        return False
    total = _integral(payload.get("sum"))
    mean = _number(payload.get("mean"))
    top3 = payload.get("top3")
    if total is None or mean is None:
        return False
    if not isinstance(top3, list):
        # A malformed field is a wrong answer, not an evaluator crash: the
        # check must survive top3=1 or any other shape without losing points.
        return False
    return (
        total == task.oracle["sum"]
        and round(mean, 2) == task.oracle["mean"]
        and top3 == task.oracle["top3"]
        and payload.get("max_file") == task.oracle["max_file"]
    )


def _eval_i1(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="I")
    payload, strict = _parse_answer(tr.final_text)
    score.add(
        "json_shape",
        strict and _shape_i1(payload),
        _CHECK,
        "нужен строгий JSON с ключами sum (целое), mean (число), top3 (три строки) и max_file (строка)",
    )
    score.add(
        "aggregate",
        _content_i1(payload, task),
        _CHECK,
        f"получено {payload!r}; ожидалось sum={task.oracle['sum']}, mean={task.oracle['mean']}, "
        f"top3={task.oracle['top3']}, max_file={task.oracle['max_file']!r}",
    )
    return score


# --------------------------------------------------------------------------- #
# I-2 — conflicting records resolved by rule
# --------------------------------------------------------------------------- #

_NODE_DIR = "configs"
_ENTITY_KEYS = ("quota", "replicas", "retry_budget")
_DISTRACTOR_KEYS = ("max_workers", "batch_size", "cache_mb")
_JSON_I2 = '{"value": <значение победившей записи>, "superseded": <сколько записей проиграло>}'


def _i2(rng: random.Random) -> core.TaskInstance:
    n_files = rng.randrange(4, 7)
    entity = rng.choice(_ENTITY_KEYS)
    entity_count = rng.randrange(3, n_files + 1)
    positions = sorted(rng.sample(range(n_files), entity_count))
    # The winner sits strictly between the first and the last record-bearing
    # file, so neither "first file wins" nor "last line wins" can coincide with
    # the rule's answer.
    winner_pos = rng.randrange(1, entity_count - 1)
    winner_file = positions[winner_pos]
    tie_file = positions[winner_pos - 1]  # shares the newest timestamp, loses on the name
    others = [pos for pos in positions if pos not in (winner_file, tie_file)]

    values = rng.sample(range(10, 999), entity_count)  # distinct -> unambiguous
    value_of = dict(zip(positions, values))

    minutes = rng.sample(range(0, 55), entity_count - 1)  # distinct and below the top
    top_minute = max(minutes) + rng.randrange(1, 5)
    assert top_minute <= 59, "минуты должны попадать в 00..59"
    minute_of: dict[int, int] = {pos: minutes[index] for index, pos in enumerate(others)}
    minute_of[winner_file] = top_minute
    minute_of[tie_file] = top_minute

    files: dict[str, str] = {}
    for index in range(n_files):
        lines: list[str] = []
        for key in rng.sample(_DISTRACTOR_KEYS, rng.randrange(1, 3)):
            minutes_noise = rng.randrange(0, 50)
            lines.append(
                f"key={key} updated=2026-03-11T09:{minutes_noise:02d} value={rng.randrange(1, 64)}"
            )
        if index in minute_of:
            lines.append(
                f"key={entity} updated=2026-03-11T09:{minute_of[index]:02d} value={value_of[index]}"
            )
        rng.shuffle(lines)
        files[f"{_NODE_DIR}/node-{index + 1:02d}.txt"] = "\n".join(lines) + "\n"

    resolved_value = value_of[winner_file]
    superseded = entity_count - 1
    naive_first = value_of[positions[0]]
    naive_last = value_of[positions[-1]]

    reference = json.dumps({"value": resolved_value, "superseded": superseded}, ensure_ascii=False)
    naive_answer = json.dumps({"value": naive_first, "superseded": superseded}, ensure_ascii=False)
    naive_last_answer = json.dumps({"value": naive_last, "superseded": superseded}, ensure_ascii=False)

    prompt = "\n".join(
        [
            f"В каталоге {_NODE_DIR} лежат файлы node-01.txt … node-{n_files:02d}.txt.",
            f"В файлах встречаются строки вида `key=<ключ> updated=<время> value=<целое>`; "
            f"ключ `{entity}` может появляться в нескольких файлах с разными значениями.",
            "",
            "Правило разрешения конфликта: побеждает запись с самым новым updated; если у "
            "нескольких записей время совпадает, побеждает запись из файла, имя которого "
            "лексикографически больше.",
            "",
            "Доступны только инструменты чтения: list_files, read_file, search_code.",
            "",
            "Верни ТОЛЬКО JSON, без пояснений и без markdown:",
            _JSON_I2,
            "",
            f"superseded — сколько записей ключа `{entity}` проиграло победившей "
            "(сколько записей перекрыто победителем).",
        ]
    )

    # --- claims the scenario makes about its own fixtures ------------------ #
    timestamps = [minute_of[pos] for pos in positions]
    assert len(values) == entity_count and len(set(values)) == entity_count
    assert len(set(timestamps)) == entity_count - 1, "ровно две записи делят новейшее время"
    assert sum(1 for stamp in timestamps if stamp == top_minute) == 2
    assert winner_file > tie_file, "при равенстве времени побеждает большее имя файла"
    assert resolved_value != naive_first and resolved_value != naive_last
    assert resolved_value != value_of[tie_file], "победитель не совпадает с записью-дублёром"
    found = sum(1 for content in files.values() for line in content.splitlines() if f"key={entity} " in line)
    assert found == entity_count
    for content in files.values():
        for line in content.splitlines():
            assert line not in prompt, "prompt не должен содержать сырых данных"

    return core.TaskInstance(
        id="I-2",
        category="I",
        title="Retrieval over files",
        prompt=prompt,
        max_points=2.0,
        workspace=files,
        tools=schemas(),
        oracle={
            "reference": reference,
            "final_answer": reference,
            "naive_answer": naive_answer,
            "naive_last_answer": naive_last_answer,
            "value": resolved_value,
            "superseded": superseded,
            "entity": entity,
            "winner_file": f"{_NODE_DIR}/node-{winner_file + 1:02d}.txt",
            "naive_first": naive_first,
            "naive_last": naive_last,
            "golden_calls": [
                {"name": "read_file", "arguments": {"path": path}} for path in sorted(files)
            ],
            # Two listing calls of headroom: the natural order is "list, then
            # read", and the shared cap otherwise always cut the last read
            # (I-1 sat at 1/2 whenever the model listed first).
            "call_headroom": 2,
            "files": sorted(files),
        },
        meta={"scenario": "I-2"},
    )


def _shape_i2(payload: Any) -> bool:
    if not _check_columns(payload) or set(payload) != {"value", "superseded"}:
        return False
    return _integral(payload["value"]) is not None and _integral(payload["superseded"]) is not None


def _content_i2(payload: Any, task: core.TaskInstance) -> bool:
    if not isinstance(payload, dict):
        return False
    value = _integral(payload.get("value"))
    superseded = _integral(payload.get("superseded"))
    return value == task.oracle["value"] and superseded == task.oracle["superseded"]


def _eval_i2(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="I")
    payload, strict = _parse_answer(tr.final_text)
    score.add(
        "json_shape",
        strict and _shape_i2(payload),
        _CHECK,
        "нужен строгий JSON с целыми ключами value и superseded",
    )
    score.add(
        "conflict",
        _content_i2(payload, task),
        _CHECK,
        f"получено {payload!r}; ожидалось value={task.oracle['value']} "
        f"(победитель {task.oracle['winner_file']!r}), superseded={task.oracle['superseded']}",
    )
    return score


# --------------------------------------------------------------------------- #
# I-3 — effective configuration of an include tree
# --------------------------------------------------------------------------- #

_CONF_DIR = "conf"
_I3_ROOT = f"{_CONF_DIR}/main.conf"
_I3_UNSET = "UNSET"
_I3_SOURCE = re.compile(r"conf/[\w.\-/]+\.conf:\d+")
_I4_SOURCE = re.compile(r"inventory/shard-\d{2}\.txt:\d+")
_I3_JSON = (
    '{"values": {"<ключ>": <значение>}, "sources": {"<ключ>": "<файл>:<строка>"}}'
)

#: Include tree. An entry is ``("include", path)`` or ``("slot", key)``; every
#: slot receives one assignment line, so the rendered line numbers are exact.
#: Depth of the deepest chain is three include levels (main → base → defaults →
#: timeouts), and every file is reachable from the root.
_I3_TREE: dict[str, list[tuple[str, str]]] = {
    "conf/main.conf": [
        ("include", "conf/base.conf"),
        ("slot", "service.port"),
        ("slot", "routes"),
        ("include", "conf/region.conf"),
    ],
    "conf/base.conf": [
        ("slot", "logging.level"),
        ("slot", "service.port"),
        ("slot", "cache.ttl"),
        ("slot", "retry.backoff"),
        ("include", "conf/defaults.conf"),
    ],
    "conf/defaults.conf": [
        ("slot", "cache.ttl"),
        ("include", "conf/timeouts.conf"),
        ("slot", "retry.backoff"),
        ("include", "conf/pools.conf"),
    ],
    "conf/timeouts.conf": [("slot", "http.timeout")],
    "conf/pools.conf": [("slot", "pool.size")],
    "conf/limits.conf": [("slot", "max_conns"), ("slot", "pool.size")],
    "conf/region.conf": [
        ("slot", "region.name"),
        ("include", "conf/limits.conf"),
        ("include", "conf/overrides.conf"),
        ("slot", "logging.level"),
    ],
    "conf/overrides.conf": [
        ("slot", "cache.ttl"),
        ("slot", "max_conns"),
        ("slot", "routes"),
    ],
}

_I3_TREE_KEYS = (
    "service.port", "routes", "logging.level", "cache.ttl", "retry.backoff",
    "http.timeout", "pool.size", "max_conns", "region.name",
)

_ROUTE_POOL = ("/v2/orders", "/v2/refunds", "/v2/users", "/v3/carts", "/v3/price", "/v3/health")
_REGION_POOL = ("eu-west", "us-east", "ap-south", "sa-east")


def _i3_plan(rng: random.Random) -> dict[str, list[Any]]:
    """One value per slot, in render order (include lines shift precedence)."""
    first_list = rng.sample(_ROUTE_POOL, 2)
    second_list = rng.sample([route for route in _ROUTE_POOL if route not in first_list], 3)
    return {
        "service.port": [8081, 8080],
        "routes": [first_list, second_list],
        "logging.level": ["info", "debug"],
        # A real value, then an inheriting layer, then a clear: the final value
        # is ``null`` and neither "first wins" nor "UNSET is absent" holds.
        "cache.ttl": [120, _I3_UNSET, None],
        "retry.backoff": [9, rng.randrange(2, 6)],
        "http.timeout": [rng.randrange(15, 46)],
        "pool.size": [8, 12],
        "max_conns": [_I3_UNSET, 900],
        "region.name": [rng.choice(_REGION_POOL)],
    }


def _i3_entries(
    plan: dict[str, list[Any]], cycle: bool
) -> dict[str, list[tuple[str, str, Any]]]:
    """Fill the tree slots with concrete lines, keeping the declared order."""
    plan = {key: list(values) for key, values in plan.items()}
    rendered: dict[str, list[tuple[str, str, Any]]] = {}
    for path, structure in _I3_TREE.items():
        lines: list[tuple[str, str, Any]] = []
        for kind, value in structure:
            if kind == "include":
                lines.append(("include", value, None))
            else:
                lines.append(("key", value, plan[value].pop(0)))
        rendered[path] = lines
    if cycle:
        # A real cycle: limits → overrides → limits. A resolver that follows
        # include lines without a stack guard never terminates.
        rendered["conf/limits.conf"].append(("include", "conf/overrides.conf", None))
        rendered["conf/overrides.conf"].append(("include", "conf/limits.conf", None))
    for key, rest in plan.items():
        assert not rest, f"слоты ключа {key} не заполнены"
    return rendered


def _i3_render(value: Any) -> str:
    if value == _I3_UNSET:
        return _I3_UNSET
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _i3_workspace(entries: dict[str, list[tuple[str, str, Any]]]) -> dict[str, str]:
    workspace: dict[str, str] = {}
    for path, lines in entries.items():
        text_lines: list[str] = []
        for kind, name, value in lines:
            text_lines.append(f"include {name}" if kind == "include" else f"{name} = {_i3_render(value)}")
        workspace[path] = "\n".join(text_lines) + "\n"
    return workspace


def _i3_resolve(
    entries: dict[str, list[tuple[str, str, Any]]], mode: str = "exact"
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply the tree semantics; ``mode`` switches to one of the wrong models.

    The resolver walks the structure (not the rendered text), so the oracle is
    computed by a different path than the model's line-oriented reading.
    """
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    stack: list[str] = []

    def visit(path: str) -> None:
        if path in stack:
            return
        stack.append(path)
        for number, (kind, name, value) in enumerate(entries[path], start=1):
            if kind == "include":
                visit(name)
                continue
            if value == _I3_UNSET:
                continue
            if value is None and mode == "null_inherit":
                continue
            if mode == "first" and name in values:
                continue
            if isinstance(value, list) and mode == "concat" and name in values:
                values[name] = list(values[name]) + list(value)
            else:
                values[name] = list(value) if isinstance(value, list) else value
            sources[name] = f"{path}:{number}"
        stack.pop()

    visit(_I3_ROOT)
    return values, sources


def _i3_answer(values: dict[str, Any], sources: dict[str, str]) -> str:
    return json.dumps({"values": values, "sources": sources}, ensure_ascii=False, sort_keys=True)


def _i3(rng: random.Random) -> core.TaskInstance:
    cycle = rng.random() < 0.5
    plan = _i3_plan(rng)
    entries = _i3_entries(plan, cycle)
    files = _i3_workspace(entries)
    values, sources = _i3_resolve(entries)
    reference = _i3_answer(values, sources)
    naive_first = _i3_answer(*_i3_resolve(entries, mode="first"))
    naive_concat = _i3_answer(*_i3_resolve(entries, mode="concat"))
    naive_null = _i3_answer(*_i3_resolve(entries, mode="null_inherit"))

    prompt = "\n".join(
        [
            f"Корень конфигурации — {_I3_ROOT}, файлы лежат в каталоге {_CONF_DIR}.",
            "Строка `key = <значение>` задаёт ключ, строка `include <путь>` применяет "
            "включаемый файл на этом месте.",
            "Файлы применяются в порядке обхода: сверху вниз, include раскрывается на своей "
            "строке. Если файл уже открыт в текущей цепочке включений, повторно он не "
            "применяется.",
            "",
            "Поздняя строка перекрывает раннюю. `UNSET` — слой ничего не задаёт, значение "
            "наследуется от предыдущего слоя. `null` — ключ очищается, итоговое значение null. "
            "Список в квадратных скобках заменяет предыдущий список целиком.",
            "",
            "Значения бывают целыми, строками без кавычек, `null` и списками строк.",
            "Доступны только инструменты чтения: list_files, read_file, search_code.",
            "",
            "Верни ТОЛЬКО JSON, без пояснений и без markdown:",
            _I3_JSON,
            "",
            "values — все ключи дерева с их итоговыми значениями; sources — для каждого ключа "
            "файл и номер строки (нумерация с 1) выигравшей строки, например "
            f"{_CONF_DIR}/overrides.conf:2.",
        ]
    )

    # --- claims the scenario makes about its own fixtures ------------------ #
    assert set(values) == set(_I3_TREE_KEYS), "все ключи дерева обязаны быть разрешены"
    assert values["cache.ttl"] is None, "null обязан очистить ключ"
    assert len(values["routes"]) == 3, "список заменяется, а не склеивается"
    assert values["service.port"] == 8081 and values["logging.level"] == "debug"
    assert values["max_conns"] == 900 and values["pool.size"] == 12
    assert naive_first != reference and naive_concat != reference and naive_null != reference
    null_naive = json.loads(naive_null)["values"]
    assert null_naive["cache.ttl"] == 120, "модель «null как отсутствие» обязана отличаться"
    assert len(files) == len(_I3_TREE)
    for content in files.values():
        for line in content.splitlines():
            assert line not in prompt, "prompt не должен содержать сырых данных"

    return core.TaskInstance(
        id="I-3",
        category="I",
        title="Effective configuration of an include tree",
        prompt=prompt,
        max_points=2.0,
        workspace=files,
        tools=schemas(),
        oracle={
            "reference": reference,
            "final_answer": reference,
            "naive_answer": naive_first,
            "naive_concat_answer": naive_concat,
            "naive_null_answer": naive_null,
            "values": values,
            "sources": sources,
            "cycle": cycle,
            "golden_calls": [
                {"name": "read_file", "arguments": {"path": path}} for path in sorted(files)
            ],
            "call_headroom": 2,
            "files": sorted(files),
        },
        meta={"scenario": "I-3"},
    )


def _i3_shape(payload: Any) -> bool:
    if not _check_columns(payload) or set(payload) != {"values", "sources"}:
        return False
    values = payload["values"]
    sources = payload["sources"]
    if not isinstance(values, dict) or not isinstance(sources, dict):
        return False
    if set(values) != set(_I3_TREE_KEYS) or set(sources) != set(_I3_TREE_KEYS):
        return False
    for value in values.values():
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            continue
        if _integral(value) is None:
            return False
    for source in sources.values():
        if not isinstance(source, str) or not _I3_SOURCE.fullmatch(source):
            return False
    return True


def _i3_values_ok(payload: dict[str, Any], task: core.TaskInstance) -> bool:
    values = payload.get("values")
    if not isinstance(values, dict) or set(values) != set(task.oracle["values"]):
        return False
    for key, expected in task.oracle["values"].items():
        got = values.get(key)
        if _integral(got) is not None and _integral(expected) is not None:
            if _integral(got) != _integral(expected):
                return False
        elif got != expected:
            return False
    return True


def _i3_sources_ok(payload: dict[str, Any], task: core.TaskInstance) -> bool:
    sources = payload.get("sources")
    return isinstance(sources, dict) and sources == task.oracle["sources"]


def _eval_i3(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="I")
    payload, strict = _parse_answer(tr.final_text)
    score.add(
        "json_shape",
        strict and _i3_shape(payload),
        _CHECK / 2,
        'нужен строгий JSON {"values": {...}, "sources": {...<файл>:<строка>}} '
        f"со всеми ключами дерева: {sorted(_I3_TREE_KEYS)}",
    )
    score.add(
        "effective_values",
        isinstance(payload, dict) and _i3_values_ok(payload, task),
        _CHECK * 3 / 4,
        f"получено {payload.get('values') if isinstance(payload, dict) else payload!r}; "
        f"ожидалось {task.oracle['values']}",
    )
    score.add(
        "provenance",
        isinstance(payload, dict) and _i3_sources_ok(payload, task),
        _CHECK * 3 / 4,
        f"получено {payload.get('sources') if isinstance(payload, dict) else payload!r}; "
        f"ожидалось {task.oracle['sources']}",
    )
    return score


# --------------------------------------------------------------------------- #
# I-4 — join versioned inventory shards by stable id
# --------------------------------------------------------------------------- #

_INVENTORY_DIR = "inventory"
_SHARD_CAP = 4
_OWNERS = ("team-alpha", "team-beta", "team-gamma", "team-delta", "team-omega", "team-sigma")
_GOODS = (
    "widget", "gadget", "bolt", "nut", "spring", "valve", "pump", "rotor",
    "shaft", "gasket", "bracket", "filter", "lever", "cam", "coupler", "flange",
)
_I4_JSON = (
    '{"active": <сколько записей действует>, "by_owner": {"<владелец>": <сколько>}, '
    '"source": {"<id>": "<файл>:<строка>"}}'
)


def _i4_shards(n_files: int, events: list[tuple[int, int, str, str, str | None]]) -> dict[str, str]:
    """Render ``events`` (shard, order, kind, id, payload) into shard files."""
    events = sorted(events, key=lambda item: (item[0], item[1]))
    texts: dict[str, list[str]] = {index: [] for index in range(n_files)}
    for shard, _, kind, ident, payload in events:
        texts[shard].append(f"DELETE {ident}" if kind == "delete" else f"{ident} {payload}")
    return {
        f"{_INVENTORY_DIR}/shard-{index + 1:02d}.txt": "\n".join(lines) + "\n"
        for index, lines in texts.items()
    }


def _i4_answer(active: dict[str, tuple[str, str]]) -> str:
    by_owner: dict[str, int] = {}
    for owner, _ in active.values():
        by_owner[owner] = by_owner.get(owner, 0) + 1
    return json.dumps(
        {
            "active": len(active),
            "by_owner": dict(sorted(by_owner.items())),
            "source": {ident: source for ident, (_, source) in sorted(active.items())},
        },
        ensure_ascii=False,
    )


def _i4_lines(events: list[tuple[int, int, str, str, str | None]]) -> dict[tuple[int, int], int]:
    """Rendered line number of every event, keyed by (shard, order)."""
    numbers: dict[tuple[int, int], int] = {}
    counters: dict[int, int] = {}
    for shard, order, *_ in sorted(events, key=lambda item: (item[0], item[1])):
        counters[shard] = counters.get(shard, 0) + 1
        numbers[(shard, order)] = counters[shard]
    return numbers


def _i4_by_name(
    events: list[tuple[int, int, str, str, str | None]],
    names: dict[str, str],
    lines: dict[tuple[int, int], int],
) -> dict[str, tuple[str, str]]:
    active: dict[str, tuple[str, str]] = {}
    for shard, order, kind, ident, payload in sorted(events, key=lambda item: (item[0], item[1])):
        name = names[ident]
        if kind == "delete":
            active.pop(name, None)
        else:
            active[name] = (
                payload or "",
                f"{_INVENTORY_DIR}/shard-{shard + 1:02d}.txt:{lines[(shard, order)]}",
            )
    return active


def _i4(rng: random.Random) -> core.TaskInstance:
    n_files = rng.randrange(8, 11)
    n_ids = rng.randrange(10, 15)
    ids = [f"sku-{100 + index}" for index in range(n_ids)]
    rng.shuffle(ids)

    # Exactly two display names are shared by two different ids: a join by name
    # collapses those pairs, a join by id does not.
    pool = rng.sample(_GOODS, n_ids - 2)
    pair_a, pair_b = rng.sample(pool, 2)
    flat = pool + [pair_a, pair_b]
    rng.shuffle(flat)
    suffix = {base: rng.randrange(10, 99) for base in flat}
    names = {ident: f"{base}-{suffix[base]}" for ident, base in zip(ids, flat)}
    return _i4_build(rng, n_files, ids, names)


def _i4_build(
    rng: random.Random, n_files: int, ids: list[str], names: dict[str, str]
) -> core.TaskInstance:
    load = [0] * n_files
    events: list[tuple[int, int, str, str, str | None]] = []
    order = 0
    last: dict[str, int] = {}

    def place(shard: int, kind: str, ident: str, payload: str | None = None) -> None:
        nonlocal order
        load[shard] += 1
        events.append((shard, order, kind, ident, payload))
        order += 1
        last[ident] = shard

    def put(
        low: int, kind: str, ident: str, payload: str | None = None, reserve: int = 0
    ) -> int:
        for shard in range(low, n_files):
            if load[shard] + reserve < _SHARD_CAP:
                place(shard, kind, ident, payload)
                return shard
        raise AssertionError("нет места в shard-файлах")

    # Every file gets at least one record; the rest of the base records spread
    # with three slots per shard, keeping the fourth for the jotail events.
    for index, ident in enumerate(ids):
        shard = index if index < n_files else rng.randrange(0, n_files)
        while load[shard] >= 3:
            shard = rng.randrange(0, n_files)
        put(shard, "record", ident, f"{names[ident]} {rng.choice(_OWNERS)}")

    duplicates = [ident for ident in ids if list(names.values()).count(names[ident]) == 2]
    # Reserve the object that will be revived while early shards are still free.
    early_pool = [ident for ident in ids if ident not in duplicates]
    revived = rng.choice(early_pool)
    remaining = [
        ident for ident in ids if last[ident] <= n_files - 2 and ident != revived
    ]
    owned: list[str] = []
    # A correction needs a later shard; retry rather than shorten a jammed tail.
    for _ in range(3):
        for ident in sorted(remaining, key=lambda item: last[item]):
            try:
                put(last[ident] + 1, "record", ident, f"{names[ident]} {rng.choice(_OWNERS)}", reserve=1)
            except AssertionError:
                continue
            owned.append(ident)
            remaining.remove(ident)
            break
        else:
            raise AssertionError("нет места для исправления владельца")

    deletable = [
        ident for ident in ids if ident not in duplicates and ident not in owned
    ]
    # The tombstone reserves the slot its resurrection record goes into; the
    # earliest object with room today wins.
    for ident in sorted(deletable, key=lambda item: last[item]):
        try:
            shard = put(last[ident], "delete", ident, reserve=1)
        except AssertionError:
            continue
        revived = ident
        place(shard, "record", revived, f"{names[revived]} {rng.choice(_OWNERS)}")
        break
    else:
        raise AssertionError("нет места для tombstone с воскрешением")
    for ident in sorted((item for item in deletable if item != revived), key=lambda item: last[item]):
        try:
            put(last[ident], "delete", ident)
        except AssertionError:
            continue
        break
    else:
        raise AssertionError("нет места для второго tombstone")

    files = _i4_shards(n_files, events)
    ordered = sorted(events, key=lambda item: (item[0], item[1]))
    lines = _i4_lines(ordered)

    def model(events_in: list[tuple[int, int, str, str, str | None]]) -> dict[str, tuple[str, str]]:
        active: dict[str, tuple[str, str]] = {}
        for shard, order, kind, ident, payload in events_in:
            if kind == "delete":
                active.pop(ident, None)
            else:
                active[ident] = (
                    payload.rsplit(" ", 1)[1],
                    f"{_INVENTORY_DIR}/shard-{shard + 1:02d}.txt:{lines[(shard, order)]}",
                )
        return active

    active = model(ordered)
    records_only = [event for event in ordered if event[2] == "record"]
    everything = model(records_only)
    reference = _i4_answer(active)
    by_name = _i4_answer(_i4_by_name(ordered, names, lines))
    with_deleted = _i4_answer(everything)

    first_source: dict[str, tuple[str, str]] = {}
    for shard, order, kind, ident, payload in ordered:
        if kind == "record":
            first_source.setdefault(
                ident,
                (
                    payload.rsplit(" ", 1)[1],
                    f"{_INVENTORY_DIR}/shard-{shard + 1:02d}.txt:{lines[(shard, order)]}",
                ),
            )
    stale = _i4_answer({ident: first_source[ident] for ident in active})

    prompt = "\n".join(
        [
            f"В каталоге {_INVENTORY_DIR} лежат shard-файлы shard-01.txt … shard-{n_files:02d}.txt "
            "с версиями инвентаря.",
            "Строка `<id> <название> <владелец>` — версия записи; строка `DELETE <id>` — "
            "tombstone, удаляющий запись.",
            "",
            "Shard-файлы применяются по возрастанию номера, строки внутри файла — сверху вниз; "
            "поздняя версия перекрывает раннюю по стабильному id. Tombstone удаляет запись, "
            "но более поздняя версия после него снова её создаёт.",
            "",
            "Доступны только инструменты чтения: list_files, read_file, search_code.",
            "",
            "Верни ТОЛЬКО JSON, без пояснений и без markdown:",
            _I4_JSON,
            "",
            "Учитывай только действующие записи: active — их число, by_owner — сколько "
            "действующих записей у каждого владельца, source — файл и номер строки (с 1) "
            "выигравшей версии каждого действующего id.",
        ]
    )

    # --- claims the scenario makes about its own fixtures ------------------ #
    live_names = [names[ident] for ident in active]
    assert len(set(live_names)) == len(live_names) - 2, "ровно две пары имён должны совпадать"
    assert len(active) == len(ids) - 1, "один tombstone остаётся окончательным"
    assert revived in active, "запись после tombstone снова действует"
    assert sum(1 for ident in ids if ident not in active) == 1
    assert len(events) <= 30 and len(events) >= 12
    assert len(files) == n_files
    assert by_name != reference and with_deleted != reference and stale != reference
    assert "DELETE" in "\n".join(files.values())
    for content in files.values():
        for line in content.splitlines():
            assert line not in prompt, "prompt не должен содержать сырых данных"

    return core.TaskInstance(
        id="I-4",
        category="I",
        title="Join inventory shards by stable id",
        prompt=prompt,
        max_points=2.0,
        workspace=files,
        tools=schemas(),
        oracle={
            "reference": reference,
            "final_answer": reference,
            "naive_answer": by_name,
            "naive_deleted_answer": with_deleted,
            "naive_stale_answer": stale,
            "active": len(active),
            "by_owner": json.loads(reference)["by_owner"],
            "source": json.loads(reference)["source"],
            "deleted": [ident for ident in ids if ident not in active],
            "revived": revived,
            "golden_calls": [
                {"name": "read_file", "arguments": {"path": path}} for path in sorted(files)
            ],
            "call_headroom": 2,
            "files": sorted(files),
        },
        meta={"scenario": "I-4"},
    )


def _i4_shape(payload: Any) -> bool:
    if not _check_columns(payload) or set(payload) != {"active", "by_owner", "source"}:
        return False
    if _integral(payload["active"]) is None:
        return False
    by_owner = payload["by_owner"]
    if not isinstance(by_owner, dict) or not by_owner:
        return False
    if not all(isinstance(key, str) and (_integral(value) or 0) > 0 for key, value in by_owner.items()):
        return False
    source = payload["source"]
    if not isinstance(source, dict):
        return False
    return all(isinstance(key, str) and isinstance(value, str) and _I4_SOURCE.fullmatch(value)
               for key, value in source.items())


def _i4_state_ok(payload: dict[str, Any], task: core.TaskInstance) -> bool:
    active = _integral(payload.get("active"))
    by_owner = payload.get("by_owner")
    return (
        active == task.oracle["active"]
        and isinstance(by_owner, dict)
        and {key: _integral(value) for key, value in by_owner.items()} == task.oracle["by_owner"]
    )


def _i4_sources_ok(payload: dict[str, Any], task: core.TaskInstance) -> bool:
    source = payload.get("source")
    return isinstance(source, dict) and source == task.oracle["source"]


def _eval_i4(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="I")
    payload, strict = _parse_answer(tr.final_text)
    score.add(
        "json_shape",
        strict and _i4_shape(payload),
        _CHECK / 2,
        'нужен строгий JSON {"active": <целое>, "by_owner": {...}, "source": {...<файл>:<строка>}}',
    )
    score.add(
        "active_state",
        isinstance(payload, dict) and _i4_state_ok(payload, task),
        _CHECK * 3 / 4,
        f"получено {payload!r}; ожидалось active={task.oracle['active']}, "
        f"by_owner={task.oracle['by_owner']}",
    )
    score.add(
        "provenance",
        isinstance(payload, dict) and _i4_sources_ok(payload, task),
        _CHECK * 3 / 4,
        f"получено {payload.get('source') if isinstance(payload, dict) else payload!r}; "
        f"ожидалось {task.oracle['source']}",
    )
    return score


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

_BUILDERS = {"I-1": _i1, "I-2": _i2, "I-3": _i3, "I-4": _i4}
_EVALUATORS = {"I-1": _eval_i1, "I-2": _eval_i2, "I-3": _eval_i3, "I-4": _eval_i4}

BASELINE_IDS = IDS
PILOT_IDS = ("I-3",)
ALL_IDS = BASELINE_IDS + PILOT_IDS + ("I-4",)


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    profile = core.profile_of(cfg)
    if profile == "pilot":
        ids = PILOT_IDS
    elif profile == "stretch-v1":
        ids = ALL_IDS
    else:
        ids = BASELINE_IDS
    return [_BUILDERS[scenario](rng) for scenario in ids]


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    return _EVALUATORS[task.id](task, tr)


core.register(
    key="I",
    title="Retrieval over files",
    points=4.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=BASELINE_IDS,
    expected_ids_by_profile={"pilot": PILOT_IDS, "stretch-v1": ALL_IDS},
)
