"""Category I — retrieval over workspace files, 2 scenarios × 2 checks × 1 point.

Both scenarios hand the model a small read-only workspace and ask for facts that
are spread over several files, so the answer cannot be produced from the prompt:

* I-1 aggregates records scattered over ``records/part-NN.txt`` (sum, mean,
  top-3 names, the file holding the maximum);
* I-2 resolves one entity recorded in several files with conflicting values
  under a stated rule (newest ``updated`` wins, ties broken by the larger file
  name) and asks how many records lost.

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
# Registration
# --------------------------------------------------------------------------- #

_BUILDERS = {"I-1": _i1, "I-2": _i2}
_EVALUATORS = {"I-1": _eval_i1, "I-2": _eval_i2}


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    return [_BUILDERS[scenario](rng) for scenario in IDS]


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    return _EVALUATORS[task.id](task, tr)


core.register(
    key="I",
    title="Retrieval over files",
    points=4.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
)
