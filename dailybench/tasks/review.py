"""Category J — code review, 5 points over two scenarios.

J-1 (3.0) — an off-by-one boundary defect in a pure window helper: it returns
one element more than asked whenever ``count`` is smaller than the input, and
the public test never exercises that case.
J-2 (2.0) — a defect that only shows under repetition: the module cache is keyed
on half of the input, so a second call with another root silently returns the
first result. A single call looks perfect; the state is what is wrong.

Both scenarios ship a small repository whose public test file is green, and a
one-line defect. The model must return strict JSON with the file, the defect
line and the complete corrected file; scoring is behavioural — the returned
``fix`` is written into a temporary copy of the repository and the hidden tests
run there, so a plausible-sounding answer that does not actually repair the
behaviour earns nothing. The reported line is derived from the fixture itself,
never hard-coded.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .. import core, text

IDS = ("J-1", "J-2")
_POINTS = {"J-1": 3.0, "J-2": 2.0}

#: Finding the defect is worth the same in both scenarios; the repair carries
#: the rest. A review that mislocates the line still gets paid for a fix that
#: really works, and vice versa.
_LOCATION_POINTS = 1.0

#: A wedged interpreter must not stall the whole run.
_HIDDEN_TIMEOUT_S = 60


# --------------------------------------------------------------------------- #
# Fixture templates
# --------------------------------------------------------------------------- #
# Every template is filled with ``_fill`` (plain ``str.replace``), so the braces
# of the Python code inside stay literal and no template needs escaping.

_J1_HELPER = '''"""Human-readable labels shared by the pipeline steps."""

from __future__ import annotations

_PREFIX = "@PREFIX@"


def label(name: str, value: object) -> str:
    """Render ``name=value`` with the service prefix."""
    return f"{_PREFIX}{name}={value}"
'''

#: ``@RETURN@`` is the slice expression: the buggy and the fixed file differ in
#: that one line only.
_J1_WINDOW = '''"""Sliding-window helpers for the metrics pipeline."""

from __future__ import annotations


def @FUNC@(values: list, count: int) -> list:
    """Return the last ``count`` values of ``values``, oldest first.

    ``count`` is clamped to the length of ``values``; a non-positive ``count``
    yields an empty list.
    """
    if count <= 0:
        return []
    return values[@RETURN@]
'''

_J1_PUBLIC_TEST = '''"""Public test: the documented behaviour of @FUNC@."""

import unittest

from @DOTTED@ import @FUNC@

SAMPLE = [@SAMPLE@]


class WindowTests(unittest.TestCase):
    def test_count_equal_to_length_keeps_everything(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, len(SAMPLE)), SAMPLE)

    def test_count_above_length_keeps_everything(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, 99), SAMPLE)

    def test_zero_count_is_empty(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, 0), [])

    def test_negative_count_is_empty(self) -> None:
        self.assertEqual(@FUNC@(SAMPLE, -3), [])

    def test_empty_input_is_empty(self) -> None:
        self.assertEqual(@FUNC@([], 4), [])


if __name__ == "__main__":
    unittest.main()
'''

_J1_HIDDEN_TESTS = '''"""Hidden acceptance tests for @DOTTED@.@FUNC@ (not shown to the model)."""

import unittest

from @DOTTED@ import @FUNC@


class BoundaryTests(unittest.TestCase):
    def test_returns_exactly_count_values(self) -> None:
        self.assertEqual(@FUNC@([1, 2, 3, 4, 5], 2), [4, 5])
        self.assertEqual(@FUNC@([1, 2, 3, 4, 5], 3), [3, 4, 5])
        self.assertEqual(@FUNC@([1, 2, 3, 4, 5], 4), [2, 3, 4, 5])

    def test_count_one_returns_the_last_value(self) -> None:
        self.assertEqual(@FUNC@([7, 8, 9], 1), [9])

    def test_count_above_length_keeps_everything(self) -> None:
        self.assertEqual(@FUNC@([1, 2, 3], 10), [1, 2, 3])

    def test_non_positive_count_is_empty(self) -> None:
        self.assertEqual(@FUNC@([1, 2, 3], 0), [])
        self.assertEqual(@FUNC@([1, 2, 3], -1), [])

    def test_empty_input_is_empty(self) -> None:
        self.assertEqual(@FUNC@([], 5), [])

    def test_duplicates_are_kept(self) -> None:
        self.assertEqual(@FUNC@([5, 5, 5, 5], 2), [5, 5])

    def test_input_is_not_mutated(self) -> None:
        values = [1, 2, 3]
        @FUNC@(values, 2)
        self.assertEqual(values, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
'''

_J2_DECOY = '''"""Static ingest configuration."""

from __future__ import annotations

DEFAULTS = {
    "shard_size": @SHARD@,
    "retries": 2,
}


def shard_size() -> int:
    """Number of records read per shard."""
    return DEFAULTS["shard_size"]
'''

#: ``@KEY@`` is the cache key: the buggy and the fixed file differ in that one
#: line only.
_J2_PATHS = '''"""Path joining helpers for the ingest service."""

from __future__ import annotations

_CACHE: dict = {}


def @FUNC@(root: str, name: str) -> str:
    """Join ``root`` and ``name`` with exactly one slash between them.

    Repeated calls with the same arguments are served from a small cache.
    """
    key = @KEY@
    if key in _CACHE:
        return _CACHE[key]
    joined = root.rstrip("/") + "/" + name.lstrip("/")
    _CACHE[key] = joined
    return joined
'''

_J2_PUBLIC_TEST = '''"""Public test: the documented behaviour of @FUNC@."""

import unittest

from @DOTTED@ import @FUNC@


class PathTests(unittest.TestCase):
    def test_join_adds_one_slash(self) -> None:
        self.assertEqual(@FUNC@("/srv/data", "report.csv"), "/srv/data/report.csv")

    def test_trailing_slashes_are_collapsed(self) -> None:
        self.assertEqual(@FUNC@("/srv/", "/tmp/out.txt"), "/srv/tmp/out.txt")

    def test_empty_name_keeps_the_root(self) -> None:
        self.assertEqual(@FUNC@("/srv/data", ""), "/srv/data/")

    def test_repeated_call_is_stable(self) -> None:
        first = @FUNC@("/srv/data", "report.csv")
        second = @FUNC@("/srv/data", "report.csv")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
'''

#: The names here are deliberately disjoint from the public test's names: with
#: the shared module cache, reusing a name would let the public run (or the
#: hidden run) change the other one's verdict on the buggy file, and the two
#: facts the fixture asserts — public green, hidden red — must both hold.
_J2_HIDDEN_TESTS = '''"""Hidden acceptance tests for @DOTTED@.@FUNC@ (not shown to the model)."""

import unittest

from @DOTTED@ import @FUNC@


class CacheTests(unittest.TestCase):
    def test_same_name_under_different_roots(self) -> None:
        self.assertEqual(@FUNC@("/srv/a", "part-1.csv"), "/srv/a/part-1.csv")
        self.assertEqual(@FUNC@("/srv/b", "part-1.csv"), "/srv/b/part-1.csv")

    def test_cache_does_not_cross_roots(self) -> None:
        self.assertEqual(@FUNC@("/mnt/one", "shard.bin"), "/mnt/one/shard.bin")
        self.assertEqual(@FUNC@("/mnt/two", "shard.bin"), "/mnt/two/shard.bin")

    def test_trailing_slashes_per_root(self) -> None:
        self.assertEqual(@FUNC@("/srv/x/", "tail.log"), "/srv/x/tail.log")
        self.assertEqual(@FUNC@("/srv/y/", "tail.log"), "/srv/y/tail.log")

    def test_repeated_call_is_stable(self) -> None:
        first = @FUNC@("/opt/app", "conf.ini")
        second = @FUNC@("/opt/app", "conf.ini")
        self.assertEqual(first, "/opt/app/conf.ini")
        self.assertEqual(second, "/opt/app/conf.ini")

    def test_name_with_leading_slash(self) -> None:
        self.assertEqual(@FUNC@("/opt/app", "/deep/file"), "/opt/app/deep/file")
        self.assertEqual(@FUNC@("/other/app", "/deep/file"), "/other/app/deep/file")


if __name__ == "__main__":
    unittest.main()
'''

_PROMPT = """В рабочей копии репозитория ровно один файл содержит дефект: поведение
расходится с тем, что обещает docstring, и с намерением публичных тестов.
Дефект укладывается ровно в одну строку — исправлять нужно только её.

Файлы рабочей копии:

{files}
Публичный тест {public_test} описывает ожидаемое поведение и сейчас проходит:
дефектный случай он не покрывает.

Верни ТОЛЬКО JSON, без markdown-обёртки и текста вокруг:

{{"file": "<путь файла с дефектом>", "line": <номер строки с дефектом>, "fix": "<полный исправленный текст файла>"}}

* "file" — путь относительно корня рабочей копии, ровно как в списке выше.
* "line" — номер дефектной строки в этом файле, нумерация с 1.
* "fix" — полный текст файла после исправления (не diff, без многоточий и сокращений).
"""

_RUNNER = '''"""Runs the named test modules and reports one JSON line per test."""

import importlib.util
import json
import sys
import unittest


class Recorder(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.rows = []

    def addSuccess(self, test) -> None:
        self.rows.append((test.id(), True, ""))

    def addFailure(self, test, err) -> None:
        self.rows.append((test.id(), False, str(err[1])))

    def addError(self, test, err) -> None:
        self.rows.append((test.id(), False, str(err[1])))

    def addSkip(self, test, reason) -> None:
        self.rows.append((test.id(), False, f"skipped: {reason}"))


def run_module(label: str, path: str) -> None:
    spec = importlib.util.spec_from_file_location(f"dailybench_review_{label}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    recorder = Recorder()
    suite.run(recorder)
    for name, ok, detail in recorder.rows:
        print(json.dumps({"module": label, "test": name, "ok": ok, "detail": detail[:400]}))


def main() -> int:
    repo = sys.argv[1]
    pairs = sys.argv[2:]
    sys.path.insert(0, repo)
    for label, path in zip(pairs[0::2], pairs[1::2]):
        try:
            run_module(label, path)
        except BaseException as exc:  # a broken import is a failed test, not a crash
            detail = f"{type(exc).__name__}: {exc}"[:400]
            print(json.dumps({"module": label, "test": f"{label}.<import>", "ok": False, "detail": detail}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# --------------------------------------------------------------------------- #
# Template filling and fixture arithmetic
# --------------------------------------------------------------------------- #


def _fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace(f"@{key}@", value)
    return template


def _render_files(files: dict[str, str]) -> str:
    blocks = []
    for rel, content in files.items():
        lang = "markdown" if rel.endswith(".md") else "python"
        blocks.append(f"### {rel}\n\n```{lang}\n{content}```\n")
    return "\n".join(blocks)


def _defect_line(source: str, marker: str) -> int:
    """Line number of the defect, taken from the fixture, never hard-coded."""
    hits = [i for i, line in enumerate(source.splitlines(), 1) if line == marker]
    assert len(hits) == 1, f"маркер дефекта {marker!r} встречается {len(hits)} раз"
    return hits[0]


def _assert_single_line_fix(task_id: str, buggy: str, reference: str, line: int) -> None:
    """The reported line really is the defect line: the only one that changes."""
    before, after = buggy.splitlines(), reference.splitlines()
    assert len(before) == len(after), f"{task_id}: правка меняет число строк файла"
    changed = [i for i, (a, b) in enumerate(zip(before, after), 1) if a != b]
    assert changed == [line], f"{task_id}: правка затрагивает строки {changed}, ожидалась только {line}"


# --------------------------------------------------------------------------- #
# Hidden tests: run them in a throwaway copy of the repository
# --------------------------------------------------------------------------- #


def _run_modules(
    repo: Path, sandbox: Path, modules: list[tuple[str, Path]]
) -> tuple[dict[str, dict[str, tuple[bool, str]]], str]:
    """Run the modules and return ``({label: {test: (ok, detail)}}, note)``."""
    runner = sandbox / "run_review_tests.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    argv = [sys.executable, str(runner), str(repo)]
    for label, path in modules:
        argv += [label, str(path)]
    empty = {label: {} for label, _ in modules}
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_HIDDEN_TIMEOUT_S,
            cwd=str(sandbox),
        )
    except subprocess.TimeoutExpired:
        return empty, f"тесты не уложились в {_HIDDEN_TIMEOUT_S} с"
    except OSError as exc:
        return empty, f"не удалось запустить python: {exc}"

    rows: dict[str, dict[str, tuple[bool, str]]] = empty
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = str(row.get("module", ""))
        if label in rows:
            rows[label][str(row.get("test", ""))] = (bool(row.get("ok")), str(row.get("detail", "")))
    note = ""
    if any(not group for group in rows.values()):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        note = f"тесты не выполнились: {tail[-1][:200] if tail else 'нет вывода'}"
    return rows, note


def _run_test_modules(
    task: core.TaskInstance, target_source: str, labels: tuple[str, ...]
) -> tuple[dict[str, dict[str, tuple[bool, str]]], str]:
    """Materialise the workspace, overwrite the reviewed file, run the tests."""
    sandbox = Path(tempfile.mkdtemp(prefix=f"dailybench-{task.id.lower()}-review-"))
    repo = sandbox / "repo"
    repo.mkdir(parents=True)
    for rel, content in (task.workspace or {}).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (repo / task.oracle["target_file"]).write_text(target_source, encoding="utf-8")

    modules: list[tuple[str, Path]] = []
    for label in labels:
        if label == "hidden":
            path = sandbox / "test_review_hidden.py"
            path.write_text(task.oracle["hidden_tests"], encoding="utf-8")
        else:
            path = repo / task.oracle["public_test_file"]
        modules.append((label, path))
    # The hidden module runs first: that is exactly what ``evaluate`` sees, so a
    # fixture claim proved here holds during scoring too.
    return _run_modules(repo, sandbox, modules)


def _hidden_summary(results: dict[str, tuple[bool, str]]) -> tuple[bool, str]:
    if not results:
        return False, "скрытые тесты не выполнились"
    failures = sorted((name, detail) for name, (ok, detail) in results.items() if not ok)
    if failures:
        name, detail = failures[0]
        return False, (
            f"{len(results) - len(failures)}/{len(results)} скрытых тестов прошли; "
            f"первый провал {name}: {detail[:200]}"
        )
    return True, f"все {len(results)} скрытых тестов прошли"


def _verify_fixture(task: core.TaskInstance, source: str, *, expect_hidden_pass: bool) -> None:
    """Assert a claim the scenario makes about its own fixture."""
    rows, note = _run_test_modules(task, source, ("hidden", "public"))
    assert not note, f"{task.id}: {note}"
    public = rows.get("public", {})
    assert public, f"{task.id}: публичные тесты не выполнились"
    assert all(ok for ok, _ in public.values()), f"{task.id}: публичные тесты упали: {public}"
    hidden = rows.get("hidden", {})
    assert hidden, f"{task.id}: скрытые тесты не выполнились"
    failed = sorted(name for name, (ok, _) in hidden.items() if not ok)
    if expect_hidden_pass:
        assert not failed, f"{task.id}: эталон провалил скрытые тесты {failed}"
    else:
        assert failed, f"{task.id}: дефектный файл не провалил ни одного скрытого теста"


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

_J1_MODULES = ("pipeline/window.py", "metrics/window.py", "pipeline/sliding.py")
_J1_FUNCS = ("tail", "last_values", "window")
_J1_PREFIXES = ("svc.", "metrics.", "pipe.")
_J2_MODULES = ("ingest/paths.py", "ingest/locations.py", "sink/paths.py")
_J2_FUNCS = ("resolve", "join_path", "full_path")


def _scenario_j1(rng: random.Random) -> core.TaskInstance:
    module_path = rng.choice(_J1_MODULES)
    package, _, filename = module_path.partition("/")
    dotted = f"{package}.{filename[:-3]}"
    func = rng.choice(_J1_FUNCS)
    prefix = rng.choice(_J1_PREFIXES)
    sample = [rng.randrange(1, 90) for _ in range(3)]

    buggy = _fill(_J1_WINDOW, FUNC=func, RETURN="-count - 1 :")
    reference = _fill(_J1_WINDOW, FUNC=func, RETURN="-count :")
    marker = "    return values[-count - 1 :]"
    line = _defect_line(buggy, marker)
    _assert_single_line_fix("J-1", buggy, reference, line)

    helper = _fill(_J1_HELPER, PREFIX=prefix)
    public_test = f"tests/test_{filename[:-3]}.py"
    public = _fill(
        _J1_PUBLIC_TEST, DOTTED=dotted, FUNC=func, SAMPLE=", ".join(str(value) for value in sample)
    )
    hidden = _fill(_J1_HIDDEN_TESTS, DOTTED=dotted, FUNC=func)

    workspace = {
        "README.md": f"# {package}\n\nSmall helpers shared by the reporting service: labels and window slices.\n",
        f"{package}/__init__.py": f'"""{package} package."""\n',
        f"{package}/labels.py": helper,
        module_path: buggy,
        public_test: public,
    }
    task = core.TaskInstance(
        id="J-1",
        category="J",
        title="Code review",
        prompt=_PROMPT.format(files=_render_files(workspace), public_test=public_test),
        max_points=_POINTS["J-1"],
        workspace=workspace,
        oracle={
            "reference": _reference_reply(module_path, line, reference),
            "reference_source": reference,
            "target_file": module_path,
            "line": line,
            "hidden_tests": hidden,
            "public_test_file": public_test,
            "defect": "граница окна: срез начинается на элемент раньше",
            "checks": ["defect_located", "fix_passes_hidden_tests"],
        },
        meta={"scenario": "J-1"},
    )
    _verify_fixture(task, buggy, expect_hidden_pass=False)
    _verify_fixture(task, reference, expect_hidden_pass=True)
    return task


def _scenario_j2(rng: random.Random) -> core.TaskInstance:
    module_path = rng.choice(_J2_MODULES)
    package, _, filename = module_path.partition("/")
    dotted = f"{package}.{filename[:-3]}"
    func = rng.choice(_J2_FUNCS)
    shard = rng.randrange(64, 4096)

    buggy = _fill(_J2_PATHS, FUNC=func, KEY="name")
    reference = _fill(_J2_PATHS, FUNC=func, KEY="(root, name)")
    marker = "    key = name"
    line = _defect_line(buggy, marker)
    _assert_single_line_fix("J-2", buggy, reference, line)

    decoy = _fill(_J2_DECOY, SHARD=str(shard))
    public_test = f"tests/test_{filename[:-3]}.py"
    public = _fill(_J2_PUBLIC_TEST, DOTTED=dotted, FUNC=func)
    hidden = _fill(_J2_HIDDEN_TESTS, DOTTED=dotted, FUNC=func)

    workspace = {
        "README.md": f"# {package}\n\nRecord ingest service: shard configuration and path joining.\n",
        f"{package}/__init__.py": f'"""{package} package."""\n',
        f"{package}/config.py": decoy,
        module_path: buggy,
        public_test: public,
    }
    task = core.TaskInstance(
        id="J-2",
        category="J",
        title="Code review",
        prompt=_PROMPT.format(files=_render_files(workspace), public_test=public_test),
        max_points=_POINTS["J-2"],
        workspace=workspace,
        oracle={
            "reference": _reference_reply(module_path, line, reference),
            "reference_source": reference,
            "target_file": module_path,
            "line": line,
            "hidden_tests": hidden,
            "public_test_file": public_test,
            "defect": "ключ кэша собран из части аргументов: повторный вызов с другим root отдаёт чужой результат",
            "checks": ["defect_located", "fix_passes_hidden_tests"],
        },
        meta={"scenario": "J-2"},
    )
    _verify_fixture(task, buggy, expect_hidden_pass=False)
    _verify_fixture(task, reference, expect_hidden_pass=True)
    return task


def _reference_reply(target_file: str, line: int, source: str) -> str:
    return json.dumps({"file": target_file, "line": line, "fix": source}, ensure_ascii=False)


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    return [_scenario_j1(rng), _scenario_j2(rng)]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _same_path(value: Any, target: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().replace("\\", "/").lstrip("./") == target


def _as_line(value: Any) -> int | None:
    """The reported line as an integer; a quoted number is still the same line."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    return None


def extract_fix(payload: Any) -> str | None:
    """The corrected file from the reply: a fenced block when present, else the string."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("fix")
    if not isinstance(raw, str) or not raw.strip():
        return None
    block = text.first_fenced_block(raw)
    if block:
        return block
    candidate = text.strip_fence(raw)[0]
    return candidate or None


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="J")
    target = str(task.oracle["target_file"])
    defect_line = int(task.oracle["line"])

    payload, strict = text.strict_json(tr.final_text)
    if not strict:
        payload = text.loose_json(tr.final_text)
    reported_file = payload.get("file") if isinstance(payload, dict) else None
    reported_line = _as_line(payload["line"]) if isinstance(payload, dict) and "line" in payload else None

    located = _same_path(reported_file, target) and reported_line == defect_line
    score.add(
        "defect_located",
        located,
        _LOCATION_POINTS,
        f"получено file={reported_file!r} line={reported_line!r}, "
        f"ожидалось file={target!r} line={defect_line}",
    )

    fix_points = task.max_points - _LOCATION_POINTS
    fix = extract_fix(payload)
    if fix is None:
        score.add("fix_passes_hidden_tests", False, fix_points, "в ответе нет полного исправленного файла")
        return score

    rows, note = _run_test_modules(task, fix, ("hidden",))
    if note:
        score.add("fix_passes_hidden_tests", False, fix_points, note)
        return score
    passed, detail = _hidden_summary(rows.get("hidden", {}))
    score.add("fix_passes_hidden_tests", passed, fix_points, detail)
    if not strict:
        score.notes.append("ответ не является строгим JSON целиком")
    return score


core.register(
    key="J",
    title="Code review",
    points=5.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=IDS,
)
