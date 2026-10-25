"""Pinned BigCodeBench-Hard adapter with the upstream scorer.

Identity is fixed in code: the ``bigcode/bigcodebench-hard`` dataset at the
``v0.1.4`` split of revision ``298d2cc7…``, ``instruct`` prompts only, and the
upstream ``bigcodebench==0.2.5`` evaluator with local execution. Only a
selected list of upstream task ids is sent to the scorer
(``--selective_evaluate``) and every task carries exactly one sample: the
README's greedy/batch-size warning makes batch size 1 part of the protocol.

The verdict is the upstream ``pass@1`` from ``eval_results.json``. It is stored
as a ratio (``0.5``), never multiplied into the native ``/100``. Upstream task
ids are preserved verbatim. When the scorer package, its pinned version or
docker is missing, the bundle says ``unverified`` and the metric stays ``None``
instead of collapsing to ``0``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ADAPTER_NAME = "bigcodebench"

#: The upstream dataset. ``v0.1.4`` is a HF split marker, so the git revision
#: is pinned separately by its commit sha: the marker alone can be repointed.
DATASET_ID = "bigcode/bigcodebench-hard"
DATASET_REVISION_LABEL = "v0.1.4"
DATASET_REVISION = "298d2cc7b96612e15e47313c3603ee124cee0c1f"
DATASET_SPLIT = "v0.1.4"
UPSTREAM_SPLIT = "instruct"
SUBSET = "hard"
PROMPT_FIELD = "instruct_prompt"

SCORER_PACKAGE = "bigcodebench"
#: First upstream release that both pins dataset ``v0.1.4`` and accepts
#: ``--execution`` / ``--selective_evaluate``. ``0.1.8.post1`` predates both:
#: it still loads ``v0.1.0_hf`` and its CLI only knows ``--split/--subset``.
SCORER_VERSION = "0.2.5"
SCORER_COMMAND = (sys.executable, "-m", "bigcodebench.evaluate")
SCORER_PROTOCOL = (
    "bigcodebench.evaluate --samples <samples.jsonl> --execution local "
    "--selective_evaluate <upstream ids> --split instruct --subset hard "
    "--parallel 1 --save_pass_rate False --calibrated False"
)
#: ``calibrated False``: samples are raw instruct answers, not the sanitized
#: calibrated bodies the leaderboard upload uses. Saved pass@k tables are
#: disabled because they prompt interactively; ``pass@1`` is computed here
#: from the per-task statuses of ``eval_results.json``.
DEFAULT_TIMEOUT_S = 6 * 60 * 60

LICENSE = {
    "spdx": "Apache-2.0",
    "applies_to": ["dataset", "scorer-code"],
    "record": "docs/licenses/bigcodebench.json",
    "sources": [
        "https://huggingface.co/datasets/bigcode/bigcodebench-hard",
        "https://github.com/bigcode-project/bigcodebench/blob/v0.2.5/LICENSE",
    ],
}

_PASS = "pass"
_STATUSES = frozenset({_PASS, "fail", "timeout"})


class AdapterError(RuntimeError):
    """The adapter cannot produce a verdict from the given inputs."""


@dataclass(frozen=True)
class Task:
    """One pinned upstream task, with the id kept exactly as upstream wrote it."""

    task_id: str
    prompt: str
    entry_point: str = ""
    canonical_solution: str = ""
    libs: str = ""
    doc_struct: str = ""

    def record(self) -> dict[str, str]:
        payload = {
            "task_id": self.task_id,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "entry_point": self.entry_point,
            "canonical_solution_sha256": hashlib.sha256(
                self.canonical_solution.encode("utf-8")
            ).hexdigest(),
            "libs": self.libs,
        }
        if self.doc_struct:
            payload["doc_struct"] = self.doc_struct
        return payload


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _task_from_row(row: Mapping[str, Any]) -> Task:
    task_id = str(row.get("task_id") or "").strip()
    prompt = row.get(PROMPT_FIELD) or row.get("prompt")
    if not task_id:
        raise AdapterError("у записи датасета нет task_id")
    if not isinstance(prompt, str) or not prompt.strip():
        raise AdapterError(f"у задачи {task_id} пустой {PROMPT_FIELD}")
    return Task(
        task_id=task_id,
        prompt=prompt,
        entry_point=str(row.get("entry_point") or ""),
        canonical_solution=str(row.get("canonical_solution") or ""),
        libs=str(row.get("libs") or ""),
        doc_struct=str(row.get("doc_struct") or ""),
    )


def _load_hf_tasks() -> list[Task]:
    """Read the pinned upstream split. ``datasets`` comes with the scorer."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised without datasets
        raise AdapterError(
            "для загрузки датасета нужен пакет datasets (ставится с bigcodebench)"
        ) from exc
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split=DATASET_SPLIT)
    return [_task_from_row(row) for row in dataset]


def _load_local_tasks(path: Path) -> list[Task]:
    """Fixture/local tasks: a JSON array or JSONL of the same fields."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise AdapterError(f"{path}: ожидается JSON-массив задач")
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [_task_from_row(row) for row in rows]


def load_tasks(
    *,
    limit: int = 0,
    tasks_path: Path | None = None,
) -> list[Task]:
    """Load the pinned split and cut a deterministic prefix.

    Order is the upstream order, so the same ``limit`` always selects the same
    tasks. ``limit <= 0`` means the whole split.
    """
    if limit < 0:
        raise AdapterError("limit не может быть отрицательным")
    tasks = (
        _load_local_tasks(tasks_path) if tasks_path is not None else _load_hf_tasks()
    )
    if not tasks:
        raise AdapterError("датасет не дал ни одной задачи")
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise AdapterError(f"повторяющийся upstream task_id: {task.task_id}")
        seen.add(task.task_id)
    return tasks[:limit] if limit else tasks


# --------------------------------------------------------------------------- #
# Readiness and samples
# --------------------------------------------------------------------------- #


def readiness() -> list[str]:
    """Reasons the pinned scorer cannot run here; an empty list means ready."""
    missing: list[str] = []
    if importlib.util.find_spec(SCORER_PACKAGE) is None:
        missing.append(f"нет пакета {SCORER_PACKAGE}=={SCORER_VERSION}")
    else:  # pragma: no cover - needs the upstream package installed
        try:
            installed = importlib.metadata.version(SCORER_PACKAGE)
        except importlib.metadata.PackageNotFoundError:
            installed = "unknown"
        if installed != SCORER_VERSION:
            missing.append(
                f"установлен {SCORER_PACKAGE}=={installed}, нужен {SCORER_VERSION}"
            )
    if shutil.which("docker") is None:
        missing.append("нет docker для локального исполнения")
    return missing


def read_predictions(path: Path) -> dict[str, str]:
    """Read ``{"task_id", "solution"}`` rows; one sample per task."""
    predictions: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{path}:{number}: не JSON ({exc})") from exc
        task_id = str(row.get("task_id") or "").strip()
        solution = row.get("solution")
        if not task_id or not isinstance(solution, str):
            raise AdapterError(f"{path}:{number}: нужны строковые task_id и solution")
        if task_id in predictions:
            raise AdapterError(
                f"{task_id}: несколько сэмплов; протокол адаптера — batch size 1"
            )
        predictions[task_id] = solution
    return predictions


def write_samples(path: Path, tasks: Sequence[Task], predictions: Mapping[str, str]) -> None:
    """Write exactly one upstream sample line per selected task."""
    expected = [task.task_id for task in tasks]
    missing = [task_id for task_id in expected if task_id not in predictions]
    extra = sorted(set(predictions) - set(expected))
    if missing:
        raise AdapterError("нет сэмплов для задач: " + ", ".join(missing))
    if extra:
        raise AdapterError("сэмплы для задач вне выборки: " + ", ".join(extra))
    rows = [
        {"task_id": task_id, "solution": predictions[task_id]} for task_id in expected
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def evaluate_arguments(samples: Path, task_ids: Sequence[str]) -> list[str]:
    """Upstream arguments after the executable prefix."""
    return [
        "--samples",
        str(samples),
        "--execution",
        "local",
        "--selective_evaluate",
        ",".join(task_ids),
        "--split",
        UPSTREAM_SPLIT,
        "--subset",
        SUBSET,
        "--parallel",
        "1",
        "--save_pass_rate",
        "False",
        "--calibrated",
        "False",
    ]


def evaluate_command(samples: Path, task_ids: Sequence[str]) -> list[str]:
    """The full pinned upstream argv, with the selection passed explicitly."""
    return [*SCORER_COMMAND, *evaluate_arguments(samples, task_ids)]


def results_path(samples: Path) -> Path:
    """Result path the pinned upstream derives from the samples path."""
    if not str(samples).endswith(".jsonl"):
        raise AdapterError("файл сэмплов должен оканчиваться на .jsonl")
    return Path(str(samples).replace(".jsonl", "_eval_results.json"))


def parse_results(path: Path, task_ids: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute upstream ``pass@1`` from per-task statuses.

    Strict: every selected task must occur exactly once and carry a known
    upstream status. A partial or foreign result file yields ``AdapterError``,
    which the caller records as ``unverified``.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"не читается {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("eval"), dict):
        raise AdapterError(f"{path}: нет объекта eval")
    results = document["eval"]

    unexpected = sorted(set(results) - set(task_ids))
    if unexpected:
        raise AdapterError("в результате чужие task_id: " + ", ".join(unexpected))

    verdicts: list[dict[str, Any]] = []
    passed = 0
    for task_id in task_ids:
        rows = results.get(task_id)
        if not isinstance(rows, list) or not rows:
            raise AdapterError(f"нет результата для задачи {task_id}")
        if len(rows) != 1:
            raise AdapterError(
                f"{task_id}: {len(rows)} сэмплов, pass@1 требует ровно один"
            )
        row = rows[0]
        status = str(row.get("status") or "")
        if status not in _STATUSES:
            raise AdapterError(f"{task_id}: неизвестный статус {status!r}")
        passed += status == _PASS
        verdicts.append(
            {
                "task_id": task_id,
                "status": status,
                "details": row.get("details") or {},
            }
        )

    metric = {
        "name": "pass@1",
        "value": passed / len(task_ids),
        "numerator": passed,
        "denominator": len(task_ids),
        "scale": "ratio",
        "unit": "ratio",
        "percent_of_100": False,
        "source": "eval_results.json:status==pass",
    }
    return metric, verdicts


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


def bundle_stem(*, tasks: Sequence[Task]) -> dict[str, Any]:
    """Identity shared by a prepared and a scored bundle."""
    return {
        "adapter": ADAPTER_NAME,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION_LABEL,
            "revision_sha": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "prompt_field": PROMPT_FIELD,
            "upstream_split": UPSTREAM_SPLIT,
            "subset": SUBSET,
            "task_ids": [task.task_id for task in tasks],
        },
        "scorer": {
            "package": SCORER_PACKAGE,
            "version": SCORER_VERSION,
            "protocol": SCORER_PROTOCOL,
            "execution": "local",
            "batch_size": 1,
        },
        "license": LICENSE,
        "tasks": [task.record() for task in tasks],
        "native_100": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_bundle(out: Path) -> dict[str, Any]:
    path = out / "bundle.json"
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"не читается подготовленный bundle {path}: {exc}") from exc
    if bundle.get("adapter") != ADAPTER_NAME:
        raise AdapterError(f"{path}: это не bundle {ADAPTER_NAME}")
    if not bundle.get("dataset", {}).get("task_ids"):
        raise AdapterError(f"{path}: пустой список task_ids")
    return bundle


def prepare(
    *,
    out: Path,
    limit: int = 0,
    tasks_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze the selection, prompts and identity; do not call the model."""
    from ..holdout import load_record_by_id

    try:
        load_record_by_id(ADAPTER_NAME)
    except ValueError as exc:
        raise AdapterError(f"импорт отклонён: {exc}") from exc
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(limit=limit, tasks_path=tasks_path)
    bundle = bundle_stem(tasks=tasks)
    bundle["selection"] = {
        "order": "upstream",
        "limit": limit,
        "source": str(tasks_path) if tasks_path is not None else DATASET_ID,
    }
    bundle["status"] = "prepared"
    bundle["reason"] = ""
    bundle["metric"] = None
    bundle["verdicts"] = []

    (out / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "entry_point": task.entry_point,
                    "canonical_solution": task.canonical_solution,
                    "libs": task.libs,
                    "doc_struct": task.doc_struct,
                }
                for task in tasks
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "predictions.example.jsonl").write_text(
        "".join(
            json.dumps(
                {"task_id": task.task_id, "solution": task.canonical_solution},
                ensure_ascii=False,
            )
            + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )
    _write_json(out / "bundle.json", bundle)
    return bundle


def _read_bundle_tasks(out: Path) -> tuple[dict[str, Any], list[Task]]:
    bundle = _read_bundle(out)
    rows = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    by_id = {str(row["task_id"]): row for row in rows}
    tasks: list[Task] = []
    for task_id in bundle["dataset"]["task_ids"]:
        row = by_id.get(task_id)
        if row is None:
            raise AdapterError(f"{out}/tasks.json потерял задачу {task_id}")
        tasks.append(_task_from_row(row))
    return bundle, tasks


def score(
    *,
    out: Path,
    samples: Path,
    evaluate_cmd: Sequence[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the pinned scorer and store its external metric.

    ``evaluate_cmd`` replaces the default argv prefix for tests or wrappers; an
    explicit override also skips the local readiness probe, because the caller
    then owns the environment.
    """
    bundle, tasks = _read_bundle_tasks(out)
    bundle["status"] = "unverified"
    bundle["reason"] = ""
    bundle["metric"] = None
    bundle["verdicts"] = []
    bundle["upstream_results"] = ""
    bundle.pop("next", None)

    try:
        predictions = read_predictions(samples)
    except AdapterError as exc:
        bundle["reason"] = str(exc)
        _write_json(out / "bundle.json", bundle)
        return bundle

    try:
        write_samples(out / "samples.jsonl", tasks, predictions)
    except AdapterError as exc:
        bundle["reason"] = str(exc)
        _write_json(out / "bundle.json", bundle)
        return bundle

    missing = [] if evaluate_cmd else readiness()
    if missing:
        bundle["reason"] = "; ".join(missing)
        bundle["next"] = (
            f"установить {SCORER_PACKAGE}=={SCORER_VERSION} и docker, "
            "затем повторить score"
        )
        _write_json(out / "bundle.json", bundle)
        return bundle

    task_ids = [t.task_id for t in tasks]
    command = (
        [*evaluate_cmd, *evaluate_arguments(out / "samples.jsonl", task_ids)]
        if evaluate_cmd
        else evaluate_command(out / "samples.jsonl", task_ids)
    )
    bundle["scorer"]["command"] = command
    _write_json(out / "bundle.json", bundle)

    # The upstream CLI asks Y/N when its result file already exists. Removing
    # the stale file keeps a rerun unconditional; stdin is closed anyway.
    upstream_results = results_path(out / "samples.jsonl")
    upstream_results.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            command,
            cwd=out,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        bundle["reason"] = f"scorer не запустился: {exc}"
        _write_json(out / "bundle.json", bundle)
        return bundle
    (out / "evaluate.log").write_text(
        "$ " + shlex.join(command) + "\n\n" + proc.stdout + "\n" + proc.stderr,
        encoding="utf-8",
    )

    if proc.returncode != 0:
        bundle["reason"] = f"scorer вышел с кодом {proc.returncode}; см. evaluate.log"
        _write_json(out / "bundle.json", bundle)
        return bundle
    if not upstream_results.is_file():
        bundle["reason"] = f"scorer не создал {upstream_results.name}; см. evaluate.log"
        _write_json(out / "bundle.json", bundle)
        return bundle

    try:
        metric, verdicts = parse_results(upstream_results, [t.task_id for t in tasks])
    except AdapterError as exc:
        bundle["reason"] = str(exc)
        _write_json(out / "bundle.json", bundle)
        return bundle

    bundle["status"] = "verified"
    bundle["metric"] = metric
    bundle["verdicts"] = verdicts
    bundle["upstream_results"] = upstream_results.name
    _write_json(out / "bundle.json", bundle)
    return bundle


# --------------------------------------------------------------------------- #
# CLI entry points
# --------------------------------------------------------------------------- #


def _print_bundle(bundle: Mapping[str, Any]) -> None:
    print(f"status: {bundle.get('status')}")
    if bundle.get("reason"):
        print(f"reason: {bundle['reason']}")
    metric = bundle.get("metric")
    if metric:
        print(
            "pass@1: "
            f"{metric['numerator']}/{metric['denominator']} = {metric['value']:.3f} "
            "(external ratio, не /100)"
        )
    print("tasks: " + ", ".join(bundle["dataset"]["task_ids"]))


def cmd_prepare(args: Any) -> int:
    out = Path(args.out)
    try:
        bundle = prepare(
            out=out,
            limit=args.limit,
            tasks_path=Path(args.tasks) if args.tasks else None,
        )
    except (AdapterError, OSError, json.JSONDecodeError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    print(f"подготовлено {len(bundle['dataset']['task_ids'])} задач: {out / 'bundle.json'}")
    return 0


def cmd_score(args: Any) -> int:
    evaluate_cmd = shlex.split(args.evaluate_cmd) if args.evaluate_cmd else None
    try:
        bundle = score(
            out=Path(args.out),
            samples=Path(args.samples),
            evaluate_cmd=evaluate_cmd,
            timeout=args.timeout,
        )
    except (AdapterError, OSError, json.JSONDecodeError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    _print_bundle(bundle)
    return 0 if bundle["status"] == "verified" else 1


def add_subcommands(sub: Any) -> None:
    """Register ``dailybench adapter bigcodebench …``."""
    adapter = sub.add_parser(
        "adapter",
        help="официальный upstream-набор: pinned identity, внешняя метрика отдельно",
    )
    families = adapter.add_subparsers(dest="adapter_name", required=True)
    bcb = families.add_parser(
        ADAPTER_NAME,
        help="BigCodeBench-Hard instruct, upstream scorer, pass@1 не переводится в /100",
    )
    actions = bcb.add_subparsers(dest="adapter_action", required=True)

    prep = actions.add_parser(
        "prepare",
        help="заморозить pinned выборку и промпты (без вызова модели)",
    )
    prep.add_argument("--out", required=True, help="каталог bundle")
    prep.add_argument(
        "--limit",
        type=int,
        default=0,
        help="сколько первых задач upstream взять; 0 — весь hard split (148)",
    )
    prep.add_argument("--tasks", default="", help="локальный JSONL/JSON вместо HF (тесты)")
    prep.set_defaults(func=cmd_prepare)

    scored = actions.add_parser(
        "score",
        help="прогнать pinned upstream evaluator по сэмплам и записать pass@1",
    )
    scored.add_argument("--out", required=True, help="каталог, подготовленный prepare")
    scored.add_argument("--samples", required=True, help="JSONL: task_id, solution")
    scored.add_argument(
        "--evaluate-cmd",
        default="",
        help="argv-префикс вместо bigcodebench.evaluate (обход readiness-проверки)",
    )
    scored.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    scored.set_defaults(func=cmd_score)
