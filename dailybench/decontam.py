"""Contamination control for the public suite layer.

Two independent checks live here:

* an n-gram overlap filter that compares every task prompt against reference
  corpora (a 13-gram match means a whole long phrase survived verbatim), and
* the private-layer gate: private tasks never leave the machine as prompt text
  and the ``stretch`` profile must keep a fifth of every A–J category private.

The n-gram filter is deliberately mechanical: normalise, tokenise, shingle,
count the intersection. A paraphrase that keeps long phrases lands in the
middle of the scale, which is exactly the signal the threshold is for. Nothing
here decides *why* a text is similar; it only measures and reports.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

NGRAM_SIZE = 13
"""Words per shingle. Long enough that a shared phrase is not a coincidence."""

CONTAMINATED = 0.5
"""Overlap share at which a task is called contaminated."""

PRIVATE_CATEGORIES = tuple("ABCDEFGHIJ")
"""Categories covered by the private-layer gate."""

STRETCH_PRIVATE_MIN = 0.20
"""Minimum private share per category for ``stretch-v1``."""

STRETCH_PROFILE = "stretch-v1"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def normalize(text: str) -> str:
    """Fold a text into the only form the shingler compares.

    NFC, lower case, collapsed whitespace: composition form and case must never
    hide a duplicate. Punctuation is dropped later by the tokeniser.
    """
    folded = unicodedata.normalize("NFC", text or "").lower()
    return _WS_RE.sub(" ", folded).strip()


def shingles(text: str, n: int = NGRAM_SIZE) -> set[tuple[str, ...]]:
    """Distinct word n-grams of a normalised text.

    A text shorter than ``n`` yields one shingle of all its words, so an exact
    short duplicate still matches; it never matches a differently worded text.
    """
    tokens = _TOKEN_RE.findall(normalize(text))
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def corpus_shingles(texts: Iterable[str], n: int = NGRAM_SIZE) -> set[tuple[str, ...]]:
    """Union of the shingles of every reference document."""
    result: set[tuple[str, ...]] = set()
    for text in texts:
        result |= shingles(text, n)
    return result


def read_corpus(paths: Iterable[Path | str]) -> list[str]:
    """Read reference documents from disk (UTF-8)."""
    return [Path(path).read_text(encoding="utf-8") for path in paths]


# --------------------------------------------------------------------------- #
# n-gram report
# --------------------------------------------------------------------------- #


@dataclass
class TaskText:
    """Minimal task shape for prompts read straight from files."""

    id: str
    prompt: str


@dataclass
class TaskOverlap:
    task_id: str
    shingles: int
    matched: int
    threshold: float = CONTAMINATED

    @property
    def share(self) -> float:
        return self.matched / self.shingles if self.shingles else 0.0

    @property
    def contaminated(self) -> bool:
        return self.share >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "shingles": self.shingles,
            "matched": self.matched,
            "share": round(self.share, 4),
            "contaminated": self.contaminated,
        }


@dataclass
class DecontamReport:
    ngram: int = NGRAM_SIZE
    threshold: float = CONTAMINATED
    corpus_documents: int = 0
    rows: list[TaskOverlap] = field(default_factory=list)

    @property
    def affected_ids(self) -> list[str]:
        return [row.task_id for row in self.rows if row.matched]

    @property
    def contaminated_ids(self) -> list[str]:
        return [row.task_id for row in self.rows if row.contaminated]

    @property
    def contaminated(self) -> bool:
        return bool(self.contaminated_ids)

    @property
    def share(self) -> float:
        total = sum(row.shingles for row in self.rows)
        return sum(row.matched for row in self.rows) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ngram": self.ngram,
            "threshold": self.threshold,
            "corpus_documents": self.corpus_documents,
            "overlap_share": round(self.share, 4),
            "affected_ids": self.affected_ids,
            "contaminated_ids": self.contaminated_ids,
            "verdict": "contaminated" if self.contaminated else "clean",
            "tasks": [row.to_dict() for row in self.rows],
        }

    def render(self) -> str:
        lines = [
            f"Фильтр {self.ngram}-грамм против {self.corpus_documents} документов корпуса, "
            f"порог {self.threshold:.2f}: {'КОНТАМИНАЦИЯ' if self.contaminated else 'чисто'}",
            f"Перекрытие: {self.share:.1%}; задето заданий: {len(self.affected_ids)}",
        ]
        for row in self.rows:
            mark = "ДУБЛЬ" if row.contaminated else ("частично" if row.matched else "—")
            lines.append(f"{row.task_id}: {row.share:.1%} ({row.matched}/{row.shingles}) {mark}")
        if self.contaminated_ids:
            lines.append("Загрязнены: " + ", ".join(self.contaminated_ids))
        return "\n".join(lines)


def check_corpus(
    tasks: Iterable[Any],
    corpus: Iterable[str],
    n: int = NGRAM_SIZE,
    threshold: float = CONTAMINATED,
) -> DecontamReport:
    """Measure every task prompt against the reference shingles."""
    if n < 1:
        raise ValueError("ngram must be >= 1")
    documents = list(corpus)
    reference = corpus_shingles(documents, n)
    report = DecontamReport(ngram=n, threshold=threshold, corpus_documents=len(documents))
    for task in tasks:
        own = shingles(task.prompt, n)
        report.rows.append(
            TaskOverlap(
                task_id=task.id,
                shingles=len(own),
                matched=len(own & reference),
                threshold=threshold,
            )
        )
    return report


def check_corpus_files(
    tasks: Iterable[Any],
    paths: Iterable[Path | str],
    n: int = NGRAM_SIZE,
    threshold: float = CONTAMINATED,
) -> DecontamReport:
    paths = list(paths)
    return check_corpus(tasks, read_corpus(paths), n=n, threshold=threshold)


# --------------------------------------------------------------------------- #
# Private layer
# --------------------------------------------------------------------------- #


def is_private(task: Any) -> bool:
    return bool(getattr(task, "private", False))


def publishable(tasks: Iterable[Any]) -> list[Any]:
    """Tasks that may carry prompt text in a public export."""
    return [task for task in tasks if not is_private(task)]


def prompt_hashes(task: Any) -> dict[str, str]:
    """Fingerprints published for a private task instead of its text.

    Accepts the task or its serialised document, so a mapping and a stub whose
    ``prompt`` is already ``None`` hash exactly what the export publishes.
    """
    from .schema import task_to_dict

    doc = task if isinstance(task, Mapping) else task_to_dict(task)

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
        ).hexdigest()

    return {
        "prompt_sha256": hashlib.sha256((doc.get("prompt") or "").encode()).hexdigest(),
        "dialogue_sha256": digest(doc.get("dialogue") or []),
        "workspace_sha256": digest(doc.get("workspace") or {}),
        "tools_sha256": digest(doc.get("tools") or []),
    }


@dataclass
class PrivateGateRow:
    category: str
    total: int
    private: int

    @property
    def share(self) -> float:
        return self.private / self.total if self.total else 0.0


@dataclass
class PrivateGateReport:
    profile: str
    required: float = STRETCH_PRIVATE_MIN
    rows: list[PrivateGateRow] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(row.total for row in self.rows)

    @property
    def private(self) -> int:
        return sum(row.private for row in self.rows)

    @property
    def share(self) -> float:
        return self.private / self.total if self.total else 0.0

    @property
    def gated(self) -> bool:
        return self.profile == STRETCH_PROFILE

    @property
    def failed(self) -> list[str]:
        """Categories below the mark, the same ones ``render`` marks."""
        if not self.gated:
            return []
        return [row.category for row in self.rows if row.share < self.required]

    @property
    def ok(self) -> bool:
        """Every category must clear the mark: a high overall share does not
        excuse a category with nothing private."""
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "gated": self.gated,
            "required": self.required,
            "share": round(self.share, 4),
            "failed_categories": self.failed,
            "verdict": "ok" if self.ok else "failed",
            "categories": {
                row.category: {
                    "total": row.total,
                    "private": row.private,
                    "share": round(row.share, 4),
                }
                for row in self.rows
            },
        }

    def render(self) -> str:
        head = (
            f"Приватный слой, профиль {self.profile}: {self.share:.1%} "
            f"(нужно ≥{self.required:.0%} на категорию A–J)"
            if self.gated
            else f"Приватный слой, профиль {self.profile}: {self.share:.1%} (гейт не применяется)"
        )
        lines = [head]
        for row in self.rows:
            mark = "" if not self.gated or row.share >= self.required else "  ПРОВАЛ"
            lines.append(f"{row.category}: {row.private}/{row.total} ({row.share:.0%}){mark}")
        if self.failed:
            lines.append("Не хватает приватных задач: " + ", ".join(self.failed))
        return "\n".join(lines)


def private_gate(
    tasks: Iterable[Any],
    profile: str,
    categories: Iterable[str] = PRIVATE_CATEGORIES,
    required: float = STRETCH_PRIVATE_MIN,
) -> PrivateGateReport:
    """Count private tasks per category; the gate applies to ``stretch-v1``."""
    tasks = list(tasks)
    report = PrivateGateReport(profile=profile, required=required)
    for category in categories:
        group = [task for task in tasks if task.category == category]
        report.rows.append(
            PrivateGateRow(
                category=category,
                total=len(group),
                private=sum(1 for task in group if is_private(task)),
            )
        )
    return report
# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def cmd_decontam(args: Any) -> int:
    """Run the n-gram filter against reference corpora."""
    from .cli import _config_from
    from .suite import build_suite

    if args.suite:
        tasks = build_suite(_config_from(args))
    else:
        if not args.input:
            print("нужен --suite или --input с проверяемыми файлами", file=sys.stderr)
            return 2
        tasks = [
            TaskText(
                id=str(path),
                prompt=Path(path).read_text(encoding="utf-8"),
            )
            for path in args.input
        ]
    report = check_corpus_files(tasks, args.corpus, n=args.ngram, threshold=args.threshold)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.render())
    return 1 if report.contaminated else 0


def cmd_private_check(args: Any) -> int:
    """Report (and gate) the private share of a profile."""
    from .cli import _config_from
    from .suite import build_suite

    cfg = _config_from(args)
    report = private_gate(build_suite(cfg), cfg.profile)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.render())
    return 0 if report.ok else 1


def add_subcommands(sub: Any, add_common: Any) -> None:
    """Register the contamination subcommands on the CLI parser."""
    command = sub.add_parser("decontam", help="13-граммный фильтр промптов против корпусов")
    add_common(command)
    command.add_argument("--corpus", nargs="+", required=True, help="файлы корпуса-справочника")
    command.add_argument("--suite", action="store_true", help="проверять сьют профиля целиком")
    command.add_argument("--input", nargs="*", default=[], help="файлы кандидатов без --suite")
    command.add_argument("--ngram", type=int, default=NGRAM_SIZE)
    command.add_argument("--threshold", type=float, default=CONTAMINATED)
    command.add_argument("--json", action="store_true")
    command.set_defaults(func=cmd_decontam)

    check = sub.add_parser("private-check", help="доля приватных задач A–J и гейт stretch")
    add_common(check)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_private_check)
