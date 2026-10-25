#!/usr/bin/env python3
"""manifest: манифест записи, статусы и поиск исходников (включая починки).

Манифест - список объектов ``{"model", "scene", "status"}``. Он единственный
источник того, что вообще должно быть записано: verify не доверяет каталогу
клипов, а сверяет его с манифестом. Здесь же проверки, без которых манифест
ничего не доказывает:

* пары модель/сцена уникальны, статус - из закрытого словаря ``timing``;
* ``ok`` не несёт причину, плашка - несёт;
* ``result-notes`` ссылаются на существующие строки и не врут статусом;
* починка (``source_mode="repairs"``) ищется раньше исходника и не подменяет
  его: запись в ``_repairs`` не трогает опубликованную страницу.

Поиск исходника отделён от записи: рекордер и проверка пользуются одним
правилом, поэтому «клип есть, а страницы нет» ловится до Chrome.
"""

from __future__ import annotations

import json
from pathlib import Path

from timing import STATUSES, is_placeholder, note_errors, valid_status

MODES = ("showcase", "repairs")
#: Каталог починок - рядом с опубликованными страницами, не внутри них.
REPAIRS_DIR = "_repairs"
SHOWCASE_DIR = "_showcase"


class ManifestError(ValueError):
    """Манифест не описывает дерево: запись и проверка бессмысленны."""


def load(path: Path) -> list[dict]:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"manifest unreadable: {exc}") from exc
    if not isinstance(rows, list):
        raise ManifestError("manifest must be a list")
    return rows


def pair(row: dict) -> tuple[str, str]:
    return row["model"], row["scene"]


def by_clip(rows: list[dict]) -> dict[str, dict]:
    return {clip_name(row): row for row in rows}


def clip_name(row: dict) -> str:
    return f"{row['model']}__{row['scene']}.mp4"


def check(rows: list[dict], *, notes: list[dict] | None = None) -> list[str]:
    """Все нарушения манифеста списком: дубли, статусы, причины, notes."""
    errors: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: not an object")
            continue
        model, scene = row.get("model"), row.get("scene")
        if not isinstance(model, str) or not model.strip():
            errors.append(f"row {index}: model must be a non-empty string")
        if not isinstance(scene, str) or not scene.strip():
            errors.append(f"row {index}: scene must be a non-empty string")
        if not (isinstance(model, str) and isinstance(scene, str)):
            continue
        key = (model, scene)
        if key in seen:
            errors.append(f"{model}/{scene}: duplicate pair (rows {seen[key]} and {index})")
        seen[key] = index
        status = row.get("status")
        if not valid_status(status):
            errors.append(f"{model}/{scene}: unknown status {status!r}")
        elif is_placeholder(status):
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                errors.append(f"{model}/{scene}: placeholder {status} without a reason")
        elif status == "ok" and row.get("reason"):
            errors.append(f"{model}/{scene}: ok carries a placeholder reason")
        if "repaired" in row and not isinstance(row["repaired"], bool):
            errors.append(f"{model}/{scene}: repaired must be a boolean")
    for index, note in enumerate(notes or []):
        for problem in note_errors(note):
            errors.append(f"note {index}: {problem}")
        if not isinstance(note, dict) or not (note.get("model"), note.get("scene")):
            continue
        index = seen.get((note["model"], note["scene"]))
        target = rows[index] if index is not None else None
        if target is None:
            errors.append(
                f"note {index}: {note['model']}/{note['scene']} is not in the manifest"
            )
            continue
        if target.get("status") == "ok" and note.get("originalStatus") == "failed":
            errors.append(
                f"note {index}: {note['model']}/{note['scene']} is ok but noted as failed"
            )
        if note.get("repair"):
            if not target.get("repaired"):
                errors.append(f"note {index}: repair without repaired=true in the manifest")
    return errors


def check_files(rows: list[dict], clips: Path) -> list[str]:
    """Наличие клипов и маркеров под манифест: ok - клип, плашка - маркер."""
    errors: list[str] = []
    for row in rows:
        clip = Path(clips) / clip_name(row)
        marker = clip.with_suffix(".missing")
        status = row.get("status")
        if not clip.is_file() or clip.stat().st_size == 0:
            errors.append(f"{row['model']}/{row['scene']}: clip missing (status {status})")
        if is_placeholder(status):
            if not marker.is_file():
                errors.append(f"{row['model']}/{row['scene']}: placeholder without marker")
            else:
                head = marker.read_text(encoding="utf-8").strip().split(":", 1)[0].strip()
                if head not in ("", status) and head in STATUSES:
                    errors.append(
                        f"{row['model']}/{row['scene']}: marker {head!r} != status {status!r}"
                    )
        elif marker.is_file():
            errors.append(f"{row['model']}/{row['scene']}: marker left for status {status}")
    return errors


def resolve_source(name: str, root: Path, mode: str = "showcase") -> tuple[Path, Path, bool]:
    """Страница по имени ``model/scene``: починка раньше исходника.

    Возвращает ``(page, server_root, repaired)``. ``repaired`` - страница взята
    из ``_repairs``; режим ``repairs`` без починки отдаёт исходник, чтобы
    запись не падала, а качество пометило подмену.
    """
    if mode not in MODES:
        raise ValueError(f"source mode must be one of {MODES}")
    root = Path(root).expanduser().resolve()
    parts = Path(name).parts
    if len(parts) < 2:
        raise ValueError("source name must be 'model/scene'")
    model, scene = parts[-2], Path(parts[-1]).stem
    relative = Path(model) / "pages" / f"{scene}.html"
    candidates: list[tuple[Path, bool]] = []
    if mode == "repairs":
        candidates.append((root.parent / REPAIRS_DIR / relative, True))
    if root.name in (SHOWCASE_DIR, REPAIRS_DIR):
        candidates.append((root / relative, root.name == REPAIRS_DIR))
    else:
        candidates += [
            (root / SHOWCASE_DIR / relative, False),
            (root / relative, False),
            (root / f"{scene}.html", False),
        ]
    for page, repaired in candidates:
        if not page.is_file():
            continue
        if page == root / f"{scene}.html":
            return page, root, False
        # Корень статики - каталог, из которого считается URL страницы.
        return page, page.parents[len(relative.parts) - 1], repaired
    raise ManifestError(f"source page for {name} not found under {root} ({mode})")

