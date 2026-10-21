"""Чистые SVG-фигуры отчёта: числа входят, разметка выходит.

Правила спецификации (FRAMEWORK_SPEC §7.2–7.3): каждая фигура — сценарий
чтения, цвет никогда не единственный носитель, а тест или скринридер декодирует
значение из ``data-*`` и ``<title>``, не вглядываясь в пиксели. Поэтому все
геометрии здесь считаются из чисел, и пустой вход даёт явное «нет данных», а не
пустую рамку.

Модуль ничего не читает из прогона сам: отчёт подаёт уже посчитанные строки.
Исключение — визуальный дифф: он принимает два файла кадров, потому что PIL
умеет только растровое сравнение.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EMPTY = "нет данных"

# A frame comes from a run artefact, not from a trusted encoder. PIL's own
# default (~89 Mpx) only warns and then materialises the bitmap; the panel
# boundary below turns both that and a MemoryError into «нет данных» instead of
# losing the whole report.
MAX_IMAGE_PIXELS = 50_000_000

STATUS_ORDER = (
    "checked",
    "not_run",
    "model_failed",
    "capture_failed",
    "evaluator_unavailable",
    "media_missing",
)

STATUS_LABELS = {
    "checked": "проверено",
    "not_run": "не запускалось",
    "model_failed": "сбой модели/транспорта",
    "capture_failed": "страница не отрисовалась",
    "evaluator_unavailable": "evaluator недоступен",
    "media_missing": "кадр/медиа отсутствует",
}

#: Подпись к цвету: статус читается без цветового зрения.
STATUS_MARKS = {
    "checked": "✓",
    "not_run": "○",
    "model_failed": "✕",
    "capture_failed": "◍",
    "evaluator_unavailable": "⚙",
    "media_missing": "▣",
}

STATUS_COLORS = {
    "checked": "#7ee787",
    "not_run": "#8b949e",
    "model_failed": "#ff7b72",
    "capture_failed": "#e3b341",
    "evaluator_unavailable": "#d2a8ff",
    "media_missing": "#56d4dd",
}

SERIES_COLORS = ("#79c0ff", "#d2a8ff", "#e3b341", "#56d4dd")



def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"), quote=True)


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _figure(*, width: float, height: float, label: str, body: str, css: str) -> str:
    return (
        f"<figure class='viz {css}' role='img' aria-label='{_esc(label)}'>"
        f"<svg viewBox='0 0 {width:g} {height:g}' role='img' aria-label='{_esc(label)}'>"
        f"<title>{_esc(label)}</title>{body}</svg></figure>"
    )


def _empty(label: str = EMPTY, *, width: float = 320, height: float = 140) -> str:
    return _figure(
        width=width,
        height=height,
        label=label,
        css="is-empty",
        body=(
            f"<rect x='0.5' y='0.5' width='{width - 1:g}' height='{height - 1:g}' "
            "fill='none' stroke='#30363d' stroke-dasharray='4 4' rx='8'/>"
            f"<text x='{width / 2:g}' y='{height / 2 + 4:g}' text-anchor='middle' "
            f"fill='#8b949e' font-size='13'>{_esc(label)}</text>"
        ),
    )


def tone(share: float | None) -> str:
    """Cell class from a 0–100 value; ``None`` is an unmeasured cell."""
    if share is None:
        return "none"
    return "full" if share >= 80 else ("part" if share >= 50 else "zero")


def polar(cx: float, cy: float, radius: float, index: int, count: int, share: float) -> tuple[float, float]:
    """Point of one radar axis at ``share`` (0–1), first axis straight up."""
    angle = -math.pi / 2 + 2 * math.pi * index / max(count, 1)
    reach = radius * max(0.0, min(1.0, share))
    return cx + reach * math.cos(angle), cy + reach * math.sin(angle)


# --------------------------------------------------------------------------- #
# Radar: score profile per category
# --------------------------------------------------------------------------- #


def radar(
    series: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    *,
    size: int = 360,
    labels: Mapping[str, str] | None = None,
) -> str:
    """One polygon per model over the category axes.

    ``series`` is ``[{"name": ..., "values": {category: percent | None}}]``.
    A missing value is a hole: the polygon falls to the centre and the vertex
    carries ``data-value=''`` instead of pretending a zero was measured.
    """
    categories = list(categories)
    drawn = list(series)[:4]
    if not drawn or not categories:
        return _empty("нет данных для радара", width=size, height=size)
    measured = [
        (s, k) for s in drawn for k in categories
        if isinstance((s.get("values") or {}).get(k), (int, float))
    ]
    if not measured:
        return _empty("нет данных для радара", width=size, height=size)

    cx = cy = size / 2
    radius = size * 0.36
    labels = labels or {}
    count = len(categories)

    rings = "".join(
        "<polygon points='{}' fill='none' stroke='#21262d'/>".format(
            " ".join(f"{x:.2f},{y:.2f}" for x, y in (
                polar(cx, cy, radius * step / 4, i, count, 1.0) for i in range(count)
            ))
        )
        for step in (1, 2, 3, 4)
    )
    spokes = "".join(
        f"<line x1='{cx:g}' y1='{cy:g}' x2='{x:.2f}' y2='{y:.2f}' stroke='#21262d'/>"
        for x, y in (polar(cx, cy, radius, i, count, 1.0) for i in range(count))
    )
    axis_labels = []
    for index, key in enumerate(categories):
        x, y = polar(cx, cy, radius + 18, index, count, 1.0)
        anchor = "middle" if abs(x - cx) < 2 else ("start" if x > cx else "end")
        axis_labels.append(
            f"<text x='{x:.2f}' y='{y + 4:.2f}' text-anchor='{anchor}' fill='#8b949e' font-size='11'>"
            f"{_esc(labels.get(key) or key)}</text>"
        )

    polygons = []
    for order, item in enumerate(drawn):
        values = item.get("values") or {}
        color = SERIES_COLORS[order % len(SERIES_COLORS)]
        points, dots, summary = [], [], []
        for index, key in enumerate(categories):
            value = values.get(key)
            share = _num(value) / 100 if isinstance(value, (int, float)) else 0.0
            x, y = polar(cx, cy, radius, index, count, share)
            points.append(f"{x:.2f},{y:.2f}")
            dots.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='{color}' data-category='{_esc(key)}' "
                f"data-value='{'' if not isinstance(value, (int, float)) else f'{_num(value):.1f}'}'>"
                f"<title>{_esc(item.get('name'))} · {_esc(key)}: "
                f"{'нет данных' if not isinstance(value, (int, float)) else f'{_num(value):.0f}%'}</title>"
                "</circle>"
            )
            summary.append(f"{key}={_num(value):.0f}%" if isinstance(value, (int, float)) else f"{key}=—")
        polygons.append(
            f"<g data-model='{_esc(item.get('name'))}' data-values='{_esc(','.join(summary))}'>"
            f"<title>{_esc(item.get('name'))}: {_esc(', '.join(summary))}</title>"
            f"<polygon points='{' '.join(points)}' fill='{color}' fill-opacity='0.16' "
            f"stroke='{color}' stroke-width='1.6'/>{''.join(dots)}</g>"
        )

    legend = "".join(
        f"<span class='viz-key'><i style='background:{SERIES_COLORS[i % len(SERIES_COLORS)]}'></i>"
        f"{_esc(item.get('name'))}</span>"
        for i, item in enumerate(drawn)
    )
    body = f"{rings}{spokes}{''.join(axis_labels)}{''.join(polygons)}"
    return _figure(
        width=size, height=size + 24, label="Радар качества по категориям", css="radar", body=body,
    ).replace("</figure>", f"</figure><div class='viz-legend'>{legend}</div>")


# --------------------------------------------------------------------------- #
# Heatmap: model × family
# --------------------------------------------------------------------------- #


def _family_name(key: str) -> str:
    return str(key).split("/")[-1]


def heatmap(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    *,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Failure heatmap with the number inside every cell.

    ``rows`` is ``[{"name": ..., "values": {family: {"pct":…, "tasks":…, "failed":…}}}]``.
    Colour alone would hide the magnitude on a projector, so each cell prints
    its percent, keeps the raw numbers in ``data-*`` and explains itself in a
    ``<title>``.
    """
    rows = list(rows)
    families = list(families)
    if not rows or not families:
        return _empty("нет данных для heatmap")

    left, top, cell_w, cell_h = 118, 46, 66, 27
    width = max(430, left + cell_w * len(families) + 20)
    height = top + cell_h * len(rows) + 52
    labels = labels or {}

    head = "".join(
        f"<text x='{left + i * cell_w + cell_w / 2:.1f}' y='{top - 12}' text-anchor='middle' "
        f"fill='#8b949e' font-size='10.5'>{_esc(_family_name(f))}</text>"
        for i, f in enumerate(families)
    )
    head += (
        f"<text x='{left + len(families) * cell_w / 2:.1f}' y='{top - 28}' text-anchor='middle' "
        "fill='#8b949e' font-size='10'>семейство (цвет + число)</text>"
    )

    cells = []
    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        name = str(row.get("name") or "")
        cells.append(
            f"<text x='{left - 8}' y='{y + 18}' text-anchor='end' fill='#c9d1d9' font-size='11.5'>"
            f"{_esc(name)}</text>"
        )
        values = row.get("values") or {}
        for column, family in enumerate(families):
            x = left + column * cell_w
            cell = values.get(family) or {}
            pct = cell.get("pct")
            pct = _num(pct) if isinstance(pct, (int, float)) else None
            tasks = int(_num(cell.get("tasks")))
            failed = int(_num(cell.get("failed")))
            css = tone(pct)
            label = (
                f"{name} · {family}: " + (
                    f"{pct:.0f}% (провалено {failed} из {tasks})" if pct is not None
                    else ("не проверялось" if tasks else "нет задач")
                )
            )
            text = f"{pct:.0f}%" if pct is not None else "—"
            cells.append(
                f"<g class='hm {css}' data-model='{_esc(name)}' data-family='{_esc(family)}' "
                f"data-pct='{'' if pct is None else f'{pct:.1f}'}' data-tasks='{tasks}' data-failed='{failed}'>"
                f"<title>{_esc(label)}</title>"
                f"<rect x='{x + 1}' y='{y + 1}' width='{cell_w - 2}' height='{cell_h - 2}' rx='4'/>"
                f"<text x='{x + cell_w / 2:.1f}' y='{y + 19}' text-anchor='middle' font-size='11.5'>"
                f"{text}</text></g>"
            )

    legend = (
        f"<g class='viz-legend-svg' transform='translate({left},{height - 24})'>"
        f"<rect x='0' y='-11' width='12' height='12' rx='3' class='full'/>"
        "<text x='17' y='0' fill='#8b949e' font-size='10.5'>≥80%</text>"
        f"<rect x='78' y='-11' width='12' height='12' rx='3' class='part'/>"
        "<text x='95' y='0' fill='#8b949e' font-size='10.5'>50–79%</text>"
        f"<rect x='160' y='-11' width='12' height='12' rx='3' class='zero'/>"
        "<text x='177' y='0' fill='#8b949e' font-size='10.5'>&lt;50%</text>"
        f"<rect x='230' y='-11' width='12' height='12' rx='3' class='none'/>"
        "<text x='247' y='0' fill='#8b949e' font-size='10.5'>нет задач</text></g>"
    )
    body = f"{head}{''.join(cells)}{legend}"
    return _figure(
        width=width, height=height, label="Heatmap ошибок: модели против семейств",
        css="heatmap", body=body,
    )


# --------------------------------------------------------------------------- #
# Completeness bar and status legend
# --------------------------------------------------------------------------- #


def classify(score: Mapping[str, Any]) -> str:
    """One status per task, infrastructure failures kept apart.

    ``not_run`` is not ``model_failed`` and a render failure is not a bad
    answer: the bar must show what was actually verified, not a flat "failed".
    """
    if score.get("skipped"):
        return "not_run"
    error = str(score.get("error") or "")
    if error:
        low = error.lower()
        if "evaluator" in low or "category not registered" in low:
            return "evaluator_unavailable"
        return "model_failed"
    for check in score.get("checks") or []:
        if check.get("passed"):
            continue
        detail = str(check.get("detail") or "").lower()
        if check.get("name") == "html" and "рендер" in detail:
            return "capture_failed"
        if "кадр" in detail or "медиа" in detail:
            return "media_missing"
    return "checked"


def task_statuses(scores: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_ORDER}
    for score in scores:
        counts[classify(score)] += 1
    return counts


def status_legend(counts: Mapping[str, Any] | None = None) -> str:
    """Swatch + icon + word per status, so colour is never the only channel."""
    items = []
    for status in STATUS_ORDER:
        count = int(_num((counts or {}).get(status)))
        amount = f" · {count}" if counts is not None else ""
        items.append(
            f"<span class='viz-key' data-status='{status}'>"
            f"<i style='background:{STATUS_COLORS[status]}'></i>"
            f"<b>{STATUS_MARKS[status]}</b> {_esc(STATUS_LABELS[status])}{amount}</span>"
        )
    return f"<div class='viz-legend statuses' role='list'>{''.join(items)}</div>"


def completeness(counts: Mapping[str, Any]) -> str:
    """Stacked bar of task statuses, proportionally from the counts."""
    total = sum(int(_num(counts.get(status))) for status in STATUS_ORDER)
    if not total:
        return _empty("нет данных: ни одного задания")

    width, bar_y, bar_h = 640, 10, 18
    segments, x = [], 0.0
    for index, status in enumerate(STATUS_ORDER):
        count = int(_num(counts.get(status)))
        if not count:
            continue
        share = count / total
        w = width - x if index == len(STATUS_ORDER) - 1 else width * share
        label = f"{STATUS_LABELS[status]}: {count} из {total} ({100 * share:.0f}%)"
        segments.append(
            f"<g data-status='{status}' data-count='{count}' data-share='{share:.4f}'>"
            f"<title>{_esc(label)}</title>"
            f"<rect x='{x:.2f}' y='{bar_y}' width='{w:.2f}' height='{bar_h}' "
            f"fill='{STATUS_COLORS[status]}' fill-opacity='0.85'/></g>"
        )
        x += w

    notes = []
    for status in STATUS_ORDER:
        count = int(_num(counts.get(status)))
        if not count:
            continue
        notes.append(
            f"<text x='0' y='{bar_y + bar_h + 20 + len(notes) * 17}' fill='#c9d1d9' font-size='11.5'>"
            f"<tspan fill='{STATUS_COLORS[status]}'>{STATUS_MARKS[status]}</tspan> "
            f"{_esc(STATUS_LABELS[status])}: {count} из {total} ({100 * count / total:.0f}%)</text>"
        )
    height = bar_y + bar_h + 32 + len(notes) * 17
    return _figure(
        width=width, height=height, label=f"Полнота прогона: проверено {100 * int(_num(counts.get('checked'))) / total:.0f}%",
        css="completeness", body=f"{''.join(segments)}{''.join(notes)}",
    )


# --------------------------------------------------------------------------- #
# Failure clusters: char trigrams → hashed features → PCA 2D
# --------------------------------------------------------------------------- #

_FEATURE_DIM = 64


def _trigrams(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if len(flat) < 3:
        return [flat or "?"]
    return [flat[i:i + 3] for i in range(len(flat) - 2)]


def features(text: str, *, seed: int = 20261021, dim: int = _FEATURE_DIM) -> list[float]:
    """L2-normalised hashed trigram vector — deterministic for a seed."""
    vector = [0.0] * dim
    for gram in _trigrams(text):
        digest = hashlib.blake2b(f"{seed}\0{gram}".encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest, "big") % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _pca(matrix: list[list[float]]) -> list[tuple[float, float]]:
    """First two principal components, or a deterministic ring on degeneracy."""
    count = len(matrix)
    if count == 1:
        return [(0.0, 0.0)]
    try:
        import numpy as np

        table = np.asarray(matrix, dtype=float)
        table = table - table.mean(axis=0)
        _, _, vt = np.linalg.svd(table, full_matrices=False)
        coords = table @ vt[:2].T
        pairs = [(float(row[0]), float(row[1]) if row.shape[0] > 1 else 0.0) for row in coords]
    except Exception:  # numpy отсутствует или вырожденная матрица
        pairs = [(row[0] if row else 0.0, row[1] if len(row) > 1 else 0.0) for row in matrix]
    if max((abs(x) for x, _ in pairs), default=0.0) < 1e-12 and max((abs(y) for _, y in pairs), default=0.0) < 1e-12:
        return [(math.cos(2 * math.pi * i / count), math.sin(2 * math.pi * i / count)) for i in range(count)]
    return pairs


def _clusters(points: Sequence[tuple[float, float]], threshold: float = 48.0) -> list[list[int]]:
    groups: list[list[int]] = []
    centres: list[tuple[float, float]] = []
    for index, (x, y) in enumerate(points):
        for group_index, (cx, cy) in enumerate(centres):
            if math.hypot(x - cx, y - cy) <= threshold:
                groups[group_index].append(index)
                size = len(groups[group_index])
                centres[group_index] = (
                    (cx * (size - 1) + x) / size, (cy * (size - 1) + y) / size,
                )
                break
        else:
            groups.append([index])
            centres.append((x, y))
    return groups


def cluster_map(items: Sequence[Mapping[str, Any]], *, width: int = 640, height: int = 320, seed: int = 20261021) -> str:
    """Semantic map of failures: trigram hashes reduced to two components.

    ``items`` is ``[{"id", "model", "category", "text", "count"?}]``. Identical
    texts are counted, not stacked, and points carry the text in ``<title>``.
    The seed fixes the hash bucket order, so two reports over one run agree.
    """
    items = list(items)
    if not items:
        return _empty("нет данных: провалов нет", width=width, height=min(height, 140))

    matrix = [features(str(item.get("text") or ""), seed=seed) for item in items]
    raw = _pca(matrix)

    span_x = max(x for x, _ in raw) - min(x for x, _ in raw)
    span_y = max(y for _, y in raw) - min(y for _, y in raw)
    pad = 34
    points = []
    for x, y in raw:
        px = width / 2 if span_x < 1e-12 else pad + (x - min(p[0] for p in raw)) / span_x * (width - 2 * pad)
        py = height / 2 if span_y < 1e-12 else pad + (y - min(p[1] for p in raw)) / span_y * (height - 2 * pad)
        points.append((round(px, 2), round(py, 2)))

    groups = _clusters(points)
    cluster_circles, points_svg = [], []
    for number, group in enumerate(groups, 1):
        members = [points[i] for i in group]
        cx = sum(p[0] for p in members) / len(members)
        cy = sum(p[1] for p in members) / len(members)
        spread = max(math.hypot(x - cx, y - cy) for x, y in members) + 13
        cluster_circles.append(
            f"<circle class='cluster' cx='{cx:.1f}' cy='{cy:.1f}' r='{spread:.1f}' "
            "fill='none' stroke='#2b3440' stroke-dasharray='3 3'>"
            f"<title>кластер C{number}: {len(group)} провал(ов)</title></circle>"
            f"<text x='{cx:.1f}' y='{cy - spread - 4:.1f}' text-anchor='middle' fill='#8b949e' "
            f"font-size='10'>C{number} · n={len(group)}</text>"
        )
        for index in group:
            item = items[index]
            x, y = points[index]
            count = max(1, int(_num(item.get("count")) or 1))
            radius = 4 + min(count, 6)
            text = f"{item.get('id')} · {item.get('model')}: {item.get('text')}"
            points_svg.append(
                f"<g class='pt' data-id='{_esc(item.get('id'))}' data-model='{_esc(item.get('model'))}' "
                f"data-category='{_esc(item.get('category'))}' data-x='{x:.2f}' data-y='{y:.2f}' "
                f"data-cluster='C{number}' data-count='{count}'>"
                f"<title>{_esc(text)}</title>"
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius}' fill='#d2a8ff' fill-opacity='0.55' "
                f"stroke='#d2a8ff'/>"
                + (f"<text x='{x + radius + 2:.1f}' y='{y + 3:.1f}' fill='#8b949e' font-size='9'>{count}×</text>"
                   if count > 1 else "")
                + "</g>"
            )
    note = (
        f"<text x='{pad}' y='{height - 8}' fill='#6e7681' font-size='10'>"
        f"char-триграммы → {_FEATURE_DIM} хеш-признаков → PCA 2D · seed {seed} · точек {len(items)}</text>"
    )
    body = "".join(cluster_circles) + "".join(points_svg) + note
    return _figure(
        width=width, height=height, label="Карта семантических кластеров провалов",
        css="clusters", body=body,
    )


# --------------------------------------------------------------------------- #
# Visual diff: changed-pixel bbox and a zoomed inset
# --------------------------------------------------------------------------- #


def diff_bbox(reference: str | Path, answer: str | Path, *, tolerance: int = 16) -> dict[str, Any]:
    """Bounding box of changed pixels in the reference's pixel space.

    ``bbox`` is ``(left, top, right, bottom)``, right/bottom exclusive — the
    same convention PIL uses. Sizes that differ are normalised to the
    reference, and a missing file is ``bbox=None`` with an explicit reason.
    """
    try:
        from PIL import Image, ImageChops
    except ImportError:  # pragma: no cover - PIL есть в dev-окружении
        return {"bbox": None, "reason": f"{EMPTY}: PIL недоступен"}

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(reference) as raw:
            ref = raw.convert("RGB")
        with Image.open(answer) as raw:
            got = raw.convert("RGB")
        if got.size != ref.size:
            got = got.resize(ref.size)
        mask = ImageChops.difference(ref, got).convert("L").point(lambda v: 255 if v > tolerance else 0)
        bbox = mask.getbbox()
        changed = mask.histogram()[255]
    except (OSError, ValueError, MemoryError):
        return {"bbox": None, "reason": f"{EMPTY}: кадр не читается"}
    return {
        "bbox": bbox,
        "width": ref.width,
        "height": ref.height,
        "changed": changed,
        "reason": None if bbox else "расхождений нет",
    }


def _thumb(path: str | Path, width: int) -> tuple[str, float] | None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        ratio = image.height / image.width
        image = image.resize((width, max(1, round(width * ratio))))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    except (OSError, ValueError, MemoryError):
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), ratio


def visual_diff(
    reference: str | Path,
    answer: str | Path,
    *,
    tolerance: int = 16,
    thumb_w: int = 240,
) -> str:
    """Reference / answer / inset, with the changed region boxed and numbered.

    The crop and the coordinates come from :func:`diff_bbox`; the report only
    embeds the resulting data URIs, so the page stays offline.
    """
    label = "Визуальный дифф: эталон и ответ"
    info = diff_bbox(reference, answer, tolerance=tolerance)
    bbox = info.get("bbox")
    if bbox is None:
        return _empty(str(info.get("reason") or EMPTY), width=thumb_w * 2 + 30, height=120)

    left, top, right, bottom = bbox
    thumbs = []
    for path, name in ((reference, "эталон"), (answer, "ответ")):
        thumb = _thumb(path, thumb_w)
        if thumb is None:
            return _empty(f"{EMPTY}: кадр не читается")
        thumbs.append((thumb[0], thumb[1]))
    ratio = thumbs[0][1]  # обе картинки растягиваются на одну высоту
    panel_h = thumb_w * ratio
    scale = thumb_w / max(1, int(info["width"]))

    width = thumb_w * 2 + 30
    height = 24 + panel_h + 172
    inset_size = 150
    pad = 5
    crop_box = (
        max(0, left - pad), max(0, top - pad),
        min(int(info["width"]), right + pad), min(int(info["height"]), bottom + pad),
    )
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        with Image.open(answer) as raw:
            crop = raw.convert("RGB").crop(crop_box)
        crop.thumbnail((inset_size, inset_size))
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        inset = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except (OSError, ValueError, MemoryError):  # pragma: no cover - врезка необязательна
        inset = thumbs[1][0]

    caption = (
        f"расхождение x {left}–{right}, y {top}–{bottom} · пикселей {int(info['changed'])}"
    )
    body = (
        f"<text x='0' y='11' fill='#c9d1d9' font-size='11'>{_esc(caption)}</text>"
        f"<text x='0' y='21' fill='#6e7681' font-size='10'>эталон</text>"
        f"<image href='{thumbs[0][0]}' x='0' y='24' width='{thumb_w}' height='{panel_h:.2f}'/>"
        f"<text x='{thumb_w + 30}' y='21' fill='#6e7681' font-size='10'>ответ</text>"
        f"<image href='{thumbs[1][0]}' x='{thumb_w + 30}' y='24' width='{thumb_w}' height='{panel_h:.2f}'/>"
        f"<rect x='{thumb_w + 30 + left * scale:.2f}' y='{24 + top * scale:.2f}' "
        f"width='{max(1.0, (right - left) * scale):.2f}' height='{max(1.0, (bottom - top) * scale):.2f}' "
        "fill='#ff6b6b' fill-opacity='0.18' stroke='#ff6b6b'/>"
        f"<text x='{thumb_w + 30}' y='{24 + panel_h + 15}' fill='#6e7681' font-size='10'>ответ, красный бокс — область</text>"
        f"<text x='0' y='{24 + panel_h + 15}' fill='#8b949e' font-size='10'>врезка той же области:</text>"
        f"<image href='{inset}' x='0' y='{24 + panel_h + 22}' width='{inset_size}' height='{inset_size}' "
        f"style='image-rendering:pixelated'/>"
        f"<rect x='0' y='{24 + panel_h + 22}' width='{inset_size}' height='{inset_size}' "
        "fill='none' stroke='#30363d'/>"
    )
    return _figure(
        width=width,
        height=height,
        label=f"{label}, {caption}",
        css="diff",
        body=body,
    ).replace("<svg ", f"<svg data-bbox='{left},{top},{right},{bottom}' data-changed='{int(info['changed'])}' ", 1)


# --------------------------------------------------------------------------- #
# Verdict strip: three conclusions and three risks, straight from the numbers
# --------------------------------------------------------------------------- #


def _pct(row: Mapping[str, Any] | None) -> float | None:
    possible = _num((row or {}).get("possible"))
    if not possible:
        return None
    return 100.0 * _num((row or {}).get("earned")) / possible


def _wrap(text: str, width: int) -> list[str]:
    """Break a line on spaces so an SVG text stays inside its column."""
    lines: list[str] = []
    current = ""
    for word in str(text).split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def verdict_strip(
    summary: Mapping[str, Any],
    statuses: Mapping[str, Any] | None = None,
    *,
    titles: Mapping[str, str] | None = None,
) -> str:
    """Three conclusions and three risks; each traces to one number."""
    titles = titles or {}
    categories = [
        (key, row, _pct(row)) for key, row in (summary.get("per_category") or {}).items()
        if (row or {}).get("possible") and row.get("skipped") != row.get("tasks")
    ]
    earned = _num(summary.get("headline_earned"))
    possible = _num(summary.get("headline_possible"))

    conclusions: list[str] = []
    if possible:
        line = f"Качество {earned:.0f} из {possible:.0f}"
        ci = summary.get("headline_ci95")
        line += f" · ДИ 95% {ci[0]:.0f}–{ci[1]:.0f}%" if ci else " · ДИ не посчитан"
        conclusions.append(line)
    else:
        conclusions.append("нет данных: ни одного измеренного балла")
    if categories:
        best_key, _, best = max(categories, key=lambda x: (x[2] if x[2] is not None else -1))
        worst_key, worst_row, worst = min(categories, key=lambda x: (x[2] if x[2] is not None else 101))
        conclusions.append(
            f"Сильнейшая ось {best_key} {titles.get(best_key) or ''}: {best:.0f}%".replace(" :", ":")
        )
        conclusions.append(
            f"Слабейшая ось {worst_key} {titles.get(worst_key) or ''}: {worst:.0f}% "
            f"({worst_row.get('tasks', 0)} заданий)".replace(" :", ":")
        )
    else:
        conclusions.extend(["сильнейшая ось не определена", "слабейшая ось не определена"])
    while len(conclusions) < 3:
        conclusions.append("—")

    risks: list[str] = []
    state = str(summary.get("admission_state") or "passed")
    if state == "failed":
        risks.append("long context не прошёл допуск: вердикт по короткому знаменателю")
    elif state == "unverified":
        risks.append("допуск long context не подтверждён")
    if summary.get("headline_provisional"):
        risks.append(f"малый n={summary.get('headline_n') or 0}: интервал ориентировочный")
    ci = summary.get("headline_ci95")
    if ci and ci[1] - ci[0] > 30:
        risks.append(f"широкий ДИ: {ci[1] - ci[0]:.0f} п.п.")
    skipped_categories = [
        key for key, row in (summary.get("per_category") or {}).items()
        if row.get("tasks") and row.get("skipped") == row.get("tasks")
    ]
    if skipped_categories:
        risks.append(f"категории не проверялись: {', '.join(sorted(skipped_categories))}")
    elif summary.get("skipped"):
        risks.append(f"заданий не проверялось: {summary['skipped']}")
    weak = [key for key, _, pct in categories if pct is not None and pct < 50]
    if weak:
        risks.append(f"слабых осей (<50%): {len(weak)} ({', '.join(sorted(weak))})")
    if summary.get("errors"):
        risks.append(f"ошибок evaluator/транспорта: {summary['errors']}")
    if statuses and statuses.get("capture_failed"):
        risks.append(f"страниц не отрисовалось: {statuses['capture_failed']}")
    if summary.get("full_suite") is False:
        risks.append("прогон частичный: не все задания сьюта")
    if not risks:
        risks = ["рисков по цифрам не видно"] * 3
    elif len(risks) < 3:
        risks.extend(["прочих числовых рисков нет"] * (3 - len(risks)))
    risks = risks[:3]

    column_width = 46
    wrapped = {
        "conclusion": [_wrap(text, column_width) for text in conclusions],
        "risk": [_wrap(text, column_width) for text in risks],
    }
    columns = {kind: 30 + sum(len(lines) * 16 + 7 for lines in blocks) for kind, blocks in wrapped.items()}
    height = max(columns.values()) + 12
    body = (
        f"<text x='0' y='16' fill='#8b949e' font-size='12'>3 вывода</text>"
        f"<text x='330' y='16' fill='#8b949e' font-size='12'>3 риска</text>"
    )
    for column, (kind, blocks) in enumerate(wrapped.items()):
        x = column * 330
        y = 38
        colour = "#c9d1d9" if kind == "conclusion" else "#e3b341"
        for index, lines in enumerate(blocks):
            text = (conclusions if kind == "conclusion" else risks)[index]
            body += (
                f"<g data-kind='{kind}' data-index='{index}' data-text='{_esc(text)}'>"
                f"<title>{_esc(text)}</title>"
                + "".join(
                    f"<text x='{x}' y='{y + line_index * 16}' fill='{colour}' font-size='12'>{_esc(line)}</text>"
                    for line_index, line in enumerate(lines)
                )
                + "</g>"
            )
            y += len(lines) * 16 + 7
    return _figure(
        width=640, height=height, label="Вердикт по цифрам: 3 вывода и 3 риска",
        css="verdict-strip", body=body,
    )
