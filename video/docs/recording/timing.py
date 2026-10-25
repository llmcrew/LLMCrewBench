#!/usr/bin/env python3
"""timing: контракт длительности, частот, полос и статусов показа.

Витрина из DailyBench выросла в этот набор: 20-секундный клип, показ 60 к/с при
мастере 120 к/с, две полосы по 960 px и занавес 0,75 с. Здесь лежит только
политика - без Chrome, ffmpeg и Remotion, чтобы её можно было проверить на
чистых числах.

Правила, которые модуль защищает:

* клип - ровно ``secs × fps`` кадров, ``secs`` кратен кадру мастера;
* мастер снимается с запасом над показом (2× по умолчанию), занавес - половина
  ``SHOWCASE_LEAD``, то есть 0,75 с при штатной политике;
* статус клипа - из закрытого словаря, а плашка и живой клип не смешиваются;
* ``result-notes`` не выдаёт провал за успех: починка без изменения исходника
  (тот же SHA) не открывает полноэкранный ход.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

#: Окно показа одного полноэкранного хода: 20 с мастера на 120 к/с = 2400 кадров.
CLIP_SECONDS = 20.0
#: Мастер пишется вдвое чаще показа: запас на замедление и плавный занавес.
MASTER_FPS = 120
DISPLAY_FPS = 60
#: Две равные половины кадра: 2 × 960 × 1080.
LANES = 2
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
#: Занавес: кадры выезда полосы на весь кадр. Половина ``SHOWCASE_LEAD``.
CURTAIN_SECONDS = 0.75
SHOWCASE_LEAD = 1.5
#: Запись сетки = ход показа (окно + занавес с двух сторон), 23 с.
SHOWCASE_TURN_SECONDS = CLIP_SECONDS + SHOWCASE_LEAD * 2
SHOWCASE_EMPTY_SECONDS = 4.0

#: Все допустимые статусы медиа. Первые два - живой клип и штатный пропуск.
STATUSES = (
    "ok",
    "skipped",
    "not_run",
    "model_failed",
    "evaluator_unavailable",
    "infra_failure",
    "capture_failed",
    "media_missing",
)
#: Статусы плашки: клип на диске есть, но это заглушка, а не запись.
PLACEHOLDER_STATUSES = frozenset(
    {"skipped", "not_run", "model_failed", "evaluator_unavailable", "infra_failure",
     "capture_failed"}
)
#: Исходный вердикт в result-notes: провал или частичный результат.
NOTE_STATUSES = ("partial", "failed")


def frame_count(seconds: float, fps: int, *, allow_zero: bool = False) -> int:
    """Кадров ровно столько, сколько требует длительность; иначе ``ValueError``."""
    if fps <= 0 or not isinstance(fps, int):
        raise ValueError("fps must be a positive integer")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration must be a finite, non-negative number")
    count = round(seconds * fps)
    if (not math.isclose(count / fps, seconds, rel_tol=0, abs_tol=1e-9)
            or (count == 0 and not allow_zero)):
        raise ValueError(f"duration must be a positive multiple of 1/{fps} second")
    return count


def stage_frames(lead: float, fps: int) -> int:
    """Кадры одного занавеса: половина лида, но не короче двух."""
    return max(2, round(lead * fps * 0.5))


def validate_timing(
    *, open: float = CLIP_SECONDS, secs: float | None = None, fps: int = MASTER_FPS,
    curtain: float = CURTAIN_SECONDS, lead: float = SHOWCASE_LEAD,
    display_fps: int = DISPLAY_FPS,
) -> dict[str, int]:
    """Проверить политику и вернуть кадровые величины. ``ValueError`` - разлад.

    ``open`` - окно показа хода, ``secs`` - вся запись (по умолчанию один ход).
    Запись обязана состоять из целых ходов: иначе занавес обрежется.
    """
    if display_fps <= 0 or (fps % display_fps and display_fps % fps):
        raise ValueError(f"master fps {fps} and display fps {display_fps} are not multiples")
    frames = frame_count(open, fps)
    curtain_frames = frame_count(curtain, fps)
    lead_frames = frame_count(lead, fps)
    if stage_frames(lead, fps) != curtain_frames:
        raise ValueError(
            f"curtain {curtain}s must be half of lead {lead}s "
            f"({stage_frames(lead, fps)} frames at {fps} fps, got {curtain_frames})"
        )
    turn_frames = frame_count(open + 2 * lead, fps)
    total = turn_frames if secs is None else frame_count(secs, fps)
    if total % turn_frames:
        raise ValueError(
            f"recording {total} frames is not a whole number of {turn_frames}-frame turns"
        )
    return {
        "frames": frames, "curtain_frames": curtain_frames, "lead_frames": lead_frames,
        "turn_frames": turn_frames, "total_frames": total, "turns": total // turn_frames,
    }


def valid_status(status: object) -> bool:
    return isinstance(status, str) and status in STATUSES


def is_placeholder(status: object) -> bool:
    return status in PLACEHOLDER_STATUSES


def media_state(status: object, *, has_clip: bool, black: bool) -> str:
    """Свести статус, наличие файла и черноту кадра в один вердикт.

    ``ok`` с чёрным клипом - подмена записи заглушкой; плашка без файла - провал
    поставки; плашка с цветным клипом - статус разошёлся с материалом.
    """
    if not valid_status(status):
        return "unknown-status"
    if not has_clip:
        return "missing-clip"
    if is_placeholder(status):
        return "placeholder-ok" if black else "placeholder-not-black"
    return "ok" if not black else "ok-black"


# --------------------------------------------------------------------------- #
# Result-notes: причина, ожидание и починка
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Repair:
    """Починка исходника: клип, суть правки и SHA до/после."""

    clip: str
    change: str
    source_sha256: str
    repaired_sha256: str

    @property
    def same_source(self) -> bool:
        return self.source_sha256 == self.repaired_sha256


@dataclasses.dataclass(frozen=True)
class ResultNote:
    model: str
    scene: str
    original_status: str
    reason: str
    expected: str
    repair: Repair | None = None


def parse_repair(raw: object) -> Repair | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("repair must be an object or null")
    fields = {}
    for key in ("clip", "change", "sourceSha256", "repairedSha256"):
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"repair.{key} must be a non-empty string")
        fields[key] = value
    return Repair(
        clip=fields["clip"], change=fields["change"],
        source_sha256=fields["sourceSha256"], repaired_sha256=fields["repairedSha256"],
    )


def parse_note(row: object) -> ResultNote:
    if not isinstance(row, dict):
        raise ValueError("result note must be an object")
    for key in ("model", "scene", "reason", "expected"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"result note.{key} must be a non-empty string")
    status = row.get("originalStatus")
    if status not in NOTE_STATUSES:
        raise ValueError(f"result note.originalStatus must be one of {NOTE_STATUSES}")
    return ResultNote(
        model=row["model"], scene=row["scene"], original_status=status,
        reason=row["reason"], expected=row["expected"], repair=parse_repair(row.get("repair")),
    )


def note_errors(row: object) -> list[str]:
    """Все нарушения одной записи result-notes, а не первое."""
    errors: list[str] = []
    if not isinstance(row, dict):
        return ["result note must be an object"]
    for key in ("model", "scene", "reason", "expected"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"{key}: пустое или не строковое значение")
    if row.get("originalStatus") not in NOTE_STATUSES:
        errors.append(f"originalStatus: {row.get('originalStatus')!r} вне {NOTE_STATUSES}")
    repair = row.get("repair")
    if repair is not None:
        if not isinstance(repair, dict):
            errors.append("repair: не объект")
        else:
            for key in ("clip", "change", "sourceSha256", "repairedSha256"):
                value = repair.get(key)
                if not isinstance(value, str) or not value:
                    errors.append(f"repair.{key}: пустое или не строковое значение")
    return errors


def presentation_clip(run: dict) -> str | None:
    """Что вообще можно показать: починку - всегда, живой клип - если не провал."""
    note = run.get("resultNote") or {}
    repair = note.get("repair") or {}
    if repair.get("clip"):
        return repair["clip"]
    if run.get("status") == "ok" and note.get("originalStatus") != "failed":
        return run.get("clip")
    return None


def can_open_fullscreen(run: dict) -> bool:
    """Полноэкранный ход: клип есть, а провал снимается только правкой кода."""
    if presentation_clip(run) is None:
        return False
    note = run.get("resultNote") or {}
    if note.get("originalStatus") != "failed":
        return True
    repair = note.get("repair") or {}
    # Дольше ждать старт и не менять код - тот же провал. Починка засчитывается,
    # только если исходник реально изменился.
    return bool(repair) and repair.get("sourceSha256") == repair.get("repairedSha256")


def fullscreen_order(runs: list[dict], offset: int) -> list[int]:
    """Индексы полос в порядке хода; фильтр не сдвигает правую полосу влево."""
    if not runs:
        return []
    return [
        index for index in ((i + offset) % len(runs) for i in range(len(runs)))
        if can_open_fullscreen(runs[index])
    ]


def build_timeline(entries: list[dict]) -> list[dict]:
    """Секции показа с абсолютным ``from``; пустая секция короткая, а не нулевая."""
    from_ = 0.0
    timeline = []
    for index, entry in enumerate(entries):
        order = fullscreen_order(entry.get("runs") or [], index)
        seconds = len(order) * SHOWCASE_TURN_SECONDS if order else SHOWCASE_EMPTY_SECONDS
        timeline.append({**entry, "from": from_, "seconds": seconds, "order": order})
        from_ += seconds
    return timeline


def timeline_frames(timeline: list[dict], fps: int) -> int:
    return sum(frame_count(section["seconds"], fps) for section in timeline)


def timeline_errors(timeline: list[dict], fps: int) -> list[str]:
    """Непрерывность и кратность кадрам - до рендера, а не после."""
    errors: list[str] = []
    cursor: Fraction | None = Fraction(0)
    for section in timeline:
        start = Fraction(section["from"]).limit_denominator(1000)
        if cursor is not None and start != cursor:
            errors.append(f"{section.get('scene')}: from {float(start)} != {float(cursor)}")
        count = frame_count(section["seconds"], fps)
        if count <= 0:
            errors.append(f"{section.get('scene')}: пустая секция")
        cursor = start + Fraction(count, fps)
    return errors
