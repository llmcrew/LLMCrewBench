#!/usr/bin/env python3
"""layout: две полосы, занавес на 0,75 с и раскадровка показа.

Полосы - 960×1080 каждая (половина кадра). Полоса стоит в своей половине, пока
не её очередь; на своём ходу выезжает на весь кадр за ``curtain`` секунд, стоит
клип и уходит обратно за те же ``curtain``. Между ходами - пауза ``lead − curtain``.

Здесь же проверка раскладки: полосы без зазоров накрывают кадр, занавес кратен
кадру, ходы не перекрываются и не выходят за секцию. Рекордер передаёт эти
величины драйверу страницы (``page-driver.js``/``curtain.js``), поэтому формулы
обязаны совпадать; совпадение проверяет ``tests/test_recording_features.py``.
"""

from __future__ import annotations

import dataclasses

from timing import (
    CLIP_SECONDS,
    CURTAIN_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LANES,
    MASTER_FPS,
    SHOWCASE_LEAD,
    validate_timing,
)


@dataclasses.dataclass(frozen=True)
class Layout:
    """Полосы и их ширина. Замораживается: кадр не пересобирается на ходу."""

    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT
    lanes: int = LANES

    def __post_init__(self) -> None:
        if self.lanes < 1:
            raise ValueError("lanes must be >= 1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame must be positive")
        if self.width % self.lanes:
            raise ValueError(f"width {self.width} is not divisible by {self.lanes} lanes")

    @property
    def lane_width(self) -> int:
        return self.width // self.lanes

    def lane(self, index: int) -> dict[str, int]:
        if not 0 <= index < self.lanes:
            raise ValueError(f"lane {index} outside 0..{self.lanes - 1}")
        return {"x": index * self.lane_width, "y": 0, "width": self.lane_width, "height": self.height}

    def open(self) -> dict[str, int]:
        return {"x": 0, "y": 0, "width": self.width, "height": self.height}

    def at(self, index: int, curtain: float) -> dict[str, int]:
        """Геометрия полосы между своей половиной и полным кадром.

        ``curtain`` - 0..1 (0 - полоса на месте, 1 - во весь кадр). Дробные
        координаты округляются: пиксель кадра - целое.
        """
        if not 0.0 <= curtain <= 1.0:
            raise ValueError("curtain must be within 0..1")
        rest, full = self.lane(index), self.open()
        return {
            key: round(rest[key] + (full[key] - rest[key]) * curtain)
            for key in ("x", "y", "width", "height")
        }

    def errors(self) -> list[str]:
        errors = []
        spans = [(lane["x"], lane["x"] + lane["width"]) for lane in
                 (self.lane(index) for index in range(self.lanes))]
        for (_, left_end), (right_start, _) in zip(spans, spans[1:]):
            if left_end != right_start:
                errors.append(f"lanes overlap or leave a gap at {left_end}..{right_start}")
        if spans[0][0] != 0 or spans[-1][1] != self.width:
            errors.append("lanes do not cover the frame")
        if any(lane["height"] != self.height for lane in
               (self.lane(index) for index in range(self.lanes))):
            errors.append("lanes do not fill the frame height")
        return errors


@dataclasses.dataclass(frozen=True)
class Turn:
    """Один полноэкранный ход: какая полоса, когда и с каким занавесом."""

    lane: int
    clip: str
    turn_start: int
    expand_from: int
    open_from: int
    open_to: int
    return_to: int
    turn_end: int

    def facing(self, frame: int, curtain_frames: int) -> float:
        """0 - полоса на месте, 1 - во весь кадр; вне хода - 0."""
        if frame < self.expand_from or frame >= self.return_to:
            return 0.0
        if frame < self.open_from:
            return (frame - self.expand_from) / curtain_frames
        if frame < self.open_to:
            return 1.0
        return (self.return_to - frame) / curtain_frames


def storyboard(
    order: list[int],
    *,
    clips: list[str] | None = None,
    open: float = CLIP_SECONDS,
    secs: float | None = None,
    fps: int = MASTER_FPS,
    lead: float = SHOWCASE_LEAD,
    curtain: float = CURTAIN_SECONDS,
    lanes: int = LANES,
    start: int = 0,
) -> list[Turn]:
    """Раскадровка записи: ходы в порядке ``order``, без перекрытий.

    Полоса расширяется ровно за ``curtain`` кадров до окна и возвращается за
    столько же после него; ход - ``open + 2·lead``, запись - целое число
    ходов. Порядок повторяется, если ходов больше, чем полос.
    """
    if lanes < 1:
        raise ValueError("lanes must be >= 1")
    if not order:
        raise ValueError("order must not be empty")
    if len(set(order)) != len(order):
        raise ValueError("order repeats a lane")
    if any(not 0 <= lane < lanes for lane in order):
        raise ValueError(f"order lane outside 0..{lanes - 1}")
    if secs is None:
        secs = len(order) * (open + 2 * lead)
    sizes = validate_timing(open=open, secs=secs, fps=fps, curtain=curtain, lead=lead)
    turns = []
    for position in range(sizes["turns"]):
        begin = start + position * sizes["turn_frames"]
        open_from = begin + sizes["lead_frames"]
        open_to = open_from + sizes["frames"]
        turns.append(Turn(
            lane=order[position % len(order)],
            clip=(clips[position % len(clips)] if clips else ""),
            turn_start=begin,
            expand_from=open_from - sizes["curtain_frames"],
            open_from=open_from,
            open_to=open_to,
            return_to=open_to + sizes["curtain_frames"],
            turn_end=begin + sizes["turn_frames"],
        ))
    return turns


def section_seconds(order: list[int], *, open: float = CLIP_SECONDS,
                    lead: float = SHOWCASE_LEAD, empty: float = 4.0) -> float:
    return len(order) * (open + 2 * lead) if order else empty


def storyboard_errors(turns: list[Turn], *, open: float = CLIP_SECONDS,
                      secs: float | None = None, fps: int = MASTER_FPS,
                      lead: float = SHOWCASE_LEAD, curtain: float = CURTAIN_SECONDS,
                      start: int = 0) -> list[str]:
    """Перекрытия, выход за запись и разлад величин - до записи."""
    errors = []
    if secs is None:
        secs = len(turns) * (open + 2 * lead)
    try:
        sizes = validate_timing(open=open, secs=secs, fps=fps, curtain=curtain, lead=lead)
    except ValueError as exc:
        return [str(exc)]
    stride = sizes["turn_frames"]
    if len(turns) != sizes["turns"]:
        errors.append(f"turns {len(turns)} != {sizes['turns']}")
    for position, turn in enumerate(turns):
        expected = start + position * stride
        if turn.turn_start != expected:
            errors.append(f"turn {position}: start {turn.turn_start} != {expected}")
        if turn.expand_from < turn.turn_start:
            errors.append(f"turn {position}: curtain starts before the turn")
        if turn.return_to > turn.turn_end:
            errors.append(f"turn {position}: curtain ends after the turn")
        if turn.open_to - turn.open_from != sizes["frames"]:
            errors.append(f"turn {position}: open window is not {sizes['frames']} frames")
        if position and turn.expand_from < turns[position - 1].return_to:
            errors.append(f"turn {position}: overlaps the previous curtain")
    return errors


def frame_turn(turns: list[Turn], frame: int) -> tuple[Turn | None, float]:
    """Ход, накрывающий кадр, и его занавес в 0..1."""
    for turn in turns:
        if turn.expand_from <= frame < turn.return_to:
            curtain_frames = max(1, turn.open_from - turn.expand_from)
            return turn, turn.facing(frame, curtain_frames)
    return None, 0.0


def storyboard_json(turns: list[Turn], *keys: str) -> list[dict]:
    """Раскадровка для страницы: только запрошенные ключи, в порядке ходов."""
    return [{key: getattr(turn, key) for key in keys} for turn in turns]


def _main(argv: list[str]) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Раскадровка занавеса для секции.")
    parser.add_argument("--lanes", type=int, default=LANES)
    parser.add_argument("--order", default="")
    parser.add_argument("--open", type=float, default=CLIP_SECONDS)
    parser.add_argument("--secs", type=float)
    parser.add_argument("--fps", type=int, default=MASTER_FPS)
    parser.add_argument("--curtain", type=float, default=CURTAIN_SECONDS)
    parser.add_argument("--lead", type=float, default=SHOWCASE_LEAD)
    args = parser.parse_args(argv)
    order = [int(part) for part in args.order.split(",") if part] or list(range(args.lanes))
    layout = Layout(lanes=args.lanes)
    turns = storyboard(order, open=args.open, secs=args.secs, fps=args.fps,
                       lead=args.lead, curtain=args.curtain, lanes=args.lanes)
    errors = layout.errors() + storyboard_errors(
        turns, open=args.open, secs=args.secs, fps=args.fps, lead=args.lead,
        curtain=args.curtain)
    print(json.dumps({"layout": dataclasses.asdict(layout), "turns": storyboard_json(
        turns, "lane", "turn_start", "expand_from", "open_from", "open_to", "return_to",
        "turn_end"), "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
