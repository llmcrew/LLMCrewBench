"""Category V — image input (VQA), three showcase scenarios outside the score.

The block is deliberately small and each scenario is unanswerable from the
prompt alone: the picture carries the fact. All three are rendered with Pillow
into one fixed geometry (``core.ASSET_SIZE``, 8-bit RGB PNG) and presented as a
local ``image_url`` part, never as base64 text and never from an external URL.

* V-1 reads a bar chart: which month has the tallest bar.
* V-2 reads spatial relations: which figure is to the left.
* V-3 traces a diagram: following the arrows from the start node.

Every oracle is computed from the same drawing code that renders the bytes, and
its ``mutant`` is the plausible wrong answer (the second-tallest bar, the right
figure, the last-drawn node). The two checks separate an exact answer from an
answer that merely contains the right token: a reply wrapped in a sentence can
earn the region half, and the mutant cannot earn either.

The category is reported next to K as a showcase block: a text-only endpoint is
skipped as *unsupported* by the runner's capability probe and is not read as an
accuracy of zero.
"""

from __future__ import annotations

import io
import random
import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .. import core

IDS: tuple[str, ...] = ("V-1", "V-2", "V-3")

TITLE = "Image input (VQA)"

_POINT_EXACT = 0.5
_POINT_REGION = 0.5

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun")
COLORS: tuple[tuple[str, str], ...] = (
    ("red", "красный"),
    ("blue", "синий"),
    ("green", "зелёный"),
    ("yellow", "жёлтый"),
)
SHAPES = ("circle", "square")
NODES = ("1", "2", "3", "4", "5")

_BG = (250, 250, 248)
_INK = (30, 34, 40)
_GRID = (210, 214, 220)


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _asset(path: str, image: Image.Image) -> core.Asset:
    """Freeze one drawn frame as a typed PNG asset."""
    raw = _png(image)
    return core.Asset(path=path, media_type="image/png", data=raw)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    width: int = 4,
) -> None:
    import math

    draw.line([start, end], fill=_INK, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    for delta in (0.45, -0.45):
        corner = (
            end[0] - length * math.cos(angle - delta),
            end[1] - length * math.sin(angle - delta),
        )
        draw.line([end, corner], fill=_INK, width=width)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def _chart(rng: random.Random) -> core.TaskInstance:
    """A bar chart; the answer is the tallest month, read off the picture."""
    values = rng.sample(range(2, 11), len(MONTHS))
    tallest = max(range(len(MONTHS)), key=lambda index: values[index])
    second = sorted(range(len(MONTHS)), key=lambda index: values[index])[-2]
    answer = MONTHS[tallest]
    mutant = MONTHS[second]

    image = Image.new("RGB", core.ASSET_SIZE, _BG)
    draw = ImageDraw.Draw(image)
    width, height = core.ASSET_SIZE
    left, right, top, bottom = 70, width - 40, 50, height - 70
    draw.line([(left, bottom), (right, bottom)], fill=_INK, width=3)
    draw.line([(left, top), (left, bottom)], fill=_INK, width=3)
    slot = (right - left) / len(MONTHS)
    font = _font(26)
    for index, (month, value) in enumerate(zip(MONTHS, values)):
        bar_height = (bottom - top - 30) * value / max(values)
        x0 = left + slot * index + slot * 0.2
        x1 = left + slot * (index + 1) - slot * 0.2
        draw.rectangle([x0, bottom - bar_height, x1, bottom], fill=(70, 120, 190))
        draw.text(
            ((x0 + x1) / 2, bottom + 10), month.upper(), fill=_INK, font=font, anchor="ma"
        )
    draw.text((left, top - 34), "assets, units", fill=_GRID, font=_font(20))

    prompt = (
        "На изображении столбчатая диаграмма за полугодие: по горизонтали месяцы "
        "JAN–JUN, по вертикали значения. Ответь одним словом: у какого месяца самый "
        "высокий столбик?"
    )
    return core.TaskInstance(
        id="V-1",
        category="V",
        title="Tallest bar of a chart",
        prompt=prompt,
        max_points=1.0,
        workspace={},
        assets=[_asset("chart.png", image)],
        oracle={
            "answer": answer,
            "mutant": mutant,
            "reference": answer,
            "layout": {"months": list(MONTHS), "values": values},
        },
        meta={"asset_kind": "bar-chart"},
    )


def _spatial(rng: random.Random) -> core.TaskInstance:
    """Two figures side by side; the answer is the colour of the left one."""
    left_color, right_color = rng.sample(COLORS, 2)
    left_shape, right_shape = rng.sample(SHAPES, 2)
    answer = left_color[1]
    mutant = right_color[1]

    image = Image.new("RGB", core.ASSET_SIZE, _BG)
    draw = ImageDraw.Draw(image)
    width, height = core.ASSET_SIZE
    radius = 92
    gap = rng.randint(70, 130)
    left_center = (width / 2 - radius - gap / 2, height / 2 + rng.randint(-40, 40))
    right_center = (width / 2 + radius + gap / 2, height / 2 + rng.randint(-40, 40))

    def figure(center: tuple[float, float], shape: str, color: tuple[int, int, int]) -> None:
        box = [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]
        if shape == "circle":
            draw.ellipse(box, fill=color, outline=_INK, width=3)
        else:
            draw.rectangle(box, fill=color, outline=_INK, width=3)

    palette = {"red": (200, 60, 55), "blue": (55, 100, 190),
               "green": (60, 150, 90), "yellow": (225, 190, 60)}
    figure(left_center, left_shape, palette[left_color[0]])
    figure(right_center, right_shape, palette[right_color[0]])
    draw.line([(width / 2, 40), (width / 2, height - 40)], fill=_GRID, width=2)

    prompt = (
        "На изображении две фигуры. Ответь одним словом: какого цвета фигура, "
        "которая находится ЛЕВЕЕ? (красный, синий, зелёный или жёлтый)"
    )
    return core.TaskInstance(
        id="V-2",
        category="V",
        title="Left figure of a pair",
        prompt=prompt,
        max_points=1.0,
        workspace={},
        assets=[_asset("spatial.png", image)],
        oracle={
            "answer": answer,
            "mutant": mutant,
            "reference": answer,
            "layout": {
                "left": {"shape": left_shape, "color": left_color[0]},
                "right": {"shape": right_shape, "color": right_color[0]},
            },
        },
        meta={"asset_kind": "spatial"},
    )


def _trace(rng: random.Random) -> core.TaskInstance:
    """Five nodes and a jumbled arrow chain; the answer is the final node."""
    order = ["1", *rng.sample(["2", "3", "4", "5"], 4)]
    answer = order[-1]
    mutant = rng.choice([node for node in NODES if node != answer])

    image = Image.new("RGB", core.ASSET_SIZE, _BG)
    draw = ImageDraw.Draw(image)
    width, height = core.ASSET_SIZE
    radius = 34
    columns = [110, 270, 430, 590, width - 80]
    rows = [130, height - 130]
    positions: dict[str, tuple[int, int]] = {}
    for index, node in enumerate(NODES):
        column, row = columns[index], rows[index % 2]
        if index == 0:
            positions[node] = (column, height // 2)
        elif index == len(NODES) - 1:
            positions[node] = (column, height // 2)
        else:
            positions[node] = (column, row)
    font = _font(30)
    for first, second in zip(order, order[1:]):
        start = positions[first]
        end = positions[second]
        # Stop the line at the circle edge so the arrowhead stays visible.
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        trim = radius + 8
        _draw_arrow(
            draw,
            (start[0] + dx / length * trim, start[1] + dy / length * trim),
            (end[0] - dx / length * trim, end[1] - dy / length * trim),
        )
    for node, (x, y) in positions.items():
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(240, 242, 246), outline=_INK, width=3)
        draw.text((x, y), node, fill=_INK, font=font, anchor="mm")

    prompt = (
        "На изображении схема из узлов и стрелок. Начни с узла 1 и двигайся строго "
        "по направлению стрелок. Ответь одной цифрой: в каком узле путь закончится?"
    )
    return core.TaskInstance(
        id="V-3",
        category="V",
        title="Trace a diagram",
        prompt=prompt,
        max_points=1.0,
        workspace={},
        assets=[_asset("trace.png", image)],
        oracle={
            "answer": answer,
            "mutant": mutant,
            "reference": answer,
            "layout": {"order": order},
        },
        meta={"asset_kind": "trace"},
    )


_SCENARIOS = (("V-1", _chart), ("V-2", _spatial), ("V-3", _trace))


# --------------------------------------------------------------------------- #
# Category module
# --------------------------------------------------------------------------- #


def generate(rng: random.Random, cfg: Any) -> list[core.TaskInstance]:
    """Only the research profile carries the VQA block."""
    if core.profile_of(cfg) != "research-v1":
        return []
    tasks = [builder(rng) for _, builder in _SCENARIOS]
    for task in tasks:
        mutant = str(task.oracle["mutant"])
        if mutant == task.oracle["answer"]:
            raise RuntimeError(f"{task.id}: мутант совпал с ответом")
    return tasks


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip(".!?,;:«»\"'")).lower()


def _mentions(text: str, answer: str) -> bool:
    """The answer as a whole token anywhere in the reply (the region check)."""
    return bool(re.search(rf"(?<!\w){re.escape(answer.lower())}(?!\w)", text.lower()))


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    """Half a point for the exact token, half for a reply containing it."""
    score = core.Score(task_id=task.id, category="V")
    answer = str(task.oracle["answer"])
    text = tr.final_text
    exact = _normalise(text) == _normalise(answer)
    region = _mentions(text, answer)
    score.add(
        "exact",
        exact,
        _POINT_EXACT,
        f"ответ {text!r} равен эталону {answer!r}" if exact else f"получено {text!r}, эталон {answer!r}",
    )
    score.add(
        "region",
        region,
        _POINT_REGION,
        f"эталон {answer!r} найден в ответе" if region else f"эталона {answer!r} в ответе нет",
    )
    return score


core.register(
    key="V",
    title=TITLE,
    points=3.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=(),
    expected_ids_by_profile={"research-v1": IDS},
)
