"""Category L — free-form art: one self-directed visual work.

The model receives a single free brief: idea, subject, style, palette and
means of expression are its own. The deliverable is a standalone HTML page
(HTML/CSS, SVG, Canvas, WebGL; the local Three.js build is available but
optional). The automatic check is technical delivery only — the page must load
and render a frame; artistic quality is judged by people, not heuristics.

L does not enter the 100-point A–J leaderboard and is reported separately.
"""

from __future__ import annotations

import random

from .. import art, core
from ..config import RunConfig

IDS = ("L-1",)

TITLE = "Free-form art"

BRIEF = """\
Создай одну самостоятельную визуальную работу. Идею, сюжет, стиль, палитру,
тональность и средства реализации выбери сам. Покажи собственное художественное
решение: работу будет оценивать человек по результату, а не по соблюдению
рецепта.
"""

# Smoke fixture for the generator/oracle path, not a model of "correct"
# aesthetics: an inline SVG scene that renders without scripts or hooks.
GOLDEN = """\
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Сумеречный маяк</title>
<style>
  html, body { margin: 0; height: 100%; background: #0b1020; }
  svg { display: block; width: 100vw; height: 100vh; }
</style>
</head>
<body>
<svg viewBox="0 0 960 540" xmlns="http://www.w3.org/2000/svg">

  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#101b3d"/>
      <stop offset="0.55" stop-color="#3d2f63"/>
      <stop offset="1" stop-color="#b3542e"/>
    </linearGradient>
    <linearGradient id="sea" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#22143a"/>
      <stop offset="1" stop-color="#070a18"/>
    </linearGradient>
  </defs>
  <rect width="960" height="360" fill="url(#sky)"/>
  <circle cx="760" cy="300" r="46" fill="#f4c95d"/>
  <rect y="360" width="960" height="180" fill="url(#sea)"/>
  <path d="M812 300 L956 258 L956 342 Z" fill="rgba(244,201,93,0.28)"/>
  <polygon points="470,360 500,240 530,240 560,360" fill="#141a2e"/>
  <rect x="497" y="208" width="36" height="34" fill="#f4c95d"/>
  <polygon points="470,360 500,240 497,242 467,360" fill="#1b2340"/>
  <g stroke="#f4c95d" stroke-width="2" opacity="0.85">
    <line x1="515" y1="140" x2="640" y2="90"/>
    <line x1="515" y1="140" x2="515" y2="60"/>
    <line x1="515" y1="140" x2="392" y2="92"/>
  </g>
  <g fill="#ffffff" opacity="0.5">
    <circle cx="120" cy="80" r="1.6"/><circle cx="210" cy="50" r="1.2"/>
    <circle cx="300" cy="110" r="1.4"/><circle cx="70" cy="180" r="1"/>
    <circle cx="380" cy="60" r="1.3"/><circle cx="175" cy="150" r="1.1"/>
  </g>
</svg>
</body>
</html>
"""


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    """One free-form work; the brief does not depend on the seed."""
    task = art.build_task(
        task_id="L-1",
        title="Свободная работа",
        prompt=BRIEF,
        golden=GOLDEN,
    )
    task.meta["seed_sensitive"] = False
    return [task]


core.register(
    key="L",
    title=TITLE,
    points=art.POINTS_PER_CHECK * len(IDS),
    generate=generate,
    evaluate=art.evaluate,
    expected_ids=IDS,
)