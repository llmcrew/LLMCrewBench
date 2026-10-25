"""Category L — free-form art: one visual work per topic.

Every task hands the model a theme and nothing else: idea, composition, palette,
style and means are its own. The deliverable is a standalone HTML page
(HTML/CSS, SVG, Canvas, WebGL; the local Three.js build is available but
optional). The only automatic check is technical delivery — the page loads and
renders a frame; artistic quality is judged by people.

The topics are the ones the earlier strict block used (sunset, neon grid,
mandala, city, bridge, galaxy, fireworks, aurora, night sky, fractal, metal,
tunnel, lava lamp, water, flow, trails, eight lofi rain scenes, five orbital
disasters), but the palette shares, composition formulas, scene API and
animation are no longer required.

L does not enter the 100-point A–J leaderboard and is reported separately.
"""

from __future__ import annotations

import random

from .. import art, core
from ..config import RunConfig

#: Theme of every task, in the order the block is played. The free brief (L-1)
#: comes first; the rest name a subject and leave everything else open.
TOPICS: dict[str, str] = {
    "L-1": "Свободная работа",
    "L-2": "Неоновая сетка",
    "L-3": "Мандала",
    "L-4": "Ночной город",
    "L-5": "Мост",
    "L-6": "Галактика из частиц",
    "L-7": "Фейерверк",
    "L-8": "Аврора",
    "L-9": "Звёздное небо над горами",
    "L-10": "Фрактал",
    "L-11": "Жидкий металл и стекло",
    "L-12": "Бесконечный туннель",
    "L-13": "Лавовая лампа",
    "L-14": "Водная поверхность",
    "L-15": "Поток частиц",
    "L-16": "Следы сфер",
    "L-17": "Комната с дождём за окном",
    "L-18": "Неоновая лапшичная под дождём",
    "L-19": "Трамвай сквозь дождливый город",
    "L-20": "Оранжерея на станции под метеорным дождём",
    "L-21": "Аналоговая рубка с дождём за стеклом",
    "L-22": "Робот-ремонтник под дождём",
    "L-23": "Космическая заправка под дождём",
    "L-24": "Автобусная остановка в неоне",
    "L-25": "Ядерные взрывы на Земле из космоса",
    "L-26": "Метеорит входит в атмосферу",
    "L-27": "Столкновение двух планет",
    "L-28": "Цепная реакция обломков",
    "L-29": "Станция уходит, Земля позади",
    "L-30": "Архив древних карт",
    "L-31": "Коралловый риф без людей",
    "L-32": "Заброшенный карьер",
    "L-33": "Читальный зал после закрытия",
    "L-34": "Мастерская часовщика",
    "L-35": "Путь перелётных птиц через водно-болотные угодья",
    "L-36": "Палеонтологический раскоп",
    "L-37": "Рынок на рассвете",
    "L-38": "Детская площадка в масштабе насекомого",
    "L-39": "Радиотелескоп на пустынном плато",
}

IDS = tuple(TOPICS)

#: Baseline still asks the first twenty-nine themes; the archive, reef, quarry,
#: reading room, clock workshop, migration, dig, market, insect-scale playground
#: and radio telescope only enter the stretch suite.
IDS_BASELINE = tuple(f"L-{index}" for index in range(1, 30))

IDS_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "pilot": IDS_BASELINE,
    "stretch-v1": IDS,
    # The research profile draws no art work.
    "research-v1": (),
}

TITLE = "Free-form art"

BRIEF = """\
Создай одну самостоятельную визуальную работу. Идею, сюжет, стиль, палитру,
тональность и средства реализации выбери сам. Покажи собственное художественное
решение: работу будет оценивать человек по результату, а не по соблюдению
рецепта.
"""

#: Тема заданной работы: сюжет назван, всё остальное — на усмотрение автора.
TOPIC_BRIEF = """\
Создай самостоятельную визуальную работу на тему «{topic}». Как её показать —
композиция, палитра, стиль, статика или движение, чем рисовать — решай сам.
Работу будет оценивать человек по результату, а не по соблюдению рецепта.
"""


def _smoke_page(title: str, hue: int) -> str:
    """Дымовая страница для selfcheck: рисуется без скриптов и хуков.

    Это не эталон вкуса и не образец композиции — только доказательство, что
    путь «страница -> кадр -> оценка» живой у каждого задания.
    """
    warm = 40 + hue % 120
    return f"""\
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: hsl({hue} 45% 8%); }}
  svg {{ display: block; width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<svg viewBox="0 0 960 540" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="hsl({hue} 60% 18%)"/>
      <stop offset="1" stop-color="hsl({(hue + 60) % 360} 55% 6%)"/>
    </linearGradient>
    <radialGradient id="sun">
      <stop offset="0" stop-color="hsl({warm} 95% 72%)"/>
      <stop offset="1" stop-color="hsl({warm} 90% 48%)"/>
    </radialGradient>
  </defs>
  <rect width="960" height="540" fill="url(#bg)"/>
  <circle cx="640" cy="210" r="86" fill="url(#sun)"/>
  <path d="M0 400 Q 240 350 480 400 T 960 400 L960 540 L0 540 Z"
        fill="hsl({(hue + 20) % 360} 40% 14%)"/>
  <g fill="hsl({(hue + 200) % 360} 80% 70%)" opacity="0.75">
    <circle cx="180" cy="120" r="3"/><circle cx="260" cy="86" r="2"/>
    <circle cx="340" cy="150" r="2.5"/><circle cx="120" cy="200" r="2"/>
  </g>
  <text x="48" y="500" font-family="ui-sans-serif, sans-serif" font-size="30"
        fill="hsl({hue} 30% 92%)" opacity="0.9">{title}</text>
</svg>
</body>
</html>
"""


GOLDEN = _smoke_page("Свободная работа", 220)


def _in_sentence(topic: str) -> str:
    """Тема внутри фразы: первая буква строчная, имена собственные сохраняются."""
    return topic[:1].lower() + topic[1:]


def build(task_id: str, rng: random.Random) -> core.TaskInstance:
    topic = TOPICS[task_id]
    free = task_id == "L-1"
    return art.build_task(
        task_id=task_id,
        title=topic if free else f"{task_id} — {topic}",
        prompt=BRIEF if free else TOPIC_BRIEF.format(topic=_in_sentence(topic)),
        golden=_smoke_page(topic, rng.randrange(360)),
    )


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    """The profile's themes; the briefs do not depend on the seed."""
    profile = core.profile_of(cfg)
    ids = IDS_BY_PROFILE.get(profile, IDS_BASELINE)
    tasks = [build(task_id, rng) for task_id in ids]
    for task in tasks:
        task.meta["seed_sensitive"] = False
    return tasks


core.register(
    key="L",
    title=TITLE,
    points=art.POINTS_PER_CHECK * len(IDS_BASELINE),
    generate=generate,
    evaluate=art.evaluate,
    expected_ids=IDS_BASELINE,
    expected_ids_by_profile={"stretch-v1": IDS, "research-v1": ()},
)
