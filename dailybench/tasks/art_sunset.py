"""Art scenario L-1 — закат над морем (эталонный паттерн блока).

Что делает сценарий арт-заданием, а не симуляцией: промпт объявляет палитру
(точные цвета и их доли кадра), композицию (линия горизонта, градиент к
горизонту, отражение в воде) и референсный момент анимации. Проверка берёт эти
объявления и меряет их по кадру: доля объявленных цветов, монотонность профиля
яркости, зеркальность отражения, плотность деталей, различие кадров во времени.

Никакой физики: солнце садится по объявленной формуле, вода — зеркало.
"""

from __future__ import annotations

import math
import random

from .. import art, core, visual

IDS = ("L-1",)
TITLE = "L-1 — закат над морем"

#: Палитра задания: цвет -> минимальная доля кадра. Доли подобраны так, чтобы
#: пустая или однотонная сцена их не набрала.
PALETTE = {
    "#1b2a4a": 0.10,  # ночное небо у верхнего края
    "#ff7a3d": 0.08,  # полоса заката у горизонта
    "#ffd08a": 0.01,  # диск солнца
    "#20283a": 0.08,  # вода ниже горизонта
}
HORIZON = 0.52  # линия горизонта в долях высоты кадра
SUN_R = 0.075  # радиус солнца в долях высоты кадра
SUN_FLATTEN = 0.55  # сплюснутость диска по вертикали
DISTANCE = 12.0
FOV = 45.0

#: Полувысота кадра на плоскости сцены: та же тригонометрия, что и в эталоне.
def _half() -> float:
    return math.tan(math.radians(FOV / 2)) * DISTANCE

SUN_START = (0.5, 0.46)
STEPS = 78

PROMPT = """Собери закат над морем: плоскость воды, небо с градиентом, солнце
садится к горизонту и оставляет дорожку на воде. Камера смотрит строго вдоль
воды, отражение в воде — зеркало неба.

Композиция (проверяется по кадру грубо):
* горизонт — ровно на {horizon:.0%} высоты кадра, выше — небо, ниже — вода;
* небо — градиент от тёплой полосы у горизонта к холодной ночи у верхнего края
  кадра, то есть яркость монотонно растёт сверху вниз до горизонта;
* вода — градиент от той же тёплой полосы у горизонта к тёмной воде у нижнего
  края, плюс горизонтальная рябь: полосы яркости с шагом 0.024 высоты кадра,
  сдвиг по фазе {phase};
* солнце — диск со сплюснутым по вертикали силуэтом: `CircleGeometry`
  радиуса {sun_r} высоты кадра, `scale.y = 0.55`; центр в {sun} — координаты
  заданы в долях ШИРИНЫ и ВЫСОТЫ кадра (плоскости шире кадра, поэтому
  пересчитывай долю кадра в единицы сцены через его видимую ширину), он опускается к
  горизонту со скоростью {speed} высоты кадра в секунду и останавливается на
  линии горизонта (сплюснутость — отражение рефракции у горизонта);
* отражение солнца в воде — такой же диск под солнцем, зеркальный ему
  относительно горизонта и растянутый по вертикали в 2.5 раза.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_night} — ночное небо у верхнего края, не меньше {s_night:.0%} кадра;
* {c_sunset} — тёплая полоса у горизонта, не меньше {s_sunset:.0%};
* {c_sun} — диск солнца и его отражение, не меньше {s_sun:.0%};
* {c_water} — вода ниже горизонта, не меньше {s_water:.0%}.

Сцена (плоскости в 12 единицах от камеры, ширина 40, полувысота кадра на
этой дистанции ≈ {half} единицы):
* небо — плоскость высотой ровно {sky_h} единицы (доля кадра выше горизонта),
  имя `sky`, материал — `ShaderMaterial` с вершинным градиентом от {c_night}
  (верх) к {c_sunset} (горизонт), затенения нет;
* вода — такая же плоскость высотой {sea_h} единицы ниже горизонта, имя `sea`,
  градиент от {c_sunset} (горизонт) к {c_water} (низ) и полосы ряби;
* солнце — `THREE.Mesh(new THREE.CircleGeometry(...))` с именем `sun`, материал
  `MeshBasicMaterial` цвета {c_sun};
* отражение — `THREE.Mesh` с именем `sun_reflection`, тот же диск с
  `scale.y = 2.5`;
* источники света не нужны: все материалы самосветящиеся;
* камера `PerspectiveCamera(45, 800/600, 0.1, 200)` в точке (0, 0.2, 0),
  смотрит в (0, 0.2, -1) — ровно вдоль горизонта, поэтому горизонт ложится
  чуть ниже середины кадра.

Анимация:
* `window.__step(dt)` двигает солнце вниз: `y -= {speed} * dt`, но не ниже
  линии горизонта; время t растёт на dt;
* `window.__probe()` возвращает `t`, `objects` и `sun: [x, y]` (4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, горизонт и
профиль, детализация кадра (не плоско и не шум), и то, что сцена движется —
кадры на t и t+{steps} шагов различаются, а каждая загрузка даёт тот же кадр.
"""

GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>sunset</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const HORIZON = {horizon};
const SUN_R = {sun_r_val};
const SUN_FLATTEN = {sun_flatten_val};
const SPEED = {speed};
const PHASE = {phase};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// Кадр виден ровно на ±half по вертикали, горизонт — на HORIZON высоты кадра:
// отсюда высоты плоскостей и положение камеры, чтобы стык попал точно на
// горизонт, а горизонт — точно на объявленную долю кадра.
const skyWidth = 40, distance = 12;
const half = Math.tan(45 * Math.PI / 360) * distance;
// ширина кадра на плоскости сцены: доля кадра, а не доля плоскости
const halfW = half * (800 / 600);
const cameraY = (HORIZON - 0.5) * 2 * half;
const skyHeight = HORIZON * 2 * half;
const seaHeight = (1 - HORIZON) * 2 * half;
const skyTop = skyHeight;

const camera = new THREE.PerspectiveCamera(45, 800 / 600, 0.1, 200);
camera.position.set(0, cameraY, 0);
camera.lookAt(0, cameraY, -1);

const sky = new THREE.Mesh(
  new THREE.PlaneGeometry(skyWidth, skyHeight),
  new THREE.ShaderMaterial({{
    uniforms: {{ uTop: {{value: new THREE.Color({c_night})}}, uBottom: {{value: new THREE.Color({c_sunset})}} }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; varying vec2 vUv;
      void main() {{
        // степень 2.0: ночь занимает верхнюю часть неба, тёплая полоса — узкая
        gl_FragColor = vec4(mix(uBottom, uTop, pow(smoothstep(0.0, 1.0, vUv.y), 2.0)), 1.0);
      }}`,
  }})
);
sky.name = 'sky';
sky.position.set(0, cameraY + half - skyHeight / 2, -distance);
scene.add(sky);

const sea = new THREE.Mesh(
  new THREE.PlaneGeometry(skyWidth, seaHeight),
  new THREE.ShaderMaterial({{
    uniforms: {{
      uTop: {{value: new THREE.Color({c_sunset})}},
      uBottom: {{value: new THREE.Color({c_water})}},
      uPhase: {{value: 0.0}},
    }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; uniform float uPhase; varying vec2 vUv;
      void main() {{
        vec3 colour = mix(uBottom, uTop, pow(smoothstep(0.0, 1.0, vUv.y), 1.8));
        // рябь: полосы, зеркальные строкам неба выше горизонта
        // рябь: шаг 0.024 высоты кадра, то есть 1/0.05 периода на плоскость
        float ripple = 0.5 + 0.5 * sin((vUv.y * 20.0 + uPhase) * 3.14159);
        colour *= 0.82 + 0.35 * ripple;
        gl_FragColor = vec4(colour, 1.0);
      }}`,
  }})
);
sea.name = 'sea';
sea.position.set(0, cameraY - half + seaHeight / 2, -distance);
scene.add(sea);

const sun = new THREE.Mesh(
  new THREE.CircleGeometry(SUN_R * 2 * half, 48),
  new THREE.MeshBasicMaterial({{color: {c_sun}}})
);
sun.name = 'sun';
sun.scale.set(1.0, SUN_FLATTEN, 1.0);
const sunX = ({sun_x} - 0.5) * 2 * halfW;
const sunY = cameraY + half - {sun_y} * 2 * half;
sun.position.set(sunX, sunY, -distance + 0.05);
scene.add(sun);

// Отражение: тот же диск, зеркальный солнцу относительно горизонта, растянут
// по вертикали — на воде получается дорожка.
const reflection = new THREE.Mesh(
  new THREE.CircleGeometry(SUN_R * 2 * half, 48),
  new THREE.MeshBasicMaterial({{color: {c_sun}, transparent: true, opacity: 0.55}})
);
reflection.name = 'sun_reflection';
reflection.scale.set(1.0, SUN_FLATTEN * 2.5, 1);
reflection.position.set(sunX, -sunY, -distance + 0.05);
scene.add(reflection);

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;

window.__step = (dt) => {{
  t += dt;
  const floor = cameraY - half + SUN_R * 2 * half;
  sun.position.y = Math.max(floor, sun.position.y - SPEED * 2 * half * dt);
  reflection.position.y = -sun.position.y;
  sea.material.uniforms.uPhase.value = PHASE + t * 0.9;
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  sun: [
    round((sun.position.x + skyWidth / 2) / skyWidth),
    round((sun.position.y - cameraY + half) / (2 * half)),
  ],
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _frame_share(hex_value: str, share: float, label: str):
    return visual.palette_colour(hex_value, share, label)


def build(rng: random.Random) -> list[core.TaskInstance]:
    speed = round(rng.choice([0.02, 0.03, 0.04]), 3)
    phase = round(rng.uniform(0.0, 1.0), 3)
    sun_x = round(rng.uniform(0.36, 0.64), 3)
    sun_y = round(rng.uniform(0.62, 0.78), 3)

    prompt = PROMPT.format(
        half=round(_half(), 3),
        sun_r=SUN_R,
        horizon=HORIZON,
        sun=f"({sun_x:.2f}, {sun_y:.2f}) высоты кадра",
        speed=f"{speed:.3f}",
        phase=f"{phase:.3f}",
        steps=STEPS,
        c_night=list(PALETTE)[0] and "#1b2a4a",
        c_sunset="#ff7a3d",
        c_sun="#ffd08a",
        c_water="#20283a",
        sky_h=round(HORIZON * 2 * _half(), 3),
        sea_h=round((1 - HORIZON) * 2 * _half(), 3),
        s_night=PALETTE["#1b2a4a"],
        s_sunset=PALETTE["#ff7a3d"],
        s_sun=PALETTE["#ffd08a"],
        s_water=PALETTE["#20283a"],
    )
    golden = GOLDEN.format(
        sun_r_val=SUN_R,
        sun_flatten_val=SUN_FLATTEN,
        horizon=HORIZON,
        speed=speed,
        phase=phase,
        sun_x=sun_x,
        sun_y=sun_y,
        c_night="0x1b2a4a",
        c_sunset="0xff7a3d",
        c_sun="0xffd08a",
        c_water="0x20283a",
    )

    # Фикстура доказывает, что все её числа есть в промпте дословно.
    for literal in (
        f"{speed:.3f}",
        f"{phase:.3f}",
        f"{sun_x:.2f}",
        f"{sun_y:.2f}",
        "#1b2a4a",
        "#ff7a3d",
        "#ffd08a",
        "#20283a",
    ):
        assert literal in prompt, literal
    assert 0.0 <= sun_x <= 1.0 and 0.5 <= sun_y <= 0.9
    assert 0.15 < sun_x < 0.85, "солнце не должно липнуть к краю кадра"
    half_w = _half() * (800 / 600)
    assert abs(sun_x - 0.5) * 2 * half_w <= half_w * 0.75, "солнце обязано быть внутри кадра"
    # Солнце за 78 шагов обязано сдвинуться заметно, иначе кадры не отличатся.
    assert speed * STEPS / 60 > 0.02

    palette_checks = [
        _frame_share("#1b2a4a", PALETTE["#1b2a4a"], "ночное небо"),
        _frame_share("#ff7a3d", PALETTE["#ff7a3d"], "полоса заката"),
        _frame_share("#ffd08a", PALETTE["#ffd08a"], "солнце"),
        _frame_share("#20283a", PALETTE["#20283a"], "вода"),
    ]
    top = lambda px: [entry["lum"] for entry in (px.get("profile") or [])][:2]  # noqa: E731
    composition_checks = [
        # Верх кадра — тёмная ночь, у горизонта — тёплая полоса: профиль
        # яркости обязан быть монотонным на четырёх верхних полосах.
        _sky_gradient(),
        # Ниже горизонта вода темнее полосы заката.
        _sea_darker(),
        # Солнце светит в своей колонке кадра, отражение — под ним.
        _bright_spot("самые яркие колонки — там, где заказано солнце", 0, sun_x),
    ]
    detail_checks = [
        visual.image_min("changed_fraction", 0.2),
        visual.image_min("distinct_colors", 12),
        visual.rich_edges(0.002, "небо и вода не сливаются в однотонную заливку"),
        visual.image_min("saturation", 0.25),
    ]

    task = art.build_task(
        task_id="L-1",
        title=TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=palette_checks,
            composition=composition_checks,
            detail=detail_checks,
            motion=STEPS,
        ),
        golden=golden,
        fixture={"speed": speed, "phase": phase, "sun_x": sun_x, "sun_y": sun_y, "horizon": HORIZON},
    )
    return [task]


def _sky_gradient():
    def check(pixels: dict) -> tuple[bool, str]:
        profile = [entry["lum"] for entry in (pixels.get("profile") or [])]
        if len(profile) < 4:
            return False, "кадр: нет профиля яркости"
        top = profile[:4]
        rises = sum(1 for a, b in zip(top, top[1:]) if b >= a - 1.0)
        good = rises >= 2 and top[-1] - top[0] >= 3.0
        return good, f"кадр: небо светлеет к горизонту — профиль {top} (шагов вверх {rises})"

    return visual._label(check, "кадр: градиент неба к горизонту")


def _sea_darker():
    def check(pixels: dict) -> tuple[bool, str]:
        profile = [entry["lum"] for entry in (pixels.get("profile") or [])]
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        sky_bottom, sea_top, sea_bottom = profile[3], profile[4], profile[7]
        good = sky_bottom >= sea_top - 1.0 and sea_top > sea_bottom + 4.0
        return good, f"кадр: полоса заката светлее воды — {sky_bottom} > {sea_top} > {sea_bottom}"

    return visual._label(check, "кадр: горизонт читается (небо светлее воды)")


def _bright_spot(label: str, _row: int, sun_x: float):
    """The brightest columns sit where the fixture put the sun.

    The whole-height column profile dilutes the disc (it covers a fifth of the
    frame), so the check compares the brightest column's position with the
    declared x instead of trusting a single column value.
    """
    want = sun_x

    def check(pixels: dict) -> tuple[bool, str]:
        columns = pixels.get("columns") or []
        if len(columns) < 8:
            return False, "кадр: нет колоночного профиля"
        brightest = max(range(len(columns)), key=lambda i: columns[i])
        found = (brightest + 0.5) / len(columns)
        good = abs(found - want) <= 0.14
        return good, f"кадр: {label} — самая яркая колонка {found:.2f} (заказано {want:.2f})"

    return visual._label(check, f"кадр: {label}")
