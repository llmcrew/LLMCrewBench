"""Art scenarios L-6..L-9 — частицы и небо (галактика, фейерверк, аврора, звёзды).

Те же правила, что и в ``art_sunset``: промпт объявляет палитру (точные цвета и
их минимальные доли кадра), композицию (радиальный спад, места залпов, горизонт
гор) и движение, а проверка меряет объявленное по кадру. Никакой физики: частицы
летят по объявленной формуле, галактика вращается дифференциально, лента авроры
идёт волной, звёзды мерцают по фазе.

Случайность — только явный ЛЦГ с константами, напечатанными в промпте, поэтому
кадр побитово воспроизводим, а фикстура (seed и производные от rng числа) лежит
в промпте дословно.
"""

from __future__ import annotations

import math
import random

from .. import art, core, visual

IDS = ("L-6", "L-7", "L-8", "L-9")
TITLES = {
    "L-6": "L-6 — галактика из частиц",
    "L-7": "L-7 — фейерверк",
    "L-8": "L-8 — аврора",
    "L-9": "L-9 — звёздное небо над горами",
}

#: Единственный генератор случайности слайса; константы печатаются в промпте.
LCG_TEXT = (
    '* случайность — только генератор: состояние `s` — целое, старт `S0 = {seed}`, '
    'шаг `s = (1664525 * s + 1013904223) % 4294967296`, число `u = s / 4294967296`; '
    'порядок обращений задан ниже — повторяй ровно, `Math.random()` запрещён;'
)

FOV = 45.0
ASPECT = 800 / 600
STEPS = {"L-6": 72, "L-7": 72, "L-8": 72, "L-9": 84}

# --------------------------------------------------------------------------- #
# L-6 — галактика из частиц
# --------------------------------------------------------------------------- #

GALAXY_PALETTE = {
    "#ffbf00": 0.0006,  # тёплое ядро и внутренние рукава
    "#ff0080": 0.0022,  # средние рукава
    "#0cb7f0": 0.0022,  # периферия и пылевое гало
    "#1d1638": 0.47,  # ночное небо вокруг диска
}
GALAXY = {
    "arms_n": 7000,
    "core_n": 2500,
    "haze_n": 1500,
    "cam_z": 10.0,
    "r_frac": 0.44,  # радиус диска в полувысотах кадра
    "arms": 2,
    "turns": 1.6,  # оборотов рукава от ядра к краю
    "scatter": 0.5,  # разброс рукава, радиан
    "w0": 0.55,  # угловая скорость на радиусе ядра, рад/с
    "a0": 0.25,  # сглаживание дифференциального вращения
    "q_min": 0.14,  # ближняя к ядру доля радиуса
    "core_until": 0.38,  # доля радиуса, до которой точки красятся как ядро
    "arm_until": 0.70,  # ... и как средние рукава
    "core_r": 0.22,  # радиус сгустка ядра, доля R
    "empty_max": 0.36,  # доля кадра, которую разрешено оставить «пустой»
}

GALAXY_PROMPT = """Собери галактику из частиц: спиральный диск из точек, тёплое
ядро-сгусток, бирюзовая периферия и ночное небо вокруг.

Камера `PerspectiveCamera(45, 800/600, 0.1, 200)` в точке (0, 0, {cam_z}),
смотрит в начало координат и видит галактику плашмя; полувысота кадра на
дистанции камеры ≈ {half} единицы.

Композиция (проверяется по кадру грубо):
* диск галактики — радиус {r_frac} полувысоты кадра, центр кадра;
* яркость монотонно падает от центра к краям: центральная область заметно
  ярче углов, полосы профиля в середине кадра — ярче верхней и нижней;
* «пустого» — пикселей, у которых максимум канала не больше 70, — не больше
  {empty_max} кадра: небо вокруг диска светлее чёрного и имеет собственный тон.

Рукава: {arms_n} точек, две ветви. Для каждой точки возьми из ЛЦГ четыре числа
подряд (`u`, `v`, `w`, `g` — последнее это `u1+u2+u3-1.5`, три числа ЛЦГ), затем
* `q = {q_min} + {q_span} * u^0.62` — доля радиуса;
* `r = {r_frac} * полувысота * q`;
* `theta = {turns} * 2*pi*q + pi*floor(v * {arms}) + g * {scatter} * (1 - 0.45*q)`
  — два рукава, закрученных на {turns} оборота, с разбросом по `theta`;
* `z = (w - 0.5) * 0.10 * r * (0.4 + q)` — тонкий диск.
Цвет точки выбирается по `q`: до {core_until} — ядро, до {arm_until} — средние
рукава, дальше — периферия. Дополнительно к рукавам: {core_n} точек ядра-сгустка (`a = 2*pi*u`,
`r = {core_r} * R * u^0.55`, `z = (u-0.5) * 0.09 * R`) и {haze_n} точек пылевого
гало (`r = R * (0.60 + 0.85*u)`, `a = 2*pi*u`) на периферии.

Точки — `THREE.Points` с `PointsMaterial({{sizeAttenuation: false, transparent:
true, blending: THREE.AdditiveBlending, depthTest: false}})`, размеры 3.2, 2.8 и
2.4 пикселя по зонам, имена `stars_core`, `stars_arm`, `stars_edge`; в ядре —
аддитивное пятно `core_glow` (плоскость с радиальным градиентом на канве,
размер 0.55 * R).

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_core} — тёплое ядро и внутренние рукава, не меньше {s_core} кадра;
* {c_arm} — средние рукава, не меньше {s_arm};
* {c_out} — периферия и гало, не меньше {s_out};
* {c_sky} — ночное небо вокруг диска, не меньше {s_sky}.

Анимация — дифференциальное вращение: `theta += {w0} * dt / ({a0} + r/R)`, то
есть внутренние точки обгоняют внешние; время t растёт на dt. За {steps} шагов
диск обязан заметно повернуться.

* `window.__step(dt)` двигает время и вращает диск;
* `window.__probe()` возвращает `t`, `spin` (накопленный угол по формуле
  `spin += {w0} * dt`), `objects` и `rings` — число точек в четырёх кольцевых
  зонах радиуса.

Проверяется: страница и кадр (5 проверок), палитра по долям, радиальный спад
яркости и заполнение кадра, детализация (не плоско и не шум) и то, что диск
вращается: кадры на t и t+{steps} шагов различаются, а каждая загрузка даёт тот
же кадр.

Свойства фикстуры, обязанные попасть в промпт дословно: seed {seed}, число
точек {arms_n}/{core_n}/{haze_n}, радиус {r_frac} полувысота кадра, скорость
{w0} рад/с.

{lcg}
"""

GALAXY_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>galaxy</title>
<style>html,body{{margin:0;background:#05060f;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const ARMS_N = {arms_n}, CORE_N = {core_n}, HAZE_N = {haze_n};
const SEED = {seed};
const CAM_Z = {cam_z}, FOV = 45, PIX = 600;
const HALF = Math.tan(FOV * Math.PI / 360) * CAM_Z;
const R = {r_frac} * HALF;
const ARMS = {arms}, TURNS = {turns}, SCATTER = {scatter}, Q_MIN = {q_min}, W0 = {w0}, A0 = {a0};
const CORE_R = {core_r};
const CORE_UNTIL = {core_until}, ARM_UNTIL = {arm_until};
const COLORS = ['{c_core}', '{c_arm}', '{c_out}'];
const SIZES = [3.2, 2.8, 2.4];

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, PIX * 1.8, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, 800 / PIX, 0.1, 200);
camera.position.set(0, 0, CAM_Z);
camera.lookAt(0, 0, 0);

// Небо: градиентная плоскость за диском; размер посчитан на её дистанции.
const skyDist = CAM_Z + 1, skyHalf = Math.tan(FOV * Math.PI / 360) * skyDist;
const skyCanvas = document.createElement('canvas');
skyCanvas.width = skyCanvas.height = 128;
const skyCtx = skyCanvas.getContext('2d');
const skyGrad = skyCtx.createRadialGradient(64, 64, 0, 64, 64, 64);
skyGrad.addColorStop(0, '{c_sky_in}');
skyGrad.addColorStop(0.55, '{c_sky}');
skyGrad.addColorStop(1, '{c_sky_out}');
skyCtx.fillStyle = skyGrad;
skyCtx.fillRect(0, 0, 128, 128);
const skyTex = new THREE.CanvasTexture(skyCanvas);
skyTex.colorSpace = THREE.SRGBColorSpace;
const sky = new THREE.Mesh(new THREE.PlaneGeometry(skyHalf * 2 * 800 / PIX, skyHalf * 2),
  new THREE.MeshBasicMaterial({{map: skyTex, depthTest: false}}));
sky.name = 'sky';
sky.position.z = -1;
scene.add(sky);

// Точки по зонам: {{r, th, z}} — из формулы рукавов и сгустков.
const bands = [[], [], []];
const zone = (q) => (q < CORE_UNTIL ? 0 : q < ARM_UNTIL ? 1 : 2);
for (let i = 0; i < ARMS_N; i++) {{
  const u = next(), v = next(), w = next();
  const g = next() + next() + next() - 1.5;
  const q = Q_MIN + (1 - Q_MIN) * Math.pow(u, 0.62);
  const theta = TURNS * 2 * Math.PI * q + Math.PI * Math.floor(v * ARMS) + g * SCATTER * (1 - 0.45 * q);
  const r = R * q;
  bands[zone(q)].push({{r: r, th: theta, z: (w - 0.5) * 0.10 * r * (0.4 + q)}});
}}
for (let i = 0; i < CORE_N; i++) {{
  const a = 2 * Math.PI * next();
  const r = CORE_R * R * Math.pow(next(), 0.55);
  bands[0].push({{r: r, th: a, z: (next() - 0.5) * 0.09 * R}});
}}
for (let i = 0; i < HAZE_N; i++) {{
  const a = 2 * Math.PI * next();
  const r = R * (0.60 + 0.85 * next());
  bands[2].push({{r: r, th: a, z: (next() - 0.5) * 0.20 * R}});
}}

const names = ['stars_core', 'stars_arm', 'stars_edge'];
const objects = [];
for (let b = 0; b < 3; b++) {{
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(bands[b].length * 3), 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({{
    size: SIZES[b], sizeAttenuation: false, color: new THREE.Color(COLORS[b]),
    transparent: true, blending: THREE.AdditiveBlending, depthTest: false,
  }}));
  pts.name = names[b];
  scene.add(pts);
  objects.push(pts);
}}

const glowCanvas = document.createElement('canvas');
glowCanvas.width = glowCanvas.height = 128;
const glowCtx = glowCanvas.getContext('2d');
const glowGrad = glowCtx.createRadialGradient(64, 64, 0, 64, 64, 64);
glowGrad.addColorStop(0, 'rgba(255,233,184,0.95)');
glowGrad.addColorStop(0.45, 'rgba(255,205,150,0.30)');
glowGrad.addColorStop(1, 'rgba(255,233,184,0)');
glowCtx.fillStyle = glowGrad;
glowCtx.fillRect(0, 0, 128, 128);
const glowTex = new THREE.CanvasTexture(glowCanvas);
glowTex.colorSpace = THREE.SRGBColorSpace;
const glow = new THREE.Mesh(new THREE.PlaneGeometry(0.55 * R * 2, 0.55 * R * 2),
  new THREE.MeshBasicMaterial({{map: glowTex, transparent: true, depthTest: false, blending: THREE.AdditiveBlending}}));
glow.name = 'core_glow';
glow.position.z = 0.01;
scene.add(glow);

const place = () => {{
  for (let b = 0; b < 3; b++) {{
    const attr = objects[b].geometry.attributes.position, list = bands[b];
    for (let i = 0; i < list.length; i++) {{
      const p = list[i];
      attr.setXYZ(i, p.r * Math.cos(p.th), p.r * Math.sin(p.th), p.z);
    }}
    attr.needsUpdate = true;
  }}
}};

let t = 0, spin = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;
const rings = () => [0, 1, 2, 3].map((k) => {{
  let n = 0;
  for (const list of bands) for (const p of list) if (Math.min(3, Math.floor(p.r / R * 4)) === k) n += 1;
  return n;
}});

window.__step = (dt) => {{
  t += dt;
  spin += W0 * dt;
  for (const list of bands) for (const p of list) p.th += W0 * dt / (A0 + p.r / R);
  place();
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  spin: round(spin),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  rings: rings(),
}});

place();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _galaxy_falloff():
    """Ядро ярче краёв: центральная область и середина профиля впереди углов."""

    def check(pixels: dict) -> tuple[bool, str]:
        regions = pixels.get("regions") or []
        profile = [entry["lum"] for entry in (pixels.get("profile") or [])]
        if len(regions) < 9 or len(profile) < 8:
            return False, "кадр: нет областей или профиля"
        lum = lambda rgb: 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]  # noqa: E731
        centre = lum(regions[4])
        corners = [lum(regions[i]) for i in (0, 2, 6, 8)]
        middle = max(profile[3], profile[4])
        edges = max(profile[0], profile[7])
        good = centre > max(corners) + 6.0 and middle > edges + 4.0
        return good, (
            f"кадр: яркость падает от центра к краям — центр {centre:.1f} против углов "
            f"{[round(c, 1) for c in corners]}, середина {middle:.1f} против краёв {edges:.1f}"
        )

    return visual._label(check, "кадр: яркость галактики падает от центра к краям")


def _galaxy_filled():
    """«Пустого» не больше четверти: почти чёрные пиксели — исключение."""

    def check(pixels: dict) -> tuple[bool, str]:
        histogram = pixels.get("histogram") or []
        if not histogram:
            return False, "кадр: нет гистограммы"
        empty = sum(float(share) for r, g, b, share in histogram if max(r, g, b) <= 70)
        good = empty <= 0.36
        return good, f"кадр: пустого {empty:.4f} (нужно <= 0.36)"

    return visual._label(check, "кадр: небо и гало закрывают кадр (пустого <= 0.36)")


# --------------------------------------------------------------------------- #
# L-7 — фейерверк
# --------------------------------------------------------------------------- #

FIRE_PALETTE = {
    "#ff0051": 0.0012,  # малиновый залп
    "#ffc300": 0.0012,  # золотой залп
    "#19cdff": 0.0025,  # голубой залп
    "#280f54": 0.38,  # ночное небо
}
FIRE = {
    "per_burst": 320,
    "bursts": 3,
    "cam_z": 12.0,
    "v0": 2.2,  # начальная скорость разлёта, единиц/с
    "grav": 1.1,  # гравитация, единиц/с^2
    "life": 1.9,  # время жизни залпа, с
    "period": 2.0,  # повтор залпа, с
    "size": 6.0,
    "delays": (0.0, 0.35, 0.7),
}


def _fire_spots(rng: random.Random) -> tuple[tuple[float, float], ...]:
    """Три объявленные точки запуска в долях кадра, с разлётом по сетке."""
    xs = sorted(rng.uniform(0.26, 0.74) for _ in range(3))
    ys = sorted(rng.uniform(0.16, 0.36) for _ in range(3))
    return tuple((round(x, 3), round(y, 3)) for x, y in zip(xs, ys))


FIRE_PROMPT = """Собери ночной фейерверк: три залпа из объявленных точек, каждый —
сфера из частиц, которая расширяется, падает под гравитацией и затухает.

Камера `PerspectiveCamera(45, 800/600, 0.1, 200)` в точке (0, 0, {cam_z}),
смотрит в начало координат; полувысота кадра на этой дистанции ≈ {half} единицы,
доля кадра переводится как `x = (fx - 0.5) * 2 * полувысота * 800/600`,
`y = (0.5 - fy) * 2 * полувысота`.

Композиция (проверяется по кадру грубо):
* залпы — в верхней половине кадра, верхние полосы кадра ярче нижних;
* небо тёмное и занимает большую часть кадра: тёплые и холодные частицы видны
  на нём как мелкие искры, а не как заливка.

Залпы: {bursts} штуки, точки запуска в долях кадра (x от левого края, y от
верхнего): {spots}. Каждый залп — {per_burst} частиц, направление по сфере:
`z = 1 - 2*u`, `phi = 2*pi*v`, `rr = sqrt(1 - z*z)`, скорость
`V = {v0} * (0.65 + 0.5*w)` (три числа ЛЦГ на частицу), вектор
`(rr*cos(phi)*V, z*V, rr*sin(phi)*V*0.5)`.

Частица живёт: `tau` — время с запуска, положение
`p = точка + вектор * tau`, `y -= 0.5 * {grav} * tau^2`, яркость
`max(0, 1 - tau / {life})`, размер {size} пикселей. Залп повторяется каждые
{period} с: `tau = (t - delay) mod {period}`, при `tau > {life}` частиц не
видно. Задержки залпов: {delays} с. Точки — `THREE.Points` с
`PointsMaterial({{sizeAttenuation: false, transparent: true, blending:
THREE.AdditiveBlending, depthTest: false}})`, имена `burst_1`, `burst_2`,
`burst_3`; цвет залпа — свой цвет палитры.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_red} — красный залп, не меньше {s_red} кадра;
* {c_gold} — золотой залп, не меньше {s_gold};
* {c_blue} — голубой залп, не меньше {s_blue};
* {c_sky} — ночное небо, не меньше {s_sky}.

Анимация: `window.__step(dt)` растит `tau`, расширяет и гасит залпы, время t
растёт на dt; за {steps} шагов залпы заметно расширяются и меняют яркость.

* `window.__probe()` возвращает `t`, `objects` и `bursts` — по объекту на залп с
  `r` (средний радиус разлёта, 4 знака) и `bright` (текущая яркость).

Проверяется: страница и кадр (5 проверок), палитра по долям, что искры в верхней
части кадра и небо тёмное, детализация (не плоско и не шум) и то, что сцена
живёт: кадры на t и t+{steps} шагов различаются, а каждая загрузка даёт тот же
кадр.

Свойства фикстуры, обязанные попасть в промпт дословно: seed {seed}, точки
запуска {spots}, задержки {delays}, скорость {v0}, гравитация {grav}, жизнь
{life}, период {period}, частиц в залпе {per_burst}.

{lcg}
"""

FIRE_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>fireworks</title>
<style>html,body{{margin:0;background:#05060f;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const PER_BURST = {per_burst}, BURSTS = {bursts};
const SEED = {seed};
const CAM_Z = {cam_z}, FOV = 45, PIX = 600;
const HALF = Math.tan(FOV * Math.PI / 360) * CAM_Z;
const V0 = {v0}, GRAV = {grav}, LIFE = {life}, PERIOD = {period}, SIZE = {size};
const SPOTS = {spots};
const DELAYS = {delays};
const COLORS = ['{c_red}', '{c_gold}', '{c_blue}'];

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, PIX * 1.8, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, 800 / PIX, 0.1, 200);
camera.position.set(0, 0, CAM_Z);
camera.lookAt(0, 0, 0);

// Ночное небо: тёмный градиент во весь кадр.
const skyCanvas = document.createElement('canvas');
skyCanvas.width = skyCanvas.height = 128;
const skyCtx = skyCanvas.getContext('2d');
const skyGrad = skyCtx.createLinearGradient(0, 0, 0, 128);
skyGrad.addColorStop(0, '{c_sky_top}');
skyGrad.addColorStop(1, '{c_sky_bottom}');
skyCtx.fillStyle = skyGrad;
skyCtx.fillRect(0, 0, 128, 128);
const skyTex = new THREE.CanvasTexture(skyCanvas);
skyTex.colorSpace = THREE.SRGBColorSpace;
const sky = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2 * 800 / PIX, HALF * 2),
  new THREE.MeshBasicMaterial({{map: skyTex, depthTest: false}}));
sky.name = 'sky';
sky.position.z = -1;
scene.add(sky);

// Частицы залпа: направления по сфере из ЛЦГ.
const bursts = [];
for (let k = 0; k < BURSTS; k++) {{
  const dirs = [], speeds = [];
  for (let i = 0; i < PER_BURST; i++) {{
    const u = next(), v = next(), w = next();
    const z = 1 - 2 * u, phi = 2 * Math.PI * v, rr = Math.sqrt(Math.max(0, 1 - z * z));
    const sp = V0 * (0.65 + 0.5 * w);
    dirs.push([rr * Math.cos(phi) * sp, z * sp, rr * Math.sin(phi) * sp * 0.5]);
    speeds.push(sp);
  }}
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(PER_BURST * 3), 3));
  const mat = new THREE.PointsMaterial({{size: SIZE, sizeAttenuation: false, transparent: true,
    color: new THREE.Color(COLORS[k]), blending: THREE.AdditiveBlending, depthTest: false}});
  const pts = new THREE.Points(geo, mat);
  pts.name = 'burst_' + (k + 1);
  scene.add(pts);
  bursts.push({{pts: pts, dirs: dirs, speeds: speeds, delay: DELAYS[k], tau: 0,
    fx: SPOTS[k][0], fy: SPOTS[k][1]}});
}}

const place = (t) => {{
  for (const b of bursts) {{
    const tau = ((t - b.delay) % PERIOD + PERIOD) % PERIOD;
    b.tau = tau;
    const fade = Math.max(0, 1 - tau / LIFE);
    b.pts.material.opacity = fade;
    b.pts.visible = fade > 0;
    const x0 = (b.fx - 0.5) * 2 * HALF * 800 / PIX, y0 = (0.5 - b.fy) * 2 * HALF;
    const attr = b.pts.geometry.attributes.position;
    for (let i = 0; i < PER_BURST; i++) {{
      const d = b.dirs[i];
      attr.setXYZ(i, x0 + d[0] * tau, y0 + d[1] * tau - 0.5 * GRAV * tau * tau, d[2] * tau);
    }}
    attr.needsUpdate = true;
  }}
}};

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;
window.__step = (dt) => {{
  t += dt;
  place(t);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  bursts: bursts.map((b) => ({{r: round(b.speeds.reduce((a, v) => a + v, 0) / PER_BURST * b.tau),
    bright: round(Math.max(0, 1 - b.tau / LIFE))}})),
}});

place(0);
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _fire_upper():
    """Искры — в верхней части кадра: верхние полосы ярче нижних."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = [entry["lum"] for entry in (pixels.get("profile") or [])]
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        top = sum(profile[:4]) / 4
        bottom = sum(profile[4:]) / 4
        good = top > bottom + 0.6
        return good, f"кадр: залпы выше горизонта — верх {top:.1f} против низа {bottom:.1f}"

    return visual._label(check, "кадр: искры в верхней половине кадра")


def _fire_sparks_in_upper_half(safe: float):
    """Тёплые искры (залпы) сидят в верхней половине кадра."""

    def check(pixels: dict) -> tuple[bool, str]:
        warm = pixels.get("warm") or {}
        centroid = warm.get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 2:
            return False, "кадр: тёплых искр не видно"
        fraction = float(warm.get("fraction") or 0.0)
        good = fraction >= 0.0015 and centroid[1] <= safe
        return good, f"кадр: тёплые искры {fraction:.4f} кадра, центр y={centroid[1]:.3f} (нужно <= {safe})"

    return visual._label(check, "кадр: искры залпов в верхней части и их видно")


def _fire_sky_dark():
    """Небо тёмное и занимает кадр: почти чёрных пикселей больше половины."""

    def check(pixels: dict) -> tuple[bool, str]:
        dark = (pixels.get("dark") or {}).get("fraction")
        if not isinstance(dark, (int, float)):
            return False, "кадр: нет класса dark"
        good = float(dark) >= 0.5
        return good, f"кадр: тёмного неба {float(dark):.4f} (нужно >= 0.5)"

    return visual._label(check, "кадр: ночное небо занимает кадр (тёмного >= 0.5)")


# --------------------------------------------------------------------------- #
# L-8 — аврора
# --------------------------------------------------------------------------- #

AURORA_PALETTE = {
    "#18f051": 0.013,  # внизу лент — зелёный
    "#2197db": 0.12,  # середина лент — бирюза
    "#ff33ee": 0.045,  # верх ленты — фиолетовый
    "#1d1638": 0.40,  # ночное небо
}
AURORA = {
    "strips": 28,
    "cam_z": 10.0,
    "w_range": (0.030, 0.085),  # ширина полосы в долях ширины кадра
    "h_range": (0.30, 0.68),  # высота полосы в долях высоты кадра
    "amp": 0.05,  # амплитуда бега полосы, доля ширины кадра
    "speed": 0.35,  # скорость волны, циклов/с
    "wave_k": 3.0,  # набег фазы по ширине кадра
    "opacity": (0.16, 0.40),
    "y": (0.02, 0.26),  # центр полосы вверх от центра кадра, доля высоты
}


AURORA_PROMPT = """Собери аврору: {strips} вертикальных полос-лент на ночном небе,
снизу зелёные, к верху через бирюзу в фиолетовый, с волной слева направо.

Камера `PerspectiveCamera(45, 800/600, 0.1, 200)` в точке (0, 0, {cam_z}),
смотрит в начало координат; полувысота кадра на этой дистанции ≈ {half} единицы,
ширина кадра — 800/600 от неё.

Композиция (проверяется по кадру грубо):
* лента стоит над нижней третью кадра: полосы висят в верхней части неба,
  верхние полосы профиля ярче нижних, горизонт чист;
* полосы вертикальные и разной ширины — кадр не заливка, а лента из языков.

Полосы: {strips} штук. Для каждой полосы возьми из ЛЦГ пять чисел подряд:
ширину `w = {w_lo} + {w_span} * u` (доля ширины кадра), высоту
`h = {h_lo} + {h_span} * v` (доля высоты кадра), центр
`x = (i + 0.5 + 0.6*(u2-0.5)) / {strips} - 0.5` (доля ширины кадра),
`y = {y_lo} + {y_span} * u3` (доля высоты кадра, вверх от центра),
прозрачность `op = {op_lo} + {op_span} * u4`, фазу `ph = 0.4 * u5`.
Каждая полоса — `THREE.Mesh` с `PlaneGeometry(1, 1)` в масштабе `(w, h)`, имя
`ribbon_1`…`ribbon_{strips}`, материал `MeshBasicMaterial({{map: канва-текстура,
transparent: true, blending: THREE.AdditiveBlending, depthTest: false}})` с
`opacity = op`. Общая канва-текстура 64x256 держит и цвет по высоте полосы — низ
{c_green}, середина {c_teal}, верх {c_violet}, — и мягкое боковое затухание
(горизонтальный градиент с непрозрачной серединой), поэтому полоса читается как
занавес, а не как прямоугольник.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_green} — низ лент, не меньше {s_green} кадра;
* {c_teal} — середина лент, не меньше {s_teal};
* {c_violet} — верх лент, не меньше {s_violet};
* {c_sky} — ночное небо, не меньше {s_sky}.

Анимация — волна слева направо: у полосы с центром `x` смещение
`dx = {amp} * sin(2*pi*{speed}*t - {wave_k}*x + ph)`, высота
`h * (1 + 0.15 * sin(2*pi*{speed}*0.7*t + ph))`; время t растёт на dt.
За {steps} шагов фаза волны проходит заметно, языки обязаны переехать.

* `window.__step(dt)` двигает время и волну;
* `window.__probe()` возвращает `t`, `objects` и `wave` — фазу волны
  `2*pi*{speed}*t` (4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, что лента висит в
верхней части и не заливает низ, детализация (не плоско и не шум) и то, что
сцена живёт: кадры на t и t+{steps} шагов различаются, а каждая загрузка даёт
тот же кадр.

Свойства фикстуры, обязанные попасть в промпт дословно: seed {seed}, полос
{strips}, ширина {w_lo}..{w_span}, высота {h_lo}..{h_span}, амплитуда {amp},
скорость волны {speed}.

{lcg}
"""

AURORA_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>aurora</title>
<style>html,body{{margin:0;background:#05060f;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const STRIPS = {strips};
const SEED = {seed};
const CAM_Z = {cam_z}, FOV = 45, PIX = 600;
const HALF = Math.tan(FOV * Math.PI / 360) * CAM_Z;
const FRAME_W = HALF * 2 * 800 / PIX, FRAME_H = HALF * 2;
const W_LO = {w_lo}, W_SPAN = {w_span}, H_LO = {h_lo}, H_SPAN = {h_span};
const AMP = {amp}, SPEED = {speed}, WAVE_K = {wave_k};
const OP_LO = {op_lo}, OP_SPAN = {op_span}, Y_LO = {y_lo}, Y_SPAN = {y_span};
const C_GREEN = new THREE.Color('{c_green}'), C_TEAL = new THREE.Color('{c_teal}');
const C_VIOLET = new THREE.Color('{c_violet}');

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, PIX * 1.8, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, 800 / PIX, 0.1, 200);
camera.position.set(0, 0, CAM_Z);
camera.lookAt(0, 0, 0);

const skyCanvas = document.createElement('canvas');
skyCanvas.width = skyCanvas.height = 128;
const skyCtx = skyCanvas.getContext('2d');
const skyGrad = skyCtx.createLinearGradient(0, 0, 0, 128);
skyGrad.addColorStop(0, '{c_sky_top}');
skyGrad.addColorStop(0.75, '{c_sky}');
skyGrad.addColorStop(1, '{c_sky_bottom}');
skyCtx.fillStyle = skyGrad;
skyCtx.fillRect(0, 0, 128, 128);
const skyTex = new THREE.CanvasTexture(skyCanvas);
skyTex.colorSpace = THREE.SRGBColorSpace;
const SKY_DIST = CAM_Z + 1, SKY_HALF = Math.tan(FOV * Math.PI / 360) * SKY_DIST;
const sky = new THREE.Mesh(new THREE.PlaneGeometry(SKY_HALF * 2 * 800 / PIX, SKY_HALF * 2),
  new THREE.MeshBasicMaterial({{map: skyTex, depthTest: false}}));
sky.name = 'sky';
sky.position.z = -1;
scene.add(sky);

// Полоса: вертикальная лента. Общая канва-текстура держит и градиент цвета
// (низ зелёный → бирюза → фиолетовый верх), и мягкое затухание к краям,
// поэтому полоса читается как занавес, а не как прямоугольник.
const ribbonCanvas = document.createElement('canvas');
ribbonCanvas.width = 64;
ribbonCanvas.height = 256;
const ribbonCtx = ribbonCanvas.getContext('2d');
const mixColour = (a, b, k) => a.clone().lerp(b, Math.min(1, Math.max(0, k)));
for (let y = 0; y < 256; y++) {{
  const v = 1 - y / 255;  // 0 — низ полосы, 1 — верх
  // непрерывный тон: зелёный низ → бирюза → фиолетовый верх; степень держит
  // зелёную кромку шире остальных
  // переход зелёный → бирюза → фиолетовый снизу вверх
  const k = Math.min(1, Math.pow(v, 0.9) * 1.14);
  const c = k < 0.55 ? mixColour(C_GREEN, C_TEAL, k / 0.55) : mixColour(C_TEAL, C_VIOLET, (k - 0.55) / 0.45);
  // альфа: плотный низ, к верху лента редеет, но не гаснет — иначе фиолетовый
  // разбавляется небом и тона не набирает
  const alpha = 0.55 + 0.45 * Math.pow(1 - v, 0.6);
  ribbonCtx.fillStyle = `rgba(${{(c.r * 255) | 0}},${{(c.g * 255) | 0}},${{(c.b * 255) | 0}},${{alpha.toFixed(3)}})`;
  ribbonCtx.fillRect(0, y, 64, 1);
}}
// боковое затухание: горизонтальный градиент прозрачности поверх строк
ribbonCtx.globalCompositeOperation = 'destination-in';
const sideGrad = ribbonCtx.createLinearGradient(0, 0, 64, 0);
sideGrad.addColorStop(0.0, 'rgba(0,0,0,0)');
sideGrad.addColorStop(0.22, 'rgba(0,0,0,1)');
sideGrad.addColorStop(0.78, 'rgba(0,0,0,1)');
sideGrad.addColorStop(1.0, 'rgba(0,0,0,0)');
ribbonCtx.fillStyle = sideGrad;
ribbonCtx.fillRect(0, 0, 64, 256);
ribbonCtx.globalCompositeOperation = 'source-over';
const ribbonTex = new THREE.CanvasTexture(ribbonCanvas);
ribbonTex.colorSpace = THREE.SRGBColorSpace;

const ribbon = (w, h, op) => {{
  const mat = new THREE.MeshBasicMaterial({{map: ribbonTex, transparent: true, opacity: op,
    blending: THREE.AdditiveBlending, depthTest: false}});
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
  mesh.scale.set(w, h, 1);
  return mesh;
}};

const strips = [];
for (let i = 0; i < STRIPS; i++) {{
  const w = W_LO + W_SPAN * next();
  const h = H_LO + H_SPAN * next();
  const x = (i + 0.5 + 0.6 * (next() - 0.5)) / STRIPS - 0.5;
  const y = Y_LO + Y_SPAN * next();
  const op = OP_LO + OP_SPAN * next();
  const ph = 0.4 * next();
  const mesh = ribbon(w * FRAME_W, h * FRAME_H, op);
  mesh.name = 'ribbon_' + (i + 1);
  mesh.userData = {{x: x * FRAME_W, y: y * FRAME_H, h: h * FRAME_H, w: w * FRAME_W, ph: ph}};
  scene.add(mesh);
  strips.push(mesh);
}}

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;
const wave = () => {{
  for (const mesh of strips) {{
    const u = mesh.userData;
    mesh.position.x = u.x + AMP * FRAME_W * Math.sin(2 * Math.PI * SPEED * t - WAVE_K * (u.x / FRAME_W) + u.ph);
    mesh.position.y = u.y;
    mesh.scale.y = u.h * (1 + 0.15 * Math.sin(2 * Math.PI * SPEED * 0.7 * t + u.ph));
  }}
}};

window.__step = (dt) => {{
  t += dt;
  wave();
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  wave: round(2 * Math.PI * SPEED * t),
}});

wave();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _aurora_top():
    """Лента висит над нижней третью: верхние полосы профиля ярче нижних."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = [entry["lum"] for entry in (pixels.get("profile") or [])]
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        top = sum(profile[:4]) / 4
        bottom = sum(profile[5:]) / 3
        good = top > bottom + 0.5
        return good, f"кадр: лента выше горизонта — верх {top:.1f} против низа {bottom:.1f}"

    return visual._label(check, "кадр: аврора висит в верхней части кадра")


def _aurora_vertical():
    """Полосы вертикальные: границы вдоль строк сильнее, чем вдоль столбцов."""

    def check(pixels: dict) -> tuple[bool, str]:
        edges = pixels.get("edges")
        regions = pixels.get("regions") or []
        if not isinstance(edges, (int, float)) or len(regions) < 9:
            return False, "кадр: нет метрик"
        lum = lambda rgb: 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]  # noqa: E731
        top = [lum(regions[i]) for i in (0, 1, 2)]
        spread = max(top) - min(top)
        good = float(edges) >= 0.0025 and spread >= 1.0
        return good, f"кадр: языки ленты читаются — границы {edges}, разброс по верхней строке {spread:.1f}"

    return visual._label(check, "кадр: полосы авроры различимы по горизонтали")


# --------------------------------------------------------------------------- #
# L-9 — звёздное небо над горами
# --------------------------------------------------------------------------- #

NIGHT_PALETTE = {
    "#150b38": 0.32,  # ночное небо
    "#00b1d9": 0.10,  # силуэт гор — сине-бирюзовый контур
    "#f2f6ff": 0.0008,  # звёзды
    "#ffd699": 0.005,  # луна
}
NIGHT = {
    "stars": 620,
    "ridge_points": 9,
    "cam_z": 10.0,
    "moon_r": 0.075,  # радиус луны в долях высоты кадра
    "moon_x": (0.20, 0.80),
    "moon_y": (0.16, 0.34),
    "drift": 0.02,  # ход луны, доля ширины кадра в секунду
    "twinkle": 1.7,  # скорость мерцания, рад/с
    "star_size": (2.0, 4.6),  # размер звезды, пиксели
    "ridge_range": (0.14, 0.46),  # высота горы над нижним краем, доля кадра
}


def _night_ridge(rng: random.Random) -> tuple[float, ...]:
    """Высоты ломаной гор (доли высоты кадра над нижним краем), от rng."""
    return tuple(round(rng.uniform(*NIGHT["ridge_range"]), 3) for _ in range(NIGHT["ridge_points"]))


NIGHT_PROMPT = """Собери звёздное небо над горами: чёрно-синее небо, силуэт гор по
ломаной, instanced-звёзды с мерцанием по фазе и тёплая луна.

Камера `PerspectiveCamera(45, 800/600, 0.1, 200)` в точке (0, 0, {cam_z}),
смотрит в начало координат; полувысота кадра на этой дистанции ≈ {half} единицы,
доля кадра переводится как `x = (fx - 0.5) * 2 * полувысота * 800/600`,
`y = (0.5 - fy) * 2 * полувысота`.

Композиция (проверяется по кадру грубо):
* горы — силуэт внизу кадра: нижние полосы профиля ярче верхних, небо над
  горами темнее гор;
* звёзды — в верхней части кадра, россыпью, небо не пустое;
* луна — тёплое пятно в верхней половине кадра, над горами.

Горы: ломаная по {ridge_points} точкам, высоты над нижним краем кадра в долях
высоты кадра: {ridge}; `THREE.Shape` из ломаной и нижних углов кадра, заливка
{c_ridge}, имя `ridge`.

Небо: плоскость во весь кадр, вертикальный градиент от {c_sky_top} у верхнего
края к {c_sky} у горизонта, имя `sky`.

Звёзды: `THREE.InstancedMesh` из {stars} четырёхугольников, имя `stars`, размер
звезды `{size_lo} + {size_span} * u` пикселя; положение — доля кадра
`fx = u2`, `fy = u3 * 0.72` (верхние две трети кадра); фаза `ph = 6.283 * u4`;
цвет {c_star}, аддитивное смешивание, `depthTest: false`.

Луна: диск радиуса {moon_r} высоты кадра, материал `MeshBasicMaterial` цвета
{c_moon}, имя `moon`; плюс аддитивное пятно вокруг. Начало в {moon}.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_sky} — небо, не меньше {s_sky} кадра;
* {c_ridge} — силуэт гор, не меньше {s_ridge};
* {c_star} — звёзды, не меньше {s_star};
* {c_moon} — луна, не меньше {s_moon}.

Анимация: звёзды мерцают по фазе — масштаб
`0.5 + 0.5 * sin(ph + {twinkle}*t)`, то есть от 0.5 до 1.0 базового размера;
луна едет вправо со скоростью {drift} ширины кадра в секунду; время t растёт на
dt. За {steps} шагов и мерцание, и ход луны обязаны быть видны.

* `window.__step(dt)` двигает время, мерцание и луну;
* `window.__probe()` возвращает `t`, `objects` и `moon` — долю x луны (4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, что горы внизу и
небо темнее гор, детализация (не плоско и не шум) и то, что сцена живёт: кадры
на t и t+{steps} шагов различаются, а каждая загрузка даёт тот же кадр.

Свойства фикстуры, обязанные попасть в промпт дословно: seed {seed}, горы
{ridge}, луна {moon}, мерцание {twinkle}, ход луны {drift}, звёзд {stars}.

{lcg}
"""

NIGHT_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>night</title>
<style>html,body{{margin:0;background:#05060f;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const STARS = {stars}, RIDGE_POINTS = {ridge_points};
const SEED = {seed};
const CAM_Z = {cam_z}, FOV = 45, PIX = 600;
const HALF = Math.tan(FOV * Math.PI / 360) * CAM_Z;
const FRAME_W = HALF * 2 * 800 / PIX, FRAME_H = HALF * 2;
const RIDGE = {ridge};
const MOON_R = {moon_r}, MOON_X = {moon_x}, MOON_Y = {moon_y}, DRIFT = {drift};
const TWINKLE = {twinkle}, SIZE_LO = {size_lo}, SIZE_SPAN = {size_span};

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, PIX * 1.8, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, 800 / PIX, 0.1, 200);
camera.position.set(0, 0, CAM_Z);
camera.lookAt(0, 0, 0);

// Небо: вертикальный градиент от чёрно-синего к чуть более светлому у горизонта.
const skyCanvas = document.createElement('canvas');
skyCanvas.width = skyCanvas.height = 128;
const skyCtx = skyCanvas.getContext('2d');
const skyGrad = skyCtx.createLinearGradient(0, 0, 0, 128);
skyGrad.addColorStop(0, '{c_sky_top}');
skyGrad.addColorStop(0.62, '{c_sky}');
skyGrad.addColorStop(1, '{c_sky_low}');
skyCtx.fillStyle = skyGrad;
skyCtx.fillRect(0, 0, 128, 128);
const skyTex = new THREE.CanvasTexture(skyCanvas);
skyTex.colorSpace = THREE.SRGBColorSpace;
const SKY_DIST = CAM_Z + 1, SKY_HALF = Math.tan(FOV * Math.PI / 360) * SKY_DIST;
const sky = new THREE.Mesh(new THREE.PlaneGeometry(SKY_HALF * 2 * 800 / PIX, SKY_HALF * 2),
  new THREE.MeshBasicMaterial({{map: skyTex, depthTest: false}}));
sky.name = 'sky';
sky.position.z = -1;
scene.add(sky);

// Горы: ломаная из объявленных высот, замкнутая нижним краем кадра.
const shape = new THREE.Shape();
shape.moveTo(-FRAME_W / 2, -FRAME_H / 2);
for (let i = 0; i < RIDGE_POINTS; i++) {{
  const x = -FRAME_W / 2 + FRAME_W * (i / (RIDGE_POINTS - 1));
  shape.lineTo(x, -FRAME_H / 2 + RIDGE[i] * FRAME_H);
}}
shape.lineTo(FRAME_W / 2, -FRAME_H / 2);
const ridge = new THREE.Mesh(new THREE.ShapeGeometry(shape),
  new THREE.MeshBasicMaterial({{color: '{c_ridge}', depthTest: false}}));
ridge.name = 'ridge';
ridge.position.z = -0.5;
scene.add(ridge);

// Звёзды: instanced-четырёхугольники, мерцают масштабом по своей фазе.
const starGeo = new THREE.PlaneGeometry(1, 1);
const starMat = new THREE.MeshBasicMaterial({{color: '{c_star}', transparent: true,
  blending: THREE.AdditiveBlending, depthTest: false}});
const starMesh = new THREE.InstancedMesh(starGeo, starMat, STARS);
starMesh.name = 'stars';
starMesh.frustumCulled = false;
scene.add(starMesh);
const stars = [];
const dummy = new THREE.Object3D();
for (let i = 0; i < STARS; i++) {{
  const size = (SIZE_LO + SIZE_SPAN * next()) / PIX * FRAME_H;
  const fx = next(), fy = next() * 0.72, ph = 6.283 * next();
  stars.push({{x: (fx - 0.5) * FRAME_W, y: (0.5 - fy) * FRAME_H, size: size, ph: ph}});
}}

const glowCanvas = document.createElement('canvas');
glowCanvas.width = glowCanvas.height = 128;
const glowCtx = glowCanvas.getContext('2d');
const glowGrad = glowCtx.createRadialGradient(64, 64, 0, 64, 64, 64);
glowGrad.addColorStop(0, 'rgba(255,217,160,0.75)');
glowGrad.addColorStop(0.5, 'rgba(255,217,160,0.18)');
glowGrad.addColorStop(1, 'rgba(255,217,160,0)');
glowCtx.fillStyle = glowGrad;
glowCtx.fillRect(0, 0, 128, 128);
const glowTex = new THREE.CanvasTexture(glowCanvas);
glowTex.colorSpace = THREE.SRGBColorSpace;
const moon = new THREE.Mesh(new THREE.CircleGeometry(MOON_R * FRAME_H, 40),
  new THREE.MeshBasicMaterial({{color: '{c_moon}', depthTest: false}}));
moon.name = 'moon';
const moonGlow = new THREE.Mesh(new THREE.PlaneGeometry(MOON_R * FRAME_H * 7, MOON_R * FRAME_H * 7),
  new THREE.MeshBasicMaterial({{map: glowTex, transparent: true, depthTest: false, blending: THREE.AdditiveBlending}}));
moonGlow.name = 'moon_glow';
scene.add(moonGlow);
scene.add(moon);

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;
const moonX = () => (MOON_X - 0.5) * FRAME_W + DRIFT * FRAME_W * t;
const palette = () => {{
  for (let i = 0; i < STARS; i++) {{
    const st = stars[i];
    const flick = 0.5 + 0.5 * Math.sin(st.ph + TWINKLE * t);
    dummy.position.set(st.x, st.y, 0);
    dummy.scale.setScalar(st.size * flick);
    dummy.updateMatrix();
    starMesh.setMatrixAt(i, dummy.matrix);
  }}
  starMesh.instanceMatrix.needsUpdate = true;
  moon.position.set(moonX(), (0.5 - MOON_Y) * FRAME_H, 0.1);
  moonGlow.position.set(moonX(), (0.5 - MOON_Y) * FRAME_H, 0.05);
}};

window.__step = (dt) => {{
  t += dt;
  palette();
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  moon: round((moonX() / FRAME_W) + 0.5),
}});

palette();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _night_ridge_lower():
    """Горы внизу кадра светлее неба над ними."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = [entry["lum"] for entry in (pixels.get("profile") or [])]
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        top = sum(profile[:3]) / 3
        bottom = max(profile[6], profile[7])
        good = bottom > top + 4.0
        return good, f"кадр: силуэт гор ниже неба — низ {bottom:.1f} против верха {top:.1f}"

    return visual._label(check, "кадр: горы внизу кадра светлее неба")


def _night_moon_upper():
    """Луна — тёплое пятно в верхней половине кадра."""

    def check(pixels: dict) -> tuple[bool, str]:
        warm = pixels.get("warm") or {}
        centroid = warm.get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 2:
            return False, "кадр: тёплого пятна не видно"
        fraction = float(warm.get("fraction") or 0.0)
        good = fraction >= 0.0012 and centroid[1] <= 0.45
        return good, f"кадр: луна — тёплых {fraction:.4f} кадра, центр y={centroid[1]:.3f} (нужно <= 0.45)"

    return visual._label(check, "кадр: луна тёплая и в верхней половине")


def _night_sky_dark():
    """Небо тёмное: почти чёрных пикселей больше половины кадра."""

    def check(pixels: dict) -> tuple[bool, str]:
        dark = (pixels.get("dark") or {}).get("fraction")
        if not isinstance(dark, (int, float)):
            return False, "кадр: нет класса dark"
        good = float(dark) >= 0.35
        return good, f"кадр: тёмного неба {float(dark):.4f} (нужно >= 0.35)"

    return visual._label(check, "кадр: ночное небо тёмное (тёмного >= 0.35)")


# --------------------------------------------------------------------------- #
# Сборка заданий
# --------------------------------------------------------------------------- #


def _share_text(share: float) -> str:
    """A declared share written the way the prompt states it."""
    if share >= 0.10:
        return f"{share:.0%}"
    if share >= 0.01:
        return f"{share * 100:.1f}%".replace(".0%", "%")
    return f"{share * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _frame_share(hex_value: str, share: float, label: str):
    return visual.palette_colour(hex_value, share, label)


def _family_checks(prefix: str, palette: dict[str, float], notes: dict[str, str]) -> list:
    return [_frame_share(colour, share, f"{prefix}: {notes[colour]}") for colour, share in palette.items()]


def build(rng: random.Random) -> list[core.TaskInstance]:
    return [
        _galaxy(rng),
        _fireworks(rng),
        _aurora(rng),
        _night(rng),
    ]


def _galaxy(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    half = math.tan(math.radians(FOV / 2)) * GALAXY["cam_z"]
    prompt = GALAXY_PROMPT.format(
        seed=seed,
        half=round(half, 3),
        cam_z=f"{GALAXY['cam_z']:.1f}",
        r_frac=f"{GALAXY['r_frac']:.2f}",
        arms_n=GALAXY["arms_n"],
        core_n=GALAXY["core_n"],
        haze_n=GALAXY["haze_n"],
        arms=GALAXY["arms"],
        turns=f"{GALAXY['turns']:.2f}",
        scatter=f"{GALAXY['scatter']:.2f}",
        w0=f"{GALAXY['w0']:.2f}",
        a0=f"{GALAXY['a0']:.2f}",
        q_min=f"{GALAXY['q_min']:.2f}",
        q_span=f"{1 - GALAXY['q_min']:.2f}",
        core_until=f"{GALAXY['core_until']:.2f}",
        arm_until=f"{GALAXY['arm_until']:.2f}",
        core_r=f"{GALAXY['core_r']:.2f}",
        empty_max=_share_text(GALAXY["empty_max"]),
        steps=STEPS["L-6"],
        c_core="#ffbf00",
        c_arm="#ff0080",
        c_out="#0cb7f0",
        c_sky="#1d1638",
        s_core=_share_text(GALAXY_PALETTE["#ffbf00"]),
        s_arm=_share_text(GALAXY_PALETTE["#ff0080"]),
        s_out=_share_text(GALAXY_PALETTE["#0cb7f0"]),
        s_sky=_share_text(GALAXY_PALETTE["#1d1638"]),
        lcg=LCG_TEXT.format(seed=seed),
    )
    golden = GALAXY_GOLDEN.format(
        seed=seed,
        arms_n=GALAXY["arms_n"],
        core_n=GALAXY["core_n"],
        haze_n=GALAXY["haze_n"],
        cam_z=f"{GALAXY['cam_z']:.1f}",
        r_frac=f"{GALAXY['r_frac']:.2f}",
        arms=GALAXY["arms"],
        turns=f"{GALAXY['turns']:.2f}",
        scatter=f"{GALAXY['scatter']:.2f}",
        w0=f"{GALAXY['w0']:.2f}",
        a0=f"{GALAXY['a0']:.2f}",
        q_min=f"{GALAXY['q_min']:.2f}",
        core_until=f"{GALAXY['core_until']:.2f}",
        arm_until=f"{GALAXY['arm_until']:.2f}",
        core_r=f"{GALAXY['core_r']:.2f}",
        c_core="#ffbf00",
        c_arm="#ff0080",
        c_out="#0cb7f0",
        c_sky="#1d1638",
        c_sky_in="#2f2080",
        c_sky_out="#181046",
    )
    for literal in (str(seed), "7000", "2500", "1500", "0.44", "0.55", "#ffbf00", "#ff0080", "#0cb7f0", "#1d1638"):
        assert literal in prompt, literal
    # Диск обязан повернуться за 72 шага: угол в ядре >> шага измерения.
    assert GALAXY["w0"] * STEPS["L-6"] / 60 / (GALAXY["a0"] + 0.5) > 0.2
    return art.build_task(
        task_id="L-6",
        title=TITLES["L-6"],
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_family_checks(
                "галактика",
                GALAXY_PALETTE,
                {
                    "#ffbf00": "ядро и внутренние рукава",
                    "#ff0080": "средние рукава",
                    "#0cb7f0": "периферия и гало",
                    "#1d1638": "ночное небо вокруг диска",
                },
            ),
            composition=[_galaxy_falloff(), _galaxy_filled()],
            detail=[
                visual.image_min("changed_fraction", 0.10),
                visual.image_min("distinct_colors", 10),
                visual.rich_edges(0.01, "частицы диска дают зерно"),
                visual.image_min("saturation", 0.25),
            ],
            motion=STEPS["L-6"],
        ),
        golden=golden,
        fixture={"seed": seed, "arms_n": GALAXY["arms_n"], "r_frac": GALAXY["r_frac"], "w0": GALAXY["w0"]},
    )


def _fireworks(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    spots = _fire_spots(rng)
    half = math.tan(math.radians(FOV / 2)) * FIRE["cam_z"]
    shown = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in spots)
    delays = ", ".join(f"{d:.1f}" for d in FIRE["delays"])
    prompt = FIRE_PROMPT.format(
        seed=seed,
        half=round(half, 3),
        cam_z=f"{FIRE['cam_z']:.1f}",
        bursts=FIRE["bursts"],
        per_burst=FIRE["per_burst"],
        spots=shown,
        delays=delays,
        v0=f"{FIRE['v0']:.2f}",
        grav=f"{FIRE['grav']:.2f}",
        life=f"{FIRE['life']:.2f}",
        period=f"{FIRE['period']:.2f}",
        size=f"{FIRE['size']:.1f}",
        steps=STEPS["L-7"],
        c_red="#ff0051",
        c_gold="#ffc300",
        c_blue="#19cdff",
        c_sky="#280f54",
        s_red=_share_text(FIRE_PALETTE["#ff0051"]),
        s_gold=_share_text(FIRE_PALETTE["#ffc300"]),
        s_blue=_share_text(FIRE_PALETTE["#19cdff"]),
        s_sky=_share_text(FIRE_PALETTE["#280f54"]),
        lcg=LCG_TEXT.format(seed=seed),
    )
    golden = FIRE_GOLDEN.format(
        seed=seed,
        per_burst=FIRE["per_burst"],
        bursts=FIRE["bursts"],
        cam_z=f"{FIRE['cam_z']:.1f}",
        v0=f"{FIRE['v0']:.2f}",
        grav=f"{FIRE['grav']:.2f}",
        life=f"{FIRE['life']:.2f}",
        period=f"{FIRE['period']:.2f}",
        size=f"{FIRE['size']:.1f}",
        spots="[" + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in spots) + "]",
        delays="[" + ", ".join(f"{d:.1f}" for d in FIRE["delays"]) + "]",
        c_red="#ff0051",
        c_gold="#ffc300",
        c_blue="#19cdff",
        c_sky_top="#241049",
        c_sky_bottom="#0a0524",
    )
    for literal in (str(seed), shown, delays, f"{FIRE['v0']:.2f}", f"{FIRE['grav']:.2f}", f"{FIRE['life']:.2f}", f"{FIRE['period']:.2f}", str(FIRE["per_burst"])):
        assert literal in prompt, literal
    for x, y in spots:
        assert 0.15 < x < 0.85 and 0.15 < y < 0.55
    return art.build_task(
        task_id="L-7",
        title=TITLES["L-7"],
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_family_checks(
                "фейерверк",
                FIRE_PALETTE,
                {
                    "#ff0051": "красный залп",
                    "#ffc300": "золотой залп",
                    "#19cdff": "голубой залп",
                    "#280f54": "ночное небо",
                },
            ),
            composition=[_fire_upper(), _fire_sparks_in_upper_half(0.60), _fire_sky_dark()],
            detail=[
                visual.image_min("changed_fraction", 0.10),
                visual.image_min("distinct_colors", 8),
                visual.image_min("saturation", 0.20),
            ],
            motion=STEPS["L-7"],
        ),
        golden=golden,
        fixture={"seed": seed, "spots": list(spots), "delays": list(FIRE["delays"])},
    )


def _aurora(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    half = math.tan(math.radians(FOV / 2)) * AURORA["cam_z"]
    w_lo, w_span = AURORA["w_range"]
    w_span = round(w_span - w_lo, 3)
    h_lo, h_span = AURORA["h_range"]
    h_span = round(h_span - h_lo, 3)
    op_lo, op_span = AURORA["opacity"]
    op_span = round(op_span - op_lo, 3)
    prompt = AURORA_PROMPT.format(
        seed=seed,
        half=round(half, 3),
        cam_z=f"{AURORA['cam_z']:.1f}",
        strips=AURORA["strips"],
        w_lo=f"{w_lo:.3f}",
        w_span=f"{w_span:.3f}",
        h_lo=f"{h_lo:.2f}",
        h_span=f"{h_span:.2f}",
        amp=f"{AURORA['amp']:.2f}",
        speed=f"{AURORA['speed']:.2f}",
        wave_k=f"{AURORA['wave_k']:.1f}",
        op_lo=f"{op_lo:.2f}",
        op_span=f"{op_span:.2f}",
        y_lo=f"{AURORA['y'][0]:.2f}",
        y_span=f"{AURORA['y'][1]:.2f}",
        steps=STEPS["L-8"],
        c_green="#18f051",
        c_teal="#2197db",
        c_violet="#ff33ee",
        c_sky="#1d1638",
        s_green=_share_text(AURORA_PALETTE["#18f051"]),
        s_teal=_share_text(AURORA_PALETTE["#2197db"]),
        s_violet=_share_text(AURORA_PALETTE["#ff33ee"]),
        s_sky=_share_text(AURORA_PALETTE["#1d1638"]),
        lcg=LCG_TEXT.format(seed=seed),
    )
    golden = AURORA_GOLDEN.format(
        seed=seed,
        strips=AURORA["strips"],
        cam_z=f"{AURORA['cam_z']:.1f}",
        w_lo=f"{w_lo:.3f}",
        w_span=f"{w_span:.3f}",
        h_lo=f"{h_lo:.2f}",
        h_span=f"{h_span:.2f}",
        amp=f"{AURORA['amp']:.2f}",
        speed=f"{AURORA['speed']:.2f}",
        wave_k=f"{AURORA['wave_k']:.1f}",
        op_lo=f"{op_lo:.2f}",
        op_span=f"{op_span:.2f}",
        y_lo=f"{AURORA['y'][0]:.2f}",
        y_span=f"{AURORA['y'][1]:.2f}",
        c_green="#18f051",
        c_teal="#2197db",
        c_violet="#ff33ee",
        c_sky="#1d1638",
        c_sky_top="#1a1148",
        c_sky_bottom="#060318",
    )
    for literal in (str(seed), str(AURORA["strips"]), f"{w_lo:.3f}", f"{h_lo:.2f}", f"{AURORA['amp']:.2f}", f"{AURORA['speed']:.2f}", f"{AURORA['wave_k']:.1f}", "#18f051", "#2197db", "#ff33ee"):
        assert literal in prompt, literal
    # Волна за 72 шага проходит больше десятой доли цикла — сдвиг виден.
    assert AURORA["speed"] * STEPS["L-8"] / 60 > 0.05
    assert AURORA["amp"] > 0.0
    return art.build_task(
        task_id="L-8",
        title=TITLES["L-8"],
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_family_checks(
                "аврора",
                AURORA_PALETTE,
                {
                    "#18f051": "зелёный низ лент",
                    "#2197db": "бирюзовая середина лент",
                    "#ff33ee": "фиолетовый верх лент",
                    "#1d1638": "ночное небо",
                },
            ),
            composition=[_aurora_top(), _aurora_vertical()],
            detail=[
                visual.image_min("changed_fraction", 0.30),
                visual.image_min("distinct_colors", 10),
                visual.rich_edges(0.002, "языки ленты дают границы"),
                visual.image_min("saturation", 0.20),
            ],
            motion=STEPS["L-8"],
        ),
        golden=golden,
        fixture={"seed": seed, "strips": AURORA["strips"], "amp": AURORA["amp"], "speed": AURORA["speed"]},
    )


def _night(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    ridge = _night_ridge(rng)
    moon_x = round(rng.uniform(*NIGHT["moon_x"]), 3)
    moon_y = round(rng.uniform(*NIGHT["moon_y"]), 3)
    half = math.tan(math.radians(FOV / 2)) * NIGHT["cam_z"]
    size_lo, size_hi = NIGHT["star_size"]
    size_span = round(size_hi - size_lo, 3)
    prompt = NIGHT_PROMPT.format(
        seed=seed,
        half=round(half, 3),
        cam_z=f"{NIGHT['cam_z']:.1f}",
        stars=NIGHT["stars"],
        ridge_points=NIGHT["ridge_points"],
        ridge="(" + ", ".join(f"{v:.3f}" for v in ridge) + ")",
        moon=f"({moon_x:.3f}, {moon_y:.3f}) доли кадра",
        drift=f"{NIGHT['drift']:.2f}",
        twinkle=f"{NIGHT['twinkle']:.1f}",
        moon_r=f"{NIGHT['moon_r']:.3f}",
        size_lo=f"{size_lo:.1f}",
        size_span=f"{size_span:.1f}",
        steps=STEPS["L-9"],
        c_sky="#150b38",
        c_sky_top="#090722",
        c_sky_low="#150b38",
        c_ridge="#00b1d9",
        c_star="#f2f6ff",
        c_moon="#ffd699",
        s_sky=_share_text(NIGHT_PALETTE["#150b38"]),
        s_ridge=_share_text(NIGHT_PALETTE["#00b1d9"]),
        s_star=_share_text(NIGHT_PALETTE["#f2f6ff"]),
        s_moon=_share_text(NIGHT_PALETTE["#ffd699"]),
        lcg=LCG_TEXT.format(seed=seed),
    )
    golden = NIGHT_GOLDEN.format(
        seed=seed,
        stars=NIGHT["stars"],
        ridge_points=NIGHT["ridge_points"],
        cam_z=f"{NIGHT['cam_z']:.1f}",
        ridge="[" + ", ".join(f"{v:.3f}" for v in ridge) + "]",
        moon_r=f"{NIGHT['moon_r']:.3f}",
        moon_x=f"{moon_x:.3f}",
        moon_y=f"{moon_y:.3f}",
        drift=f"{NIGHT['drift']:.2f}",
        twinkle=f"{NIGHT['twinkle']:.1f}",
        size_lo=f"{size_lo:.1f}",
        size_span=f"{size_span:.1f}",
        c_sky="#150b38",
        c_sky_top="#090722",
        c_sky_low="#0f0b2e",
        c_ridge="#00b1d9",
        c_star="#f2f6ff",
        c_moon="#ffd699",
    )
    for literal in (str(seed), f"{moon_x:.3f}", f"{moon_y:.3f}", f"{NIGHT['drift']:.2f}", f"{NIGHT['twinkle']:.1f}", f"{NIGHT['moon_r']:.3f}"):
        assert literal in prompt, literal
    for value in ridge:
        assert f"{value:.3f}" in prompt, value
    assert NIGHT["twinkle"] * STEPS["L-9"] / 60 > 1.0  # фаза мерцания успевает пройти
    assert NIGHT["drift"] * STEPS["L-9"] / 60 > 0.02  # луна обязана уехать
    return art.build_task(
        task_id="L-9",
        title=TITLES["L-9"],
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_family_checks(
                "звёздное небо",
                NIGHT_PALETTE,
                {
                    "#150b38": "ночное небо",
                    "#00b1d9": "силуэт гор",
                    "#f2f6ff": "звёзды",
                    "#ffd699": "луна",
                },
            ),
            composition=[_night_ridge_lower(), _night_moon_upper(), _night_sky_dark()],
            detail=[
                visual.image_min("changed_fraction", 0.10),
                visual.image_min("distinct_colors", 8),
                visual.rich_edges(0.002, "горы и звёзды дают границы"),
            ],
            motion=STEPS["L-9"],
        ),
        golden=golden,
        fixture={"seed": seed, "ridge": list(ridge), "moon": [moon_x, moon_y], "drift": NIGHT["drift"]},
    )
