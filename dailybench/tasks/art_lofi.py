"""Art scenarios L-17..L-24 — LoFi Rain Sci-Fi.

Восемь атмосферных сцен «дождь за стеклом»: уютная комната, неоновая
лапшичная, трамвай, оранжерея, аналоговая рубка, робот-ремонтник, космическая
заправка, автобусная остановка.

Общий приём слайса — объёмная рамка. Каркас даёт свет (точечный, с затуханием),
материал ``MeshStandardMaterial`` и тела вместо вырезок: стены, полы, прилавки
и приборы собираются коробками, поэтому кадр читается как комната в перспективе,
а не как плоская картинка. «Улица» ставится за стекло, капли — аддитивные точки,
которые едут вниз по ``__step``. Габариты по-прежнему выводятся из доли кадра и
дистанции до плоскости (``_fw``/``_fh``), поэтому сцена детерминирована и
совпадает с числами, напечатанными в промпте.

Палитра каждой сцены — тёмная база, два-три насыщенных акцента (неон, тёплые
лампы, растения) и бледный холодный цвет дождя: он малонасыщенный, поэтому
проверка по тону отличает акценты, а не штрихи на стекле. Все числа фикстуры
печатаются в промпт дословно и перепроверяются ``assert``-ами в :func:`build`.
"""

from __future__ import annotations

import math
import random

from .. import art, core, visual

IDS = (
    "L-17", "L-18", "L-19", "L-20",
    "L-21", "L-22", "L-23", "L-24",
)

TITLES = {
    "L-17": "L-17 — комната с дождём за окном",
    "L-18": "L-18 — неоновая лапшичная под дождём",
    "L-19": "L-19 — трамвай сквозь дождливый город",
    "L-20": "L-20 — оранжерея на станции под метеорным дождём",
    "L-21": "L-21 — аналоговая рубка с дождём за стеклом",
    "L-22": "L-22 — робот-ремонтник под дождём",
    "L-23": "L-23 — космическая заправка под дождём",
    "L-24": "L-24 — автобусная остановка в неоне",
}

#: Шаги до второго кадра: движение обязано быть видно за это время.
STEPS = {
    "L-17": 90, "L-18": 84, "L-19": 72, "L-20": 84,
    "L-21": 84, "L-22": 78, "L-23": 90, "L-24": 84,
}

FOV = 45.0
ASPECT = 800.0 / 600.0
CAM_Z = 4.0  # камера стоит здесь; дистанция до плоскости z — это (CAM_Z - z)

#: LCG, который объявляется в промпте и одинаков во всех эталонах.
LCG_A, LCG_C, LCG_M = 1103515245, 12345, 2147483648


def _half_at(distance: float) -> float:
    """Полувысота кадра на данной дистанции — та же тригонометрия, что в L-1."""
    return math.tan(math.radians(FOV / 2)) * distance


def _fw(share: float, distance: float) -> float:
    """Доля ШИРИНЫ кадра в единицах сцены на данной дистанции."""
    return share * 2 * _half_at(distance) * ASPECT


def _fh(share: float, distance: float) -> float:
    """Доля ВЫСОТЫ кадра в единицах сцены на данной дистанции."""
    return share * 2 * _half_at(distance)


def _share_text(share: float) -> str:
    if share >= 0.10:
        return f"{share:.0%}"
    if share >= 0.01:
        return f"{share * 100:.1f}%".replace(".0%", "%")
    return f"{share * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _frame_share(hex_value: str, share: float, label: str):
    return visual.palette_colour(hex_value, share, label)


def _share_checks(prefix: str, palette: dict[str, float], notes: dict[str, str]) -> list:
    return [_frame_share(colour, share, f"{prefix}: {notes[colour]}") for colour, share in palette.items()]


def _lcg_text(seed: int) -> str:
    """Одна строка про детерминизм: и генератор, и запрет, и побитовое совпадение."""
    return (
        f"Случайность — только явный LCG: `lcg = (lcg * {LCG_A} + {LCG_C}) % {LCG_M}`, "
        f"`rnd() = lcg / {LCG_M}`, начальное `seed = {seed}`; никаких `Math.random()`. "
        "Две загрузки одной страницы обязаны дать побитово одинаковый кадр."
    )


LCG_JS = """const SEED = {seed};
let lcg = SEED;
const rnd = () => {{ lcg = (lcg * 1103515245 + 12345) % 2147483648; return lcg / 2147483648; }};"""

#: Общий для всех восьми эталонов каркас страницы: перевод доли кадра в единицы
#: сцены, размещение плоскости по доле кадра и аддитивные точки дождя.
COMMON_JS = """const H = (d) => Math.tan(45 * Math.PI / 360) * d;      // полувысота кадра на дистанции d
const FW = (u, d) => u * 2 * H(d) * (800 / 600);    // доля ширины кадра -> единицы сцены
const FH = (u, d) => u * 2 * H(d);                  // доля высоты кадра -> единицы сцены
const CAM_Z = 4;

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x241638, 14, 34);
const camera = new THREE.PerspectiveCamera(45, 800 / 600, 0.1, 200);
camera.position.set(0, 0, CAM_Z);
camera.lookAt(0, 0, -1);

const solid = (c) => new THREE.MeshBasicMaterial({{color: c}});
// Градиент из двух близких оттенков: плоскость перестаёт быть заливкой. Цвета
// принимаются и в '#rrggbb', и в '0xrrggbb'.
const hx = (c) => c.slice(0, 2) === '0x' ? '#' + c.slice(2) : c;   // '0x..' и '#..'
const vgrad = (top, bottom) => new THREE.ShaderMaterial({{
  uniforms: {{uTop: {{value: new THREE.Color(hx(top))}}, uBottom: {{value: new THREE.Color(hx(bottom))}}}},
  vertexShader: `varying vec2 vUv; void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  // uniform-цвета приходят в линейном пространстве: mix по линейным и перевод
  // обратно в sRGB — иначе градиент вдвое светлее того же цвета у MeshBasicMaterial.
  fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; varying vec2 vUv;
    void main() {{
      gl_FragColor = vec4(mix(uBottom, uTop, smoothstep(0.0, 1.0, vUv.y)), 1.0);
      #include <colorspace_fragment>
    }}`,
}});
const haze = (c, o) => new THREE.MeshBasicMaterial({{color: c, transparent: true, opacity: o, depthWrite: false}});
// Объёмный вариант тех же деталей: свет и материал вместо эмиссивной заливки.
// Поверхность получает блик и затенение, кадр перестаёт быть «плоским».
const lit = (c, o) => new THREE.MeshStandardMaterial({{color: c, roughness: 0.6, metalness: 0.05, opacity: o ?? 1, transparent: (o ?? 1) < 1}});
// Точечный свет с затуханием: на плоскости он даёт градиент, а не ровную
// заливку, — именно этого не хватало плоским эмиссивным кадрам. Второй, слабый
// и холодный, стоит со стороны окна и подсвечивает края.
// Свет ставится по сцене: тёплый — у киоска (его подставляет L-18), холодный —
// у окна. Значения намеренно слабые: тёмная база должна остаться тёмной, а
// свет только выявляет объём.
const lamp = new THREE.PointLight(0xffd9a8, 5.5, 26, 2);
lamp.position.set(-1.2, 1.0, 1.4);
scene.add(lamp);
scene.add(new THREE.AmbientLight(0x4a5674, 1.1));
const rim = new THREE.PointLight(0x9fc4d8, 3.2, 34, 2);
rim.position.set(0.8, 0.2, -4.0);
scene.add(rim);
const place = (mesh, u, v, z) => {{
  mesh.position.set((u - 0.5) * FW(1, CAM_Z - z), (0.5 - v) * FH(1, CAM_Z - z), z);
  scene.add(mesh);
  return mesh;
}};
const quad = (u, v, material, name, z) => {{
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(FW(u, CAM_Z - z), FH(v, CAM_Z - z)), material);
  mesh.name = name;
  return place(mesh, 0.5, 0.5, z);
}};
const box = (u, v, w, material, name, z) => {{
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(FW(u, CAM_Z - z), FH(v, CAM_Z - z), w), material);
  mesh.name = name;
  return place(mesh, 0.5, 0.5, z);
}};
// Улица: здания телами в глубине — дают перспективу и затенение по граням.
// Объёмное тело: те же габариты кадра, но с толщиной — при наезде камеры видны
// боковые грани, и предмет читается как тело, а не как вырезанный
// прямоугольник. position — доля кадра, как у place.
const solid3 = (u, v, d, material, name, z) => {{
  const depth = d * FW(u, CAM_Z - z);
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(FW(u, CAM_Z - z), FH(v, CAM_Z - z), depth), material);
  mesh.name = name;
  return place(mesh, 0.5, 0.5, z);
}};
const spray = (n, size, colour) => {{
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
  const cloud = new THREE.Points(geometry, new THREE.PointsMaterial({{
    size: size, sizeAttenuation: false, color: colour, transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthTest: false,
  }}));
  scene.add(cloud);
  return cloud;
}};
const round = (v) => Math.round(v * 1e4) / 1e4;
const names = () => scene.children.map((o) => o.name).filter(Boolean);"""


# --------------------------------------------------------------------------- #
# L-17 — комната с дождём за окном
# --------------------------------------------------------------------------- #

L17_PALETTE = {
    "#2e1636": 0.30,  # стены, пол и потолок комнаты
    "#ffa83c": 0.010,  # тёплая лампа и полоса подоконника
    "#7bbf6a": 0.006,  # растение на подоконнике
    "#7fe3e0": 0.010,  # окна неонового города и монитор
    "#a8c8dc": 0.012,  # капли на стекле и дождь за окном
}
L17 = {
    "glass_z": -6.0,  # плоскость окна
    "glass_w": 0.78,  # ширина окна в долях кадра
    "glass_v": 0.11,  # верх окна в долях высоты кадра
    "glass_h": 0.34,  # высота окна в долях кадра
    "sill_v": 0.50,  # полоса подоконника по центру
    "drift": 1.4,  # наезд камеры к окну, единиц/с
    "drops": 56,
    "speed": 1.9,  # скорость капель по стеклу, единиц/с
}
L17_TITLE = TITLES["L-17"]

L17_PROMPT = """Собери ночную сцену «комната с дождём за окном», как рисуют lofi-фоны:
тёплая комната, большое окно во всю стену, по стеклу едут капли, за окном —
неоновый город, внутри стол с монитором и растение. Камера медленно едет к окну.

Композиция (проверяется по кадру грубо):
* окно занимает верхнюю половину кадра: от {glass_v:.0%} до {glass_bottom:.0%} высоты,
  шириной {glass_w:.0%} кадра по центру;
* ниже окна — тёплая полоса подоконника на {sill_v:.0%} высоты, под ней тёмный пол;
* кадр в полосе окна светлее низа кадра минимум на 6 единиц средней яркости;
* холодные пиксели (город и капли) лежат в полосе окна, тёплые (лампа и
  подоконник) — по центру кадра.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_room} — стены, потолок и пол комнаты, не меньше {s_room};
* {c_lamp} — тёплая лампа под потолком и полоса подоконника, не меньше {s_lamp};
* {c_plant} — листва растения на подоконнике, не меньше {s_plant};
* {c_city} — окна города за стеклом и монитор на столе, не меньше {s_city};
* {c_rain} — капли на стекле и струи дождя за окном, не меньше {s_rain}.

Геометрия (координаты — в долях ШИРИНЫ и ВЫСОТЫ кадра, дистанции — по оси Z;
камера стоит в (0, 0, 4) и смотрит в (0, 0, -1)):
Комната собирается телами, а не картинкой: пол и потолок — `BoxGeometry`,
уходящие в глубину от камеры, у окна — откос (рама с толщиной), предметы —
коробки. Поэтому при наезде камеры видно, как меняется перспектива и как
точечный свет ложится градиентом на стены: кадр обязан быть объёмным.
* `wall_top` — тёмная полоса стены над окном: 0..{glass_v:.0%} высоты, цвет {c_room};
* `glass` — стекло в окне: ширина {glass_w:.0%} кадра, от {glass_v:.0%} до
  {glass_bottom:.0%} высоты, z = {glass_z}, полупрозрачное (opacity 0.35);
* `city` — стена дальнего города сразу за стеклом, z = -20: полосы-небоскрёбы
  высотой 0.20..0.45 высоты кадра, тёмные, с окнами цвета {c_city} (мелкие
  аддитивные квадраты 0.012x0.012 кадра, {windows} штук, положение — LCG);
* `rain` — струи дождя за окном: {drops_out} аддитивных точек цвета {c_rain} на
  плоскости z = -12, размер 5 px, полосы падают вниз со скоростью {rain_speed} в
  секунду (единиц сцены) и заворачиваются;
* `plant` — растение на подоконнике слева (u = {plant_u}): три листа-плоскости
  цвета {c_plant} (0.09x0.15, 0.08x0.12 и 0.07x0.10 кадра) и горшок {c_room};
  силуэт высотой {plant_h:.0%} кадра;
* `desk` — стол справа внизу (u = {desk_u}, v = {desk_v}), монитор цвета {c_city}
  {mon_w:.0%} кадра шириной, тёплое пятно {c_lamp} на столе;
* `lamp` — тёплая лампа цвета {c_lamp} под потолком, u = {lamp_u}, v = {lamp_v};
* камера едет к окну: z уменьшается на {drift} единицы в секунду до {cam_min}, x не меняется.

Анимация:
* `window.__step(dt)` двигает капли и струи вниз, двигает камеру к окну и
  возвращает `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects` и `camera: [x, z]` (4 знака).

Проверяется: страница и кадр, палитра по долям, композиция (окно светлее пола,
тёплое у подоконника, холодное в окне), детализация и движение; кадры на t и
t+{steps} шагов обязаны различаться, а каждая загрузка — совпадать побитово.

{lcg}
"""

L17_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>rain room</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const GLASS_Z = {glass_z};
const DRIFT = {drift};
const SPEED = {speed};

// Комната телами: потолок и пол уходят в глубину (BoxGeometry), боковые стены
// стоят под углом, стена с окном — за проёмом. Всё это даёт перспективу и
// затенение вместо плоской картинки.
const room = new THREE.Group();
room.name = 'room';
scene.add(room);
const boxAt = (w, h, d, x, y, z, material, name) => {{
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  mesh.name = name;
  mesh.position.set(x, y, z);
  room.add(mesh);
  return mesh;
}};
const D = CAM_Z - GLASS_Z;                       // глубина комнаты в единицах сцены
const H2 = (d) => Math.tan(45 * Math.PI / 360) * (CAM_Z - d);   // полувысота кадра
boxAt(FW(1.0, 3.0), 0.14, D, 0, FH(0.5, 3.0) - 0.06, GLASS_Z + D / 2, vgrad('{c_room_hi}', '{c_room_deep}'), 'ceiling');
boxAt(FW(1.0, 3.0), 0.18, D, 0, -FH(0.5, 3.0) + 0.06, GLASS_Z + D / 2, vgrad('{c_room_deep}', '0x0f0714'), 'floor_bottom');
boxAt(0.16, FH(1.0, CAM_Z - 4.0), D, -FW(0.5, 2.0), 0, GLASS_Z + D / 2, lit('{c_room}'), 'wall_left');
boxAt(0.16, FH(1.0, CAM_Z - 4.0), D, FW(0.5, 2.0), 0, GLASS_Z + D / 2, lit('{c_room}'), 'wall_right');
place(quad(1.0, 0.16, lit({c_room_hi}), 'wall_top', -6), 0.5, 0.18, -6);
// Откос окна: рама с толщиной, стекло утоплено — видно глубину проёма.
place(boxAt(FW(0.02, CAM_Z - GLASS_Z), FH(0.46, CAM_Z - GLASS_Z), 0.12, 0, 0, GLASS_Z, lit('{c_frame}'), 'frame_left'), 0, 0, GLASS_Z);
place(quad(1.0, 0.10, lit({c_lamp}), 'sill', -5.5), 0.5, {sill_v}, -5.5);
place(quad(1.0, 0.34, vgrad('{c_room}', '{c_room_deep}'), 'wall_low', -6), 0.5, 0.80, -6);
place(quad(1.0, 0.22, vgrad('{c_room_deep}', '0x0f0714'), 'floor', -3), 0.5, 0.92, -3);
// Рама окна: тёмные импосты поверх проёма, чтобы окно читалось как окно.
place(quad(0.02, 0.46, solid({c_frame}), 'frame_left', -5.4), 0.09, 0.37, -5.4);
place(quad(0.02, 0.46, solid({c_frame}), 'frame_right', -5.4), 0.91, 0.37, -5.4);
place(quad(0.90, 0.03, solid({c_frame}), 'frame_top', -5.4), 0.5, 0.15, -5.4);
place(quad(0.014, 0.46, solid({c_frame}), 'mullion_a', -5.35), 0.44, 0.37, -5.35);
place(quad(0.014, 0.46, solid({c_frame}), 'mullion_b', -5.35), 0.56, 0.37, -5.35);

// Окно: стекло в проёме, город за ним и струи дождя.
const glass = place(quad({glass_w}, {glass_h}, lit(0x9fc4d8, 0.22), 'glass', GLASS_Z), 0.5, {glass_mid}, GLASS_Z);
const city = place(quad(0.98, 0.42, lit({c_city_dark}, 1), 'city', -20), 0.5, 0.36, -20);
for (let i = 0; i < {windows}; i++) {{
  const w = 0.012, u = 0.10 + rnd() * 0.80, v = {city_top} + rnd() * 0.22;
  place(quad(w, 0.012, haze({c_city}, 0.85), 'window' + i, -19.5), u, v, -19.5);
}}
place(quad(0.16, 0.05, haze({c_city}, 0.6), 'neon_sign', -19.4), 0.22, 0.24, -19.4);
place(quad(0.10, 0.10, haze({c_city}, 0.4), 'neon_tall', -19.4), 0.74, 0.30, -19.4);
const rain = spray({drops_out}, 5, {c_rain});
{{
  const p = rain.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    p.setXYZ(i, (rnd() - 0.5) * FW(0.86, 16), (rnd() - 0.5) * FH(0.46, 16), -12 - rnd() * 0.01);
  }}
  p.needsUpdate = true;
}}

// Капли, которые едут по стеклу, — ближе стекла, чтобы читались поверх него.
const drops = spray({drops}, 6, {c_rain});
const dropSpan = FH({glass_h}, 10);
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    p.setXYZ(i, (rnd() - 0.5) * FW({glass_w} * 0.94, 10), (rnd() - 0.5) * dropSpan, GLASS_Z + 0.12);
  }}
  p.needsUpdate = true;
}}

// Растение на подоконнике, стол и монитор.
const leaf = lit({c_plant});
place(quad(0.09, 0.15, leaf, 'plant_a', -5.3), {plant_u}, 0.50, -5.3);
place(quad(0.08, 0.12, leaf, 'plant_b', -5.3), {plant_u} - 0.06, 0.53, -5.3);
place(quad(0.07, 0.10, leaf, 'plant_c', -5.3), {plant_u} + 0.06, 0.53, -5.3);
place(solid3(0.06, 0.06, 0.06, lit({c_room_deep}), 'pot', -5.3), {plant_u}, 0.60, -5.3);
place(solid3(0.34, 0.06, 0.04, lit({c_room_deep}), 'desk', -2), {desk_u}, {desk_v}, -2);
place(solid3(0.18, 0.16, 0.03, lit({c_frame}, 1), 'monitor', -2), {desk_u}, {desk_v} - 0.13, -2);
place(quad(0.16, 0.12, haze({c_city}, 0.55), 'screen', -2), {desk_u}, {desk_v} - 0.13, -1.9);
place(quad(0.20, 0.06, haze({c_lamp}, 0.45), 'desk_glow', -2), {desk_u}, {desk_v} + 0.02, -1.9);
place(quad(0.14, 0.05, haze({c_lamp}, 0.75), 'lamp', -5), {lamp_u}, {lamp_v}, -5);
place(quad(0.30, 0.04, haze({c_lamp}, 0.5), 'lamp_glow', -4.9), {lamp_u}, {lamp_v} + 0.05, -4.9);

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);

window.__step = (dt) => {{
  t += dt;
  const fall = (cloud, low, high, rate, dt) => {{
    const p = cloud.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {{
      let y = p.getY(i) - rate * fizz[i % 64] * dt;
      if (y < low) y += (high - low);
      p.setY(i, y);
    }}
    p.needsUpdate = true;
  }};
  fall(drops, -dropSpan / 2, dropSpan / 2, SPEED, dt);
  fall(rain, -FH(0.46, 16) / 2, FH(0.46, 16) / 2, SPEED * 1.7, dt);
  camera.position.z = Math.max(GLASS_Z + 2.4, camera.position.z - DRIFT * dt);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  camera: [round(camera.position.x), round(camera.position.z)],
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# L-18 — неоновая лапшичная под дождём
# --------------------------------------------------------------------------- #

L18_PALETTE = {
    "#1e1145": 0.34,  # индиговая ночь, стена и асфальт
    "#12c7d4": 0.006,  # циановая вывеска киоска
    "#ffb347": 0.008,  # тёплые лампы, пар и фонарь
    "#ff5fa8": 0.003,  # малиновые голопанели
    "#a8c8dc": 0.012,  # капли и дождь
}
L18 = {
    "sign_u": 0.13, "sign_v": 0.26, "sign_w": 0.06, "sign_h": 0.28,
    "board_u": 0.36, "board_v": 0.40, "board_w": 0.19,
    "drift": 0.05, "drops": 64, "speed": 2.1, "steam": 0.6,
}
L18_TITLE = TITLES["L-18"]

L18_PROMPT = """Собери ночную сцену «неоновая лапшичная под дождём»: кадр снят с
улицы сквозь мокрое стекло, за стеклом — киоск с неоновой вывеской, пар над
котлами, голопанели и асфальт с отражениями.

Композиция (проверяется по кадру грубо):
* верх кадра — стена и вывески, низ кадра занят асфальтом: средняя яркость
  трёх нижних полос профиля минимум на 2 единицы ниже трёх средних полос;
* тёплые пиксели (вывеска и лампы) — в верхней половине кадра;
* холодных пикселей на кадре не меньше, чем накрывает дождь, и лежат они ниже
  середины кадра (отражения и мокрый асфальт).

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_night} — ночное небо, стена и асфальт, не меньше {s_night};
* {c_sign} — циановая вывеска киоска (вертикальная, справа), не меньше {s_sign};
* {c_lamp} — тёплые лампы над стойкой и пар, не меньше {s_lamp};
* {c_holo} — малиновые голопанели слева, не меньше {s_holo};
* {c_rain} — капли на стекле и струи дождя, не меньше {s_rain}.

Геометрия (координаты — в долях кадра, камера в (0, 0, 4) смотрит в (0, 0, -1)):
* `sky` — глухая стена ночи z = -45 во весь кадр, цвет {c_night};
* `wall` — стена дома за киоском (z = -12), `floor` — асфальт от 0.62 высоты
  кадра вниз (z = -8), оба цвета {c_night};
* `kiosk` — стойка киоска слева: корпус 0.20x0.20 кадра на u = 0.24, v = 0.47,
  прилавок 0.26x0.05 кадра на v = 0.62; цвет корпуса {c_night}, прилавок — {c_lamp};
* `board` — вывеска над стойкой: ширина {board_w:.0%} кадра, u = {board_u},
  v = {board_v}, цвет {c_sign}, со строками-иероглифами (поперечные полосы 0.01 высоты);
* `sign` — вертикальная неоновая вывеска справа: {sign_w:.0%} на {sign_h:.0%} кадра,
  u = {sign_u}, v = {sign_v}, цвет {c_sign}, с миганием;
* `holo` — две голопанели слева ({holo_w:.0%} на {holo_h:.0%} кадра, u = {holo_u}, v = 0.30 и 0.32),
  цвет {c_holo}, по панелям бегут строки развёртки (фаза по времени);
* `lamps` — три тёплые лампы {c_lamp} над стойкой, u = 0.18 / 0.26 / 0.34, v = 0.12;
* `puddles` — четыре пятна на асфальте (0.16x0.05 кадра, u = 0.20/0.40/0.60/0.78,
  v = 0.82), цвет {c_rain}, по ним бегут круги-полосы;
* `glass` — мокрое стекло перед камерой: z = -4, 0.80x0.80 кадра, полупрозрачное;
* `drops` — {drops} капель цвета {c_rain} ПЕРЕД стеклом (z = -3.9), размер 6 px,
  падают вниз со скоростью {speed} единиц/с и заворачиваются;
* `rain` — струи дождя за стеклом (z = -12), те же точки, на треть медленнее;
* `steam` — три столба пара над стойкой: u = 0.20/0.24/0.28, v = 0.40, цвет {c_lamp},
  прозрачность качается с частотой {steam} в секунду.

Анимация:
* `window.__step(dt)` двигает капли, фазу развёртки и прозрачность пара, дрейфует
  камеру (`x` колеблется на {drift} в каждую сторону с частотой 0.25) и возвращает
  `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects`, `drops: [x, y]` первой капли и
  `steam` — текущую прозрачность пара (4 знака).

Проверяется: страница и кадр, палитра по долям, композиция (низ темнее верха,
тёплое сверху, холодное снизу), детализация и движение.

{lcg}
"""

L18_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>rain noodle bar</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const SPEED = {speed};
const DRIFT = {drift};
const STEAM = {steam};

// Ночь, стена, асфальт.
place(quad(2.0, 2.0, vgrad('0x1a0d2c', '0x2a1240'), 'sky', -60), 0.5, 0.5, -60);
place(quad(0.92, 0.30, lit({c_wall}), 'wall_far', -24), 0.5, 0.16, -24);
for (let i = 0; i < 5; i++) place(solid3(0.09, 0.16 + i * 0.05, 0.9, lit(i % 2 ? 0x241134 : 0x1a0c26), 'block' + i, -22), 0.13 + i * 0.19, 0.22 + i * 0.02, -22);
place(quad(0.86, 0.46, lit('{c_wall}'), 'wall', -16), 0.50, 0.42, -16);
place(quad(1.0, 0.40, vgrad('{c_ground}', '0x120819'), 'floor', -8), 0.5, 0.80, -8);
// Тумбы на асфальте: без них пол кажется бумагой.
solid3(0.10, 0.06, 0.30, lit('0x241133'), 'curb_a', -8).position.set(FW(0.22, CAM_Z + 8) / 2, -FH(0.30, CAM_Z + 8), -8);
place(solid3(0.86, 0.05, 0.25, lit({c_roof}), 'roof', -12), 0.50, 0.05, -12);
place(quad(0.60, 0.03, solid({c_lamp}), 'roof_edge', -11.9), 0.50, 0.085, -11.9);
for (let i = 0; i < 3; i++) place(quad(0.03, 0.34, solid({c_roof}), 'pillar' + i, -11.8), 0.14 + i * 0.36, 0.32, -11.8);

// Киоск: корпус, прилавок, тёмный проём.
box(0.20, 0.22, 1.2, lit({c_roof}), 'kiosk', -11);
place(solid3(0.22, 0.18, 0.04, lit(0x0e0718), 'kiosk_mouth', -10.7), 0.24, 0.55, -10.7);
place(solid3(0.26, 0.05, 0.10, lit({c_lamp}), 'counter', -10.6), 0.24, 0.655, -10.6);
place(quad(0.24, 0.03, haze({c_lamp}, 0.5), 'counter_glow', -10.5), 0.24, 0.695, -10.5);
const board = place(solid3({board_w}, 0.06, 0.04, lit({c_sign}), 'board', -11.5), {board_u}, {board_v}, -11.5);
for (let i = 0; i < 4; i++) {{
  place(quad({board_w} * 0.09, 0.014, solid(0x08262c), 'glyph' + i, -11.4), {board_u} - 0.06 + i * 0.04, {board_v}, -11.4);
}}
const sign = place(solid3({sign_w}, {sign_h}, 0.06, lit({c_sign}), 'sign', -11.6), {sign_u}, {sign_v}, -11.6);
place(quad({sign_w} * 0.46, {sign_h} * 0.86, solid(0x0b3a42), 'sign_core', -11.5), {sign_u}, {sign_v}, -11.5);

// Голопанели.
const holoMat = new THREE.ShaderMaterial({{
  uniforms: {{uPhase: {{value: 0.0}}}},
  vertexShader: `varying vec2 vUv; void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  fragmentShader: `uniform float uPhase; varying vec2 vUv;
    void main() {{
      float line = step(0.72, fract(vUv.y * 9.0 + uPhase));
      gl_FragColor = vec4(mix(vec3(0.30, 0.10, 0.28), vec3({holo_rgb}), line), 1.0);
    }}`,
}});
for (let i = 0; i < 2; i++) {{
  const panel = place(quad({holo_w}, {holo_h}, holoMat, 'holo' + i, -11.2), {holo_u}, 0.30 + i * 0.02, -11.2);
  place(quad({holo_w} * 1.08, {holo_h} * 0.04, solid(0x5a1240), 'holo_frame' + i, -11.1), {holo_u}, 0.30 + i * 0.02, -11.1);
}}

// Лампы, пар, лужи.
for (let i = 0; i < 3; i++) place(quad(0.03, 0.03, haze({c_lamp}, 0.85), 'lamp' + i, -8), 0.18 + i * 0.08, 0.12, -8);
const steam = haze({c_lamp}, 0.30);
for (let i = 0; i < 3; i++) place(quad(0.05, 0.13, steam, 'steam' + i, -10.4), 0.19 + i * 0.04, 0.40, -10.4);
for (let i = 0; i < 4; i++) place(quad(0.16, 0.05, haze({c_rain}, 0.5), 'puddle' + i, -9), 0.20 + i * 0.19, 0.82, -9);

// Стекло и капли.
place(quad(0.80, 0.80, haze(0x9fc4d8, 0.18), 'glass', -4), 0.5, 0.5, -4);
const drops = spray({drops}, 6, {c_rain});
const dropSpan = FH(0.70, 7.9);
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * FW(0.60, 7.9), (rnd() - 0.5) * dropSpan, -3.9);
  p.needsUpdate = true;
}}
const rain = spray(40, 4, {c_rain});
const rainSpan = FH(0.60, 16);
{{
  const p = rain.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * FW(0.80, 16), (rnd() - 0.5) * rainSpan, -12);
  p.needsUpdate = true;
}}

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);
const fall = (cloud, low, high, rate, dt) => {{
  const p = cloud.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    let y = p.getY(i) - rate * fizz[i % 64] * dt;
    if (y < low) y += (high - low);
    p.setY(i, y);
  }}
  p.needsUpdate = true;
}};

window.__step = (dt) => {{
  t += dt;
  fall(drops, -dropSpan / 2, dropSpan / 2, SPEED, dt);
  fall(rain, -rainSpan / 2, rainSpan / 2, SPEED * 0.66, dt);
  holoMat.uniforms.uPhase.value = t * 0.9;
  steam.opacity = 0.22 + 0.12 * Math.sin(t * STEAM * 6.2832);
  sign.material.color.setHex(Math.sin(t * 1.7) > -0.6 ? {c_sign} : 0x0d5f68);
  camera.position.x = Math.sin(t * 0.25) * DRIFT;
  camera.lookAt(camera.position.x, 0, -1);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  drops: [round(drops.geometry.attributes.position.getX(0)), round(drops.geometry.attributes.position.getY(0))],
  steam: round(steam.opacity),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# L-19 — трамвай сквозь дождливый город
# --------------------------------------------------------------------------- #

L19_PALETTE = {
    "#1a1230": 0.30,  # салон трамвая и ночь
    "#12c7d4": 0.010,  # неоновые пятна за окном
    "#ffa63c": 0.010,  # тёплые блики салона на стекле
    "#a8c8dc": 0.012,  # капли и струи по стеклу
}
L19 = {
    "glass_v": 0.10, "glass_h": 0.62, "glass_w": 0.78,
    "sway": 0.03, "drops": 60, "speed": 1.7, "blobs": 22,
}
L19_TITLE = TITLES["L-19"]

L19_PROMPT = """Собери сцену «трамвай сквозь дождливый город»: вид из вагона, большое
окно, по стеклу стекают капли, за окном проезжают неоновые пятна города, сверху
и снизу — тёплые блики салона на стекле. Камера почти статична.

Композиция (проверяется по кадру грубо):
* окно занимает центр кадра (от {glass_v:.0%} до {glass_bottom:.0%} высоты,
  ширина {glass_w:.0%}), кадр в полосе окна светлее нижней кромки кадра;
* тёплые блики лежат в верхней трети кадра;
* холодные пиксели (капли и пятна) — в полосе окна, ниже {cool_top:.0%} высоты.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_night} — салон, простенки и ночь за окном, не меньше {s_night};
* {c_neon} — неоновые пятна города за окном, не меньше {s_neon};
* {c_warm} — тёплые блики салона (потолочный и оконный), не меньше {s_warm};
* {c_rain} — капли и струи по стеклу, не меньше {s_rain}.

Геометрия (координаты — в долях кадра, камера в (0, 0, 4) смотрит в (0, 0, -1)):
* `night` — глухая ночь во весь кадр (z = -45), цвет {c_night};
* `cabin_top` — потолок салона (0..{glass_v:.0%} высоты), `cabin_low` — пол и
  стена под окном (от {glass_bottom:.0%} вниз), слева и справа — простенки шириной
  0.09 кадра, все цвета {c_night};
* `mullion` — два вертикальных импоста окна (0.012 кадра шириной) цвета {c_night},
  u = {mul_a} и u = {mul_b};
* `glass` — стекло окна {glass_w:.0%} x {glass_h:.0%} кадра на z = -4, opacity 0.30;
* `blobs` — {blobs} аддитивных пятен города за стеклом: z = -14, размер 7..13 px,
  цвет {c_neon}, каждое едет вбок со своей скоростью 2..4 единиц/с и
  заворачивается (неон проезжает мимо);
* `rail` — горизонтальная полоса отражения города на стекле (0.02 высоты, v = 0.58),
  цвет {c_neon}, opacity 0.4;
* `glow_top` — тёплый блик потолочной лампы на стекле: 0.70x0.05 кадра, v = {glow_v:.2f},
  цвет {c_warm};
* `glow_low` — тёплая полоса подоконника: 0.62x0.04 кадра, v = {glow_low:.2f}, {c_warm};
* `drops` — {drops} капель цвета {c_rain} по стеклу (z = -3.9), размер 6 px, едут
  вниз со скоростью {speed} единиц/с с разбросом по LCG.

Анимация:
* `window.__step(dt)` двигает капли вниз, двигает пятна вбок, чуть качает камеру
  (`x` колеблется на {sway} в каждую сторону) и возвращает `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects` и `blob: [x, y]` первого пятна (4 знака).

Проверяется: страница и кадр, палитра по долям, композиция, детализация и движение.

{lcg}

"""

L19_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>rain tram</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const SWAY = {sway};
const SPEED = {speed};

place(quad(1.8, 1.8, vgrad('0x241f38', '0x120f1c'), 'night', -60), 0.5, 0.5, -60);
place(quad(1.0, {glass_v}, vgrad('0x1c1436', '0x0e091c'), 'cabin_top', -6), 0.5, {cabin_top_v}, -6);
for (let i = 0; i < 3; i++) place(quad(0.08, 0.035, haze({c_warm}, 0.85), 'ceiling_light' + i, -5.7), 0.25 + i * 0.25, 0.045, -5.7);
place(quad(1.0, 0.05, solid(0x241840), 'sill_low', -5.5), 0.5, 0.72, -5.5);
place(quad(1.0, 0.10, vgrad('0x1d1438', '0x0f0a1e'), 'seat', -4.2), 0.5, 0.90, -4.2);
place(quad(1.0, {cabin_low}, solid({c_night}), 'cabin_low', -6), 0.5, {cabin_low_v}, -6);
place(quad(0.09, 1.0, solid({c_night}), 'wall_left', -6), 0.045, 0.5, -6);
place(quad(0.09, 1.0, solid({c_night}), 'wall_right', -6), 0.955, 0.5, -6);
place(quad(0.012, {glass_h}, solid(0x140d26), 'mullion_a', -5.9), {mul_a}, {glass_mid}, -5.9);
place(quad(0.012, {glass_h}, solid(0x140d26), 'mullion_b', -5.9), {mul_b}, {glass_mid}, -5.9);
place(quad({glass_w}, {glass_h}, haze(0x9fc4d8, 0.18), 'glass', -4), 0.5, {glass_mid}, -4);

const blobs = spray({blobs}, 9, {c_neon});
const blobSpeed = [];
{{
  const p = blobs.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    p.setXYZ(i, (rnd() - 0.5) * FW(1.4, 18), (0.06 + rnd() * 0.40) * FH(1, 18), -14);
    blobSpeed.push(2 + rnd() * 2);
  }}
  p.needsUpdate = true;
}}
place(quad(0.70, 0.02, haze({c_neon}, 0.4), 'rail', -4.6), 0.5, 0.58, -4.6);
place(quad(0.70, 0.05, haze({c_warm}, 0.75), 'glow_top', -4.5), 0.5, {glow_v}, -4.5);
place(quad(0.62, 0.04, haze({c_warm}, 0.55), 'glow_low', -4.5), 0.5, {glow_low}, -4.5);

const drops = spray({drops}, 6, {c_rain});
const dropSpan = FH(0.60, 7.9);
const dropHalf = FW(0.36, 7.9);
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * 2 * dropHalf, (rnd() - 0.5) * dropSpan, -3.9);
  p.needsUpdate = true;
}}

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);

window.__step = (dt) => {{
  t += dt;
  const pd = drops.geometry.attributes.position;
  for (let i = 0; i < pd.count; i++) {{
    let y = pd.getY(i) - SPEED * fizz[i % 64] * dt;
    if (y < -dropSpan / 2) y += dropSpan;
    pd.setY(i, y);
  }}
  pd.needsUpdate = true;
  const pb = blobs.geometry.attributes.position;
  for (let i = 0; i < pb.count; i++) {{
    let x = pb.getX(i) + blobSpeed[i] * dt;
    if (x > dropHalf * 2.2) x -= dropHalf * 4.4;
    pb.setX(i, x);
  }}
  pb.needsUpdate = true;
  camera.position.x = Math.sin(t * 0.6) * SWAY;
  camera.lookAt(camera.position.x, 0, -1);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  blob: [round(blobs.geometry.attributes.position.getX(0)), round(blobs.geometry.attributes.position.getY(0))],
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# L-20 — оранжерея на станции под метеорным дождём
# --------------------------------------------------------------------------- #

L20_PALETTE = {
    "#123038": 0.30,  # купол, корпус станции, ночь
    "#7fd054": 0.010,  # листва оранжереи
    "#ffc06a": 0.004,  # мягкие лампы под куполом
    "#a8c8dc": 0.010,  # дождь и капли по куполу
}
L20 = {
    "dome_z": -14.0, "pan_w": 0.34, "dome_h": 0.72,
    "lamp_v": 0.10, "drops": 64, "speed": 2.0, "meteors": 26, "fly": 2.6,
}
L20_TITLE = TITLES["L-20"]

L20_PROMPT = """Собери сцену «оранжерея на станции под метеорным дождём»: стеклянный
купол, растения в кадках, мягкие лампы под куполом, снаружи падают метеоры.
Камера пролетает между растениями.

Композиция (проверяется по кадру грубо):
* купол занимает верхнюю часть кадра: кадр выше {dome_line:.0%} ярче нижней
  трети минимум на 4 единицы (свет станции против тёмного пола);
* зелёные (листва) и тёплые (лампы) пиксели лежат в средней полосе кадра, не
  ниже {warm_top:.0%} и не выше {warm_bottom:.0%} высоты;
* холодные пиксели (дождь) — за куполом, в верхней половине кадра.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_dome} — стекло купола, корпус станции и ночь, не меньше {s_dome};
* {c_leaf} — листва растений, не меньше {s_leaf};
* {c_lamp} — мягкие лампы и блики на куполе, не меньше {s_lamp};
* {c_rain} — метеорный дождь и капли по стеклу, не меньше {s_rain}.

Геометрия (координаты — в долях кадра, камера в (0, 0, 4) смотрит в (0, 0, -1)):
* `dome_back` — задняя панель купола z = {dome_z}: {pan_back:.0%} x {dome_h:.0%} кадра (v = 0.31),
  цвет {c_dome}, opacity 0.85; две боковые панели повёрнуты на ±0.6 рад у
  u = 0.16 и u = 0.84 и стоят на z = -12;
* `floor` — тёмный пол станции от {dome_line:.0%} высоты вниз (z = -8), цвет {c_dome}, с тёплой кромкой {c_lamp};
* `rib` — рёбра купола: четыре наклонные тёмные полосы по 0.012 ширины;
* `lamp` — {lamps} мягких ламп цвета {c_lamp} под куполом (u = 0.28..0.72, v = {lamp_v}),
  каждая — плоскость 0.09x0.07 кадра с opacity 0.75;
* `plant` — {plants} растений: цилиндр-ствол {c_dome} и три листа {c_leaf} на
  z = -4..-9, высота 0.30..0.55 кадра, u = 0.12/0.30/0.68/0.86;
* `meteor` — {meteors} метеоров снаружи: z = -22, аддитивные точки размера 5 px,
  цвет {c_rain}, падают вниз-вбок со скоростью 6..9 единиц/с и заворачиваются;
* `drops` — {drops} капель по куполу: z = -13.5, размер 5 px, скорость {speed} единиц/с.

Анимация:
* `window.__step(dt)` двигает метеоры и капли, летит камерой между растениями
  (z уменьшается на {fly} единицы в секунду до 1.0, x колеблется на 0.6 единицы) и возвращает
  `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects`, `meteor: [x, y]` первого метеора и
  `camera: [x, z]` (4 знака).

{lcg}

Проверяется: страница и кадр, палитра по долям, композиция, детализация и движение.

{lcg}

"""

L20_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>meteor greenhouse</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const FLY = {fly};
const SPEED = {speed};
const DOME_Z = {dome_z};

place(quad(2.4, 2.4, vgrad('0x16323c', '0x061014'), 'night', -60), 0.5, 0.5, -60);
place(quad({pan_w} * 2.1, {dome_h}, haze({c_dome}, 0.85), 'dome_back', DOME_Z), 0.5, 0.31, DOME_Z);
place(quad(1.9, 1.5, haze({c_dome}, 0.5), 'dome_far', -34), 0.5, 0.34, -34);
place(quad({pan_w}, {dome_h} * 0.9, haze({c_dome}, 0.6), 'dome_left', -12), 0.14, 0.34, -12).rotation.y = 0.55;
place(quad({pan_w}, {dome_h} * 0.9, haze({c_dome}, 0.6), 'dome_right', -12), 0.86, 0.34, -12).rotation.y = -0.55;
place(quad(2.4, 0.44, vgrad('0x12242c', '0x061014'), 'floor', -10), 0.5, 0.80, -10);
place(quad(2.4, 0.02, solid({c_lamp}), 'floor_lip', -9.8), 0.5, 0.605, -9.8);
for (let i = 0; i < 4; i++) place(quad(0.016, 0.62, solid(0x08161c), 'rib' + i, -13), 0.20 + i * 0.20, 0.32, -13).rotation.z = (i % 2 ? 0.12 : -0.12);

for (let i = 0; i < {lamps}; i++) place(quad(0.09, 0.07, haze({c_lamp}, 0.8), 'lamp' + i, -9), 0.30 + i * 0.10, {lamp_v}, -9);

const leaf = solid({c_leaf});
const plants = [];
for (let i = 0; i < {plants}; i++) {{
  const u = 0.12 + (i % 4) * 0.25, h = 0.34 + (i % 3) * 0.12, z = -4 - (i % 3) * 2.4;
  const base = 0.88;
  plants.push(place(quad(0.04, h, solid(0x08202a), 'stem' + i, z), u, base - h / 2, z));
  place(quad(0.08, 0.05, leaf, 'leaf_a' + i, z + 0.15), u - 0.055, base - h * 0.72, z + 0.15);
  place(quad(0.07, 0.05, leaf, 'leaf_b' + i, z + 0.15), u + 0.055, base - h * 0.52, z + 0.15);
  place(quad(0.10, 0.06, leaf, 'leaf_top' + i, z + 0.15), u, base - h - 0.02, z + 0.15);
  place(quad(0.08, 0.07, solid(0x08202a), 'pot' + i, z), u, base - 0.03, z);
}}

const meteors = spray({meteors}, 5, {c_rain});
const meteorSpan = FH(0.9, 26), meteorWide = FW(1.3, 26);
const meteorSpeed = [];
{{
  const p = meteors.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    p.setXYZ(i, (rnd() - 0.5) * meteorWide, (rnd() - 0.5) * meteorSpan, -22);
    meteorSpeed.push(6 + rnd() * 3);
  }}
  p.needsUpdate = true;
}}
const drops = spray({drops}, 5, {c_rain});
const dropSpan = FH(0.80, 10), dropWide = FW(0.86, 10);
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * dropWide, (rnd() - 0.5) * dropSpan, -13.5);
  p.needsUpdate = true;
}}

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);

window.__step = (dt) => {{
  t += dt;
  const pm = meteors.geometry.attributes.position;
  for (let i = 0; i < pm.count; i++) {{
    let y = pm.getY(i) - meteorSpeed[i] * dt;
    let x = pm.getX(i) + meteorSpeed[i] * 0.45 * dt;
    if (y < -meteorSpan / 2) {{ y += meteorSpan; x -= meteorWide * 0.45; }}
    if (x > meteorWide / 2) x -= meteorWide;
    pm.setXYZ(i, x, y, -22);
  }}
  pm.needsUpdate = true;
  const pd = drops.geometry.attributes.position;
  for (let i = 0; i < pd.count; i++) {{
    let y = pd.getY(i) - SPEED * fizz[i % 64] * dt;
    if (y < -dropSpan / 2) y += dropSpan;
    pd.setY(i, y);
  }}
  pd.needsUpdate = true;
  camera.position.z = Math.max(1.0, camera.position.z - FLY * dt);
  camera.position.x = Math.sin(t * 0.35) * 0.6;
  camera.lookAt(camera.position.x * 0.4, 0, -1);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  meteor: [round(meteors.geometry.attributes.position.getX(0)), round(meteors.geometry.attributes.position.getY(0))],
  camera: [round(camera.position.x), round(camera.position.z)],
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# L-21 — аналоговая рубка с дождём за стеклом
# --------------------------------------------------------------------------- #

L21_PALETTE = {
    "#2a1c4a": 0.30,  # корпус рубки, панель, пол
    "#57e08a": 0.008,  # зелёные экраны приборов
    "#ffd166": 0.002,  # янтарные индикаторы и тумблеры
    "#a8c8dc": 0.012,  # дождь за стеклом и капли
}
L21 = {
    "panel_v": 0.30, "screen_u": 0.22, "glass_v": 0.12, "glass_h": 0.46,
    "drift": 0.04, "drops": 58, "speed": 1.8, "blink": 1.1, "flash": 0.35,
}
L21_TITLE = TITLES["L-21"]

L21_PROMPT = """Собери сцену «аналоговая рубка с дождём за стеклом»: пульт с
тумблерами и индикаторами, четыре экрана приборов, за широким стеклом — дождь и
вспышки грозы. Камера медленно дрейфует.

Композиция (проверяется по кадру грубо):
* приборная панель — яркая полоса в верхней трети кадра: средняя яркость верхних
  двух полос профиля минимум на 5 единиц выше средней;
* низ кадра (пол и корпус) — тёмный: яркость последней полосы профиля ниже 60;
* янтарные и зелёные пиксели приборов лежат в полосе {panel_from:.0%}..{panel_to:.0%}
  высоты кадра (экраны слева и справа от центра, индикаторы по центру);
* холодные пиксели (дождь) — в полосе стекла, выше полосы панели.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_hull} — корпус рубки, панель и пол, не меньше {s_hull};
* {c_screen} — зелёные экраны приборов (u = {screen_u} и {screen_u2}), не меньше {s_screen};
* {c_amber} — янтарные индикаторы и тумблеры по центру панели, не меньше {s_amber};
* {c_rain} — дождь за стеклом и капли по стеклу, не меньше {s_rain}.

Геометрия (координаты — в долях кадра, камера в (0, 0, 4) смотрит в (0, 0, -1)):
* `night` — глухая ночь за стеклом (z = -45), цвет {c_hull};
* `glass` — стекло рубки: 0.78x{glass_h:.0%} кадра, верх на {glass_v:.0%} высоты
  кадра, z = -5, opacity 0.35; над стеклом — козырёк {c_hull};
* `rain` — струи дождя за стеклом: z = -12, {drops_out} аддитивных точек цвета {c_rain},
  размер 5 px, падают со скоростью {speed} единиц/с;
* `skyline` — пять силуэтов домов за стеклом (z = -14), цвет тёмно-синий 0x1a2140;
* `drops` — {drops} капель по стеклу (z = -4.9), размер 6 px, та же скорость;
* `panel` — приборная панель: {panel_w:.0%} x {panel_h:.0%} кадра, u = {panel_u},
  v = {panel_v}, цвет {c_hull}; наклонена на 0.25 рад;
* `screen` — четыре экрана {c_screen} (0.11x0.10 кадра) в ряд по панели
  (u = {screen_u}..{screen_u2}), с бегущей развёрткой;
* `dial` — {dials} янтарных индикаторов {c_amber} на панели (u = 0.45..0.55, v по панели);
* `toggle` — {toggles} тумблеров {c_amber} под индикаторами;
* `floor` — тёмный пол от {floor_line:.0%} вниз, цвет {c_hull};
* `flash` — вспышка грозы: аддитивное пятно {c_rain} 0.5x0.4 кадра на u = {flash_u},
  v = {flash_v} за стеклом (z = -14), яркость качается с частотой {blink} в секунду.

Анимация:
* `window.__step(dt)` двигает капли и струи, качает яркость вспышки и индикаторов,
  дрейфует камеру на {drift} в секунду по x и возвращает `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects`, `flash` — яркость вспышки (4 знака).

{lcg}
"""

L21_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>analog rain cockpit</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const SPEED = {speed};
const BLINK = {blink};
const DRIFT = {drift};

place(quad(1.8, 1.8, vgrad('0x241a44', '0x0f0a1e'), 'night', -60), 0.5, 0.5, -60);
place(quad(1.0, {glass_v}, solid(0x241a44), 'canopy', -6), 0.5, {canopy_v}, -6);
place(quad(1.0, {below}, solid(0x1f1538), 'hull_low', -6), 0.5, {below_v}, -6);
place(quad(1.0, 0.05, solid(0x0d0818), 'window_sill', -5.6), 0.5, {sill_v}, -5.6);
place(quad(1.0, {glass_h}, haze(0x9fc4d8, 0.16), 'glass', -5), 0.5, {glass_mid}, -5);
for (let i = 0; i < 5; i++) place(quad(0.10, 0.10 + i * 0.02, solid(0x1a2140), 'skyline' + i, -14), 0.12 + i * 0.18, 0.30, -14);

const rain = spray(48, 6, {c_rain});
const rainSpan = FH(0.40, 16), rainWide = FW(0.80, 16);
{{
  const p = rain.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * rainWide, {glass_mid_units} + (rnd() - 0.5) * rainSpan, -12);
  p.needsUpdate = true;
}}
const drops = spray({drops}, 6, {c_rain});
const dropSpan = FH(0.40, 8.9), dropWide = FW(0.72, 8.9);
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * dropWide, {glass_mid_units} + (rnd() - 0.5) * dropSpan, -4.9);
  p.needsUpdate = true;
}}

// Панель с экранами, индикаторами и тумблерами.
const panel = place(quad({panel_w}, {panel_h}, solid({c_hull}), 'panel', -3), {panel_u}, {panel_v}, -3);
panel.rotation.x = -0.25;
place(quad({panel_w} * 1.04, 0.02, solid(0x1d1436), 'panel_lip', -2.9), {panel_u}, {panel_v} + 0.10, -2.9);

const screenMat = new THREE.ShaderMaterial({{
  uniforms: {{uPhase: {{value: 0.0}}}},
  vertexShader: `varying vec2 vUv; void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  fragmentShader: `uniform float uPhase; varying vec2 vUv;
    void main() {{
      float scan = 0.55 + 0.45 * step(0.66, fract(vUv.y * 7.0 - uPhase));
      float bars = 0.7 + 0.3 * step(0.5, fract(vUv.x * 5.0 + uPhase * 0.5));
      gl_FragColor = vec4(vec3(0.10, 0.55, 0.28) * scan * bars + vec3(0.02, {screen_b}, 0.05), 1.0);
    }}`,
}});
for (let i = 0; i < 4; i++) {{
  place(quad(0.10, 0.09, screenMat, 'screen' + i, -2.85), 0.22 + i * 0.28, {panel_v} - 0.02, -2.85);
  place(quad(0.12, 0.11, solid(0x120c26), 'screen_frame' + i, -2.95), 0.20 + i * 0.20, {panel_v} - 0.02, -2.95);
}}
for (let i = 0; i < {dials}; i++) {{
  place(quad(0.032, 0.030, solid({c_amber}), 'dial' + i, -2.8), 0.46 + i * 0.03, {panel_v} - 0.062, -2.8);
}}
for (let i = 0; i < {toggles}; i++) {{
  place(quad(0.018, 0.038, solid({c_amber}), 'toggle' + i, -2.8), 0.46 + i * 0.03, {panel_v} + 0.028, -2.8);
}}
const flash = place(quad(0.50, 0.40, haze({c_rain}, 0.25), 'flash', -14), {flash_u}, {flash_v}, -14);
place(quad(1.0, 0.26, vgrad('0x241a44', '0x100a20'), 'floor', -1), 0.5, {floor_v}, -1);
place(quad(0.34, 0.16, solid(0x241a42), 'console', -1.6), 0.20, {console_v}, -1.6);
for (let i = 0; i < 4; i++) place(quad(0.05, 0.012, solid({c_amber}), 'led' + i, -1.5), 0.11 + i * 0.06, {console_v} + 0.02, -1.5);
place(quad(0.30, 0.04, solid(0x120c26), 'seat_base', -1.2), 0.72, {seat_v}, -1.2);
place(quad(0.26, 0.12, solid(0x241a42), 'seat_back', -1.1), 0.72, {seat_v} - 0.07, -1.1);
place(quad(0.30, 0.03, haze({c_screen}, 0.25), 'panel_glow', -2.7), 0.50, {panel_v} + 0.10, -2.7);

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);

window.__step = (dt) => {{
  t += dt;
  const fall = (cloud, low, high, rate, dt) => {{
    const p = cloud.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {{
      let y = p.getY(i) - rate * fizz[i % 64] * dt;
      if (y < low) y += (high - low);
      p.setY(i, y);
    }}
    p.needsUpdate = true;
  }};
  fall(drops, -dropSpan / 2, dropSpan / 2, SPEED, dt);
  fall(rain, -rainSpan / 2, rainSpan / 2, SPEED * 1.4, dt);
  screenMat.uniforms.uPhase.value = t * 0.6;
  flash.material.opacity = 0.10 + 0.30 * (0.5 + 0.5 * Math.sin(t * BLINK * 6.2832));
  camera.position.x = Math.sin(t * 0.3) * DRIFT;
  camera.lookAt(camera.position.x, 0, -1);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  flash: round(flash.material.opacity),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# L-22 — робот-ремонтник под дождём
# --------------------------------------------------------------------------- #

L22_PALETTE = {
    "#1b1a2a": 0.30,  # асфальт, стены и ночь
    "#ff2f7a": 0.006,  # неоновая вывеска и её отражение
    "#ffd94a": 0.001,  # искры и лампа робота
    "#a8c8dc": 0.0015,  # дождь, капли и блики
}
L22 = {
    "sign_u": 0.30, "sign_v": 0.16, "sign_w": 0.30, "sign_h": 0.12,
    "robot_u": 0.42, "robot_v": 0.66, "robot_w": 0.20,
    "approach": 1.15, "drops": 54, "speed": 2.0, "sparks": 22,
}
L22_TITLE = TITLES["L-22"]

L22_PROMPT = """Собери сцену «робот-ремонтник под дождём»: маленький робот у
неоновой вывески на мокром асфальте, из-под его руки летят искры, вокруг лужи с
отражением вывески. Камера приближается к роботу.

Композиция (проверяется по кадру грубо):
* вывеска — яркая полоса в верхней четверти кадра (тёплые и малиновые пиксели
  выше {sign_line:.0%} высоты);
* низ кадра темнее верха: средняя яркость нижних двух полос профиля минимум на
  3 единицы ниже верхних двух;
* холодные пиксели (дождь и лужи) лежат ниже середины кадра.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_night} — асфальт, стены и ночь, не меньше {s_night};
* {c_sign} — неоновая вывеска и её отражение в лужах, не меньше {s_sign};
* {c_spark} — искры и лампа робота, не меньше {s_spark};
* {c_rain} — дождь, капли и блики на асфальте, не меньше {s_rain}.

Геометрия (координаты — в долях кадра, камера в (0, 0, 4) смотрит в (0, 0, -1)):
* `night` — глухая ночь (z = -45), `wall` — стена дома (z = -16), `floor` (цвет {c_ground}) —
  асфальт от {floor_line:.0%} высоты кадра вниз (z = -8), всё цвета {c_night};
* `sign` — вывеска: {sign_w:.0%} x {sign_h:.0%} кадра, u = {sign_u}, v = {sign_v},
  цвет {c_sign}, с миганием (раз в 1.4 с);
* `robot` — робот на асфальте u = {robot_u}, v = {robot_v}, ширина {robot_w:.0%} кадра:
  корпус 0.55 высоты робота, голова 0.30, две ноги 0.35, две руки 0.45,
  глаз {c_rain} (0.12 ширины робота), лампа {c_spark} на плече; корпус и ноги — {c_night};
* `wet` — полоса мокрого асфальта с бликом {c_rain} (0.96x0.06 кадра, v = 0.945);
* `puddle` — три лужи под роботом и вывеской (0.20x0.05 кадра, v = 0.88),
  цвет {c_sign} с opacity 0.4 (отражение вывески);
* `drops` — {drops} капель цвета {c_rain} перед камерой (z = -3.9), размер 8 px,
  падают со скоростью {speed} единиц/с;
* `sparks` — {sparks} искр цвета {c_spark}: аддитивные точки размера 6 px у плеча
  робота (z = -6), падают вниз со скоростью 3..4.5 единиц/с и заворачиваются;
* камера приближается: z уменьшается на {approach} единицы в секунду до 0.8.

Анимация:
* `window.__step(dt)` двигает капли и искры, мигает вывеска и возвращает
  `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects`, `spark: [x, y]` первой искры и
  `camera: [x, z]` (4 знака).

{lcg}
"""

L22_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>repair robot in rain</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const APPROACH = {approach};
const SPEED = {speed};

place(quad(1.8, 1.8, vgrad('0x241f38', '0x120f1c'), 'night', -60), 0.5, 0.5, -60);
place(quad(1.0, 0.86, vgrad('0x2b2640', '0x191624'), 'wall', -16), 0.5, 0.43, -16);
place(quad(1.0, 0.46, vgrad('{c_ground}', '0x0c0b13'), 'floor', -8), 0.5, 0.84, -8);

const sign = place(quad({sign_w}, {sign_h}, solid({c_sign}), 'sign', -13), {sign_u}, {sign_v}, -13);
place(quad({sign_w} * 0.86, {sign_h} * 0.62, solid(0x3b0a1e), 'sign_core', -12.9), {sign_u}, {sign_v}, -12.9);
place(quad({sign_w} * 1.05, 0.02, solid(0x2a141f), 'sign_bar', -12.8), {sign_u}, {sign_v} + 0.07, -12.8);
place(quad({sign_w} * 1.3, {sign_h} * 1.6, haze({c_sign}, 0.18), 'sign_halo', -12.7), {sign_u}, {sign_v}, -12.7);

// Робот: корпус, голова, глаз, лампа, руки и ноги.
const R = {robot_w} * FH(1, 9.4);
const robot = new THREE.Group();
robot.name = 'robot';
{{
  const body = new THREE.Mesh(new THREE.BoxGeometry(R * 0.55 * FH(1, 9.4) / FH(1, 9.4) * 1.33, R * 0.55, R * 0.30), solid({c_night}));
  body.position.y = 0;
  const head = new THREE.Mesh(new THREE.BoxGeometry(R * 0.38 * 1.33, R * 0.30, R * 0.26), solid(0x2a2740));
  head.position.y = R * 0.44;
  const eye = new THREE.Mesh(new THREE.PlaneGeometry(R * 0.20 * 1.33, R * 0.09), solid({c_rain}));
  eye.position.set(0, R * 0.46, R * 0.14);
  const lamp = new THREE.Mesh(new THREE.PlaneGeometry(R * 0.14 * 1.33, R * 0.12), haze({c_spark}, 0.9));
  lamp.position.set(R * 0.22 * 1.33, R * 0.34, R * 0.16);
  for (const s of [-1, 1]) {{
    const leg = new THREE.Mesh(new THREE.BoxGeometry(R * 0.12 * 1.33, R * 0.35, R * 0.14), solid(0x100f1c));
    leg.position.set(s * R * 0.16 * 1.33, -R * 0.45, 0);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(R * 0.10 * 1.33, R * 0.45, R * 0.12), solid(0x2a2740));
    arm.position.set(s * R * 0.32 * 1.33, R * 0.02, 0);
    arm.rotation.z = s * 0.25;
    robot.add(leg, arm);
  }}
  robot.add(body, head, eye, lamp);
}}
robot.position.set(({robot_u} - 0.5) * FW(1, 9.4), (0.5 - {robot_v}) * FH(1, 9.4) - FH(0.03, 9.4), -6 + 0.4);
scene.add(robot);

for (let i = 0; i < 3; i++) place(quad(0.20, 0.05, haze({c_sign}, 0.4), 'puddle' + i, -7), 0.22 + i * 0.26, 0.845, -7);
place(quad(0.92, 0.05, haze({c_rain}, 0.4), 'wet_asphalt', -7.4), 0.5, 0.925, -7.4);

const drops = spray({drops}, 8, {c_rain});
const dropSpan = FH(0.86, 7.9), dropWide = FW(0.92, 7.9);
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * dropWide, (rnd() - 0.5) * dropSpan, -3.9);
  p.needsUpdate = true;
}}
const sparks = spray({sparks}, 8, {c_spark});
const sparkSpan = FH(0.30, 10), sparkWide = FW(0.10, 10);
const sparkSpeed = [];
{{
  const p = sparks.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    p.setXYZ(i, (rnd() - 0.5) * sparkWide, R * 0.30 + (rnd() - 0.5) * 0.3, -6);
    sparkSpeed.push(3 + rnd() * 1.5);
  }}
  p.needsUpdate = true;
}}

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);

window.__step = (dt) => {{
  t += dt;
  const pd = drops.geometry.attributes.position;
  for (let i = 0; i < pd.count; i++) {{
    let y = pd.getY(i) - SPEED * fizz[i % 64] * dt;
    if (y < -dropSpan / 2) y += dropSpan;
    pd.setY(i, y);
  }}
  pd.needsUpdate = true;
  const ps = sparks.geometry.attributes.position;
  for (let i = 0; i < ps.count; i++) {{
    let y = ps.getY(i) - sparkSpeed[i] * dt;
    let x = ps.getX(i) + (fizz[i % 64] - 1) * 0.5 * dt;
    if (y < sparkSpan / 2 - sparkSpan) {{ y += sparkSpan; x = (fizz[i % 64] - 1) * 0.3; }}
    ps.setXYZ(i, x, y, -6);
  }}
  ps.needsUpdate = true;
  sign.material.color.setHex(Math.sin(t * 4.5) > -0.5 ? {c_sign} : 0x6b1236);
  camera.position.z = Math.max(0.8, camera.position.z - APPROACH * dt);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  spark: [round(sparks.geometry.attributes.position.getX(0)), round(sparks.geometry.attributes.position.getY(0))],
  camera: [round(camera.position.x), round(camera.position.z)],
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# L-23 — космическая заправка под дождём
# --------------------------------------------------------------------------- #

L23_PALETTE = {
    "#241644": 0.30,  # корпус станции, платформа и ночь
    "#35d5e6": 0.006,  # циановые голограммы и неон заправки
    "#ffb347": 0.005,  # янтарные огни ровера и башни
    "#a8c8dc": 0.010,  # дождь, туман и капли
}
L23 = {
    "tower_u": 0.20, "tower_v": 0.30, "hover": 0.06,
    "drops": 76, "speed": 2.2, "orbit": 1.6, "fog": 0.22,
}
L23_TITLE = TITLES["L-23"]

L23_PROMPT = """Собери сцену «космическая заправка под дождём»: ночная станция, ровер
на светящейся платформе, заправочная башня с шлангом, туман и неон. Камера облетает
ровер по дуге.

Композиция (проверяется по кадру грубо):
* платформа светлее фона: средняя яркость полос 4..6 профиля минимум на 4 единицы
  выше полос 0..1 (туман и небо);
* ровер с огнями — в средней полосе кадра (тёплые пиксели между {warm_top:.0%} и
  {warm_bottom:.0%} высоты);
* холодные пиксели (голограммы, туман, дождь) шире {cool_share:.0%} кадра.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_hull} — корпус станции, платформа и ночь, не меньше {s_hull};
* {c_neon} — циановые голограммы, неон заправки и зарядные полосы, не меньше {s_neon};
* {c_amber} — янтарные огни ровера и башни, не меньше {s_amber};
* {c_rain} — дождь, туман и капли, не меньше {s_rain}.

Геометрия (координаты — в долях кадра, камера в (0, 0, 4) смотрит в (0, 0, -1)):
* `night` — глухая ночь (z = -45), `hull` — корпус станции (z = -18), оба {c_hull};
* `pad` — платформа 0.52x0.10 кадра на v = {pad_v}, u = 0.52, цвет {c_neon} с
  opacity 0.5, под ней тёмное основание {c_hull};
* `rover` — ровер на платформе u = {rover_u}, v = {rover_v}: корпус 0.60 высоты
  объекта, кабина 0.30, два колеса {c_hull}, три огня {c_amber} (0.10 ширины);
* `tower` — заправочная башня u = {tower_u}, v = {tower_v}, 0.05x0.46 кадра,
  цвет {c_hull}; шланг {c_neon} (0.02 ширины, изогнут дугой из трёх отрезков);
* `holo` — три циановые голопанели за башней (0.09x0.12 кадра, v = {holo_v});
* `fog` — два слоя тумана: ровные полосы {c_rain} с opacity {fog} на v = 0.72 и
  v = 0.44 поперёк кадра;
* `drops` — {drops} капель {c_rain} перед камерой (z = -3.9), размер 6 px, падают
  со скоростью {speed} единиц/с;
* камера облетает ровер: x колеблется на 0.7 единицы с частотой 0.25, z уменьшается
  на {orbit} единицы в секунду до 2.0, взгляд смещён на 0.3 влево.

Анимация:
* `window.__step(dt)` двигает капли, мигает неон и облетает ровер, возвращая
  `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects`, `camera: [x, z]` и `pad` —
  прозрачность платформы (4 знака).

{lcg}
"""

L23_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>rain space fuel stop</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const SPEED = {speed};
const ORBIT = {orbit};

place(quad(2.6, 2.6, vgrad('0x2c1b52', '0x1a0f34'), 'night', -60), 0.5, 0.5, -60);
place(quad(2.2, 1.6, vgrad('0x281845', '0x150d2a'), 'hull', -26), 0.5, 0.42, -26);
for (let i = 0; i < 3; i++) {{  // силуэты надстроек станции
  place(quad(0.34, 0.10 + i * 0.05, solid(0x160e30), 'block' + i, -25), 0.22 + i * 0.28, 0.14, -25);
}}
place(quad(2.2, 0.30, solid(0x120b28), 'far_floor', -30), 0.5, 0.88, -30);

const padGlow = haze({c_neon}, 0.55);
place(quad(0.30, 0.05, padGlow, 'pad', -15), 0.54, {pad_v}, -15);
place(quad(0.33, 0.04, solid(0x150e2c), 'pad_base', -14.9), 0.54, {pad_v} + 0.045, -14.9);
place(quad(0.33, 0.010, solid({c_neon}), 'pad_rim', -14.8), 0.54, {pad_v} - 0.028, -14.8);
place(quad(0.36, 0.05, solid(0x140d2c), 'pad_shadow', -14.6), 0.54, {pad_v} + 0.075, -14.6);

// Ровер на платформе.
const roverU = {rover_u}, roverV = {rover_v};
place(quad(0.15, 0.055, solid(0x170f30), 'rover_body', -14.2), roverU, roverV, -14.2);
place(quad(0.09, 0.05, solid(0x2a1d4c), 'rover_cab', -14.1), roverU - 0.015, roverV - 0.052, -14.1);
place(quad(0.045, 0.028, solid({c_amber}), 'rover_window', -14.0), roverU - 0.015, roverV - 0.052, -14.0);
for (const s of [-1, 1]) {{
  place(quad(0.045, 0.045, solid(0x0d0918), 'wheel', -14.1), roverU + s * 0.048, roverV + 0.030, -14.1);
  place(quad(0.012, 0.030, solid(0x2a1d4c), 'strut', -14.0), roverU + s * 0.055, roverV - 0.005, -14.0);
}}
place(quad(0.09, 0.07, haze({c_amber}, 0.95), 'rover_lamp', -14.0), roverU + 0.062, roverV - 0.026, -14.0);
for (let i = 0; i < 3; i++) place(quad(0.036, 0.028, solid({c_amber}), 'rover_light' + i, -14.0), roverU - 0.060 + i * 0.034, roverV + 0.032, -14.0);
place(quad(0.26, 0.03, haze({c_amber}, 0.35), 'rover_glow', -14.0), roverU, roverV + 0.05, -14.0);

// Заправочная башня со шлангом к роверу.
// Заправочная ферма: мачта с рукой над платформой и шлангом к роверу.
place(quad(0.030, 0.46, solid(0x1b1236), 'tower', -18), {tower_u}, {tower_v} + 0.02, -18);
place(quad(0.032, 0.020, solid(0x0f0a22), 'tower_foot', -17.9), {tower_u}, {tower_v} + 0.25, -17.9);
place(quad(0.10, 0.020, haze({c_amber}, 0.8), 'tower_lamp', -17.9), {tower_u} + 0.02, {tower_v} - 0.21, -17.9);
place(quad(0.24, 0.022, solid(0x241a3e), 'gantry', -17.8), {tower_u} + 0.11, {tower_v} - 0.10, -17.8);
place(quad(0.012, 0.070, solid({c_neon}), 'hose', -17.7), {tower_u} + 0.20, {tower_v} - 0.06, -17.7);

// Голопанели и туман.
for (let i = 0; i < 3; i++) {{
  const hu = 0.62 + i * 0.15, hv = {holo_v} + (i % 2) * 0.04;
  place(quad(0.11, 0.15, haze({c_neon}, 0.35), 'holo' + i, -28), hu, hv, -28);
  place(quad(0.115, 0.006, solid({c_neon}), 'holo_top' + i, -27.9), hu, hv - 0.065, -27.9);
  place(quad(0.06, 0.006, solid(0x0d3b42), 'holo_row' + i, -27.8), hu - 0.02, hv + 0.02, -27.8);
}}
place(quad(2.2, 0.10, haze({c_rain}, {fog}), 'fog_low', -20), 0.5, 0.80, -20);
place(quad(2.2, 0.08, haze({c_rain}, {fog}), 'fog_mid', -24), 0.5, 0.44, -24);

const drops = spray({drops}, 8, {c_rain});
const dropSpan = FH(0.90, 7.9), dropWide = FW(0.98, 7.9);
const plink = spray(24, 3, {c_rain});   // мелкая морось у платформы
{{
  const p = plink.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * FW(1.2, 20), (0.1 + rnd() * 0.2) * FH(1, 20), -18);
  p.needsUpdate = true;
}}
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * dropWide, (rnd() - 0.5) * dropSpan, -3.9);
  p.needsUpdate = true;
}}

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);

window.__step = (dt) => {{
  t += dt;
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    let y = p.getY(i) - SPEED * fizz[i % 64] * dt;
    if (y < -dropSpan / 2) y += dropSpan;
    p.setY(i, y);
  }}
  p.needsUpdate = true;
  padGlow.opacity = 0.38 + 0.16 * Math.sin(t * 1.9);
  camera.position.x = Math.sin(t * 0.25) * 0.7;
  camera.position.z = Math.max(2.0, camera.position.z - ORBIT * dt);
  camera.lookAt(-0.3, 0, -1);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  camera: [round(camera.position.x), round(camera.position.z)],
  pad: round(padGlow.opacity),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# L-24 — автобусная остановка в неоне
# --------------------------------------------------------------------------- #

L24_PALETTE = {
    "#1c1236": 0.30,  # ночь, навес, асфальт
    "#22d3c5": 0.012,  # голорасписание
    "#ffa83c": 0.004,  # тёплая лампа под навесом
    "#ff4d9d": 0.006,  # малиновый неон за стеклом и в лужах
    "#a8c8dc": 0.010,  # капли и дождь
}
L24 = {
    "glass_v": 0.16, "glass_h": 0.44, "glass_w": 0.62,
    "board_v": 0.30, "lamp_v": 0.14,
    "sway": 0.022, "drops": 56, "speed": 1.9,
}
L24_TITLE = TITLES["L-24"]

L24_PROMPT = """Собери сцену «автобусная остановка в неоне»: навес, стеклянная
панель с каплями, голорасписание с бегущими строками, лужи и малиновый неон за
стеклом. Камера почти статична.

Композиция (проверяется по кадру грубо):
* стекло с улицей занимает центр кадра ({glass_w:.0%} ширины, от {glass_v:.0%} до
  {glass_bottom:.0%} высоты), навес — верхняя полоса до {glass_v:.0%};
* холодные пиксели (дождь, стекло, лужи) лежат ниже {cool_top:.0%} высоты кадра;
* тёплая лампа и неон — в верхней половине кадра, лужи с отражением — в нижней
  четверти.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_night} — ночь, навес, стойки и асфальт, не меньше {s_night};
* {c_holo} — голорасписание с бегущими строками, не меньше {s_holo};
* {c_lamp} — тёплая лампа под навесом, не меньше {s_lamp};
* {c_neon} — малиновый неон за стеклом и его отражение в лужах, не меньше {s_neon};
* {c_rain} — капли по стеклу и струи дождя, не меньше {s_rain}.

Геометрия (координаты — в долях кадра, камера в (0, 0, 4) смотрит в (0, 0, -1)):
* `night` — глухая ночь за стеклом (z = -45), `floor` — асфальт от
  {floor_line:.0%} высоты кадра вниз (z = -8), `canopy` — навес сверху
  (0..{glass_v:.0%} высоты, z = -5), всё цвета {c_night};
* `glass` — стеклянная панель {glass_w:.0%} x {glass_h:.0%} кадра, верх на
  {glass_v:.0%} высоты, z = -5, opacity 0.32;
* `post` — две стойки навеса (0.02 ширины кадра) по краям стекла, цвет {c_night};
* `board` — голорасписание u = {board_u}, v = {board_v}, 0.22x0.14 кадра, цвет
  {c_holo}: по нему бегут строки (пять полос, фаза по времени) и мигает заголовок;
* `lamp` — тёплая лампа под навесом u = {lamp_u}, v = {lamp_v}, 0.13x0.06 кадра,
  цвет {c_lamp}, opacity 0.9, плюс тёплый ореол 0.20x0.10 кадра;
* `neon` — малиновая вывеска за стеклом: 0.30x0.06 кадра на u = {neon_u}, v = {neon_v};
* `bench` — скамья {c_night} под стеклом (0.30x0.05 кадра, v = 0.66);
* `puddle` — три лужи с отражением (0.18x0.04 кадра, v = 0.90, цвет {c_neon}, opacity 0.35);
* `drops` — {drops} капель {c_rain} перед стеклом (z = -4.9), размер 6 px, падают
  со скоростью {speed} единиц/с;
* `rain` — струи дождя за стеклом (z = -12), те же точки, на треть медленнее.

Анимация:
* `window.__step(dt)` двигает капли, фазу строк расписания, чуть качает камеру
  (x на {sway}) и возвращает `window.__probe()`;
* `window.__probe()` возвращает `t`, `objects` и `board_phase` — фазу строк (4 знака).

{lcg}
"""

L24_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>neon bus stop</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + COMMON_JS + """

const SPEED = {speed};
const SWAY = {sway};

place(quad(1.8, 1.8, vgrad('0x241f38', '0x120f1c'), 'night', -60), 0.5, 0.5, -60);
quad(1.0, {glass_v}, solid(0x150c2a), 'canopy', -5);
place(quad(1.0, {below}, vgrad('0x241749', '0x100a1e'), 'floor', -8), 0.5, {below_v}, -8);
place(quad(0.02, {post_h}, solid(0x150c2a), 'post_a', -5), {post_a}, 0.60, -5);
place(quad(0.02, {post_h}, solid(0x150c2a), 'post_b', -5), {post_b}, 0.60, -5);
place(quad({glass_w}, {glass_h}, haze(0x9fc4d8, 0.20), 'glass', -5), 0.5, {glass_mid}, -5);

const boardMat = new THREE.ShaderMaterial({{
  uniforms: {{uPhase: {{value: 0.0}}}},
  vertexShader: `varying vec2 vUv; void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  fragmentShader: `uniform float uPhase; varying vec2 vUv;
    void main() {{
      float head = step(0.78, vUv.y);
      float rows = step(0.55, fract(vUv.y * 6.0 - uPhase));
      float tail = step(fract(vUv.x * 1.5), 0.92);
      vec3 base = mix(vec3(0.02, 0.22, 0.24), vec3(0.13, 0.83, 0.77), rows * (1.0 - head));
      gl_FragColor = vec4(mix(base, vec3(0.75, 1.0, 0.95), head * 0.8) * (0.7 + 0.3 * tail), 1.0);
    }}`,
}});
place(quad(0.22, 0.14, boardMat, 'board', -4.6), {board_u}, {board_v}, -4.6);
place(quad(0.24, 0.16, solid(0x120a26), 'board_frame', -4.8), {board_u}, {board_v}, -4.8);
place(quad(0.13, 0.06, haze({c_lamp}, 0.9), 'lamp', -4.5), {lamp_u}, {lamp_v}, -4.5);
place(quad(0.20, 0.10, haze({c_lamp}, 0.35), 'lamp_halo', -4.4), {lamp_u}, {lamp_v}, -4.4);
place(quad(0.30, 0.06, haze({c_neon}, 0.7), 'neon', -13), {neon_u}, {neon_v}, -13);
place(quad(0.30, 0.05, solid(0x1a1030), 'bench', -4.2), 0.5, 0.66, -4.2);
for (let i = 0; i < 3; i++) place(quad(0.18, 0.04, haze({c_neon}, 0.35), 'puddle' + i, -7), 0.24 + i * 0.26, 0.90, -7);

const drops = spray({drops}, 6, {c_rain});
const dropSpan = FH(0.70, 8.9), dropWide = FW(0.72, 8.9);
{{
  const p = drops.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * dropWide, {glass_mid_units} + (rnd() - 0.5) * dropSpan, -4.9);
  p.needsUpdate = true;
}}
const rain = spray(34, 4, {c_rain});
const rainSpan = FH(0.50, 16), rainWide = FW(0.80, 16);
{{
  const p = rain.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, (rnd() - 0.5) * rainWide, {glass_mid_units} + (rnd() - 0.5) * rainSpan, -12);
  p.needsUpdate = true;
}}

let t = 0;
const fizz = [];
for (let i = 0; i < 64; i++) fizz.push(0.6 + rnd() * 0.8);
const fall = (cloud, low, high, rate, dt) => {{
  const p = cloud.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {{
    let y = p.getY(i) - rate * fizz[i % 64] * dt;
    if (y < low) y += (high - low);
    p.setY(i, y);
  }}
  p.needsUpdate = true;
}};

window.__step = (dt) => {{
  t += dt;
  const fall2 = (cloud, low, high, rate) => {{
    const p = cloud.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {{
      let y = p.getY(i) - rate * fizz[i % 64] * dt;
      if (y < low) y += (high - low);
      p.setY(i, y);
    }}
    p.needsUpdate = true;
  }};
  fall2(drops, -dropSpan / 2, dropSpan / 2, SPEED);
  fall2(rain, -rainSpan / 2, rainSpan / 2, SPEED * 0.66);
  boardMat.uniforms.uPhase.value = t * 0.5;
  camera.position.x = Math.sin(t * 0.22) * SWAY;
  camera.lookAt(camera.position.x, 0, -1);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: names(),
  board_phase: round(boardMat.uniforms.uPhase.value),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Сборка заданий
# --------------------------------------------------------------------------- #


def _l17(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    plant_u = round(rng.uniform(0.12, 0.20), 2)
    desk_u = round(rng.uniform(0.66, 0.76), 2)
    lamp_u = round(rng.uniform(0.24, 0.34), 2)
    drift = round(rng.uniform(1.1, 1.7), 2)
    prompt = L17_PROMPT.format(steps=STEPS['L-17'],
        glass_v=L17["glass_v"],
        glass_w=L17["glass_w"],
        glass_bottom=L17["glass_v"] + L17["glass_h"],
        sill_v=L17["sill_v"],
        plant_h=0.20,
        c_room="#2e1636", c_lamp="#ffa83c", c_plant="#7bbf6a", c_city="#7fe3e0", c_rain="#a8c8dc",
        s_room=_share_text(L17_PALETTE["#2e1636"]),
        s_lamp=_share_text(L17_PALETTE["#ffa83c"]),
        s_plant=_share_text(L17_PALETTE["#7bbf6a"]),
        s_city=_share_text(L17_PALETTE["#7fe3e0"]),
        s_rain=_share_text(L17_PALETTE["#a8c8dc"]),
        plant_u=f"{plant_u:.2f}", desk_u=f"{desk_u:.2f}", desk_v="0.72", lamp_u=f"{lamp_u:.2f}", lamp_v="0.06",
        windows=26, drops_out=64, rain_speed="2.4", drops=L17["drops"], speed=f"{L17['speed']:.2f}",
        mon_w=0.14, glass_z=f"{L17['glass_z']:.1f}", cam_min=f"{L17['glass_z'] + 2.4:.1f}", drift=f"{drift:.2f}",
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L17_GOLDEN.format(
        desk_v="0.72",
        seed=seed,
        glass_z=f"{L17['glass_z']:.1f}", glass_w=L17["glass_w"], glass_h=L17["glass_h"],
        glass_mid=round(L17["glass_v"] + L17["glass_h"] / 2, 3),
        city_top=0.14, city_dark="0x170c20",
        windows=26, drops_out=64, drops=L17["drops"],
        rain_speed="2.4", speed=f"{L17['speed']:.2f}", drift=f"{drift:.2f}",
        c_room="0x2e1636", c_room_hi="0x41205a", c_room_deep="0x1a0c22", c_frame="0x120a1a",
        c_lamp="0xffa83c", c_plant="0x7bbf6a",
        c_city="0x7fe3e0", c_city_dark="0x170c20", c_rain="0xa8c8dc",
        plant_u=f"{plant_u:.2f}", desk_u=f"{desk_u:.2f}",
        lamp_u=f"{lamp_u:.2f}", lamp_v="0.06", sill_v=f"{L17['sill_v']:.2f}",
    )
    for literal in (
        f"{plant_u:.2f}", f"{desk_u:.2f}", f"{lamp_u:.2f}", f"{drift:.2f}",
        str(seed), "#2e1636", "#ffa83c", "#7bbf6a", "#7fe3e0", "#a8c8dc",
        _share_text(L17_PALETTE["#2e1636"]), _share_text(L17_PALETTE["#a8c8dc"]),
    ):
        assert literal in prompt, literal
    # Габарит окна внутри кадра: 0.78 ширины и 0.11..0.45 высоты.
    assert L17["glass_w"] <= 0.9 and 0.05 <= L17["glass_v"] < L17["glass_v"] + L17["glass_h"] <= 0.95
    assert L17["glass_w"] * 2 * _half_at(CAM_Z - L17["glass_z"]) * ASPECT <= _fw(0.98, CAM_Z - L17["glass_z"])
    assert 0.15 <= L17["glass_w"] * L17["glass_h"] <= 0.80
    # Наезд камеры обязан заметно изменить кадр.
    assert drift * STEPS["L-17"] / 60.0 > 1.0
    return art.build_task(
        task_id="L-17",
        title=L17_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("комната", L17_PALETTE, {
                "#2e1636": "стены и пол комнаты",
                "#ffa83c": "лампа и подоконник",
                "#7bbf6a": "растение",
                "#7fe3e0": "окна города и монитор",
                "#a8c8dc": "капли и дождь",
            }),
            composition=[
                _window_brighter("окно светлее пола", 4, 6),
                _warm_towards("тёплое у подоконника", 0.20, 0.80),
                _cool_upper("холодный город в полосе окна", 0.60),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.20),
                visual.image_min("distinct_colors", 20),
                visual.rich_edges(0.012, "комната и город не заливка"),
                visual.image_min("saturation", 0.25),
            ],
            motion=STEPS["L-17"],
        ),
        golden=golden,
        fixture={"seed": seed, "plant_u": plant_u, "desk_u": desk_u, "lamp_u": lamp_u, "drift": drift},
    )


def _l18(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    sign_u = round(rng.uniform(0.10, 0.16), 2)
    board_u = round(rng.uniform(0.30, 0.40), 2)
    drift = round(rng.uniform(0.04, 0.07), 3)
    steam = round(rng.uniform(0.5, 0.8), 2)
    prompt = L18_PROMPT.format(steps=STEPS['L-18'],
        sign_u=f"{sign_u:.2f}", sign_v=f"{L18['sign_v']:.2f}", sign_w=L18["sign_w"], sign_h=L18["sign_h"],
        board_u=f"{board_u:.2f}", board_v=f"{L18['board_v']:.2f}", board_w=L18["board_w"],
        holo_u="0.14", holo_w=0.08, holo_h=0.14,
        c_night="#1e1145", c_sign="#12c7d4", c_lamp="#ffb347", c_holo="#ff5fa8", c_rain="#a8c8dc",
        s_night=_share_text(L18_PALETTE["#1e1145"]),
        s_sign=_share_text(L18_PALETTE["#12c7d4"]),
        s_lamp=_share_text(L18_PALETTE["#ffb347"]),
        s_holo=_share_text(L18_PALETTE["#ff5fa8"]),
        s_rain=_share_text(L18_PALETTE["#a8c8dc"]),
        drops=L18["drops"], speed=f"{L18['speed']:.2f}", steam=f"{steam:.2f}", drift=f"{drift:.3f}",
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L18_GOLDEN.format(
        seed=seed,
        c_night="0x1e1145", c_wall="0x2f1642", c_ground="0x1b0c28", c_roof="0x160c22",
        c_sign="0x12c7d4", c_lamp="0xffb347", c_holo="0xff5fa8", c_rain="0xa8c8dc",
        holo_rgb="0.95, 0.24, 0.50",
        board_w=L18["board_w"], board_u=f"{board_u:.2f}", board_v=f"{L18['board_v']:.2f}",
        sign_w=L18["sign_w"], sign_h=L18["sign_h"], sign_u=f"{sign_u:.2f}", sign_v=f"{L18['sign_v']:.2f}",
        holo_w="0.08", holo_h="0.14", holo_u="0.14", glass_mid_units=0.0,
        drops=L18["drops"], speed=f"{L18['speed']:.2f}", drift=f"{drift:.3f}", steam=f"{steam:.2f}",
    )
    for literal in (
        f"{sign_u:.2f}", f"{board_u:.2f}", f"{drift:.3f}", f"{steam:.2f}", str(seed),
        "#1e1145", "#12c7d4", "#ffb347", "#ff5fa8", "#a8c8dc",
    ):
        assert literal in prompt, literal
    # Вывеска и голопанели внутри кадра, не липнут к краям.
    assert 0.05 <= sign_u - L18["sign_w"] / 2 and sign_u + L18["sign_w"] / 2 <= 0.95
    assert L18["sign_h"] / 2 <= L18["sign_v"] <= 0.60
    assert 0.08 <= board_u <= 0.60
    assert L18["speed"] * STEPS["L-18"] / 60.0 > 1.5, "капли обязаны успеть проехать кадр"
    return art.build_task(
        task_id="L-18",
        title=L18_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("лапшичная", L18_PALETTE, {
                "#1e1145": "ночь и асфальт",
                "#12c7d4": "циановая вывеска",
                "#ffb347": "лампы и пар",
                "#ff5fa8": "голопанели",
                "#a8c8dc": "капли и дождь",
            }),
            composition=[
                _ground_darker("асфальт темнее стены", 2.0),
                _warm_upper("вывеска и лампы сверху", 0.60),
                _cool_lower("мокрый асфальт и дождь снизу", 0.30),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.18),
                visual.image_min("distinct_colors", 20),
                visual.rich_edges(0.009, "киоск, пар и дождь дают детали"),
                visual.image_min("saturation", 0.30),
            ],
            motion=STEPS["L-18"],
        ),
        golden=golden,
        fixture={"seed": seed, "sign_u": sign_u, "board_u": board_u, "drift": drift, "steam": steam},
    )


def _l19(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    mul_a = round(rng.uniform(0.30, 0.36), 2)
    mul_b = round(rng.uniform(0.64, 0.70), 2)
    sway = round(rng.uniform(0.02, 0.05), 3)
    speed = round(rng.uniform(1.5, 2.0), 2)
    prompt = L19_PROMPT.format(steps=STEPS['L-19'],
        glass_v=L19["glass_v"], glass_h=L19["glass_h"], glass_w=L19["glass_w"],
        glass_bottom=L19["glass_v"] + L19["glass_h"], cool_top=0.20, glow_v=0.17, glow_low=0.66,
        c_night="#1a1230", c_neon="#12c7d4", c_warm="#ffa63c", c_rain="#a8c8dc",
        s_night=_share_text(L19_PALETTE["#1a1230"]),
        s_neon=_share_text(L19_PALETTE["#12c7d4"]),
        s_warm=_share_text(L19_PALETTE["#ffa63c"]),
        s_rain=_share_text(L19_PALETTE["#a8c8dc"]),
        blobs=L19["blobs"], drops=L19["drops"], speed=f"{speed:.2f}", sway=f"{sway:.3f}",
        mul_a=f"{mul_a:.2f}", mul_b=f"{mul_b:.2f}",
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L19_GOLDEN.format(
        seed=seed,
        glass_v=L19["glass_v"], glass_h=L19["glass_h"], glass_w=L19["glass_w"],
        glass_mid=round(L19["glass_v"] + L19["glass_h"] / 2, 3),
        cabin_low=round(1 - L19["glass_v"] - L19["glass_h"], 3),
        cabin_low_v=round(L19["glass_v"] + L19["glass_h"] + (1 - L19["glass_v"] - L19["glass_h"]) / 2, 3),
        cabin_top_v=round(L19["glass_v"] / 2, 3),
        mul_a=f"{mul_a:.2f}", mul_b=f"{mul_b:.2f}", glow_v="0.17", glow_low="0.66",
        c_night="0x1a1230", c_neon="0x12c7d4", c_warm="0xffa63c", c_rain="0xa8c8dc",
        blobs=L19["blobs"], drops=L19["drops"], speed=f"{speed:.2f}", sway=f"{sway:.3f}",
    )
    for literal in (f"{mul_a:.2f}", f"{mul_b:.2f}", f"{sway:.3f}", f"{speed:.2f}", str(seed),
                    "#1a1230", "#12c7d4", "#ffa63c", "#a8c8dc"):
        assert literal in prompt, literal
    assert mul_a < mul_b
    assert L19["glass_h"] * 2 * _half_at(9) <= _fh(0.98, 9)  # стекло в кадре
    assert L19["speed"] * STEPS["L-19"] / 60.0 > 1.2, "капли обязаны успеть проехать кадр"
    return art.build_task(
        task_id="L-19",
        title=L19_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("трамвай", L19_PALETTE, {
                "#1a1230": "салон и ночь",
                "#12c7d4": "неон за окном",
                "#ffa63c": "тёплые блики салона",
                "#a8c8dc": "капли по стеклу",
            }),
            composition=[
                _window_brighter("окно светлее кромки салона", 3, 6),
                _warm_upper("тёплые блики сверху", 0.50),
                _cool_lower("холодное в полосе окна", 0.20),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.15),
                visual.image_min("distinct_colors", 16),
                visual.rich_edges(0.009, "капли и город дают детали"),
                visual.image_min("saturation", 0.25),
            ],
            motion=STEPS["L-19"],
        ),
        golden=golden,
        fixture={"seed": seed, "mullion_a": mul_a, "mullion_b": mul_b, "sway": sway, "speed": speed},
    )


def _l20(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    dome_z = round(rng.uniform(-15.0, -12.5), 1)
    fly = round(rng.uniform(2.2, 3.0), 2)
    lamp_u = round(rng.uniform(0.28, 0.34), 2)
    plants = rng.choice([4, 5, 6])
    prompt = L20_PROMPT.format(steps=STEPS['L-20'],
        dome_z=f"{dome_z:.1f}", pan_w=L20["pan_w"], pan_back=round(L20["pan_w"] * 2.1, 2), dome_h=L20["dome_h"], dome_line=0.62,
        lamp_v=L20["lamp_v"], lamps=5, plants=plants,
        warm_top=0.25, warm_bottom=0.80,
        c_dome="#123038", c_leaf="#7fd054", c_lamp="#ffc06a", c_rain="#a8c8dc",
        s_dome=_share_text(L20_PALETTE["#123038"]),
        s_leaf=_share_text(L20_PALETTE["#7fd054"]),
        s_lamp=_share_text(L20_PALETTE["#ffc06a"]),
        s_rain=_share_text(L20_PALETTE["#a8c8dc"]),
        meteors=L20["meteors"], drops=L20["drops"], speed=f"{L20['speed']:.2f}", fly=f"{fly:.2f}",
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L20_GOLDEN.format(
        seed=seed, dome_z=f"{dome_z:.1f}", pan_w=L20["pan_w"], dome_h=L20["dome_h"],
        lamp_v=L20["lamp_v"], lamps=5, plants=plants,
        c_dome="0x123038", c_leaf="0x7fd054", c_lamp="0xffc06a", c_rain="0xa8c8dc",
        meteors=L20["meteors"], drops=L20["drops"], speed=f"{L20['speed']:.2f}", fly=f"{fly:.2f}",
    )
    for literal in (f"{dome_z:.1f}", f"{fly:.2f}", str(seed), "#123038", "#7fd054", "#ffc06a", "#a8c8dc"):
        assert literal in prompt, literal
    assert CAM_Z - dome_z > 6.0, "купол не должен упираться в камеру"
    assert L20["pan_w"] * 2 + 0.30 <= 1.05, "три панели купола шире кадра не нужны"
    assert fly * STEPS["L-20"] / 60.0 > 2.0, "пролёт обязан быть виден"
    return art.build_task(
        task_id="L-20",
        title=L20_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("оранжерея", L20_PALETTE, {
                "#123038": "купол и ночь",
                "#7fd054": "листва растений",
                "#ffc06a": "лампы под куполом",
                "#a8c8dc": "метеоры и капли",
            }),
            composition=[
                _dome_lighter("купол светлее пола", 4.0),
                _warm_band("лампы и листва в средней полосе", 0.30, 0.85),
                _cool_upper("метеоры за куполом сверху", 0.65),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.15),
                visual.image_min("distinct_colors", 12),
                visual.rich_edges(0.012, "листва и рёбра купола дают детали"),
                visual.image_min("saturation", 0.22),
            ],
            motion=STEPS["L-20"],
        ),
        golden=golden,
        fixture={"seed": seed, "dome_z": dome_z, "fly": fly, "lamp_u": lamp_u, "plants": plants},
    )


def _l21(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    flash_u = round(rng.uniform(0.20, 0.34), 2)
    flash_v = round(rng.uniform(0.26, 0.36), 2)
    blink = round(rng.uniform(0.9, 1.4), 2)
    drift = round(rng.uniform(0.03, 0.06), 3)
    prompt = L21_PROMPT.format(steps=STEPS['L-21'],
        panel_v=L21["panel_v"], panel_w=0.84, panel_h=0.14, panel_u="0.50",
        panel_from=0.22, panel_to=0.42, floor_line=0.52,
        screen_u="0.20", screen_u2="0.80",
        glass_v=L21["glass_v"], glass_h=L21["glass_h"],
        c_hull="#2a1c4a", c_screen="#57e08a", c_amber="#ffd166", c_rain="#a8c8dc",
        s_hull=_share_text(L21_PALETTE["#2a1c4a"]),
        s_screen=_share_text(L21_PALETTE["#57e08a"]),
        s_amber=_share_text(L21_PALETTE["#ffd166"]),
        s_rain=_share_text(L21_PALETTE["#a8c8dc"]),
        dials=5, toggles=5, drops=L21["drops"], drops_out=44, speed=f"{L21['speed']:.2f}",
        blink=f"{blink:.2f}", drift=f"{drift:.3f}", flash_u=f"{flash_u:.2f}", flash_v=f"{flash_v:.2f}",
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L21_GOLDEN.format(
        seed=seed,
        glass_v=L21["glass_v"], glass_h=L21["glass_h"], glass_mid=round(L21["glass_v"] + L21["glass_h"] / 2, 3),
        below=round(1 - L21["glass_v"] - L21["glass_h"], 3),
        canopy_v=round(L21["glass_v"] / 2, 3), sill_v=round(L21["glass_v"] + L21["glass_h"] + 0.03, 3),
        below_v=round(L21["glass_v"] + L21["glass_h"] + (1 - L21["glass_v"] - L21["glass_h"]) / 2, 3),
        panel_w=0.84, panel_h=0.14, panel_u="0.50", panel_v=f"{L21['panel_v']:.2f}",
        floor_v=round(L21["panel_v"] + 0.22, 3), console_v=round(L21["panel_v"] + 0.17, 3), seat_v=round(L21["panel_v"] + 0.24, 3),
        glass_mid_units=round((0.5 - L21["glass_v"] - L21["glass_h"] / 2) * 2 * _half_at(9), 3),
        screen_b="0.72",
        c_hull="0x2a1c4a", c_screen="0x57e08a", c_amber="0xffd166", c_rain="0xa8c8dc",
        dials=5, toggles=5, drops=L21["drops"], speed=f"{L21['speed']:.2f}",
        blink=f"{blink:.2f}", drift=f"{drift:.3f}", flash_u=f"{flash_u:.2f}", flash_v=f"{flash_v:.2f}",
    )
    for literal in (f"{blink:.2f}", f"{drift:.3f}", f"{flash_u:.2f}", f"{flash_v:.2f}", "30%", str(seed),
                    "#2a1c4a", "#57e08a", "#ffd166", "#a8c8dc"):
        assert literal in prompt, literal
    assert 0.05 <= flash_u <= 0.95 and 0.05 <= flash_v <= 0.95
    assert L21["speed"] * STEPS["L-21"] / 60.0 > 1.5, "капли обязаны успеть проехать кадр"
    return art.build_task(
        task_id="L-21",
        title=L21_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("рубка", L21_PALETTE, {
                "#2a1c4a": "корпус и панель",
                "#57e08a": "экраны приборов",
                "#ffd166": "индикаторы и тумблеры",
                "#a8c8dc": "дождь за стеклом",
            }),
            composition=[
                _panel_bright("панель приборов сверху", 5.0, 60.0),
                _warm_band("приборы в полосе панели", 0.20, 0.55),
                _cool_upper("дождь в полосе стекла", 0.60),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.15),
                visual.image_min("distinct_colors", 16),
                visual.rich_edges(0.012, "тумблеры и капли дают детали"),
                visual.image_min("saturation", 0.22),
            ],
            motion=STEPS["L-21"],
        ),
        golden=golden,
        fixture={"seed": seed, "flash_u": flash_u, "flash_v": flash_v, "blink": blink, "drift": drift},
    )


def _l22(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    sign_u = round(rng.uniform(0.26, 0.34), 2)
    robot_u = round(rng.uniform(0.38, 0.46), 2)
    approach = round(rng.uniform(1.0, 1.3), 2)
    speed = round(rng.uniform(1.8, 2.3), 2)
    prompt = L22_PROMPT.format(steps=STEPS['L-22'],
        sign_u=f"{sign_u:.2f}", sign_v=f"{L22['sign_v']:.2f}", sign_w=L22["sign_w"], sign_h=L22["sign_h"], sign_line=0.25,
        robot_u=f"{robot_u:.2f}", robot_v=f"{L22['robot_v']:.2f}", robot_w=L22["robot_w"], floor_line=0.40,
        c_night="#1b1a2a", c_ground="#14121f", c_sign="#ff2f7a", c_spark="#ffd94a", c_rain="#a8c8dc",
        s_night=_share_text(L22_PALETTE["#1b1a2a"]),
        s_sign=_share_text(L22_PALETTE["#ff2f7a"]),
        s_spark=_share_text(L22_PALETTE["#ffd94a"]),
        s_rain=_share_text(L22_PALETTE["#a8c8dc"]),
        drops=L22["drops"], speed=f"{speed:.2f}", approach=f"{approach:.2f}", sparks=L22["sparks"],
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L22_GOLDEN.format(
        seed=seed,
        sign_u=f"{sign_u:.2f}", sign_v=f"{L22['sign_v']:.2f}", sign_w=L22["sign_w"], sign_h=L22["sign_h"],
        robot_u=f"{robot_u:.2f}", robot_v=f"{L22['robot_v']:.2f}", robot_w=L22["robot_w"],
        c_night="0x1b1a2a", c_ground="0x14121f", c_sign="0xff2f7a", c_spark="0xffd94a", c_rain="0xa8c8dc",
        drops=L22["drops"], speed=f"{speed:.2f}", approach=f"{approach:.2f}", sparks=L22["sparks"],
    )
    for literal in (f"{sign_u:.2f}", f"{robot_u:.2f}", f"{approach:.2f}", f"{speed:.2f}", "40%", str(seed),
                    "#1b1a2a", "#ff2f7a", "#ffd94a", "#a8c8dc"):
        assert literal in prompt, literal
    assert 0.05 <= sign_u - L22["sign_w"] / 2 and sign_u + L22["sign_w"] / 2 <= 0.95
    assert 0.05 <= robot_u - L22["robot_w"] / 2 and robot_u + L22["robot_w"] / 2 <= 0.95
    assert L22["robot_w"] * 2 * _half_at(9.4) * ASPECT <= _fw(0.60, 9.4)  # робот внутри кадра
    assert approach * STEPS["L-22"] / 60.0 * 6 > 1.0, "приближение обязано быть видно"
    return art.build_task(
        task_id="L-22",
        title=L22_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("робот", L22_PALETTE, {
                "#1b1a2a": "асфальт и ночь",
                "#ff2f7a": "вывеска и её отражение",
                "#ffd94a": "искры и лампа робота",
                "#a8c8dc": "дождь и лужи",
            }),
            composition=[
                _sign_high("вывеска сверху в верхней четверти кадра", "#ff2f7a", 0.10),
                _ground_darker("низ кадра темнее верха", 3.0),
                _cool_lower("дождь и лужи снизу", 0.08),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.15),
                visual.image_min("distinct_colors", 16),
                visual.rich_edges(0.009, "робот и капли дают детали"),
                visual.image_min("saturation", 0.22),
            ],
            motion=STEPS["L-22"],
        ),
        golden=golden,
        fixture={"seed": seed, "sign_u": sign_u, "robot_u": robot_u, "approach": approach, "speed": speed},
    )


def _l23(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    tower_u = round(rng.uniform(0.18, 0.26), 2)
    rover_u = round(rng.uniform(0.46, 0.56), 2)
    orbit = round(rng.uniform(1.4, 1.9), 2)
    fog = round(rng.uniform(0.18, 0.28), 2)
    prompt = L23_PROMPT.format(steps=STEPS['L-23'],
        tower_u=f"{tower_u:.2f}", tower_v=f"{L23['tower_v']:.2f}", rover_u=f"{rover_u:.2f}", rover_v="0.55",
        pad_v="0.62", holo_v="0.28", warm_top=0.30, warm_bottom=0.75, cool_share=0.03,
        c_hull="#241644", c_neon="#35d5e6", c_amber="#ffb347", c_rain="#a8c8dc",
        s_hull=_share_text(L23_PALETTE["#241644"]),
        s_neon=_share_text(L23_PALETTE["#35d5e6"]),
        s_amber=_share_text(L23_PALETTE["#ffb347"]),
        s_rain=_share_text(L23_PALETTE["#a8c8dc"]),
        drops=L23["drops"], speed=f"{L23['speed']:.2f}", orbit=f"{orbit:.2f}", fog=f"{fog:.2f}",
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L23_GOLDEN.format(
        seed=seed,
        tower_u=f"{tower_u:.2f}", tower_v=f"{L23['tower_v']:.2f}", rover_u=f"{rover_u:.2f}",
        rover_v="0.55", pad_v="0.62", holo_v="0.28",
        c_hull="0x241644", c_neon="0x35d5e6", c_amber="0xffb347", c_rain="0xa8c8dc",
        drops=L23["drops"], speed=f"{L23['speed']:.2f}", orbit=f"{orbit:.2f}", fog=f"{fog:.2f}",
    )
    for literal in (f"{tower_u:.2f}", f"{rover_u:.2f}", f"{orbit:.2f}", f"{fog:.2f}", str(seed),
                    "#241644", "#35d5e6", "#ffb347", "#a8c8dc"):
        assert literal in prompt, literal
    assert 0.05 <= tower_u <= 0.4 and 0.2 <= rover_u <= 0.8
    assert orbit * STEPS["L-23"] / 60.0 > 2.0, "облёт обязан быть виден"
    return art.build_task(
        task_id="L-23",
        title=L23_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("заправка", L23_PALETTE, {
                "#241644": "станция и ночь",
                "#35d5e6": "голограммы и платформа",
                "#ffb347": "огни ровера",
                "#a8c8dc": "дождь и туман",
            }),
            composition=[
                _pad_lighter("платформа светлее ночи", 4.0),
                _warm_band("ровер с огнями в средней полосе", 0.25, 0.85),
                _cool_share("туман и голограммы широко", 0.02),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.15),
                visual.image_min("distinct_colors", 16),
                visual.rich_edges(0.01, "станция и туман не заливка"),
                visual.image_min("saturation", 0.22),
            ],
            motion=STEPS["L-23"],
        ),
        golden=golden,
        fixture={"seed": seed, "tower_u": tower_u, "rover_u": rover_u, "orbit": orbit, "fog": fog},
    )


def _l24(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    board_u = round(rng.uniform(0.26, 0.34), 2)
    lamp_u = round(rng.uniform(0.60, 0.70), 2)
    neon_u = round(rng.uniform(0.58, 0.70), 2)
    sway = round(rng.uniform(0.015, 0.035), 3)
    prompt = L24_PROMPT.format(steps=STEPS['L-24'],
        glass_v=L24["glass_v"], glass_h=L24["glass_h"], glass_w=L24["glass_w"],
        glass_bottom=L24["glass_v"] + L24["glass_h"], cool_top=0.30, floor_line=0.62,
        board_u=f"{board_u:.2f}", board_v=f"{L24['board_v']:.2f}", lamp_u=f"{lamp_u:.2f}", lamp_v=f"{L24['lamp_v']:.2f}",
        neon_u=f"{neon_u:.2f}", neon_v="0.24",
        c_night="#1c1236", c_holo="#22d3c5", c_lamp="#ffa83c", c_neon="#ff4d9d", c_rain="#a8c8dc",
        s_night=_share_text(L24_PALETTE["#1c1236"]),
        s_holo=_share_text(L24_PALETTE["#22d3c5"]),
        s_lamp=_share_text(L24_PALETTE["#ffa83c"]),
        s_neon=_share_text(L24_PALETTE["#ff4d9d"]),
        s_rain=_share_text(L24_PALETTE["#a8c8dc"]),
        drops=L24["drops"], speed=f"{L24['speed']:.2f}", sway=f"{sway:.3f}",
        seed=seed, lcg=_lcg_text(seed),
    )
    golden = L24_GOLDEN.format(
        seed=seed,
        glass_v=L24["glass_v"], glass_h=L24["glass_h"], glass_w=L24["glass_w"],
        glass_mid=round(L24["glass_v"] + L24["glass_h"] / 2, 3),
        below=round(1 - L24["glass_v"] - L24["glass_h"], 3),
        below_v=round(L24["glass_v"] + L24["glass_h"] + (1 - L24["glass_v"] - L24["glass_h"]) / 2, 3),
        glass_mid_units=round((0.5 - L24["glass_v"] - L24["glass_h"] / 2) * 2 * _half_at(9), 3),
        post_a=round(0.5 - L24["glass_w"] / 2 - 0.02, 3), post_b=round(0.5 + L24["glass_w"] / 2 + 0.02, 3),
        post_h=round(L24["glass_h"] + 0.06, 3),
        board_u=f"{board_u:.2f}", board_v=f"{L24['board_v']:.2f}", lamp_u=f"{lamp_u:.2f}", lamp_v=f"{L24['lamp_v']:.2f}",
        neon_u=f"{neon_u:.2f}", neon_v="0.24",
        c_night="0x1c1236", c_holo="0x22d3c5", c_lamp="0xffa83c", c_neon="0xff4d9d", c_rain="0xa8c8dc",
        drops=L24["drops"], speed=f"{L24['speed']:.2f}", sway=f"{sway:.3f}",
    )
    for literal in (f"{board_u:.2f}", f"{lamp_u:.2f}", f"{neon_u:.2f}", f"{sway:.3f}", str(seed),
                    "#1c1236", "#22d3c5", "#ffa83c", "#ff4d9d", "#a8c8dc"):
        assert literal in prompt, literal
    assert L24["glass_w"] <= 0.9 and L24["glass_v"] + L24["glass_h"] <= 0.95
    assert L24["glass_w"] * 2 * _half_at(9) * ASPECT <= _fw(0.96, 9)
    assert L24["speed"] * STEPS["L-24"] / 60.0 > 1.5, "капли обязаны успеть проехать кадр"
    return art.build_task(
        task_id="L-24",
        title=L24_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks("остановка", L24_PALETTE, {
                "#1c1236": "ночь, навес и асфальт",
                "#22d3c5": "голорасписание",
                "#ffa83c": "тёплая лампа",
                "#ff4d9d": "неон и лужи",
                "#a8c8dc": "капли и дождь",
            }),
            composition=[
                _window_brighter("стекло с улицей светлее асфальта", 2, 6),
                _warm_upper("лампа и неон сверху", 0.65),
                _cool_lower("дождь и лужи ниже середины", 0.30),
            ],
            detail=[
                visual.image_min("changed_fraction", 0.15),
                visual.image_min("distinct_colors", 16),
                visual.rich_edges(0.012, "расписание и капли дают детали"),
                visual.image_min("saturation", 0.22),
            ],
            motion=STEPS["L-24"],
        ),
        golden=golden,
        fixture={"seed": seed, "board_u": board_u, "lamp_u": lamp_u, "neon_u": neon_u, "sway": sway},
    )


# --------------------------------------------------------------------------- #
# Общие предикаты композиции
# --------------------------------------------------------------------------- #


def _profile(pixels: dict) -> list[float]:
    return [float(entry["lum"]) for entry in (pixels.get("profile") or [])]


def _ground_darker(label: str, margin: float):
    """Низ кадра темнее верха: асфальт/пол против стены и неба."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        top = sum(profile[1:4]) / 3
        low = sum(profile[5:8]) / 3
        good = low + margin <= top
        return good, f"кадр: {label} — низ {low:.1f} против верха {top:.1f} (нужно >= {margin})"

    return visual._label(check, label)


def _window_brighter(label: str, margin: float, low_band: int):
    """Полоса окна светлее нижней кромки кадра."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        window = sum(profile[1:4]) / 3
        bottom = profile[low_band]
        good = window >= bottom + margin
        return good, f"кадр: {label} — окно {window:.1f} против низа {bottom:.1f} (нужно >= {margin})"

    return visual._label(check, label)


def _dome_lighter(label: str, margin: float):
    """Купол/платформа светлее пола и дальнего фона."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        dome = sum(profile[2:5]) / 3
        bottom = sum(profile[6:8]) / 2
        good = dome >= bottom + margin
        return good, f"кадр: {label} — купол {dome:.1f} против низа {bottom:.1f} (нужно >= {margin})"

    return visual._label(check, label)


def _pad_lighter(label: str, margin: float):
    """Светящаяся платформа/купол в середине кадра ярче дальнего фона."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        mid = max(profile[3], profile[4])
        top = sum(profile[0:2]) / 2
        good = mid >= top + margin
        return good, f"кадр: {label} — середина {mid:.1f} против фона {top:.1f} (нужно >= {margin})"

    return visual._label(check, label)


def _panel_bright(label: str, margin: float, floor_cap: float):
    """Панель приборов светлее средней яркости, а низ кадра тёмный."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        mean = float(pixels.get("mean") or 0.0)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        panel = sum(profile[0:2]) / 2
        good = panel >= mean + margin and profile[7] <= floor_cap
        return good, (
            f"кадр: {label} — панель {panel:.1f} против средней {mean:.1f} (нужно >= {margin}), "
            f"низ {profile[7]:.1f} (нужно <= {floor_cap})"
        )

    return visual._label(check, label)


def _class_centroid(name: str, axis: int, low: float, high: float, label: str):
    def check(pixels: dict) -> tuple[bool, str]:
        stats = pixels.get(name) or {}
        centroid = stats.get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 2:
            return False, f"кадр: {label} — пикселей класса нет"
        value = float(centroid[axis])
        good = low <= value <= high
        return good, f"кадр: {label} — центр по {'xy'[axis]} {value:.3f} (нужно {low}..{high})"

    return visual._label(check, label)


def _warm_upper(label: str, cap: float):
    """Тёплые пиксели держатся верха кадра."""

    def check(pixels: dict) -> tuple[bool, str]:
        warm = pixels.get("warm") or {}
        fraction = float(warm.get("fraction") or 0.0)
        centroid = warm.get("centroid")
        if not isinstance(centroid, list):
            return False, f"кадр: {label} — тёплых пикселей нет"
        good = fraction >= 0.002 and float(centroid[1]) <= cap
        return good, f"кадр: {label} — доля {fraction:.4f}, центр y {float(centroid[1]):.3f} (нужно <= {cap})"

    return visual._label(check, label)


def _warm_towards(label: str, low: float, high: float):
    return _class_centroid("warm", 1, low, high, label)


def _warm_band(label: str, low: float, high: float):
    return _class_centroid("warm", 1, low, high, label)


def _cool_lower(label: str, floor: float):
    """Холодные пиксели лежат не выше объявленной границы."""

    def check(pixels: dict) -> tuple[bool, str]:
        cool = pixels.get("cool") or {}
        fraction = float(cool.get("fraction") or 0.0)
        centroid = cool.get("centroid")
        if not isinstance(centroid, list):
            return False, f"кадр: {label} — холодных пикселей нет"
        good = fraction >= 0.005 and float(centroid[1]) >= floor
        return good, f"кадр: {label} — доля {fraction:.4f}, центр y {float(centroid[1]):.3f} (нужно >= {floor})"

    return visual._label(check, label)


def _cool_upper(label: str, cap: float):
    def check(pixels: dict) -> tuple[bool, str]:
        cool = pixels.get("cool") or {}
        fraction = float(cool.get("fraction") or 0.0)
        centroid = cool.get("centroid")
        if not isinstance(centroid, list):
            return False, f"кадр: {label} — холодных пикселей нет"
        good = fraction >= 0.005 and float(centroid[1]) <= cap
        return good, f"кадр: {label} — доля {fraction:.4f}, центр y {float(centroid[1]):.3f} (нужно <= {cap})"

    return visual._label(check, label)


def _cool_share(label: str, low: float):
    def check(pixels: dict) -> tuple[bool, str]:
        cool = pixels.get("cool") or {}
        fraction = float(cool.get("fraction") or 0.0)
        good = fraction >= low
        return good, f"кадр: {label} — холодных {fraction:.4f} (нужно >= {low})"

    return visual._label(check, label)


def _hex(value: str) -> str:
    """Цвет приводится к '#rrggbb': в промпте он печатается так, в JS пишется 0x."""
    return value if value.startswith("#") else "#" + value[2:]


def _sign_high(label: str, colour: str = "#ff2f7a", low: float = 0.10):
    """Пиксели вывески — не меньше объявленной доли кадра."""

    want = _hex(colour)

    def check(pixels: dict) -> tuple[bool, str]:
        histogram = pixels.get("hue_histogram") or []
        if not histogram:
            return False, "кадр: нет гистограммы"
        share = 0.0
        for hue, sat, val, fraction in histogram:
            if visual.hsv_error(want, hue, sat, val) <= 0.22:
                share += float(fraction)
        good = share >= low
        return good, f"кадр: {label} — {want} занимает {share:.4f} (нужно >= {low})"

    return visual._label(check, label)


def build(rng: random.Random) -> list[core.TaskInstance]:
    return [_l17(rng), _l18(rng), _l19(rng), _l20(rng), _l21(rng), _l22(rng), _l23(rng), _l24(rng)]
