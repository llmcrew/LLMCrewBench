"""Art scenarios L-10..L-12 — шейдеры и материалы.

Три сцены, где картинка делается фрагментным/вершинным шейдером, а не набором
геометрии:

* L-10 — фрактал Мандельброта на весь кадр: квадрат под орто-камерой, итерации
  ``z = z*z + c`` во фрагментном шейдере, раскраска по логарифму итераций;
  движение — рост зума в объявленную точку;
* L-11 — жидкий металл и стекло: ``MeshPhysicalMaterial`` с бликами от
  градиентной ``DataTexture`` (создана в JS, без внешних файлов), тёмный фон;
  движение — вращение объектов и сдвиг блика;
* L-12 — бесконечный туннель: кольца в перспективе, глубина на тумане в
  шейдере, цветовые полосы по кольцам; движение — кольца летят на камеру.

Палитра объявляется долями кадра и проверяется строго по тону, композиция —
широкими порогами, как и в остальном блоке. Числа фикстуры печатаются в промпт
дословно и перепроверяются ``assert``-ами в :func:`build`.
"""

from __future__ import annotations

import math
import random

from .. import art, core, visual

IDS = ("L-10", "L-11", "L-12")

FRACTAL_TITLE = "L-10 — фрактал Мандельброта (фрагментный шейдер)"
METAL_TITLE = "L-11 — жидкий металл и стекло (MeshPhysicalMaterial)"
TUNNEL_TITLE = "L-12 — бесконечный туннель (туман в шейдере)"

# --------------------------------------------------------------------------- #
# L-10 — фрактал
# --------------------------------------------------------------------------- #

#: Палитра: цвет -> минимальная доля кадра. Полосы по t = log(iter+1)/log(161):
#: внешние (тёмно-синий, бирюзовый) и приграничные (салатовый), тело множества
#: (розовый) — четыре цвета, различимые по тону. Пороги — половина измеренного.
L10_PALETTE = {
    "#1b1240": 0.06,  # внешние полосы, мало итераций (измерено 0.12)
    "#18b7ac": 0.10,  # средние полосы (измерено 0.21)
    "#b8f24a": 0.01,  # полосы у самой границы множества (измерено 0.019)
    "#ff5f7e": 0.21,  # тело множества, t = 1 (измерено 0.43)
}
L10_CENTRE = (-0.600, 0.000)  # центр области — точно на вещественной оси
L10_ITER = 160  # максимум итераций
L10_ESCAPE = 2.0  # радиус выхода
L10_HALF = 1.2  # полувысота области при зуме 1.0
L10_ZOOMS = (1.2, 1.35, 1.5)
L10_RATES = (0.25, 0.35, 0.45)
L10_STEPS = 48

# --------------------------------------------------------------------------- #
# L-11 — металл и стекло
# --------------------------------------------------------------------------- #

L11_PALETTE = {
    "#0a0714": 0.37,  # тёмный фон и пол (измерено 0.75)
    "#33d1c6": 0.05,  # металлическая сфера (измерено 0.11)
    "#e0733c": 0.03,  # стеклянный тор (измерено 0.06)
    "#f2f6fb": 0.34,  # белые блики (измерено 0.70)
}
L11_SPHERE_R = 0.8  # радиус сферы
L11_ROUGHNESS = (0.12, 0.18, 0.24)
L11_CLEARCOAT = 0.6
L11_METAL_SPIN = (0.8, 1.1, 1.4)  # рад/с
L11_LIGHT_R = 3.2  # радиус орбиты блика
L11_LIGHT_SPIN = (0.6, 0.9, 1.2)  # рад/с
L11_CAMERA = (0.0, 0.6, 6.0)
L11_TARGET = (0.0, 0.40, 0.0)
L11_SPHERE = (0.0, 0.40, 0.60)
L11_TORUS = (-1.45, -0.25, -0.6)
L11_STEPS = 48

# --------------------------------------------------------------------------- #
# L-12 — туннель
# --------------------------------------------------------------------------- #

L12_PALETTE = {
    "#0a0612": 0.05,  # туман и фон (измерено 0.10)
    "#33d1c6": 0.15,  # бирюзовые полосы колец (измерено 0.30)
    "#e0733c": 0.14,  # оранжевые полосы колец (измерено 0.30)
    "#eaf2ff": 0.008,  # светлое пятно в конце туннеля (измерено 0.017)
}
L12_COUNT = 34  # колец
L12_PITCH = 2.3  # шаг между кольцами
L12_Z_FAR = -72.0  # самое дальнее кольцо
L12_RING_R = 2.0  # радиус кольца
L12_TUBE = 0.34  # радиус трубки кольца
L12_WOBBLE = 0.16  # разброс радиусов колец
L12_WOBBLE_FREQ = 0.7
L12_BANDS = 4  # полос по кольцу
L12_BAND_PHASE = 0.4  # скорость бега полос
L12_RING_PHASE = 0.4  # сдвиг фазы на кольцо
L12_DENSITY = (0.030, 0.038, 0.046)  # плотность тумана
L12_SPEEDS = (7.0, 9.0, 11.0)  # скорость полёта колец
L12_CORE_R = 1.2  # радиус светлого пятна
L12_CORE_SCALE = 5.0  # растяжка пятна в кадре
L12_CORE_Z = -42.0
L12_CAMERA_Z = 8.0
L12_FOV = 55.0
L12_STEPS = 48

PROMPT_FRACTAL = """Собери фрактал Мандельброта на весь кадр: один квадрат 2x2 под
`OrthographicCamera(-1, 1, 1, -1, 0, 1)`, вся картинка считается во фрагментном
шейдере `ShaderMaterial` — итерации `z = z*z + c`, раскраска по числу итераций.
Свет и текстуры не нужны.

Область и итерации (числа обязаны стоять в шейдере дословно):
* центр области — c = {centre}, ровно на вещественной оси;
* стартовый зум `uZoom` = {zoom}, полувысота кадра-области {half} / uZoom, ширина
  в 4/3 раза больше (кадр 1920x1080), центр области — в середине кадра;
* максимум итераций — ровно {iter}, точка считается ушедшей при |z| > {escape}
  (радиус выхода {escape});
* доля t = log(итераций + 1) / log({iter_plus}); точки внутри множества дают
  t = 1.0.

Раскраска (цвета — из палитры ниже, переходы `smoothstep` между полосами):
* t = 0 → {c_outer}; переход к {c_mid} в t = 0.30..0.40;
* переход к тёмному {c_outer} обратно в t = 0.55..0.62;
* переход к {c_rim} в t = 0.70..0.76; переход к {c_body} в t = 0.90..0.96;
* всё, что внутри множества (t = 1.0), — заливка {c_body}.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_outer} — внешние тёмные полосы, не меньше {s_outer:.0%} кадра;
* {c_mid} — средние полосы, не меньше {s_mid:.0%};
* {c_rim} — узкие полосы у самой границы, не меньше {s_rim:.0%};
* {c_body} — тело множества, не меньше {s_body:.0%}. Сумма долей палитры меньше
  1.0: оставшееся — плавные переходы между полосами.

Композиция (проверяется по кадру грубо):
* центр области — в середине кадра: тело множества ложится в центр, внешние
  тёмные полосы уходят к краям, центр кадра светлее углов;
* множество симметрично относительно вещественной оси: центр области стоит на
  ней, поэтому верх и низ кадра зеркальны (профиль яркости сверху вниз
  повторяется в обратном порядке);
* полосы идут по всему кадру — плотность границ высокая, однотонной заливки нет.

Анимация:
* `window.__step(dt)`: время t растёт на dt, `uZoom = {zoom} * exp({rate} * t)` —
  непрерывный зум в центр области, кадр перерисовывается;
* `window.__probe()` возвращает `t`, `objects` и `zoom` (4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, симметрия и
плотность границ, и то, что сцена движется — кадры на t и t+{steps} шагов
различаются (зум успевает заметно вырасти), а каждая загрузка даёт тот же кадр.
"""

GOLDEN_FRACTAL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>mandelbrot</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const CENTRE = new THREE.Vector2({cx}, {cy});
const ZOOM = {zoom}, RATE = {rate}, HALF = {half}, ASPECT = 800 / 600;
const MAX_ITER = {iter};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const material = new THREE.ShaderMaterial({{
  uniforms: {{
    uZoom: {{value: ZOOM}},
    uCentre: {{value: CENTRE}},
    uPal0: {{value: new THREE.Color({c_outer})}},
    uPal1: {{value: new THREE.Color({c_mid})}},
    uPal2: {{value: new THREE.Color({c_rim})}},
    uPal3: {{value: new THREE.Color({c_body})}},
  }},
  vertexShader: `varying vec2 vUv;
    void main() {{ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }}`,
  fragmentShader: `precision highp float;
    uniform float uZoom; uniform vec2 uCentre;
    uniform vec3 uPal0; uniform vec3 uPal1; uniform vec3 uPal2; uniform vec3 uPal3;
    varying vec2 vUv;
    const int MAX_ITER = {iter};
    const float ESCAPE = {escape};
    const float HALF = {half};
    void main() {{
      // Полувысота области кадра — HALF / uZoom, ширина в 4/3 раза больше.
      vec2 step = vec2(2.0 * (HALF / uZoom) * (800.0 / 600.0), 2.0 * (HALF / uZoom));
      vec2 c = uCentre + (vUv - 0.5) * step;
      vec2 z = vec2(0.0);
      int iter = MAX_ITER;
      for (int i = 0; i < MAX_ITER; i++) {{
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (dot(z, z) > ESCAPE * ESCAPE) {{ iter = i; break; }}
      }}
      // t = log(итераций + 1) / log(161); внутри множества t = 1.0.
      float t = (iter >= MAX_ITER) ? 1.0 : log(float(iter) + 1.0) / log(161.0);
      vec3 colour = uPal0;
      colour = mix(colour, uPal1, smoothstep(0.30, 0.40, t));
      colour = mix(colour, uPal0, smoothstep(0.55, 0.62, t));
      colour = mix(colour, uPal2, smoothstep(0.70, 0.76, t));
      colour = mix(colour, uPal3, smoothstep(0.90, 0.96, t));
      gl_FragColor = vec4(colour, 1.0);
    }}`,
}});
const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
quad.name = 'fractal';
scene.add(quad);

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;

window.__step = (dt) => {{
  t += dt;
  material.uniforms.uZoom.value = ZOOM * Math.exp(RATE * t);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  zoom: round(material.uniforms.uZoom.value),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""

PROMPT_METAL = """Собери сцену «жидкий металл и стекло»: металлическая сфера и
стеклянный тор на тёмном фоне, блики — от градиентной карты окружения,
нарисованной в JS (`THREE.DataTexture` 8x64, без внешних файлов) и от трёх
источников света.

Композиция (проверяется по кадру грубо):
* сфера — в центре кадра, её блики читаются как светлое пятно около центра;
* фон и углы кадра тёмные, центр занят сферой и её отражением в полу;
* контур сферы и тора читается: границы в кадре есть, заливки нет.

Геометрия и материал (числа — в сцену дословно):
* сфера `SphereGeometry({radius}, 48, 32)` в точке (0, 0.40, 0.60), имя `sphere`,
  материал `MeshPhysicalMaterial`: цвет {c_metal}, metalness 1.0,
  roughness {roughness}, clearcoat {clearcoat}, `envMapIntensity` 3.0;
* тор `TorusGeometry(0.80, 0.28, 16, 64)`, имя `torus`, материал
  `MeshPhysicalMaterial`: цвет {c_glass}, metalness 0.15, roughness 0.06,
  clearcoat 1.0 — стекло, стоит слева и ближе к камере: (-1.45, -0.25, -0.6);
* пол — плоскость 40x40 в y = -1.20, цвет {c_floor}, metalness 0.85,
  roughness 0.30;
* фон — плоскость 40x40 за сценой в z = -6, тот же градиент, что у карты
  окружения, снизу {c_bg} вверх до {c_mid_sky}.

Карта окружения (`DataTexture` 8x64, colourSpace = SRGBColorSpace, mapping —
EquirectangularReflectionMapping, `needsUpdate = true`, положена в
`scene.environment`): вертикальный градиент снизу вверх {c_bg} → {c_night} →
{c_sky} → {c_top}, где {c_night} — глубокая ночь: низ тёмный, верх светлый,
только у самого верха небо светлеет до {c_top}, поэтому у металла две узкие
яркие полосы-блика, а тело сферы остаётся {c_metal} в полутени.

Свет:
* `DirectionalLight(0xffffff, 4.0)` из точки (2.0, 4.0, 3.0);
* острая белая искра: `PointLight({c_highlight}, 120, 25)` в точке (0.6, 2.6, 2.2)
  — она даёт на металле крошечное яркое пятно {c_highlight};
* два `PointLight` (цвета {c_metal} и {c_glass}, intensity 45, distance 25) —
  они едут по орбите радиуса {light_r} в плоскости xz со скоростью {light_spin}
  от начальной фазы {phase}: `x = {light_r} * cos({light_spin} * t + {phase})`,
  `z = {light_r} * sin({light_spin} * t + {phase}) + 2.0`; стартовая точка
  (2.0, 4.0, 3.0) у `DirectionalLight` — фиксированная;
* слабый `AmbientLight(0x223044, 0.3)`, чтобы тёмные части не проваливались;
* туман сцены `THREE.Fog({c_bg}, 14, 45)` — дальний пол и фон гаснут в темноте.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — тёмный фон и пол, не меньше {s_bg:.0%} кадра;
* {c_metal} — металлическая сфера, не меньше {s_metal:.0%};
* {c_glass} — стеклянный тор, не меньше {s_glass:.0%};
* {c_highlight} — белые блики на металле и стекле, не меньше {s_highlight:.0%}.

Анимация:
* `window.__step(dt)`: сфера вращается `sphere.rotation.y += {spin} * dt` и
  `sphere.rotation.x += 0.4 * dt`, тор — `torus.rotation.y += 0.7 * dt`,
  источники света едут по своей орбите (сдвиг блика), кадр перерисовывается;
* `window.__probe()` возвращает `t`, `objects`, `angle` (поворот сферы) и
  `light` — позицию блика (4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, тёмные края и
светлый центр, детализация (блики и контуры) и то, что сцена движется — кадры
на t и t+{steps} шагов различаются.
"""

GOLDEN_METAL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>liquid metal</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const SPIN = {spin}, LIGHT_R = {light_r}, LIGHT_SPIN = {light_spin}, PHASE = {phase};
const C_BG = new THREE.Color({c_bg}), C_METAL = new THREE.Color({c_metal});
const C_GLASS = new THREE.Color({c_glass}), C_TOP = new THREE.Color({c_top});
const C_SKY = new THREE.Color({c_sky}), C_NIGHT = new THREE.Color({c_night});

const renderer = new THREE.WebGLRenderer({{antialias: true, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(C_BG, 14, 45);
const camera = new THREE.PerspectiveCamera(45, 800 / 600, 0.1, 200);
camera.position.set({cam_x}, {cam_y}, {cam_z});
camera.lookAt({look_x}, {look_y}, {look_z});

// Карта окружения: градиент, нарисованный в JS, без единого внешнего файла.
const W = 8, H = 64;
const data = new Uint8Array(W * H * 4);
const mix = (a, b, f) => a.clone().lerp(b, Math.min(1, Math.max(0, f)));
for (let y = 0; y < H; y++) {{
  const f = y / (H - 1);
  // Низ — глубокая ночь: тело металла остаётся своим цветом, ярким делают
  // только узкую верхнюю полосу неба.
  const c = f > 0.88 ? mix(C_SKY, C_TOP, (f - 0.88) / 0.12)
    : f > 0.50 ? mix(C_NIGHT, C_SKY, (f - 0.50) / 0.38)
    : mix(C_NIGHT, C_METAL, (f / 0.50) * 0.7);
  for (let x = 0; x < W; x++) {{
    const i = (y * W + x) * 4;
    data[i] = c.r * 255; data[i + 1] = c.g * 255; data[i + 2] = c.b * 255; data[i + 3] = 255;
  }}
}}
const env = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
env.mapping = THREE.EquirectangularReflectionMapping;
env.colorSpace = THREE.SRGBColorSpace;
env.needsUpdate = true;
scene.environment = env;

const background = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.ShaderMaterial({{
    uniforms: {{ uTop: {{value: C_SKY}}, uBottom: {{value: C_BG}} }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; varying vec2 vUv;
      void main() {{
        // Фон темнеет вниз: только узкая полоса неба наверху светлая.
        gl_FragColor = vec4(mix(uBottom, uTop, pow(vUv.y, 3.0)), 1.0);
      }}`,
  }})
);
background.name = 'backdrop';
background.position.set(0, 0, -6);
scene.add(background);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshPhysicalMaterial({{color: {c_floor}, metalness: 0.85, roughness: 0.30}})
);
floor.name = 'floor';
floor.rotation.x = -Math.PI / 2;
floor.position.y = -1.20;
scene.add(floor);

const sphere = new THREE.Mesh(
  new THREE.SphereGeometry({radius}, 48, 32),
  new THREE.MeshPhysicalMaterial({{
    color: C_METAL, metalness: 1.0, roughness: {roughness}, clearcoat: {clearcoat},
    clearcoatRoughness: 0.05, envMapIntensity: 3.0,
  }})
);
sphere.name = 'sphere';
sphere.position.set({sph_x}, {sph_y}, {sph_z});
scene.add(sphere);

const torus = new THREE.Mesh(
  new THREE.TorusGeometry(0.80, 0.28, 16, 64),
  new THREE.MeshPhysicalMaterial({{
    color: C_GLASS, metalness: 0.15, roughness: 0.06, clearcoat: 1.0,
    clearcoatRoughness: 0.03, envMapIntensity: 2.2,
  }})
);
torus.name = 'torus';
torus.position.set({tor_x}, {tor_y}, {tor_z});
scene.add(torus);

scene.add(new THREE.AmbientLight(0x223044, 0.3));
const key = new THREE.DirectionalLight(0xffffff, 4.0);
key.position.set(2.0, 4.0, 3.0);
scene.add(key);
const spark = new THREE.PointLight({c_highlight}, 120, 25);
spark.position.set(0.6, 2.6, 2.2);
scene.add(spark);
const lightA = new THREE.PointLight(C_METAL, 45, 25);
const lightB = new THREE.PointLight(C_GLASS, 45, 25);
lightA.position.set(LIGHT_R, 1.8, 2.0);
lightB.position.set(-LIGHT_R, 0.2, 2.0);
scene.add(lightA);
scene.add(lightB);

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;

window.__step = (dt) => {{
  t += dt;
  // Поворот объекта и облёт блика: кадр обязан сдвинуться.
  sphere.rotation.y += SPIN * dt;
  sphere.rotation.x += 0.4 * dt;
  torus.rotation.y += 0.7 * dt;
  lightA.position.x = LIGHT_R * Math.cos(LIGHT_SPIN * t + PHASE);
  lightA.position.z = LIGHT_R * Math.sin(LIGHT_SPIN * t + PHASE) + 2.0;
  lightB.position.x = -LIGHT_R * Math.cos(LIGHT_SPIN * t + PHASE);
  lightB.position.z = -LIGHT_R * Math.sin(LIGHT_SPIN * t + PHASE) + 2.0;
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  angle: round(sphere.rotation.y),
  light: [round(lightA.position.x), round(lightA.position.y), round(lightA.position.z)],
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""

PROMPT_TUNNEL = """Собери бесконечный туннель: кольца в перспективе летят на камеру,
глубина — на тумане, полосы по кольцам бегут. Камера стоит внутри туннеля и
смотрит вдаль.

Геометрия и движение (числа — в сцену дословно):
* кольцо — `TorusGeometry({ring_r}, {tube}, 10, 48)`, в сцене {count} колец;
* кольца стоят вдоль z с шагом {pitch} от z = {z_far} до z = {z_far} + {count} * {pitch},
  имя кольца `ring<i>`, радиус кольца i домножается на
  `1 + {wobble} * sin(i * {wobble_freq})` (лёгкий разброс, туннель живой);
* камера `PerspectiveCamera(55, 800/600, 0.1, 200)` в точке (0, 0, {cam_z}) и
  смотрит в (0, 0, -1) — внутрь туннеля;
* конец туннеля закрыт светлым пятном: `MeshBasicMaterial` цвета {c_core} с
  `fog: false`, `CircleGeometry({core_r}, 48)` в (0, 0, {core_z}) со `scale` {core_scale} —
  свет в конце туннеля, пятно около центра кадра;
* кольцо при подходе к камере гаснет в туман: его `scale` растёт
  `max(1, (4 - z) / 12)`, поэтому вплывающее кольцо успевает уйти в фон.

Заворот и туман (в шейдере кольца):
* z кольца i: `{z_far} + ((i * {pitch} + t * {speed}) mod ({count} * {pitch}))` —
  кольца летят на камеру со скоростью {speed} и заворачиваются по z;
* туман — в фрагментном шейдере по глубине: `vDepth` = -z во view-пространстве
  (`modelViewMatrix`), `fog = 1 - exp(-{density} * vDepth)`, цвет тумана {c_fog};
* ближние кольца притухают: множитель `smoothstep(0.4, 4.0, vDepth)` — кольцо
  вплывает из темноты, поэтому края кадра тёмные, а середина светлее;
* по кольцу {bands} цветовых полос: `sin(vUv.x * 3.14159 * {bands} + uPhase)`
  (период 2 полосы на кольцо, как у настоящей решётки), цвета полос
  {c_teal} и {c_orange} попеременно; фаза бежит: `uPhase = {band_phase} * t + {ring_phase} * i`;
* дальние кольца подсвечиваются светом из конца туннеля: `far = smoothstep(40, 68, vDepth)`,
  `colour = mix(colour, {c_core}, far * 0.30)` (умеренно, иначе дальние кольца
  сливаются со светлым пятном и фон перестаёт читаться);
* `scene.background` = {c_fog}; в сцене также `THREE.Fog({c_fog}, 8, 90)` — общий
  туман сцены и цвет фона за кольцами.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_fog} — туман и фон туннеля, не меньше {s_fog:.0%} кадра;
* {c_teal} — бирюзовые полосы колец, не меньше {s_teal:.0%};
* {c_orange} — оранжевые полосы колец, не меньше {s_orange:.0%};
* {c_core} — светлое пятно в конце туннеля, не меньше {s_core:.0%}.

Композиция (проверяется по кадру грубо):
* середина кадра ярче краёв: светлое пятно в центре и притухание ближних колец;
* кольца симметричны относительно центра кадра (кадр зеркален по вертикальной
  оси — кольца это окружности);
* кадр не плоско: кольца дают плотные границы, а полосы — насыщенный цвет;
* палитра: {c_fog} — цвет тумана, сама доля мала, потому что кольца закрывают
  кадр; это цвет затухания, к которому уходит глубина.

Анимация:
* `window.__step(dt)`: время t растёт на dt, кольца едут по завороту (z
  пересчитывается по формуле выше), фаза полос бежит, кадр перерисовывается;
* `window.__probe()` возвращает `t`, `objects`, `speed` и `z` — z первых трёх
  колец (4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, светлый центр,
симметрия, детализация и то, что сцена движется — кадры на t и t+{steps} шагов
различаются (кольца успевают пролететь несколько шагов решётки).
"""

GOLDEN_TUNNEL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>tunnel</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const COUNT = {count}, PITCH = {pitch}, SPAN = COUNT * PITCH, Z_FAR = {z_far};
const SPEED = {speed}, DENSITY = {density}, WOBBLE = {wobble}, WOBBLE_FREQ = {wobble_freq};
const BAND_PHASE = {band_phase}, RING_PHASE = {ring_phase};
const C_FOG = new THREE.Color({c_fog}), C_TEAL = new THREE.Color({c_teal});
const C_ORANGE = new THREE.Color({c_orange}), C_CORE = new THREE.Color({c_core});
const PAIRS = [[C_TEAL, C_ORANGE], [C_ORANGE, C_TEAL], [C_TEAL, C_ORANGE]];

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = C_FOG;
scene.fog = new THREE.Fog(C_FOG, 8, 90);
const camera = new THREE.PerspectiveCamera({fov}, 800 / 600, 0.1, 200);
camera.position.set(0, 0, {cam_z});
camera.lookAt(0, 0, -1);

const geometry = new THREE.TorusGeometry({ring_r}, {tube}, 10, 48);
const rings = [];
for (let i = 0; i < COUNT; i++) {{
  const pair = PAIRS[i % 3];
  const material = new THREE.ShaderMaterial({{
    uniforms: {{
      uA: {{value: pair[0]}}, uB: {{value: pair[1]}},
      uFog: {{value: C_FOG}}, uCore: {{value: C_CORE}},
      uPhase: {{value: 0.0}}, uDensity: {{value: DENSITY}},
    }},
    vertexShader: `varying vec2 vUv; varying float vDepth;
      void main() {{
        vUv = uv;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }}`,
    fragmentShader: `uniform vec3 uA; uniform vec3 uB; uniform vec3 uFog; uniform vec3 uCore;
      uniform float uPhase; uniform float uDensity;
      varying vec2 vUv; varying float vDepth;
      void main() {{
        // 8 полос по кольцу (период — 2 полосы), фаза бежит; туман по глубине.
        float band = 0.5 + 0.5 * sin(vUv.x * 3.14159 * float({bands}) + uPhase);
        vec3 colour = mix(uA, uB, band);
        colour *= 0.25 + 0.75 * smoothstep(0.4, 4.0, vDepth);
        // Дальние кольца подсвечиваются светом из конца туннеля.
        float far = smoothstep(40.0, 68.0, vDepth);
        colour = mix(colour, uCore, far * 0.30);
        float fog = 1.0 - exp(-uDensity * vDepth);
        gl_FragColor = vec4(mix(colour, uFog, clamp(fog, 0.0, 1.0)), 1.0);
      }}`,
  }});
  const ring = new THREE.Mesh(geometry, material);
  ring.name = 'ring' + i;
  ring.scale.setScalar(1.0 + WOBBLE * Math.sin(i * WOBBLE_FREQ));
  ring.position.z = Z_FAR + i * PITCH;
  scene.add(ring);
  rings.push(ring);
}}

const core = new THREE.Mesh(
  new THREE.CircleGeometry({core_r}, 48),
  new THREE.MeshBasicMaterial({{color: {c_core}, fog: false}})
);
core.name = 'core';
core.position.set(0, 0, {core_z});
core.scale.setScalar({core_scale});
scene.add(core);

const round = (v) => Math.round(v * 1e4) / 1e4;
const place = (ring, i, time) => {{
  const depth = Z_FAR + ((i * PITCH + time * SPEED) % SPAN);
  ring.position.z = depth;
  // Перспектива: по мере приближения кольцо раскрывается на весь кадр,
  // поэтому оно гаснет, а между ним и камерой остаётся фон-туман.
  ring.scale.setScalar((1.0 + WOBBLE * Math.sin(i * WOBBLE_FREQ)) * Math.max(1.0, (4.0 - depth) / 12.0));
  ring.material.uniforms.uPhase.value = BAND_PHASE * time + RING_PHASE * i;
}};

let t = 0;
window.__step = (dt) => {{
  t += dt;
  for (let i = 0; i < rings.length; i++) place(rings[i], i, t);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  speed: SPEED,
  z: rings.slice(0, 3).map((ring) => round(ring.position.z)),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _frame_share(hex_value: str, share: float, label: str):
    return visual.palette_colour(hex_value, share, label)


# --------------------------------------------------------------------------- #
# Предикаты по кадру
# --------------------------------------------------------------------------- #


def _updown_mirror(max_difference: float, label: str):
    """Верх и низ кадра совпадают: множество симметрично вещественной оси."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = [entry["lum"] for entry in (pixels.get("profile") or [])]
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        pairs = [(0, 7), (1, 6), (2, 5), (3, 4)]
        diffs = [round(abs(profile[a] - profile[b]), 1) for a, b in pairs]
        good = max(diffs) <= max_difference
        return good, f"кадр: {label} — расхождения полос {diffs} (нужно <= {max_difference})"

    return visual._label(check, f"кадр: {label} (расхождение <= {max_difference})")


def _dark_corners(low: float, label: str):
    """Углы кадра тёмные: внешние полосы/фон лежат по краям."""

    def check(pixels: dict) -> tuple[bool, str]:
        regions = pixels.get("regions") or []
        if len(regions) < 9:
            return False, "кадр: нет областей"
        corners = [regions[0], regions[2], regions[6], regions[8]]
        sums = [round(sum(c) / 3, 1) for c in corners]
        worst = max(sums)
        good = worst <= low
        return good, f"кадр: {label} — яркость углов {sums} (нужно <= {low})"

    # Порог берётся вдвое выше измеренного максимума: проверка на максимум, а не
    # на среднее, поэтому запас вверх, а не вниз.

    return visual._label(check, f"кадр: {label} (углы <= {low})")


def _bright_centre(factor: float, label: str):
    """Центр кадра ярче углов: свет в конце туннеля и притухание ближних колец."""

    def check(pixels: dict) -> tuple[bool, str]:
        regions = pixels.get("regions") or []
        if len(regions) < 9:
            return False, "кадр: нет областей"
        centre = sum(regions[4]) / 3
        corners = [sum(regions[i]) / 3 for i in (0, 2, 6, 8)]
        mean = sum(corners) / 4
        ratio = centre / max(1.0, mean)
        good = ratio >= factor
        return good, f"кадр: {label} — центр {centre:.1f} против углов {mean:.1f} (отношение {ratio:.2f})"

    return visual._label(check, f"кадр: {label} (центр/углы >= {factor})")


# --------------------------------------------------------------------------- #
# Сценарии
# --------------------------------------------------------------------------- #


def _fractal(rng: random.Random) -> core.TaskInstance:
    zoom = rng.choice(L10_ZOOMS)
    rate = rng.choice(L10_RATES)

    prompt = PROMPT_FRACTAL.format(
        zoom=f"{zoom:.3f}",
        rate=f"{rate:.3f}",
        centre=f"({L10_CENTRE[0]:.3f}, {L10_CENTRE[1]:.3f})",
        half=L10_HALF,
        escape=L10_ESCAPE,
        iter=L10_ITER,
        iter_plus=L10_ITER + 1,
        steps=L10_STEPS,
        c_outer="#1b1240",
        c_mid="#18b7ac",
        c_rim="#b8f24a",
        c_body="#ff5f7e",
        s_outer=L10_PALETTE["#1b1240"],
        s_mid=L10_PALETTE["#18b7ac"],
        s_rim=L10_PALETTE["#b8f24a"],
        s_body=L10_PALETTE["#ff5f7e"],
    )
    golden = GOLDEN_FRACTAL.format(
        cx=f"{L10_CENTRE[0]:.3f}",
        cy=f"{L10_CENTRE[1]:.3f}",
        zoom=f"{zoom:.3f}",
        rate=f"{rate:.3f}",
        half=L10_HALF,
        escape=L10_ESCAPE,
        iter=L10_ITER,
        c_outer="0x1b1240",
        c_mid="0x18b7ac",
        c_rim="0xb8f24a",
        c_body="0xff5f7e",
    )
    for literal in (f"{zoom:.3f}", f"{rate:.3f}", f"{L10_CENTRE[0]:.3f}", f"{L10_CENTRE[1]:.3f}", "#1b1240", "#18b7ac", "#b8f24a", "#ff5f7e"):
        assert literal in prompt, literal
    assert str(L10_ITER) in prompt
    # Зум за steps шагов обязан заметно вырасти, иначе кадры не различатся.
    assert rate * L10_STEPS / 60.0 > 0.1

    return art.build_task(
        task_id="L-10",
        title=FRACTAL_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=[
                _frame_share("#1b1240", L10_PALETTE["#1b1240"], "внешние полосы"),
                _frame_share("#18b7ac", L10_PALETTE["#18b7ac"], "средние полосы"),
                _frame_share("#b8f24a", L10_PALETTE["#b8f24a"], "полосы у границы"),
                _frame_share("#ff5f7e", L10_PALETTE["#ff5f7e"], "тело множества"),
            ],
            composition=[
                _updown_mirror(10.0, "верх и низ зеркальны (ось симметрии множества)"),
                _bright_centre(1.15, "центр кадра (тело множества) светлее углов"),
            ],
            detail=[
                visual.rich_edges(0.04, "фрактал даёт плотные границы, а не заливку"),
                visual.image_min("distinct_colors", 15),
                visual.image_min("changed_fraction", 0.30),
                visual.image_min("saturation", 0.35),
            ],
            motion=L10_STEPS,
        ),
        golden=golden,
        fixture={"zoom": zoom, "rate": rate, "centre": list(L10_CENTRE), "iter": L10_ITER},
    )


def _metal(rng: random.Random) -> core.TaskInstance:
    roughness = rng.choice(L11_ROUGHNESS)
    spin = rng.choice(L11_METAL_SPIN)
    light_spin = rng.choice(L11_LIGHT_SPIN)
    phase = round(rng.uniform(0.0, math.tau), 3)

    prompt = PROMPT_METAL.format(
        radius=L11_SPHERE_R,
        roughness=f"{roughness:.2f}",
        clearcoat=L11_CLEARCOAT,
        light_r=L11_LIGHT_R,
        light_spin=f"{light_spin:.3f}",
        phase=f"{phase:.3f}",
        spin=f"{spin:.3f}",
        steps=L11_STEPS,
        c_bg="#0a0714",
        c_floor="#121a2e",
        c_sky="#1b4a5e",
        c_night="#0d1b2e",
        c_mid_sky="#4d7fa8",
        c_top="#8fd3ff",
        c_metal="#33d1c6",
        c_glass="#e0733c",
        c_highlight="#f2f6fb",
        s_bg=L11_PALETTE["#0a0714"],
        s_metal=L11_PALETTE["#33d1c6"],
        s_glass=L11_PALETTE["#e0733c"],
        s_highlight=L11_PALETTE["#f2f6fb"],
    )
    golden = GOLDEN_METAL.format(
        spin=f"{spin:.3f}",
        light_r=L11_LIGHT_R,
        light_spin=f"{light_spin:.3f}",
        phase=phase,
        radius=L11_SPHERE_R,
        roughness=f"{roughness:.2f}",
        clearcoat=L11_CLEARCOAT,
        sph_x=L11_SPHERE[0], sph_y=L11_SPHERE[1], sph_z=L11_SPHERE[2],
        tor_x=L11_TORUS[0], tor_y=L11_TORUS[1], tor_z=L11_TORUS[2],
        cam_x=L11_CAMERA[0], cam_y=L11_CAMERA[1], cam_z=L11_CAMERA[2],
        look_x=L11_TARGET[0], look_y=L11_TARGET[1], look_z=L11_TARGET[2],
        c_bg="0x0a0714",
        c_floor="0x121a2e",
        c_sky="0x1b4a5e",
        c_night="0x0d1b2e",
        c_top="0x8fd3ff",
        c_metal="0x33d1c6",
        c_glass="0xe0733c",
        c_highlight="0xf2f6fb",
    )
    for literal in (f"{spin:.3f}", f"{light_spin:.3f}", f"{phase:.3f}", "#0a0714", "#33d1c6", "#e0733c", "#f2f6fb"):
        assert literal in prompt, literal
    assert f"{roughness:.2f}" in prompt and f"{roughness:.2f}" in golden
    # Блик обязан сдвинуться, а сфера — повернуться: кадры различатся.
    assert spin * L11_STEPS / 60.0 > 0.5 and light_spin * L11_STEPS / 60.0 > 0.4

    return art.build_task(
        task_id="L-11",
        title=METAL_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=[
                _frame_share("#0a0714", L11_PALETTE["#0a0714"], "тёмный фон и пол"),
                _frame_share("#33d1c6", L11_PALETTE["#33d1c6"], "металлическая сфера"),
                _frame_share("#e0733c", L11_PALETTE["#e0733c"], "стеклянный тор"),
                _frame_share("#f2f6fb", L11_PALETTE["#f2f6fb"], "белые блики"),
            ],
            composition=[
                _dark_corners(70.0, "углы кадра притемнены (фон и пол, без светлых пятен)"),
                visual.image_color("dark", 0.25, "тёмная масса фона"),
                visual.image_centroid("dark", 1, 0.35, 1.0, "тёмная масса внизу кадра"),
                visual.image_centroid("bright", 0, 0.25, 0.75, "блики стоят по центру кадра"),
                visual.image_centroid("bright", 1, 0.05, 0.85, "блики не прижаты к краю"),
            ],
            detail=[
                visual.rich_edges(0.006, "контур сферы и тора читается"),
                visual.image_min("distinct_colors", 60),
                visual.image_min("saturation", 0.30),
                visual.image_min("changed_fraction", 0.15),
            ],
            motion=L11_STEPS,
        ),
        golden=golden,
        fixture={"roughness": roughness, "spin": spin, "light_spin": light_spin, "phase": phase},
    )


def _tunnel(rng: random.Random) -> core.TaskInstance:
    density = rng.choice(L12_DENSITY)
    speed = rng.choice(L12_SPEEDS)
    span = L12_COUNT * L12_PITCH

    prompt = PROMPT_TUNNEL.format(
        count=L12_COUNT,
        pitch=L12_PITCH,
        z_far=L12_Z_FAR,
        fov=L12_FOV,
        ring_r=L12_RING_R,
        tube=L12_TUBE,
        wobble=L12_WOBBLE,
        wobble_freq=L12_WOBBLE_FREQ,
        cam_z=L12_CAMERA_Z,
        core_r=L12_CORE_R,
        core_z=L12_CORE_Z,
        core_scale=L12_CORE_SCALE,
        speed=f"{speed:.3f}",
        density=f"{density:.3f}",
        bands=L12_BANDS,
        band_phase=L12_BAND_PHASE,
        ring_phase=L12_RING_PHASE,
        steps=L12_STEPS,
        c_fog="#0a0612",
        c_teal="#33d1c6",
        c_orange="#e0733c",
        c_core="#eaf2ff",
        s_fog=L12_PALETTE["#0a0612"],
        s_teal=L12_PALETTE["#33d1c6"],
        s_orange=L12_PALETTE["#e0733c"],
        s_core=L12_PALETTE["#eaf2ff"],
    )
    golden = GOLDEN_TUNNEL.format(
        count=L12_COUNT,
        pitch=L12_PITCH,
        z_far=L12_Z_FAR,
        fov=L12_FOV,
        ring_r=L12_RING_R,
        tube=L12_TUBE,
        wobble=L12_WOBBLE,
        wobble_freq=L12_WOBBLE_FREQ,
        cam_z=L12_CAMERA_Z,
        core_r=L12_CORE_R,
        core_z=L12_CORE_Z,
        core_scale=L12_CORE_SCALE,
        speed=f"{speed:.3f}",
        density=f"{density:.3f}",
        bands=L12_BANDS,
        band_phase=L12_BAND_PHASE,
        ring_phase=L12_RING_PHASE,
        c_fog="0x0a0612",
        c_teal="0x33d1c6",
        c_orange="0xe0733c",
        c_core="0xeaf2ff",
    )
    for literal in (f"{speed:.3f}", f"{density:.3f}", "#0a0612", "#33d1c6", "#e0733c", "#eaf2ff"):
        assert literal in prompt, literal
    assert str(L12_COUNT) in prompt and str(L12_PITCH) in prompt
    # За steps шагов кольца пролетают больше одного шага решётки.
    assert speed * L12_STEPS / 60.0 > L12_PITCH * 1.5
    assert span > 0

    return art.build_task(
        task_id="L-12",
        title=TUNNEL_TITLE,
        prompt=prompt,
        spec=art.ArtSpec(
            palette=[
                _frame_share("#0a0612", L12_PALETTE["#0a0612"], "туман и фон туннеля"),
                _frame_share("#33d1c6", L12_PALETTE["#33d1c6"], "бирюзовые полосы колец"),
                _frame_share("#e0733c", L12_PALETTE["#e0733c"], "оранжевые полосы колец"),
                _frame_share("#eaf2ff", L12_PALETTE["#eaf2ff"], "свет в конце туннеля"),
            ],
            composition=[
                _bright_centre(1.10, "середина кадра ярче краёв"),
                visual.mirror(True, 30.0, "кольца симметричны относительно центра кадра"),
            ],
            detail=[
                visual.rich_edges(0.012, "кольца дают плотные границы"),
                visual.image_min("distinct_colors", 25),
                visual.image_min("changed_fraction", 0.35),
                visual.image_min("saturation", 0.25),
            ],
            motion=L12_STEPS,
        ),
        golden=golden,
        fixture={"density": density, "speed": speed, "count": L12_COUNT, "pitch": L12_PITCH},
    )


def build(rng: random.Random) -> list[core.TaskInstance]:
    return [_fractal(rng), _metal(rng), _tunnel(rng)]
