"""Art scenario L-2..L-5 — геометрия и архитектура (неон, мандала, город, мост).

Каждая сцена — одна страница Three.js и один фиксированный ракурс: проверка
меряет кадр, а не физику. Промпт объявляет палитру (точные цвета и их доли
кадра), композицию (горизонт, симметрия, профиль, створ) и движение; эталонные
страницы собраны так, чтобы объявленное было видно на кадре, а пороги проверок
стояли примерно вдвое ниже измеренного запаса.

Сцены:
* L-2 «неоновая сетка» — перспективная сетка бежит на камеру, полосатое солнце
  опускается к горизонту, небо градиентом от циана к фиолетовому;
* L-3 «мандала» — 12-лучевая мандала с поворотом на 15° за проверку;
* L-4 «ночной город» — ряд домов со светящимися окнами, их фаза мигает по
  индексу окна, дымка ходит вверх-вниз;
* L-5 «мост» — наклонная изометрия из примитивов с рёбрами `EdgesGeometry`,
  медленный поворот всей конструкции.

Три вещи, без которых сцена не читается на кадре (проверено на живом рендере):
объявленный хекс совпадает с кадром только если шейдер заканчивается
`#include <colorspace_fragment>` — иначе three отдаёт линейные значения и цвета
уезжают; у шейдерных плоскостей должна быть разбита геометрия и ближний край не
может стоять в плоскости камеры, иначе `varying` интерполируется вкривь; сетка
линиями `LineSegments` толщиной в пиксель не набирает объявленную долю — её
надо рисовать шейдерными полосами.
"""

from __future__ import annotations

import math
import random

from .. import art, core, visual

IDS = ("L-2", "L-3", "L-4", "L-5",)
TITLE = "L-2..L-5 — геометрия и архитектура"

FOV = 45.0
STEPS = 78  # кадров до второго снимка: движение обязано быть заметно


def _frame_height(distance: float) -> float:
    """Высота кадра в единицах сцены на данной дистанции (та же тригонометрия,
    что в эталоне L-1): из неё выводятся все размеры в долях кадра."""
    return 2 * math.tan(math.radians(FOV / 2)) * distance


# --------------------------------------------------------------------------- #
# Общие куски предикатов
# --------------------------------------------------------------------------- #


def _lums(pixels: dict) -> list[float]:
    return [float(entry.get("lum") or 0.0) for entry in (pixels.get("profile") or [])]


def _sats(pixels: dict) -> list[float]:
    return [float(entry.get("sat") or 0.0) for entry in (pixels.get("profile") or [])]


def _regions(pixels: dict) -> list[list[int]]:
    return [list(rgb) for rgb in (pixels.get("regions") or [])]


def _bright(rgb) -> float:
    return rgb[0] + rgb[1] + rgb[2]


def _share(hex_value: str, share: float, label: str):
    return visual.palette_colour(hex_value, share, label)


def _profile_check(label: str, body):
    """Предикат по восьми полосам профиля кадра."""

    def check(pixels: dict) -> tuple[bool, str]:
        profile = _lums(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        return body(profile)

    return visual._label(check, label)


# =========================================================================== #
# L-2 — неоновая сетка (synthwave)
# =========================================================================== #

L2_HORIZON = 0.55  # горизонт: доля высоты кадра
L2_SUN_R = 0.10  # радиус солнца: доля высоты кадра
L2_SUN_X = 0.50  # центр солнца по горизонтали: доля ширины кадра
L2_SPACING = 1.6  # шаг линий сетки, единицы сцены
L2_BACK = 30.0  # дистанция неба и солнца, единицы сцены
L2_PALETTE = {
    "#1b0f3c": 0.20,  # фиолетовое небо и земля
    "#ff2fb0": 0.010,  # пурпурные линии сетки
    "#19e0e8": 0.010,  # циановая полоса у горизонта
    "#ffb03a": 0.006,  # жёлто-оранжевый диск солнца
}

PROMPT_L2 = """Собери неоновую сетку в стиле synthwave: пурпурная перспективная сетка
бежит на камеру, над горизонтом — полосатое солнце, над ним яркая циановая полоса
и фиолетовая ночь.

Композиция (проверяется по кадру грубо):
* горизонт — ровно на {horizon:.0%} высоты кадра: выше — небо, ниже — земля;
* небо — тёмно-фиолетовое {c_bg} сверху, у самого горизонта — яркая циановая
  полоса {c_cyan}: яркость профиля монотонно растёт сверху вниз до горизонта;
* небо у горизонта заметно ярче земли под ним — горизонт обязан читаться полосой;
* солнце — диск радиусом {sun_r} высоты кадра, центр по горизонтали кадра и на
  {sun_y:.1%} высоты кадра (над горизонтом); в нижних двух третях диска вырезаны
  горизонтальные полосы, из них просвечивает небо;
* кадр симметричен относительно вертикальной оси (солнце по центру, сетка
  разложена от нуля в обе стороны).

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — фиолетовое небо и земля, не меньше {s_bg:.0%} кадра;
* {c_cyan} — циановая полоса над горизонтом, не меньше {s_cyan:.1%} кадра;
* {c_sun} — диск солнца, не меньше {s_sun:.1%} кадра;
* {c_line} — линии сетки, не меньше {s_line:.1%} кадра.

Сцена (камера в начале координат, смотрит в (0, 0, -1), угол 45, кадр 1920x1080;
полувысота кадра на дистанции {back:.0f} равна {half} единицы):
* небо — плоскость на z = -{back:.0f} высотой {sky_h} единицы, верхний край ровно
  на верхнем краю кадра, нижний — ровно на линии горизонта; материал
  `ShaderMaterial`, градиент от {c_bg} (верх) к {c_cyan} (низ) по
  `smoothstep(0.0, 0.32, vUv.y)`, без света;
* земля — плоскость от камеры на 120 единиц вглубь на высоте {ground} единицы
  (ровно под линией горизонта), материал `ShaderMaterial`: основа — градиент от
  {c_ground} у нижнего края к {c_bg} у горизонта (по -vWorld.z), поверх —
  сетка;
* сетка рисуется в шейдере земли по мировым координатам: полосы поперёк кадра
  `fract((vWorld.z + uOffset) / {spacing})` и вдоль кадра `fract(vWorld.x /
  {spacing})` с шагом {spacing} единиц, ширина линии растёт с глубиной так, что на кадре
  цвет {c_line}, дальние линии гаснут к горизонту; ширина линий не меньше
  пикселя, иначе на кадре их не видно;
* солнце — `THREE.Mesh` с именем `sun` на z = -{sun_near} (перед небом), радиус
  {sun_units} единицы, `ShaderMaterial`: сверху {c_sun}, книзу темнее, а в нижних
  двух третях диска `discard` вырезает горизонтальные полосы;
* имена объектов: `sky`, `ground`, `sun`; света нет — всё самосветящееся.

Анимация:
* сетка бежит на камеру: `uOffset` растёт на {speed}*dt и заворачивается
  остатком от деления на {spacing} (картинка при завороте не рвётся);
* солнце опускается на {sun_fall} высоты кадра в секунду, пока его центр не
  окажется на 0.52 высоты кадра;
* `window.__probe()` возвращает `t`, `objects`, `sun: [x, y]` (доли кадра) и
  `grid_z` (текущий uOffset).

Проверяется: страница и кадр (5 проверок), палитра по долям, горизонт и профиль
неба, симметрия кадра, детализация (сетка не сливается в заливку) и то, что
сцена движется — кадры на t и t+{steps} шагов различаются.
"""

GOLDEN_L2 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>synthwave</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const W = 1920, H = 1080, FOV = 45;
const HORIZON = {horizon};
const SUN_X = {sun_x}, SUN_Y = {sun_y}, SUN_R = {sun_r};
const SUN_FALL = {sun_fall};
const SPACING = {spacing}, SPEED = {speed};
const BACK = {back};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, W / H, 0.1, 400);
camera.position.set(0, 0, 0);
camera.lookAt(0, 0, -1);

// Полувысота кадра на дистанции фона: из неё выводятся все размеры в долях кадра.
const halfBack = Math.tan(FOV * Math.PI / 360) * BACK;
const frameH = 2 * halfBack;
const groundY = (0.5 - HORIZON) * frameH;   // мировая высота линии горизонта

const sky = new THREE.Mesh(
  new THREE.PlaneGeometry(160, HORIZON * frameH),
  new THREE.ShaderMaterial({{
    uniforms: {{
      uTop: {{value: new THREE.Color({c_bg})}},
      uBottom: {{value: new THREE.Color({c_cyan})}},
    }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; varying vec2 vUv;
      void main() {{
        // У PlaneGeometry uv.y = 1 — верхний край кадра, uv.y = 0 — горизонт:
        // uBottom (циан) живёт у горизонта, uTop (ночь) — у верхнего края.
        // colorspace_fragment обязателен: без него шейдер отдаёт линейные
        // значения и хекс не совпадает с тем, что видно на кадре.
        vec3 colour = mix(uBottom, uTop, smoothstep(0.0, 0.7, vUv.y));
        gl_FragColor = vec4(colour, 1.0);
        #include <colorspace_fragment>
      }}`,
  }})
);
sky.name = 'sky';
sky.position.set(0, halfBack - HORIZON * frameH / 2, -BACK);
scene.add(sky);

// Земля несёт сетку в шейдере: у `LineSegments` ширина всегда один пиксель, а
// шейдерная полоса видна и даёт перспективный неоновый след к горизонту.
// Ближний край земли обязан стоять перед камерой (z = -1), иначе точка с w = 0
// ломает интерполяцию `varying` и сетка не читается.
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(160, 100, 32, 20),
  new THREE.ShaderMaterial({{
    uniforms: {{
      uNear: {{value: new THREE.Color({c_ground})}},
      uFar: {{value: new THREE.Color({c_bg})}},
      uLine: {{value: new THREE.Color({c_line})}},
      uSpacing: {{value: SPACING}},
      uOffset: {{value: 0.0}},
    }},
    vertexShader: `varying vec3 vWorld;
      void main() {{
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
      }}`,
    fragmentShader: `uniform vec3 uNear; uniform vec3 uFar; uniform vec3 uLine;
      uniform float uSpacing; uniform float uOffset; varying vec3 vWorld;
      void main() {{
        float depth = -vWorld.z;
        vec3 base = mix(uNear, uFar, clamp(depth / 30.0, 0.0, 1.0));
        // ширина линии пропорциональна глубине — на кадре она выходит
        // постоянной (около трёх пикселей), иначе у горизонта сетка исчезает
        float w = clamp(0.004 * depth, 0.012, 0.09);
        float gx = smoothstep(w, 0.0, abs(fract(vWorld.x / uSpacing + 0.5) - 0.5));
        float gz = smoothstep(w, 0.0, abs(fract((vWorld.z + uOffset) / uSpacing + 0.5) - 0.5));
        float line = max(gx, gz) * (1.0 - smoothstep(50.0, 95.0, depth));
        gl_FragColor = vec4(mix(base, uLine, line), 1.0);
        #include <colorspace_fragment>
      }}`,
  }})
);
ground.name = 'ground';
ground.rotation.x = -Math.PI / 2;
ground.position.set(0, groundY, -51);
scene.add(ground);

const sun = new THREE.Mesh(
  new THREE.CircleGeometry(SUN_R * frameH, 64),
  new THREE.ShaderMaterial({{
    uniforms: {{
      uTop: {{value: new THREE.Color({c_sun})}},
      uBottom: {{value: new THREE.Color({c_sun_low})}},
    }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; varying vec2 vUv;
      void main() {{
        // полосатый низ диска: щели прорезаны в двух нижних третях
        if (vUv.y < 0.62 && fract(vUv.y * 22.0) < 0.5) discard;
        gl_FragColor = vec4(mix(uBottom, uTop, vUv.y), 1.0);
        #include <colorspace_fragment>
      }}`,
  }})
);
sun.name = 'sun';
sun.position.set((SUN_X - 0.5) * 2 * halfBack * (W / H), halfBack - SUN_Y * frameH, -BACK + 0.5);
scene.add(sun);

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;

window.__step = (dt) => {{
  t += dt;
  ground.material.uniforms.uOffset.value = (ground.material.uniforms.uOffset.value + SPEED * dt) % SPACING;
  const floor = halfBack - 0.52 * frameH;
  sun.position.y = Math.max(floor, sun.position.y - SUN_FALL * frameH * dt);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  sun: [round((sun.position.x / (2 * halfBack * (W / H))) + 0.5), round((halfBack - sun.position.y) / frameH)],
  grid_z: round(ground.material.uniforms.uOffset.value),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _l2_sky_gradient():
    """Небо — от тёмной ночи к яркой циановой полосе: профиль обязан расти."""
    return _profile_check(
        "кадр: небо светлеет к горизонту",
        lambda profile: (
            profile[3] - profile[0] >= 8.0 and all(b >= a - 1.5 for a, b in zip(profile[:3], profile[1:4])),
            f"кадр: профиль неба {profile[:4]} (нужен рост к горизонту)",
        ),
    )


def _l2_horizon_band():
    """Полоса у горизонта ярче неба над ней и ярче земли под ней."""
    def check(pixels: dict) -> tuple[bool, str]:
        profile = _lums(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        good = profile[3] >= profile[2] + 4.0 and profile[3] >= profile[4] + 20.0
        return good, f"кадр: полоса горизонта {profile[3]:.1f} против неба {profile[2]:.1f} и земли {profile[4]:.1f}"

    return visual._label(check, "кадр: над горизонтом яркая полоса, ниже — тёмная земля")


def _l2_ground_grid_reads():
    """Земля под горизонтом не светлее полосы горизонта и не однотонна:
    сетка на ней видна полосами профиля ярче фона."""
    def check(pixels: dict) -> tuple[bool, str]:
        profile = _lums(pixels)
        sats = _sats(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        ground = profile[4:8]
        spread = max(ground) - min(ground)
        good = max(ground) <= profile[3] and (spread >= 1.5 or min(sats[4:8]) >= 0.5)
        return good, f"кадр: земля {ground} (разброс {spread:.1f}) под полосой {profile[3]:.1f}"

    return visual._label(check, "кадр: земля темнее горизонта, но не однотонна")


def _l2_sun_centred():
    def check(pixels: dict) -> tuple[bool, str]:
        stats = pixels.get("warm") or {}
        centroid = stats.get("centroid")
        if not isinstance(centroid, list):
            return False, "кадр: солнце не найдено"
        x = float(centroid[0])
        return 0.42 <= x <= 0.58, f"кадр: центр солнца по x = {x:.3f} (нужно 0.42..0.58)"

    return visual._label(check, "кадр: солнце стоит по центру кадра")


# =========================================================================== #
# L-3 — мандала
# =========================================================================== #

L3_R_OUT = 0.34  # внешний радиус мандалы: доля высоты кадра
L3_R_IN = 0.10  # внутренний радиус: доля высоты кадра
L3_DISTANCE = 8.0  # дистанция камеры
L3_PALETTE = {
    "#0a0a24": 0.25,  # ночной фон
    "#ffd257": 0.012,  # золотое ядро и кольца
    "#ff3aa8": 0.012,  # розовые лепестки
    "#2ee6ff": 0.010,  # циановые лепестки
}
L3_OMEGA = round(math.radians(15.0) * 60 / STEPS, 4)  # рад/с: ровно 15° за 78 шагов проверки

PROMPT_L3 = """Собери мандалу: 12-лучевая радиальная симметрия вокруг яркого ядра,
кольца и лепестки из торов, цилиндров и плоскостей, вид строго в центр.

Композиция (проверяется по кадру грубо):
* мандала висит в центре кадра: её внешний радиус {r_out} высоты кадра, то есть
  от 0.5 - {r_out} до 0.5 + {r_out} по вертикали; углы кадра остаются фоном;
* кадр зеркально симметричен относительно вертикальной оси (лучи разложены от
  нуля в обе стороны), поэтому мандала поворачивается на угол, кратный 15°;
* яркость кадра падает от центра к краям: средние полосы профиля ярче верхней и
  нижней полос;
* лучи читаются: в четырёх боковых областях кадра (верх, низ, лево, право) свет
  доходит до края, а углы кадра остаются тёмным фоном.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — ночной фон, не меньше {s_bg:.0%} кадра;
* {c_gold} — ядро и кольца, не меньше {s_gold:.1%} кадра;
* {c_pink} — розовые лепестки, не меньше {s_pink:.1%} кадра;
* {c_cyan} — циановые лепестки, не меньше {s_cyan:.1%} кадра.

Сцена (камера в (0, 0, {distance:.0f}), смотрит в (0, 0, 0), угол 45, кадр 1920x1080;
высота кадра на плоскости мандалы {frame_h} единицы):
* ядро — `THREE.Mesh` с именем `core`, диск радиуса {core_r} высоты кадра,
  цвет {c_gold}, поверх него кольцо `TorusGeometry` того же золота;
* лучи — имя группы `petals`: 12 плоскостей шириной {arm_w} высоты кадра от
  радиуса {r_in} до {r_out} высоты кадра, через одну {c_pink} и {c_cyan}, плюс
  12 кружков-наконечников радиуса {tip_r} высоты кадра на внешнем радиусе;
* вторая, короткая серия лепестков того же числа, повёрнутая на 15°, от радиуса
  {r_in} до {short_r} высоты кадра, шириной {short_w};
* кольца — торы `TorusGeometry` радиуса {ring_r} и {r_out} высоты кадра, труба
  {tube} и {tube_thin} высоты кадра, цвета {c_gold} и {c_cyan};
* фон — чёрный {c_bg} (цвет очистки кадра), света нет.

Анимация:
* вся мандала — группа `wheel` — поворачивается на `wheel.rotation.z = {omega} * t`,
  то есть ровно на 15° за {steps} шагов: за проверку лучи обязаны заметно
  сдвинуться, а кадр остаться зеркально симметричным;
* `window.__probe()` возвращает `t`, `objects`, `angle` (градусы, 4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, центр ярче краёв,
симметрия, детализация лучей и то, что мандала движется.
"""

GOLDEN_L3 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>mandala</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const W = 1920, H = 1080, FOV = 45;
const DIST = {distance};
const R_IN = {r_in}, R_OUT = {r_out}, R_SHORT = {short_r};
const CORE_R = {core_r}, RING_R = {ring_r}, TUBE = {tube}, TUBE_THIN = {tube_thin};
const ARM_W = {arm_w}, SHORT_W = {short_w}, TIP_R = {tip_r};
const OMEGA = {omega};
const BG = {c_bg};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
renderer.setClearColor(BG);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, W / H, 0.1, 100);
camera.position.set(0, 0, DIST);
camera.lookAt(0, 0, 0);

const frameH = 2 * Math.tan(FOV * Math.PI / 360) * DIST;

// Мандала — группа wheel: поворот целиком, лучи и кольца внутри неё.
// Радиальный градиент на каждой детали даёт кадру больше двенадцати оттенков.
const wheel = new THREE.Group();
wheel.name = 'wheel';
scene.add(wheel);

const radial = (inner, outer) => new THREE.ShaderMaterial({{
  uniforms: {{
    uInner: {{value: new THREE.Color(inner)}},
    uOuter: {{value: new THREE.Color(outer)}},
  }},
  vertexShader: `varying vec2 vUv;
    void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  fragmentShader: `uniform vec3 uInner; uniform vec3 uOuter; varying vec2 vUv;
    void main() {{
      gl_FragColor = vec4(mix(uInner, uOuter, smoothstep(0.0, 1.0, vUv.y)), 1.0);
      #include <colorspace_fragment>
    }}`,
}});

const core = new THREE.Mesh(new THREE.CircleGeometry(CORE_R * frameH, 64), radial({c_gold_hi}, {c_gold}));
core.name = 'core';
core.position.z = 0.2;
wheel.add(core);
const coreRing = new THREE.Mesh(new THREE.TorusGeometry(CORE_R * frameH * 1.7, TUBE * frameH, 10, 96),
  radial({c_pink}, {c_gold}));
coreRing.name = 'core_ring';
coreRing.position.z = 0.15;
wheel.add(coreRing);

const petals = new THREE.Group();
petals.name = 'petals';
wheel.add(petals);
for (let k = 0; k < 12; k++) {{
  const a = k * Math.PI / 6;
  const pink = (k % 2) === 0;
  const mat = radial(pink ? {c_pink_hi} : {c_cyan_hi}, pink ? {c_pink} : {c_cyan});
  const mid = (R_IN + R_OUT) / 2;
  const arm = new THREE.Mesh(new THREE.PlaneGeometry(ARM_W * frameH, (R_OUT - R_IN) * frameH), mat);
  arm.position.set(Math.cos(a) * mid * frameH, Math.sin(a) * mid * frameH, 0.05);
  arm.rotation.z = a - Math.PI / 2;
  petals.add(arm);
  const tip = new THREE.Mesh(new THREE.CircleGeometry(TIP_R * frameH, 24), mat);
  tip.position.set(Math.cos(a) * R_OUT * frameH, Math.sin(a) * R_OUT * frameH, 0.09);
  petals.add(tip);
  // короткая серия: те же 12 лучей, сдвинутые на 15°, другой оттенок
  const b = a + Math.PI / 12;
  const shortMat = radial(pink ? {c_cyan} : {c_pink}, {c_void});
  const midd = (R_IN + R_SHORT) / 2;
  const short = new THREE.Mesh(new THREE.PlaneGeometry(SHORT_W * frameH, (R_SHORT - R_IN) * frameH), shortMat);
  short.position.set(Math.cos(b) * midd * frameH, Math.sin(b) * midd * frameH, 0.02);
  short.rotation.z = b - Math.PI / 2;
  petals.add(short);
}}

const midRing = new THREE.Mesh(new THREE.TorusGeometry(RING_R * frameH, TUBE * frameH, 10, 120),
  radial({c_gold}, {c_gold_hi}));
midRing.name = 'mid_ring';
midRing.position.z = 0.12;
wheel.add(midRing);
const outerRing = new THREE.Mesh(new THREE.TorusGeometry((R_OUT + 0.03) * frameH, TUBE_THIN * frameH, 10, 160),
  radial({c_cyan}, {c_cyan}));
outerRing.name = 'outer_ring';
outerRing.position.z = 0.12;
wheel.add(outerRing);

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;

window.__step = (dt) => {{
  t += dt;
  wheel.rotation.z = OMEGA * t;
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean).concat(['petals']),
  angle: round(wheel.rotation.z * 180 / Math.PI),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _l3_centre_bright():
    return _profile_check(
        "кадр: яркость падает от центра к краям",
        lambda profile: (
            (profile[3] + profile[4]) / 2 >= max(profile[0], profile[7]) * 1.30
            and (profile[3] + profile[4]) / 2 >= max(profile[1], profile[6]) * 1.02,
            f"кадр: центр {(profile[3] + profile[4]) / 2:.1f} против краёв {profile[0]}/{profile[7]}",
        ),
    )


def _l3_petals_read():
    def check(pixels: dict) -> tuple[bool, str]:
        regions = _regions(pixels)
        if len(regions) < 9:
            return False, "кадр: нет областей"
        sides = [_bright(regions[i]) for i in (1, 3, 5, 7)]
        corners = [_bright(regions[i]) for i in (0, 2, 6, 8)]
        good = min(sides) >= max(corners) + 12
        return good, f"кадр: лучи на краях {sides} против углов {corners}"

    return visual._label(check, "кадр: лепестки доходят до краёв кадра")


def _l3_centre_region():
    def check(pixels: dict) -> tuple[bool, str]:
        regions = _regions(pixels)
        if len(regions) < 9:
            return False, "кадр: нет областей"
        centre = _bright(regions[4])
        sides = [_bright(regions[i]) for i in (1, 3, 5, 7)]
        good = centre >= max(sides) * 0.85
        return good, f"кадр: ядро {centre} против лучей {max(sides)}"

    return visual._label(check, "кадр: ядро мандалы читается в центре")


# =========================================================================== #
# L-4 — ночной город
# =========================================================================== #

L4_EYE = 8.0  # высота камеры
L4_DISTANCE = 26.0  # дистанция до фасада
L4_PALETTE = {
    "#2a1b5c": 0.10,  # ночное небо
    "#0b0b14": 0.15,  # силуэты зданий и улица
    "#22e0d0": 0.012,  # бирюзовые окна
    "#ffb066": 0.010,  # тёплые окна
}

PROMPT_L4 = """Собери ночной город: ряд домов-коробок вдоль улицы, светящиеся окна на
фасадах, синяя ночь и ползущая дымка.

Композиция (проверяется по кадру грубо):
* камера в ({eye:.0f}) по высоте и {distance:.0f} единицах от фасада, смотрит
  горизонтально — горизонт ложится на середину кадра, верх кадра занимает небо;
* небо заметно светлее улицы, а крыши домов образуют ступенчатый силуэт: профиль
  яркости падает от верхних полос к нижним, и в средней части кадра есть
  перепады (крыши разной высоты);
* окна светятся в средней зоне кадра: тёплые окна попадают в полосу от 0.15 до
  0.70 высоты кадра и разложены по горизонтали через весь кадр.

Здания (все числа — фикстура, x — центр по горизонтали, ширина, высота):
{rows}

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — ночное небо, не меньше {s_bg:.0%} кадра;
* {c_sil} — силуэты домов и улица, не меньше {s_sil:.0%} кадра;
* {c_mint} — бирюзовые окна, не меньше {s_mint:.1%} кадра;
* {c_warm} — тёплые окна, не меньше {s_warm:.1%} кадра.

Сцена (кадр 1920x1080, угол 45, света нет — окна самосветящиеся):
* земля — плоскость y = 0 цвета {c_sil}, дома стоят на ней;
* дом — `THREE.Mesh` из `BoxGeometry(ширина, высота, 3)` в цвете {c_sil} с именем
  `house`, центр по x из таблицы выше, низ на y = 0;
* окна — один `THREE.InstancedMesh` с именем `windows`: прямоугольники 0.5x0.7 на
  фасаде, сетка «столбцы x этажи» (столбец через 0.95 по ширине дома, этаж через
  1.45 по высоте, начиная с 1.0), у дома с чётным номером окна бирюзовые
  {c_mint}, у нечётного — тёплые {c_warm};
* дымка — плоскость перед домами цвета {c_haze} с прозрачностью 0.22 и
  градиентом по вертикали, имя `haze`;
* небо — цвет очистки кадра {c_bg}.

Анимация:
* окно с номером i светится, если остаток от деления (i * {phase} + t * {blink})
  на 1 меньше 0.78, иначе гаснет до 15% яркости: мигание детерминировано фазой
  по индексу окна;
* дымка ходит по вертикали: `haze.position.y = {haze_y} + 0.9 * sin(t * 0.25)`;
* `window.__probe()` возвращает `t`, `objects`, `lit` (сколько окон светится) и
  `haze_y`.

Проверяется: страница и кадр (5 проверок), палитра по долям, силуэт и небо,
детализация (окна и рёбра домов), движение — кадры на t и t+{steps} шагов
различаются, дымка и фаза окон меняются.
"""

GOLDEN_L4 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>night city</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const W = 1920, H = 1080, FOV = 45;
const EYE = {eye}, DIST = {distance};
const BG = {c_bg}, SIL = {c_sil}, MINT = {c_mint}, WARM = {c_warm}, EDGE = {c_edge};
const PHASE = {phase}, BLINK = {blink}, HAZE_Y = {haze_y};
const HOUSES = {houses};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
renderer.setClearColor(BG);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
// Небо — градиентная плоскость: фон обязан светлеть к горизонту, а не быть
// однотонной заливкой (иначе кадр вырождается в четыре цвета).
const frameH = 2 * Math.tan(FOV * Math.PI / 360) * DIST;
const sky = new THREE.Mesh(
  new THREE.PlaneGeometry(120, 60),
  new THREE.ShaderMaterial({{
    uniforms: {{ uTop: {{value: new THREE.Color({c_sky_hi})}}, uBottom: {{value: new THREE.Color(BG)}} }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; varying vec2 vUv;
      void main() {{
        gl_FragColor = vec4(mix(uBottom, uTop, smoothstep(0.05, 0.70, vUv.y)), 1.0);
        #include <colorspace_fragment>
      }}`,
  }})
);
sky.name = 'sky';
sky.position.set(0, 16, -45);
scene.add(sky);

const camera = new THREE.PerspectiveCamera(FOV, W / H, 0.1, 200);
camera.position.set(0, EYE, DIST);
camera.lookAt(0, EYE, 0);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(120, 120),
  new THREE.ShaderMaterial({{
    uniforms: {{ uNear: {{value: new THREE.Color(SIL)}}, uFar: {{value: new THREE.Color({c_street})}} }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uNear; uniform vec3 uFar; varying vec2 vUv;
      void main() {{
        gl_FragColor = vec4(mix(uFar, uNear, smoothstep(0.0, 0.65, vUv.y)), 1.0);
        #include <colorspace_fragment>
      }}`,
  }})
);
ground.name = 'street';
ground.rotation.x = -Math.PI / 2;
ground.position.set(0, 0, 0);
scene.add(ground);

const edgeMat = new THREE.LineBasicMaterial({{ color: EDGE }});
const cells = [];
for (let b = 0; b < HOUSES.length; b++) {{
  const [x, w, h] = HOUSES[b];
  // фасад чуть светлее к низу: тонкая разница оттенков держит кадр живым
  const shade = new THREE.Color(SIL).offsetHSL(0.0, 0.0, 0.012 * b);
  const geo = new THREE.BoxGeometry(w, h, 3);
  const house = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({{ color: shade }}));
  house.name = 'house';
  house.position.set(x, h / 2, 0);
  scene.add(house);
  // контур крыши и рёбер: даёт силуэту читаемую линию и добавляет деталей
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
  edges.position.copy(house.position);
  scene.add(edges);
  const cols = Math.max(2, Math.floor(w / 0.95));
  const rows = Math.max(3, Math.floor(h / 1.45));
  for (let c = 0; c < cols; c++) {{
    for (let r = 0; r < rows; r++) {{
      const wx = x + (c - (cols - 1) / 2) * 0.95;
      const wy = 1.0 + r * 1.45;
      cells.push([wx, wy, (b % 2) ? 1 : 0]);
    }}
  }}
}}

const winGeo = new THREE.PlaneGeometry(0.5, 0.7);
const winMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
const windows = new THREE.InstancedMesh(winGeo, winMat, cells.length);
windows.name = 'windows';
const lit = [], dim = [];
const m4 = new THREE.Matrix4();
for (let i = 0; i < cells.length; i++) {{
  const [x, y, warm] = cells[i];
  m4.makeTranslation(x, y, 1.56);
  windows.setMatrixAt(i, m4);
  const base = new THREE.Color(warm ? WARM : MINT);
  lit.push(base);
  dim.push(base.clone().multiplyScalar(0.15));
  windows.setColorAt(i, base);
}}
scene.add(windows);

const haze = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 16),
  new THREE.ShaderMaterial({{
    uniforms: {{ uColor: {{value: new THREE.Color({c_haze})}} }},
    transparent: true, depthWrite: false,
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uColor; varying vec2 vUv;
      void main() {{ gl_FragColor = vec4(uColor, 0.22 * (1.0 - abs(vUv.y - 0.35))); }}`,
  }})
);
haze.name = 'haze';
haze.position.set(0, HAZE_Y, 14);
scene.add(haze);

const house = windows;
let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;
const litCount = () => {{
  let n = 0;
  for (let i = 0; i < cells.length; i++) if (((i * PHASE + t * BLINK) % 1) < 0.78) n++;
  return n;
}};

window.__step = (dt) => {{
  t += dt;
  let n = 0;
  for (let i = 0; i < cells.length; i++) {{
    const on = ((i * PHASE + t * BLINK) % 1) < 0.78;
    windows.setColorAt(i, on ? lit[i] : dim[i]);
    if (on) n++;
  }}
  windows.instanceColor.needsUpdate = true;
  haze.position.y = HAZE_Y + 0.9 * Math.sin(t * 0.25);
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  lit: litCount(),
  haze_y: round(haze.position.y),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _l4_sky_brighter():
    return _profile_check(
        "кадр: небо светлее улицы",
        lambda profile: (
            profile[0] + profile[1] >= profile[6] + profile[7] + 40.0,
            f"кадр: небо {profile[0] + profile[1]:.1f} против улицы {profile[6] + profile[7]:.1f}",
        ),
    )


def _l4_skyline_steps():
    return _profile_check(
        "кадр: крыши на разной высоте",
        lambda profile: (
            profile[1] >= profile[5] + 10.0 and max(profile[2:6]) - min(profile[2:6]) >= 8.0,
            f"кадр: силуэт {profile[1:6]} (нужен ступенчатый спад)",
        ),
    )


def _l4_windows_area():
    def check(pixels: dict) -> tuple[bool, str]:
        stats = pixels.get("warm") or {}
        centroid = stats.get("centroid")
        fraction = float(stats.get("fraction") or 0.0)
        if not isinstance(centroid, list):
            return False, "кадр: тёплых окон не видно"
        x, y = float(centroid[0]), float(centroid[1])
        good = 0.30 <= x <= 0.70 and 0.15 <= y <= 0.70 and fraction >= 0.004
        return good, f"кадр: центр тёплых окон ({x:.3f}, {y:.3f}), доля {fraction:.4f}"

    return visual._label(check, "кадр: окна светятся в средней зоне")


def _l4_haze_soft():
    def check(pixels: dict) -> tuple[bool, str]:
        stats = pixels.get("dark") or {}
        fraction = float(stats.get("fraction") or 0.0)
        good = fraction >= 0.10
        return good, f"кадр: тёмный силуэт занимает {fraction:.4f} кадра (нужно >= 0.10)"

    return visual._label(check, "кадр: силуэт домов и улицы читается")


# =========================================================================== #
# L-5 — мост в изометрии
# =========================================================================== #

L5_CAM = (13.0, 7.0, 21.0)
L5_LOOK = (0.0, 1.0, 0.0)
L5_PALETTE = {
    "#15171f": 0.25,  # земля и пустота
    "#3ce8f0": 0.06,  # рёбра, перила и арка
    "#ffc247": 0.008,  # яркая дорожка
}
L5_OMEGA = 0.10  # рад/с: медленный поворот всей конструкции

PROMPT_L5 = """Собери мост в наклонной изометрии: балка пролёта с ярким тёплым
полотном дороги, арка над ней, пилоны вниз; рёбра подсвечены `EdgesGeometry` и
`LineSegments`.

Композиция (проверяется по кадру грубо):
* камера в ({cam}), смотрит в ({look}) — ракурс в три четверти, поэтому рёбра
  идут по кадру наклонно, а не по осям, и мост пересекает кадр по диагонали;
* яркая дорожка {c_warm} идёт по центру кадра: центральная область кадра заметно
  ярче левой и правой;
* конструкция занимает середину кадра и насыщена там (бирюзовые рёбра): в верхней
  полосе кадра — пустой фон, углы кадра — тоже фон {c_bg};
* плотность границ на кадре высокая: у каждой коробки видны рёбра.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — фон и пустота, не меньше {s_bg:.0%} кадра;
* {c_teal} — рёбра, перила, арка и подвесы, не меньше {s_teal:.0%} кадра;
* {c_warm} — полотно дороги, не меньше {s_warm:.1%} кадра.

Сцена (кадр 1920x1080, угол 45, коробки — `BoxGeometry`, рёбра —
`new THREE.LineSegments(new THREE.EdgesGeometry(geometry), lineMat)` поверх каждой
коробки, вся конструкция — группа `rig` в начале координат):
* балка пролёта — коробка 18 x 1.0 x 5.0 на y = 0 в тёмной бирюзе {c_deck}, с
  рёбрами;
* полотно дороги — коробка 18 x 0.35 x 4.6 на y = 0.68, цвет {c_warm}, с рёбрами;
* перила — две коробки 18 x 0.4 x 0.22 на y = 1.05 по z = ±2.3, цвет {c_teal};
* четыре пилона — коробки 1.4 x 8.0 x 1.4 на y = -4.0 по x = ±6.5 и z = ±2.0, в
  тёмной бирюзе {c_deck}, с рёбрами;
* арка — `TorusGeometry` радиуса {arch_r} с трубой 0.30, полукруг
  (`thetaLength = Math.PI`), стоит в плоскости XY на y = 1.2, цвет {c_teal}, с
  рёбрами (`EdgesGeometry` с порогом 24);
* подвесы — две коробки 0.16 высотой {arch_h} по x = ±{hang}, от арки вниз до
  полотна, цвет {c_teal};
* фон — {c_bg} (цвет очистки кадра), света нет: материалы самосветящиеся.

Анимация:
* вся конструкция — группа `rig` — поворачивается на `rig.rotation.y = {omega} * t`,
  то есть примерно на 6° за {steps} шагов проверки: рёбра обязаны заметно
  сдвинуться, иначе кадры не различаются;
* `window.__probe()` возвращает `t`, `objects`, `yaw` (градусы, 4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, центральная дорожка,
наклон и плотность рёбер, и то, что конструкция движется.
"""

GOLDEN_L5 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>bridge</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const W = 1920, H = 1080, FOV = 45;
const BG = {c_bg}, TEAL = {c_teal}, WARM = {c_warm}, DECK = {c_deck};
const OMEGA = {omega}, ARCH_R = {arch_r}, ARCH_H = {arch_h}, HANG = {hang};
const CAM = [{cam}], LOOK = [{look}];

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
renderer.setClearColor(BG);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, W / H, 0.1, 300);
camera.position.set(CAM[0], CAM[1], CAM[2]);
camera.lookAt(LOOK[0], LOOK[1], LOOK[2]);

const rig = new THREE.Group();
rig.name = 'rig';
scene.add(rig);

const grad = (tags) => new THREE.ShaderMaterial({{
  uniforms: Object.fromEntries(tags.map((t, i) => ['u' + i, {{value: new THREE.Color(t)}}])),
  vertexShader: `varying vec3 vWorld;
    void main() {{ vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
      gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0); }}`,
  fragmentShader: `uniform vec3 u0; uniform vec3 u1; varying vec3 vWorld;
    void main() {{
      gl_FragColor = vec4(mix(u0, u1, clamp((vWorld.y + 6.0) / 14.0, 0.0, 1.0)), 1.0);
      #include <colorspace_fragment>
    }}`,
}});
// дорожка: тёплый градиент вдоль себя — яркое пятно в центре кадра
const roadMat = (() => {{
  const m = grad([WARM, {c_warm_hi}]);
  m.fragmentShader = `uniform vec3 u0; uniform vec3 u1; varying vec3 vWorld;
    void main() {{
      gl_FragColor = vec4(mix(u1, u0, clamp((vWorld.z + 9.0) / 18.0, 0.0, 1.0)), 1.0);
      #include <colorspace_fragment>
    }}`;
  return m;
}})();

const lineMat = new THREE.LineBasicMaterial({{ color: TEAL }});

function solid(w, h, d, x, y, z, mat, edges) {{
  const geo = new THREE.BoxGeometry(w, h, d);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, y, z);
  rig.add(mesh);
  if (edges) {{
    const line = new THREE.LineSegments(new THREE.EdgesGeometry(geo), lineMat);
    line.position.set(x, y, z);
    rig.add(line);
  }}
}}

// Мост целиком в плоскости XY: пролёт идёт по оси X (в кадре — по диагонали),
// арка стоит над ним, пилоны уходят вниз. Наклон ракурса даёт изометрию.
solid(18, 1.0, 5.0, 0, 0, 0, grad([DECK, {c_deck_hi}]), true);       // балка пролёта
solid(18, 0.35, 4.6, 0, 0.68, 0, roadMat, true);                      // полотно дороги
for (const sz of [-2.3, 2.3]) {{
  solid(18, 0.4, 0.22, 0, 1.05, sz, new THREE.MeshBasicMaterial({{color: TEAL}}), true);  // перила
}}
for (const sx of [-6.5, 6.5]) {{
  for (const sz of [-2.0, 2.0]) solid(1.4, 8.0, 1.4, sx, -4.0, sz, grad([{c_pylon}, DECK]), true);
}}
const archGeo = new THREE.TorusGeometry(ARCH_R, 0.30, 8, 48, Math.PI);
const arch = new THREE.Mesh(archGeo, new THREE.MeshBasicMaterial({{color: TEAL}}));
arch.name = 'arch';
arch.position.set(0, 1.2, 0);
rig.add(arch);
const archEdges = new THREE.LineSegments(new THREE.EdgesGeometry(archGeo, 24), lineMat);
archEdges.position.copy(arch.position);
rig.add(archEdges);
// подвесы арки — вертикальные стойки: добавляют рёбра и читаемый наклон
for (const sx of [-HANG, HANG]) {{
  solid(0.16, ARCH_H, 0.16, sx, 1.2 + 0.2 + ARCH_H / 2, 0, new THREE.MeshBasicMaterial({{color: TEAL}}), true);
}}

let t = 0;
const round = (v) => Math.round(v * 1e4) / 1e4;

window.__step = (dt) => {{
  t += dt;
  rig.rotation.y = OMEGA * t;
  renderer.render(scene, camera);
  return window.__probe();
}};

window.__probe = () => ({{
  t: round(t),
  objects: ['rig'],
  yaw: round(rig.rotation.y * 180 / Math.PI),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _l5_walkway_centre():
    def check(pixels: dict) -> tuple[bool, str]:
        regions = _regions(pixels)
        if len(regions) < 9:
            return False, "кадр: нет областей"
        middle = _bright(regions[4])
        sides = (_bright(regions[3]) + _bright(regions[5])) / 2
        good = middle >= sides * 1.25
        return good, f"кадр: центр {middle} против соседей {sides:.0f} (нужна яркая дорожка по центру)"

    return visual._label(check, "кадр: яркая дорожка идёт по центру кадра")


def _l5_tilted_edges():
    return _profile_check(
        "кадр: конструкция в середине, фон сверху",
        lambda profile: (
            profile[0] <= min(profile[3], profile[4]) and profile[0] <= 60.0,
            f"кадр: верх {profile[0]:.1f}, середина {profile[3]:.1f}/{profile[4]:.1f}",
        ),
    )


def _l5_saturation_middle():
    def check(pixels: dict) -> tuple[bool, str]:
        sats = _sats(pixels)
        if len(sats) < 8:
            return False, "кадр: нет профиля насыщенности"
        good = max(sats[3], sats[4]) >= max(sats[0], sats[7]) * 1.25
        return good, f"кадр: насыщенность середины {sats[3]:.3f}/{sats[4]:.3f} против краёв {sats[0]:.3f}/{sats[7]:.3f}"

    return visual._label(check, "кадр: конструкция насыщена, пустота — нет")


def _l5_corners_free():
    def check(pixels: dict) -> tuple[bool, str]:
        regions = _regions(pixels)
        if len(regions) < 9:
            return False, "кадр: нет областей"
        corners = [_bright(regions[i]) for i in (0, 2)]
        centre = _bright(regions[4])
        good = max(corners) <= centre * 0.75
        return good, f"кадр: верхние углы {corners} против центра {centre}"

    return visual._label(check, "кадр: верхние углы остаются фоном")


# --------------------------------------------------------------------------- #
# Сборка заданий
# --------------------------------------------------------------------------- #


def _build_l2(rng: random.Random) -> core.TaskInstance:
    speed = round(rng.choice([0.9, 1.0, 1.1]), 3)
    sun_y = round(rng.uniform(0.34, 0.42), 3)
    sun_fall = round(rng.choice([0.025, 0.03, 0.035]), 3)
    half = _frame_height(L2_BACK)

    prompt = PROMPT_L2.format(
        horizon=L2_HORIZON,
        sun_r=L2_SUN_R,
        sun_y=sun_y,
        spacing=L2_SPACING,
        back=L2_BACK,
        half=round(half, 3),
        sun_units=round(L2_SUN_R * 2 * half, 3),
        sun_near=round(L2_BACK - 0.5, 1),
        sky_h=round(L2_HORIZON * 2 * half, 3),
        ground=round((0.5 - L2_HORIZON) * 2 * half, 3),
        speed=f"{speed:.2f}",
        sun_fall=f"{sun_fall:.3f}",
        steps=STEPS,
        c_bg="#1b0f3c",
        c_ground="#120a2a",
        c_line="#ff2fb0",
        c_cyan="#19e0e8",
        c_sun="#ffb03a",
        c_sun_low="#ff5a2e",
        s_bg=L2_PALETTE["#1b0f3c"],
        s_cyan=L2_PALETTE["#19e0e8"],
        s_sun=L2_PALETTE["#ffb03a"],
        s_line=L2_PALETTE["#ff2fb0"],
    )
    golden = GOLDEN_L2.format(
        horizon=L2_HORIZON,
        sun_x=L2_SUN_X,
        sun_y=sun_y,
        sun_r=L2_SUN_R,
        sun_fall=sun_fall,
        spacing=L2_SPACING,
        speed=speed,
        back=L2_BACK,
        c_bg="0x1b0f3c",
        c_ground="0x120a2a",
        c_line="0xff2fb0",
        c_cyan="0x19e0e8",
        c_sun="0xffb03a",
        c_sun_low="0xff5a2e",
    )

    for literal in (f"{sun_y:.1%}", f"{speed:.2f}", f"{sun_fall:.3f}", *L2_PALETTE, "#19e0e8", "#ffb03a"):
        assert literal in prompt, literal
    assert 0.30 <= sun_y <= 0.45, "солнце обязано стоять над горизонтом и под верхом кадра"
    travel = speed * STEPS / 60
    assert travel <= L2_SPACING * 2.5, "за проверку сетка не должна убегать слишком далеко"
    assert travel >= 0.6, "сдвиг сетки обязан быть заметен на кадре"
    assert sun_fall * STEPS / 60 >= 0.03, "солнце обязано заметно опуститься"

    palette = [
        _share("#1b0f3c", L2_PALETTE["#1b0f3c"], "небо и земля"),
        _share("#19e0e8", L2_PALETTE["#19e0e8"], "циановая полоса"),
        _share("#ffb03a", L2_PALETTE["#ffb03a"], "солнце"),
        _share("#ff2fb0", L2_PALETTE["#ff2fb0"], "линии сетки"),
    ]
    composition = [
        visual.mirror(True, 10.0, "сцена симметрична относительно вертикали"),
        _l2_sky_gradient(),
        _l2_horizon_band(),
        _l2_sun_centred(),
    ]
    detail = [
        visual.image_min("edges", 0.012),
        visual.image_min("distinct_colors", 20),
        visual.image_min("changed_fraction", 0.12),
        visual.image_min("saturation", 0.35),
    ]
    return art.build_task(
        task_id="L-2",
        title="L-2 — неоновая сетка (synthwave)",
        prompt=prompt,
        spec=art.ArtSpec(palette=palette, composition=composition, detail=detail, motion=STEPS),
        golden=golden,
        fixture={"speed": speed, "sun_y": sun_y, "sun_fall": sun_fall, "horizon": L2_HORIZON},
    )


def _build_l3(rng: random.Random) -> core.TaskInstance:
    ring_r = round(rng.choice([0.16, 0.18, 0.20]), 3)
    short_r = round(rng.choice([0.20, 0.22]), 3)
    frame_h = _frame_height(L3_DISTANCE)

    prompt = PROMPT_L3.format(
        r_out=L3_R_OUT,
        r_in=L3_R_IN,
        short_r=short_r,
        core_r=0.06,
        ring_r=ring_r,
        tube=0.012,
        tube_thin=0.006,
        arm_w=0.030,
        short_w=0.042,
        tip_r=0.022,
        distance=L3_DISTANCE,
        frame_h=round(frame_h, 3),
        omega=f"{L3_OMEGA:.4f}",
        steps=STEPS,
        c_bg="#0a0a24",
        c_gold="#ffd257",
        c_pink="#ff3aa8",
        c_cyan="#2ee6ff",
        s_bg=L3_PALETTE["#0a0a24"],
        s_gold=L3_PALETTE["#ffd257"],
        s_pink=L3_PALETTE["#ff3aa8"],
        s_cyan=L3_PALETTE["#2ee6ff"],
    )
    golden = GOLDEN_L3.format(
        c_gold_hi="0xffe89a",
        c_gold="0xffd257",
        c_pink_hi="0xff86c8",
        c_pink="0xff3aa8",
        c_cyan_hi="0x8ef2ff",
        c_cyan="0x2ee6ff",
        c_void="0x2a1550",
        distance=L3_DISTANCE,
        r_in=L3_R_IN,
        r_out=L3_R_OUT,
        short_r=short_r,
        core_r=0.06,
        ring_r=ring_r,
        tube=0.012,
        tube_thin=0.006,
        arm_w=0.030,
        short_w=0.042,
        tip_r=0.022,
        omega=L3_OMEGA,
        c_bg="0x0a0a24",
    )

    for literal in (str(ring_r), str(short_r), f"{L3_OMEGA:.4f}", f"{STEPS}", *L3_PALETTE):
        assert literal in prompt, literal
    assert L3_R_IN < short_r <= L3_R_OUT, "короткая серия лепестков обязана попадать между кольцами"
    # Поворот обязан быть заметным и кратным 15° на момент проверки: иначе
    # ломается зеркальная симметрия кадра.
    turn = math.degrees(L3_OMEGA * STEPS / 60)
    assert abs((turn + 7.5) % 15.0 - 7.5) < 0.01, f"поворот за проверку {turn}° — не кратен 15°"
    assert turn >= 10.0, "поворот обязан быть заметен"

    palette = [
        _share("#0a0a24", L3_PALETTE["#0a0a24"], "ночной фон"),
        _share("#ffd257", L3_PALETTE["#ffd257"], "ядро и кольца"),
        _share("#ff3aa8", L3_PALETTE["#ff3aa8"], "розовые лепестки"),
        _share("#2ee6ff", L3_PALETTE["#2ee6ff"], "циановые лепестки"),
    ]
    composition = [
        _l3_centre_bright(),
        _l3_petals_read(),
        _l3_centre_region(),
        visual.mirror(True, 12.0, "мандала симметрична относительно вертикали"),
    ]
    detail = [
        visual.image_min("edges", 0.03),
        visual.image_min("distinct_colors", 24),
        visual.image_min("changed_fraction", 0.08),
        visual.image_min("saturation", 0.45),
    ]
    return art.build_task(
        task_id="L-3",
        title="L-3 — мандала",
        prompt=prompt,
        spec=art.ArtSpec(palette=palette, composition=composition, detail=detail, motion=STEPS),
        golden=golden,
        fixture={"ring_r": ring_r, "short_r": short_r, "omega": L3_OMEGA},
    )


def _skyline(rng: random.Random) -> list[dict]:
    """Ряд домов: случайные ширины и высоты, центр ряда — в начале координат."""
    count = rng.randint(6, 7)
    heights: set[float] = set()
    houses: list[dict] = []
    x = 0.0
    for _ in range(count):
        width = round(rng.uniform(1.5, 2.6), 1)
        # высоты обязаны различаться: округление до 0.1 легко даёт дубли
        while True:
            height = round(rng.uniform(8.5, 17.0), 1)
            if height not in heights:
                break
        heights.add(height)
        houses.append({"x": x + width / 2, "w": width, "h": height})
        x += width + round(rng.uniform(0.6, 1.2), 1)
    shift = (x - houses[-1]["w"]) / 2
    for house in houses:
        house["x"] = round(house["x"] - shift, 1)
    return houses


def _build_l4(rng: random.Random) -> core.TaskInstance:
    houses = _skyline(rng)
    phase = round(rng.uniform(0.11, 0.16), 3)
    blink = round(rng.choice([0.5, 0.6, 0.7]), 3)
    haze_y = round(rng.uniform(6.5, 8.0), 2)
    rows = "\n".join(
        f"  * дом {index + 1}: x = {house['x']:.1f}, ширина {house['w']:.1f}, высота {house['h']:.1f}"
        for index, house in enumerate(houses)
    )

    prompt = PROMPT_L4.format(
        eye=L4_EYE,
        distance=L4_DISTANCE,
        rows=rows,
        phase=f"{phase:.3f}",
        blink=f"{blink:.3f}",
        haze_y=f"{haze_y:.2f}",
        steps=STEPS,
        c_bg="#2a1b5c",
        c_sil="#0b0b14",
        c_mint="#22e0d0",
        c_warm="#ffb066",
        c_haze="#4a5a8a",
        s_bg=L4_PALETTE["#2a1b5c"],
        s_sil=L4_PALETTE["#0b0b14"],
        s_mint=L4_PALETTE["#22e0d0"],
        s_warm=L4_PALETTE["#ffb066"],
    )
    golden = GOLDEN_L4.format(
        eye=L4_EYE,
        distance=L4_DISTANCE,
        phase=phase,
        blink=blink,
        haze_y=haze_y,
        houses="[" + ", ".join(f"[{h['x']:.1f}, {h['w']:.1f}, {h['h']:.1f}]" for h in houses) + "]",
        c_bg="0x2a1b5c",
        c_sil="0x0b0b14",
        c_mint="0x22e0d0",
        c_warm="0xffb066",
        c_haze="0x4a5a8a",
        c_street="0x141428",
        c_sky_hi="0x4a2f7a",
        c_edge="0x2f6f8f",
    )

    mentioned = [f"{house[key]:.1f}" for house in houses for key in ("x", "w", "h")]
    for literal in (*mentioned, f"{phase:.3f}", f"{blink:.3f}", f"{haze_y:.2f}", *L4_PALETTE):
        assert literal in prompt, literal
    assert 6 <= len(houses) <= 7
    assert len({house["h"] for house in houses}) == len(houses), "высоты обязаны различаться"
    assert max(house["h"] for house in houses) >= 12.5, "крыши обязаны подниматься в небо"
    assert all(abs(house["x"]) + house["w"] / 2 <= 14.0 for house in houses), "ряд обязан влезать в кадр"
    assert blink * STEPS / 60 >= 0.6, "фаза окон обязана успеть сместиться на проверке"

    palette = [
        _share("#2a1b5c", L4_PALETTE["#2a1b5c"], "ночное небо"),
        _share("#0b0b14", L4_PALETTE["#0b0b14"], "силуэты домов"),
        _share("#22e0d0", L4_PALETTE["#22e0d0"], "бирюзовые окна"),
        _share("#ffb066", L4_PALETTE["#ffb066"], "тёплые окна"),
    ]
    composition = [
        _l4_sky_brighter(),
        _l4_skyline_steps(),
        _l4_windows_area(),
        _l4_haze_soft(),
    ]
    detail = [
        visual.image_min("edges", 0.02),
        visual.image_min("distinct_colors", 24),
        visual.image_min("changed_fraction", 0.15),
        visual.image_min("saturation", 0.25),
    ]
    return art.build_task(
        task_id="L-4",
        title="L-4 — ночной город",
        prompt=prompt,
        spec=art.ArtSpec(palette=palette, composition=composition, detail=detail, motion=STEPS),
        golden=golden,
        fixture={"houses": houses, "phase": phase, "blink": blink, "haze_y": haze_y},
    )


def _build_l5(rng: random.Random) -> core.TaskInstance:
    arch_r = round(rng.choice([3.0, 3.4, 3.8]), 2)
    omega = round(rng.choice([0.08, 0.10, 0.12]), 3)
    hang = round(arch_r * 0.72, 2)
    arch_h = round(math.sqrt(max(0.0, arch_r * arch_r - hang * hang)), 2)

    prompt = PROMPT_L5.format(
        cam=", ".join(f"{value:.0f}" for value in L5_CAM),
        look=", ".join(f"{value:.1f}" for value in L5_LOOK),
        arch_r=arch_r,
        omega=f"{omega:.3f}",
        arch_h=f"{arch_h:.2f}",
        hang=f"{hang:.2f}",
        steps=STEPS,
        c_bg="#15171f",
        c_teal="#3ce8f0",
        c_warm="#ffc247",
        c_deck="#123f4a",
        c_deck_hi="#1d5a68",
        c_glow="#1d2029",
        c_warm_hi="#ffe08a",
        s_bg=L5_PALETTE["#15171f"],
        s_teal=L5_PALETTE["#3ce8f0"],
        s_warm=L5_PALETTE["#ffc247"],
    )
    golden = GOLDEN_L5.format(
        cam=", ".join(f"{value:.0f}" for value in L5_CAM),
        look=", ".join(f"{value:.1f}" for value in L5_LOOK),
        arch_r=arch_r,
        omega=omega,
        arch_h=arch_h,
        hang=hang,
        c_bg="0x15171f",
        c_teal="0x3ce8f0",
        c_warm="0xffc247",
        c_deck="0x123f4a",
        c_deck_hi="0x1d5a68",
        c_glow="0x1d2029",
        c_warm_hi="0xffe08a",
        c_pylon="0x0f2530",
    )

    for literal in (str(arch_r), f"{omega:.3f}", f"{arch_h:.2f}", f"{hang:.2f}", *L5_PALETTE):
        assert literal in prompt, literal
    assert 2.8 <= arch_r <= 4.0 and arch_r + 0.28 < 9.0, "арка обязана влезать в кадр"
    rotation = math.degrees(omega * STEPS / 60)
    assert rotation >= 5.5, f"поворот за проверку {rotation:.1f}° — не виден"

    palette = [
        _share("#15171f", L5_PALETTE["#15171f"], "фон и земля"),
        _share("#3ce8f0", L5_PALETTE["#3ce8f0"], "рёбра и перила"),
        _share("#ffc247", L5_PALETTE["#ffc247"], "дорожка"),
    ]
    composition = [
        _l5_walkway_centre(),
        _l5_tilted_edges(),
        _l5_saturation_middle(),
        _l5_corners_free(),
    ]
    detail = [
        visual.image_min("edges", 0.011),
        visual.image_min("distinct_colors", 6),
        visual.image_min("changed_fraction", 0.08),
        visual.image_min("saturation", 0.30),
    ]
    return art.build_task(
        task_id="L-5",
        title="L-5 — мост в изометрии",
        prompt=prompt,
        spec=art.ArtSpec(palette=palette, composition=composition, detail=detail, motion=STEPS),
        golden=golden,
        fixture={"arch_r": arch_r, "omega": omega, "arch_h": arch_h, "hang": hang},
    )


def build(rng: random.Random) -> list[core.TaskInstance]:
    return [_build_l2(rng), _build_l3(rng), _build_l4(rng), _build_l5(rng)]
