"""Art scenarios L-13..L-16 — поля, вода и следы.

Четыре сцены, в которых красота картины держится на распределении по кадру, а
не на отдельном объекте: лавовая лампа (поле метаболлов), водная поверхность
(вершинное смещение и блик), поток частиц по аналитическому полю и орбиты со
следями. Промпт объявляет палитру (точные цвета и их минимальные доли кадра),
геометрию (доли кадра), формулу поля и фазы движения — проверка меряет ровно
это по кадру: доля объявленных цветов, профиль яркости, детализация и различие
кадров во времени.

Физика простая: метаболлы — сумма обратных квадратов, волны — три синуса,
поток — поле скоростей от функции тока, орбиты — углы, растущие по периоду.
"""

from __future__ import annotations

import math
import random

from .. import art, core, visual

IDS = ("L-13", "L-14", "L-15", "L-16")
TITLES = {
    "L-13": "L-13 — лавовая лампа",
    "L-14": "L-14 — водная поверхность",
    "L-15": "L-15 — поток частиц",
    "L-16": "L-16 — следы сфер",
}

STEPS = 78  # шагов до второго кадра: 1.3 секунды анимации
FOV = 45.0

# --------------------------------------------------------------------------- #
# L-13 — лавовая лампа: поле метаболлов на плоскости, покрывающей кадр
# --------------------------------------------------------------------------- #

CAM13 = 2.0  # дистанция камеры: плоскость вплотную занимает кадр
#: Палитра: цвет -> минимальная доля кадра. Кадр покрыт одной плоскостью, поэтому
#: разделение идёт по порогам поля, а цвета — тёплым рядом с разрывом по тону.
PALETTE13 = {
    "#16092b": 0.13,  # фон вокруг капель
    "#ff2d6b": 0.15,  # ореол вокруг капель
    "#ff6b1a": 0.15,  # тело капли
    "#ffdf3d": 0.06,  # ядро капли
}
#: Полосы поля: числа объявлены в промпте дословно и стоят в шейдере.
BLOB_RIM = (0.30, 0.62)  # ореол вокруг капли
BLOB_EDGE = (0.85, 1.30)  # поверхность капли
BLOB_CORE = (2.40, 3.20)  # ядро капли

PROMPT13 = """Собери лавовую лампу: тёмный кадр, по нему медленно качаются
капли-метаболлы, сливающиеся силуэтами, — тёплое красно-оранжевое свечение на
тёмно-сливовом фоне.

Кадр целиком занят одной плоскостью `PlaneGeometry` в {cam} единицах от камеры
({fov:.0f}°): полувысота кадра там равна {half} единицы, полуширина — в 4/3
раза больше, поэтому плоскость покрывает кадр ровно и ничего не висит в пустоте.

Поле и капли:
* капель {count}; каждая — центр (x, y) в долях ширины и высоты кадра, радиус r
  в долях высоты кадра, размах A и частота w качания, фаза p:
{rows}
* поле в точке p — сумма вкладов капель: F(p) = Σ r_i² / max(|p − c_i|², 0.0001);
  цвета набираются по полосам поля: ореол — F от {r0} до {r1}, поверхность капли —
  F от {e0} до {e1}, ядро — F от {c0} до {c1}; координаты в шейдере берутся как
  `p = (vUv − 0.5) * vec2(1.3333, 1.0)`, то есть x в долях высоты кадра;
* капля качается только по вертикали: y(t) = y₀ + A · sin(w · t + p), x не
  меняется; на t = 0 капля стоит в (x, y₀);
* капли живут в полосе 0.15..0.85 высоты кадра: верхняя и нижняя полосы кадра —
  фон.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — фон, не меньше {s_bg:.0%} кадра;
* {c_rim} — ореол вокруг капель (полоса F {r0}..{r1}), не меньше {s_rim:.0%} кадра;
* {c_body} — тело капель (полоса F {e0}..{e1}), не меньше {s_body:.0%} кадра;
* {c_core} — ядра капель (полоса F {c0}..{c1}), не меньше {s_core:.0%}.

Анимация:
* `window.__step(dt)` двигает время на dt, пересчитывает y капель и рисует кадр;
* `window.__probe()` возвращает `t`, `objects` и `blobs` — центры капель в долях
  кадра (4 знака).

Проверяется: страница и кадр (5 проверок), доли палитры, капли в средней полосе
кадра, детализация (не плоско и не шум), и то, что сцена движется — кадры на t и
t+{steps} шагов различаются, а каждая загрузка даёт тот же кадр.
"""

GOLDEN13 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>lava lamp</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

// Плоскость в 2 единицах от камеры занимает кадр целиком: полувысота кадра
// tan(22.5°)*2 = 0.8284, полуширина — в 4/3 раза больше.
const COUNT = {count};
const HALF_H = Math.tan(45 * Math.PI / 360) * 2, HALF_W = HALF_H * 4 / 3;
const BLOBS = [
{blob_rows}
];

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 800 / 600, 0.1, 20);
camera.position.set(0, 0, 2);
camera.lookAt(0, 0, 0);

// Центры капель в координатах шейдера: x в долях высоты кадра, y от −0.5 до 0.5.
const centres = BLOBS.map((b) => new THREE.Vector3((b[0] - 0.5) * 4 / 3, b[1] - 0.5, b[2]));

const lamp = new THREE.Mesh(
  new THREE.PlaneGeometry(HALF_W * 2, HALF_H * 2),
  new THREE.ShaderMaterial({{
    uniforms: {{
      uBg: {{value: new THREE.Color({c_bg})}},
      uRim: {{value: new THREE.Color({c_rim})}},
      uBody: {{value: new THREE.Color({c_body})}},
      uCore: {{value: new THREE.Color({c_core})}},
      uBlobs: {{value: centres}},
    }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `#define BLOB_COUNT {count}
      #define RIM_LO 0.30
      #define RIM_HI 0.62
      #define EDGE_LO 0.85
      #define EDGE_HI 1.30
      #define CORE_LO 2.40
      #define CORE_HI 3.20
      uniform vec3 uBg; uniform vec3 uRim; uniform vec3 uBody; uniform vec3 uCore;
      uniform vec3 uBlobs[BLOB_COUNT];
      varying vec2 vUv;
      void main() {{
        // поле метаболлов: сумма r^2/d^2, цвета — по полосам поля
        vec2 p = (vUv - 0.5) * vec2(1.3333, 1.0);
        float field = 0.0;
        for (int i = 0; i < BLOB_COUNT; i++) {{
          vec2 d = p - uBlobs[i].xy;
          field += uBlobs[i].z * uBlobs[i].z / max(dot(d, d), 1e-4);
        }}
        vec3 colour = mix(uBg, uRim, smoothstep(RIM_LO, RIM_HI, field));
        colour = mix(colour, uBody, smoothstep(EDGE_LO, EDGE_HI, field));
        colour = mix(colour, uCore, smoothstep(CORE_LO, CORE_HI, field));
        gl_FragColor = vec4(colour, 1.0);
      }}`,
  }})
);
lamp.name = 'lamp';
scene.add(lamp);

const round = (v) => Math.round(v * 1e4) / 1e4;
let t = 0;
function update() {{
  for (let i = 0; i < COUNT; i++) {{
    const b = BLOBS[i];
    centres[i].set((b[0] - 0.5) * 4 / 3, b[1] - 0.5 + b[3] * Math.sin(b[4] * t + b[5]), b[2]);
  }}
}}

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  blobs: centres.map((v) => [round(v.x / (4 / 3) + 0.5), round(v.y + 0.5)]),
}});
window.__step = (dt) => {{ t += dt; update(); renderer.render(scene, camera); return window.__probe(); }};

update();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""

# --------------------------------------------------------------------------- #
# L-14 — водная поверхность: вершинное смещение, нормаль и солнечный блик
# --------------------------------------------------------------------------- #

#: Полувысота кадра на единичной дистанции: кадр виден от −span до +span по y при
#: дистанции 1 (45° по вертикали).
SPAN14 = math.tan(math.radians(FOV / 2))
HORIZON14 = 0.38  # линия горизонта в долях высоты кадра: отсюда наклон камеры
#: tan угла наклона камеры вниз: горизонт ложится ровно на HORIZON14.
PITCH14 = (0.5 - HORIZON14) * 2 * SPAN14
CAM_Y14 = 0.9  # высота камеры над водой
SKY_DIST14 = 500.0  # дистанция плоскости неба (дальше дальней кромки воды)
SUN_Z14 = 300.0  # солнце между водой и небом
SUN_ANGLE14 = 0.012  # центр солнца над горизонтом в долях высоты кадра
SUN_R14 = 0.05  # радиус солнца в долях высоты кадра
SEA14_NEAR = 0.4  # ближняя кромка воды перед камерой
SEA_LEN14 = 400.0  # и насколько далеко она уходит
#: Волны: направление в плоскости xz, частота, круговая скорость, амплитуда.
WAVES14 = (
    ((0.924, 0.383), 0.38, 0.90, 0.26),
    ((-0.643, 0.766), 0.72, 1.50, 0.16),
    ((0.309, -0.951), 1.25, 2.40, 0.09),
)
PALETTE14 = {
    "#14224f": 0.06,  # холодный верх неба
    "#8c5f86": 0.02,  # сиреневая полоса в середине неба
    "#2f86b0": 0.08,  # вода у горизонта
    "#0b2740": 0.10,  # вода у нижнего края кадра
    "#ffb066": 0.03,  # тёплая полоса неба у горизонта
    "#ffe6b0": 0.01,  # диск солнца и дорожка блика
}

PROMPT14 = """Собери водную поверхность: волны от трёх синусов, пересчитанная
нормаль, тёплый блик от низкого солнца у горизонта и холодное небо.

Камера и горизонт (проверяется по кадру грубо):
* `PerspectiveCamera(45, 800/600, 0.1, 1000)` в точке (0, {cam_y}, 0), наклонена
  вниз так, что линия горизонта — ровно {horizon:.3f} высоты кадра ({horizon:.0%}):
  смотрит в (0, {cam_y} − {pitch}, −1), где {pitch} — тангенс угла наклона;
* полувысота кадра на единичной дистанции равна {span} (45° по вертикали),
  поэтому всё, что ниже горизонта, обязана занимать вода, а выше — небо;
* вода — `PlaneGeometry(300, {sea_len}, 180, 200)` с `rotateX(-PI/2)`: от
  z = +{sea_near} (под камерой) до z = −{sea_far}; дальняя кромка воды приходится
  на горизонт, поэтому кадр ниже горизонта — сплошная вода;
* небо — плоскость на дистанции {sky_dist} перед камерой, высотой от горизонта до
  верхнего края кадра; градиент снизу вверх: {c_warm} у горизонта, {c_mid} на
  середине высоты, {c_sky} у верхнего края — яркость монотонно падает от горизонта
  кверху;
* солнце — диск радиуса {sun_r} высоты кадра в центре по горизонтали, материал
  `MeshBasicMaterial` цвета {c_sun}, дистанция {sun_z}, центр на {sun_angle} высоты
  кадра выше горизонта: водой срезана его нижняя часть.

Волны (вершинный шейдер, только по времени t):
* смещение h(p, t) = Σ a_i · sin(k_i · dot(d_i, p.xz) − w_i · t + p_i), амплитуды
  масштабируются на {wave}; три волны (d — направление в плоскости xz, k —
  частота, w — скорость, a — амплитуда):
{waves}
* нормаль пересчитывается из производных смещения:
  N = normalize(vec3(−∂h/∂x, 1, −∂h/∂z));
* фазы: p_i = {phase} + i · 1.7.

Вода (фрагментный шейдер):
* цвет — от {c_water} у горизонта к {c_deep} у нижнего края кадра (по Френелю:
  чем «скользящее» взгляд, тем светлее вода);
* блик — зеркальное отражение солнца: `pow(max(dot(reflect(−L, N), V), 0.0), 32)`
  плюс тёплая дорожка {c_sun} от берега к горизонту по оси x
  (`exp(−x²/0.35) · smoothstep(−2, −25, z)`), так что под солнцем по центру кадра
  тянется дорожка отражения.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_sky} — холодный верх неба, не меньше {s_sky:.0%} кадра;
* {c_mid} — сиреневая полоса в середине неба, не меньше {s_mid:.0%};
* {c_water} — светлая вода у горизонта, не меньше {s_water:.0%};
* {c_deep} — тёмная вода у нижнего края, не меньше {s_deep:.0%};
* {c_warm} — тёплая полоса неба у горизонта, не меньше {s_warm:.0%};
* {c_sun} — диск солнца и дорожка блика, не меньше {s_sun:.0%}.

Анимация:
* `window.__step(dt)` растит время t, шейдер сдвигает волны, кадр перерисовывается;
* `window.__probe()` возвращает `t`, `objects`, `horizon` и `wave` (время в фазе
  волн, 4 знака).

Проверяется: страница и кадр (5 проверок), палитра по долям, горизонт и профиль
яркости (небо светлее воды, вода темнеет книзу, блик по центру), детализация
кадра, и то, что сцена движется — кадры на t и t+{steps} шагов различаются, а
каждая загрузка даёт тот же кадр.
"""

GOLDEN14 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>water</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const HORIZON = {horizon};
const CAM_Y = {cam_y};
const PITCH = {pitch};
const SPAN = Math.tan(45 * Math.PI / 360);
const SUN_R = {sun_r};
const SUN_ANGLE = {sun_angle};
const SUN_Z = {sun_z};
const SKY_DIST = {sky_dist};
const WAVE = {wave};
const PHASE = {phase};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 800 / 600, 0.1, 1200);
// Наклон вниз ровно настолько, чтобы горизонт лёг на HORIZON высоты кадра.
camera.position.set(0, CAM_Y, 0);
camera.lookAt(0, CAM_Y - PITCH, -1);

// Направление на солнце: над горизонтом, строго по центру кадра.
const sunDir = new THREE.Vector3(0, SUN_ANGLE * 2 * SPAN, -1).normalize();

// Небо: низ плоскости — ровно линия горизонта (y камеры на этой дистанции),
// верх — верхний край кадра.
const SKY_H = (PITCH + SPAN) * SKY_DIST;
const sky = new THREE.Mesh(
  new THREE.PlaneGeometry(3000, SKY_H),
  new THREE.ShaderMaterial({{
    uniforms: {{
      uSky: {{value: new THREE.Color({c_sky})}},
      uMid: {{value: new THREE.Color({c_mid})}},
      uWarm: {{value: new THREE.Color({c_warm})}},
    }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uSky; uniform vec3 uMid; uniform vec3 uWarm; varying vec2 vUv;
      void main() {{
        // vUv.y = 0 — линия горизонта: тёплая полоса у горизонта, сирень выше,
        // холодный верх у верхнего края кадра.
        vec3 colour = mix(uMid, uSky, smoothstep(0.18, 1.0, vUv.y));
        colour = mix(uWarm, colour, smoothstep(0.0, 0.18, vUv.y));
        gl_FragColor = vec4(colour, 1.0);
      }}`,
  }})
);
sky.name = 'sky';
sky.position.set(0, CAM_Y + SKY_H / 2, -SKY_DIST);
scene.add(sky);

// Вода: смещение тремя синусами, нормаль — из производных смещения.
const SEA_LEN = {sea_len}, SEA_NEAR = {sea_near};
const sea = new THREE.Mesh(
  new THREE.PlaneGeometry(300, SEA_LEN, 180, 200).rotateX(-Math.PI / 2),
  new THREE.ShaderMaterial({{
    uniforms: {{
      uTime: {{value: 0.0}},
      uWave: {{value: WAVE}},
      uDeep: {{value: new THREE.Color({c_deep})}},
      uWater: {{value: new THREE.Color({c_water})}},
      uGlint: {{value: new THREE.Color({c_sun})}},
      uSunDir: {{value: sunDir}},
    }},
    vertexShader: `uniform float uTime; uniform float uWave;
      varying vec3 vWorld; varying vec3 vN;
      void main() {{
        vec2 D1 = vec2(0.924, 0.383), D2 = vec2(-0.643, 0.766), D3 = vec2(0.309, -0.951);
        float p1 = PH, p2 = PH + 1.7, p3 = PH + 3.4;
        vec3 p = position;
        float h = 0.0, dhx = 0.0, dhz = 0.0, c;
        c = 0.26 * uWave * cos(0.38 * dot(D1, p.xz) - 0.90 * uTime + p1);
        h += 0.26 * uWave * sin(0.38 * dot(D1, p.xz) - 0.90 * uTime + p1); dhx += c * 0.38 * D1.x; dhz += c * 0.38 * D1.y;
        c = 0.16 * uWave * cos(0.72 * dot(D2, p.xz) - 1.50 * uTime + p2);
        h += 0.16 * uWave * sin(0.72 * dot(D2, p.xz) - 1.50 * uTime + p2); dhx += c * 0.72 * D2.x; dhz += c * 0.72 * D2.y;
        c = 0.09 * uWave * cos(1.25 * dot(D3, p.xz) - 2.40 * uTime + p3);
        h += 0.09 * uWave * sin(1.25 * dot(D3, p.xz) - 2.40 * uTime + p3); dhx += c * 1.25 * D3.x; dhz += c * 1.25 * D3.y;
        p.y += h;
        vN = normalize(vec3(-dhx, 1.0, -dhz));
        vec4 world = modelMatrix * vec4(p, 1.0);
        vWorld = world.xyz;
        gl_Position = projectionMatrix * viewMatrix * world;
      }}`.split('PH').join(PHASE.toFixed(3)),
    fragmentShader: `uniform vec3 uDeep; uniform vec3 uWater; uniform vec3 uGlint; uniform vec3 uSunDir;
      varying vec3 vWorld; varying vec3 vN;
      void main() {{
        vec3 N = normalize(vN);
        vec3 V = normalize(cameraPosition - vWorld);
        vec3 L = normalize(uSunDir);
        float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        vec3 colour = mix(uDeep, uWater, fres);
        float spec = pow(max(dot(reflect(-L, N), V), 0.0), 32.0);
        // Дорожка: узкая полоса по x, живущая на дальней воде под солнцем.
        float column = exp(-(vWorld.x * vWorld.x) / 0.35) * smoothstep(-2.0, -25.0, vWorld.z);
        colour = mix(colour, uGlint, clamp(0.85 * spec + 0.75 * column, 0.0, 0.9));
        gl_FragColor = vec4(colour, 1.0);
      }}`,
  }})
);
sea.name = 'sea';
sea.position.set(0, 0, SEA_NEAR - SEA_LEN / 2);
scene.add(sea);

// Солнце: диск у горизонта, впереди неба и позади ближней воды.
const sun = new THREE.Mesh(
  new THREE.CircleGeometry(SUN_R * 2 * SPAN * SUN_Z, 48),
  new THREE.MeshBasicMaterial({{color: {c_sun}}})
);
sun.name = 'sun';
sun.position.set(0, CAM_Y + SUN_ANGLE * 2 * SPAN * SUN_Z, -SUN_Z);
scene.add(sun);

const round = (v) => Math.round(v * 1e4) / 1e4;
let t = 0;
window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  horizon: HORIZON,
  wave: round(t * 0.9 + PHASE),
}});
window.__step = (dt) => {{ t += dt; sea.material.uniforms.uTime.value = t; renderer.render(scene, camera); return window.__probe(); }};

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""

# --------------------------------------------------------------------------- #
# L-15 — поток: поле скоростей от функции тока, штрихи-частицы
# --------------------------------------------------------------------------- #

COUNT15 = 1800
PALETTE15 = {
    "#0b1424": 0.40,  # фон
    "#28d7e8": 0.004,  # штрихи поперёк, холодный
    "#e8399b": 0.004,  # штрихи вдоль, тёплый
    "#f2c53d": 0.004,  # штрихи наискось, золото
}
COLOURS15 = ("#28d7e8", "#e8399b", "#f2c53d")
FIELD15 = (
    # амплитуда, k по x, m по y, круговая скорость: ψ = Σ a·sin(k·x + m·y − w·t + p)
    (0.50, 1.40, 0.90, 0.70),
    (0.35, 0.80, -1.70, 1.10),
)
SPEED15 = 0.18  # долей кадра в секунду при поле единичной величины
LIFE15 = 2.6  # секунд жизни штриха

PROMPT15 = """Собери поток: {count} частиц-штрихов, которые текут по
аналитическому полю скоростей, и след затухает по возрасту частицы.

Камера ортографическая, кадр 1920x1080: `OrthographicCamera(-4/3, 4/3, 1, -1, 0.1, 10)`
в (0, 0, 2), смотрит в начало координат; единица сцены — половина высоты кадра,
то есть y от −1 до 1, x от −4/3 до 4/3.

Поле (объявлено формулой, а не картинкой):
* функция тока ψ(x, y, t) = Σ a_i · sin(k_i · x + m_i · y − w_i · t + p_i), две
  волны: (a, k, m, w) = {f1} и {f2}; фазы p_i = {phase} + i · 2.1;
* скорость — вихрь функции тока: v = (∂ψ/∂y, −∂ψ/∂x), то есть
  v = Σ a_i · cos(k_i·x + m_i·y − w_i·t + p_i) · (m_i, −k_i);
* за шаг частица сдвигается на v · {speed} · dt.

Частицы и следы:
* частиц {count}; штрих — `InstancedMesh` из `PlaneGeometry(1, 1)`, масштаб
  экземпляра (0.010, длина, 1), поворот вокруг z на угол направления;
* частица рождается по LCG: seed = ({a} · seed + {c}) mod 2^32,
  x = (seed / 2^32 · 2 − 1) · 4/3, y = seed / 2^32 · 2 − 1; начальный seed = {seed};
* длина и яркость штриха падают с возрастом: длина от 0.090 до 0.040, яркость от
  1.0 до 0.25; возраст растёт на dt / {life} и, дойдя до 1 (или при выходе за
  кадр), частица рождается заново;
* цвет штриха — по направлению скорости, три сектора по углу atan2(vy, vx):
  −180..−60° — {c_a}, −60..60° — {c_b}, 60..180° — {c_c}.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — фон, не меньше {s_bg:.0%} кадра;
* {c_a} — штрихи одного направления, не меньше {s_a:.0%};
* {c_b} — штрихи второго направления, не меньше {s_b:.0%};
* {c_c} — штрихи третьего направления, не меньше {s_c:.0%}.

Анимация:
* `window.__step(dt)` двигает все частицы по полю, обновляет цвета и длины и
  рисует кадр;
* `window.__probe()` возвращает `t`, `objects`, `particles` (первые 8 частиц
  [x, y] в долях кадра, 4 знака) и `speed`.

Проверяется: страница и кадр (5 проверок), доли палитры, заполнение кадра
(штрихи есть в каждой полосе), детализация, и то, что поток течёт — кадры на t и
t+{steps} шагов различаются, а каждая загрузка даёт тот же кадр.
"""

GOLDEN15 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>flow</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const COUNT = {count};
const SPEED = {speed};
const LIFE = {life};
const PHASE = {phase};
const COLOURS = [{c_a}, {c_b}, {c_c}];
const A1 = {a1}, K1 = {k1}, M1 = {m1}, W1 = {w1};
const A2 = {a2}, K2 = {k2}, M2 = {m2}, W2 = {w2};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-4 / 3, 4 / 3, 1, -1, 0.1, 10);
camera.position.set(0, 0, 2);
camera.lookAt(0, 0, 0);

// Фон: две тёмные синие полосы, чтобы кадр не был заливкой.
const back = new THREE.Mesh(
  new THREE.PlaneGeometry(4 / 3 * 2, 2),
  new THREE.ShaderMaterial({{
    uniforms: {{ uTop: {{value: new THREE.Color({c_bg})}}, uBottom: {{value: new THREE.Color(0x050a14)}} }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    fragmentShader: `uniform vec3 uTop; uniform vec3 uBottom; varying vec2 vUv;
      void main() {{ gl_FragColor = vec4(mix(uBottom, uTop, vUv.y), 1.0); }}`,
  }})
);
back.name = 'back';
back.position.z = -0.5;
scene.add(back);

const strokes = new THREE.InstancedMesh(
  new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({{color: 0xffffff}}),
  COUNT
);
strokes.name = 'strokes';
scene.add(strokes);

// Поле скоростей: вихрь функции тока ψ = Σ a·sin(k·x + m·y − w·t + p).
function field(x, y, t) {{
  const c1 = A1 * Math.cos(K1 * x + M1 * y - W1 * t + PHASE);
  const c2 = A2 * Math.cos(K2 * x + M2 * y - W2 * t + PHASE + 2.1);
  return [c1 * M1 + c2 * M2, -(c1 * K1 + c2 * K2)];
}}

const A = 1664525, C = 1013904223, MOD = 4294967296;
let seed = {seed};
function born(i) {{
  seed = (A * seed + C) % MOD;
  const u = seed / MOD;
  seed = (A * seed + C) % MOD;
  pos[i * 2] = (u * 2 - 1) * 4 / 3;
  pos[i * 2 + 1] = seed / MOD * 2 - 1;
  age[i] = 0;
}}

const pos = new Float32Array(COUNT * 2);
const age = new Float32Array(COUNT);
const colour = new THREE.Color();
const mat = new THREE.Matrix4();
const quat = new THREE.Quaternion();
const vec = new THREE.Vector3();
const scale = new THREE.Vector3();
const axis = new THREE.Vector3(0, 0, 1);
for (let i = 0; i < COUNT; i++) {{ born(i); age[i] = i / COUNT; }}

let t = 0;
function update(dt) {{
  t += dt;
  for (let i = 0; i < COUNT; i++) {{
    let x = pos[i * 2], y = pos[i * 2 + 1];
    const v = field(x, y, t);
    x += v[0] * SPEED * dt;
    y += v[1] * SPEED * dt;
    age[i] += dt / LIFE;
    if (age[i] >= 1 || Math.abs(x) > 1.40 || Math.abs(y) > 1.05) {{ born(i); x = pos[i * 2]; y = pos[i * 2 + 1]; }}
    pos[i * 2] = x;
    pos[i * 2 + 1] = y;
    const a = age[i], len = 0.090 - 0.050 * a;
    const ang = Math.atan2(v[1], v[0]) - Math.PI / 2;
    quat.setFromAxisAngle(axis, ang);
    vec.set(x, y, 0);
    scale.set(0.010, len, 1);
    mat.compose(vec, quat, scale);
    strokes.setMatrixAt(i, mat);
    const sector = Math.floor(((Math.atan2(v[1], v[0]) + Math.PI) / (2 * Math.PI)) * 3) % 3;
    colour.set(COLOURS[sector]).multiplyScalar(1.0 - 0.75 * a);
    strokes.setColorAt(i, colour);
  }}
  strokes.instanceMatrix.needsUpdate = true;
  if (strokes.instanceColor) strokes.instanceColor.needsUpdate = true;
}}

window.__probe = () => {{
  const first = [];
  for (let i = 0; i < 8; i++) first.push([round(pos[i * 2] / (4 / 3) / 2 + 0.5), round(pos[i * 2 + 1] / 2 + 0.5)]);
  return {{ t: round(t), objects: scene.children.map((o) => o.name).filter(Boolean), particles: first, speed: SPEED }};
}};
function round(v) {{ return Math.round(v * 1e4) / 1e4; }}
window.__step = (dt) => {{ update(dt); renderer.render(scene, camera); return window.__probe(); }};

update(0);
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""

# --------------------------------------------------------------------------- #
# L-16 — следы сфер: орбиты с наклонением и затухающим хвостом
# --------------------------------------------------------------------------- #

HALF16 = math.tan(math.radians(FOV / 2)) * 4.0  # полувысота кадра на плоскости сцены
CAM16 = 4.0
TRAIL16 = 60  # точек в хвосте
PALETTE16 = {
    "#080f22": 0.40,  # фон
    "#ff4d6d": 0.0015,  # первая сфера и её хвост
    "#ffd166": 0.002,  # вторая
    "#4dd0e1": 0.0015,  # третья
    "#9b6dff": 0.0015,  # четвёртая
}
COLOURS16 = ("#ff4d6d", "#ffd166", "#4dd0e1", "#9b6dff")
CORE16 = "#ffe9a8"

PROMPT16 = """Собери орбиты: несколько светящихся шаров ходят по наклонным
орбитам вокруг яркого центра, за каждым тянется затухающий хвост.

Камера `PerspectiveCamera(45, 800/600, 0.1, 100)` в (0, 0, {cam}) смотрит в
начало координат; полувысота кадра на плоскости сцены — {half} единицы, поэтому
доля кадра равна координате, делённой на {pu}.

Орбиты:
* сфер {count}; у каждой: радиус орбиты a в долях высоты кадра, наклонение inc
  (радианы, поворот орбиты вокруг оси x), начальный угол t0, период T в секундах
  и цвет:
{rows}
* положение: p = a · {pu}/2 · (cos θ, sin θ · cos inc, sin θ · sin inc), θ(t) = t0 + 2π · t / T;
* на t = 0 сфера стоит в θ = t0;
* хвост — цепочка точек (`Points`) из {trail} светящихся спрайтов: яркость падает
  от 1.0 у сферы до 0.06 на конце; точка хвоста на t = 0 — положение сферы в
  момент −i · dt (шаг {dstep} с);
* центр — диск радиуса {core_r} высоты кадра с радиальным градиентом от {c_core} в
  центре к фону — компактное пятно, а не заливка;
* фон — плоскость во весь кадр ровного тёмного цвета {c_bg}: углы кадра не должны
  быть чёрной пустотой.

Палитра (проверяется по кадру, доли — минимум кадра):
* {c_bg} — фон, не меньше {s_bg:.0%} кадра;
* {c_1} — первая сфера и её хвост, не меньше {s_1:.1%};
* {c_2} — вторая, не меньше {s_2:.1%};
* {c_3} — третья, не меньше {s_3:.1%};
* {c_4} — четвёртая, не меньше {s_4:.1%}.

Анимация:
* `window.__step(dt)` растит угол каждой сферы на 2π · dt / T, сдвигает хвост и
  рисует кадр;
* `window.__probe()` возвращает `t`, `objects`, `orbits` (углы в радианах, 4
  знака) и `planets` (положения в долях кадра).

Проверяется: страница и кадр (5 проверок), доли палитры, центр кадра ярче краёв,
детализация (хвосты читаются, кадр не пустой), и то, что орбиты движутся — кадры
на t и t+{steps} шагов различаются, а каждая загрузка даёт тот же кадр.
"""

GOLDEN16 = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>orbits</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const COUNT = {count};
const TRAIL = {trail};
const HALF = Math.tan(45 * Math.PI / 360) * 4, PU = 2 * HALF;
const PLANETS = [
{rows}
];

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 800 / 600, 0.1, 100);
camera.position.set(0, 0, 4);
camera.lookAt(0, 0, 0);

// Фон: плоскость во весь кадр — ровный тёмный цвет без чёрных углов.
const back = new THREE.Mesh(
  new THREE.PlaneGeometry(2 * PU * 1.4, 2 * PU * 1.4),
  new THREE.MeshBasicMaterial({{color: {c_bg}}})
);
back.name = 'back';
back.position.z = -0.5;
scene.add(back);

// Центр: компактное пятно с радиальным градиентом от {c_core} к фону.
const core = new THREE.Mesh(
  new THREE.CircleGeometry(0.22 * PU, 64),
  new THREE.ShaderMaterial({{
    uniforms: {{ uCore: {{value: new THREE.Color({c_core})}}, uBg: {{value: new THREE.Color({c_bg})}} }},
    vertexShader: `varying vec2 vUv;
      void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    fragmentShader: `uniform vec3 uCore; uniform vec3 uBg; varying vec2 vUv;
      void main() {{
        float r = length(vUv - 0.5) * 2.0;
        gl_FragColor = vec4(uCore, 1.0 - smoothstep(0.0, 0.9, r));
      }}`,
  }})
);
core.name = 'core';
core.position.z = -0.2;
scene.add(core);

const point = (p, theta) => new THREE.Vector3(
  p[0] * PU / 2 * Math.cos(theta),
  p[0] * PU / 2 * Math.sin(theta) * Math.cos(p[1]),
  p[0] * PU / 2 * Math.sin(theta) * Math.sin(p[1])
);
const thetaOf = (p, t) => p[2] + 2 * Math.PI * t / p[3];
const fadeOf = (k) => 1.0 - 0.94 * (k / (TRAIL - 1));

// Хвост — цепочка светящихся спрайтов: Line тоньше пикселя и на кадре почти
// не читается, а спрайты дают и ширину, и мягкое затухание.
const dot = document.createElement('canvas');
dot.width = dot.height = 32;
const dctx = dot.getContext('2d');
const dgrad = dctx.createRadialGradient(16, 16, 0, 16, 16, 16);
dgrad.addColorStop(0.0, 'rgba(255,255,255,0.8)');
dgrad.addColorStop(0.5, 'rgba(255,255,255,0.4)');
dgrad.addColorStop(1.0, 'rgba(255,255,255,0)');
dctx.fillStyle = dgrad;
dctx.fillRect(0, 0, 32, 32);
const sprite = new THREE.CanvasTexture(dot);

const balls = [];
const tails = [];
for (let i = 0; i < COUNT; i++) {{
  const p = PLANETS[i];
  const colour = new THREE.Color(p[4]);
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.030 * PU, 20, 12),
    new THREE.MeshBasicMaterial({{color: colour}})
  );
  ball.name = 'ball' + i;
  scene.add(ball);
  balls.push(ball);

  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(TRAIL * 3);
  const colours = new Float32Array(TRAIL * 3);
  for (let k = 0; k < TRAIL; k++) {{
    const fade = fadeOf(k);
    colours[k * 3] = colour.r * fade; colours[k * 3 + 1] = colour.g * fade; colours[k * 3 + 2] = colour.b * fade;
  }}
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colours, 3));
  const tail = new THREE.Points(geo, new THREE.PointsMaterial({{
    size: 0.038 * PU, map: sprite, vertexColors: true,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
  }}));
  tail.name = 'tail' + i;
  scene.add(tail);
  tails.push({{geo: geo, positions: positions}});
}}

const round = (v) => Math.round(v * 1e4) / 1e4;
let t = 0;
function update() {{
  for (let i = 0; i < COUNT; i++) {{
    const p = PLANETS[i];
    const head = point(p, thetaOf(p, t));
    balls[i].position.copy(head);
    const buf = tails[i].positions, omega = 2 * Math.PI / p[3];
    buf.copyWithin(3, 0, (TRAIL - 1) * 3);
    buf[0] = head.x; buf[1] = head.y; buf[2] = head.z;
    tails[i].geo.attributes.position.needsUpdate = true;
  }}
}}

// Хвост на t = 0: точки траектории назад по времени.
for (let i = 0; i < COUNT; i++) {{
  const p = PLANETS[i], buf = tails[i].positions;
  for (let k = 0; k < TRAIL; k++) {{
    const q = point(p, thetaOf(p, -k / 60));
    buf[k * 3] = q.x; buf[k * 3 + 1] = q.y; buf[k * 3 + 2] = q.z;
  }}
  tails[i].geo.attributes.position.needsUpdate = true;
}}

window.__probe = () => ({{
  t: round(t),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  orbits: PLANETS.map((p) => round(thetaOf(p, t))),
  planets: PLANETS.map((p) => {{
    const q = point(p, thetaOf(p, t));
    return [round(q.x / PU * 2 + 0.5), round(0.5 - q.y / PU * 2)];
  }}),
}});
window.__step = (dt) => {{ t += dt; update(); renderer.render(scene, camera); return window.__probe(); }};

update();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Предикаты по кадру
# --------------------------------------------------------------------------- #


def _profile(pixels: dict) -> list[float]:
    return [float(entry["lum"]) for entry in (pixels.get("profile") or [])]


def _mean_lum(color) -> float:
    return sum(float(v) for v in color) / 3.0


def _lamp_blobs_in_middle():
    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        middle = (profile[3] + profile[4]) / 2
        edge = (profile[0] + profile[1] + profile[6] + profile[7]) / 4
        good = middle > edge + 4.0
        return good, f"кадр: капли в средней полосе — середина {middle:.1f} против краёв {edge:.1f}"

    return visual._label(check, "кадр: капли держатся в средней полосе кадра")


def _lamp_warm_centroid():
    def check(pixels: dict) -> tuple[bool, str]:
        warm = pixels.get("warm") or {}
        centroid = warm.get("centroid")
        if not centroid:
            return False, "кадр: тёплых пикселей не нашлось"
        x, y = float(centroid[0]), float(centroid[1])
        good = warm.get("fraction", 0) >= 0.05 and 0.25 <= x <= 0.75 and 0.2 <= y <= 0.8
        return good, f"кадр: свечение распределено по кадру — центр ({x:.2f}, {y:.2f}), доля {warm.get('fraction')}"

    return visual._label(check, "кадр: свечение в центре кадра, а не в углу")


def _sky_over_water():
    def check(pixels: dict) -> tuple[bool, str]:
        regions = pixels.get("regions") or []
        if len(regions) < 9:
            return False, "кадр: нет областей"
        # Кадр симметричен ровно по центру: дорожка блика стоит в средней колонке,
        # поэтому горизонт читается по боковым областям — светлая полоса у
        # горизонта и тёмная вода под ней.
        top = (_mean_lum(regions[0]) + _mean_lum(regions[2])) / 2
        horizon = (_mean_lum(regions[3]) + _mean_lum(regions[5])) / 2
        bottom = (_mean_lum(regions[6]) + _mean_lum(regions[8])) / 2
        # Либо у горизонта светлее всего, либо полоса неба над водой явно светлее
        # воды у нижнего края: на слабом контрасте хватает второго признака.
        good = (horizon > top + 4.0 and horizon > bottom + 8.0) or top > bottom + 15.0
        return good, (
            f"кадр: горизонт читается — у горизонта {horizon:.0f}, "
            f"небо {top:.0f}, вода у нижнего края {bottom:.0f}"
        )

    return visual._label(check, "кадр: горизонт читается (полоса у горизонта светлее неба и воды)")


def _water_sky_gradient():
    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        # Горизонт объявлен в 0.34..0.42: три верхние полосы — чистое небо.
        top = profile[:3]
        good = top[-1] > top[0] + 0.1 * top[0] and sum(1 for a, b in zip(top, top[1:]) if b >= a - 1.0) >= 1
        return good, f"кадр: небо светлеет к горизонту — профиль {[round(v, 1) for v in top]}"

    return visual._label(check, "кадр: градиент неба к горизонту")


def _glint_column():
    def check(pixels: dict) -> tuple[bool, str]:
        regions = pixels.get("regions") or []
        if len(regions) < 9:
            return False, "кадр: нет областей"
        # Средняя строка областей — вода сразу под горизонтом: там живёт дорожка.
        left, centre, right = (_mean_lum(regions[3]), _mean_lum(regions[4]), _mean_lum(regions[5]))
        good = centre > left + 6.0 and centre > right + 6.0
        return good, f"кадр: дорожка блика по центру — {centre:.0f} против {left:.0f}/{right:.0f}"

    return visual._label(check, "кадр: блик идёт по центру кадра")


def _flow_fills_frame():
    def check(pixels: dict) -> tuple[bool, str]:
        profile = _profile(pixels)
        if len(profile) < 8:
            return False, "кадр: нет профиля яркости"
        darkest = min(profile)
        good = darkest >= 4.0
        return good, f"кадр: штрихи есть в каждой полосе — минимальная яркость полосы {darkest:.1f}"

    return visual._label(check, "кадр: поток покрывает все полосы кадра")


def _flow_not_mirrored():
    def check(pixels: dict) -> tuple[bool, str]:
        difference = pixels.get("mirror_difference")
        if not isinstance(difference, (int, float)):
            return False, "кадр: нет метрики зеркала"
        good = float(difference) >= 6.0
        return good, f"кадр: поток не симметричен — зеркальная разница {difference} (нужно >= 6)"

    return visual._label(check, "кадр: поток асимметричен, а не зеркало")


def _core_brighter():
    def check(pixels: dict) -> tuple[bool, str]:
        regions = pixels.get("regions") or []
        if len(regions) < 9:
            return False, "кадр: нет областей"
        centre = _mean_lum(regions[4])
        corners = [_mean_lum(regions[i]) for i in (0, 2, 6, 8)]
        edge = sum(corners) / len(corners)
        good = centre > edge + 8.0
        return good, f"кадр: центр ярче краёв — {centre:.0f} против {edge:.0f}"

    return visual._label(check, "кадр: центр кадра ярче краёв")


def _orbits_spread():
    def check(pixels: dict) -> tuple[bool, str]:
        bright = pixels.get("bright") or {}
        centroid = bright.get("centroid")
        if not centroid:
            return False, "кадр: ярких пикселей не нашлось"
        x, y = float(centroid[0]), float(centroid[1])
        good = bright.get("fraction", 0.0) >= 0.004 and 0.3 <= x <= 0.7 and 0.3 <= y <= 0.7
        return good, f"кадр: светящиеся тела вокруг центра — ({x:.2f}, {y:.2f}), доля {bright.get('fraction')}"

    return visual._label(check, "кадр: сферы и хвосты держатся вокруг центра")


def _frame_share(hex_value: str, share: float, label: str):
    return visual.palette_colour(hex_value, share, label)


# --------------------------------------------------------------------------- #
# Сборка заданий
# --------------------------------------------------------------------------- #


def _task_lava(rng: random.Random) -> core.TaskInstance:
    count = rng.randrange(6, 10)
    blobs = [
        (
            round(rng.uniform(0.10, 0.90), 3),
            round(rng.uniform(0.28, 0.72), 3),
            round(rng.uniform(0.070, 0.105), 3),
            round(rng.uniform(0.10, 0.18), 3),
            round(rng.uniform(0.45, 1.15), 3),
            round(rng.uniform(0.0, 6.28), 2),
        )
        for _ in range(count)
    ]
    while len({(b[0], b[1]) for b in blobs}) < count:
        blobs = [(round(rng.uniform(0.10, 0.90), 3), round(rng.uniform(0.28, 0.72), 3), *b[2:]) for b in blobs]
        break

    prompt = PROMPT13.format(
        cam=CAM13,
        fov=FOV,
        half=round(math.tan(math.radians(FOV / 2)) * CAM13, 4),
        count=count,
        rows="\n".join(
            f"* капля {i + 1}: центр ({b[0]:.3f}, {b[1]:.3f}) кадра, радиус {b[2]:.3f}, "
            f"размах {b[3]:.3f}, частота {b[4]:.3f}, фаза {b[5]:.2f};"
            for i, b in enumerate(blobs)
        ),
        r0=BLOB_RIM[0], r1=BLOB_RIM[1],
        e0=BLOB_EDGE[0], e1=BLOB_EDGE[1],
        c0=BLOB_CORE[0], c1=BLOB_CORE[1],
        c_bg="#16092b",
        c_rim="#ff2d6b",
        c_body="#ff6b1a",
        c_core="#ffdf3d",
        s_bg=PALETTE13["#16092b"],
        s_rim=PALETTE13["#ff2d6b"],
        s_body=PALETTE13["#ff6b1a"],
        s_core=PALETTE13["#ffdf3d"],
        steps=STEPS,
    )
    golden = GOLDEN13.format(
        count=count,
        blob_rows="\n".join(
            f"  [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}, {b[4]:.3f}, {b[5]:.2f}]," for b in blobs
        ),
        c_bg="0x16092b",
        c_rim="0xff2d6b",
        c_body="0xff6b1a",
        c_core="0xffdf3d",
    )

    for blob in blobs:
        for number in (f"{blob[0]:.3f}", f"{blob[1]:.3f}", f"{blob[2]:.3f}", f"{blob[3]:.3f}", f"{blob[4]:.3f}", f"{blob[5]:.2f}"):
            assert number in prompt, number
    for literal in ("#16092b", "#ff2d6b", "#ff6b1a", "#ffdf3d"):
        assert literal in prompt, literal
    assert count in range(6, 10)
    # Кадр на 78 шагов обязан отличаться: размах качания даёт заметный сдвиг.
    assert min(b[3] for b in blobs) * STEPS / 60 > 0.08

    palette_checks = [
        _frame_share("#16092b", PALETTE13["#16092b"], "фон лампы"),
        _frame_share("#ff2d6b", PALETTE13["#ff2d6b"], "ореол капель"),
        _frame_share("#ff6b1a", PALETTE13["#ff6b1a"], "тело капель"),
        _frame_share("#ffdf3d", PALETTE13["#ffdf3d"], "ядро капель"),
    ]
    composition_checks = [
        _lamp_blobs_in_middle(),
        _lamp_warm_centroid(),
    ]
    detail_checks = [
        visual.image_min("changed_fraction", 0.12),
        visual.image_min("distinct_colors", 24),
        visual.rich_edges(0.002, "силуэты капель читаются, фон не залит"),
        visual.image_min("saturation", 0.30),
    ]
    return art.build_task(
        task_id="L-13",
        title=TITLES["L-13"],
        prompt=prompt,
        spec=art.ArtSpec(palette=palette_checks, composition=composition_checks, detail=detail_checks, motion=STEPS),
        golden=golden,
        fixture={
            "count": count,
            "blobs": [list(b) for b in blobs],
            "bands": [list(BLOB_RIM), list(BLOB_EDGE), list(BLOB_CORE)],
        },
    )


def _task_water(rng: random.Random) -> core.TaskInstance:
    horizon = round(rng.uniform(0.34, 0.42), 3)
    wave = round(rng.uniform(0.80, 1.20), 3)
    phase = round(rng.uniform(0.0, 6.28), 3)
    pitch = round((0.5 - horizon) * 2 * SPAN14, 4)

    prompt = PROMPT14.format(
        horizon=horizon,
        cam_y=CAM_Y14,
        pitch=f"{pitch:.4f}",
        span=round(SPAN14, 4),
        sun_r=SUN_R14,
        sun_angle=SUN_ANGLE14,
        sun_z=SUN_Z14,
        sky_dist=SKY_DIST14,
        sea_near=SEA14_NEAR,
        sea_len=SEA_LEN14,
        sea_far=SEA_LEN14 - SEA14_NEAR,
        c_sky="#14224f",
        c_mid="#8c5f86",
        c_water="#2f86b0",
        c_deep="#0b2740",
        c_warm="#ffb066",
        c_sun="#ffe6b0",
        s_sky=PALETTE14["#14224f"],
        s_mid=PALETTE14["#8c5f86"],
        s_water=PALETTE14["#2f86b0"],
        s_deep=PALETTE14["#0b2740"],
        s_warm=PALETTE14["#ffb066"],
        s_sun=PALETTE14["#ffe6b0"],
        wave=f"{wave:.3f}",
        phase=f"{phase:.3f}",
        waves="\n".join(
            f"* волна {i + 1}: d = ({d[0]:.3f}, {d[1]:.3f}), k = {k:.2f}, w = {w:.2f}, a = {a:.2f};"
            for i, (d, k, w, a) in enumerate(WAVES14)
        ),
        steps=STEPS,
    )
    golden = GOLDEN14.format(
        horizon=horizon,
        cam_y=CAM_Y14,
        pitch=f"{pitch:.4f}",
        sun_r=SUN_R14,
        sun_angle=SUN_ANGLE14,
        sun_z=SUN_Z14,
        sky_dist=SKY_DIST14,
        sea_near=SEA14_NEAR,
        sea_len=SEA_LEN14,
        wave=f"{wave:.3f}",
        phase=f"{phase:.3f}",
        c_sky="0x14224f",
        c_mid="0x8c5f86",
        c_water="0x2f86b0",
        c_deep="0x0b2740",
        c_warm="0xffb066",
        c_sun="0xffe6b0",
    )

    for literal in (
        f"{horizon:.3f}",
        str(CAM_Y14),
        f"{pitch:.4f}",
        str(SUN_R14),
        str(SUN_ANGLE14),
        f"{wave:.3f}",
        f"{phase:.3f}",
        "#14224f",
        "#8c5f86",
        "#2f86b0",
        "#0b2740",
        "#ffb066",
        "#ffe6b0",
    ):
        assert literal in prompt, literal
    for (d, k, w, a) in WAVES14:
        for number in (f"{d[0]:.3f}", f"{d[1]:.3f}", f"{k:.2f}", f"{w:.2f}", f"{a:.2f}"):
            assert number in prompt, number
    assert 0.3 <= horizon <= 0.45
    # Волны обязаны уехать за 78 шагов: фронт идёт со скоростью w/k единиц в секунду.
    assert min(w / k for _, k, w, _ in WAVES14) * STEPS / 60 > 2.0
    # Дальняя кромка воды обязана уйти за горизонт: иначе под ним чёрная полоса.
    assert SEA_LEN14 - SEA14_NEAR > 50.0

    palette_checks = [
        _frame_share("#14224f", PALETTE14["#14224f"], "небо вверху"),
        _frame_share("#8c5f86", PALETTE14["#8c5f86"], "сиреневая полоса неба"),
        _frame_share("#2f86b0", PALETTE14["#2f86b0"], "светлая вода у горизонта"),
        _frame_share("#0b2740", PALETTE14["#0b2740"], "тёмная вода у нижнего края"),
        _frame_share("#ffb066", PALETTE14["#ffb066"], "тёплая полоса неба"),
        _frame_share("#ffe6b0", PALETTE14["#ffe6b0"], "солнце и дорожка"),
    ]
    composition_checks = [
        _water_sky_gradient(),
        _sky_over_water(),
        _glint_column(),
    ]
    detail_checks = [
        visual.image_min("changed_fraction", 0.4),
        visual.image_min("distinct_colors", 32),
        visual.rich_edges(0.004, "волны и блик читаются, а не заливка"),
        visual.image_min("saturation", 0.30),
    ]
    return art.build_task(
        task_id="L-14",
        title=TITLES["L-14"],
        prompt=prompt,
        spec=art.ArtSpec(palette=palette_checks, composition=composition_checks, detail=detail_checks, motion=STEPS),
        golden=golden,
        fixture={"horizon": horizon, "wave": wave, "phase": phase, "pitch": pitch, "cam_y": CAM_Y14},
    )


def _task_flow(rng: random.Random) -> core.TaskInstance:
    count = int(rng.choice([1500, 1800, 2100]))
    phase = round(rng.uniform(0.0, 6.28), 3)
    seed = int(rng.randrange(1, 1 << 31))
    waves = [tuple(row) for row in FIELD15]

    prompt = PROMPT15.format(
        count=count,
        f1="(a, k, m, w) = (%.2f, %.2f, %.2f, %.2f)" % waves[0],
        f2="(a, k, m, w) = (%.2f, %.2f, %.2f, %.2f)" % waves[1],
        phase=f"{phase:.3f}",
        speed=SPEED15,
        life=LIFE15,
        seed=seed,
        a=1664525,
        c=1013904223,
        c_bg="#0b1424",
        c_a="#28d7e8",
        c_b="#e8399b",
        c_c="#f2c53d",
        s_bg=PALETTE15["#0b1424"],
        s_a=PALETTE15["#28d7e8"],
        s_b=PALETTE15["#e8399b"],
        s_c=PALETTE15["#f2c53d"],
        steps=STEPS,
    )
    golden = GOLDEN15.format(
        count=count,
        speed=SPEED15,
        life=LIFE15,
        phase=f"{phase:.3f}",
        seed=seed,
        c_bg="0x0b1424",
        c_a="0x28d7e8",
        c_b="0xe8399b",
        c_c="0xf2c53d",
        a1=f"{waves[0][0]:.2f}",
        k1=f"{waves[0][1]:.2f}",
        m1=f"{waves[0][2]:.2f}",
        w1=f"{waves[0][3]:.2f}",
        a2=f"{waves[1][0]:.2f}",
        k2=f"{waves[1][1]:.2f}",
        m2=f"{waves[1][2]:.2f}",
        w2=f"{waves[1][3]:.2f}",
    )

    for literal in (
        str(count),
        f"{phase:.3f}",
        str(seed),
        str(SPEED15),
        "#0b1424",
        "#28d7e8",
        "#e8399b",
        "#f2c53d",
        "(%.2f, %.2f, %.2f, %.2f)" % waves[0],
        "(%.2f, %.2f, %.2f, %.2f)" % waves[1],
    ):
        assert literal in prompt, literal
    assert 1200 <= count <= 2500
    # Поток обязан сдвинуть частицы за 78 шагов заметно.
    assert SPEED15 * min(w[0] for w in waves) * STEPS / 60 > 0.05

    palette_checks = [
        _frame_share("#0b1424", PALETTE15["#0b1424"], "фон потока"),
        _frame_share("#28d7e8", PALETTE15["#28d7e8"], "штрихи одного направления"),
        _frame_share("#e8399b", PALETTE15["#e8399b"], "штрихи второго направления"),
        _frame_share("#f2c53d", PALETTE15["#f2c53d"], "штрихи третьего направления"),
    ]
    composition_checks = [
        _flow_fills_frame(),
        _flow_not_mirrored(),
    ]
    detail_checks = [
        visual.image_min("changed_fraction", 0.12),
        visual.image_min("distinct_colors", 32),
        visual.rich_edges(0.008, "штрихи читаются, а не сливаются в дымку"),
        visual.image_min("saturation", 0.30),
    ]
    return art.build_task(
        task_id="L-15",
        title=TITLES["L-15"],
        prompt=prompt,
        spec=art.ArtSpec(palette=palette_checks, composition=composition_checks, detail=detail_checks, motion=STEPS),
        golden=golden,
        fixture={"count": count, "phase": phase, "seed": seed, "speed": SPEED15, "life": LIFE15},
    )


def _task_orbits(rng: random.Random) -> core.TaskInstance:
    count = rng.randrange(5, 9)
    planets = [
        (
            round(rng.uniform(0.30, 0.82), 3),
            round(rng.uniform(0.15, 0.85), 3),
            round(rng.uniform(0.0, 6.28), 3),
            round(rng.uniform(5.0, 9.0), 2),
            i % len(COLOURS16),
        )
        for i in range(count)
    ]
    # Первые четыре сферы обязаны нести разные цвета: иначе палитра не проверяется.
    for i, colour in enumerate(COLOURS16):
        planets[i] = (*planets[i][:4], i)

    prompt = PROMPT16.format(
        cam=CAM16,
        half=round(HALF16, 3),
        pu=round(2 * HALF16, 3),
        count=count,
        rows="\n".join(
            f"* сфера {i + 1}: a = {p[0]:.3f}, inc = {p[1]:.3f}, t0 = {p[2]:.3f}, T = {p[3]:.2f}, цвет {COLOURS16[p[4]]};"
            for i, p in enumerate(planets)
        ),
        trail=TRAIL16,
        dstep="0.0167",
        core_r=0.22,
        c_core=CORE16,
        c_bg="#080f22",
        c_1=COLOURS16[0],
        c_2=COLOURS16[1],
        c_3=COLOURS16[2],
        c_4=COLOURS16[3],
        s_bg=PALETTE16["#080f22"],
        s_1=PALETTE16[COLOURS16[0]],
        s_2=PALETTE16[COLOURS16[1]],
        s_3=PALETTE16[COLOURS16[2]],
        s_4=PALETTE16[COLOURS16[3]],
        steps=STEPS,
    )
    golden = GOLDEN16.format(
        count=count,
        trail=TRAIL16,
        rows="\n".join(
            f"  [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {p[3]:.2f}, 0x{COLOURS16[p[4]][1:]}]," for p in planets
        ),
        c_core="0xffe9a8",
        c_bg="0x080f22",
    )

    for p in planets:
        for number in (f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", f"{p[3]:.2f}"):
            assert number in prompt, number
    for literal in ("#080f22", "#ff4d6d", "#ffd166", "#4dd0e1", "#9b6dff", "#ffe9a8"):
        assert literal in prompt, literal
    assert count in range(5, 9)
    assert len({p[4] for p in planets}) == len(COLOURS16)
    # Углы обязаны сдвинуться за 78 шагов на заметную величину.
    assert min(2 * math.pi * STEPS / 60 / p[3] for p in planets) > 0.6

    palette_checks = [
        _frame_share("#080f22", PALETTE16["#080f22"], "тёмный фон"),
        _frame_share(COLOURS16[0], PALETTE16[COLOURS16[0]], "след первой сферы"),
        _frame_share(COLOURS16[1], PALETTE16[COLOURS16[1]], "след второй сферы"),
        _frame_share(COLOURS16[2], PALETTE16[COLOURS16[2]], "след третьей сферы"),
        _frame_share(COLOURS16[3], PALETTE16[COLOURS16[3]], "след четвёртой сферы"),
    ]
    composition_checks = [
        _core_brighter(),
        _orbits_spread(),
    ]
    detail_checks = [
        visual.image_min("changed_fraction", 0.05),
        visual.image_min("distinct_colors", 20),
        visual.rich_edges(0.002, "хвосты читаются, фон не залит"),
        visual.image_min("saturation", 0.25),
    ]
    return art.build_task(
        task_id="L-16",
        title=TITLES["L-16"],
        prompt=prompt,
        spec=art.ArtSpec(palette=palette_checks, composition=composition_checks, detail=detail_checks, motion=STEPS),
        golden=golden,
        fixture={
            "count": count,
            "planets": [list(p) for p in planets],
            "trail": TRAIL16,
        },
    )


BUILDERS = (_task_lava, _task_water, _task_flow, _task_orbits)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Все четыре сцены слайса, каждая берёт фикстуру из общего потока."""
    return [builder(rng) for builder in BUILDERS]
