"""Visual scenarios K-31..K-32 — гравитационная задача двух тел и RLC-ступенька.

Same shape as ``visual_structures``: the prompt states the scene, the exact
scheme and the exact probe fields, ``build()`` draws the fixture numbers from
``rng`` and renders *both* the prompt and the reference page from them, and the
expected numbers are recomputed here in Python from the stated scheme — a check
asks "does the probe obey the equations it was given", never "does it look like
the reference".

Both scenarios are integrated by an explicit-ish Euler on purpose, and the
fixtures choose constants that keep the scheme stable and the answer readable:

* K-31 is a two-body Kepler pair on a circular orbit around the barycentre.
  The probe carries positions, velocities, the total energy and the momentum,
  and the checks pair the replay with the invariants — the momentum stays zero,
  the energy and the separation stay put — so a page that merely reports the
  initial state, drops the force or flips its sign cannot pass.
* K-32 is a series RLC step response: the inductor current and the capacitor
  voltage are advanced by explicit Euler, the underdamped fixture overshoots
  the source, and the checks follow the replay through the snapshot points,
  the overshoot peak and the zero crossings of the current.

Both reference pages use a fixed orthographic camera, so the two answers are
compared on the same picture.
"""

from __future__ import annotations

import math
import random
from string import Template

from .. import core, visual

IDS = ("K-31", "K-32")

#: ``pilot`` carries the cheapest proof that the slice works; ``stretch`` takes
#: both scenarios.
IDS_BY_PROFILE = {"pilot": ("K-31",), "stretch-v1": IDS}

#: Orthographic frustum (left, right, top, bottom) and camera of each scene.
ORTHO31 = (-8.0, 8.0, 6.0, -6.0)
ORTHO32 = (-9.0, 9.0, 6.0, -6.0)
CAMERA = (0.0, 0.0, 30.0)
LIGHT31 = (6.0, 10.0, 18.0)
LIGHT32 = (7.0, 11.0, 16.0)

#: Gravity constant in scene units; both bodies move under the mutual force.
G = 1.0
DT31 = 1 / 240
STEPS31 = 480
#: Best-of-half snapshot of K-31, in steps; the invariants are read there too.
HALF31 = STEPS31 // 2
BODY_RADIUS = 0.36

#: Series RLC fixture: underdamped by construction, overshoot is required.
DT32 = 0.002
STEPS32 = 280
SNAP32 = (70, 140)
GAUGE_MAX = 4.0

FIELDS31 = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 6 знаков>,
  "objects": ["body_a", "body_b", "trail_a", "trail_b"],
  "camera": [0, 0, 30],
  "bodies": [
    {"mass": m1, "x": <x1>, "y": <y1>, "vx": <vx1>, "vy": <vy1>},
    {"mass": m2, "x": <x2>, "y": <y2>, "vx": <vx2>, "vy": <vy2>}
  ],
  "energy": <полная энергия E>,
  "energy0": <начальная энергия E0>,
  "momentum": [m1*vx1 + m2*vx2, m1*vy1 + m2*vy2],
  "distance": <расстояние между телами>,
  "half": {"x1": ..., "y1": ..., "x2": ..., "y2": ...,
           "vx1": ..., "vy1": ..., "vx2": ..., "vy2": ...,
           "energy": ..., "momentum": [px, py], "distance": ...}
}"""

FIELDS32 = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 6 знаков>,
  "objects": ["board", "source", "inductor", "resistor", "capacitor", "gauge", "wires"],
  "camera": [0, 0, 30],
  "vc0": 0, "i0": 0,
  "vc1": <Vc ровно через 70 шагов>, "i1": <I через 70 шагов>,
  "vc2": <Vc ровно через 140 шагов>, "i2": <I через 140 шагов>,
  "vc": <Vc после всех шагов>, "i": <I после всех шагов>,
  "peak": <максимальное Vc за прогон>, "peak_step": <номер шага пика>,
  "crossings": <сколько раз ток сменил знак>,
  "gauge": {"x": 3.4, "y": <высота индикатора>}
}"""

PROMPT31 = """Собери сцену двух тел, связанных ньютоновской гравитацией: тела
обращаются вокруг общего барицентра на плоскости XY.

Сцена (плоскость XY, z = 0):
* тело A — сфера радиуса {radius}, цвет 0xff8a3d, имя `body_a`;
* тело B — сфера радиуса {radius}, цвет 0x5aa0ff, имя `body_b`;
* след A — линия цвета 0xff8a3d, имя `trail_a`; след B — линия цвета
  0x5aa0ff, имя `trail_b`; в след дописывается точка на каждом кадре;
* камера `OrthographicCamera(-8, 8, 6, -6, 0.1, 200)` в точке {cam} смотрит в
  (0, 0, 0); свет — `AmbientLight(0xffffff, 0.6)` и
  `DirectionalLight(0xffffff, 1.5)` из точки {light}.

Симуляция:
* тела: массы m1 = {m1} и m2 = {m2}; гравитационная постоянная G = {g};
* начальное расстояние d = {d}; тела стоят на оси X по разные стороны от
  барицентра: r1 = d*m2/(m1+m2), r2 = d*m1/(m1+m2), старт
  A = (-r1, 0) со скоростью (0, -v1), B = (r2, 0) со скоростью (0, v2), где
  v1 = sqrt(G*(m1+m2)/d)*m2/(m1+m2), v2 = sqrt(G*(m1+m2)/d)*m1/(m1+m2);
* шаг делается только вызовом `window.__step(dt)`, `dt = {dt}` с; ровно
  {steps} шагов, снимок половинного времени — ровно после {half} шагов;
* схема шага — полунеявный Эйлер ровно в таком порядке:
  `dx = x2 - x1; dy = y2 - y1; r = hypot(dx, dy); f = G/(r*r*r);`
  затем `vx1 += f*m2*dx*dt; vy1 += f*m2*dy*dt;`
  `vx2 -= f*m1*dx*dt; vy2 -= f*m1*dy*dt;`
  затем `x1 += vx1*dt; y1 += vy1*dt; x2 += vx2*dt; y2 += vy2*dt`;
* энергия `E = 0.5*m1*(vx1²+vy1²) + 0.5*m2*(vx2²+vy2²) - G*m1*m2/r`,
  импульс `[m1*vx1 + m2*vx2, m1*vy1 + m2*vy2]`; `energy0` — энергия старта;
* снимок `half` хранит те же поля и заполняется один раз, на шаге {half}; до
  снимка в нём начальные значения;
* `window.__probe()` сначала перерисовывает сцену
  (`renderer.render(scene, camera)`), потом возвращает ровно такие поля
  (числа — с 6 знаками):

{fields}

Проверяется: старт (шаги, время, энергия, нулевой импульс, расстояние),
положения, скорости, энергия и импульс на снимке и в конце (сверка с этой же
схемой, допуск 1e-3), инварианты — импульс остаётся нулевым, а энергия и
расстояние не плывут (допуск 1e-3) — и кадр: оба тела и следы видны.

Мгновенная «наивная» страница без силы даёт прямые линии и другие координаты;
страница с перевёрнутым знаком силы даёт расходящуюся пару.
"""

PROMPT32 = """Собери сцену последовательной RLC-цепи: источник подаёт ступеньку
напряжения, ток в индуктивности и напряжение на конденсаторе идут по явной
схеме Эйлера.

Сцена (плоскость XY, z = 0):
* доска — параллелепипед 9.4x0.3x5.4 в точке (0, -0.15, 0), цвет 0x16202c,
  имя `board`;
* источник — куб 1x1.2x1 в точке (-4, 0.6, 0), цвет 0x2f6fb3, имя `source`;
* индуктивность — три кольца радиуса 0.34 с шагом 0.55 по X в точке
  (-2.2, 0.8, 0), цвет 0xb35b2f, имя `inductor`;
* резистор — брусок 1.4x0.5x0.5 в точке (-0.2, 0.6, 0), цвет 0x8a4a2f, имя
  `resistor`;
* конденсатор — две пластины 0.12x1.6x1.2 в точках (1.6, 0.8, 0) и
  (2.1, 0.8, 0), цвет 0x9aa7b8, имя `capacitor`;
* провода — три бруска 2.2x0.08x0.08 на высоте 0.05 вдоль X, цвет 0x55606b,
  имя `wires`;
* индикатор — брусок 0.5x1x0.5 в точке (3.4, h/2 + 0.15, 0), высота
  `h = 0.2 + 3 * min(1.5, max(0, Vc / Vin))`, цвет
  `lerp(0x1d3f7a, 0xff5a1e, min(1, Vc / Vin))`, имя `gauge`;
* камера `OrthographicCamera(-9, 9, 6, -6, 0.1, 200)` в точке {cam} смотрит в
  (0, 1, 0); свет — `AmbientLight(0xffffff, 0.6)` и
  `DirectionalLight(0xffffff, 1.5)` из точки {light}.

Симуляция:
* `Vin = {vin}` В, `R = {r}` Ом, `L = {l}` Гн, `C = {c}` Ф; старт `I = 0`,
  `Vc = 0`;
* шаг делается только вызовом `window.__step(dt)`, `dt = {dt}` с; ровно
  {steps} шагов; один вызов — один шаг явной схемы Эйлера ровно в таком
  порядке: `I += dt * (Vin - Vc - R*I) / L;` затем `Vc += dt * I / C;`;
* снимки берутся ровно после {s1} шагов и после {s2} шагов; до своего момента
  снимок хранит 0;
* `peak` — максимальное достигнутое `Vc`, `peak_step` — номер шага, на
  котором оно впервые достигнуто; `crossings` — сколько раз ток сменил знак
  относительно предыдущего шага;
* `window.__probe()` сначала обновляет индикатор и перерисовывает сцену
  (`renderer.render(scene, camera)`), потом возвращает ровно такие поля
  (числа — с 6 знаками):

{fields}

Проверяется: старт, снимки, конец прогона, пик и число смен знака тока
(сверка с этой же схемой, допуск 1e-3), пределы — `Vc` перелетает `Vin`,
но остаётся ниже 1.8*`Vin`, к концу подходит к `Vin`, ток затухает, — и кадр:
цепь и выросший индикатор видны.

Мгновенная страница без демпфера (`R*I` выброшено) даёт пик 2*`Vin`;
страница со знаком `+R*I` разгоняет цепь.
"""

GOLDEN31 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>two-body kepler</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const G = $g, M1 = $m1, M2 = $m2;
const D = $d;
const R1 = D * M2 / (M1 + M2), R2 = D * M1 / (M1 + M2);
const VREL = Math.sqrt(G * (M1 + M2) / D);
const V1 = VREL * M2 / (M1 + M2), V2 = VREL * M1 / (M1 + M2);
const HALF = $half;

let x1 = -R1, y1 = 0, vx1 = 0, vy1 = -V1;
let x2 = R2, y2 = 0, vx2 = 0, vy2 = V2;
let elapsed = 0, steps = 0;

const energy = () => 0.5 * M1 * (vx1 * vx1 + vy1 * vy1)
  + 0.5 * M2 * (vx2 * vx2 + vy2 * vy2) - G * M1 * M2 / Math.hypot(x2 - x1, y2 - y1);
const momentum = () => [M1 * vx1 + M2 * vx2, M1 * vy1 + M2 * vy2];
const distance = () => Math.hypot(x2 - x1, y2 - y1);
const ENERGY0 = energy();
let half = {x1, y1, x2, y2, vx1, vy1, vx2, vy2, energy: ENERGY0, momentum: momentum(), distance: distance()};

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.OrthographicCamera(-8, 8, 6, -6, 0.1, 200);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(...$light);
scene.add(key);

const MAX = 1024;
function trail(color, name) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(MAX * 3);
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setDrawRange(0, 0);
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({color}));
  line.frustumCulled = false;
  line.name = name;
  scene.add(line);
  return {geometry, positions, count: 0};
}
const trailA = trail(0xff8a3d, 'trail_a');
const trailB = trail(0x5aa0ff, 'trail_b');

function ball(color, name) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry($radius, 16, 12),
    new THREE.MeshStandardMaterial({color})
  );
  mesh.name = name;
  scene.add(mesh);
  return mesh;
}
const bodyA = ball(0xff8a3d, 'body_a');
const bodyB = ball(0x5aa0ff, 'body_b');

function push(store, x, y) {
  if (store.count >= MAX) return;
  store.positions[store.count * 3] = x;
  store.positions[store.count * 3 + 1] = y;
  store.positions[store.count * 3 + 2] = 0;
  store.count += 1;
  store.geometry.setDrawRange(0, store.count);
  store.geometry.attributes.position.needsUpdate = true;
}

function draw() {
  bodyA.position.set(x1, y1, 0);
  bodyB.position.set(x2, y2, 0);
  push(trailA, x1, y1);
  push(trailB, x2, y2);
  renderer.render(scene, camera);
}

window.__step = (dt) => {
  const dx = x2 - x1, dy = y2 - y1;
  const r = Math.hypot(dx, dy);
  const f = G / (r * r * r);
  vx1 += f * M2 * dx * dt;
  vy1 += f * M2 * dy * dt;
  vx2 -= f * M1 * dx * dt;
  vy2 -= f * M1 * dy * dt;
  x1 += vx1 * dt;
  y1 += vy1 * dt;
  x2 += vx2 * dt;
  y2 += vy2 * dt;
  elapsed += dt;
  steps += 1;
  if (steps === HALF) {
    half = {x1, y1, x2, y2, vx1, vy1, vx2, vy2, energy: energy(), momentum: momentum(), distance: distance()};
  }
};

const round6 = (value) => Math.round(value * 1e6) / 1e6;

window.__probe = () => {
  draw();
  return {
    steps,
    t: round6(elapsed),
    objects: scene.children.map((o) => o.name).filter(Boolean),
    camera: [camera.position.x, camera.position.y, camera.position.z],
    bodies: [
      {mass: M1, x: round6(x1), y: round6(y1), vx: round6(vx1), vy: round6(vy1)},
      {mass: M2, x: round6(x2), y: round6(y2), vx: round6(vx2), vy: round6(vy2)},
    ],
    energy: round6(energy()),
    energy0: round6(ENERGY0),
    momentum: momentum().map(round6),
    distance: round6(distance()),
    half: {
      x1: round6(half.x1), y1: round6(half.y1), x2: round6(half.x2), y2: round6(half.y2),
      vx1: round6(half.vx1), vy1: round6(half.vy1), vx2: round6(half.vx2), vy2: round6(half.vy2),
      energy: round6(half.energy), momentum: half.momentum.map(round6), distance: round6(half.distance),
    },
  };
};

draw();
window.__ready = true;
</script>
</body></html>
""")

GOLDEN32 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>series rlc step</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const VIN = $vin, R = $r, L = $l, C = $c;
const S1 = $s1, S2 = $s2;

let current = 0, vc = 0, elapsed = 0, steps = 0;
let vc1 = 0, i1 = 0, vc2 = 0, i2 = 0;
let peak = 0, peakStep = 0, crossings = 0, previous = 0;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.OrthographicCamera(-9, 9, 6, -6, 0.1, 200);
camera.position.set(...$camera);
camera.lookAt(0, 1, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(...$light);
scene.add(key);

function board(x, y, z, sx, sy, sz, color, name) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(sx, sy, sz),
    new THREE.MeshStandardMaterial({color})
  );
  mesh.position.set(x, y, z);
  mesh.name = name;
  scene.add(mesh);
  return mesh;
}

board(0, -0.15, 0, 9.4, 0.3, 5.4, 0x16202c, 'board');
board(-4, 0.6, 0, 1, 1.2, 1, 0x2f6fb3, 'source');
for (const dx of [-0.55, 0, 0.55]) {
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.34, 0.09, 8, 18),
    new THREE.MeshStandardMaterial({color: 0xb35b2f})
  );
  ring.position.set(-2.2 + dx, 0.8, 0);
  ring.name = 'inductor';
  scene.add(ring);
}
board(-0.2, 0.6, 0, 1.4, 0.5, 0.5, 0x8a4a2f, 'resistor');
board(1.6, 0.8, 0, 0.12, 1.6, 1.2, 0x9aa7b8, 'capacitor');
board(2.1, 0.8, 0, 0.12, 1.6, 1.2, 0x9aa7b8, 'capacitor');
board(-3, 0.05, 0, 1.6, 0.08, 0.08, 0x55606b, 'wires');
board(0.6, 0.05, 0, 1.6, 0.08, 0.08, 0x55606b, 'wires');
board(2.85, 0.05, 0, 0.9, 0.08, 0.08, 0x55606b, 'wires');

const gauge = board(3.4, 0.35, 0, 0.5, 1, 0.5, 0x1d3f7a, 'gauge');
const cold = new THREE.Color(0x1d3f7a);
const hot = new THREE.Color(0xff5a1e);
const tint = new THREE.Color();

function level() {
  return Math.min(1.5, Math.max(0, vc / VIN));
}

function paint() {
  const l = level();
  const h = 0.2 + 3 * l;
  gauge.scale.set(1, h, 1);
  gauge.position.set(3.4, h / 2 + 0.15, 0);
  tint.copy(cold).lerp(hot, Math.min(1, l));
  gauge.material.color.copy(tint);
  renderer.render(scene, camera);
}

window.__step = (dt) => {
  current += dt * (VIN - vc - R * current) / L;
  vc += dt * current / C;
  elapsed += dt;
  steps += 1;
  if ((current < 0) !== (previous < 0)) crossings += 1;
  previous = current;
  if (vc > peak) { peak = vc; peakStep = steps; }
  if (steps === S1) { vc1 = vc; i1 = current; }
  if (steps === S2) { vc2 = vc; i2 = current; }
};

const round6 = (value) => Math.round(value * 1e6) / 1e6;

window.__probe = () => {
  paint();
  const h = 0.2 + 3 * level();
  return {
    steps,
    t: round6(elapsed),
    objects: scene.children.map((o) => o.name).filter(Boolean),
    camera: [camera.position.x, camera.position.y, camera.position.z],
    vc0: 0, i0: 0,
    vc1: round6(vc1), i1: round6(i1),
    vc2: round6(vc2), i2: round6(i2),
    vc: round6(vc), i: round6(current),
    peak: round6(peak), peak_step: peakStep,
    crossings,
    gauge: {x: 3.4, y: round6(h)},
  };
};

paint();
window.__ready = true;
</script>
</body></html>
""")


# --------------------------------------------------------------------------- #
# Python replicas of the stated schemes
# --------------------------------------------------------------------------- #


def round6(value: float) -> float:
    return math.floor(value * 1e6 + 0.5) / 1e6


def kepler(
    m1: float,
    m2: float,
    d: float,
    steps: int,
    *,
    force: bool = True,
    sign: float = 1.0,
) -> dict[str, object]:
    """The stated semi-implicit Euler, replayed independently of the page.

    ``force=False`` and ``sign=-1`` are the two named mutants: a page without
    the force draws straight lines, a page with the wrong sign lets the pair
    fly apart.
    """
    r1 = d * m2 / (m1 + m2)
    r2 = d * m1 / (m1 + m2)
    vrel = math.sqrt(G * (m1 + m2) / d)
    x1, y1, x2, y2 = -r1, 0.0, r2, 0.0
    vx1, vy1 = 0.0, -vrel * m2 / (m1 + m2)
    vx2, vy2 = 0.0, vrel * m1 / (m1 + m2)

    def energy() -> float:
        r = math.hypot(x2 - x1, y2 - y1)
        return (
            0.5 * m1 * (vx1 * vx1 + vy1 * vy1)
            + 0.5 * m2 * (vx2 * vx2 + vy2 * vy2)
            - G * m1 * m2 / r
        )

    energy0 = energy()
    half = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "energy": energy0, "momentum": [0.0, 0.0], "distance": d,
    }
    for step in range(1, steps + 1):
        dx, dy = x2 - x1, y2 - y1
        r = math.hypot(dx, dy)
        if force:
            f = sign / (r * r * r)
            vx1 += f * m2 * dx * DT31
            vy1 += f * m2 * dy * DT31
            vx2 -= f * m1 * dx * DT31
            vy2 -= f * m1 * dy * DT31
        x1 += vx1 * DT31
        y1 += vy1 * DT31
        x2 += vx2 * DT31
        y2 += vy2 * DT31
        if step == HALF31:
            half = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "energy": energy(), "momentum": [m1 * vx1 + m2 * vx2, m1 * vy1 + m2 * vy2],
                "distance": math.hypot(x2 - x1, y2 - y1),
            }
    return {
        "bodies": [
            {"mass": m1, "x": x1, "y": y1, "vx": vx1, "vy": vy1},
            {"mass": m2, "x": x2, "y": y2, "vx": vx2, "vy": vy2},
        ],
        "energy": energy(),
        "energy0": energy0,
        "momentum": [m1 * vx1 + m2 * vx2, m1 * vy1 + m2 * vy2],
        "distance": math.hypot(x2 - x1, y2 - y1),
        "half": half,
    }


def rlc(
    vin: float,
    r: float,
    l: float,
    c: float,
    *,
    steps: int = STEPS32,
    damping: bool = True,
    sign: float = -1.0,
) -> dict[str, object]:
    """Explicit Euler on the series RLC loop, replayed independently.

    ``damping=False`` and ``sign=+1`` are the mutants: the first drops ``R*I``,
    the second feeds the drop back with the wrong sign.
    """
    current = 0.0
    vc = 0.0
    peak = 0.0
    peak_step = 0
    crossings = 0
    previous = 0.0
    vc1 = i1 = vc2 = i2 = 0.0
    for step in range(1, steps + 1):
        drop = r * current if damping else 0.0
        current += DT32 * (vin - vc + sign * drop) / l
        vc += DT32 * current / c
        if (current < 0) != (previous < 0):
            crossings += 1
        previous = current
        if vc > peak:
            peak, peak_step = vc, step
        if step == SNAP32[0]:
            vc1, i1 = vc, current
        if step == SNAP32[1]:
            vc2, i2 = vc, current
    return {
        "vc1": vc1, "i1": i1, "vc2": vc2, "i2": i2,
        "vc": vc, "i": current, "peak": peak, "peak_step": peak_step,
        "crossings": crossings,
    }


def _in_ortho(point: tuple[float, float], ortho: tuple[float, ...], margin: float = 0.9) -> bool:
    left, right, top, bottom = ortho
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_w, half_h = (right - left) / 2, (top - bottom) / 2
    return abs(point[0] - cx) <= margin * half_w and abs(point[1] - cy) <= margin * half_h


def _assert_in_frame(ortho: tuple[float, ...], points, label: str) -> None:
    outside = [p for p in points if not _in_ortho(p, ortho)]
    assert not outside, f"{label}: {len(outside)} точек вне кадра, например {outside[:2]}"


def _assert_literals(prompt: str, *values: str) -> None:
    missing = [v for v in values if v not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


def _camera_is(camera: tuple[float, ...]) -> visual.Predicate:
    def check(payload: dict) -> tuple[bool, str]:
        actual = visual.path_of(payload, "camera", None)
        good = isinstance(actual, list) and len(actual) == 3 and all(
            abs(float(a) - b) <= 1e-6 for a, b in zip(actual, camera)
        )
        return good, f"camera = {actual} (ожидалось {list(camera)})"

    return visual._label(check, "камера на месте")


def _zero_momentum(path: str, tolerance: float) -> visual.Predicate:
    """The barycentre does not move: both components stay at zero."""

    def check(payload: dict) -> tuple[bool, str]:
        values = visual.path_of(payload, path, None)
        if not isinstance(values, list) or len(values) != 2:
            return False, f"{path}: не пара чисел ({values!r})"
        good = all(abs(float(value)) <= tolerance for value in values)
        return good, f"{path} = {values} (обязан быть нулевым)"

    return visual._label(check, f"|{path}| <= {tolerance}")


def _invariant(left: str, right: str, tolerance: float) -> visual.Predicate:
    """Two readings of the same conserved quantity must agree."""

    def check(payload: dict) -> tuple[bool, str]:
        a, problem = visual._number(payload, left)
        if problem:
            return False, problem
        b, problem = visual._number(payload, right)
        if problem:
            return False, problem
        good = abs(a - b) <= tolerance
        return good, f"{left} = {a}, {right} = {b} (разница <= {tolerance})"

    return visual._label(check, f"|{left} - {right}| <= {tolerance}")


def _rising(values) -> visual.Predicate:
    def check(payload: dict) -> tuple[bool, str]:
        actual = [visual.path_of(payload, path, None) for path in values]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in actual):
            return False, f"{values}: не числа ({actual})"
        good = all(a < b for a, b in zip(actual, actual[1:]))
        return good, f"Vc = {actual} (обязано строго расти)"

    return visual._label(check, "Vc строго растёт")


def _below(path: str, value: float) -> visual.Predicate:
    def check(payload: dict) -> tuple[bool, str]:
        actual, problem = visual._number(payload, path)
        if problem:
            return False, problem
        return actual <= value, f"{path} = {actual} (обязано быть <= {value})"

    return visual._label(check, f"{path} <= {value}")


def _between(path: str, low: float, high: float) -> visual.Predicate:
    def check(payload: dict) -> tuple[bool, str]:
        actual, problem = visual._number(payload, path)
        if problem:
            return False, problem
        return low <= actual <= high, f"{path} = {actual} (ожидалось {low}..{high})"

    return visual._label(check, f"{path} в {low}..{high}")


def _at_least(path: str, value: float) -> visual.Predicate:
    def check(payload: dict) -> tuple[bool, str]:
        actual, problem = visual._number(payload, path)
        if problem:
            return False, problem
        return actual >= value, f"{path} = {actual} (обязано быть >= {value})"

    return visual._label(check, f"{path} >= {value}")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _task_kepler(rng: random.Random) -> core.TaskInstance:
    for _ in range(80):
        m1 = round(rng.uniform(1.0, 2.0), 2)
        m2 = round(rng.uniform(1.0, 2.0), 2)
        d = round(rng.uniform(3.0, 3.8), 2)
        if m1 == m2:
            continue
        good = kepler(m1, m2, d, STEPS31)
        naive = kepler(m1, m2, d, STEPS31, force=False)
        repel = kepler(m1, m2, d, STEPS31, sign=-1.0)
        far = max(
            math.hypot(body["x"], body["y"]) for body in good["bodies"]
        )
        movement = max(
            abs(good["bodies"][i][axis] - naive["bodies"][i][axis])
            for i in range(2)
            for axis in ("x", "y")
        )
        flipped = max(
            abs(good["bodies"][i][axis] - repel["bodies"][i][axis])
            for i in range(2)
            for axis in ("x", "y")
        )
        if movement > 0.2 and flipped > 0.2 and far < 4.0:
            break
    else:  # pragma: no cover - the ranges are wide enough for a hit every time
        raise AssertionError("не удалось подобрать фикстуру K-31")

    energy0 = good["energy0"]
    half = good["half"]
    # The fixture proves itself: the circular orbit really is an orbit, the
    # invariants hold, and both named mutants are far from the answer.
    assert energy0 < -0.1 and abs(good["energy"] - energy0) < 1e-5
    assert abs(good["distance"] - d) < 2e-3
    assert abs(half["distance"] - d) < 2e-3
    assert max(abs(v) for v in good["momentum"]) < 1e-9
    assert good["bodies"][0]["vx"] != 0.0

    prompt = PROMPT31.format(
        cam=list(CAMERA), light=list(LIGHT31), m1=m1, m2=m2, g=G, d=d, radius=BODY_RADIUS,
        dt=DT31, steps=STEPS31, half=HALF31, fields=FIELDS31,
    )
    golden = GOLDEN31.substitute(
        g=G, m1=m1, m2=m2, d=d, half=HALF31, radius=BODY_RADIUS,
        camera=list(CAMERA), light=list(LIGHT31),
    )
    _assert_literals(
        prompt,
        f"m1 = {m1}", f"m2 = {m2}", f"G = {G}", f"d = {d}",
        f"dt = {DT31}", f"{STEPS31} шагов", f"после {HALF31} шагов",
        "-8, 8, 6, -6", str(list(CAMERA)), str(list(LIGHT31)),
        "0xff8a3d", "0x5aa0ff", "trail_a", "trail_b",
    )
    sampled = [(-3.8, 0.0), (3.8, 0.0), (0.0, 3.8), (0.0, -3.8)]
    sampled.append((half["x1"], half["y1"]))
    sampled.append((half["x2"], half["y2"]))
    sampled.extend((body["x"], body["y"]) for body in good["bodies"])  # type: ignore[union-attr]
    _assert_in_frame(ORTHO31, sampled, "K-31: орбита")

    def body_checks(stage: dict, tags: str) -> list[visual.Predicate]:
        bodies = stage["bodies"]
        checks = [
            visual.near(f"{tags}bodies.0.x", round6(bodies[0]["x"]), 1e-3),
            visual.near(f"{tags}bodies.0.y", round6(bodies[0]["y"]), 1e-3),
            visual.near(f"{tags}bodies.0.vx", round6(bodies[0]["vx"]), 1e-3),
            visual.near(f"{tags}bodies.0.vy", round6(bodies[0]["vy"]), 1e-3),
            visual.near(f"{tags}bodies.1.x", round6(bodies[1]["x"]), 1e-3),
            visual.near(f"{tags}bodies.1.y", round6(bodies[1]["y"]), 1e-3),
            visual.near(f"{tags}bodies.1.vx", round6(bodies[1]["vx"]), 1e-3),
            visual.near(f"{tags}bodies.1.vy", round6(bodies[1]["vy"]), 1e-3),
        ]
        return checks

    def half_checks(stage: dict) -> list[visual.Predicate]:
        return [
            visual.near("half.x1", round6(stage["x1"]), 1e-3),
            visual.near("half.y1", round6(stage["y1"]), 1e-3),
            visual.near("half.x2", round6(stage["x2"]), 1e-3),
            visual.near("half.y2", round6(stage["y2"]), 1e-3),
            visual.near("half.energy", round6(stage["energy"]), 1e-3),
            _zero_momentum("half.momentum", 1e-3),
            visual.near("half.distance", round6(stage["distance"]), 1e-3),
        ]

    scene_checks = [
        visual.near("steps", 0, 0),
        visual.near("t", 0.0, 1e-9),
        visual.near("energy0", round6(energy0), 1e-6),
        visual.near("energy", round6(energy0), 1e-6),
        visual.near("distance", d, 1e-6),
        _zero_momentum("momentum", 1e-6),
        _camera_is(CAMERA),
        visual.contains_all("objects", ["body_a", "body_b", "trail_a", "trail_b"]),
    ]
    simulation = [
        visual.near("steps", STEPS31, 0),
        visual.near("t", round6(STEPS31 * DT31), 1e-6),
        *body_checks(good, ""),
        *half_checks(half),
        visual.near("energy", round6(good["energy"]), 1e-3),
        _zero_momentum("momentum", 1e-3),
        visual.near("distance", round6(good["distance"]), 1e-3),
        _invariant("energy", "energy0", 1e-3),
        _invariant("energy", "half.energy", 1e-3),
        _invariant("distance", "half.distance", 3e-3),
    ]
    image = [
        visual.image_min("changed_fraction", 0.003),
        visual.image_min("distinct_colors", 4),
        visual.image_color("warm", 0.0015, "тело A и его след различимы"),
        visual.image_color("cool", 0.0015, "тело B и его след различимы"),
        visual.image_min("mean", 5.0),
    ]
    return visual.build_task(
        task_id="K-31",
        title="K-31 — гравитация двух тел",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks, simulation=simulation, image=image, steps=STEPS31, dt=DT31
        ),
        golden=golden,
        rng_meta={
            "m1": m1, "m2": m2, "d": d, "g": G, "steps": STEPS31, "dt": DT31, "half": HALF31,
            "energy0": round6(energy0), "energy": round6(good["energy"]),
            "distance": round6(good["distance"]),
            "bodies": [[round6(body[key]) for key in ("x", "y", "vx", "vy")] for body in good["bodies"]],
            "naive": [[round6(body[key]) for key in ("x", "y")] for body in naive["bodies"]],
            "repel": [[round6(body[key]) for key in ("x", "y")] for body in repel["bodies"]],
        },
    )


def _task_rlc(rng: random.Random) -> core.TaskInstance:
    for _ in range(80):
        vin = float(rng.choice([10.0, 12.0]))
        r = round(rng.uniform(1.5, 2.2), 2)
        l = round(rng.uniform(0.18, 0.26), 3)
        c = round(rng.uniform(0.008, 0.012), 4)
        good = rlc(vin, r, l, c)
        ratio = good["peak"] / vin
        if not (1.35 <= ratio <= 1.75):
            continue
        if not (2 <= good["crossings"] <= 6):
            continue
        if abs(good["vc"] - vin) > 0.25 * vin or min(good["vc1"], good["vc2"]) < 0.0:
            continue
        if abs(good["vc1"] - good["vc2"]) < 0.5 or abs(good["i1"]) < 0.5:
            continue
        break
    else:  # pragma: no cover
        raise AssertionError("не удалось подобрать фикстуру K-32")

    undamped = rlc(vin, r, l, c, damping=False)
    wrong_sign = rlc(vin, r, l, c, sign=1.0)
    # The fixture proves itself: real overshoot, a decaying current, and both
    # named mutants visibly different at the snapshot points.
    assert 0.0 < good["vc1"] < good["peak"]
    assert abs(good["i"]) < 0.5 and abs(good["i1"]) > 0.5
    assert undamped["peak"] / vin > 1.9
    assert abs(undamped["peak"] - good["peak"]) > 0.1 * vin
    assert abs(wrong_sign["vc1"] - good["vc1"]) > 0.5
    assert abs(wrong_sign["vc"] - good["vc"]) > 0.05

    prompt = PROMPT32.format(
        cam=list(CAMERA), light=list(LIGHT32), vin=vin, r=r, l=l, c=c, dt=DT32,
        steps=STEPS32, s1=SNAP32[0], s2=SNAP32[1], fields=FIELDS32,
    )
    golden = GOLDEN32.substitute(
        vin=vin, r=r, l=l, c=c, s1=SNAP32[0], s2=SNAP32[1],
        camera=list(CAMERA), light=list(LIGHT32),
    )
    _assert_literals(
        prompt,
        f"Vin = {vin}", f"R = {r}", f"L = {l}", f"C = {c}", f"dt = {DT32}",
        f"{STEPS32} шагов", f"после {SNAP32[0]}", f"после {SNAP32[1]}",
        "-9, 9, 6, -6", str(list(CAMERA)), str(list(LIGHT32)),
        "0xff5a1e", "0x1d3f7a",
    )
    _assert_in_frame(
        ORTHO32,
        [(-4.5, 0.0), (4.5, 0.0), (-4.5, 1.6), (4.5, 1.6), (3.4, 0.2 + 3 * 1.5)],
        "K-32: цепь",
    )
    scene_checks = [
        visual.near("steps", 0, 0),
        visual.near("vc0", 0.0, 0),
        visual.near("i0", 0.0, 0),
        visual.near("vc", 0.0, 1e-9),
        visual.near("crossings", 0, 0),
        _camera_is(CAMERA),
        visual.contains_all(
            "objects", ["board", "source", "inductor", "resistor", "capacitor", "gauge", "wires"]
        ),
    ]
    simulation = [
        visual.near("steps", STEPS32, 0),
        visual.near("t", round6(STEPS32 * DT32), 1e-6),
        visual.near("vc1", round6(good["vc1"]), 1e-3),
        visual.near("i1", round6(good["i1"]), 1e-3),
        visual.near("vc2", round6(good["vc2"]), 1e-3),
        visual.near("i2", round6(good["i2"]), 1e-3),
        visual.near("vc", round6(good["vc"]), 1e-3),
        visual.near("i", round6(good["i"]), 1e-3),
        visual.near("peak", round6(good["peak"]), 1e-3),
        visual.near("peak_step", good["peak_step"], 0),
        visual.near("crossings", good["crossings"], 0),
        _rising(("vc0", "vc1")),
        _below("vc2", good["peak"]),
        _between("peak", 1.3 * vin, 1.85 * vin),
        _between("vc", 0.75 * vin, 1.15 * vin),
        _below("i2", abs(good["i1"])),
        _at_least("crossings", 2),
    ]
    image = [
        visual.image_min("changed_fraction", 0.01),
        visual.image_min("distinct_colors", 6),
        visual.image_color("warm", 0.002, "индикатор и резистор различимы"),
        visual.image_min("mean", 8.0),
    ]
    return visual.build_task(
        task_id="K-32",
        title="K-32 — ступенька последовательной RLC-цепи",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks, simulation=simulation, image=image, steps=STEPS32, dt=DT32
        ),
        golden=golden,
        rng_meta={
            "vin": vin, "r": r, "l": l, "c": c, "steps": STEPS32, "dt": DT32,
            "vc1": round6(good["vc1"]), "i1": round6(good["i1"]),
            "vc2": round6(good["vc2"]), "i2": round6(good["i2"]),
            "vc": round6(good["vc"]), "i": round6(good["i"]),
            "peak": round6(good["peak"]), "peak_step": good["peak_step"],
            "crossings": good["crossings"],
            "undamped_peak": round6(undamped["peak"]),
            "wrong_sign": [round6(wrong_sign["vc1"]), round6(wrong_sign["vc"])],
        },
    )


BUILDERS = (_task_kepler, _task_rlc)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Every scenario of the slice, each drawing its fixture from the same stream."""
    return [builder(rng) for builder in BUILDERS]
