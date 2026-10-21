"""Visual scenarios K-23 and K-26 — гидравлика: сеть с клапаном и резервуары.

Same shape as ``visual_fields``: the prompt states the scene, the exact scheme
and the exact probe fields, ``build()`` draws the fixture numbers from ``rng``
and renders *both* the prompt and the reference page from them, and the
expected numbers are recomputed here in Python from the stated law — the check
asks "does the network satisfy Kirchhoff" and "does the exchange conserve
volume", never "does it look like the reference".

Both scenarios have a host-*clickable* valve: a real, accessible ``<button>``
inside the viewport, and the page dispatches an ordinary click on it at a fixed
step. A prediction page that only flips a boolean without a button, or opens
the valve at the wrong time, is caught by the scene and the snapshots.

K-23 is a four-node pipe network solved with Kirchhoff's law: the valve state
changes the conductance of one edge, the branch flow reverses sign, and mass
balance at every free node must stay within ``1e-4``. K-26 is two
communicating tanks: ``q = G * (h1 - h2)``, heads updated as ``h = V / A``
after the volume exchange, so the total volume is conserved while the levels
approach each other and are read back through screen markers.
"""

from __future__ import annotations

import math
import random
from string import Template
from typing import Any

from .. import core, visual

IDS = ("K-23", "K-26")
IDS_BY_PROFILE = {"pilot": ("K-23",), "stretch-v1": IDS}

TITLE23 = "K-23 — сеть труб с клапаном, Кирхгоф и баланс массы"
TITLE26 = "K-26 — сообщающиеся резервуары, баланс объёмов"

#: Frame the scenarios are drawn in: an orthographic box, so a world point
#: maps to a fixed pixel and the fixture is provably inside the frame.
CAM23 = (-6.0, 6.0, 4.5, -4.5)
CAM26 = (-7.0, 7.0, 5.25, -5.25)
ZPLANE = 20.0

#: Nodes of the K-23 network: 0 is the source, 3 the drain.
P0_POS = (-4.6, 1.4, 0.0)
P1_POS = (-1.5, -1.5, 0.0)
P2_POS = (1.5, 1.2, 0.0)
P3_POS = (4.6, -1.3, 0.0)

#: Pipe edges as ``(name, from, to)``; ``valve`` is the switchable one.
EDGES = (
    ("feed", 0, 2),
    ("branch", 2, 1),
    ("drain", 2, 3),
    ("outlet", 1, 3),
    ("valve", 0, 1),
)

#: K-26 tank geometry and the screen mapping of a head (0 is the top edge).
TANK_H = 6.0
TANK_W = 2.2
TANK_BASE = -3.0
CAM_TOP = CAM26[2]
VIEW_H = CAM26[2] - CAM26[3]
FRAC_MARGIN = 0.02

STEP_DT = 1 / 60
STEP_CLICK = 60
STEPS23 = 300
STEPS26 = 240
DAMP = 0.25

MUTANT_TOL = 1e-2

FIELDS23 = """{
  "nodes": 4,
  "pipes": ["feed", "branch", "drain", "outlet", "valve"],
  "valve": 0 | 1,
  "ui": {"tag": "<тег кнопки>", "text": "<надпись>", "in_viewport": true,
         "aria_pressed": "false" | "true"},
  "closed": null | {
    "pressures": [P1, P2],
    "flows": {"feed": q, "branch": q, "drain": q, "outlet": q, "valve": q},
    "balance": <максимум из |q02 - q23 - q21| и |q21 + q01 - q13|>,
    "t": <1.0 по схеме>
  },
  "open": null | { <те же поля, состояние открытого клапана> },
  "t": <прошедшее время, 4 знака>
}"""

FIELDS26 = """{
  "tanks": 2,
  "valve": 0 | 1,
  "ui": {"tag": "<тег кнопки>", "text": "<надпись>", "in_viewport": true,
         "aria_pressed": "false" | "true"},
  "initial": [h1, h2],
  "at60": null | [h1, h2],
  "levels": [h1, h2],
  "volumes": [V1, V2],
  "q": <поток q, 4 знака>,
  "balance": <|V1 + V2 - V0|, 4 знака>,
  "t": <прошедшее время, 4 знака>
}"""

PROMPT23 = """Собери сцену гидравлической сети из четырёх узлов: узел 0 — источник,
узел 3 — сток, между ними четыре трубы и пятая — клапан. Клапан закрыт, пока по
нему не кликнут настоящей кнопкой в кадре.

Сцена:
* четыре узла — сферы: источник радиуса 0.55 цвета 0xff5533 в точке {p0},
  узел 1 радиуса 0.42 цвета 0xb8b8b8 в точке {p1}, узел 2 радиуса 0.42 того же
  цвета в точке {p2}, сток радиуса 0.55 цвета 0x2f8fff в точке {p3};
* трубы — цилиндры радиуса 0.09 цвета 0x707070 между парами узлов, каждая
  ориентирована по своей оси: `feed` 0-2, `branch` 2-1, `drain` 2-3,
  `outlet` 1-3; клапан — цилиндр радиуса 0.13 между узлами 0 и 1, его цвет
  0xff3b30 при закрытом клапане и 0x9ee0a8 при открытом;
* камера ортографическая `OrthographicCamera(-6, 6, 4.5, -4.5, 0.1, 100)` в
  точке (0, 0, 20) и смотрит в (0, 0, 0); свет — `AmbientLight(0xffffff, 0.6)`
  и `DirectionalLight(0xffffff, 1.4)` из точки (4, 8, 10); фон 0x0c1118;
* в левом нижнем углу кадра, целиком внутри 800x600, стоит доступная кнопка
  `<button id="valve" aria-pressed="false" aria-label="Открыть клапан">` с
  надписью «Открыть клапан»; клик по ней открывает клапан (`aria-pressed`
  становится `"true"`), повторные клики состояние не меняют.

Симуляция:
* давления в узлах 1 и 2 — `P1` и `P2`, стартуют с нуля; давления границ
  фиксированы: `P0 = {p0v}`, `P3 = 0`;
* проводимости труб: `feed` `C02 = {c02}`, `branch` `C21 = {c21}`,
  `drain` `C23 = {c23}`, `outlet` `C13 = {c13}`; проводимость клапана
  `Cv = {cv}` при открытом клапане и 0 при закрытом;
* поток по трубе `a -> b` равен `q_ab = (P_a - P_b) * C_ab`, направление
  хранения — по записи: `q02 = (P0 - P2) * C02`, `q21 = (P2 - P1) * C21`,
  `q23 = (P2 - P3) * C23`, `q13 = (P1 - P3) * C13`,
  `q01 = (P0 - P1) * Cv`;
* шаг делается только вызовом `window.__step(dt)`; `dt = {dt}`, время `t`
  растёт на `dt` за вызов. На каждом шаге сначала решается Кирхгоф — система
  из двух уравнений, по одному на свободный узел 2 и узел 1:
  `(P0 - P2) * C02 + (P3 - P2) * C23 + (P1 - P2) * C21 = 0` и
  `(P0 - P1) * Cv + (P2 - P1) * C21 + (P3 - P1) * C13 = 0`
  (проводимость клапана берётся по его текущему положению), затем давления
  подтягиваются к решению с коэффициентом `{damp}`:
  `P1 += {damp} * (P1_реш - P1)`, `P2 += {damp} * (P2_реш - P2)`;
* на 60-м шаге (`t = 1` с) снимается снимок закрытого состояния, и сразу
  после него по кнопке диспатчится обычный клик `document.getElementById('valve').click()`;
  дальше клапан открыт, и остальные шаги идут с `Cv = {cv}`;
* на 300-м шаге (`t = 5` с) снимается снимок открытого состояния;
* `window.__probe()` сначала рисует кадр (`paint()`, затем
  `renderer.render(scene, camera)`), потом возвращает поля; `window.__step(dt)`
  ничего не рисует. Снимок — это `{{"pressures": [P1, P2], "flows": {{...}},
  "balance": ..., "t": ...}}`, числа с 4 знаками; до своего шага снимок `null`.
  `balance` — максимум из `|q02 - q23 - q21|` (узел 2) и `|q21 + q01 - q13|`
  (узел 1);
* `window.__probe()` возвращает ровно такие поля:

{fields}

Проверяется: четыре узла, пять труб, закрытый клапан и кнопка в кадре сразу
после загрузки; давления, потоки и баланс в обоих снимках (сверка с этой же
системой, допуск 1e-3), баланс массы не больше 1e-4, знак `q21` меняется
(закрытый клапан: `q21 > 0`, открытый: `q21 < 0`, через закрытый клапан
`q01 = 0`), и кадр — узлы и трубы видны, картинка воспроизводима при повторной
загрузке.
"""

PROMPT26 = """Собери сцену двух сообщающихся резервуаров: слева узкий, справа
широкий, между ними труба с клапаном. Клапан закрыт, пока по нему не кликнут
настоящей кнопкой в кадре; после клика уровни выравниваются.

Сцена:
* два резервуара шириной {w} и высотой {h}, левый центром в x = {x1}, правый —
  в x = {x2}, дно обоих на y = {base}; стенки — рамки цвета 0x9aa0a6 толщиной
  0.06, вода — прямоугольник цвета 0x505050 от дна до уровня;
* уровень каждого резервуара отмечен горизонтальной планкой шириной 1.9 и
  высотой 0.12 на высоте воды: левая планка цвета 0xff3b30, правая 0x38f0ff;
* рядом с каждым резервуаром вертикальная шкала высотой 6 с делениями через
  0.5 цвета 0xd0d6dd;
* соединительная труба на высоте 0.6 от дна — цилиндр радиуса 0.12 цвета
  0x707070, клапан на ней — цилиндр радиуса 0.17 цвета 0xff3b30 при закрытом
  клапане и 0x9ee0a8 при открытом;
* камера ортографическая `OrthographicCamera(-7, 7, 5.25, -5.25, 0.1, 100)` в
  точке (0, 0, 20) и смотрит в (0, 0, 0); свет — `AmbientLight(0xffffff, 0.65)`
  и `DirectionalLight(0xffffff, 1.3)` из точки (4, 8, 10); фон 0x0b0f14;
* в левом нижнем углу кадра, целиком внутри 800x600, стоит доступная кнопка
  `<button id="valve" aria-pressed="false" aria-label="Открыть клапан">` с
  надписью «Открыть клапан»; клик открывает клапан (`aria-pressed` становится
  `"true"`), повторные клики ничего не меняют.

Симуляция:
* площади дна `A1 = {a1}` и `A2 = {a2}`, начальные уровни `h1 = {h10}` и
  `h2 = {h20}`, объёмы `V1 = A1 * h1`, `V2 = A2 * h2`, полный объём
  `V0 = V1 + V2` неизменен;
* шаг делается только вызовом `window.__step(dt)`, `dt = {dt}`; до 60-го шага
  клапан закрыт и уровни не меняются;
* на 60-м шаге снимается снимок `at60` (уровни как были) и сразу после него по
  кнопке диспатчится обычный клик `document.getElementById('valve').click()`;
* после открытия на каждом шаге строго в таком порядке:
  `q = {g} * (h1 - h2)`; `V1 = V1 - q * dt`; `V2 = V2 + q * dt`;
  затем `h1 = V1 / A1`; `h2 = V2 / A2`; поток `q` при закрытом клапане 0;
* `window.__probe()` сначала рисует кадр (`paint()`, затем
  `renderer.render(scene, camera)`), потом возвращает поля; `window.__step(dt)`
  ничего не рисует. Вода рисуется по текущим уровням, планки — на уровнях
  воды, так что уровень читается по кадру;
* `window.__probe()` возвращает ровно такие поля (числа с 4 знаками):

{fields}

Проверяется: два резервуара, закрытый клапан и кнопка в кадре сразу после
загрузки; уровни на 60-м шаге равны начальным; конечные уровни после 240
шагов (сверка с этой же схемой, допуск 1e-3); `V1 + V2` отличается от `V0` не
больше чем на 1e-6; левый уровень упал, правый вырос, `q > 0`; планки уровней
на кадре стоят на своих местах по шкале.
"""

GOLDEN23 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>valve network</title>
<style>
html,body{margin:0;background:#0c1118;overflow:hidden}canvas{display:block}
#valve{position:fixed;left:20px;top:545px;width:170px;height:38px;
  font:15px system-ui;background:#ff8c1a;color:#12161c;border:2px solid #ffd9a0;
  border-radius:6px;cursor:pointer}
</style></head>
<body>
<button id="valve" aria-pressed="false" aria-label="Открыть клапан">Открыть клапан</button>
<script type="module">
import * as THREE from './vendor/three.module.js';

const P0 = $p0v, P3 = 0, C02 = $c02, C23 = $c23, C21 = $c21, C13 = $c13, CV = $cv;
const DT = $dt, DAMP = $damp, CLICK = 60, TOTAL = 300;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c1118);
const camera = new THREE.OrthographicCamera(-6, 6, 4.5, -4.5, 0.1, 100);
camera.position.set(0, 0, 20);
camera.lookAt(0, 0, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(4, 8, 10);
scene.add(key);

const nodes = [$p0, $p1, $p2, $p3];
const nodeMat = [0xff5533, 0xb8b8b8, 0xb8b8b8, 0x2f8fff];
nodes.forEach((pos, i) => {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(i === 0 || i === 3 ? 0.55 : 0.42, 24, 16),
    new THREE.MeshStandardMaterial({color: nodeMat[i]})
  );
  mesh.position.set(...pos);
  scene.add(mesh);
});

function pipe(a, b, radius, color) {
  const pa = new THREE.Vector3(...nodes[a]), pb = new THREE.Vector3(...nodes[b]);
  const mid = pa.clone().add(pb).multiplyScalar(0.5);
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, pa.distanceTo(pb), 12),
    new THREE.MeshStandardMaterial({color})
  );
  mesh.position.copy(mid);
  mesh.rotation.z = Math.atan2(-(pb.x - pa.x), pb.y - pa.y);
  scene.add(mesh);
  return mesh;
}

pipe(0, 2, 0.09, 0x707070);
pipe(2, 1, 0.09, 0x707070);
pipe(2, 3, 0.09, 0x707070);
pipe(1, 3, 0.09, 0x707070);
const valvePipe = pipe(0, 1, 0.13, 0xff3b30);

const valveBtn = document.getElementById('valve');
let valve = 0, n = 0;
let p1 = 0, p2 = 0;
let closed = null, open = null;

const r4 = (v) => Math.round(v * 1e4) / 1e4;

function solve(cv) {
  const a = C02 + C23 + C21, b = cv + C21 + C13;
  const det = a * b - C21 * C21;
  const rhs2 = P0 * C02, rhs1 = P0 * cv;
  return [(rhs2 * b + C21 * rhs1) / det, (a * rhs1 + C21 * rhs2) / det];
}

function flows() {
  const q02 = (P0 - p2) * C02, q23 = (p2 - P3) * C23;
  const q21 = (p2 - p1) * C21, q13 = p1 * C13, q01 = (P0 - p1) * CV * valve;
  const balance = Math.max(Math.abs(q02 - q23 - q21), Math.abs(q21 + q01 - q13));
  return {q02, q23, q21, q13, q01, balance};
}

function snapshot() {
  const f = flows();
  return {
    pressures: [r4(p1), r4(p2)],
    flows: {feed: r4(f.q02), branch: r4(f.q21), drain: r4(f.q23), outlet: r4(f.q13), valve: r4(f.q01)},
    balance: r4(f.balance), t: r4(n * DT),
  };
}

function paint() {
  valvePipe.material.color.set(valve ? 0x9ee0a8 : 0xff3b30);
}

window.__step = (dt) => {
  const sol = solve(valve ? CV : 0);
  p1 += DAMP * (sol[1] - p1);
  p2 += DAMP * (sol[0] - p2);
  n += 1;
  if (n === CLICK && closed === null) {
    closed = snapshot();
    valveBtn.click();
  }
  if (n === TOTAL && valve) open = snapshot();
};

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  const r = valveBtn.getBoundingClientRect();
  return {
    nodes: 4,
    pipes: ['feed', 'branch', 'drain', 'outlet', 'valve'],
    valve,
    ui: {
      tag: valveBtn.tagName, text: valveBtn.textContent.trim(),
      in_viewport: r.width > 0 && r.height > 0 && r.left >= 0 && r.top >= 0 && r.right <= 800 && r.bottom <= 600,
      aria_pressed: valveBtn.getAttribute('aria-pressed'),
    },
    closed, open, t: r4(n * DT),
  };
};

valveBtn.addEventListener('click', () => {
  valve = 1;
  valveBtn.setAttribute('aria-pressed', 'true');
  paint();
});

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN26 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>communicating tanks</title>
<style>
html,body{margin:0;background:#0b0f14;overflow:hidden}canvas{display:block}
#valve{position:fixed;left:20px;top:545px;width:170px;height:38px;
  font:15px system-ui;background:#c7ced6;color:#12161c;border:2px solid #e8eef4;
  border-radius:6px;cursor:pointer}
</style></head>
<body>
<button id="valve" aria-pressed="false" aria-label="Открыть клапан">Открыть клапан</button>
<script type="module">
import * as THREE from './vendor/three.module.js';

const A1 = $a1, A2 = $a2, G = $g, DT = $dt;
const H = 6, W = 2.2, BASE = -3, X1 = -2.6, X2 = 2.6;
const CLICK = 60, TOTAL = 240;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
const camera = new THREE.OrthographicCamera(-7, 7, 5.25, -5.25, 0.1, 100);
camera.position.set(0, 0, 20);
camera.lookAt(0, 0, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const key = new THREE.DirectionalLight(0xffffff, 1.3);
key.position.set(4, 8, 10);
scene.add(key);

const r4 = (v) => Math.round(v * 1e4) / 1e4;
const valveBtn = document.getElementById('valve');
let valve = 0, n = 0;

function water(cx) {
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(W - 0.12, 1),
    new THREE.MeshStandardMaterial({color: 0x505050})
  );
  mesh.position.x = cx;
  scene.add(mesh);
  return mesh;
}
const leftWater = water(X1), rightWater = water(X2);

function marker(cx, color) {
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1.9, 0.12),
    new THREE.MeshStandardMaterial({color})
  );
  mesh.position.x = cx; mesh.position.z = 0.06;
  scene.add(mesh);
  return mesh;
}
const leftMark = marker(X1, 0xff3b30), rightMark = marker(X2, 0x38f0ff);

const wallMat = new THREE.MeshStandardMaterial({color: 0x9aa0a6});
const tickMat = new THREE.MeshStandardMaterial({color: 0xd0d6dd});
for (const cx of [X1, X2]) {
  for (const dx of [-W / 2, W / 2]) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(0.06, H), wallMat);
    wall.position.set(cx + dx, BASE + H / 2, 0.02);
    scene.add(wall);
  }
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(W, 0.06), wallMat);
  floor.position.set(cx, BASE, 0.02);
  scene.add(floor);
}
for (let i = 0; i <= 12; i++) {
  for (const [cx, side] of [[X1, -1], [X2, 1]]) {
    const tick = new THREE.Mesh(new THREE.PlaneGeometry(i % 2 ? 0.12 : 0.2, 0.035), tickMat);
    tick.position.set(cx + side * (W / 2 + 0.18), BASE + i * 0.5, 0.03);
    scene.add(tick);
  }
}

const linkMat = new THREE.MeshStandardMaterial({color: 0x707070});
const link = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, X2 - X1, 12), linkMat);
link.rotation.z = Math.PI / 2;
link.position.set(0, BASE + 0.6, 0.1);
scene.add(link);
const valveMesh = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), new THREE.MeshStandardMaterial({color: 0xff3b30}));
valveMesh.position.set(0, BASE + 0.6, 0.15);
scene.add(valveMesh);

let h1 = $h10, h2 = $h20;
let V1 = A1 * h1, V2 = A2 * h2;
const V0 = V1 + V2;
let q = 0, at60 = null;

function paint() {
  leftWater.scale.y = h1; leftWater.position.y = BASE + h1 / 2;
  rightWater.scale.y = h2; rightWater.position.y = BASE + h2 / 2;
  leftMark.position.y = BASE + h1;
  rightMark.position.y = BASE + h2;
  valveMesh.material.color.set(valve ? 0x9ee0a8 : 0xff3b30);
}

window.__step = (dt) => {
  if (valve) {
    q = G * (h1 - h2);
    V1 = V1 - q * dt;
    V2 = V2 + q * dt;
    h1 = V1 / A1;
    h2 = V2 / A2;
  }
  n += 1;
  if (n === CLICK && at60 === null) {
    at60 = [r4(h1), r4(h2)];
    valveBtn.click();
  }
};

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  const r = valveBtn.getBoundingClientRect();
  return {
    tanks: 2,
    valve,
    ui: {
      tag: valveBtn.tagName, text: valveBtn.textContent.trim(),
      in_viewport: r.width > 0 && r.height > 0 && r.left >= 0 && r.top >= 0 && r.right <= 800 && r.bottom <= 600,
      aria_pressed: valveBtn.getAttribute('aria-pressed'),
    },
    initial: [r4($h10), r4($h20)],
    at60,
    levels: [r4(h1), r4(h2)],
    volumes: [r4(V1), r4(V2)],
    q: r4(q),
    balance: r4(Math.abs(V1 + V2 - V0)),
    t: r4(n * DT),
  };
};

valveBtn.addEventListener('click', () => {
  valve = 1;
  valveBtn.setAttribute('aria-pressed', 'true');
  paint();
});

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")


# --------------------------------------------------------------------------- #
# Python replicas of the stated laws
# --------------------------------------------------------------------------- #


def round4(value: float) -> float:
    return math.floor(value * 1e4 + 0.5) / 1e4


def network(p0: float, c: dict[str, float], valve_open: bool) -> dict[str, Any]:
    """One Kirchhoff solve: pressures at the free nodes and every edge flow."""
    c02, c23, c21, c13 = c["c02"], c["c23"], c["c21"], c["c13"]
    cv = c["cv"] if valve_open else 0.0
    a = c02 + c23 + c21
    b = cv + c21 + c13
    det = a * b - c21 * c21
    rhs2, rhs1 = p0 * c02, p0 * cv
    p2 = (rhs2 * b + c21 * rhs1) / det
    p1 = (a * rhs1 + c21 * rhs2) / det
    q02 = (p0 - p2) * c02
    q23 = p2 * c23
    q21 = (p2 - p1) * c21
    q13 = p1 * c13
    q01 = (p0 - p1) * cv
    balance = max(abs(q02 - q23 - q21), abs(q21 + q01 - q13))
    return {
        "pressures": [p1, p2],
        "flows": {"feed": q02, "branch": q21, "drain": q23, "outlet": q13, "valve": q01},
        "balance": balance,
    }


def settle(
    p0: float,
    c: dict[str, float],
    valve_open: bool,
    steps: int,
    start: tuple[float, float] = (0.0, 0.0),
    damping: float = DAMP,
) -> dict[str, Any]:
    """Damped relaxation towards the Kirchhoff solution, exactly as stated."""
    p1, p2 = start
    for _ in range(steps):
        target = network(p0, c, valve_open)["pressures"]
        p1 += damping * (target[0] - p1)
        p2 += damping * (target[1] - p2)
    cv = c["cv"] if valve_open else 0.0
    q02 = (p0 - p2) * c["c02"]
    q23 = p2 * c["c23"]
    q21 = (p2 - p1) * c["c21"]
    q13 = p1 * c["c13"]
    q01 = (p0 - p1) * cv
    return {
        "pressures": [p1, p2],
        "flows": {"feed": q02, "branch": q21, "drain": q23, "outlet": q13, "valve": q01},
        "balance": max(abs(q02 - q23 - q21), abs(q21 + q01 - q13)),
    }


def tanks(
    a1: float,
    a2: float,
    h1: float,
    h2: float,
    g: float,
    steps: int,
    dt: float = STEP_DT,
    invert: bool = False,
) -> dict[str, Any]:
    """Closed valve for 60 steps, then the stated exchange for the rest."""
    v1, v2 = a1 * h1, a2 * h2
    v0 = v1 + v2
    q = 0.0
    for step in range(1, steps + 1):
        if step > STEP_CLICK:
            q = g * ((h2 - h1) if invert else (h1 - h2))
            v1 -= q * dt
            v2 += q * dt
            h1 = v1 / a1
            h2 = v2 / a2
    return {
        "levels": [h1, h2],
        "volumes": [v1, v2],
        "q": q,
        "balance": abs(v1 + v2 - v0),
    }


def frac_of_head(head: float) -> float:
    """Screen y (0 top, 1 bottom) of a head, the mapping the page draws with."""
    return (CAM_TOP - (TANK_BASE + head)) / VIEW_H


def _equal(path: str, value: Any):
    def check(payload: dict[str, Any]) -> tuple[bool, str]:
        actual = visual.path_of(payload, path)
        if actual is visual.MISSING:
            return False, f"{path}: в probe нет такого поля"
        return actual == value, f"{path} = {actual!r} (ожидалось {value!r})"

    return check


def _truth(path: str):
    def check(payload: dict[str, Any]) -> tuple[bool, str]:
        actual = visual.path_of(payload, path)
        if actual is visual.MISSING:
            return False, f"{path}: в probe нет такого поля"
        return bool(actual), f"{path} = {actual!r}"

    return check


def _ortho_in_frame(point: tuple[float, float], half: tuple[float, float], margin: float = 0.9) -> bool:
    x, y = point
    return abs(x) <= margin * half[0] and abs(y) <= margin * half[1]


def _assert_in_frame(points, label: str) -> None:
    half = ((CAM23[1] - CAM23[0]) / 2, (CAM23[2] - CAM23[3]) / 2)
    outside = [p for p in points if not _ortho_in_frame(p, half)]
    assert not outside, f"{label}: {len(outside)} точек вне кадра, например {outside[:2]}"


def _assert_literals(prompt: str, *values: str) -> None:
    missing = [v for v in values if v not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _task_network(rng: random.Random) -> core.TaskInstance:
    p0 = float(rng.choice([10.0, 12.0, 14.0]))
    c = {
        "c02": float(rng.choice([0.8, 1.0, 1.2])),
        "c23": float(rng.choice([0.8, 1.0, 1.2])),
        "c21": float(rng.choice([1.6, 2.0, 2.4])),
        "c13": float(rng.choice([0.8, 1.0, 1.2])),
        "cv": float(rng.choice([4.0, 6.0, 8.0])),
    }
    closed = settle(p0, c, False, STEP_CLICK)
    opened = settle(p0, c, True, STEPS23 - STEP_CLICK, (closed["pressures"][0], closed["pressures"][1]))

    # The frame is a fixture too: the fixture must not leave it.
    _assert_in_frame([P0_POS[:2], P1_POS[:2], P2_POS[:2], P3_POS[:2]], "K-23: узлы")

    # Closed: source feeds node 2, the branch runs 2 -> 1 (q21 > 0).
    assert closed["pressures"][0] > 0.0 and closed["pressures"][1] > closed["pressures"][0]
    assert closed["pressures"][1] < p0, "давление не может превысить источник"
    assert closed["flows"]["branch"] > 1e-3, "закрытый клапан: поток 2 -> 1 обязан быть"
    assert closed["flows"]["valve"] == 0.0, "через закрытый клапан потока нет"
    assert closed["balance"] <= 1e-4, f"баланс закрытого состояния {closed['balance']}"

    # Open: the valve pressurises node 1 above node 2, the branch reverses.
    assert opened["pressures"][0] > opened["pressures"][1], "открытый клапан обязан развернуть знак"
    assert opened["flows"]["branch"] < -1e-3, "открытый клапан: поток 1 -> 2 обязан быть"
    assert opened["flows"]["valve"] > 1e-3, "открытый клапан обязан пропускать поток"
    assert opened["balance"] <= 1e-4, f"баланс открытого состояния {opened['balance']}"

    # Discriminating power: swapped resistances and a leaking closed valve must
    # land outside the published tolerance.
    swapped = settle(p0, {k: 1.0 / v for k, v in c.items()}, False, STEP_CLICK)
    assert abs(swapped["pressures"][0] - closed["pressures"][0]) > MUTANT_TOL, "перепутанные R не отличить"
    leak = network(p0, c, True)
    assert abs(leak["flows"]["branch"] - closed["flows"]["branch"]) > MUTANT_TOL, "протечку не отличить"

    prompt = PROMPT23.format(
        p0=list(P0_POS), p1=list(P1_POS), p2=list(P2_POS), p3=list(P3_POS),
        p0v=p0, c02=c["c02"], c21=c["c21"], c23=c["c23"], c13=c["c13"], cv=c["cv"],
        dt=STEP_DT, damp=DAMP, fields=FIELDS23,
    )
    golden = GOLDEN23.substitute(
        p0=list(P0_POS), p1=list(P1_POS), p2=list(P2_POS), p3=list(P3_POS),
        p0v=p0, c02=c["c02"], c21=c["c21"], c23=c["c23"], c13=c["c13"], cv=c["cv"],
        dt=STEP_DT, damp=DAMP,
    )
    _assert_literals(
        prompt,
        f"P0 = {p0}", f"C02 = {c['c02']}", f"C21 = {c['c21']}", f"C23 = {c['c23']}",
        f"C13 = {c['c13']}", f"Cv = {c['cv']}", f"dt = {STEP_DT}", f"{DAMP} *",
        str(list(P0_POS)), str(list(P1_POS)), str(list(P2_POS)), str(list(P3_POS)),
    )

    def snap(state: dict[str, Any], t: float) -> dict[str, Any]:
        return {
            "pressures": [round4(v) for v in state["pressures"]],
            "flows": {k: round4(v) for k, v in state["flows"].items()},
            "balance": round4(state["balance"]),
            "t": round4(t),
        }

    closed_meta = snap(closed, STEP_CLICK * STEP_DT)
    open_meta = snap(opened, STEPS23 * STEP_DT)

    def state_checks(state: dict[str, Any], prefix: str, branch_positive: bool):
        checks = [visual.near(f"{prefix}.pressures.0", round4(state["pressures"][0]), 1e-3),
                  visual.near(f"{prefix}.pressures.1", round4(state["pressures"][1]), 1e-3)]
        for name, value in state["flows"].items():
            checks.append(visual.near(f"{prefix}.flows.{name}", round4(value), 1e-3))
        checks.append(visual.between(f"{prefix}.balance", 0.0, 1e-4))
        if branch_positive:
            checks.append(visual.between(f"{prefix}.flows.branch", 1e-3, 100.0))
        else:
            checks.append(visual.between(f"{prefix}.flows.branch", -100.0, -1e-3))
        return checks

    scene_checks = [
        visual.near("nodes", 4, 0),
        visual.contains_all("pipes", ["feed", "branch", "drain", "outlet", "valve"]),
        visual.near("valve", 0, 0),
        _equal("ui.tag", "BUTTON"),
        _truth("ui.in_viewport"),
        _equal("ui.aria_pressed", "false"),
    ]
    sim_checks = (
        [visual.near("valve", 1, 0), _equal("ui.aria_pressed", "true")]
        + state_checks(closed, "closed", True)
        + state_checks(opened, "open", False)
        + [visual.near("closed.flows.valve", 0.0, 1e-9)]
    )
    image_checks = [
        visual.image_min("changed_fraction", 0.02),
        visual.image_min("distinct_colors", 8),
        visual.rich_edges(0.004, "трубы и узлы"),
        visual.image_centroid("warm", 0, 0.0, 0.35, "источник слева"),
        visual.image_centroid("cool", 0, 0.65, 1.0, "сток справа"),
    ]
    return visual.build_task(
        task_id="K-23",
        title=TITLE23,
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks, simulation=sim_checks, image=image_checks,
            steps=STEPS23, dt=STEP_DT,
        ),
        golden=golden,
        rng_meta={
            "p0": p0, **c, "damping": DAMP, "steps": STEPS23,
            "closed": closed_meta, "open": open_meta,
        },
    )


def _task_tanks(rng: random.Random) -> core.TaskInstance:
    a1 = float(rng.choice([1.5, 2.0, 2.5]))
    a2 = float(rng.choice([2.5, 3.0, 3.5]))
    h1 = float(rng.choice([3.5, 4.0, 4.5]))
    h2 = float(rng.choice([0.8, 1.0, 1.2]))
    g = float(rng.choice([0.4, 0.5, 0.6]))

    final = tanks(a1, a2, h1, h2, g, STEPS26)
    v0 = a1 * h1 + a2 * h2

    assert final["balance"] <= 1e-9, f"объём уплыл: {final['balance']}"
    assert final["levels"][0] < h1, "левый уровень обязан упасть"
    assert final["levels"][1] > h2, "правый уровень обязан вырасти"
    assert final["q"] > 0.0, "поток идёт от высокого уровня к низкому"
    assert abs(v0 - (a1 * final["levels"][0] + a2 * final["levels"][1])) <= 1e-9

    # A coarser dt must stay stable and conserve volume just the same.
    flushed = tanks(a1, a2, h1, h2, g, STEPS26, dt=1 / 12)
    assert abs(a1 * flushed["levels"][0] + a2 * flushed["levels"][1] - v0) <= 1e-9

    # The mutant (flow from the low tank) must diverge from the oracle.
    mutant = tanks(a1, a2, h1, h2, g, STEPS26, invert=True)
    assert mutant["levels"][0] > h1 and mutant["levels"][1] < h2, "мутант уводит уровни прочь"
    assert abs(mutant["levels"][0] - final["levels"][0]) > MUTANT_TOL

    f1, f2 = frac_of_head(final["levels"][0]), frac_of_head(final["levels"][1])
    assert 0.05 < f1 < f2 < 0.95, f"планки обязаны быть выше и различимы ({f1}, {f2})"
    assert f2 - f1 > 0.03, "планки обязаны стоять в разных полосах кадра"
    assert _ortho_in_frame((2.6 + TANK_W / 2, TANK_BASE + TANK_H), ((CAM26[1] - CAM26[0]) / 2, VIEW_H / 2))
    assert _ortho_in_frame((-2.6 - TANK_W / 2, TANK_BASE + TANK_H), ((CAM26[1] - CAM26[0]) / 2, VIEW_H / 2))

    prompt = PROMPT26.format(
        w=TANK_W, h=TANK_H, x1=-2.6, x2=2.6, base=TANK_BASE,
        a1=a1, a2=a2, h10=h1, h20=h2, g=g, dt=STEP_DT, fields=FIELDS26,
    )
    golden = GOLDEN26.substitute(a1=a1, a2=a2, h10=h1, h20=h2, g=g, dt=STEP_DT)
    _assert_literals(
        prompt,
        f"A1 = {a1}", f"A2 = {a2}", f"h1 = {h1}", f"h2 = {h2}", f"{g} *",
        f"dt = {STEP_DT}", str(TANK_W), str(TANK_H),
    )

    scene_checks = [
        visual.near("tanks", 2, 0),
        visual.near("valve", 0, 0),
        visual.near("levels.0", round4(h1), 1e-9),
        visual.near("levels.1", round4(h2), 1e-9),
        visual.near("balance", 0.0, 1e-9),
        _equal("ui.tag", "BUTTON"),
        _truth("ui.in_viewport"),
        _equal("ui.aria_pressed", "false"),
    ]
    sim_checks = [
        visual.near("valve", 1, 0),
        _equal("ui.aria_pressed", "true"),
        visual.near("at60.0", round4(h1), 1e-4),
        visual.near("at60.1", round4(h2), 1e-4),
        visual.near("levels.0", round4(final["levels"][0]), 1e-3),
        visual.near("levels.1", round4(final["levels"][1]), 1e-3),
        visual.between("balance", 0.0, 1e-6),
        visual.between("q", 1e-6, 100.0),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.02),
        visual.image_min("distinct_colors", 8),
        visual.rich_edges(0.004, "стенки, шкалы и вода"),
        visual.image_centroid("warm", 0, 0.15, 0.45, "левый уровень"),
        visual.image_centroid("cool", 0, 0.55, 0.85, "правый уровень"),
        visual.image_centroid("warm", 1, f1 - FRAC_MARGIN, f1 + FRAC_MARGIN, "левая планка на шкале"),
        visual.image_centroid("cool", 1, f2 - FRAC_MARGIN, f2 + FRAC_MARGIN, "правая планка на шкале"),
    ]
    return visual.build_task(
        task_id="K-26",
        title=TITLE26,
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks, simulation=sim_checks, image=image_checks,
            steps=STEPS26, dt=STEP_DT,
        ),
        golden=golden,
        rng_meta={
            "a1": a1, "a2": a2, "h1_0": h1, "h2_0": h2, "g": g,
            "dt": STEP_DT, "steps": STEPS26,
            "levels": [round4(v) for v in final["levels"]],
            "balance": final["balance"],
        },
    )


BUILDERS = (_task_network, _task_tanks)


def build(rng: random.Random) -> list[core.TaskInstance]:
    return [builder(rng) for builder in BUILDERS]
