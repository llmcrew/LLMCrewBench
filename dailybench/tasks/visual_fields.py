"""Visual scenarios K-12..K-16 — сетки, клеточные автоматы, среды.

Same shape as ``visual_projectiles``: the prompt states the scene, the exact
update rule and the exact probe fields, ``build()`` draws the fixture numbers
from ``rng`` and renders *both* the prompt and the reference page from them, and
the expected numbers are recomputed here in Python from the stated rule — a
check asks "does the grid obey the scheme it was given", never "does it look
like the reference".

Every scenario here is a grid, so the reference page draws it with a single
``THREE.InstancedMesh``: one instance per cell, the cell colour carrying the
state. Randomness is never ``Math.random()`` — where a fixture needs a random
start, the constants of an explicit LCG are written into the prompt, so the
Python replica walks the very same sequence.
"""

from __future__ import annotations

import math
import random
from string import Template

from .. import core, visual

IDS = ("K-12", "K-13", "K-14", "K-15", "K-16")

#: Fixed camera and frame size per scenario, so two answers are compared on the
#: same picture.
CAMERA12 = (0.0, 40.0, 40.0)
CAMERA13 = (0.0, 50.0, 46.0)
CAMERA14 = (0.0, 34.0, 23.0)
CAMERA15 = (0.0, 0.0, 60.0)
CAMERA16 = (56.0, 43.0, 43.0)
LIGHT12 = (12.0, 40.0, 16.0)
LIGHT13 = (16.0, 46.0, 18.0)
LIGHT14 = (10.0, 28.0, 14.0)
LIGHT15 = (0.0, 20.0, 40.0)
LIGHT16 = (30.0, 40.0, 20.0)

#: The LCG the fixture prompts hand out: identical in JS and Python, because
#: every product stays below 2**53 and both languages mask the low 32 bits.
LCG_A = 1664525
LCG_C = 1013904223
LCG_M = 1 << 32

#: The three rows the Life probe reports: first, middle (11 of 24) and last.
LIFE_ROWS = (0, 11, 23)

FIELDS12 = """{
  "grid": 32,
  "mean": <средняя температура по всей сетке, 4 знака>,
  "max": <максимальная температура, 4 знака>,
  "samples": [T[8][8], T[16][16], T[24][8]]
}"""

FIELDS13 = """{
  "trees": <клеток-деревьев>,
  "burning": <горящих клеток>,
  "burned": <выгоревших клеток>,
  "steps": <число сделанных шагов>
}"""

FIELDS14 = """{
  "alive": <живых клеток на всей сетке>,
  "generation": <число сделанных шагов>,
  "rows": [<живых в строке 0>, <живых в строке 11>, <живых в строке 23>]
}"""

FIELDS15 = """{
  "samples": [u[16], u[32], u[48]],
  "energy": <энергия по формуле из задания, 4 знака>
}"""

FIELDS16 = """{
  "heights": [h(2, 3), h(10, 10), h(20, 5)],
  "min": <минимум по всей решётке, 4 знака>,
  "max": <максимум по всей решётке, 4 знака>
}"""

PROMPT12 = """Собери сцену диффузии тепла на квадратной пластине: сетка 32x32,
тепло растекается от одной нагретой точки, пластина нарисована столбиками,
высота и цвет которых зависят от температуры ячейки.

Сцена:
* пластина — 1024 столбика, по одному на ячейку `(i, j)` (i — строка,
  j — столбец), всё одним `THREE.InstancedMesh` с именем `plate`; основание
  кубика 0.92x0.92, ячейка `(i, j)` стоит в точке `(j - 15.5, h/2, i - 15.5)`
  с высотой `h = 0.05 + 3 * t`, где `t` — температура ячейки, делённая на
  максимальную температуру текущего состояния; цвет столбика —
  `lerp(0x14306b, 0xff5a1e, t)`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 0, 0); свет — `AmbientLight(0xffffff, 0.55)` и
  `DirectionalLight(0xffffff, 1.6)` из точки {light}.

Симуляция:
* температура хранится как 1024 числа, строка за строкой, индекс `i*32 + j`;
* в начале всё нули, кроме `T[8][8] = 1000` — одна нагретая точка;
* шаг делается только вызовом `window.__step(dt)`; `dt` на схему не влияет —
  один вызов это ровно один шаг;
* состояние рисуется в `window.__probe()`: сначала `paint()`, потом
  `renderer.render(scene, camera)`, и только затем возвращаются поля;
  `window.__step(dt)` состояние только продвигает и ничего не рисует — так
  проверка снимает кадр ровно после нужного числа шагов;
* схема шага — явная, массив обновляется на месте, одним проходом
  `i = 0..31`, внутри `j = 0..31`, ровно в таком порядке:
  `c = T[i][j]; up = (i > 0) ? T[i-1][j] : c; dn = (i < 31) ? T[i+1][j] : c;
  lf = (j > 0) ? T[i][j-1] : c; rt = (j < 31) ? T[i][j+1] : c;
  T[i][j] = c + a * (up + dn + lf + rt - 4 * c)` при `a = {a}`;
  сосед за границей сетки равен самой ячейке — это нулевой поток;
  соседи сверху и слева берутся уже обновлёнными, потому что проход идёт по
  порядку и меняет массив на месте;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: размер сетки и состояние сразу после загрузки, температуры в трёх
узлах `T[8][8]`, `T[16][16]`, `T[24][8]`, среднее и максимум после заданного
числа шагов (сверка с этой же схемой, допуск 1e-3), и кадр — пластина видна,
горячая область отличается по цвету, картинка воспроизводима при повторной
загрузке.
"""

PROMPT13 = """Собери сцену лесного пожара: сетка 40x40, часть клеток занята
деревьями, от одной горящей клетки огонь перекидывается на восемь соседей.

Сцена:
* сетка 40x40, по одному кубику 0.9x0.6x0.9 на клетку `(i, j)` в точке
  `(j - 19.5, 0.3, i - 19.5)`, всё одним `THREE.InstancedMesh` (1600
  экземпляров) с именем `forest`;
* цвет клетки по состоянию: пустая — 0x0f1620, дерево — 0x2f8f46,
  горящая — 0xff5a1e, выгоревшая — 0x4a3a2a;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 0, 0); свет — `AmbientLight(0xffffff, 0.6)` и
  `DirectionalLight(0xffffff, 1.5)` из точки {light}.

Симуляция:
* три состояния: пусто (0), дерево (1), огонь (2), выгоревшее (3);
* старт детерминирован линейным конгруэнтным генератором: `seed = {seed}`;
  функция `next()` делает `seed = (seed * 1664525 + 1013904223) % 4294967296`
  и возвращает `seed / 4294967296`; обходи сетку по строкам (`i = 0..39`,
  внутри `j = 0..39`) и на каждую клетку вызывай `next()` ровно один раз:
  дерево, если значение меньше {p}, иначе пустая клетка;
* затем клетка (20, 20) становится горящей;
* шаг делается только вызовом `window.__step(dt)`; `dt` на схему не влияет —
  один вызов это ровно один шаг;
* состояние рисуется в `window.__probe()`: сначала пересчитай цвета клеток,
  потом `renderer.render(scene, camera)`, и только затем возвращай поля;
  `window.__step(dt)` состояние только продвигает и ничего не рисует — так
  проверка снимает кадр ровно после нужного числа шагов;
* схема шага (все клетки считаются по старому состоянию, потом состояние
  заменяется целиком): горящая клетка становится выгоревшей; дерево
  загорается, если хотя бы один из восьми соседей горит (за границей сетки
  соседей нет, замыкания нет); пустая и выгоревшая клетки не меняются;
* `window.__probe()` возвращает ровно такие поля (числа — целые):

{fields}

Проверяется: состояние сразу после загрузки (одна горящая клетка, число
деревьев из генератора), числа клеток каждого состояния после заданного числа
шагов (сверка с этой же схемой) и кадр — лес виден, горящий фронт отличается по
цвету, картинка воспроизводима при повторной загрузке.
"""

PROMPT14 = """Собери игру «Жизнь» на сетке 24x24: клетки рождаются и умирают по
правилам B3/S23, живые клетки нарисованы кубиками.

Сцена:
* сетка 24x24, по одному кубику 0.92x0.4x0.92 на клетку `(i, j)` в точке
  `(j - 11.5, 0.2, i - 11.5)`, всё одним `THREE.InstancedMesh` (576
  экземпляров) с именем `life`;
* цвет клетки: живая — 0xffa726, мёртвая — 0x1a2432;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 0, 0); свет — `AmbientLight(0xffffff, 0.6)` и
  `DirectionalLight(0xffffff, 1.5)` из точки {light}.

Симуляция:
* старт детерминирован линейным конгруэнтным генератором: `seed = {seed}`;
  функция `next()` делает `seed = (seed * 1664525 + 1013904223) % 4294967296`
  и возвращает `seed / 4294967296`; обходи сетку по строкам (`i = 0..23`,
  внутри `j = 0..23`) и на каждую клетку вызывай `next()` ровно один раз:
  клетка жива, если значение меньше {p}, иначе мертва;
* шаг делается только вызовом `window.__step(dt)`; `dt` на схему не влияет —
  один вызов это ровно одно поколение;
* состояние рисуется в `window.__probe()`: сначала пересчитай цвета клеток,
  потом `renderer.render(scene, camera)`, и только затем возвращай поля;
  `window.__step(dt)` состояние только продвигает и ничего не рисует — так
  проверка снимает кадр ровно после нужного числа поколений;
* правила B3/S23: живая клетка остаётся живой с 2 или 3 живыми соседями и
  умирает иначе; мёртвая оживает ровно с 3 живыми соседями; соседей восемь,
  за границей сетки сосед считается мёртвым (замыкания нет);
  все клетки считаются по старому состоянию сразу, новое состояние заменяет
  старое целиком;
* `window.__probe()` возвращает ровно такие поля (числа — целые):

{fields}

Проверяется: состояние сразу после загрузки из генератора, число живых клеток
в трёх строках (0, 11 и 23) и во всей сетке после заданного числа поколений
(сверка с этой же схемой) и кадр — живые клетки видны и различимы, картинка
воспроизводима при повторной загрузке.
"""

PROMPT15 = """Собери сцену колебаний струны: 64 точки, закреплённые концы,
начальное отклонение — колокол, дальше волна бежит по явной схеме.

Сцена:
* струна — 64 сферы радиуса 0.55, всё одним `THREE.InstancedMesh` с именем
  `string`; i-я сфера стоит в точке `((i - 31.5) * 0.7, u[i] * 4.5, 0)`;
* цвет сферы — от высоты: `lerp(0x1d3f7a, 0xffd27f, |u[i]| / A)`;
* вдоль сфер проведи `THREE.Line` (цвет 0xffd27f) по тем же точкам, чтобы
  профиль струны читался;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 0, 0); свет — `AmbientLight(0xffffff, 0.6)` и
  `DirectionalLight(0xffffff, 1.6)` из точки {light}.

Симуляция:
* 64 числа `u[i]`, причём `u[0] = u[63] = 0` на каждом шаге;
* начальное отклонение — колокол `u[i] = A * exp(-((i - center) / sigma)^2)`
  при A = {amp}, center = {center}, sigma = {sigma}; начальная скорость
  нулевая, то есть `u_prev = u` (копия начального профиля);
* шаг делается только вызовом `window.__step(dt)`; `dt` на схему не влияет —
  один вызов это ровно один шаг;
* состояние рисуется в `window.__probe()`: сначала пересчитай положения сфер и
  линию, потом `renderer.render(scene, camera)`, и только затем возвращай поля;
  `window.__step(dt)` состояние только продвигает и ничего не рисует — так
  проверка снимает кадр ровно после нужного числа шагов;
* схема шага (новый массив считается по старому, потом массивы сдвигаются):
  `next[i] = 2 * u[i] - u_prev[i] + c2 * (u[i-1] - 2 * u[i] + u[i+1])` для
  `i = 1..62`, `next[0] = next[63] = 0`, где `c2 = {c2}`; затем
  `u_prev = u`, `u = next`;
* энергия считается по текущему и предыдущему профилю:
  `energy = sum for i = 1..62 of (0.5 * (u[i] - u_prev[i])^2 +
  0.5 * c2 * (u[i+1] - u[i])^2)`;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: профиль и энергия сразу после загрузки, значения u[16], u[32],
u[48] и энергия после заданного числа шагов (сверка с этой же схемой, допуск
1e-3) и кадр — струна видна, её профиль различим, картинка воспроизводима при
повторной загрузке.
"""

PROMPT16 = """Собери терраин из сеяного вэлью-шума: высота в узле решётки
задаётся хешем от координат, рельеф — плоская геометрия с displacement.

Сцена:
* рельеф — `THREE.PlaneGeometry(40, 40, 40, 40)`, повёрнутая в плоскость XZ
  (`rotateX(-Math.PI / 2)`), после чего каждой вершине в узел `(x, z)`
  (целые координаты от -20 до 20) ставится `y = h(x, z)`;
  нормали пересчитай;
* материал — `MeshStandardMaterial` цвета 0x8a6b3d, `flatShading: true`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 0, 0); свет — `AmbientLight(0xffffff, 0.8)` и
  `DirectionalLight(0xffffff, 1.8)` из точки {light}.

Формула высоты (считай её ровно так, все действия — над 32-битными
беззнаковыми числами):
* `hash(x, y)`: `h = (x * 374761393 + y * 668265263) mod 2^32;
  h = ((h xor (h >> 13)) * 1274126177) mod 2^32;
  h = h xor (h >> 16);` результат — `h / 2^32`, то есть число из [0, 1);
  в JS удобно `Math.imul`, и результат каждой строки приводится к
  беззнаковому через `>>> 0`;
* коэффициент масштаба `A = {amp}`, высота узла `h(x, y) = A * (hash(x, y) - 0.5)`;
* между узлами высота интерполируется билинейно, но в целых узлах она равна
  ровно `h(x, y)`, и решётка как раз стоит в целых узлах.

Симуляция:
* терраин статичен: `window.__step(dt)` ничего не меняет — высоты заданы
  хешем и на шагах не пересчитываются;
* кадр рисуется в `window.__probe()` (`renderer.render(scene, camera)`) перед
  возвратом полей — так проверка снимает кадр после нужного числа шагов;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`min` и `max` — минимум и максимум `h(x, y)` по всем 41x41 узлам решётки.

Проверяется: форма рельефа сразу после загрузки, значения h(2, 3), h(10, 10),
h(20, 5), минимум и максимум по решётке после заданного числа шагов (сверка с
этой же формулой хеша, допуск 1e-3) и кадр — рельеф виден и освещён, картинка
воспроизводима при повторной загрузке.
"""

# Templates rather than ``str.format``: the reference pages are full of JS
# braces, and one stray ``{`` would turn a page into a KeyError.
GOLDEN12 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>heat plate</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = 32;
const ALPHA = $alpha;
const HOT = 8;
const HOT_VALUE = 1000;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 500);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(...$light);
scene.add(key);

const T = new Float64Array(N * N);
T[HOT * N + HOT] = HOT_VALUE;

const plate = new THREE.InstancedMesh(
  new THREE.BoxGeometry(0.92, 1, 0.92),
  new THREE.MeshStandardMaterial({}),
  N * N
);
plate.name = 'plate';
scene.add(plate);

const dummy = new THREE.Object3D();
const cold = new THREE.Color(0x14306b);
const hot = new THREE.Color(0xff5a1e);
const tint = new THREE.Color();

function paint() {
  let peak = 0;
  for (let k = 0; k < N * N; k++) if (T[k] > peak) peak = T[k];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j;
      const t = peak > 0 ? T[k] / peak : 0;
      const h = 0.05 + 3 * t;
      dummy.position.set(j - (N - 1) / 2, h / 2, i - (N - 1) / 2);
      dummy.scale.set(1, h, 1);
      dummy.updateMatrix();
      plate.setMatrixAt(k, dummy.matrix);
      tint.copy(cold).lerp(hot, t);
      plate.setColorAt(k, tint);
    }
  }
  plate.instanceMatrix.needsUpdate = true;
  if (plate.instanceColor) plate.instanceColor.needsUpdate = true;
}

let steps = 0;

window.__step = (dt) => {
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j;
      const c = T[k];
      const up = i > 0 ? T[(i - 1) * N + j] : c;
      const dn = i < N - 1 ? T[(i + 1) * N + j] : c;
      const lf = j > 0 ? T[i * N + j - 1] : c;
      const rt = j < N - 1 ? T[i * N + j + 1] : c;
      T[k] = c + ALPHA * (up + dn + lf + rt - 4 * c);
    }
  }
  steps += 1;
};

const round = (v) => Math.round(v * 1e4) / 1e4;

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  let sum = 0, peak = 0;
  for (let k = 0; k < N * N; k++) { sum += T[k]; if (T[k] > peak) peak = T[k]; }
  return {
    grid: N,
    mean: round(sum / (N * N)),
    max: round(peak),
    samples: [round(T[8 * N + 8]), round(T[16 * N + 16]), round(T[24 * N + 8])],
  };
};

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN13 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>forest fire</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = 40;
const P = $p;
let seed = $seed;
const rnd = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const EMPTY = 0, TREE = 1, BURN = 2, BURNED = 3;
const state = new Uint8Array(N * N);
for (let k = 0; k < N * N; k++) state[k] = rnd() < P ? TREE : EMPTY;
state[20 * N + 20] = BURN;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 500);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(...$light);
scene.add(key);

const forest = new THREE.InstancedMesh(
  new THREE.BoxGeometry(0.9, 0.6, 0.9),
  new THREE.MeshStandardMaterial({}),
  N * N
);
forest.name = 'forest';
scene.add(forest);

const PALETTE = [
  new THREE.Color(0x0f1620),
  new THREE.Color(0x2f8f46),
  new THREE.Color(0xff5a1e),
  new THREE.Color(0x4a3a2a),
];
const dummy = new THREE.Object3D();

function paint() {
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j;
      dummy.position.set(j - (N - 1) / 2, 0.3, i - (N - 1) / 2);
      dummy.updateMatrix();
      forest.setMatrixAt(k, dummy.matrix);
      forest.setColorAt(k, PALETTE[state[k]]);
    }
  }
  forest.instanceMatrix.needsUpdate = true;
  if (forest.instanceColor) forest.instanceColor.needsUpdate = true;
}

let steps = 0;

window.__step = (dt) => {
  const old = state.slice();
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j;
      const s = old[k];
      if (s === BURN) { state[k] = BURNED; continue; }
      if (s !== TREE) { state[k] = s; continue; }
      let hit = false;
      for (let di = -1; di <= 1 && !hit; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          if (di === 0 && dj === 0) continue;
          const a = i + di, b = j + dj;
          if (a >= 0 && a < N && b >= 0 && b < N && old[a * N + b] === BURN) { hit = true; break; }
        }
      }
      state[k] = hit ? BURN : TREE;
    }
  }
  steps += 1;
};

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  const count = [0, 0, 0, 0];
  for (let k = 0; k < N * N; k++) count[state[k]] += 1;
  return { trees: count[TREE], burning: count[BURN], burned: count[BURNED], steps };
};

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN14 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>game of life</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = 24;
const P = $p;
let seed = $seed;
const rnd = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const grid = new Uint8Array(N * N);
for (let k = 0; k < N * N; k++) grid[k] = rnd() < P ? 1 : 0;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 500);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(...$light);
scene.add(key);

const life = new THREE.InstancedMesh(
  new THREE.BoxGeometry(0.92, 0.4, 0.92),
  new THREE.MeshStandardMaterial({}),
  N * N
);
life.name = 'life';
scene.add(life);

const ALIVE = new THREE.Color(0xffa726);
const DEAD = new THREE.Color(0x1a2432);
const dummy = new THREE.Object3D();

function paint() {
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j;
      dummy.position.set(j - (N - 1) / 2, 0.2, i - (N - 1) / 2);
      dummy.updateMatrix();
      life.setMatrixAt(k, dummy.matrix);
      life.setColorAt(k, grid[k] ? ALIVE : DEAD);
    }
  }
  life.instanceMatrix.needsUpdate = true;
  if (life.instanceColor) life.instanceColor.needsUpdate = true;
}

const neighbours = (old, i, j) => {
  let n = 0;
  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      if (di === 0 && dj === 0) continue;
      const a = i + di, b = j + dj;
      if (a >= 0 && a < N && b >= 0 && b < N && old[a * N + b]) n += 1;
    }
  }
  return n;
};

let steps = 0;

window.__step = (dt) => {
  const old = grid.slice();
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j;
      const n = neighbours(old, i, j);
      grid[k] = old[k] ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
    }
  }
  steps += 1;
};

const rowCount = (i) => {
  let n = 0;
  for (let j = 0; j < N; j++) n += grid[i * N + j] ? 1 : 0;
  return n;
};

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  let alive = 0;
  for (let k = 0; k < N * N; k++) alive += grid[k] ? 1 : 0;
  return { alive, generation: steps, rows: [rowCount(0), rowCount(11), rowCount(23)] };
};

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN15 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>wave on a string</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = 64;
const AMP = $amp;
const CENTER = $center;
const SIGMA = $sigma;
const C2 = $c2;

let u = new Float64Array(N);
for (let i = 0; i < N; i++) u[i] = AMP * Math.exp(-Math.pow((i - CENTER) / SIGMA, 2));
u[0] = 0; u[N - 1] = 0;
let prev = u.slice();

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 500);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(...$light);
scene.add(key);

const beads = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.55, 12, 8),
  new THREE.MeshStandardMaterial({}),
  N
);
beads.name = 'string';
scene.add(beads);

const profile = new THREE.BufferGeometry();
profile.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
const line = new THREE.Line(profile, new THREE.LineBasicMaterial({color: 0xffd27f}));
line.name = 'profile';
scene.add(line);

const dummy = new THREE.Object3D();
const low = new THREE.Color(0x1d3f7a);
const high = new THREE.Color(0xffd27f);
const tint = new THREE.Color();

function paint() {
  const positions = profile.attributes.position;
  for (let i = 0; i < N; i++) {
    const x = (i - (N - 1) / 2) * 0.7;
    const y = u[i] * 4.5;
    dummy.position.set(x, y, 0);
    dummy.updateMatrix();
    beads.setMatrixAt(i, dummy.matrix);
    tint.copy(low).lerp(high, Math.min(1, Math.abs(u[i]) / AMP));
    beads.setColorAt(i, tint);
    positions.setXYZ(i, x, y, 0);
  }
  beads.instanceMatrix.needsUpdate = true;
  if (beads.instanceColor) beads.instanceColor.needsUpdate = true;
  positions.needsUpdate = true;
  profile.computeBoundingSphere();
}

let steps = 0;

window.__step = (dt) => {
  const next = new Float64Array(N);
  for (let i = 1; i < N - 1; i++) {
    next[i] = 2 * u[i] - prev[i] + C2 * (u[i - 1] - 2 * u[i] + u[i + 1]);
  }
  prev = u;
  u = next;
  steps += 1;
};

const round = (v) => Math.round(v * 1e4) / 1e4;

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  let energy = 0;
  for (let i = 1; i < N - 1; i++) {
    const du = u[i] - prev[i];
    const dr = u[i + 1] - u[i];
    energy += 0.5 * du * du + 0.5 * C2 * dr * dr;
  }
  return { samples: [round(u[16]), round(u[32]), round(u[48])], energy: round(energy) };
};

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN16 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>seeded value noise terrain</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const AMP = $amp;
const HALF = 20;

function hash(x, y) {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
}

const height = (x, y) => AMP * (hash(x, y) - 0.5);

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 500);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(...$light);
scene.add(key);

const geometry = new THREE.PlaneGeometry(2 * HALF, 2 * HALF, 2 * HALF, 2 * HALF);
geometry.rotateX(-Math.PI / 2);
const position = geometry.attributes.position;
for (let v = 0; v < position.count; v++) {
  const x = Math.round(position.getX(v));
  const z = Math.round(position.getZ(v));
  position.setY(v, height(x, z));
}
position.needsUpdate = true;
geometry.computeVertexNormals();

const terrain = new THREE.Mesh(
  geometry,
  new THREE.MeshStandardMaterial({color: 0x8a6b3d, flatShading: true})
);
terrain.name = 'terrain';
scene.add(terrain);

const round = (v) => Math.round(v * 1e4) / 1e4;

window.__probe = () => {
  renderer.render(scene, camera);
  let low = Infinity, high = -Infinity;
  for (let x = -HALF; x <= HALF; x++) {
    for (let y = -HALF; y <= HALF; y++) {
      const h = height(x, y);
      if (h < low) low = h;
      if (h > high) high = h;
    }
  }
  return {
    heights: [round(height(2, 3)), round(height(10, 10)), round(height(20, 5))],
    min: round(low),
    max: round(high),
  };
};

window.__step = (dt) => {};

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")


# --------------------------------------------------------------------------- #
# Python replicas of the stated schemes
# --------------------------------------------------------------------------- #


def round4(value: float) -> float:
    """``Math.round(v * 1e4) / 1e4`` — the rounding the pages are asked for."""
    return math.floor(value * 1e4 + 0.5) / 1e4


def lcg_values(seed: int, count: int) -> list[float]:
    """The exact LCG sequence the prompts spell out."""
    out: list[float] = []
    state = seed
    for _ in range(count):
        state = (state * LCG_A + LCG_C) % LCG_M
        out.append(state / LCG_M)
    return out


def diffuse(alpha: float, size: int, hot: int, hot_value: float, steps: int):
    """Gauss-Seidel heat step, in place, exactly as the prompt writes it."""
    grid = [[0.0] * size for _ in range(size)]
    grid[hot][hot] = hot_value
    for _ in range(steps):
        for i in range(size):
            for j in range(size):
                c = grid[i][j]
                up = grid[i - 1][j] if i > 0 else c
                dn = grid[i + 1][j] if i < size - 1 else c
                lf = grid[i][j - 1] if j > 0 else c
                rt = grid[i][j + 1] if j < size - 1 else c
                grid[i][j] = c + alpha * (up + dn + lf + rt - 4 * c)
    total = sum(sum(row) for row in grid)
    peak = max(max(row) for row in grid)
    return grid, total / (size * size), peak


def fire(seed: int, density: float, size: int, steps: int, ignite: tuple[int, int]):
    """The forest-fire automaton: 0 empty, 1 tree, 2 burning, 3 burned."""
    values = lcg_values(seed, size * size)
    grid = [[1 if values[i * size + j] < density else 0 for j in range(size)] for i in range(size)]
    grid[ignite[0]][ignite[1]] = 2
    initial_trees = sum(row.count(1) for row in grid)
    for _ in range(steps):
        new = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                state = grid[i][j]
                if state == 2:
                    new[i][j] = 3
                elif state == 1:
                    hit = False
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            a, b = i + di, j + dj
                            if 0 <= a < size and 0 <= b < size and grid[a][b] == 2:
                                hit = True
                    new[i][j] = 2 if hit else 1
                else:
                    new[i][j] = state  # пустая и выгоревшая клетки не меняются
        grid = new
    counts = [sum(row.count(v) for row in grid) for v in (1, 2, 3)]
    return grid, initial_trees, counts


def life(seed: int, density: float, size: int, steps: int):
    """Conway's B3/S23, no wrapping, simultaneous update."""
    values = lcg_values(seed, size * size)
    grid = [1 if v < density else 0 for v in values]
    initial_alive = sum(grid)
    initial_rows = [sum(grid[i * size : (i + 1) * size]) for i in LIFE_ROWS]

    def neighbours(old: list[int], i: int, j: int) -> int:
        n = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                a, b = i + di, j + dj
                if 0 <= a < size and 0 <= b < size:
                    n += old[a * size + b]
        return n

    for _ in range(steps):
        old = grid[:]
        for i in range(size):
            for j in range(size):
                k = i * size + j
                n = neighbours(old, i, j)
                grid[k] = 1 if (old[k] and n in (2, 3)) or (not old[k] and n == 3) else 0
    rows = [sum(grid[i * size : (i + 1) * size]) for i in LIFE_ROWS]
    return grid, initial_alive, initial_rows, rows


def wave(amplitude: float, centre: int, sigma: float, c2: float, size: int, steps: int):
    """Explicit three-level scheme for u_tt = c^2 u_xx, fixed ends."""
    u = [amplitude * math.exp(-((i - centre) / sigma) ** 2) for i in range(size)]
    u[0] = 0.0
    u[size - 1] = 0.0
    prev = list(u)
    initial_energy = wave_energy(u, prev, c2)
    for _ in range(steps):
        nxt = [0.0] * size
        for i in range(1, size - 1):
            nxt[i] = 2 * u[i] - prev[i] + c2 * (u[i - 1] - 2 * u[i] + u[i + 1])
        prev, u = u, nxt
    return u, prev, initial_energy, wave_energy(u, prev, c2)


def wave_energy(u: list[float], prev: list[float], c2: float) -> float:
    energy = 0.0
    for i in range(1, len(u) - 1):
        du = u[i] - prev[i]
        dr = u[i + 1] - u[i]
        energy += 0.5 * du * du + 0.5 * c2 * dr * dr
    return energy


def hash01(x: int, y: int) -> float:
    """The 32-bit hash from the prompt, on unsigned conventions in both languages."""
    mask = 0xFFFFFFFF
    h = (x * 374761393 + y * 668265263) & mask
    h = ((h ^ (h >> 13)) * 1274126177) & mask
    h = (h ^ (h >> 16)) & mask
    return h / 4294967296.0


def noise_height(x: int, y: int, amplitude: float) -> float:
    return amplitude * (hash01(x, y) - 0.5)


def noise_extremes(amplitude: float, half: int) -> tuple[float, float]:
    values = [noise_height(x, y, amplitude) for x in range(-half, half + 1) for y in range(-half, half + 1)]
    return min(values), max(values)


def _terrain_shape(payload: dict) -> tuple[bool, str]:
    """A shape predicate over the terrain probe, defined here because it is local."""
    heights = visual.path_of(payload, "heights", None)
    low = visual.path_of(payload, "min", None)
    high = visual.path_of(payload, "max", None)
    if not isinstance(heights, list) or len(heights) != 3:
        return False, f"heights: не три числа ({heights!r})"
    if not all(isinstance(v, (int, float)) for v in [*heights, low, high]):
        return False, f"heights: не числа ({heights!r})"
    good = low < high and all(low <= v <= high for v in heights)
    return good, f"min = {low}, max = {high}, heights = {heights}"


# --------------------------------------------------------------------------- #
# Frame geometry: the fixtures must be inside the frustum by construction, not
# "by eye" — an object outside the frame fails the image check for a reason the
# model cannot fix.
# --------------------------------------------------------------------------- #

FOV_DEG = 50.0
ASPECT = 800 / 600
FRAME_MARGIN = 0.7


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = math.sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def _camera_basis(camera, look):
    forward = _unit(_sub(look, camera))
    right = _unit(_cross(forward, (0.0, 1.0, 0.0)))
    up = _cross(right, forward)
    return forward, right, up


def _in_frame(point, camera, look, margin: float = FRAME_MARGIN) -> bool:
    """Is ``point`` inside the view frustum, with a margin against the edges?"""
    forward, right, up = _camera_basis(camera, look)
    rel = _sub(point, camera)
    depth = _dot(rel, forward)
    if depth <= 0.0:
        return False
    half_h = depth * math.tan(math.radians(FOV_DEG / 2))
    half_w = half_h * ASPECT
    return abs(_dot(rel, right)) <= margin * half_w and abs(_dot(rel, up)) <= margin * half_h


def _box_corners(x0, x1, y0, y1, z0, z1):
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def _assert_in_frame(camera, look, points, label: str) -> None:
    outside = [p for p in points if not _in_frame(p, camera, look)]
    assert not outside, f"{label}: {len(outside)} точек вне кадра, например {outside[:2]}"


def _assert_literals(prompt: str, *values: str) -> None:
    """Every fixture number must be spelled out in the prompt verbatim."""
    missing = [v for v in values if v not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _task_heat(rng: random.Random) -> core.TaskInstance:
    alpha = float(rng.choice([0.15, 0.2, 0.25]))
    steps = int(rng.choice([40, 60, 80]))
    size, hot, hot_value, dt = 32, 8, 1000.0, 1 / 60
    grid, mean, peak = diffuse(alpha, size, hot, hot_value, steps)
    samples = [grid[8][8], grid[16][16], grid[24][8]]

    assert len({round4(v) for v in samples}) == 3, "три узла обязаны различаться"
    assert 0.0 < samples[1] < samples[0], "тепло обязано растечься от нагретой точки"
    assert 0.0 < samples[2] < samples[1], "дальний узел обязан быть холоднее среднего"
    assert abs(peak - hot_value) > 1.0, "максимум обязан отличаться от начального"

    prompt = PROMPT12.format(
        cam=list(CAMERA12), light=list(LIGHT12), a=alpha, fields=FIELDS12,
    )
    golden = GOLDEN12.substitute(
        alpha=alpha, camera=list(CAMERA12), light=list(LIGHT12),
    )
    _assert_literals(
        prompt,
        f"a = {alpha}",
        "T[8][8] = 1000",
        "[8][8]",
        "[16][16]",
        "[24][8]",
        str(list(CAMERA12)),
        str(list(LIGHT12)),
    )
    _assert_in_frame(
        CAMERA12,
        (0.0, 0.0, 0.0),
        _box_corners(-16.0, 16.0, 0.0, 3.05, -16.0, 16.0),
        "K-12: пластина",
    )
    scene_checks = [
        visual.near("grid", size, 0),
        visual.near("max", hot_value, 1e-4),
        visual.near("mean", round4(hot_value / (size * size)), 1e-4),
        visual.near("samples.0", hot_value, 1e-4),
        visual.near("samples.1", 0.0, 1e-6),
        visual.near("samples.2", 0.0, 1e-6),
    ]
    sim_checks = [
        visual.near("samples.0", round4(samples[0]), 1e-3),
        visual.near("samples.1", round4(samples[1]), 1e-3),
        visual.near("samples.2", round4(samples[2]), 1e-3),
        visual.near("mean", round4(mean), 1e-3),
        visual.near("max", round4(peak), 1e-3),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.09),
        visual.image_min("distinct_colors", 16),
        visual.image_color("warm", 0.012, "горячая область пластины различима"),
        visual.image_min("mean", 12.0),
    ]
    return visual.build_task(
        task_id="K-12",
        title="K-12 — диффузия тепла на пластине 32x32",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=steps, dt=dt),
        golden=golden,
        rng_meta={
            "alpha": alpha, "steps": steps, "dt": dt,
            "mean": round4(mean), "max": round4(peak), "samples": [round4(v) for v in samples],
        },
    )


def _task_fire(rng: random.Random) -> core.TaskInstance:
    seed = int(rng.randrange(1, 1 << 31))
    density = float(rng.choice([0.6, 0.65, 0.7]))
    steps = int(rng.choice([6, 7, 8]))
    size, ignite, dt = 40, (20, 20), 1 / 60
    grid, initial_trees, counts = fire(seed, density, size, steps, ignite)
    trees, burning, burned = counts

    # Measured over 200 seeds per parameter set: burning >= 9 and burned >= 7
    # everywhere in this space, so these thresholds hold for any seed.
    assert burning > 5, "к моменту проверки фронт обязан быть заметным"
    assert burned > 4, "часть клеток обязана уже выгореть"
    assert trees > 0, "лес не должен выгореть целиком"
    assert trees != burning and trees != burned, "состояния обязаны различаться"
    assert burning + burned > 10, "пожар обязан распространиться от единственной затравки"

    prompt = PROMPT13.format(
        cam=list(CAMERA13), light=list(LIGHT13), seed=seed, p=density, fields=FIELDS13,
    )
    golden = GOLDEN13.substitute(
        p=density, seed=seed, camera=list(CAMERA13), light=list(LIGHT13),
    )
    _assert_literals(
        prompt,
        f"seed = {seed}",
        f"меньше {density}",
        "1664525",
        "1013904223",
        "4294967296",
        "(20, 20)",
        str(list(CAMERA13)),
        str(list(LIGHT13)),
    )
    _assert_in_frame(
        CAMERA13,
        (0.0, 0.0, 0.0),
        _box_corners(-19.5, 19.5, 0.0, 0.6, -19.5, 19.5),
        "K-13: лес",
    )
    scene_checks = [
        visual.near("trees", initial_trees, 0),
        visual.near("burning", 1, 0),
        visual.near("burned", 0, 0),
        visual.near("steps", 0, 0),
    ]
    sim_checks = [
        visual.near("trees", trees, 0),
        visual.near("burning", burning, 0),
        visual.near("burned", burned, 0),
        visual.near("steps", steps, 0),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.10),
        visual.image_min("distinct_colors", 8),
        visual.image_color("warm", 0.0014, "горящий фронт различим"),
        visual.image_color("cool", 0.02, "пустые клетки видны как тёмная сетка"),
    ]
    return visual.build_task(
        task_id="K-13",
        title="K-13 — лесной пожар 40x40",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=steps, dt=dt),
        golden=golden,
        rng_meta={
            "seed": seed, "density": density, "steps": steps, "initial_trees": initial_trees,
            "trees": trees, "burning": burning, "burned": burned,
        },
    )


def _task_life(rng: random.Random) -> core.TaskInstance:
    seed = int(rng.randrange(1, 1 << 31))
    density = float(rng.choice([0.3, 0.35, 0.4]))
    steps = int(rng.choice([20, 30, 40]))
    size, dt = 24, 1 / 60
    grid, initial_alive, initial_rows, rows = life(seed, density, size, steps)
    alive = sum(grid)

    assert alive > 0, "поколение обязано быть непустым"
    assert alive != initial_alive, "поколение обязано отличаться от старта"
    assert rows != initial_rows, "три строки обязаны измениться за поколения"

    prompt = PROMPT14.format(
        cam=list(CAMERA14), light=list(LIGHT14), seed=seed, p=density, fields=FIELDS14,
    )
    golden = GOLDEN14.substitute(
        p=density, seed=seed, camera=list(CAMERA14), light=list(LIGHT14),
    )
    _assert_literals(
        prompt,
        f"seed = {seed}",
        f"меньше {density}",
        "1664525",
        "1013904223",
        "4294967296",
        str(list(CAMERA14)),
        str(list(LIGHT14)),
    )
    _assert_in_frame(
        CAMERA14,
        (0.0, 0.0, 0.0),
        _box_corners(-11.5, 11.5, 0.0, 0.4, -11.5, 11.5),
        "K-14: сетка «Жизни»",
    )
    scene_checks = [
        visual.near("generation", 0, 0),
        visual.near("alive", initial_alive, 0),
        visual.size_between("rows", 3, 3),
    ]
    sim_checks = [
        visual.near("generation", steps, 0),
        visual.near("alive", alive, 0),
        visual.near("rows.0", rows[0], 0),
        visual.near("rows.1", rows[1], 0),
        visual.near("rows.2", rows[2], 0),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.05),
        visual.image_min("distinct_colors", 3),
        visual.image_color("warm", 0.005, "живые клетки различимы"),
        visual.image_centroid("warm", 1, 0.0, 0.98, "живые клетки видны в кадре"),
    ]
    return visual.build_task(
        task_id="K-14",
        title="K-14 — игра «Жизнь» 24x24",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=steps, dt=dt),
        golden=golden,
        rng_meta={
            "seed": seed, "density": density, "steps": steps,
            "initial_alive": initial_alive, "alive": alive, "rows": rows,
        },
    )


def _task_wave(rng: random.Random) -> core.TaskInstance:
    amplitude = float(rng.choice([0.8, 1.0, 1.2]))
    centre = int(rng.choice([16, 18]))
    sigma = float(rng.choice([7.0, 8.0]))
    courant = float(rng.choice([0.5, 0.6]))
    steps = 40
    size, dt = 64, 1 / 60
    c2 = courant * courant
    u, _prev, initial_energy, energy = wave(amplitude, centre, sigma, c2, size, steps)
    samples = [u[16], u[32], u[48]]

    # The three probe nodes must stay apart and non-trivial after the steps, or
    # the check would pass on a page that only drew a flat line.
    assert len({round4(v) for v in samples}) == 3, "три узла обязаны различаться"
    assert all(abs(v) > 5e-3 for v in samples), "все три узла обязаны быть возбуждены"
    assert energy > 0.0, "энергия обязана быть положительной"
    assert max(abs(v) for v in u) <= amplitude * 1.05, "схема обязана быть устойчивой"

    prompt = PROMPT15.format(
        cam=list(CAMERA15), light=list(LIGHT15), amp=amplitude, center=centre,
        sigma=sigma, c2=round(c2, 4), fields=FIELDS15,
    )
    golden = GOLDEN15.substitute(
        amp=amplitude, center=centre, sigma=sigma, c2=c2,
        camera=list(CAMERA15), light=list(LIGHT15),
    )
    _assert_literals(
        prompt,
        f"A = {amplitude}",
        f"center = {centre}",
        f"sigma = {sigma}",
        f"c2 = {round(c2, 4)}",
        "радиуса 0.55",
        "((i - 31.5) * 0.7",
        "u[i] * 4.5",
        str(list(CAMERA15)),
        str(list(LIGHT15)),
    )
    _assert_in_frame(
        CAMERA15,
        (0.0, 0.0, 0.0),
        # The whole reachable envelope: every sphere position of every
        # reachable state, not just the ones this fixture happens to visit.
        _box_corners(-22.05, 22.05, -amplitude * 1.05 * 4.5, amplitude * 1.05 * 4.5, -0.7, 0.7),
        "K-15: струна",
    )
    initial_samples = [amplitude * math.exp(-((i - centre) / sigma) ** 2) for i in (16, 32, 48)]
    scene_checks = [
        visual.near("samples.0", round4(initial_samples[0]), 1e-3),
        visual.near("samples.1", round4(initial_samples[1]), 1e-3),
        visual.near("samples.2", round4(initial_samples[2]), 1e-3),
        visual.near("energy", round4(initial_energy), 1e-3),
    ]
    sim_checks = [
        visual.near("samples.0", round4(samples[0]), 1e-3),
        visual.near("samples.1", round4(samples[1]), 1e-3),
        visual.near("samples.2", round4(samples[2]), 1e-3),
        visual.near("energy", round4(energy), 1e-3),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.005),
        visual.image_min("distinct_colors", 16),
        visual.image_color("warm", 0.0007, "струна различима"),
        visual.image_color("bright", 0.0005, "профиль струны светлый"),
    ]
    return visual.build_task(
        task_id="K-15",
        title="K-15 — волновое уравнение на струне",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=steps, dt=dt),
        golden=golden,
        rng_meta={
            "amplitude": amplitude, "centre": centre, "sigma": sigma,
            "courant": courant, "c2": c2, "steps": steps,
            "samples": [round4(v) for v in samples], "energy": round4(energy),
            "initial_energy": round4(initial_energy),
        },
    )


def _task_terrain(rng: random.Random) -> core.TaskInstance:
    amplitude = float(rng.choice([3.0, 4.0, 5.0]))
    steps = int(rng.choice([12, 18, 24]))
    half, dt = 20, 1 / 60
    low, high = noise_extremes(amplitude, half)
    heights = [noise_height(x, y, amplitude) for x, y in ((2, 3), (10, 10), (20, 5))]

    assert len({round4(v) for v in heights}) == 3, "три узла обязаны различаться"
    assert low < -0.2 * amplitude and high > 0.2 * amplitude, "рельеф обязан быть выраженным"
    assert all(low < v < high for v in heights), "узлы обязаны лежать внутри диапазона"

    prompt = PROMPT16.format(
        cam=list(CAMERA16), light=list(LIGHT16), amp=amplitude, fields=FIELDS16,
    )
    golden = GOLDEN16.substitute(amp=amplitude, camera=list(CAMERA16), light=list(LIGHT16))
    _assert_literals(
        prompt,
        f"A = {amplitude}",
        "374761393",
        "668265263",
        "1274126177",
        str(list(CAMERA16)),
        str(list(LIGHT16)),
    )
    _assert_in_frame(
        CAMERA16,
        (0.0, 0.0, 0.0),
        _box_corners(-20.0, 20.0, low, high, -20.0, 20.0),
        "K-16: рельеф",
    )
    scene_checks = [
        _terrain_shape,
        visual.between("min", -0.5 * amplitude - 0.05, -0.2 * amplitude),
        visual.between("max", 0.2 * amplitude, 0.5 * amplitude + 0.05),
    ]
    sim_checks = [
        visual.near("heights.0", round4(heights[0]), 1e-3),
        visual.near("heights.1", round4(heights[1]), 1e-3),
        visual.near("heights.2", round4(heights[2]), 1e-3),
        visual.near("min", round4(low), 1e-3),
        visual.near("max", round4(high), 1e-3),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.05),
        visual.image_min("distinct_colors", 4),
        visual.image_color("warm", 0.05, "рельеф занимает заметную часть кадра"),
        visual.image_min("mean", 12.0),
    ]
    return visual.build_task(
        task_id="K-16",
        title="K-16 — терраин из сеяного вэлью-шума",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=steps, dt=dt),
        golden=golden,
        rng_meta={
            "amplitude": amplitude, "steps": steps,
            "heights": [round4(v) for v in heights], "min": round4(low), "max": round4(high),
        },
    )


BUILDERS = (_task_heat, _task_fire, _task_life, _task_wave, _task_terrain)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Every scenario of the slice, each drawing its fixture from the same stream."""
    return [builder(rng) for builder in BUILDERS]
