"""Visual scenarios K-33..K-34 — a finite-difference PDE and an Atwood machine.

Two scenarios of opposite temper: the first is a grid scheme whose *invariant*
is the point, the second is a closed-form mechanism whose acceleration formula
is the point.

K-33 integrates the 2D heat equation explicitly with zero-flux (Neumann) edges:
a ghost neighbour behind the border mirrors the boundary cell, so the total
energy of the plate is conserved by construction. The probe publishes three
nodes and the energy; the oracle is a Python replay of the same scheme *and*
the conservation invariant, read as a predicate of its own. A clamped (Dirichlet)
border and a mass loss both break the invariant and are rejected.

K-34 is an Atwood machine with a massive pulley: ``a = (m1 - m2) g /
(m1 + m2 + J/R^2)`` and the state is that constant acceleration integrated in
closed form. The inertia term is what makes the task non-trivial — dropping it
or flipping the mass difference are the two published defects.

Both cameras are fixed: the plate is seen at an angle from above, the machine
through an orthographic front view, so the two pans keep their pixel column and
the frame can be read back independently of the probe.
"""

from __future__ import annotations

import math
import random
from string import Template

from .. import core, visual

IDS = ("K-33", "K-34")

#: Both scenarios live in the stretch suite only.
IDS_BY_PROFILE = {"stretch-v1": IDS}

DT = 1 / 60

#: K-33 grid and its two hot cells. ``ALPHA`` stays below 1/4, so the explicit
#: five-point stencil is stable for every fixture.
SIZE = 24
HOT_POS = ((5, 7), (11, 16))
SAMPLES = ((4, 7), (6, 7), (12, 16))
ALPHA = (0.16, 0.2, 0.22)
CAMERA33 = (0.0, 30.0, 30.0)
LOOK33 = (0.0, 0.0, 0.0)
LIGHT33 = (11.0, 27.0, 15.0)

#: K-34 frame: orthographic, world units, 800x600 keeps the 4:3 aspect.
ORTHO = (-6.5, 6.5, 4.875, -4.875)
CAMERA34 = (0.0, 4.0, 20.0)
LOOK34 = (0.0, 4.0, 0.0)
LIGHT34 = (5.0, 12.0, 14.0)
AXIS = (0.0, 7.5)
PAN = 2.0
GRAVITY = 9.81

FIELDS33 = """{
  "grid": 24,
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "mean": <средняя температура по всей сетке, 4 знака>,
  "max": <максимальная температура, 4 знака>,
  "energy": <сумма всех температур, 4 знака>,
  "samples": [T[4][7], T[6][7], T[12][16]]
}"""

FIELDS34 = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["floor", "beam", "pulley", "rope-left", "rope-right", "mass-1", "mass-2"],
  "camera": [0, 4, 20],
  "x1": <путь левой гири вниз от оси, м>,
  "x2": <путь правой гири вниз от оси, м>,
  "v": <скорость, м/с>,
  "a": <ускорение, м/с²>,
  "m1": <масса 1>,
  "m2": <масса 2>,
  "R": <радиус шкива>,
  "J": <момент инерции шкива>
}"""

PROMPT33 = """Собери сцену явной теплопроводности на квадратной пластине 24x24:
периметр теплоизолирован (нулевой поток), тепло растекается от двух нагретых
точек, пластина нарисована столбиками.

Сцена:
* пластина — 576 столбиков, по одному на ячейку `(i, j)` (i — строка,
  j — столбец, обе от 0 до 23), всё одним `THREE.InstancedMesh` с именем
  `plate`; основание кубика 0.92x0.92, ячейка `(i, j)` стоит в точке
  `(j - 11.5, h/2, i - 11.5)` с высотой `h = 0.05 + 3 * t`, где `t` —
  температура ячейки, делённая на максимальную температуру текущего состояния;
  цвет столбика — `lerp(0x14306b, 0xff5a1e, t)`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 0, 0); свет — `AmbientLight(0xffffff, 0.55)` и
  `DirectionalLight(0xffffff, 1.6)` из точки {light}.

Симуляция:
* температура хранится как 576 чисел, строка за строкой, индекс `i*24 + j`;
  есть два массива — текущий `T` и буфер `next` того же размера;
* в начале всё нули, кроме `T[5][7] = {hot1}` и `T[11][16] = {hot2}`;
* шаг делается только вызовом `window.__step(dt)`; `dt` на схему не влияет —
  один вызов это ровно один шаг;
* состояние рисуется в `window.__probe()`: сначала `paint()`, потом
  `renderer.render(scene, camera)`, и только затем возвращаются поля;
  `window.__step(dt)` состояние только продвигает и ничего не рисует;
* схема шага — явная по двум массивам: соседи читаются из `T`, результат
  пишется в `next`, затем массивы меняются местами; проход `i = 0..23`, внутри
  `j = 0..23`, ровно в таком порядке, `a = {alpha}`:
  `c = T[i][j]; next[i][j] = c + a * (up + dn + lf + rt - 4 * c)`,
  где `up`, `dn`, `lf`, `rt` — соседи `T[i-1][j]`, `T[i+1][j]`, `T[i][j-1]`,
  `T[i][j+1]`; за границей сетки сосед равен самой ячейке — это нулевой поток,
  то есть край остаётся теплоизолированным и, в частности,
  `next[0][j] = c + a * (T[1][j] + lf + rt - 3 * c)`;
* энергия пластины — просто сумма всех 576 температур;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: размер сетки, состояние и энергия сразу после загрузки,
температуры в трёх узлах `T[4][7]`, `T[6][7]`, `T[12][16]`, среднее, максимум и
энергия после заданного числа шагов (сверка с этой же схемой, допуск 1e-3,
энергия обязана сохраниться), и кадр — пластина видна, горячая область
отличается по цвету, картинка воспроизводима при повторной загрузке.
"""

PROMPT34 = """Собери сцену машины Атвуда: через шкив радиуса {R} с моментом инерции
{J} перекинута нерастяжимая нить, на её концах гири {m1} кг (слева) и {m2} кг
(справа). Левая гиря тяжелее и опускается, правая поднимается; нить по шкиву
не проскальзывает, шкив раскручивается.

Сцена (плоскость XY, камера ортографическая, фронтально по оси Z):
* ось шкива — в точке {axis}; шкив — диск радиуса {R} толщиной 0.25 цвета
  0x9a9a9a с именем `pulley` (ось диска вдоль Z), на ободе метка 0.12x0.2
  цвета 0xffd27f, повёрнутая вместе со шкивом;
* балка — `BoxGeometry(5, 0.4, 0.4)` цвета 0x4a4f57 в точке (0, {beam_y}, 0)
  с именем `beam`, шкив висит под ней;
* гири — кубы 0.6x0.6x0.6: левая цвета 0xff8a3d с именем `mass-1` в точке
  (-{pan}, {axis_y} - x1, 0), правая цвета 0x3d7bff с именем `mass-2` в точке
  ({pan}, {axis_y} - x2, 0); `x1` и `x2` — пути гирь вниз от оси шкива;
* нити — два цилиндра радиуса 0.05 цвета 0xbfbfbf с именами `rope-left` и
  `rope-right`: от точек касания (-{r_tan}, {axis_y}) и ({r_tan}, {axis_y}) до
  верхней грани своей гири;
* пол — `BoxGeometry(20, 0.4, 1)` цвета 0x2b2b2b в точке (0, -0.7, 0) с именем
  `floor`;
* камера `THREE.OrthographicCamera({left}, {right}, {top}, {bottom}, 0.1, 100)`
  в точке {cam} и смотрит в {look}; свет — `AmbientLight(0xffffff, 0.75)` и
  `DirectionalLight(0xffffff, 1.1)` из точки {light}; фон 0x121212.

Симуляция (аналитика, время — только число шагов):
* `t = steps / 60`, шаг `dt = 1/60`, всего {steps} шагов; состояние зависит
  только от `steps`, накопления нет;
* ускорение постоянно: `a = (m1 - m2) * g / (m1 + m2 + J / (R * R))` при
  g = {g}, m1 = {m1}, m2 = {m2}, J = {J}, R = {R};
* начальная скорость нулевая, начальные пути `x1 = {x1_0}`, `x2 = {x2_0}`;
  тогда `v = a * t`, `x1 = {x1_0} + a * t^2 / 2`, `x2 = {x2_0} - a * t^2 / 2`;
* шкив поворачивается на угол `(x1 - {x1_0}) / R`;
* шаг делает только `window.__step(dt)`; `window.__probe()` сначала
  пересчитывает положения гирь, нити, шкива и метки, потом
  `renderer.render(scene, camera)`, и только затем возвращает поля;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие всех перечисленных
имён, лишние имена не мешают.

Проверяется: состав сцены и камера, пути, скорость и ускорение после
заданного числа шагов (сверка с этой формулой, допуск 1e-3), и кадр — обе гири
различимы по цвету и стоят на своих сторонах, картинка воспроизводима при
повторной загрузке.
"""

GOLDEN33 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>heat plate neumann</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = 24;
const ALPHA = $alpha;

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

let T = new Float64Array(N * N);
let next = new Float64Array(N * N);
T[5 * N + 7] = $hot1;
T[11 * N + 16] = $hot2;

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

function at(i, j) {
  const ii = i < 0 ? 0 : (i > N - 1 ? N - 1 : i);
  const jj = j < 0 ? 0 : (j > N - 1 ? N - 1 : j);
  return T[ii * N + jj];
}

window.__step = (dt) => {
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j;
      const c = T[k];
      next[k] = c + ALPHA * (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - 4 * c);
    }
  }
  const swap = T; T = next; next = swap;
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
    steps,
    t: round(steps / 60),
    mean: round(sum / (N * N)),
    max: round(peak),
    energy: round(sum),
    samples: [round(T[4 * N + 7]), round(T[6 * N + 7]), round(T[12 * N + 16])],
  };
};

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN34 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>atwood</title>
<style>html,body{margin:0;background:#121212;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const G = $g, M1 = $m1, M2 = $m2, J = $j, R = $r;
const X1 = $x1_0, X2 = $x2_0;
const AXIS_Y = $axis_y, PAN = $pan;
const ACC = (M1 - M2) * G / (M1 + M2 + J / (R * R));

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x121212);

const camera = new THREE.OrthographicCamera($left, $right, $top, $bottom, 0.1, 100);
camera.position.set(...$camera);
camera.lookAt(...$look);
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(...$light);
scene.add(key);

const floor = new THREE.Mesh(
  new THREE.BoxGeometry(20, 0.4, 1),
  new THREE.MeshStandardMaterial({color: 0x2b2b2b})
);
floor.position.set(0, -0.7, 0);
floor.name = 'floor';
scene.add(floor);

const beam = new THREE.Mesh(
  new THREE.BoxGeometry(5, 0.4, 0.4),
  new THREE.MeshStandardMaterial({color: 0x4a4f57})
);
beam.position.set(0, AXIS_Y + 0.65, 0);
beam.name = 'beam';
scene.add(beam);

const pulley = new THREE.Group();
pulley.name = 'pulley';
pulley.position.set(0, AXIS_Y, 0);
scene.add(pulley);

const disc = new THREE.Mesh(
  new THREE.CylinderGeometry(R, R, 0.25, 32),
  new THREE.MeshStandardMaterial({color: 0x9a9a9a})
);
disc.rotation.x = Math.PI / 2;
pulley.add(disc);

const mark = new THREE.Mesh(
  new THREE.BoxGeometry(0.12, 0.2, 0.3),
  new THREE.MeshStandardMaterial({color: 0xffd27f})
);
mark.position.set(0.68 * R, 0, 0);
pulley.add(mark);

function pan(tag, hex) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.6, 0.6, 0.6),
    new THREE.MeshStandardMaterial({color: hex})
  );
  mesh.name = tag;
  scene.add(mesh);
  return mesh;
}

const mass1 = pan('mass-1', 0xff8a3d);
const mass2 = pan('mass-2', 0x3d7bff);

function rope(tag) {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.05, 0.05, 1, 8),
    new THREE.MeshStandardMaterial({color: 0xbfbfbf})
  );
  mesh.name = tag;
  scene.add(mesh);
  return mesh;
}

const ropeLeft = rope('rope-left');
const ropeRight = rope('rope-right');

function link(mesh, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  mesh.position.set((ax + bx) / 2, (ay + by) / 2, 0);
  mesh.rotation.z = Math.atan2(-dx, dy);
  mesh.scale.y = Math.hypot(dx, dy);
}

let steps = 0;

function draw() {
  const t = steps / 60;
  const x1 = X1 + ACC * t * t / 2;
  const x2 = X2 - ACC * t * t / 2;
  mass1.position.set(-PAN, AXIS_Y - x1, 0);
  mass2.position.set(PAN, AXIS_Y - x2, 0);
  link(ropeLeft, -R, AXIS_Y, -PAN, AXIS_Y - x1 + 0.3);
  link(ropeRight, R, AXIS_Y, PAN, AXIS_Y - x2 + 0.3);
  pulley.rotation.z = -(x1 - X1) / R;
  renderer.render(scene, camera);
}

const round = (v) => Math.round(v * 1e4) / 1e4;

window.__step = (dt) => {
  steps += 1;
};

window.__probe = () => {
  draw();
  const t = steps / 60;
  return {
    steps,
    t: round(t),
    objects: scene.children.map((o) => o.name).filter(Boolean),
    camera: [camera.position.x, camera.position.y, camera.position.z],
    x1: round(X1 + ACC * t * t / 2),
    x2: round(X2 - ACC * t * t / 2),
    v: round(ACC * t),
    a: round(ACC),
    m1: M1,
    m2: M2,
    R: R,
    J: J,
  };
};

draw();
window.__ready = true;
</script>
</body></html>
""")


def round4(value: float) -> float:
    return round(value + 0.0, 4)


def _in_frame(point: tuple[float, float, float], camera: tuple[float, float, float],
              look: tuple[float, float, float], margin: float = 0.88) -> bool:
    """Is ``point`` inside the perspective frame, with a margin at the border?"""
    fov, aspect = 50.0, 800 / 600
    fwd = [look[i] - camera[i] for i in range(3)]
    norm = math.sqrt(sum(v * v for v in fwd))
    fwd = [v / norm for v in fwd]
    right = [fwd[2], 0.0, -fwd[0]]
    rn = math.sqrt(right[0] ** 2 + right[2] ** 2) or 1.0
    right = [right[0] / rn, 0.0, right[2] / rn]
    up = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ]
    delta = [point[i] - camera[i] for i in range(3)]
    depth = sum(delta[i] * fwd[i] for i in range(3))
    if depth <= 0.1:
        return False
    half_h = math.tan(math.radians(fov / 2)) * depth
    half_w = half_h * aspect
    x = sum(delta[i] * right[i] for i in range(3))
    y = sum(delta[i] * up[i] for i in range(3))
    return abs(x) <= half_w * margin and abs(y) <= half_h * margin


def _assert_in_frame(points, label: str) -> None:
    outside = [p for p in points if not _in_frame(p, CAMERA33, LOOK33)]
    assert not outside, f"{label}: {len(outside)} точек вне кадра, например {outside[:2]}"


def _assert_in_ortho(points, label: str) -> None:
    left, right, top, bottom = ORTHO
    margin = 0.92
    outside = [
        p for p in points
        if abs(p[0]) > right * margin
        or not (LOOK34[1] + bottom * margin <= p[1] <= LOOK34[1] + top * margin)
    ]
    assert not outside, f"{label}: {len(outside)} точек вне кадра, например {outside[:2]}"


def _assert_literals(prompt: str, *values: str) -> None:
    missing = [value for value in values if value not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


def _energy(grid: list[float]) -> float:
    return sum(grid)


def heat_state(
    alpha: float, steps: int, hot1: float, hot2: float, clamp: bool = False,
) -> list[float]:
    """The stated scheme, replayed in Python: Jacobi with mirrored ghosts.

    ``clamp`` swaps the mirrored ghost for a zero one — a Dirichlet border,
    which is the published defect: it drains heat through the rim.
    """
    size = SIZE
    grid = [0.0] * (size * size)
    for (i, j), value in zip(HOT_POS, (hot1, hot2)):
        grid[i * size + j] = value

    def at(i: int, j: int) -> float:
        if clamp and (i < 0 or j < 0 or i > size - 1 or j > size - 1):
            return 0.0
        ii = min(max(i, 0), size - 1)
        jj = min(max(j, 0), size - 1)
        return grid[ii * size + jj]

    for _ in range(steps):
        nxt = list(grid)
        for i in range(size):
            for j in range(size):
                k = i * size + j
                c = grid[k]
                nxt[k] = c + alpha * (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - 4 * c)
        grid = nxt
    return grid


def _build_k33(rng: random.Random) -> core.TaskInstance:
    alpha = float(rng.choice(ALPHA))
    steps = int(rng.choice([90, 120]))
    hot1 = float(rng.choice([900.0, 1000.0, 1100.0]))
    hot2 = float(rng.choice([250.0, 300.0, 350.0]))
    initial = hot1 + hot2

    grid = heat_state(alpha, steps, hot1, hot2)
    samples = [grid[i * SIZE + j] for i, j in SAMPLES]
    mean = _energy(grid) / (SIZE * SIZE)
    peak = max(grid)
    energy = _energy(grid)

    # The invariant is the point of the scenario: with a mirrored border the
    # total heat never leaves the plate. And the three nodes must stay apart and
    # warm, otherwise a flat plate would pass the replay check.
    assert abs(energy - initial) <= 1e-6, "нулевой поток обязан сохранять энергию"
    assert len({round4(v) for v in samples}) == 3, "три узла обязаны различаться"
    assert all(v > 5e-3 for v in samples), "все три узла обязаны прогреться"
    assert peak < 1000.0, "максимум обязан упасть: тепло растекается"

    prompt = PROMPT33.format(
        cam=list(CAMERA33), light=list(LIGHT33), alpha=alpha, hot1=hot1, hot2=hot2,
        fields=FIELDS33,
    )
    _assert_literals(
        prompt, f"a = {alpha}", f"T[5][7] = {hot1}", f"T[11][16] = {hot2}",
        str(list(CAMERA33)), str(list(LIGHT33)),
    )
    _assert_in_frame(
        [(x, y, z) for x in (-11.5, 11.5) for y in (0.0, 3.05) for z in (-11.5, 11.5)],
        "K-33: пластина",
    )
    golden = GOLDEN33.substitute(
        alpha=alpha, hot1=hot1, hot2=hot2, camera=list(CAMERA33), light=list(LIGHT33),
    )

    scene_checks = [
        visual.near("grid", SIZE, 0),
        visual.near("steps", 0, 0),
        visual.near("max", hot1, 1e-4),
        visual.near("energy", initial, 1e-4),
        visual.near("mean", round4(initial / (SIZE * SIZE)), 1e-4),
        visual.near("samples.0", 0.0, 1e-9),
        visual.near("samples.1", 0.0, 1e-9),
        visual.near("samples.2", 0.0, 1e-9),
    ]
    sim_checks = [
        visual.near("steps", steps, 0),
        visual.near("t", round4(steps * DT), 1e-3),
        visual.near("samples.0", round4(samples[0]), 1e-3),
        visual.near("samples.1", round4(samples[1]), 1e-3),
        visual.near("samples.2", round4(samples[2]), 1e-3),
        visual.near("mean", round4(mean), 1e-3),
        visual.near("max", round4(peak), 1e-3),
        # The conservation invariant, as published: a clamped border or a mass
        # loss leaves the plate with less heat than it started with.
        visual.near("energy", round4(initial), 1e-3),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.09),
        visual.image_min("distinct_colors", 16),
        visual.image_color("warm", 0.008, "горячая область различима"),
        visual.image_centroid("warm", 0, 0.2, 0.7, "горячая область в кадре"),
        visual.image_min("mean", 12.0),
    ]
    return visual.build_task(
        task_id="K-33",
        title="K-33 — теплопроводность с нулевым потоком на краю",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks, simulation=sim_checks, image=image_checks,
            steps=steps, dt=DT,
        ),
        golden=golden,
        rng_meta={
            "alpha": alpha, "steps": steps, "hot1": hot1, "hot2": hot2,
            "samples": [round4(v) for v in samples],
            "energy": round4(energy), "initial_energy": round4(initial),
            "max": round4(peak), "mean": round4(mean),
        },
    )


def atwood_state(
    m1: float, m2: float, inertia: float, radius: float,
    x1: float, x2: float, steps: int,
) -> tuple[float, float, float, float]:
    """Closed form of the machine, independent of the reference page."""
    acc = (m1 - m2) * GRAVITY / (m1 + m2 + inertia / (radius * radius))
    t = steps * DT
    return acc, acc * t, x1 + acc * t * t / 2, x2 - acc * t * t / 2


def _build_k34(rng: random.Random) -> core.TaskInstance:
    for _ in range(80):
        m1 = round(rng.uniform(0.8, 1.1), 2)
        m2 = round(rng.uniform(0.35, 0.55), 2)
        radius = float(rng.choice([0.2, 0.25, 0.3]))
        inertia = round(0.5 * radius * radius * rng.uniform(0.8, 1.2), 4)
        x1_0 = round(rng.uniform(2.8, 3.2), 2)
        x2_0 = round(rng.uniform(4.4, 4.8), 2)
        steps = 90
        acc, _v, _x1, _x2 = atwood_state(m1, m2, inertia, radius, x1_0, x2_0, steps)
        if 1.5 <= acc <= 3.0:
            break
    else:  # pragma: no cover - the ranges are wide enough for a hit every time
        raise AssertionError("не удалось подобрать фикстуру K-34")

    acc, speed, x1, x2 = atwood_state(m1, m2, inertia, radius, x1_0, x2_0, steps)
    t = round4(steps * DT)
    naive = (m1 - m2) * GRAVITY / (m1 + m2)
    # The inertia term has to matter: without it the answer is a different one.
    assert abs(naive - acc) > 0.5, "инерция шкива обязана менять ускорение"
    assert x1 > x2 and x2 > 0.0, "гири обязаны разойтись и не встретиться"

    prompt = PROMPT34.format(
        R=radius, J=inertia, m1=m1, m2=m2, axis=list(AXIS), axis_y=AXIS[1],
        beam_y=AXIS[1] + 0.65, pan=PAN, r_tan=round4(radius), left=ORTHO[0],
        right=ORTHO[1], top=ORTHO[2], bottom=ORTHO[3], cam=list(CAMERA34),
        look=list(LOOK34), light=list(LIGHT34), g=GRAVITY, x1_0=x1_0, x2_0=x2_0,
        steps=steps, fields=FIELDS34,
    )
    _assert_literals(
        prompt, f"m1 = {m1}", f"m2 = {m2}", f"J = {inertia}", f"R = {radius}",
        f"x1 = {x1_0}", f"x2 = {x2_0}", str(list(CAMERA34)),
    )
    _assert_in_ortho(
        [
            (x, y, 0.0)
            for x in (-PAN - 0.3, -PAN + 0.3, PAN - 0.3, PAN + 0.3)
            for y in (AXIS[1] - x1 - 0.3, AXIS[1] - x2 + 0.3)
        ]
        + [(0.0, AXIS[1] + 0.65 + 0.2, 0.0), (0.0, AXIS[1] + radius, 0.0)],
        "K-34: машина",
    )
    golden = GOLDEN34.substitute(
        g=GRAVITY, m1=m1, m2=m2, j=inertia, r=radius, x1_0=x1_0, x2_0=x2_0,
        left=ORTHO[0], right=ORTHO[1], top=ORTHO[2], bottom=ORTHO[3],
        camera=list(CAMERA34), look=list(LOOK34), light=list(LIGHT34),
        axis=list(AXIS), axis_y=AXIS[1], pan=PAN,
    )

    scene_checks = [
        visual.contains_all(
            "objects",
            ["floor", "beam", "pulley", "rope-left", "rope-right", "mass-1", "mass-2"],
        ),
        visual.near("camera.1", CAMERA34[1], 1e-4),
        visual.near("camera.2", CAMERA34[2], 1e-4),
        visual.near("steps", 0, 0),
        visual.near("x1", x1_0, 1e-9),
        visual.near("x2", x2_0, 1e-9),
        visual.near("v", 0.0, 1e-9),
        visual.near("a", round4(acc), 1e-3),
        visual.near("m1", m1, 1e-4),
        visual.near("m2", m2, 1e-4),
        visual.near("R", radius, 1e-4),
        visual.near("J", inertia, 1e-4),
    ]
    sim_checks = [
        visual.near("steps", steps, 0),
        visual.near("t", t, 1e-3),
        visual.near("x1", round4(x1), 1e-3),
        visual.near("x2", round4(x2), 1e-3),
        visual.near("v", round4(speed), 1e-3),
        visual.near("a", round4(acc), 1e-3),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.004),
        visual.image_min("distinct_colors", 5),
        # The two pans keep their sides: the warm one is left, the cool one right.
        visual.image_centroid("warm", 0, 0.25, 0.45, "левая гиря слева"),
        visual.image_centroid("cool", 0, 0.55, 0.75, "правая гиря справа"),
    ]
    return visual.build_task(
        task_id="K-34",
        title="K-34 — машина Атвуда со шкивом",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks, simulation=sim_checks, image=image_checks,
            steps=steps, dt=DT,
        ),
        golden=golden,
        rng_meta={
            "m1": m1, "m2": m2, "J": inertia, "R": radius,
            "x1_0": x1_0, "x2_0": x2_0, "steps": steps,
            "a": round4(acc), "v": round4(speed),
            "x1": round4(x1), "x2": round4(x2), "naive_a": round4(naive),
        },
    )


BUILDERS = (_build_k33, _build_k34)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Both scenarios, each drawing its fixture from the same stream."""
    return [builder(rng) for builder in BUILDERS]
