"""Visual scenarios K-29..K-30 — статика фермы и PD-серво.

Same shape as ``visual_fields``: the prompt states the scene, the exact scheme
and the exact probe fields, ``build()`` draws the fixture numbers from ``rng``
and renders *both* the prompt and the reference page from them, and the expected
numbers are recomputed here in Python from the stated scheme — a check asks
"does the probe obey the equilibrium / the controller it was given", never
"does it look like the reference".

K-29 is a statically determinate triangle: reactions and member forces follow
from six joint-equilibrium equations, solved here with an independent Gaussian
elimination. K-30 is a PD servo whose target is changed by a host event in the
middle of the run; the Python replay integrates the same clamped controller.

Both reference pages use a fixed orthographic camera, so the drawing is a stable
structural diagram and two answers are compared on the same picture.
"""

from __future__ import annotations

import math
import random
from string import Template

from .. import core, visual

IDS = ("K-29", "K-30")

#: ``pilot`` carries the scenarios that are not admitted to the ranking yet;
#: ``stretch`` grows the showcase to every new scenario.
IDS_BY_PROFILE = {"pilot": IDS, "stretch-v1": IDS}

#: Orthographic frustum (left, right, top, bottom) and camera of each scene.
ORTHO29 = (-10.0, 10.0, 7.5, -7.5)
ORTHO30 = (-10.0, 10.0, 6.0, -6.0)
CAMERA29 = (0.0, 0.0, 30.0)
CAMERA30 = (0.0, 0.0, 30.0)
LIGHT29 = (8.0, 12.0, 18.0)
LIGHT30 = (6.0, 10.0, 16.0)

#: Truss geometry (XY plane, z = 0). A is pinned, B rides on a roller.
NODE_A = (-4.0, 0.0)
NODE_B = (4.0, 0.0)
MEMBERS = ("AB", "AC", "BC")
GROUND_HALF29 = 6.0

#: Servo fixture. The target is retargeted once, at step 60 of 120.
DT = 1 / 60
STEPS30 = 120
RETARGET_AT = 60
X0 = 0.0
RAIL_HALF = 8.0
U_MAX = 5.0

FIELDS29 = """{
  "nodes": {"A": [-4, 0], "B": [4, 0], "C": [cx, 3]},
  "members": ["AB", "AC", "BC"],
  "load": <P>,
  "reactions": [Ax, Ay, By],
  "forces": [F_AB, F_AC, F_BC],
  "states": [<"tension" или "compression" для AB, AC, BC>],
  "residual": [ΣFx, ΣFy, ΣM_A],
  "steps": <сколько раз вызвали __step>
}"""

FIELDS30 = """{
  "t": <время в секундах, 6 знаков>,
  "steps": <сколько раз вызвали __step>,
  "target": <текущая цель>,
  "x": <позиция каретки>,
  "v": <скорость каретки>,
  "u": <последнее управляющее ускорение после clamp>,
  "saturated_steps": <сколько шагов |u_raw| > u_max>,
  "rail_half": 8
}"""

PROMPT29 = """Собери статическую сцену треугольной фермы и посчитай её
равновесие. Кадр статический: t = 0, время не идёт.

Сцена (плоскость XY, z = 0):
* узлы фермы: A = {ax}, B = {bx}, C = ({cx}, 3.0);
* стержни AB, AC, BC — цилиндры радиуса 0.09 между узлами, цвет по знаку
  усилия: растяжение — 0x59d98a, сжатие — 0xff6b3d;
* узлы — сферы радиуса 0.18 цвета 0xf2f2f2; в A шарнирная опора (треугольник
  под узлом), в B катковая опора (два катка под узлом), обе 0x5aa0ff;
* в узле C — стрелка нагрузки вниз длиной 1.4;
* земля — полоса от x = -6.0 до x = 6.0 на y = -0.9;
* камера `OrthographicCamera(-10, 10, 7.5, -7.5, 0.1, 200)` в точке {cam}
  смотрит в (0, 0, 0); свет — `AmbientLight(0xffffff, 0.7)` и
  `DirectionalLight(0xffffff, 1.5)` из точки {light}.

Равновесие:
* в узле C приложена вертикальная вниз сила P = {load};
* A — шарнир с реакциями Ax, Ay; B — каток на горизонтальной плоскости,
  только вертикальная реакция By; реакции положительны вправо и вверх;
* стержни несут только осевое усилие: F положительна при растяжении и
  отрицательна при сжатии; на узел стержень действует вдоль стержня ОТ узла;
* реши равновесие каждого узла: сумма сил в узле равна нулю (по два уравнения
  на узел, шесть неизвестных Ax, Ay, By, F_AB, F_AC, F_BC);
* невязка мировой системы — residual = [ΣFx, ΣFy, ΣM_A]: суммы по всем силам
  (обе реакции, нагрузка и оба конца каждого стержня), момент — по абсолютным
  координатам относительно A.

Симуляция:
* `window.__step(dt)` ничего не меняет — сцена статична, — но считает вызовы;
* `window.__probe()` сначала `renderer.render(scene, camera)`, потом
  возвращает ровно такие поля (числа — с 4 знаками, residual — с 6):

{fields}

Проверяется: узлы, стержни и нагрузка, реакции, усилия и состояния
растяжение/сжатие после одного шага (сверка с решением равновесия, допуск
1e-3), невязка (допуск 1e-6) и кадр — ферма видна и различима.
"""

PROMPT30 = """Собери сцену PD-серво с ограничением управления: каретка едет по
рельсу к цели, цель меняется host-событием ровно на шаге 60 из 120.

Сцена (плоскость XY, z = 0):
* рельс — балка от x = -8.0 до x = 8.0 на y = -0.35, цвет 0x394152;
* каретка — куб 0.9x0.6x0.6 в точке `(x, 0, 0)`, цвет 0xffb347;
* цель — вертикальная полоса высотой 1.8 в точке `(target, 0.55, 0)`, цвет
  0x5aa0ff;
* управляющее ускорение — стрелка из каретки по x: длина
  `0.3 + 1.4 * |u| / u_max`, направление по знаку `u`;
* камера `OrthographicCamera(-10, 10, 6, -6, 0.1, 200)` в точке {cam}
  смотрит в (0, 0, 0); свет — `AmbientLight(0xffffff, 0.75)` и
  `DirectionalLight(0xffffff, 1.4)` из точки {light}.

Симуляция:
* состояние: `x` (позиция), `v` (скорость), `target` (цель), `step`;
  старт `x = {x0}`, `v = 0`, `target = {target_a}`, `step = 0`;
* шаг делается только вызовом `window.__step(dt)`, ровно `dt = {dt}` секунд;
* PD-закон: `u_raw = {kp} * (target - x) - {kd} * v`;
  `u = clamp(u_raw, -{umax}, {umax})`; затем `v += u * dt`, `x += v * dt`;
  `saturated_steps` увеличивается на 1, когда `|u_raw| > {umax}`;
* на шаге с номером 60, ПЕРЕД применением закона, страница обязана отправить
  host-событие `window.dispatchEvent(new CustomEvent('bench:target',
  {{ detail: {{ target: {target_b} }} }}))`; цель меняет только слушатель
  `window.addEventListener('bench:target', (e) => {{ target = e.detail.target; }})`,
  сама симуляция новую цель не подставляет;
* `window.__probe()` сначала перерисовывает сцену
  (`renderer.render(scene, camera)`), потом возвращает ровно такие поля
  (числа — с 6 знаками, кроме `steps`, `rail_half`):

{fields}

Проверяется: состояние в t = 0, позиция, скорость, управление, цель, число
насыщенных шагов и время после 120 шагов (сверка с этой же схемой, допуск
1e-3) и кадр — каретка, цель и стрелка видны.
"""

GOLDEN29 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>truss</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const A = [$ax, $ay], B = [$bx, $by], C = [$cx, $cy];
const LOAD = $load;
const REACTIONS = [$rax, $ray, $rby];
const FORCES = {AB: $fab, AC: $fac, BC: $fbc};
const P = {A, B, C};
const EDGES = [['AB', 'A', 'B'], ['AC', 'A', 'C'], ['BC', 'B', 'C']];

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.OrthographicCamera(-10, 10, 7.5, -7.5, 0.1, 200);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(...$light);
scene.add(key);

const mat = (color) => new THREE.MeshStandardMaterial({color, roughness: 0.45});
const tension = mat(0x59d98a), compression = mat(0xff6b3d);
const metal = mat(0xf2f2f2), support = mat(0x5aa0ff), dark = mat(0x394152);

function bar(a, b, material) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, len, 12), material);
  mesh.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0);
  mesh.rotation.z = Math.atan2(dy, dx) - Math.PI / 2;
  scene.add(mesh);
}

for (const [name, from, to] of EDGES) bar(P[from], P[to], FORCES[name] > 0 ? tension : compression);
for (const node of [A, B, C]) {
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.18, 16, 12), metal);
  ball.position.set(node[0], node[1], 0);
  scene.add(ball);
}

// Pin under A, two rollers under B: the support layout the equilibrium uses.
const pin = new THREE.Mesh(new THREE.ConeGeometry(0.3, 0.6, 3), support);
pin.position.set(A[0], A[1] - 0.5, 0);
scene.add(pin);
for (const dx of [-0.2, 0.2]) {
  const roller = new THREE.Mesh(new THREE.SphereGeometry(0.14, 12, 10), support);
  roller.position.set(B[0] + dx, B[1] - 0.44, 0);
  scene.add(roller);
}

const ground = new THREE.Mesh(new THREE.BoxGeometry(12, 0.08, 0.3), dark);
ground.position.set(0, -0.9, 0);
scene.add(ground);

const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.4, 10), compression);
stem.position.set(C[0], C[1] - 0.7, 0);
scene.add(stem);
const tip = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.34, 12), compression);
tip.position.set(C[0], C[1] - 1.4 - 0.15, 0);
tip.rotation.z = Math.PI;
scene.add(tip);

const round4 = (v) => Math.round(v * 1e4) / 1e4;
const round6 = (v) => Math.round(v * 1e6) / 1e6;

function residual() {
  let sx = 0, sy = 0, m = 0;
  const add = (p, fx, fy) => { sx += fx; sy += fy; m += p[0] * fy - p[1] * fx; };
  add(A, REACTIONS[0], REACTIONS[1]);
  add(B, 0, REACTIONS[2]);
  add(C, 0, -LOAD);
  for (const [name, from, to] of EDGES) {
    const f = FORCES[name];
    const a = P[from], b = P[to];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
    add(a, f * ux, f * uy);
    add(b, -f * ux, -f * uy);
  }
  return [sx, sy, m];
}

let steps = 0;

window.__step = (dt) => { steps += 1; };

window.__probe = () => {
  renderer.render(scene, camera);
  const r = residual();
  return {
    nodes: {A: A, B: B, C: C},
    members: ['AB', 'AC', 'BC'],
    load: LOAD,
    reactions: REACTIONS.map(round4),
    forces: [FORCES.AB, FORCES.AC, FORCES.BC].map(round4),
    states: EDGES.map(([name]) => FORCES[name] > 0 ? 'tension' : 'compression'),
    residual: r.map(round6),
    steps,
  };
};

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN30 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>pd servo</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const DT = $dt;
const KP = $kp, KD = $kd, UMAX = $umax;
const TARGET_A = $target_a, TARGET_B = $target_b;
const X0 = $x0;

let x = X0, v = 0, target = TARGET_A, step = 0, saturated = 0, u = 0;
window.addEventListener('bench:target', (e) => { target = e.detail.target; });

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.OrthographicCamera(-10, 10, 6, -6, 0.1, 200);
camera.position.set(...$camera);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(...$light);
scene.add(key);

const mat = (color) => new THREE.MeshStandardMaterial({color, roughness: 0.45});
const rail = new THREE.Mesh(new THREE.BoxGeometry(16, 0.12, 0.3), mat(0x394152));
rail.position.set(0, -0.35, 0);
scene.add(rail);

const cart = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.6, 0.6), mat(0xffb347));
scene.add(cart);
const marker = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.8, 0.08), mat(0x5aa0ff));
marker.position.set(target, 0.55, 0);
scene.add(marker);

const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.4, 0xff5a1e, 0.22, 0.12);
scene.add(arrow);

function paint() {
  cart.position.set(x, 0, 0);
  marker.position.set(target, 0.55, 0);
  arrow.position.set(x, 0.45, 0);
  arrow.visible = Math.abs(u) > 1e-9;
  if (arrow.visible) {
    arrow.setDirection(new THREE.Vector3(u > 0 ? 1 : -1, 0, 0));
    arrow.setLength(0.3 + 1.4 * Math.abs(u) / UMAX, 0.22, 0.12);
  }
  renderer.render(scene, camera);
}

window.__step = (dt) => {
  if (step === 60) {
    window.dispatchEvent(new CustomEvent('bench:target', {detail: {target: TARGET_B}}));
  }
  const raw = KP * (target - x) - KD * v;
  if (Math.abs(raw) > UMAX) saturated += 1;
  u = Math.max(-UMAX, Math.min(UMAX, raw));
  v += u * dt;
  x += v * dt;
  step += 1;
};

const round6 = (value) => Math.round(value * 1e6) / 1e6;

window.__probe = () => {
  paint();
  return {
    t: round6(step * DT),
    steps: step,
    target: round6(target),
    x: round6(x),
    v: round6(v),
    u: round6(u),
    saturated_steps: saturated,
    rail_half: 8,
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


def _unit(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def solve_truss(
    nodes: dict[str, tuple[float, float]], load: float
) -> tuple[list[float], list[float]]:
    """Joint equilibrium as a 6x6 system: reactions first, member forces after.

    Unknowns are ordered ``Ax, Ay, By, F_AB, F_AC, F_BC``; tension is positive.
    A member pushes the joint it is attached to *away* from that joint, so the
    coefficient of ``F`` is the unit vector from that joint to the other end.
    Solved with plain Gaussian elimination and partial pivoting — no library,
    and nothing taken from the reference page.
    """
    size = 3 + len(MEMBERS)
    matrix = [[0.0] * (size + 1) for _ in range(size)]
    row = 0
    for joint in ("A", "B", "C"):
        fx, fy = matrix[row], matrix[row + 1]
        if joint == "A":
            fx[0] = 1.0
            fy[1] = 1.0
        if joint == "B":
            fy[2] = 1.0
        if joint == "C":
            fy[size] = -load
        for index, member in enumerate(MEMBERS):
            first, second = member[0], member[1]
            if first == joint:
                ux, uy = _unit(nodes[first], nodes[second])
            elif second == joint:
                ux, uy = _unit(nodes[second], nodes[first])
            else:
                continue
            fx[3 + index] += ux
            fy[3 + index] += uy
        row += 2

    for line in matrix:
        line[size] = -line[size]

    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(matrix[r][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        head = matrix[column][column]
        for j in range(column, size + 1):
            matrix[column][j] /= head
        for r in range(size):
            if r == column or not matrix[r][column]:
                continue
            factor = matrix[r][column]
            for j in range(column, size + 1):
                matrix[r][j] -= factor * matrix[column][j]

    solution = [matrix[i][size] for i in range(size)]
    return solution[:3], solution[3:]


def truss_residual(
    nodes: dict[str, tuple[float, float]],
    reactions: list[float],
    forces: list[float],
    load: float,
) -> list[float]:
    """Global ΣFx, ΣFy and ΣM about A of every force in the free body."""
    sx = sy = moment = 0.0

    def add(point: tuple[float, float], fx: float, fy: float) -> None:
        nonlocal sx, sy, moment
        sx += fx
        sy += fy
        moment += point[0] * fy - point[1] * fx

    add(nodes["A"], reactions[0], reactions[1])
    add(nodes["B"], 0.0, reactions[2])
    add(nodes["C"], 0.0, -load)
    for index, member in enumerate(MEMBERS):
        first, second = member[0], member[1]
        ux, uy = _unit(nodes[first], nodes[second])
        force = forces[index]
        add(nodes[first], force * ux, force * uy)
        add(nodes[second], -force * ux, -force * uy)
    return [sx, sy, moment]


def servo(
    target_a: float,
    target_b: float,
    kp: float,
    kd: float,
    *,
    steps: int = STEPS30,
    clamped: bool = True,
    retarget: bool = True,
) -> tuple[float, float, float, float, int]:
    """The stated controller, step for step; returns x, v, u, target, saturated."""
    x = X0
    v = 0.0
    target = target_a
    saturated = 0
    u = 0.0
    for step in range(steps):
        if retarget and step == RETARGET_AT:
            target = target_b
        raw = kp * (target - x) - kd * v
        if abs(raw) > U_MAX:
            saturated += 1
        u = max(-U_MAX, min(U_MAX, raw)) if clamped else raw
        v += u * DT
        x += v * DT
    return x, v, u, target, saturated


def round4(value: float) -> float:
    return math.floor(value * 1e4 + 0.5) / 1e4


def round6(value: float) -> float:
    return math.floor(value * 1e6 + 0.5) / 1e6


def _states(wanted: tuple[str, ...]) -> visual.Predicate:
    def check(payload: dict) -> tuple[bool, str]:
        actual = visual.path_of(payload, "states")
        if not isinstance(actual, list):
            return False, f"states: не список ({actual!r})"
        good = list(actual) == list(wanted)
        return good, f"states = {actual} (ожидалось {list(wanted)})"

    return check


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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _task_truss(rng: random.Random) -> core.TaskInstance:
    apex = float(rng.choice([0.6, 0.8, 1.0]))
    load = float(rng.choice([5.0, 6.0, 9.0, 12.0]))
    nodes = {"A": NODE_A, "B": NODE_B, "C": (apex, 3.0)}
    reactions, forces = solve_truss(nodes, load)
    residual = truss_residual(nodes, reactions, forces, load)
    states = tuple("tension" if force > 0 else "compression" for force in forces)

    assert max(abs(v) for v in residual) < 1e-9, "решение обязано быть равновесным"
    assert all(abs(force) > 1.0 for force in forces), "все стержни обязаны быть нагружены"
    assert states.count("tension") == 1 and states.count("compression") == 2
    assert reactions[1] > 0.0 and reactions[2] > 0.0, "опоры обязаны быть нагружены"
    assert abs(reactions[1] + reactions[2] - load) < 1e-9, "вертикальный баланс"

    prompt = PROMPT29.format(
        ax=NODE_A[0], ay=NODE_A[1], bx=NODE_B[0], by=NODE_B[1],
        cx=apex, cam=list(CAMERA29), light=list(LIGHT29), load=load, fields=FIELDS29,
    )
    golden = GOLDEN29.substitute(
        ax=NODE_A[0], ay=NODE_A[1], bx=NODE_B[0], by=NODE_B[1], cx=apex, cy=3.0,
        load=load, rax=reactions[0], ray=reactions[1], rby=reactions[2],
        fab=forces[0], fac=forces[1], fbc=forces[2],
        camera=list(CAMERA29), light=list(LIGHT29),
    )
    _assert_literals(
        prompt,
        f"A = {NODE_A[0]}, B = {NODE_B[0]}",
        f"C = ({apex}, 3.0)",
        f"P = {load}",
        "-10, 10, 7.5, -7.5",
        str(list(CAMERA29)),
        str(list(LIGHT29)),
        "0x59d98a",
        "0xff6b3d",
        "от x = -6.0 до x = 6.0",
    )
    _assert_in_frame(
        ORTHO29,
        [NODE_A, NODE_B, nodes["C"], (apex, 1.45), (-GROUND_HALF29, -0.9), (GROUND_HALF29, -0.9)],
        "K-29: ферма",
    )
    scene_checks = [
        visual.contains_all("members", MEMBERS),
        visual.near("nodes.A.0", NODE_A[0], 1e-6),
        visual.near("nodes.B.0", NODE_B[0], 1e-6),
        visual.near("nodes.C.0", apex, 1e-6),
        visual.near("load", load, 1e-6),
        visual.near("steps", 0, 0),
    ]
    simulation = [
        visual.near("reactions.0", round4(reactions[0]), 1e-3),
        visual.near("reactions.1", round4(reactions[1]), 1e-3),
        visual.near("reactions.2", round4(reactions[2]), 1e-3),
        visual.near("forces.0", round4(forces[0]), 1e-3),
        visual.near("forces.1", round4(forces[1]), 1e-3),
        visual.near("forces.2", round4(forces[2]), 1e-3),
        _states(states),
        visual.near("residual.0", 0.0, 1e-6),
        visual.near("residual.1", 0.0, 1e-6),
        visual.near("residual.2", 0.0, 1e-6),
        visual.near("steps", 1, 0),
    ]
    image = [
        visual.image_min("changed_fraction", 0.01),
        visual.image_min("distinct_colors", 4),
        visual.image_color("warm", 0.002, "сжатые стержни и стрелка нагрузки различимы"),
        visual.image_color("bright", 0.0006, "узлы светятся"),
    ]
    return visual.build_task(
        task_id="K-29",
        title="K-29 — равновесие треугольной фермы",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=simulation, image=image, steps=1, dt=DT),
        golden=golden,
        rng_meta={
            "apex": apex, "load": load, "nodes": {"A": list(NODE_A), "B": list(NODE_B), "C": [apex, 3.0]},
            "reactions": [round4(v) for v in reactions],
            "forces": [round4(v) for v in forces],
            "states": list(states), "residual": [round6(v) for v in residual],
        },
    )


def _task_servo(rng: random.Random) -> core.TaskInstance:
    target_a = float(rng.choice([4.2, 4.5, 4.8]))
    target_b = -target_a
    kp = float(rng.choice([1.4, 1.5, 1.6]))
    kd = float(rng.choice([0.9, 1.0, 1.1]))
    x, v, u, target, saturated = servo(target_a, target_b, kp, kd)

    without_clamp = servo(target_a, target_b, kp, kd, clamped=False)
    old_target = servo(target_a, target_b, kp, kd, retarget=False)

    assert 20 < saturated < STEPS30, "clamp обязан срабатывать, но не на каждом шаге"
    assert abs(without_clamp[0] - x) > 0.2, "без clamp геометрия обязана разойтись"
    assert abs(old_target[0] - x) > 0.3, "старая цель обязана дать другую позицию"
    assert old_target[3] == target_a and target == target_b
    assert -RAIL_HALF < x < RAIL_HALF and -RAIL_HALF < old_target[0] < RAIL_HALF

    prompt = PROMPT30.format(
        cam=list(CAMERA30), light=list(LIGHT30), x0=X0, target_a=target_a,
        target_b=target_b, dt=DT, kp=kp, kd=kd, umax=U_MAX, fields=FIELDS30,
    )
    golden = GOLDEN30.substitute(
        dt=DT, kp=kp, kd=kd, umax=U_MAX, target_a=target_a, target_b=target_b,
        x0=X0, camera=list(CAMERA30), light=list(LIGHT30),
    )
    _assert_literals(
        prompt,
        f"x = {X0}",
        f"target = {target_a}",
        f"target: {target_b}",
        f"dt = {DT}",
        f"u_raw = {kp} * (target - x) - {kd} * v",
        f"-{U_MAX}, {U_MAX}",
        "-10, 10, 6, -6",
        str(list(CAMERA30)),
        str(list(LIGHT30)),
        "0xffb347",
        "0x5aa0ff",
        "шаге с номером 60",
    )
    _assert_in_frame(
        ORTHO30,
        [(-RAIL_HALF, -0.35), (RAIL_HALF, -0.35), (target_a, 1.45), (target_b, 1.45), (-5.5, 1.0), (5.5, 1.0)],
        "K-30: рельс",
    )
    scene_checks = [
        visual.near("t", 0.0, 1e-9),
        visual.near("steps", 0, 0),
        visual.near("x", X0, 1e-6),
        visual.near("v", 0.0, 1e-6),
        visual.near("target", target_a, 1e-6),
        visual.near("saturated_steps", 0, 0),
        visual.near("rail_half", RAIL_HALF, 0),
    ]
    simulation = [
        visual.near("t", round6(STEPS30 * DT), 1e-9),
        visual.near("steps", STEPS30, 0),
        visual.near("target", round6(target), 1e-6),
        visual.near("x", round6(x), 1e-3),
        visual.near("v", round6(v), 1e-3),
        visual.near("u", round6(u), 1e-3),
        visual.near("saturated_steps", saturated, 0),
    ]
    image = [
        visual.image_min("changed_fraction", 0.005),
        visual.image_min("distinct_colors", 4),
        visual.image_color("warm", 0.0012, "каретка различима"),
        visual.image_color("cool", 0.0004, "метка цели различима"),
    ]
    return visual.build_task(
        task_id="K-30",
        title="K-30 — PD-серво с ограничением управления",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=simulation, image=image, steps=STEPS30, dt=DT),
        golden=golden,
        rng_meta={
            "x0": X0, "target_a": target_a, "target_b": target_b,
            "kp": kp, "kd": kd, "u_max": U_MAX, "steps": STEPS30,
            "x": round6(x), "v": round6(v), "u": round6(u), "target": round6(target),
            "saturated_steps": saturated,
            "without_clamp": [round6(value) for value in without_clamp[:4]] + [without_clamp[4]],
            "old_target": [round6(value) for value in old_target[:4]] + [old_target[4]],
        },
    )


BUILDERS = (_task_truss, _task_servo)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Every scenario of the slice, each drawing its fixture from the same stream."""
    return [builder(rng) for builder in BUILDERS]
