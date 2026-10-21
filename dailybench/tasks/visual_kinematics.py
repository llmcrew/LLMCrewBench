"""Visual scenarios K-21, K-22 — kinematics: an external gear pair and a
four-bar linkage.

Both scenarios are *analytic*: the state at time ``t`` is a closed-form function
of ``t``, not an incremental solver, and the expected numbers are recomputed
here in Python from the stated geometry — a check asks "does the mechanism obey
the kinematic relation it was given", never "does it look like the reference".

The camera is orthographic in both scenarios: the observed landmarks are
positions on a plane, and a perspective view would make their pixel columns
depend on depth. The reference page draws the very numbers the prompt spells
out, and paints each observable landmark in its own colour so the frame can be
read back independently of the probe.
"""

from __future__ import annotations

import math
import random
from string import Template

from .. import core, visual

IDS = ("K-21", "K-22")

#: K-21 is in the pilot wave; the closed mechanism (K-22) follows in stretch-v1.
IDS_BY_PROFILE = {"pilot": ("K-21",), "stretch-v1": IDS}

TAU = 2.0 * math.pi

#: Orthographic frame, world units. 800x600 keeps the 4:3 aspect.
ORTHO_W = 13.0
ORTHO_H = 9.75
FRAME_MARGIN = 0.9

#: The ten tracked steps land exactly on t = 0, 1, 2 s at dt = 1/60.
DT = 1 / 60
STEPS = 120
FRAME_STEPS = (0, 60, 120)

FIELDS21 = """{
  "count": <число колёс>,
  "r": [<радиус 1>, <радиус 2>],
  "centers": [[<x1>, <y1>], [<x2>, <y2>]],
  "omega": [<угловая скорость 1>, <угловая скорость 2>],
  "ratio": <omega2 / omega1>,
  "contact_angle": <угол линии центров, рад>,
  "angles": [<угол метки 1>, <угол метки 2>],
  "marks": [[<x метки 1>, <y метки 1>], [<x метки 2>, <y метки 2>]],
  "steps": <сделанных шагов>,
  "frames": [[<t>, <угол 1>, <угол 2>, <x метки 1>, <y метки 1>, <x метки 2>, <y метки 2>], ...]
}"""

FIELDS22 = """{
  "count": <число шарниров>,
  "lengths": [<AB>, <BC>, <CD>, <DA>],
  "joints": [[<Ax>, <Ay>], [<Bx>, <By>], [<Cx>, <Cy>], [<Dx>, <Dy>]],
  "input_angle": <угол кривошипа, рад>,
  "output_angle": <угол коромысла D->C, рад>,
  "closure": <остаток замыкания, длина>,
  "steps": <сделанных шагов>,
  "frames": [[<t>, <Bx>, <By>, <Cx>, <Cy>, <вход>, <выход>, <замыкание>], ...]
}"""

PROMPT21 = """Собери сцену пары внешних зубчатых колёс: ведущее колесо радиуса
{r1} вращается с постоянной угловой скоростью {w1} рад/с, ведомое колесо
радиуса {r2} сцеплено с ним без проскальзывания.

Сцена:
* оба колеса лежат в плоскости XY, смотрят на зрителя; ось ведущего колеса —
  в точке ({cx1}, 0), ось ведомого — в точке ({cx2}, 0), так что расстояние
  между осями ровно `{r1} + {r2}`, а точка контакта — в (0, 0);
* каждое колесо — это тёмный обод (кольцо), ступица и {teeth1} (у ведущего) и
  {teeth2} (у ведомого) прямоугольных зубьев по внешнему радиусу; всё одного
  нейтрального цвета 0x4a4a4a; ступицы — белые диски радиуса 0.22;
* на каждом колесе есть круглая метка радиуса 0.35, стоящая на расстоянии
  `0.72 * R` от оси под углом метки: у ведущего 0xd63f2f (красная), у ведомого
  0x2f5f9f (синяя);
* камера ортографическая, ровно по оси Z: `OrthographicCamera(-6.5, 6.5, 4.875,
  -4.875, 0.1, 100)` в точке (0, 0, 20) и смотрит в (0, 0, 0); свет —
  `AmbientLight(0xffffff, 0.7)` и `DirectionalLight(0xffffff, 1.1)` из
  (4, 6, 12), `renderer.setSize(800, 600, false)`.

Симуляция:
* время — только число шагов: `t = steps / 60`, шаг `dt = 1/60`, всего 120
  шагов; шаг делается только вызовом `window.__step(dt)`, состояние зависит
  только от `steps`;
* начальные углы меток: у ведущего {t1}, у ведомого {t2} рад;
* угол метки ведущего в момент `t`: `norm({t1} + {w1} * t)`; угол метки
  ведомого: `norm({t2} - ({r1} / {r2}) * {w1} * t)`, где
  `norm(a) = ((a % (2*Math.PI)) + 2*Math.PI) % (2*Math.PI)`;
* угловые скорости постоянны: `omega1 = {w1}`, `omega2 = -({r1} / {r2}) *
  {w1}`, то есть `omega2 / omega1 = -{r1} / {r2}`; знак `omega2`
  отрицательный относительно `omega1`;
* координаты метки: `(cx + 0.72 * R * cos(angle), 0.72 * R * sin(angle))`,
  где `cx` — ось своего колеса;
* колесо поворачивается целиком: обод, зубья и метка жёстко связаны, ступица
  стоит на месте; на кадре видны обе метки и обе оси;
* `window.__probe()` сначала обновляет повороты и метки, потом
  `renderer.render(scene, camera)`, и только затем возвращает поля;
  `window.__step(dt)` состояние только продвигает и ничего не рисует;
* кадры записываются в `frames` при `steps`, равном 0, 60 и 120 (t = 0, 1, 2),
  и больше нигде; до шагов в `frames` ровно один кадр;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками,
  углы нормализованы, `count` и `steps` — целые):

{fields}

Проверяется: число колёс, радиусы, оси, точка контакта, скорости и
передаточное отношение, углы и метки на кадрах t = 0, 1, 2 (сверка с этой же
формулой, допуск 1e-3), и кадр — обе метки и обе оси различимы, левая метка
стоит на левом колесе, правая — на правом, картинка воспроизводима при
повторной загрузке.
"""

PROMPT22 = """Собери сцену плоского четырёхзвенного механизма в плоскости XY:
кривошип AB равномерно вращается вокруг неподвижной оси A, шатун BC передаёт
движение коромыслу CD, которое качается вокруг неподвижной оси D.

Сцена:
* оси: A = (0, 0), D = ({d}, 0); длины звеньев: AB = {a}, BC = {b},
  CD = {c}, стойка DA = {d};
* звенья — прямоугольные штанги толщиной 0.18 нейтрального цвета 0xbfc0c0,
  шарниры — диски радиуса 0.2: A — 0xd63f2f, B — 0xd6b32f, C — 0x2f7fff,
  D — 0x9a9a3a; все шарниры видны на кадре;
* камера ортографическая, фронтально по оси Z: `OrthographicCamera(-6.5, 6.5,
  4.875, -4.875, 0.1, 100)` в точке ({cx}, {cy}, 20) и смотрит в ({cx}, {cy}, 0);
  свет — `AmbientLight(0xffffff, 0.75)` и `DirectionalLight(0xffffff, 1.1)`
  из (4, 6, 12), `renderer.setSize(800, 600, false)`.

Геометрия (решается на каждом шаге заново, без накопления):
* время — только число шагов: `t = steps / 60`, шаг `dt = 1/60`, всего 120
  шагов; шаг делается только вызовом `window.__step(dt)`;
* входной угол `theta = norm({t0} + {w} * t)`, `norm` — по модулю `2*Math.PI`
  в диапазон [0, 2*pi); `B = A + {a} * (cos(theta), sin(theta))`;
* `C` — пересечение окружности `(B, {b})` и окружности `(D, {c})`;
  из двух точек пересечения берётся ветвь «локоть вверх»: та, для которой
  `cross(D - B, C - B) > 0` (перпендикуляр к `D - B` повёрнут против часовой
  стрелки); вторая точка отбрасывается;
* пересечение считается так: `u = (D - B) / |D - B|`, `q = B + u * (({b}^2 -
  {c}^2 + |D - B|^2) / (2 * |D - B|))`, `h = sqrt({b}^2 - |q - B|^2)`, и
  `C = q + h * (-u.y, u.x)`;
* `input_angle = theta`, `output_angle = norm(atan2(C.y - D.y, C.x - D.x))`;
  `closure = ||B - C| - {b}| + ||C - D| - {c}| + ||A - B| - {a}|`;
* звенья и шарниры перерисовываются из координат; `window.__probe()` сначала
  перерисовывает, потом `renderer.render(scene, camera)`, и только затем
  возвращает поля; `window.__step(dt)` состояние только продвигает;
* кадры записываются в `frames` при `steps`, равном 0, 60 и 120 (t = 0, 1, 2),
  и больше нигде; до шагов в `frames` ровно один кадр;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками,
  `count` и `steps` — целые):

{fields}

Проверяется: число шарниров, длины звеньев, координаты всех четырёх шарниров,
входной и выходной углы и замыкание на кадрах t = 0, 1, 2 (сверка с этой же
геометрией, допуск 1e-3; замыкание <= 1e-3), выбранная ветвь (зеркальная
ветвь не проходит) и кадр — все четыре шарнира различимы по цвету, звенья
видны, картинка воспроизводима при повторной загрузке.
"""

GOLDEN21 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>external gears</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const R1 = $r1, R2 = $r2, W1 = $w1, T1 = $t1, T2 = $t2;
const C1 = $cx1, C2 = $cx2;

const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.OrthographicCamera(-6.5, 6.5, 4.875, -4.875, 0.1, 100);
camera.position.set(0, 0, 20);
camera.lookAt(0, 0, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(4, 6, 12);
scene.add(key);

const body = new THREE.MeshStandardMaterial({color: 0x4a4a4a});
const white = new THREE.MeshBasicMaterial({color: 0xffffff});
const red = new THREE.MeshBasicMaterial({color: 0xd63f2f});
const blue = new THREE.MeshBasicMaterial({color: 0x2f5f9f});

function wheel(cx, r, teeth, markMaterial) {
  const group = new THREE.Group();
  group.position.set(cx, 0, 0);
  const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.38, r, 64), body);
  group.add(ring);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.04, 24), white);
  hub.rotation.x = Math.PI / 2;
  hub.position.z = 0.02;
  group.add(hub);
  for (let s = 0; s < 4; s++) {
    const spoke = new THREE.Mesh(new THREE.BoxGeometry(r - 0.38, 0.18, 0.05), body);
    spoke.rotation.z = s * Math.PI / 4;
    spoke.position.z = 0.01;
    group.add(spoke);
  }
  for (let k = 0; k < teeth; k++) {
    const a = k * 2 * Math.PI / teeth;
    const tooth = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.3, 0.06), body);
    tooth.position.set(Math.cos(a) * (r + 0.04), Math.sin(a) * (r + 0.04), 0.01);
    tooth.rotation.z = a;
    group.add(tooth);
  }
  // Метка стоит на оси X группы: абсолютный угол метки равен повороту группы.
  const mark = new THREE.Mesh(new THREE.CircleGeometry(0.35, 32), markMaterial);
  mark.position.set(0.72 * r, 0, 0.03);
  group.add(mark);
  scene.add(group);
  return group;
}

const g1 = wheel(C1, R1, $teeth1, red);
const g2 = wheel(C2, R2, $teeth2, blue);

const norm = (a) => ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
const round = (v) => Math.round(v * 1e4) / 1e4;
const markAt = (cx, r, a) => [round(cx + 0.72 * r * Math.cos(a)), round(0.72 * r * Math.sin(a))];

const ratio = -R1 / R2;
let steps = 0;
const frames = [];

const angle1 = () => norm(T1 + W1 * steps / 60);
const angle2 = () => norm(T2 + ratio * W1 * steps / 60);

function paint() {
  g1.rotation.z = angle1();
  g2.rotation.z = angle2();
}

function record() {
  const a1 = angle1(), a2 = angle2();
  const m1 = markAt(C1, R1, a1), m2 = markAt(C2, R2, a2);
  frames.push([round(steps / 60), round(a1), round(a2), m1[0], m1[1], m2[0], m2[1]]);
}

window.__step = (dt) => {
  steps += 1;
  if (steps === 60 || steps === 120) record();
};

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  const a1 = angle1(), a2 = angle2();
  const m1 = markAt(C1, R1, a1), m2 = markAt(C2, R2, a2);
  return {
    count: 2, r: [R1, R2], centers: [[C1, 0], [C2, 0]],
    omega: [W1, round(ratio * W1)], ratio: round(ratio), contact_angle: 0,
    angles: [round(a1), round(a2)], marks: [m1, m2], steps, frames,
  };
};

record();
paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN22 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>four-bar linkage</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const A = [0, 0], D = [$d, 0];
const LA = $a, LB = $b, LC = $c, LD = $d, W = $w, T0 = $t0;

const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.OrthographicCamera(-6.5, 6.5, 4.875, -4.875, 0.1, 100);
camera.position.set($cx, $cy, 20);
camera.lookAt($cx, $cy, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(4, 6, 12);
scene.add(key);

const barMat = new THREE.MeshBasicMaterial({color: 0xbfc0c0});
const colors = [0xd63f2f, 0xd6b32f, 0x2f7fff, 0x9a9a3a];
const jointMat = colors.map((c) => new THREE.MeshBasicMaterial({color: c}));

function bar() {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 0.18, 0.04), barMat);
  scene.add(mesh);
  return mesh;
}
const bars = [bar(), bar(), bar(), bar()];
const nodes = jointMat.map((m) => {
  const node = new THREE.Mesh(new THREE.CircleGeometry(0.2, 24), m);
  node.position.z = 0.05;
  scene.add(node);
  return node;
});

function place(link, p, q) {
  const dx = q[0] - p[0], dy = q[1] - p[1];
  link.scale.x = Math.hypot(dx, dy);
  link.rotation.z = Math.atan2(dy, dx);
  link.position.set((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, 0);
}

const norm = (a) => ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
const round = (v) => Math.round(v * 1e4) / 1e4;
let steps = 0;
const frames = [];

function solve() {
  const theta = norm(T0 + W * steps / 60);
  const B = [LA * Math.cos(theta), LA * Math.sin(theta)];
  const ux = (D[0] - B[0]) / Math.hypot(D[0] - B[0], D[1] - B[1]);
  const uy = (D[1] - B[1]) / Math.hypot(D[0] - B[0], D[1] - B[1]);
  const bd = Math.hypot(D[0] - B[0], D[1] - B[1]);
  const along = (LB * LB - LC * LC + bd * bd) / (2 * bd);
  const qx = B[0] + ux * along, qy = B[1] + uy * along;
  const h = Math.sqrt(LB * LB - along * along);
  const C = [qx - h * uy, qy + h * ux];
  const closure = Math.abs(Math.hypot(B[0] - C[0], B[1] - C[1]) - LB)
    + Math.abs(Math.hypot(C[0] - D[0], C[1] - D[1]) - LC)
    + Math.abs(Math.hypot(A[0] - B[0], A[1] - B[1]) - LA);
  return { B, C, theta, out: norm(Math.atan2(C[1] - D[1], C[0] - D[0])), closure };
}

function paint(state) {
  const p = [A, state.B, state.C, D];
  place(bars[0], p[0], p[1]);
  place(bars[1], p[1], p[2]);
  place(bars[2], p[2], p[3]);
  place(bars[3], p[3], p[0]);
  for (let k = 0; k < 4; k++) nodes[k].position.set(p[k][0], p[k][1], 0.05);
}

function record() {
  const s = solve();
  frames.push([round(steps / 60), round(s.B[0]), round(s.B[1]), round(s.C[0]),
    round(s.C[1]), round(s.theta), round(s.out), round(s.closure)]);
}

window.__step = (dt) => {
  steps += 1;
  if (steps === 60 || steps === 120) record();
};

window.__probe = () => {
  const s = solve();
  paint(s);
  renderer.render(scene, camera);
  return {
    count: 4, lengths: [LA, LB, LC, LD], joints: [A, [round(s.B[0]), round(s.B[1])],
      [round(s.C[0]), round(s.C[1])], D],
    input_angle: round(s.theta), output_angle: round(s.out), closure: round(s.closure),
    steps, frames,
  };
};

record();
paint(solve());
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")


# --------------------------------------------------------------------------- #
# Independent Python oracles
# --------------------------------------------------------------------------- #


def norm_angle(value: float) -> float:
    return value % TAU


def gear_state(
    r1: float,
    r2: float,
    omega1: float,
    theta1_0: float,
    theta2_0: float,
    steps: int,
    same_direction: bool = False,
    unit_ratio: bool = False,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Frames ``(t, a1, a2, m1x, m1y, m2x, m2y)`` from the stated relation.

    ``same_direction`` flips the sign of ``omega2``; ``unit_ratio`` keeps the
    sign but makes the magnitude 1. Both are the named negative example.
    """
    ratio = -1.0 if unit_ratio else -r1 / r2
    if same_direction:
        ratio = -ratio
    c1, c2 = -(r1 + r2) / 2, (r1 + r2) / 2
    frames = []
    for step in FRAME_STEPS:
        if step > steps:
            continue
        t = step / 60
        a1 = norm_angle(theta1_0 + omega1 * t)
        a2 = norm_angle(theta2_0 + ratio * omega1 * t)
        m1 = (c1 + 0.72 * r1 * math.cos(a1), 0.72 * r1 * math.sin(a1))
        m2 = (c2 + 0.72 * r2 * math.cos(a2), 0.72 * r2 * math.sin(a2))
        frames.append((t, a1, a2, m1[0], m1[1], m2[0], m2[1]))
    return frames


def gear_omega2(r1: float, r2: float, omega1: float) -> float:
    return -r1 / r2 * omega1


def fourbar(
    a: float, b: float, c: float, d: float, theta: float, branch: int = 1
) -> tuple[list[list[float]], float, float, float]:
    """Circle-circle intersection in the direction ``B -> D``, chosen branch.

    ``branch = +1`` is the "elbow up" side, ``-1`` the mirror side.
    Returns ``(joints, input_angle, output_angle, closure)``.
    """
    angle = norm_angle(theta)
    bx, by = a * math.cos(angle), a * math.sin(angle)
    dx, dy = d, 0.0
    span = math.hypot(dx - bx, dy - by)
    assert abs(b - c) <= span <= b + c, "звенья обязаны замыкаться"
    ux, uy = (dx - bx) / span, (dy - by) / span
    along = (b * b - c * c + span * span) / (2 * span)
    qx, qy = bx + ux * along, by + uy * along
    h = math.sqrt(max(0.0, b * b - along * along))
    cx, cy = qx - branch * h * uy, qy + branch * h * ux
    joints = [[0.0, 0.0], [bx, by], [cx, cy], [d, 0.0]]
    closure = (
        abs(math.hypot(bx - cx, by - cy) - b)
        + abs(math.hypot(cx - dx, cy - dy) - c)
        + abs(math.hypot(bx, by) - a)
    )
    out = norm_angle(math.atan2(cy - dy, cx - dx))
    return joints, angle, out, closure


def fourbar_frames(
    a: float, b: float, c: float, d: float, theta0: float, omega: float,
    branch: int = 1,
) -> list[tuple[float, float, float, float, float, float, float, float]]:
    frames = []
    for step in FRAME_STEPS:
        t = step / 60
        joints, inp, out, closure = fourbar(a, b, c, d, theta0 + omega * t, branch)
        frames.append((t, joints[1][0], joints[1][1], joints[2][0], joints[2][1], inp, out, closure))
    return frames


def round4(value: float) -> float:
    return math.floor(value * 1e4 + 0.5) / 1e4


# --------------------------------------------------------------------------- #
# Frame geometry
# --------------------------------------------------------------------------- #


def _assert_in_ortho(point: tuple[float, float], cx: float, cy: float, label: str) -> None:
    assert abs(point[0] - cx) <= FRAME_MARGIN * ORTHO_W / 2, f"{label}: x вне кадра"
    assert abs(point[1] - cy) <= FRAME_MARGIN * ORTHO_H / 2, f"{label}: y вне кадра"


def _assert_literals(prompt: str, *values: str) -> None:
    missing = [v for v in values if v not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _task_gears(rng: random.Random) -> core.TaskInstance:
    r1 = float(rng.choice([1.6, 1.8, 2.0]))
    r2 = float(rng.choice([2.6, 2.8, 3.0]))
    omega1 = float(rng.choice([1.4, 1.8, 2.2]))
    theta1_0 = float(rng.choice([0.4, 1.0, 1.7]))
    theta2_0 = float(rng.choice([2.2, 3.0, 3.8]))
    teeth1, teeth2 = int(round(r1 * 10)), int(round(r2 * 10))
    cx1, cx2 = -round4((r1 + r2) / 2), round4((r1 + r2) / 2)
    omega2 = gear_omega2(r1, r2, omega1)
    frames = gear_state(r1, r2, omega1, theta1_0, theta2_0, STEPS)
    initial = frames[0]
    final = frames[-1]

    assert len(frames) == 3, "нужны кадры t = 0, 1, 2"
    assert abs(initial[1] - theta1_0) < 1e-9 and abs(initial[2] - theta2_0) < 1e-9
    assert abs(final[0] - 2.0) < 1e-9
    assert len({round4(f[1]) for f in frames}) == 3, "метка 1 обязана двигаться"
    assert len({round4(f[2]) for f in frames}) == 3, "метка 2 обязана двигаться"
    assert omega2 < 0.0, "ведомое колесо обязано вращаться в другую сторону"

    prompt = PROMPT21.format(
        r1=r1, r2=r2, w1=omega1, t1=theta1_0, t2=theta2_0, cx1=cx1, cx2=cx2,
        teeth1=teeth1, teeth2=teeth2, fields=FIELDS21,
    )
    golden = GOLDEN21.substitute(
        r1=r1, r2=r2, w1=omega1, t1=theta1_0, t2=theta2_0, cx1=cx1, cx2=cx2,
        teeth1=teeth1, teeth2=teeth2,
    )
    _assert_literals(
        prompt, f"радиуса\n{r1}", f"радиуса {r2}", f"{omega1} рад/с",
        f"({cx1}, 0)", f"({cx2}, 0)", f"{r1} + {r2}", f"у ведущего {theta1_0}",
        f"у ведомого {theta2_0}",
    )
    for point in ((cx1 - r1, 0.0), (cx1 + r1, 0.0), (cx2 - r2, 0.0), (cx2 + r2, 0.0)):
        _assert_in_ortho(point, 0.0, 0.0, "K-21: колесо")
    assert abs(cx2 - cx1 - (r1 + r2)) < 1e-9, "оси обязаны отстоять на сумму радиусов"

    scene_checks = [
        visual.near("count", 2, 0),
        visual.near("r.0", r1, 1e-9),
        visual.near("r.1", r2, 1e-9),
        visual.near("centers.0.0", cx1, 1e-9),
        visual.near("centers.1.0", cx2, 1e-9),
        visual.near("centers.0.1", 0.0, 1e-9),
        visual.near("centers.1.1", 0.0, 1e-9),
        visual.near("omega.0", omega1, 1e-9),
        visual.near("omega.1", round4(omega2), 1e-6),
        visual.near("ratio", round4(-r1 / r2), 1e-6),
        visual.near("contact_angle", 0.0, 1e-9),
        visual.near("angles.0", round4(initial[1]), 1e-3),
        visual.near("angles.1", round4(initial[2]), 1e-3),
        visual.near("marks.0.0", round4(initial[3]), 1e-3),
        visual.near("marks.1.0", round4(initial[5]), 1e-3),
        visual.near("steps", 0, 0),
        visual.size_between("frames", 1, 1),
    ]
    sim_checks = [
        visual.near("steps", STEPS, 0),
        visual.size_between("frames", 3, 3),
        visual.near("frames.1.0", 1.0, 0),
        visual.near("frames.2.0", 2.0, 0),
        visual.near("frames.1.1", round4(frames[1][1]), 1e-3),
        visual.near("frames.1.2", round4(frames[1][2]), 1e-3),
        visual.near("frames.2.1", round4(final[1]), 1e-3),
        visual.near("frames.2.2", round4(final[2]), 1e-3),
        visual.near("frames.1.3", round4(frames[1][3]), 1e-3),
        visual.near("frames.1.4", round4(frames[1][4]), 1e-3),
        visual.near("frames.2.5", round4(final[5]), 1e-3),
        visual.near("frames.2.6", round4(final[6]), 1e-3),
        visual.near("angles.0", round4(final[1]), 1e-3),
        visual.near("angles.1", round4(final[2]), 1e-3),
        visual.near("marks.0.0", round4(final[3]), 1e-3),
        visual.near("marks.1.0", round4(final[5]), 1e-3),
        visual.near("omega.1", round4(omega2), 1e-6),
    ]
    image_checks = [
        visual.image_color("warm", 0.0012, "красная метка ведущего колеса"),
        visual.image_color("cool", 0.0012, "синяя метка ведомого колеса"),
        visual.image_centroid("warm", 0, 0.1, 0.46, "красная метка на левом колесе"),
        visual.image_centroid("cool", 0, 0.49, 0.95, "синяя метка на правом колесе"),
        visual.image_centroid("bright", 0, 0.45, 0.55, "обе ступицы видны"),
        visual.image_region(4, lambda color: sum(color) / 3 > 22, "центр кадра занят колёсами"),
        visual.image_min("changed_fraction", 0.08),
        visual.image_min("distinct_colors", 6),
    ]
    return visual.build_task(
        task_id="K-21",
        title="K-21 — пара внешних зубчатых колёс",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=STEPS, dt=DT),
        golden=golden,
        rng_meta={
            "r1": r1, "r2": r2, "omega1": omega1, "omega2": round4(omega2),
            "ratio": round4(-r1 / r2), "theta1_0": theta1_0, "theta2_0": theta2_0,
            "steps": STEPS,
        },
    )


def _task_fourbar(rng: random.Random) -> core.TaskInstance:
    a = float(rng.choice([0.9, 1.0, 1.1]))
    b = float(rng.choice([3.2, 3.5, 3.8]))
    c = float(rng.choice([2.3, 2.5, 2.7]))
    d = float(rng.choice([2.8, 3.0, 3.2]))
    theta0 = float(rng.choice([0.5, 1.1, 1.8]))
    omega = float(rng.choice([1.0, 1.2, 1.4]))
    cx = round4(d / 2)
    cy = 0.5

    assert a + b <= c + d, "механизм обязан быть кривошип-коромыслом (Грасгоф)"
    frames = fourbar_frames(a, b, c, d, theta0, omega)
    joints, inp, out, closure = fourbar(a, b, c, d, theta0)
    final_joints, _, _, final_closure = fourbar(a, b, c, d, theta0 + omega * 2)

    assert closure <= 1e-9 and final_closure <= 1e-9, "оракул обязан замыкаться"
    assert abs(inp - theta0) < 1e-9
    assert len({round4(f[5]) for f in frames}) == 3, "кривошип обязан двигаться"
    assert len({round4(f[3]) for f in frames}) == 3, "коромысло обязано двигаться"
    assert final_joints[2] != joints[2], "зеркальная ветвь обязана отличаться"

    prompt = PROMPT22.format(
        a=a, b=b, c=c, d=d, t0=theta0, w=omega, cx=cx, cy=cy, fields=FIELDS22,
    )
    golden = GOLDEN22.substitute(
        a=a, b=b, c=c, d=d, t0=theta0, w=omega, cx=cx, cy=cy,
    )
    _assert_literals(
        prompt, f"AB = {a}", f"BC = {b}", f"CD = {c}", f"стойка DA = {d}",
        f"({d}, 0)", f"{theta0} + {omega} * t",
    )
    for joint in joints:
        _assert_in_ortho((joint[0], joint[1]), cx, cy, "K-22: шарнир")
    final_inp = norm_angle(theta0 + omega * 2)
    final_out = fourbar(a, b, c, d, final_inp)[2]
    for frame in frames:
        _assert_in_ortho((frame[1], frame[2]), cx, cy, "K-22: шатун")
        _assert_in_ortho((frame[3], frame[4]), cx, cy, "K-22: коромысло")

    scene_checks = [
        visual.near("count", 4, 0),
        visual.near("lengths.0", a, 1e-3),
        visual.near("lengths.1", b, 1e-3),
        visual.near("lengths.2", c, 1e-3),
        visual.near("lengths.3", d, 1e-3),
        visual.near("closure", 0.0, 1e-3),
        visual.near("joints.0.0", 0.0, 1e-3),
        visual.near("joints.3.0", d, 1e-3),
        visual.near("joints.1.0", round4(joints[1][0]), 1e-3),
        visual.near("joints.1.1", round4(joints[1][1]), 1e-3),
        visual.near("joints.2.0", round4(joints[2][0]), 1e-3),
        visual.near("joints.2.1", round4(joints[2][1]), 1e-3),
        visual.near("input_angle", round4(inp), 1e-3),
        visual.near("output_angle", round4(out), 1e-3),
        visual.near("steps", 0, 0),
        visual.size_between("frames", 1, 1),
    ]
    sim_checks = [
        visual.near("steps", STEPS, 0),
        visual.size_between("frames", 3, 3),
        visual.near("frames.1.0", 1.0, 0),
        visual.near("frames.2.0", 2.0, 0),
        visual.near("frames.1.5", round4(frames[1][5]), 1e-3),
        visual.near("frames.1.6", round4(frames[1][6]), 1e-3),
        visual.near("frames.2.6", round4(frames[2][6]), 1e-3),
        visual.near("frames.1.7", 0.0, 1e-3),
        visual.near("frames.2.7", 0.0, 1e-3),
        visual.near("joints.1.0", round4(final_joints[1][0]), 1e-3),
        visual.near("joints.1.1", round4(final_joints[1][1]), 1e-3),
        visual.near("joints.2.0", round4(final_joints[2][0]), 1e-3),
        visual.near("joints.2.1", round4(final_joints[2][1]), 1e-3),
        visual.near("input_angle", round4(final_inp), 1e-3),
        visual.near("output_angle", round4(final_out), 1e-3),
        visual.near("closure", 0.0, 1e-3),
        visual.near("lengths.1", b, 1e-3),
    ]
    image_checks = [
        visual.image_color("warm", 0.0012, "тёплые шарниры A и B"),
        visual.image_color("cool", 0.0005, "синий шарнир C"),
        visual.image_centroid("cool", 0, 0.42, 1.0, "шарнир C справа от A"),
        visual.image_centroid("cool", 1, 0.05, 0.98, "шарнир C виден в кадре"),
        visual.image_region(4, lambda color: sum(color) / 3 > 20, "звенья видны в центре"),
        visual.image_min("changed_fraction", 0.012),
        visual.image_min("distinct_colors", 5),
        visual.image_min("mean", 10.0),
    ]
    return visual.build_task(
        task_id="K-22",
        title="K-22 — четырёхзвенный механизм: выбранная ветвь",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=STEPS, dt=DT),
        golden=golden,
        rng_meta={
            "a": a, "b": b, "c": c, "d": d, "theta0": theta0, "omega": omega,
            "steps": STEPS, "joints": [[round4(v) for v in joint] for joint in joints],
        },
    )


BUILDERS = (_task_gears, _task_fourbar)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Both scenarios, drawing their fixtures from the same stream."""
    return [builder(rng) for builder in BUILDERS]
