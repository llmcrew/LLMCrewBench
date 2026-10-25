"""Visual scenarios K-35..K-36 — квадратичное сопротивление и цепочка Verlet.

Two scenarios in the same shape as the rest of the K block: the prompt states
the scene, the exact discrete scheme and the exact probe fields, and the
expected numbers are recomputed here in Python from that very scheme. The
reference page is a real answer — ``selfcheck`` renders it in the browser and
requires all four checks (html, scene, simulation, image) to pass.

K-35 is a shell fired horizontally into a quadratic drag field. The scheme is
semi-implicit Euler with the acceleration ``-c * |v| * v - g`` and the probe
publishes the analytic terminal velocity ``sqrt(g / c)``: the Python replay
shows both the state after the published number of steps and the limit the
speed is heading for. A page that integrates linear drag, or no drag at all,
lands in a different place after the same steps.

K-36 is a six-link chain under gravity, integrated with Verlet and a fixed
number of constraint-relaxation iterations per step. Probe publishes node
coordinates, current link lengths and the total length residual; a page that
integrates without the relaxation keeps a visibly stretched chain.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

from .. import core, visual

IDS = ("K-35", "K-36")

#: Both scenarios join the pilot slice as hosts and the whole stretch suite.
IDS_BY_PROFILE = {"pilot": IDS, "stretch-v1": IDS}
TITLE = "K-35..K-36 — квадратичное сопротивление и цепочка Verlet"

#: The frame every scenario is rendered in.
DT = 1 / 60
GRAVITY = 9.81
FOV = 50.0
ASPECT = 800 / 600

# --------------------------------------------------------------------------- #
# Shared page skeleton
# --------------------------------------------------------------------------- #

_PAGE_HEAD = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;background:#101318;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 200);
camera.position.set(...{camera});
camera.lookAt(...{look});
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const key = new THREE.DirectionalLight(0xffffff, 1.9);
key.position.set(...{light});
scene.add(key);

const round = (value) => Math.round(value * 1e4) / 1e4;
"""

_PAGE_TAIL = """</script>
</body></html>
"""

#: Aligns a unit-length cylinder with the segment ``a -> b``: the shared way a
#: link is drawn, so its length is the distance and its angle the direction.
_LINK = """
function link(mesh, ax, ay, bx, by) {{
  const dx = bx - ax, dy = by - ay;
  mesh.position.set((ax + bx) / 2, (ay + by) / 2, 0);
  mesh.rotation.z = Math.atan2(-dx, dy);
  mesh.scale.y = Math.hypot(dx, dy);
}}
"""


def _page(body: str, **values: object) -> str:
    return (_PAGE_HEAD + body + _PAGE_TAIL).format(**values)


# --------------------------------------------------------------------------- #
# Frame proof
# --------------------------------------------------------------------------- #


def _in_frame(
    point: tuple[float, float, float],
    radius: float,
    camera_position: tuple[float, float, float],
    look: tuple[float, float, float],
    margin: float = 0.72,
) -> bool:
    """Is a sphere of ``radius`` around ``point`` inside the central ``margin``?

    The frame is what the ``image`` check reads, so a subject that leaves the
    viewport is a broken fixture rather than a wrong answer. This is the
    geometric proof that the drawn numbers keep their subject on screen: a
    camera basis from ``position``/``lookAt``, ``tan(fov/2)`` for the half
    extent, and a margin for the slack a real render leaves at the border.
    """
    fwd = [look[i] - camera_position[i] for i in range(3)]
    norm = math.sqrt(sum(value * value for value in fwd))
    fwd = [value / norm for value in fwd]
    # right = normalize(cross(fwd, world_up)) with world_up = (0, 1, 0); none of
    # the cameras look straight down, so this basis is never degenerate.
    right = [fwd[2], 0.0, -fwd[0]]
    rn = math.sqrt(right[0] ** 2 + right[2] ** 2)
    right = [right[0] / rn, 0.0, right[2] / rn]
    up = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ]
    delta = [point[i] - camera_position[i] for i in range(3)]
    depth = sum(delta[i] * fwd[i] for i in range(3))
    if depth <= 0.1:
        return False
    half_h = math.tan(math.radians(FOV / 2)) * depth
    half_w = half_h * ASPECT
    x = sum(delta[i] * right[i] for i in range(3))
    y = sum(delta[i] * up[i] for i in range(3))
    return abs(x) + radius <= half_w * margin and abs(y) + radius <= half_h * margin


def _assert_in_frame(
    camera_position: tuple[float, float, float],
    look: tuple[float, float, float],
    points: Iterable[tuple[float, float, float]],
    radius: float,
    what: str,
    margin: float = 0.72,
) -> None:
    """Every drawn point of the subject must be on screen by construction."""
    outside = [
        point
        for point in points
        if not _in_frame(point, radius, camera_position, look, margin)
    ]
    assert not outside, f"{what} выходит за кадр: {outside[:2]} (радиус {radius})"


def _assert_numbers_in_prompt(prompt: str, values: Iterable[float]) -> None:
    """Every fixture number must be readable in the prompt."""
    missing = [value for value in values if str(float(value)) not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


# --------------------------------------------------------------------------- #
# K-35 — shell in a quadratic drag field
# --------------------------------------------------------------------------- #

K35_CAMERA = (0.0, 6.5, 20.0)
K35_LOOK = (0.0, 6.5, 0.0)
K35_LIGHT = (10.0, 22.0, 18.0)
K35_RADIUS = 0.7
K35_X0 = -7.0
K35_Y0 = 11.0
K35_VY0 = 0.0

#: Coefficient/step-count pairs. They are drawn together because the fall depth,
#: not the step count alone, has to stay inside the frame: a stronger drag
#: spends the same time budget on less travel.
K35_RUNS = (
    (0.12, 75),
    (0.15, 75),
    (0.15, 90),
    (0.19, 75),
    (0.19, 90),
    (0.24, 75),
    (0.24, 90),
)

K35_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "ledge", "shell", "..."],
  "camera": [0, 6.5, 20],
  "x": <координата снаряда>,
  "y": <высота снаряда>,
  "vx": <горизонтальная скорость>,
  "vy": <вертикальная скорость>,
  "speed": <модуль скорости>,
  "terminal": <предельная скорость sqrt(g/c)>,
  "c": <коэффициент сопротивления>
}"""

K35_PROMPT = """Собери сцену: снаряд выпущен горизонтально с обрыва и падает в поле
квадратичного сопротивления воздуха.

Сцена (плоскость XY, z = 0):
* земля — плоскость 80x80 в плоскости y = -3.5, материал тёмно-синий
  (0x2b3a4a), имя `ground`;
* обрыв — параллелепипед 2.6x0.8x2.0 в точке (-8, 11.4, 0), материал
  серо-синий (0x8a93a6), имя `ledge`;
* снаряд — сфера радиуса {r} в точке `(x, y, 0)`, материал оранжевый
  (0xff8a3d), имя `shell`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.5)` и
  `DirectionalLight(0xffffff, 1.9)` из точки {light}.

Симуляция:
* шаг делается только вызовом `window.__step(dt)`;
* состояние — `x`, `y`, `vx`, `vy`; старт ровно `x = {x0}, y = {y0}`,
  `vx = {vx0}, vy = {vy0}`;
* сопротивление квадратично по модулю скорости, g = {g} вниз; схема шага —
  полунеявный Эйлер ровно в таком порядке:

      const speed = Math.hypot(vx, vy);
      vx += (-C * speed * vx) * dt;
      vy += (-G - C * speed * vy) * dt;
      x += vx * dt;
      y += vy * dt;

  где C = {c};
* предельная скорость падения — `sqrt(g / C)` (число проверяется в поле
  `terminal`), снаряд разгоняется к ней снизу и не превышает её;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `ground`, `ledge` и
`shell`, лишние имена не мешают.

Проверяется: состав сцены, `terminal` = sqrt(g/C), координаты, обе компоненты
скорости и модуль скорости после {steps} шагов (Python повторяет ту же схему
шаг в шаг, допуск 1e-3) и кадр — снаряд виден в кадре, картинка
воспроизводима при повторной загрузке.
"""

K35_BODY = """
const G = {g}, C = {c};
let x = {x0}, y = {y0}, vx = {vx0}, vy = {vy0};
let elapsed = 0, steps = 0;

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(80, 80),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -3.5;
ground.name = 'ground';
scene.add(ground);

const ledge = new THREE.Mesh(
  new THREE.BoxGeometry(2.6, 0.8, 2.0),
  new THREE.MeshStandardMaterial({{color: 0x8a93a6}})
);
ledge.position.set(-8, 11.4, 0);
ledge.name = 'ledge';
scene.add(ledge);

const shell = new THREE.Mesh(
  new THREE.SphereGeometry({r}, 24, 16),
  new THREE.MeshStandardMaterial({{color: 0xff8a3d}})
);
shell.name = 'shell';
scene.add(shell);

function draw() {{
  shell.position.set(x, y, 0);
  renderer.render(scene, camera);
}}

window.__step = (dt) => {{
  const speed = Math.hypot(vx, vy);
  vx += (-C * speed * vx) * dt;
  vy += (-G - C * speed * vy) * dt;
  x += vx * dt;
  y += vy * dt;
  elapsed += dt;
  steps += 1;
  draw();
  return window.__probe();
}};

window.__probe = () => ({{
  steps,
  t: round(elapsed),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  camera: [camera.position.x, camera.position.y, camera.position.z],
  x: round(x),
  y: round(y),
  vx: round(vx),
  vy: round(vy),
  speed: round(Math.hypot(vx, vy)),
  terminal: round(Math.sqrt(G / C)),
  c: C,
}});

draw();
window.__ready = true;
"""


def _drag_state(
    c: float, vx: float, vy: float, x: float, y: float, steps: int, mode: str = "quadratic"
) -> tuple[float, float, float, float]:
    """The stated semi-implicit scheme, replayed in Python.

    ``mode`` selects the acceleration: the stated quadratic drag, the linear
    drag a wrong answer would write, or no drag at all.
    """
    for _ in range(steps):
        speed = math.hypot(vx, vy)
        if mode == "quadratic":
            ax, ay = -c * speed * vx, -GRAVITY - c * speed * vy
        elif mode == "linear":
            ax, ay = -c * vx, -GRAVITY - c * vy
        else:
            ax, ay = 0.0, -GRAVITY
        vx += ax * DT
        vy += ay * DT
        x += vx * DT
        y += vy * DT
    return x, y, vx, vy


def _drag_path(c: float, vx: float, steps: int) -> list[tuple[float, float]]:
    """Every drawn position of the shell, for the frame proof."""
    vy, x, y = K35_VY0, K35_X0, K35_Y0
    points = [(x, y)]
    for _ in range(steps):
        x, y, vx, vy = _drag_state(c, vx, vy, x, y, 1)
        points.append((x, y))
    return points


def _build_k35(rng: random.Random) -> core.TaskInstance:
    c, steps = rng.choice(K35_RUNS)
    c = float(c)
    vx0 = round(rng.uniform(4.6, 6.2), 2)
    terminal = math.sqrt(GRAVITY / c)

    expected = _drag_state(c, vx0, K35_VY0, K35_X0, K35_Y0, steps)
    ex, ey, evx, evy = expected
    assert ey < K35_Y0 - 4.0, "снаряд не упал: проверка вырождена"
    assert abs(evy) < terminal, "вертикальная скорость обязана не превышать предел"
    assert evx < vx0, "сопротивление обязано тормозить горизонтально"

    linear = _drag_state(c, vx0, K35_VY0, K35_X0, K35_Y0, steps, "linear")
    free = _drag_state(c, vx0, K35_VY0, K35_X0, K35_Y0, steps, "none")
    assert abs(linear[0] - ex) > 0.05, "линейное сопротивление не отличимо по x"
    assert abs(free[1] - ey) > 0.05, "свободное падение не отличимо по y"

    prompt = K35_PROMPT.format(
        cam=list(K35_CAMERA),
        look=list(K35_LOOK),
        light=list(K35_LIGHT),
        r=K35_RADIUS,
        x0=K35_X0,
        y0=K35_Y0,
        vx0=vx0,
        vy0=K35_VY0,
        g=GRAVITY,
        c=c,
        steps=steps,
        fields=K35_FIELDS,
    )
    _assert_numbers_in_prompt(
        prompt,
        [K35_RADIUS, K35_X0, K35_Y0, vx0, K35_VY0, GRAVITY, c, *K35_CAMERA, *K35_LOOK, *K35_LIGHT],
    )
    _assert_in_frame(
        K35_CAMERA,
        K35_LOOK,
        [(px, py, 0.0) for px, py in _drag_path(c, vx0, steps + steps // 8 + 2)],
        K35_RADIUS,
        "снаряд",
    )
    _assert_in_frame(K35_CAMERA, K35_LOOK, [(-8.0, 11.8, 0.0)], 1.4, "обрыв", margin=0.95)

    golden = _page(
        K35_BODY,
        title="quadratic drag",
        camera=list(K35_CAMERA),
        look=list(K35_LOOK),
        light=list(K35_LIGHT),
        g=GRAVITY,
        c=c,
        x0=K35_X0,
        y0=K35_Y0,
        vx0=vx0,
        vy0=K35_VY0,
        r=K35_RADIUS,
    )

    task = visual.build_task(
        task_id="K-35",
        title="K-35 — снаряд с квадратичным сопротивлением",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=[
                visual.contains_all("objects", ["ground", "ledge", "shell"]),
                visual.near("camera.0", K35_CAMERA[0], 1e-4),
                visual.near("camera.1", K35_CAMERA[1], 1e-4),
                visual.near("camera.2", K35_CAMERA[2], 1e-4),
                visual.near("c", c, 1e-3),
                visual.near("terminal", round(terminal, 4), 1e-3),
            ],
            simulation=[
                visual.near("x", round(ex, 4), 1e-3),
                visual.near("y", round(ey, 4), 1e-3),
                visual.near("vx", round(evx, 4), 1e-3),
                visual.near("vy", round(evy, 4), 1e-3),
                visual.near("speed", round(math.hypot(evx, evy), 4), 1e-3),
                visual.near("t", round(steps * DT, 4), 1e-3),
                visual.near("steps", steps, 0),
            ],
            image=[
                visual.image_min("changed_fraction", 0.2),
                visual.image_min("distinct_colors", 5),
                visual.image_color("warm", 0.0012, "оранжевый снаряд различим"),
                visual.image_centroid("warm", 1, 0.1, 0.9, "снаряд виден в кадре"),
                visual.image_region(7, visual.cool, "низ кадра — тёмная земля"),
            ],
            steps=steps,
            dt=DT,
        ),
        golden=golden,
        rng_meta={
            "c": c,
            "vx0": vx0,
            "vy0": K35_VY0,
            "x0": K35_X0,
            "y0": K35_Y0,
            "steps": steps,
            "terminal": round(terminal, 4),
            "expected": [round(ex, 4), round(ey, 4), round(evx, 4), round(evy, 4)],
            "linear": [round(value, 4) for value in linear],
            "free": [round(value, 4) for value in free],
        },
    )
    return task


# --------------------------------------------------------------------------- #
# K-36 — Verlet chain with fixed relaxation
# --------------------------------------------------------------------------- #

K36_CAMERA = (0.0, 4.5, 16.0)
K36_LOOK = (0.0, 4.5, 0.0)
K36_LIGHT = (8.0, 16.0, 12.0)
K36_RADIUS = 0.16
K36_LINKS = 6
K36_LEN = 0.9
K36_ITERATIONS = 8
K36_DAMP = 0.99
K36_ROOT = (0.0, 9.0)
#: Angles (degrees, from the downward vertical) the six links start at.
K36_ANGLES = (-88.0, -84.0, -86.0, -82.0, -85.0, -80.0)

K36_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "anchor", "node-0", "...", "link-0", "..."],
  "camera": [0, 4.5, 16],
  "nodes": [[x0, y0], [x1, y1], ...],
  "lengths": [d0, d1, ...],
  "residual": <сумма |длина - L| по всем звеньям>,
  "len": <L>,
  "iterations": <число итераций релаксации>
}"""

K36_PROMPT = """Собери сцену: цепочка из шести звеньев висит на закреплённом узле и
качается под гравитацией.

Сцена (плоскость XY, z = 0):
* земля — плоскость 80x80 в плоскости y = -1.6, материал тёмно-синий
  (0x2b3a4a), имя `ground`;
* якорь — параллелепипед 1.2x0.4x0.4 в точке (0, 9.2, 0), материал
  серо-синий (0x8a93a6), имя `anchor`;
* узлы 0..6 — сферы радиуса {r}, имя `node-i`; узел 0 закреплён в точке
  {root} и не двигается, остальные свободны; материал оранжевый (0xff8a3d);
* звенья 0..5 — цилиндры радиуса 0.045 между узлами i и i+1, материал
  тёмно-серый (0x5b6472), имя `link-i`; длина звена — расстояние между
  узлами;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.5)` и
  `DirectionalLight(0xffffff, 1.9)` из точки {light}.

Симуляция:
* звеньев ровно 6, у всех одинаковая длина L = {L};
* шаг делается только вызовом `window.__step(dt)`; в шаге ровно {iterations}
  итераций релаксации, не больше;
* на шаге сначала интегрируются свободные узлы, каждый по отдельности:
  `nx = x + (x - px) * {damp};` затем
  `ny = y + (y - py) * {damp} - g * dt * dt;` (g = {g}, `px`/`py` —
  предыдущие координаты, узел 0 не интегрируется), после чего `px`/`py`
  заменяются на текущие `x`/`y`;
* затем {iterations} раз подряд, для звеньев i = 0..5 по возрастанию:
  `dx = x[i+1] - x[i]; dy = y[i+1] - y[i]; d = Math.hypot(dx, dy);
  diff = (d - L) / d;` далее, если i == 0 (узел 0 закреплён), сдвигается только
  узел i+1: `x[i+1] -= dx * diff; y[i+1] -= dy * diff;` иначе оба конца
  сдвигаются поровну: `x[i] += dx * diff / 2; y[i] += dy * diff / 2;
  x[i+1] -= dx * diff / 2; y[i+1] -= dy * diff / 2;`
  (если d == 0, звено пропускается);
* `residual` = сумма `|d - L|` по всем шести звеньям, посчитанная после
  последней итерации;
* `lengths` = текущие расстояния между узлами.

Начальные координаты узлов — цепочка, выведенная из стартовых углов
(в градусах от вертикали вниз): {angles}; узлы ставятся последовательно:
`x[i+1] = x[i] + L * sin(a)`, `y[i+1] = y[i] - L * cos(a)`, где a —
угол i-го звена. Начальные `px`/`py` равны текущим координатам.

`window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `ground`, `anchor`,
всех `node-*` и всех `link-*`, лишние имена не мешают.

Проверяется: состав сцены, все координаты узлов, длины звеньев и остаток
`residual` после {steps} шагов (Python повторяет ту же схему шаг в шаг, допуск
1e-3) и кадр — цепочка видна целиком, картинка воспроизводима при повторной
загрузке.
"""

K36_BODY = """
const G = {g}, L = {L}, DAMP = {damp}, ITER = {iterations};
const N = 7;
const POS = {angles};

let prev = POS.map((p) => [p[0], p[1]]);
let elapsed = 0, steps = 0;

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(80, 80),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -1.6;
ground.name = 'ground';
scene.add(ground);

const anchor = new THREE.Mesh(
  new THREE.BoxGeometry(1.2, 0.4, 0.4),
  new THREE.MeshStandardMaterial({{color: 0x8a93a6}})
);
anchor.position.set(0, 9.2, 0);
anchor.name = 'anchor';
scene.add(anchor);

const nodes = POS.map((_, i) => {{
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry({r}, 16, 12),
    new THREE.MeshStandardMaterial({{color: 0xff8a3d}})
  );
  mesh.name = `node-${{i}}`;
  scene.add(mesh);
  return mesh;
}});

const links = [];
for (let i = 0; i + 1 < N; i++) {{
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.045, 0.045, 1, 10),
    new THREE.MeshStandardMaterial({{color: 0x5b6472}})
  );
  mesh.name = `link-${{i}}`;
  scene.add(mesh);
  links.push(mesh);
}}

""" + _LINK + """
function draw() {{
  for (let i = 0; i < N; i++) nodes[i].position.set(POS[i][0], POS[i][1], 0);
  for (let i = 0; i + 1 < N; i++) link(links[i], POS[i][0], POS[i][1], POS[i + 1][0], POS[i + 1][1]);
  renderer.render(scene, camera);
}}

function residual() {{
  let sum = 0;
  for (let i = 0; i + 1 < N; i++) {{
    sum += Math.abs(Math.hypot(POS[i + 1][0] - POS[i][0], POS[i + 1][1] - POS[i][1]) - L);
  }}
  return sum;
}}

window.__step = (dt) => {{
  for (let i = 1; i < N; i++) {{
    const x = POS[i][0], y = POS[i][1];
    const nx = x + (x - prev[i][0]) * DAMP;
    const ny = y + (y - prev[i][1]) * DAMP - G * dt * dt;
    prev[i][0] = x;
    prev[i][1] = y;
    POS[i][0] = nx;
    POS[i][1] = ny;
  }}
  for (let it = 0; it < ITER; it++) {{
    for (let i = 0; i + 1 < N; i++) {{
      const dx = POS[i + 1][0] - POS[i][0];
      const dy = POS[i + 1][1] - POS[i][1];
      const d = Math.hypot(dx, dy);
      if (d === 0) continue;
      const diff = (d - L) / d;
      if (i === 0) {{
        POS[1][0] -= dx * diff;
        POS[1][1] -= dy * diff;
      }} else {{
        POS[i][0] += dx * diff / 2;
        POS[i][1] += dy * diff / 2;
        POS[i + 1][0] -= dx * diff / 2;
        POS[i + 1][1] -= dy * diff / 2;
      }}
    }}
  }}
  elapsed += dt;
  steps += 1;
  draw();
  return window.__probe();
}};

window.__probe = () => ({{
  steps,
  t: round(elapsed),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  camera: [camera.position.x, camera.position.y, camera.position.z],
  nodes: POS.map((p) => [round(p[0]), round(p[1])]),
  lengths: links.map((_, i) => round(Math.hypot(POS[i + 1][0] - POS[i][0], POS[i + 1][1] - POS[i][1]))),
  residual: round(residual()),
  len: L,
  iterations: ITER,
}});

draw();
window.__ready = true;
"""


def _chain_start(length: float, tilt: float = 0.0) -> list[tuple[float, float]]:
    """The published start: the chain laid out link by link from the angles."""
    x, y = K36_ROOT
    points = [(x, y)]
    for angle in K36_ANGLES:
        rad = math.radians(angle + tilt)
        x += length * math.sin(rad)
        y -= length * math.cos(rad)
        points.append((x, y))
    return points


def _chain_state(
    steps: int, length: float, damp: float, tilt: float, relax: bool = True
) -> tuple[list[list[float]], list[float], float]:
    """The stated Verlet scheme, replayed in Python, relaxation included per step."""
    pos = [[x, y] for x, y in _chain_start(length, tilt)]
    prev = [list(point) for point in pos]
    for _ in range(steps):
        for i in range(1, len(pos)):
            x, y = pos[i]
            nx = x + (x - prev[i][0]) * damp
            ny = y + (y - prev[i][1]) * damp - GRAVITY * DT * DT
            prev[i][0] = x
            prev[i][1] = y
            pos[i][0] = nx
            pos[i][1] = ny
        if relax:
            for _ in range(K36_ITERATIONS):
                for i in range(len(pos) - 1):
                    dx = pos[i + 1][0] - pos[i][0]
                    dy = pos[i + 1][1] - pos[i][1]
                    d = math.hypot(dx, dy)
                    if d == 0.0:
                        continue
                    diff = (d - length) / d
                    if i == 0:
                        pos[1][0] -= dx * diff
                        pos[1][1] -= dy * diff
                    else:
                        pos[i][0] += dx * diff / 2
                        pos[i][1] += dy * diff / 2
                        pos[i + 1][0] -= dx * diff / 2
                        pos[i + 1][1] -= dy * diff / 2
    lengths = [
        math.hypot(pos[i + 1][0] - pos[i][0], pos[i + 1][1] - pos[i][1])
        for i in range(len(pos) - 1)
    ]
    return pos, lengths, sum(abs(value - length) for value in lengths)


def _build_k36(rng: random.Random) -> core.TaskInstance:
    length = float(rng.choice([0.8, 0.88, 0.96]))
    damp = float(rng.choice([0.985, 0.99, 0.995]))
    tilt = round(rng.uniform(-4.0, 4.0), 2)
    steps = int(rng.choice([90, 120, 150]))
    pos, lengths, residual = _chain_state(steps, length, damp, tilt)
    assert residual < 8e-3, f"релаксация не сошлась: остаток {residual}"
    assert max(abs(value - length) for value in lengths) < 4e-3, (
        "звенья растянуты сильнее допуска"
    )

    loose, loose_lengths, loose_residual = _chain_state(
        steps, length, damp, tilt, relax=False
    )
    assert loose_residual > 20 * residual, "релаксация не отличима от её отсутствия"
    assert max(loose_lengths) > length * 1.05, "без релаксации звенья не растянуты"

    start = _chain_start(length, tilt)
    assert max(
        math.hypot(pos[i][0] - start[i][0], pos[i][1] - start[i][1]) for i in range(1, len(pos))
    ) > 2.0, "цепочка не сдвинулась: проверка вырождена"

    prompt = K36_PROMPT.format(
        cam=list(K36_CAMERA),
        look=list(K36_LOOK),
        light=list(K36_LIGHT),
        r=K36_RADIUS,
        root=list(K36_ROOT),
        L=length,
        g=GRAVITY,
        damp=damp,
        iterations=K36_ITERATIONS,
        angles=[round(angle + tilt, 2) for angle in K36_ANGLES],
        steps=steps,
        fields=K36_FIELDS,
    )
    _assert_numbers_in_prompt(
        prompt,
        [
            K36_RADIUS, length, GRAVITY, damp, K36_ITERATIONS,
            *K36_ROOT, *K36_CAMERA, *K36_LOOK, *K36_LIGHT,
            *[round(angle + tilt, 2) for angle in K36_ANGLES],
        ],
    )
    _assert_in_frame(
        K36_CAMERA,
        K36_LOOK,
        [(x, y, 0.0) for x, y in start + pos],
        K36_RADIUS,
        "цепочка",
    )

    golden = _page(
        K36_BODY,
        title="verlet chain",
        camera=list(K36_CAMERA),
        look=list(K36_LOOK),
        light=list(K36_LIGHT),
        g=GRAVITY,
        L=length,
        damp=damp,
        iterations=K36_ITERATIONS,
        angles=[list(point) for point in start],
        r=K36_RADIUS,
    )

    task = visual.build_task(
        task_id="K-36",
        title="K-36 — цепочка Verlet с релаксацией",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=[
                visual.contains_all("objects", ["ground", "anchor"]),
                visual.contains_all(
                    "objects",
                    [f"node-{i}" for i in range(K36_LINKS + 1)]
                    + [f"link-{i}" for i in range(K36_LINKS)],
                ),
                visual.near("camera.0", K36_CAMERA[0], 1e-4),
                visual.near("camera.1", K36_CAMERA[1], 1e-4),
                visual.near("camera.2", K36_CAMERA[2], 1e-4),
                visual.near("len", length, 1e-3),
                visual.near("iterations", K36_ITERATIONS, 0),
            ],
            simulation=[
                *[
                    predicate
                    for index, point in enumerate(pos)
                    for predicate in (
                        visual.near(f"nodes.{index}.0", round(point[0], 4), 1e-3),
                        visual.near(f"nodes.{index}.1", round(point[1], 4), 1e-3),
                    )
                ],
                *[
                    visual.near(f"lengths.{index}", round(length, 4), 4e-3)
                    for index, value in enumerate(lengths)
                ],
                visual.near("residual", round(residual, 4), 1e-3),
                visual.near("t", round(steps * DT, 4), 1e-3),
                visual.near("steps", steps, 0),
            ],
            image=[
                visual.image_min("changed_fraction", 0.2),
                visual.image_min("distinct_colors", 5),
                visual.image_color("warm", 0.0012, "оранжевые узлы различимы"),
                visual.image_centroid("warm", 1, 0.1, 0.9, "цепочка видна в кадре"),
                visual.image_region(7, visual.cool, "низ кадра — тёмная земля"),
            ],
            steps=steps,
            dt=DT,
        ),
        golden=golden,
        rng_meta={
            "length": length,
            "damp": damp,
            "steps": steps,
            "lengths": [round(length, 4) for length in lengths],
            "residual": round(residual, 4),
            "expected": [[round(x, 4), round(y, 4)] for x, y in pos],
            "loose_residual": round(loose_residual, 4),
            "loose": [[round(x, 4), round(y, 4)] for x, y in loose],
        },
    )
    return task


# --------------------------------------------------------------------------- #


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Both scenarios of this slice, in id order, drawn from the shared stream."""
    return [_build_k35(rng), _build_k36(rng)]
