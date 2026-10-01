"""Visual scenarios K-2..K-6 — point dynamics and a rigid body.

Five scenarios in the same shape as ``visual_projectiles``: the prompt states
the scene, the exact integration scheme and the exact probe fields, and the
expected numbers are recomputed here in Python from that very scheme. The
reference page is a real answer — ``selfcheck`` renders it in the browser and
requires all four checks (html, scene, simulation, image) to pass.

The schemes are all *deterministic and non-chaotic* on purpose: a free rigid
body spinning about its short axis and a spring chain do not amplify a
rounding difference the way a double pendulum or an n-body system would, so a
1e-3 tolerance is a statement about the physics rather than about float luck.
"""

from __future__ import annotations

import json
import math
import random
from typing import Iterable

from .. import core, visual

IDS = ("K-2", "K-3", "K-4", "K-5", "K-6")
TITLE = "K-2..K-6 — динамика: пружина, маятник, соударения, цепочка, твёрдое тело"

#: The frame every scenario is rendered in. Fixed per scenario so two answers
#: are compared on the same picture.
DT = 1 / 60

# --------------------------------------------------------------------------- #
# Shared page skeleton
# --------------------------------------------------------------------------- #

#: Identical for all five: the renderer contract from the common HTML block, a
#: fixed camera and the two lights. ``{title}``/``{camera}``/``{look}``/
#: ``{light}`` are filled per scenario.
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
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(...{light});
scene.add(key);

const round = (value) => Math.round(value * 1e4) / 1e4;
"""

_PAGE_TAIL = """</script>
</body></html>
"""

#: Aligns a unit-length cylinder with the segment ``a -> b``: the shared way a
#: rod or a spring is drawn, so its length is the distance and its angle is the
#: direction. Only the scenarios that draw a link include it.
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


def _vec(values: list[float]) -> str:
    return json.dumps(values)


FOV = 50.0
ASPECT = 800 / 600


def _in_frame(
    point: tuple[float, float, float],
    radius: float,
    camera_position: tuple[float, float, float],
    look: tuple[float, float, float],
    margin: float = 0.7,
) -> bool:
    """Is a sphere of ``radius`` around ``point`` inside the central ``margin``?

    The frame is what the ``image`` check reads, so an object that leaves the
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


def _assert_visible(
    point: tuple[float, float, float],
    radius: float,
    camera_position: tuple[float, float, float],
    look: tuple[float, float, float],
    what: str,
    margin: float = 0.7,
) -> None:
    """The subject of the frame must be on screen by construction."""
    assert _in_frame(point, radius, camera_position, look, margin), (
        f"{what} выходит за кадр: {point} (радиус {radius})"
    )


def _assert_numbers_in_prompt(prompt: str, values: Iterable[float]) -> None:
    """Every fixture number must be readable in the prompt.

    A scenario whose constants are not printed cannot be answered: the model is
    asked to reproduce a program, not to guess its parameters.
    """
    missing = [value for value in values if str(float(value)) not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


# --------------------------------------------------------------------------- #
# K-2 — mass on a horizontal spring
# --------------------------------------------------------------------------- #

K2_CAMERA = (0.0, 3.5, 15.0)
K2_LOOK = (0.0, 0.2, 0.0)
K2_LIGHT = (8.0, 14.0, 10.0)

K2_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "wall", "mass", "spring", "..."],
  "camera": [0, 3.5, 15],
  "x": <координата груза>,
  "v": <скорость груза>,
  "k": <жёсткость>,
  "m": <масса>
}"""

K2_PROMPT = """Собери сцену: груз на горизонтальной пружине колеблется вдоль оси X.

Сцена:
* земля — плоскость 60x60 в плоскости y = -0.3, материал тёмно-синий
  (0x2b3a4a), имя объекта `ground`;
* стена — параллелепипед 0.8x3.2x3.2 в точке (-7, 0.9, 0), материал
  серо-синий (0x8a93a6), имя `wall`;
* груз — куб 1.2x1.2x1.2, центр в точке (x, 0.3, 0), материал оранжевый
  (0xff8a3d), имя `mass`;
* пружина — цилиндр радиуса 0.09, соединяющий точку (-6.6, 0.3, 0) с центром
  груза, материал светлый (0xd8dde6), имя `spring`; пружина обязана тянуться
  за грузом (её длина — расстояние между этими точками);
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.45)` и
  `DirectionalLight(0xffffff, 2.0)` из точки {light}.

Симуляция:
* шаг делается только вызовом `window.__step(dt)`;
* состояние — координата `x` и скорость `v`;
* схема шага — полунеявный Эйлер ровно в таком порядке:
  `v += (-k/m * x - c*v) * dt;` затем `x += v * dt`;
* константы: k = {k}, m = {m}, c = {c}, начальное состояние x = {x0}, v = {v0};
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `ground`, `wall`, `mass`
и `spring`, лишние имена не мешают.

Проверяется: состав сцены и камера, координата и скорость груза после
заданного числа шагов (сверка с той же схемой Эйлера, допуск 1e-3), и кадр —
груз и пружина видны, картинка воспроизводима при повторной загрузке.
"""

K2_BODY = """
const K = {k}, M = {m}, C = {c};
let x = {x0}, v = {v0};
let elapsed = 0, steps = 0;

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(60, 60),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.3;
ground.name = 'ground';
scene.add(ground);

const wall = new THREE.Mesh(
  new THREE.BoxGeometry(0.8, 3.2, 3.2),
  new THREE.MeshStandardMaterial({{color: 0x8a93a6}})
);
wall.position.set(-7, 0.9, 0);
wall.name = 'wall';
scene.add(wall);

const mass = new THREE.Mesh(
  new THREE.BoxGeometry(1.2, 1.2, 1.2),
  new THREE.MeshStandardMaterial({{color: 0xff8a3d}})
);
mass.name = 'mass';
scene.add(mass);

const spring = new THREE.Mesh(
  new THREE.CylinderGeometry(0.09, 0.09, 1, 10),
  new THREE.MeshStandardMaterial({{color: 0xd8dde6}})
);
spring.name = 'spring';
scene.add(spring);
"""
K2_BODY += _LINK + """
const ANCHOR_X = -6.6, ANCHOR_Y = 0.3, MASS_Y = 0.3;

function draw() {{
  mass.position.set(x, MASS_Y, 0);
  link(spring, ANCHOR_X, ANCHOR_Y, x, MASS_Y);
  renderer.render(scene, camera);
}}

window.__step = (dt) => {{
  v += (-K / M * x - C * v) * dt;
  x += v * dt;
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
  v: round(v),
  k: K,
  m: M,
}});

draw();
window.__ready = true;
"""


def _spring_state(k: float, m: float, c: float, x: float, v: float, steps: int) -> tuple[float, float]:
    """Semi-implicit Euler, recomputed independently of the reference page."""
    for _ in range(steps):
        v += (-k / m * x - c * v) * DT
        x += v * DT
    return x, v


def _build_k2(rng: random.Random) -> core.TaskInstance:
    for _ in range(80):
        k = round(rng.uniform(7.0, 13.0), 2)
        m = float(rng.choice([0.5, 0.75, 1.0, 1.5]))
        c = round(rng.uniform(0.15, 0.7), 2)
        x0 = round(rng.uniform(-2.0, 2.0), 2)
        v0 = round(rng.uniform(-1.6, 1.6), 2)
        steps = int(rng.choice([60, 90, 120]))
        expected_x, expected_v = _spring_state(k, m, c, x0, v0, steps)
        # The naive answer — "the mass just keeps its velocity" — must be a
        # different answer, otherwise the check would not be about the equation,
        # and the correct answer must stay inside the camera.
        naive_x = x0 + v0 * steps * DT
        constants = [f"{value:.2f}" for value in (k, m, c, x0, v0)]
        if (
            abs(x0) >= 0.5
            and abs(v0) >= 0.4
            and len(set(constants)) == 5
            and abs(naive_x - expected_x) > 0.3
            and abs(expected_x) < 4.5
        ):
            break
    else:  # pragma: no cover - the ranges are wide enough for a hit every time
        raise AssertionError("не удалось подобрать фикстуру K-2")

    prompt = K2_PROMPT.format(
        cam=list(K2_CAMERA),
        look=list(K2_LOOK),
        light=list(K2_LIGHT),
        k=k,
        m=m,
        c=c,
        x0=x0,
        v0=v0,
        fields=K2_FIELDS,
    )
    _assert_numbers_in_prompt(
        prompt, [k, m, c, x0, v0, *K2_CAMERA, *K2_LOOK, *K2_LIGHT]
    )
    _assert_visible((expected_x, 0.3, 0.0), 0.85, K2_CAMERA, K2_LOOK, "груз")
    _assert_visible(
        (-7.0, 0.9, 0.0), 0.5, K2_CAMERA, K2_LOOK, "стена", margin=0.95
    )
    golden = _page(
        K2_BODY,
        title="spring-mass",
        camera=list(K2_CAMERA),
        look=list(K2_LOOK),
        light=list(K2_LIGHT),
        k=k,
        m=m,
        c=c,
        x0=x0,
        v0=v0,
    )

    task = visual.build_task(
        task_id="K-2",
        title="K-2 — груз на пружине",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=[
                visual.contains_all("objects", ["ground", "wall", "mass", "spring"]),
                visual.near("camera.0", K2_CAMERA[0], 1e-4),
                visual.near("camera.1", K2_CAMERA[1], 1e-4),
                visual.near("camera.2", K2_CAMERA[2], 1e-4),
            ],
            simulation=[
                visual.near("x", round(expected_x, 4), 1e-3),
                visual.near("v", round(expected_v, 4), 1e-3),
                visual.near("k", k, 1e-3),
                visual.near("m", m, 1e-3),
                visual.near("t", round(steps * DT, 4), 1e-3),
                visual.near("steps", steps, 0),
            ],
            image=[
                visual.image_min("changed_fraction", 0.2),
                visual.image_min("distinct_colors", 5),
                visual.image_color("warm", 0.003, "оранжевый груз различим"),
                visual.image_centroid("warm", 1, 0.2, 0.8, "груз виден в кадре"),
                visual.image_region(7, visual.cool, "низ кадра — тёмная земля"),
            ],
            steps=steps,
            dt=DT,
        ),
        golden=golden,
        rng_meta={
            "k": k,
            "m": m,
            "c": c,
            "x0": x0,
            "v0": v0,
            "steps": steps,
            "expected": [round(expected_x, 4), round(expected_v, 4)],
            "naive": round(naive_x, 4),
        },
    )
    return task


# --------------------------------------------------------------------------- #
# K-3 — damped pendulum
# --------------------------------------------------------------------------- #

K3_CAMERA = (0.0, 1.8, 9.0)
K3_LOOK = (0.0, 1.4, 0.0)
K3_LIGHT = (5.0, 10.0, 9.0)
K3_GRAVITY = 9.81

K3_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "support", "rod", "bob", "..."],
  "camera": [0, 1.8, 9],
  "theta": <угол в радианах>,
  "omega": <угловая скорость>,
  "L": <длина подвеса>
}"""

K3_PROMPT = """Собери сцену: затухающий маятник, камера смотрит сбоку (вдоль оси Z).

Сцена:
* земля — плоскость 60x60 в плоскости y = -1.2, материал тёмно-синий
  (0x2b3a4a), имя `ground`;
* опора — параллелепипед 3.5x0.3x0.5 в точке (0, 3.15, 0), материал
  серо-синий (0x8a93a6), имя `support`;
* стержень — цилиндр радиуса 0.07, соединяющий точку подвеса (0, 3, 0) с
  центром шара, материал тёмно-серый (0x5b6472), имя `rod`; длина стержня —
  расстояние от подвеса до шара;
* шар — сфера радиуса 0.3 в точке (L*sin(theta), 3 - L*cos(theta), 0),
  материал оранжевый (0xff8a3d), имя `bob`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.45)` и
  `DirectionalLight(0xffffff, 2.0)` из точки {light}.

Симуляция:
* шаг делается только вызовом `window.__step(dt)`;
* состояние — угол `theta` (отсчитывается от нижнего положения, положительный
  в сторону +X) и угловая скорость `omega`;
* схема шага — полунеявный Эйлер ровно в таком порядке:
  `omega += (-g/L * Math.sin(theta) - c*omega) * dt;` затем `theta += omega * dt`;
* константы: g = {g}, L = {L}, c = {c}, начальное состояние theta = {theta0},
  omega = {omega0};
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `ground`, `support`,
`rod` и `bob`, лишние имена не мешают.

Проверяется: состав сцены и камера, угол и угловая скорость маятника после
заданного числа шагов (сверка с той же схемой Эйлера, допуск 1e-3), и кадр —
шар и стержень видны, картинка воспроизводима при повторной загрузке.
"""

K3_BODY = """
const G = {g}, L = {L}, C = {c};
let theta = {theta0}, omega = {omega0};
let elapsed = 0, steps = 0;

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(60, 60),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -1.2;
ground.name = 'ground';
scene.add(ground);

const support = new THREE.Mesh(
  new THREE.BoxGeometry(3.5, 0.3, 0.5),
  new THREE.MeshStandardMaterial({{color: 0x8a93a6}})
);
support.position.set(0, 3.15, 0);
support.name = 'support';
scene.add(support);

const rod = new THREE.Mesh(
  new THREE.CylinderGeometry(0.07, 0.07, 1, 10),
  new THREE.MeshStandardMaterial({{color: 0x5b6472}})
);
rod.name = 'rod';
scene.add(rod);

const bob = new THREE.Mesh(
  new THREE.SphereGeometry(0.3, 24, 16),
  new THREE.MeshStandardMaterial({{color: 0xff8a3d}})
);
bob.name = 'bob';
scene.add(bob);
"""
K3_BODY += _LINK + """
const PIVOT_X = 0, PIVOT_Y = 3;

function draw() {{
  const bx = PIVOT_X + L * Math.sin(theta);
  const by = PIVOT_Y - L * Math.cos(theta);
  bob.position.set(bx, by, 0);
  link(rod, PIVOT_X, PIVOT_Y, bx, by);
  renderer.render(scene, camera);
}}

window.__step = (dt) => {{
  omega += (-G / L * Math.sin(theta) - C * omega) * dt;
  theta += omega * dt;
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
  theta: round(theta),
  omega: round(omega),
  L: L,
}});

draw();
window.__ready = true;
"""


def _pendulum_state(L: float, c: float, theta: float, omega: float, steps: int) -> tuple[float, float]:
    """Semi-implicit Euler, recomputed independently of the reference page."""
    for _ in range(steps):
        omega += (-K3_GRAVITY / L * math.sin(theta) - c * omega) * DT
        theta += omega * DT
    return theta, omega


def _build_k3(rng: random.Random) -> core.TaskInstance:
    L = round(rng.uniform(1.6, 2.4), 2)
    c = round(rng.uniform(0.05, 0.4), 2)
    theta0 = round(rng.uniform(0.7, 1.5), 3)
    omega0 = 0.0
    steps = int(rng.choice([60, 90, 120]))

    expected_theta, expected_omega = _pendulum_state(L, c, theta0, omega0, steps)
    assert abs(expected_theta - theta0) > 0.2, "маятник не сдвинулся: проверка вырождена"
    assert abs(expected_theta) < 1.6, "маятник ушёл за пределы кадра"

    prompt = K3_PROMPT.format(
        cam=list(K3_CAMERA),
        look=list(K3_LOOK),
        light=list(K3_LIGHT),
        g=K3_GRAVITY,
        L=L,
        c=c,
        theta0=theta0,
        omega0=f"{omega0:.1f}",
        fields=K3_FIELDS,
    )
    _assert_numbers_in_prompt(
        prompt, [K3_GRAVITY, L, c, theta0, omega0, *K3_CAMERA, *K3_LOOK, *K3_LIGHT]
    )
    _assert_visible(
        (L * math.sin(expected_theta), 3.0 - L * math.cos(expected_theta), 0.0),
        0.3,
        K3_CAMERA,
        K3_LOOK,
        "шар маятника",
    )
    _assert_visible(
        (0.0, 3.15, 0.0), 0.5, K3_CAMERA, K3_LOOK, "опора", margin=0.9
    )
    golden = _page(
        K3_BODY,
        title="pendulum",
        camera=list(K3_CAMERA),
        look=list(K3_LOOK),
        light=list(K3_LIGHT),
        g=K3_GRAVITY,
        L=L,
        c=c,
        theta0=theta0,
        omega0=f"{omega0:.1f}",
    )

    task = visual.build_task(
        task_id="K-3",
        title="K-3 — затухающий маятник",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=[
                visual.contains_all("objects", ["ground", "support", "rod", "bob"]),
                visual.near("camera.0", K3_CAMERA[0], 1e-4),
                visual.near("camera.1", K3_CAMERA[1], 1e-4),
                visual.near("camera.2", K3_CAMERA[2], 1e-4),
                visual.near("L", L, 1e-3),
            ],
            simulation=[
                visual.near("theta", round(expected_theta, 4), 1e-3),
                visual.near("omega", round(expected_omega, 4), 1e-3),
                visual.near("t", round(steps * DT, 4), 1e-3),
                visual.near("steps", steps, 0),
            ],
            image=[
                visual.image_min("changed_fraction", 0.2),
                visual.image_min("distinct_colors", 5),
                visual.image_color("warm", 0.0015, "оранжевый шар различим"),
                visual.image_centroid("warm", 1, 0.2, 0.8, "шар виден в кадре"),
                visual.image_region(7, visual.cool, "низ кадра — тёмная земля"),
            ],
            steps=steps,
            dt=DT,
        ),
        golden=golden,
        rng_meta={
            "L": L,
            "c": c,
            "theta0": theta0,
            "omega0": omega0,
            "steps": steps,
            "expected": [round(expected_theta, 4), round(expected_omega, 4)],
        },
    )
    return task


# --------------------------------------------------------------------------- #
# K-4 — elastic collisions on a line
# --------------------------------------------------------------------------- #

K4_CAMERA = (0.0, 3.8, 12.0)
K4_LOOK = (0.0, 0.2, 0.0)
K4_LIGHT = (7.0, 13.0, 10.0)
K4_RADIUS = 0.4

K4_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["floor", "wall-left", "wall-right", "ball-0", "...", "..."],
  "camera": [0, 4.5, 15],
  "balls": [{"x": <координата>, "v": <скорость>}, ...],
  "collisions": <сколько столкновений обработано с начала>
}"""

K4_PROMPT = """Собери сцену: три шара упруго соударяются на прямой в коробке.

Сцена:
* пол — плоскость {floor_w} x 60 в плоскости y = -R (R — радиус шара),
  материал тёмно-синий (0x2b3a4a), имя `floor`;
* две стенки коробки — параллелепипеды 0.3x2.4x1.6 в точках
  ({wall_left}, 0.6, 0) и ({wall_right}, 0.6, 0), материал серо-синий
  (0x8a93a6), имена `wall-left` и `wall-right`;
* три шара — сферы радиуса R = {r}, материал оранжевый (0xff8a3d), имена
  `ball-0`, `ball-1`, `ball-2`. Шар i рисуется в точке (x_i - W/2, 0, 0):
  начало коробки x = 0 сдвинуто в центр кадра;
* коробка занимает отрезок [0, W], W = {w}; скорости и координаты шаров
  хранятся в этой системе координат (без сдвига);
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.45)` и
  `DirectionalLight(0xffffff, 2.0)` из точки {light}.

Симуляция (одна и та же дискретная схема, шаг за шагом, без непрерывной
физики):
1. `x_i += v_i * dt` для всех шаров;
2. столкновения пар обрабатываются по порядку индексов: сначала пара (0, 1),
   затем пара (1, 2); для пары (a, b): если `x_b - x_a < 2*R`, то
   `d = (2*R - (x_b - x_a)) / 2`, затем `x_a -= d`, `x_b += d` и скорости
   `v_a` и `v_b` меняются местами (массы равны), счётчик `collisions`
   увеличивается на 1;
3. отскок от стен: для каждого шара, если `x_i - R < 0`, то `x_i = R` и
   `v_i = -v_i`; если `x_i + R > W`, то `x_i = W - R` и `v_i = -v_i`; каждое
   такое событие увеличивает `collisions` на 1.

Шары нумеруются в порядке массива `balls`; других столкновений нет.
Начальное состояние — `balls = {balls0}`: координаты и скорости в системе
координат коробки [0, W], время t = 0.

`window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `floor`, `wall-left`,
`wall-right` и всех трёх `ball-*`, лишние имена не мешают.

Проверяется: состав сцены и камера, координаты и скорости всех шаров после
заданного числа шагов (Python повторяет ту же дискретную схему шаг в шаг,
допуск 1e-3), число обработанных столкновений, и кадр — шары видны, картинка
воспроизводима при повторной загрузке.
"""

K4_BODY = """
const W = {w}, R = {r};
const X = {x0};
const V = {v0};
let collisions = 0;
let elapsed = 0, steps = 0;

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(2 * W + 4, 60),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -R;
floor.name = 'floor';
scene.add(floor);

for (const side of [-1, 1]) {{
  const wall = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 2.4, 1.6),
    new THREE.MeshStandardMaterial({{color: 0x8a93a6}})
  );
  wall.position.set(side * (W / 2 + 0.15), 0.6, 0);
  wall.name = side < 0 ? 'wall-left' : 'wall-right';
  scene.add(wall);
}}

const meshes = X.map((_, i) => {{
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(R, 24, 16),
    new THREE.MeshStandardMaterial({{color: 0xff8a3d}})
  );
  mesh.name = `ball-${{i}}`;
  mesh.position.set(X[i] - W / 2, 0, 0);
  scene.add(mesh);
  return mesh;
}});

function draw() {{
  for (let i = 0; i < meshes.length; i++) meshes[i].position.set(X[i] - W / 2, 0, 0);
  renderer.render(scene, camera);
}}

window.__step = (dt) => {{
  for (let i = 0; i < X.length; i++) X[i] += V[i] * dt;
  for (let i = 0; i + 1 < X.length; i++) {{
    const gap = X[i + 1] - X[i];
    if (gap < 2 * R) {{
      const d = (2 * R - gap) / 2;
      X[i] -= d;
      X[i + 1] += d;
      const swap = V[i];
      V[i] = V[i + 1];
      V[i + 1] = swap;
      collisions += 1;
    }}
  }}
  for (let i = 0; i < X.length; i++) {{
    if (X[i] - R < 0) {{ X[i] = R; V[i] = -V[i]; collisions += 1; }}
    if (X[i] + R > W) {{ X[i] = W - R; V[i] = -V[i]; collisions += 1; }}
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
  balls: X.map((x, i) => ({{ x: round(x), v: round(V[i]) }})),
  collisions,
}});

draw();
window.__ready = true;
"""


def _collision_state(
    xs: list[float], vs: list[float], w: float, r: float, steps: int
) -> tuple[list[float], list[float], int]:
    """The stated discrete scheme, replayed step by step in Python."""
    xs = list(xs)
    vs = list(vs)
    collisions = 0
    for _ in range(steps):
        for i in range(len(xs)):
            xs[i] += vs[i] * DT
        for i in range(len(xs) - 1):
            gap = xs[i + 1] - xs[i]
            if gap < 2 * r:
                d = (2 * r - gap) / 2
                xs[i] -= d
                xs[i + 1] += d
                vs[i], vs[i + 1] = vs[i + 1], vs[i]
                collisions += 1
        for i in range(len(xs)):
            if xs[i] - r < 0:
                xs[i] = r
                vs[i] = -vs[i]
                collisions += 1
            if xs[i] + r > w:
                xs[i] = w - r
                vs[i] = -vs[i]
                collisions += 1
    return xs, vs, collisions


def _build_k4(rng: random.Random) -> core.TaskInstance:
    w = float(rng.choice([8.0, 9.0, 10.0]))
    r = K4_RADIUS
    steps = int(rng.choice([120, 150]))

    xs: list[float] = []
    vs: list[float] = []
    collisions = 0
    expected: list[dict[str, float]] = []
    naive: list[float] = []
    for _ in range(80):
        xs = [round(w * fraction + rng.uniform(-0.3, 0.3), 2) for fraction in (0.2, 0.5, 0.8)]
        vs = [
            round(rng.uniform(1.5, 2.5) * rng.choice([-1.0, 1.0]), 2)
            for _ in range(3)
        ]
        xs_after, vs_after, collisions = _collision_state(xs, vs, w, r, steps)
        expected = [
            {"x": round(x, 4), "v": round(v, 4)} for x, v in zip(xs_after, vs_after)
        ]
        # Without the collision handling the balls would simply fly out of the
        # box: the fixture must separate the two answers, exercise both the pair
        # and the wall branch of the scheme, and keep the balls in the frame.
        naive = [round(x + v * steps * DT, 4) for x, v in zip(xs, vs)]
        if (
            collisions >= 3
            and all(r <= x <= w - r for x in xs_after)
            and len({abs(v) for v in vs}) == 3
            and min(xs[index + 1] - xs[index] for index in range(2)) > 2 * r
            and max(abs(n - e["x"]) for n, e in zip(naive, expected)) > 0.5
        ):
            break
    else:  # pragma: no cover - the draw is wide enough that this never happens
        raise AssertionError("не удалось подобрать фикстуру со столкновениями")

    prompt = K4_PROMPT.format(
        r=r,
        w=w,
        wall_left=round(-(w / 2 + 0.15), 2),
        wall_right=round(w / 2 + 0.15, 2),
        floor_w=round(2 * w + 4, 2),
        cam=list(K4_CAMERA),
        look=list(K4_LOOK),
        light=list(K4_LIGHT),
        balls0=json.dumps(
            [{"x": x, "v": v} for x, v in zip(xs, vs)], ensure_ascii=False
        ),
        fields=K4_FIELDS,
    )
    _assert_numbers_in_prompt(
        prompt, [w, r, round(-(w / 2 + 0.15), 2), round(w / 2 + 0.15, 2), round(2 * w + 4, 2)] + xs + vs
    )
    for ball in expected:
        _assert_visible(
            (ball["x"] - w / 2, 0.0, 0.0),
            r,
            K4_CAMERA,
            K4_LOOK,
            f"шар на x = {ball['x']}",
        )
    golden = _page(
        K4_BODY,
        title="collisions",
        camera=list(K4_CAMERA),
        look=list(K4_LOOK),
        light=list(K4_LIGHT),
        w=w,
        r=r,
        x0=_vec(xs),
        v0=_vec(vs),
    )

    task = visual.build_task(
        task_id="K-4",
        title="K-4 — упругие соударения на прямой",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=[
                visual.contains_all(
                    "objects",
                    ["floor", "wall-left", "wall-right", "ball-0", "ball-1", "ball-2"],
                ),
                visual.near("camera.0", K4_CAMERA[0], 1e-4),
                visual.near("camera.1", K4_CAMERA[1], 1e-4),
                visual.near("camera.2", K4_CAMERA[2], 1e-4),
                visual.size_between("balls", 3, 3),
            ],
            simulation=[
                visual.near(f"balls.{index}.x", ball["x"], 1e-3)
                for index, ball in enumerate(expected)
            ]
            + [
                visual.near(f"balls.{index}.v", ball["v"], 1e-3)
                for index, ball in enumerate(expected)
            ]
            + [
                visual.near("collisions", collisions, 0),
                visual.near("t", round(steps * DT, 4), 1e-3),
                visual.near("steps", steps, 0),
            ],
            image=[
                visual.image_min("changed_fraction", 0.2),
                visual.image_min("distinct_colors", 5),
                visual.image_color("warm", 0.004, "оранжевые шары различимы"),
                visual.image_centroid("warm", 1, 0.2, 0.8, "шары видны в кадре"),
                visual.image_region(7, visual.cool, "низ кадра — тёмный пол"),
            ],
            steps=steps,
            dt=DT,
        ),
        golden=golden,
        rng_meta={
            "w": w,
            "r": r,
            "x0": xs,
            "v0": vs,
            "steps": steps,
            "collisions": collisions,
            "expected": expected,
        },
    )
    return task


# --------------------------------------------------------------------------- #
# K-5 — three masses on two springs
# --------------------------------------------------------------------------- #

K5_CAMERA = (0.0, 3.2, 13.0)
K5_LOOK = (0.0, 0.6, 0.0)
K5_LIGHT = (7.0, 13.0, 10.0)

K5_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["floor", "rail", "cart-0", "cart-1", "cart-2", "spring-01", "spring-12"],
  "camera": [0, 3.2, 13],
  "x": [<x1>, <x2>, <x3>],
  "v": [<v1>, <v2>, <v3>]
}"""

K5_PROMPT = """Собери сцену: цепочка из трёх грузов, соединённых пружинами.

Сцена:
* пол — плоскость 40x40 в плоскости y = 0, материал тёмно-синий (0x2b3a4a),
  имя `floor`;
* рельс — параллелепипед 24x0.12x0.6 в точке (0, 0.06, 0), материал
  серо-синий (0x8a93a6), имя `rail`;
* три груза — кубы 0.9x0.9x0.9 с центрами (x_i, 0.45, 0), материал оранжевый
  (0xff8a3d), имена `cart-0`, `cart-1`, `cart-2`;
* две пружины — цилиндры радиуса 0.06, соединяющие центры соседних грузов:
  `spring-01` между грузами 0 и 1, `spring-12` между грузами 1 и 2; материал
  светлый (0xd8dde6);
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.45)` и
  `DirectionalLight(0xffffff, 2.0)` из точки {light}.

Симуляция:
* шаг делается только вызовом `window.__step(dt)`;
* состояние — координаты x1, x2, x3 и скорости v1, v2, v3;
* жёсткость пружин k = {k}, масса каждого груза m = {m}, коэффициент демпфера
  c = {c}, длина покоя пружины d0 = {d0};
* силы считаются по текущим координатам и скоростям (до обновления):
  `e12 = x2 - x1 - d0`, `e23 = x3 - x2 - d0`,
  `F1 = k*e12 + c*(v2 - v1)`,
  `F2 = -k*e12 + c*(v1 - v2) + k*e23 + c*(v3 - v2)`,
  `F3 = -k*e23 + c*(v2 - v3)`;
* затем полунеявный Эйлер: `v_i += F_i/m * dt` для всех трёх, и только после
  этого `x_i += v_i * dt` для всех трёх;
* начальное состояние: x = {x0}, v = {v0};
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `floor`, `rail`, всех
трёх `cart-*` и обеих `spring-*`, лишние имена не мешают.

Проверяется: состав сцены и камера, координаты и скорости всех трёх грузов
после заданного числа шагов (сверка с той же схемой, допуск 1e-3), и кадр —
грузы и пружины видны, картинка воспроизводима при повторной загрузке.
"""

K5_BODY = """
const K = {k}, M = {m}, C = {c}, D0 = {d0};
const X = {x0};
const V = {v0};
let elapsed = 0, steps = 0;

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
floor.rotation.x = -Math.PI / 2;
floor.name = 'floor';
scene.add(floor);

const rail = new THREE.Mesh(
  new THREE.BoxGeometry(24, 0.12, 0.6),
  new THREE.MeshStandardMaterial({{color: 0x8a93a6}})
);
rail.position.set(0, 0.06, 0);
rail.name = 'rail';
scene.add(rail);

const carts = X.map((_, i) => {{
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.9, 0.9, 0.9),
    new THREE.MeshStandardMaterial({{color: 0xff8a3d}})
  );
  mesh.name = `cart-${{i}}`;
  scene.add(mesh);
  return mesh;
}});

const springs = ['spring-01', 'spring-12'].map((name) => {{
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.06, 1, 10),
    new THREE.MeshStandardMaterial({{color: 0xd8dde6}})
  );
  mesh.name = name;
  scene.add(mesh);
  return mesh;
}});
"""
K5_BODY += _LINK + """
const CART_Y = 0.45;

function draw() {{
  for (let i = 0; i < carts.length; i++) carts[i].position.set(X[i], CART_Y, 0);
  for (let i = 0; i < springs.length; i++) {{
    link(springs[i], X[i], CART_Y, X[i + 1], CART_Y);
  }}
  renderer.render(scene, camera);
}}

window.__step = (dt) => {{
  const e12 = X[1] - X[0] - D0;
  const e23 = X[2] - X[1] - D0;
  const f1 = K * e12 + C * (V[1] - V[0]);
  const f2 = -K * e12 + C * (V[0] - V[1]) + K * e23 + C * (V[2] - V[1]);
  const f3 = -K * e23 + C * (V[1] - V[2]);
  const force = [f1, f2, f3];
  for (let i = 0; i < 3; i++) V[i] += force[i] / M * dt;
  for (let i = 0; i < 3; i++) X[i] += V[i] * dt;
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
  x: X.map((value) => round(value)),
  v: V.map((value) => round(value)),
}});

draw();
window.__ready = true;
"""


def _chain_state(
    k: float, m: float, c: float, d0: float, xs: list[float], vs: list[float], steps: int
) -> tuple[list[float], list[float]]:
    """Explicit-Euler forces, then semi-implicit position update."""
    xs = list(xs)
    vs = list(vs)
    for _ in range(steps):
        e12 = xs[1] - xs[0] - d0
        e23 = xs[2] - xs[1] - d0
        force = [
            k * e12 + c * (vs[1] - vs[0]),
            -k * e12 + c * (vs[0] - vs[1]) + k * e23 + c * (vs[2] - vs[1]),
            -k * e23 + c * (vs[1] - vs[2]),
        ]
        for i in range(3):
            vs[i] += force[i] / m * DT
        for i in range(3):
            xs[i] += vs[i] * DT
    return xs, vs


def _build_k5(rng: random.Random) -> core.TaskInstance:
    xs_after: list[float] = []
    vs_after: list[float] = []
    for _ in range(80):
        k = round(rng.uniform(9.0, 16.0), 2)
        m = float(rng.choice([0.5, 1.0, 2.0]))
        c = round(rng.uniform(0.2, 0.7), 2)
        d0 = round(rng.uniform(1.5, 2.2), 2)
        steps = int(rng.choice([60, 90, 120]))
        offsets = [round(rng.uniform(-0.4, 0.4), 2) for _ in range(3)]
        speed = round(rng.uniform(0.6, 1.4), 2)
        xs = [round(-d0 + offsets[0], 2), offsets[1], round(d0 + offsets[2], 2)]
        # Zero total momentum: the centre of mass stands still and the chain
        # oscillates instead of drifting out of the camera.
        vs = [speed, 0.0, -speed]
        xs_after, vs_after = _chain_state(k, m, c, d0, xs, vs, steps)
        naive = [round(x + v * steps * DT, 4) for x, v in zip(xs, vs)]
        if (
            len(set(offsets)) == 3
            and abs(naive[0] - xs_after[0]) > 0.3
            and max(abs(value) for value in xs_after) < 4.5
        ):
            break
    else:  # pragma: no cover - the draw is wide enough for a hit every time
        raise AssertionError("не удалось подобрать фикстуру K-5")
    assert max(abs(x) for x in xs_after) < 4.5, "груз уезжает из кадра"
    assert abs(sum(vs)) < 1e-9, "центр масс должен стоять на месте"

    prompt = K5_PROMPT.format(
        cam=list(K5_CAMERA),
        look=list(K5_LOOK),
        light=list(K5_LIGHT),
        k=k,
        m=m,
        c=c,
        d0=d0,
        x0=_vec(xs),
        v0=_vec(vs),
        fields=K5_FIELDS,
    )
    _assert_numbers_in_prompt(
        prompt, [k, m, c, d0, *xs, *vs, *K5_CAMERA, *K5_LOOK, *K5_LIGHT]
    )
    for index, value in enumerate(xs_after):
        _assert_visible(
            (value, 0.45, 0.0), 0.64, K5_CAMERA, K5_LOOK, f"груз {index}"
        )
    golden = _page(
        K5_BODY,
        title="spring-chain",
        camera=list(K5_CAMERA),
        look=list(K5_LOOK),
        light=list(K5_LIGHT),
        k=k,
        m=m,
        c=c,
        d0=d0,
        x0=_vec(xs),
        v0=_vec(vs),
    )

    task = visual.build_task(
        task_id="K-5",
        title="K-5 — цепочка из трёх грузов на пружинах",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=[
                visual.contains_all(
                    "objects",
                    ["floor", "rail", "cart-0", "cart-1", "cart-2", "spring-01", "spring-12"],
                ),
                visual.near("camera.0", K5_CAMERA[0], 1e-4),
                visual.near("camera.1", K5_CAMERA[1], 1e-4),
                visual.near("camera.2", K5_CAMERA[2], 1e-4),
                visual.size_between("x", 3, 3),
                visual.size_between("v", 3, 3),
            ],
            simulation=[
                visual.near(f"x.{index}", round(value, 4), 1e-3)
                for index, value in enumerate(xs_after)
            ]
            + [
                visual.near(f"v.{index}", round(value, 4), 1e-3)
                for index, value in enumerate(vs_after)
            ]
            + [
                visual.near("t", round(steps * DT, 4), 1e-3),
                visual.near("steps", steps, 0),
            ],
            image=[
                visual.image_min("changed_fraction", 0.2),
                visual.image_min("distinct_colors", 5),
                visual.image_color("warm", 0.0075, "оранжевые грузы различимы"),
                visual.image_centroid("warm", 1, 0.2, 0.8, "грузы видны в кадре"),
                visual.image_region(7, visual.cool, "низ кадра — тёмный пол"),
            ],
            steps=steps,
            dt=DT,
        ),
        golden=golden,
        rng_meta={
            "k": k,
            "m": m,
            "c": c,
            "d0": d0,
            "x0": xs,
            "v0": vs,
            "steps": steps,
            "expected_x": [round(value, 4) for value in xs_after],
            "expected_v": [round(value, 4) for value in vs_after],
        },
    )
    return task


# --------------------------------------------------------------------------- #
# K-6 — torque-free rigid body
# --------------------------------------------------------------------------- #

K6_CAMERA = (0.0, 2.4, 7.5)
K6_LOOK = (0.0, 1.8, 0.0)
K6_LIGHT = (5.0, 9.0, 8.0)
K6_MASS = 1.0

K6_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "body", "..."],
  "camera": [0, 2.4, 7.5],
  "omega": [<w1>, <w2>, <w3>],
  "inertia": [<I1>, <I2>, <I3>]
}"""

K6_PROMPT = """Собери сцену: свободное вращение твёрдого тела вокруг центра масс
(уравнения Эйлера, внешних моментов нет).

Сцена:
* земля — плоскость 30x30 в плоскости y = 0, материал тёмно-синий
  (0x2b3a4a), имя `ground`;
* тело — вытянутый бокс размеров ({a}, {b}, {c}) с центром в точке (0, 1.8, 0),
  материал медно-оранжевый (0xd97a3a), имя `body`; тело обязано
  поворачиваться в соответствии с угловой скоростью (ориентацию веди
  кватернионом);
* на конце длинной оси тела — метка, сфера радиуса 0.14 светлого цвета
  (0x9fd8ff), добавленная дочерним объектом тела в точку ({marker_x}, 0, 0):
  по ней видно, что тело действительно вращается;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.45)` и
  `DirectionalLight(0xffffff, 2.0)` из точки {light}.

Физика:
* главные моменты инерции тела (масса {m}): I1 = {i1}, I2 = {i2}, I3 = {i3};
  оси 1, 2, 3 — оси X, Y, Z тела;
* уравнения Эйлера без моментов:
  `I1*w1' = (I2 - I3)*w2*w3`, `I2*w2' = (I3 - I1)*w3*w1`,
  `I3*w3' = (I1 - I2)*w1*w2`;
* шаг делается только вызовом `window.__step(dt)`;
* схема шага — полунеявный Эйлер, строго в этом порядке (обновлённая
  компонента используется следующими, для которых она уже новая):
  `w1 += (I2 - I3)/I1 * w2 * w3 * dt;`
  `w2 += (I3 - I1)/I2 * w3 * w1 * dt;`
  `w3 += (I1 - I2)/I3 * w1 * w2 * dt;`
* начальное состояние: omega = {omega0};
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `ground` и `body`, лишние
имена не мешают.

Проверяется: состав сцены и камера, вектор угловой скорости после заданного
числа шагов (сверка с той же схемой Эйлера, допуск 1e-3), и кадр — тело видно,
картинка воспроизводима при повторной загрузке.
"""

K6_BODY = """
const I = {inertia};
const W = {omega0};
let elapsed = 0, steps = 0;

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(30, 30),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
ground.rotation.x = -Math.PI / 2;
ground.name = 'ground';
scene.add(ground);

const body = new THREE.Mesh(
  new THREE.BoxGeometry({a}, {b}, {c}),
  new THREE.MeshStandardMaterial({{color: 0xd97a3a}})
);
body.position.set(0, 1.8, 0);
body.name = 'body';
scene.add(body);

// Метка на конце длинной оси: она вращается вместе с телом, поэтому по кадру
// видно, что тело действительно крутится, а не стоит на месте.
const marker = new THREE.Mesh(
  new THREE.SphereGeometry(0.14, 16, 12),
  new THREE.MeshStandardMaterial({{color: 0x9fd8ff}})
);
marker.position.set({a} / 2 + 0.12, 0, 0);
marker.name = 'marker';
body.add(marker);

function draw() {{
  renderer.render(scene, camera);
}}

window.__step = (dt) => {{
  W[0] += (I[1] - I[2]) / I[0] * W[1] * W[2] * dt;
  W[1] += (I[2] - I[0]) / I[1] * W[2] * W[0] * dt;
  W[2] += (I[0] - I[1]) / I[2] * W[0] * W[1] * dt;
  const spin = new THREE.Quaternion(W[0] * dt / 2, W[1] * dt / 2, W[2] * dt / 2, 1).normalize();
  body.quaternion.multiply(spin).normalize();
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
  omega: W.map((value) => round(value)),
  inertia: I,
}});

draw();
window.__ready = true;
"""


def _spin_state(I: list[float], w: list[float], steps: int) -> list[float]:
    """The stated semi-implicit Euler, component by component, in Python."""
    out = list(w)
    for _ in range(steps):
        out[0] += (I[1] - I[2]) / I[0] * out[1] * out[2] * DT
        out[1] += (I[2] - I[0]) / I[1] * out[2] * out[0] * DT
        out[2] += (I[0] - I[1]) / I[2] * out[0] * out[1] * DT
    return out


def _build_k6(rng: random.Random) -> core.TaskInstance:
    a = round(rng.uniform(2.4, 2.9), 2)
    b = round(rng.uniform(1.0, 1.2), 2)
    c = 0.8
    mass = K6_MASS
    inertia = [
        round(mass * (b * b + c * c) / 12, 4),
        round(mass * (a * a + c * c) / 12, 4),
        round(mass * (a * a + b * b) / 12, 4),
    ]
    # The body is a real box, so its projected area depends on where the
    # rotation left it: the thin side must still be a visible object on its own
    # (b and c are close to a, not a plank), otherwise the frame check would
    # measure the orientation rather than the scene.
    assert a > 1.8 * b, "бокс не вытянутый"
    assert inertia[0] < inertia[1] < inertia[2] - 0.02, "моменты инерции не различимы"
    steps = int(rng.choice([90, 120, 150]))

    omega0: list[float] = []
    final: list[float] = []
    for _ in range(40):
        w1 = round(rng.uniform(3.0, 4.5), 2)
        w2 = round(rng.uniform(0.15, 0.45) * rng.choice([-1.0, 1.0]), 2)
        w3 = round(rng.uniform(0.1, 0.4) * rng.choice([-1.0, 1.0]), 2)
        if len({f"{w1:.2f}", f"{w2:.2f}", f"{w3:.2f}"}) < 3:
            continue
        omega0 = [w1, w2, w3]
        final = _spin_state(inertia, omega0, steps)
        if max(abs(f - w) for f, w in zip(final, omega0)) > 0.08:
            break
    else:  # pragma: no cover - the oscillation always moves omega
        raise AssertionError("не удалось подобрать фикстуру, где omega меняется")

    prompt = K6_PROMPT.format(
        a=a,
        b=b,
        c=c,
        cam=list(K6_CAMERA),
        look=list(K6_LOOK),
        light=list(K6_LIGHT),
        m=mass,
        i1=inertia[0],
        i2=inertia[1],
        i3=inertia[2],
        omega0=_vec(omega0),
        marker_x=round(a / 2 + 0.12, 2),
        fields=K6_FIELDS,
    )
    _assert_numbers_in_prompt(
        prompt,
        [a, b, c, mass, round(a / 2 + 0.12, 2), *inertia, *omega0, *K6_CAMERA, *K6_LOOK, *K6_LIGHT],
    )
    _assert_visible(
        (0.0, 1.8, 0.0),
        math.hypot(a / 2 + 0.26, b / 2),
        K6_CAMERA,
        K6_LOOK,
        "тело",
    )
    golden = _page(
        K6_BODY,
        title="rigid-body",
        camera=list(K6_CAMERA),
        look=list(K6_LOOK),
        light=list(K6_LIGHT),
        inertia=_vec(inertia),
        omega0=_vec(omega0),
        a=a,
        b=b,
        c=c,
    )

    task = visual.build_task(
        task_id="K-6",
        title="K-6 — свободное вращение твёрдого тела",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=[
                visual.contains_all("objects", ["ground", "body"]),
                visual.near("camera.0", K6_CAMERA[0], 1e-4),
                visual.near("camera.1", K6_CAMERA[1], 1e-4),
                visual.near("camera.2", K6_CAMERA[2], 1e-4),
                visual.size_between("omega", 3, 3),
                visual.near("inertia.1", inertia[1], 1e-3),
                visual.near("inertia.2", inertia[2], 1e-3),
            ],
            simulation=[
                visual.near(f"omega.{index}", round(value, 4), 1e-3)
                for index, value in enumerate(final)
            ]
            + [
                visual.near("inertia.0", inertia[0], 1e-3),
                visual.near("t", round(steps * DT, 4), 1e-3),
                visual.near("steps", steps, 0),
            ],
            image=[
                visual.image_min("changed_fraction", 0.2),
                visual.image_color("cool", 0.2, "тёмная земля видна в кадре"),
                visual.image_color("warm", 0.015, "медное тело различимо"),
                visual.image_centroid("warm", 1, 0.2, 0.8, "тело видно в кадре"),
                visual.image_region(7, visual.cool, "низ кадра — тёмная земля"),
            ],
            steps=steps,
            dt=DT,
        ),
        golden=golden,
        rng_meta={
            "a": a,
            "b": b,
            "c": c,
            "inertia": inertia,
            "omega0": omega0,
            "steps": steps,
            "expected": [round(value, 4) for value in final],
        },
    )
    return task


# --------------------------------------------------------------------------- #


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Every scenario of this slice, in id order, drawn from the shared stream."""
    return [_build_k2(rng), _build_k3(rng), _build_k4(rng), _build_k5(rng), _build_k6(rng)]
