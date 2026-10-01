"""Visual scenario K-1 — баллистика трёх снарядов (this is the pattern file).

Every visual scenario has the same shape:

* the prompt states the scene, the exact integration scheme and the exact probe
  fields, so the check is a definition rather than a guess;
* ``build()`` draws the fixture numbers from ``rng`` and renders *both* the
  prompt and the reference page from them — the reference is not a picture, it
  is the same program the model is asked to write, with the same constants;
* the expected numbers are recomputed here in Python from the stated scheme, so
  a check never asks "does it look like the reference" but "does the simulation
  obey the physics it was given".

The reference page is a real answer: ``selfcheck`` renders it in the browser and
requires all four checks (html, scene, simulation, image) to pass.
"""

from __future__ import annotations

import math
import random

from .. import core, visual

IDS = ("K-1",)
TITLE = "K-1 — баллистика: три снаряда под гравитацией"

#: Fixed camera and frame size: identical for every model, so two answers are
#: compared on the same picture.
FOV = 50.0
ASPECT = 800 / 600

#: Fixed framing: every fixture keeps the projectiles inside the same frame, so
#: the pictures of two models are comparable and the spheres keep their size.
CAMERA = (0.0, 7.0, 16.0)
LOOK_AT = (0.0, 3.5, 0.0)
LIGHT = (6.0, 12.0, 8.0)


def _visible(points: list[tuple[float, float, float]], camera: tuple[float, float, float],
             look: tuple[float, float, float], margin: float = 0.7) -> bool:
    """Do all three points sit comfortably inside the camera frustum?

    The frame is the evidence for the image check, so a fixture that pushes a
    projectile out of the picture would score a correct simulation as a partial
    one. The frustum is rebuilt here from the same camera the prompt states.
    """
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def unit(a):
        n = math.sqrt(dot(a, a)) or 1.0
        return (a[0] / n, a[1] / n, a[2] / n)

    forward = unit(sub(look, camera))
    right = unit(cross(forward, (0.0, 1.0, 0.0)))
    up = cross(right, forward)
    tan_v = math.tan(math.radians(FOV / 2))
    tan_h = tan_v * ASPECT
    for point in points:
        offset = sub(point, camera)
        depth = dot(offset, forward)
        if depth <= 0:
            return False
        if abs(dot(offset, right)) > tan_h * depth * margin:
            return False
        if abs(dot(offset, up)) > tan_v * depth * margin:
            return False
    return True

PROBE_FIELDS = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "projectile-0", "projectile-1", "projectile-2", "..."],
  "camera": [0, 6, 14],
  "projectiles": [{"x": <x>, "y": <y>, "vy": <vy>}, ...]
}"""

PROMPT = """Собери сцену баллистики: три снаряда летят под гравитацией над
плоскостью земли, камера смотрит на них сбоку.

Сцена:
* земля — плоскость 60x60 в плоскости y=0, материал тёмно-синий (0x2b3a4a), имя объекта `ground`;
* три снаряда — сферы радиуса 0.35, материал оранжевый (0xff8a3d), имена
  `projectile-0`, `projectile-1`, `projectile-2`;
* старты заданы поштучно (x, y, vx, vy):
{starts};
* гравитация — {g} м/с² вниз. Землю снаряды не пробивают и не отскакивают: они
  падают сквозь неё, столкновений в этой задаче нет;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {cam} и смотрит в
  {look}; свет — `AmbientLight(0xffffff, 0.35)` и
  `DirectionalLight(0xffffff, 2.2)` из точки {light}.

Симуляция:
* шаг делается только вызовом `window.__step(dt)`;
* схема шага — полунеявный (симплектический) Эйлер ровно в таком порядке:
  `vy += g_negative * dt; y += vy * dt; x += vx * dt`, где `vx` постоянна;
* снаряды начинают с y = 0.25 и летят вниз, пока y не станет меньше -40 — после
  этого считай их остановившимися на y = -40 (дальше не двигай);
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

`objects` — имена объектов сцены; проверяется наличие `ground` и всех трёх
`projectile-*`, лишние имена (например следы траекторий) не мешают.

Проверяется: состав сцены и камера, координаты всех трёх снарядов после
заданного числа шагов (сверка с той же схемой Эйлера, допуск 1e-3), и кадр —
снаряды видны и различимы, картинка воспроизводима при повторной загрузке.
"""

GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>ballistics</title>
<style>html,body{{margin:0;background:#101318;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const G = {g};
const RUNNERS = {runners};

const renderer = new THREE.WebGLRenderer({{antialias: false, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 200);
camera.position.set(...{camera});
camera.lookAt(...{look});

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(...{light});
scene.add(key);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(60, 60),
  new THREE.MeshStandardMaterial({{color: 0x2b3a4a}})
);
ground.rotation.x = -Math.PI / 2;
ground.name = 'ground';
scene.add(ground);

const TRAIL = 80;

const bodies = RUNNERS.map((spec, index) => {{
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.35, 24, 16),
    new THREE.MeshStandardMaterial({{color: 0xff8a3d}})
  );
  mesh.name = `projectile-${{index}}`;
  mesh.position.set(spec.x, spec.y, 0);
  scene.add(mesh);

  // След траектории: делает движение читаемым на кадре и остаётся
  // детерминированным — он зависит только от шагов симуляции.
  const trail = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(Array.from({{length: TRAIL}}, () => new THREE.Vector3(0, -100, 0))),
    new THREE.LineBasicMaterial({{color: 0xffb35d}})
  );
  trail.name = `trail-${{index}}`;
  scene.add(trail);

  const body = {{mesh, trail, points: [], vx: spec.vx, vy: spec.vy, stopped: false}};
  pushTrail(body);
  return body;
}});

function pushTrail(body) {{
  body.points.push(body.mesh.position.clone());
  if (body.points.length > TRAIL) body.points.shift();
  const positions = body.trail.geometry.attributes.position;
  for (let i = 0; i < TRAIL; i++) {{
    const point = body.points[Math.max(0, body.points.length - TRAIL + i)] || body.mesh.position;
    positions.setXYZ(i, point.x, point.y, point.z);
  }}
  positions.needsUpdate = true;
  body.trail.geometry.setDrawRange(Math.max(0, TRAIL - body.points.length), body.points.length);
  body.trail.geometry.computeBoundingSphere();
}}

let elapsed = 0;
let steps = 0;

function frame() {{
  renderer.render(scene, camera);
}}

window.__step = (dt) => {{
  for (const body of bodies) {{
    if (body.stopped) continue;
    body.vy -= G * dt;
    body.mesh.position.y += body.vy * dt;
    body.mesh.position.x += body.vx * dt;
    if (body.mesh.position.y < -40) {{ body.mesh.position.y = -40; body.stopped = true; }}
    pushTrail(body);
  }}
  elapsed += dt;
  steps += 1;
  frame();
  return window.__probe();
}};

const round = (value) => Math.round(value * 1e4) / 1e4;

window.__probe = () => ({{
  steps,
  t: round(elapsed),
  objects: scene.children.map((o) => o.name).filter(Boolean),
  camera: [camera.position.x, camera.position.y, camera.position.z],
  projectiles: bodies.map((body) => ({{
    x: round(body.mesh.position.x),
    y: round(body.mesh.position.y),
    vy: round(body.vy),
  }})),
}});

frame();
window.__ready = true;
</script>
</body></html>
"""


def _positions(spec: list[dict[str, float]], gravity: float, steps: int, dt: float) -> list[tuple[float, float, float]]:
    """Semi-implicit Euler, recomputed independently of the reference page."""
    out: list[tuple[float, float, float]] = []
    for body in spec:
        x, y, vy = body["x0"], body["y0"], body["vy"]
        for _ in range(steps):
            if y < -40:
                y = -40.0
                vy = 0.0
                continue
            vy -= gravity * dt
            y += vy * dt
            x += body["vx"] * dt
            if y < -40:
                y = -40.0
                vy = 0.0
        out.append((round(x, 4), round(y, 4), round(vy, 4)))
    return out


def _starts(specs: list[dict[str, float]]) -> str:
    return "\n".join(
        f"  `projectile-{index}`: x={spec['x0']}, y={spec['y0']}, vx={spec['vx']}, vy={spec['vy']}"
        for index, spec in enumerate(specs)
    )


def build(rng: random.Random) -> list[core.TaskInstance]:
    dt = 1 / 60
    gravity = float(rng.choice([9.81, 3.71, 1.62]))
    specs: list[dict[str, float]] = []
    expected: list[tuple[float, float, float]] = []
    camera: tuple[float, float, float] = CAMERA
    look: tuple[float, float, float] = LOOK_AT
    steps = 0

    # Draw launches that are distinct by construction and keep all three
    # projectiles inside the frame. The check compares three positions, and a
    # fixture where one projectile hides another would prove nothing.
    for _ in range(40):
        vx0 = round(rng.uniform(0.6, 1.2), 2)
        vx_step = round(rng.uniform(0.2, 0.4), 2)
        # vy is drawn around the launch speed whose apex lands mid-frame for
        # this gravity: a low-gravity fixture must not throw everything out of
        # the picture, and a high-gravity one must not hug the ground.
        vy_floor = math.sqrt(2 * gravity * 2.75)
        vy0 = round(vy_floor + rng.uniform(0.0, 0.8), 2)
        vy_step = round(rng.uniform(0.3, 0.6), 2)
        specs = [
            {
                "x0": float(-4 + 4 * index),
                "y0": 0.25,
                "vx": round(vx0 + index * vx_step, 2),
                "vy": round(vy0 + index * vy_step, 2),
            }
            for index in range(3)
        ]
        steps = int(rng.choice([45, 60, 75]))
        expected = _positions(specs, gravity, steps, dt)
        if all(y > 1.0 for _, y, _ in expected) and _visible(expected, CAMERA, LOOK_AT):
            break
    else:  # pragma: no cover - the loop above is wide enough for every seed
        raise AssertionError("не удалось подобрать траектории, попадающие в кадр")

    camera, look = CAMERA, LOOK_AT
    assert len({spec["vx"] for spec in specs}) == 3
    assert len({spec["vy"] for spec in specs}) == 3
    assert all(y > 1.0 for _, y, _ in expected), "к моменту проверки снаряды должны быть в воздухе"
    assert _visible(expected, camera, look), "все три снаряда обязаны быть в кадре"
    for spec in specs:
        assert f"vx={spec['vx']}, vy={spec['vy']}" in _starts(specs)

    light = LIGHT
    prompt = PROMPT.format(
        starts=_starts(specs),
        g=f"{gravity:.2f}",
        cam=list(camera),
        look=list(look),
        light=list(light),
        fields=PROBE_FIELDS,
    )

    runners = ", ".join(
        "{x: %.2f, y: %.2f, vx: %.2f, vy: %.2f}" % (spec["x0"], spec["y0"], spec["vx"], spec["vy"])
        for spec in specs
    )
    golden = GOLDEN.format(
        g=gravity,
        runners="[" + runners + "]",
        camera=list(camera),
        look=list(look),
        light=list(light),
    )

    scene_checks = [
        visual.contains_all("objects", ["ground", "projectile-0", "projectile-1", "projectile-2"]),
        visual.near("camera.0", camera[0], 1e-4),
        visual.near("camera.1", camera[1], 1e-4),
        visual.near("camera.2", camera[2], 1e-4),
        visual.size_between("projectiles", 3, 3),
    ]
    sim_checks = [
        visual.near(f"projectiles.{index}.x", x, 1e-3)
        for index, (x, _, _) in enumerate(expected)
    ] + [
        visual.near(f"projectiles.{index}.y", y, 1e-3)
        for index, (_, y, _) in enumerate(expected)
    ] + [
        visual.near("t", round(steps * dt, 4), 1e-3),
        visual.near("steps", steps, 0),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.15),
        visual.image_min("distinct_colors", 8),
        visual.image_color("warm", 0.0015, "оранжевые снаряды различимы"),
        visual.image_centroid("warm", 1, 0.0, 0.8, "снаряды видны в кадре, а не у его края"),
        visual.image_region(7, visual.cool, "низ кадра — тёмная земля"),
    ]

    task = visual.build_task(
        task_id="K-1",
        title=TITLE,
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks,
            simulation=sim_checks,
            image=image_checks,
            steps=steps,
            dt=dt,
        ),
        golden=golden,
        rng_meta={
            "gravity": gravity,
            "steps": steps,
            "dt": dt,
            "camera": list(camera),
            "look": list(look),
            "launches": specs,
            "expected": expected,
        },
    )
    return [task]
