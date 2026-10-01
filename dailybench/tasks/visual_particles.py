"""Visual scenarios K-7..K-11 — particles, agents and flows.

Five scenarios, all built the same way as ``visual_projectiles``: the prompt
states the scene, the exact integration scheme, the exact sequence of random
numbers and the exact probe fields; ``build()`` renders both the prompt and the
reference page from the same fixture numbers, and the expected values are
recomputed here in Python from that same scheme, so the check is a definition,
not a comparison with a picture.

Randomness — both in the fixture and in the page — is a single deterministic
LCG with constants fixed in the prompt:

    s = (1664525 * s + 1013904223) % 4294967296,   u = s / 4294967296

so the page reproduces the same stream as Python.  No `Math.random()`, no
clock, no `requestAnimationFrame` for physics.
"""

from __future__ import annotations

import math
import random

from .. import core, visual

IDS = ("K-7", "K-8", "K-9", "K-10", "K-11")

#: The one random generator every scenario in this slice uses, spelled out in
#: each prompt so the page and Python consume the identical stream.
LCG_TEXT = (
    "`s` — целое состояние, старт `S0 = {seed}`; шаг: "
    "`s = (1664525 * s + 1013904223) % 4294967296`; число `u = s / 4294967296`. "
    "Порядок обращений задан ниже — повторяй его ровно."
)


def _lcg(seed: int):
    """The exact generator the prompts spell out; yields ``u`` in ``[0, 1)``."""
    state = seed
    while True:
        state = (1664525 * state + 1013904223) % 4294967296
        yield state / 4294967296.0


FOV_DEG = 50.0
ASPECT = 800 / 600
FRAME_MARGIN = 0.7


def _visible(point: tuple[float, float, float], camera: tuple[float, ...],
             look: tuple[float, ...], margin: float = FRAME_MARGIN) -> bool:
    """Is ``point`` inside the camera frustum, with room to spare?

    Rebuilt from ``position``/``lookAt`` rather than guessed: the point must sit
    within ``margin`` of the half-extents at its depth, so a whole object cannot
    hang off the edge of the frame.
    """
    fx, fy, fz = (look[0] - camera[0], look[1] - camera[1], look[2] - camera[2])
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / norm, fy / norm, fz / norm
    # right = forward x up(0,1,0); up' = right x forward
    rx, ry, rz = -fz * 1.0, 0.0, fx * 1.0  # (f x up) with up=(0,1,0) -> (-fz, 0, fx)
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rn < 1e-12:  # camera looking straight up or down: fall back to +x as right
        rx, ry, rz, rn = 1.0, 0.0, 0.0, 1.0
    rx, ry, rz = rx / rn, ry / rn, rz / rn
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx  # right x forward
    vx, vy, vz = point[0] - camera[0], point[1] - camera[1], point[2] - camera[2]
    depth = vx * fx + vy * fy + vz * fz
    if depth <= 0.1:
        return False
    x_cam = vx * rx + vy * ry + vz * rz
    y_cam = vx * ux + vy * uy + vz * uz
    half_h = math.tan(math.radians(FOV_DEG) / 2) * depth
    half_w = half_h * ASPECT
    return abs(x_cam) <= margin * half_w and abs(y_cam) <= margin * half_h


def _assert_in_frame(points, camera, look, label: str) -> None:
    outside = [p for p in points if not _visible((p[0], p[1], 0.0), camera, look)]
    assert not outside, f"{label}: {len(outside)} объектов вне кадра, например {outside[:2]}"


def _assert_in_prompt(prompt: str, values: list[str]) -> None:
    missing = [value for value in values if value not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


def _plain_checks(camera: tuple[float, ...], objects: list[str]) -> list[visual.Predicate]:
    """Scene checks every scenario shares: the named objects and the camera."""
    return [
        visual.contains_all("objects", objects),
        visual.near("camera.0", camera[0], 1e-4),
        visual.near("camera.1", camera[1], 1e-4),
        visual.near("camera.2", camera[2], 1e-4),
    ]


# --------------------------------------------------------------------------- #
# K-7 — фонтан частиц
# --------------------------------------------------------------------------- #

K7_CAMERA = (0.0, 4.0, 14.0)
K7_LOOK = (0.0, 3.0, 0.0)
K7_LIGHT = (6.0, 12.0, 8.0)

K7_FIELDS = """{
  "steps": <число сделанных шагов>,
  "alive": <сколько частиц ещё живо>,
  "mean_height": <средняя высота живых частиц, 4 знака>,
  "particles": [{"x": <x>, "y": <y>}, {"x": <x>, "y": <y>}, {"x": <x>, "y": <y>}],
  "camera": [<камера сцены, 3 числа>],
  "objects": ["ground", "particles", "..."]
}"""

K7_PROMPT = """Собери сцену «фонтан частиц»: из одной точки кверху вылетают
частицы, камера смотрит на фонтан сбоку.

Сцена:
* земля — плоскость 80x80 в плоскости y=0, материал тёмно-синий (0x2b3a4a),
  имя объекта `ground`;
* частицы — ровно одна `THREE.InstancedMesh` из {n} сфер радиуса 0.25,
  материал оранжевый (0xff8a3d), имя `particles`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {camera} и смотрит
  в {look}; свет — `AmbientLight(0xffffff, 0.45)` и
  `DirectionalLight(0xffffff, 2.0)` из точки {light}.

Начальное состояние — все частицы стартуют из точки (0, 0.25) в плоскости
z=0, скорости берутся из ЛЦГ:
* {lcg}
* для частицы i = 0..{n_minus}, по порядку: `vx[i] = -3 + 6*u`, затем
  `vy[i] = 5 + 4*u`.

Симуляция (`window.__step(dt)` — один шаг), для живых частиц по индексу:
* `vy -= g*dt`; `y += vy*dt`; `x += vx*dt` (полунеявный Эйлер, g = {g});
* если после шага `y < 0`, частица мертва: `y = 0`, дальше она не двигается;
* `alive` — число живых частиц, `mean_height` — средняя y живых (0, если живых
  нет).

`window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: состав сцены и камера, число живых, средняя высота и координаты
первых трёх частиц после заданного числа шагов (сверка с той же схемой, допуск
1e-3), и кадр — частицы видны, картинка воспроизводима при повторной загрузке.
"""

K7_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>fountain</title>
<style>html,body{{margin:0;background:#101318;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = {n};
const G = {g};
const SEED = {seed};

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

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

const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({{color: 0x2b3a4a}}));
ground.rotation.x = -Math.PI / 2;
ground.name = 'ground';
scene.add(ground);

const px = new Array(N).fill(0), py = new Array(N).fill(0.25);
const vx = new Array(N), vy = new Array(N), alive = new Array(N).fill(true);
for (let i = 0; i < N; i++) {{ vx[i] = -3 + 6 * next(); vy[i] = 5 + 4 * next(); }}

const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.25, 12, 8), new THREE.MeshStandardMaterial({{color: 0xff8a3d}}), N);
mesh.name = 'particles';
mesh.frustumCulled = false;
scene.add(mesh);
const dummy = new THREE.Object3D();
const round = (v) => Math.round(v * 1e4) / 1e4;
const sync = () => {{
  for (let i = 0; i < N; i++) {{ dummy.position.set(px[i], py[i], 0); dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix); }}
  mesh.instanceMatrix.needsUpdate = true;
}};

let steps = 0;
window.__step = (dt) => {{
  for (let i = 0; i < N; i++) {{
    if (!alive[i]) continue;
    vy[i] -= G * dt;
    py[i] += vy[i] * dt;
    px[i] += vx[i] * dt;
    if (py[i] < 0) {{ py[i] = 0; alive[i] = false; }}
  }}
  steps += 1;
  sync();
  renderer.render(scene, camera);
  return window.__probe();
}};
window.__probe = () => {{
  let count = 0, sum = 0;
  for (let i = 0; i < N; i++) if (alive[i]) {{ count += 1; sum += py[i]; }}
  return {{
    camera: [camera.position.x, camera.position.y, camera.position.z],
    steps,
    alive: count,
    mean_height: count ? round(sum / count) : 0,
    particles: [0, 1, 2].map((i) => ({{ x: round(px[i]), y: round(py[i]) }})),
    objects: scene.children.map((o) => o.name).filter(Boolean),
  }};
}};

sync();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _fountain(n: int, g: float, seed: int, steps: int, dt: float):
    """The K-7 page program in Python; returns ``(alive, mean_height, first3, positions)``."""
    gen = _lcg(seed)
    vx = []
    vy = []
    for _ in range(n):
        vx.append(-3.0 + 6.0 * next(gen))
        vy.append(5.0 + 4.0 * next(gen))
    px = [0.0] * n
    py = [0.25] * n
    alive = [True] * n
    for _ in range(steps):
        for i in range(n):
            if not alive[i]:
                continue
            vy[i] -= g * dt
            py[i] += vy[i] * dt
            px[i] += vx[i] * dt
            if py[i] < 0.0:
                py[i] = 0.0
                alive[i] = False
    count = sum(1 for a in alive if a)
    mean = sum(py[i] for i in range(n) if alive[i]) / count if count else 0.0
    return count, mean, [(px[i], py[i]) for i in range(3)], list(zip(px, py))


def _build_k7(rng: random.Random) -> core.TaskInstance:
    n = int(rng.choice([24, 32, 40]))
    g = float(rng.choice([9.0, 12.0, 15.0]))
    seed = int(rng.randrange(1, 2**32))
    dt = 1 / 60
    # Pick the sampled moment deterministically: the golden must show both the
    # living and the fallen, and every particle must sit inside the frame.
    for steps in (45, 60, 75, 90):
        alive, mean, first3, points = _fountain(n, g, seed, steps, dt)
        if 0 < alive < n and mean > 0.05 and all(
                _visible((x, y, 0.0), K7_CAMERA, K7_LOOK) for x, y in points):
            break
    else:  # pragma: no cover - the sweep above covers a wide margin
        raise AssertionError("K-7: не удалось подобрать число шагов")

    prompt = K7_PROMPT.format(
        n=n, n_minus=n - 1, g=f"{g:.2f}", camera=list(K7_CAMERA), look=list(K7_LOOK),
        light=list(K7_LIGHT), fields=K7_FIELDS, lcg=LCG_TEXT.format(seed=seed),
    )
    # Fixture proves itself: the launches are distinct, the fountain is airborne
    # and the height is nowhere near the "no gravity" straight-line answer.
    assert len({(round(x, 4), round(y, 4)) for x, y in first3}) == 3, \
        "первые три частицы должны быть различимы"
    naive = 0.25 + 7.0 * steps * dt  # mean launch speed 7 m/s, no gravity
    assert abs(mean - naive) > 0.5, "гравитация обязана изменить среднюю высоту"
    _assert_in_frame(points, K7_CAMERA, K7_LOOK, "K-7")
    _assert_in_prompt(prompt, [str(n), f"{g:.2f}", f"S0 = {seed}"])

    golden = K7_GOLDEN.format(
        n=n, g=g, seed=seed, camera=list(K7_CAMERA), look=list(K7_LOOK), light=list(K7_LIGHT),
    )
    scene = _plain_checks(K7_CAMERA, ["ground", "particles"])
    simulation = [
        visual.near("steps", steps, 0),
        visual.near("alive", alive, 0),
        visual.near("mean_height", mean, 1e-3),
    ] + [visual.near(f"particles.{i}.{axis}", value, 1e-3)
         for i, point in enumerate(first3) for axis, value in zip(("x", "y"), point)]
    image = [
        visual.image_min("changed_fraction", 0.2),
        visual.image_min("distinct_colors", 9),
        visual.image_color("warm", 0.006, "частицы фонтана различимы"),
        visual.image_centroid("warm", 1, 0.05, 0.95, "частицы видны в кадре"),
        visual.image_region(8, visual.cool, "низ кадра — тёмная земля"),
    ]
    return visual.build_task(
        task_id="K-7", title="K-7 — фонтан частиц", prompt=prompt,
        spec=visual.VisualSpec(scene=scene, simulation=simulation, image=image, steps=steps, dt=dt),
        golden=golden,
        rng_meta={"n": n, "g": g, "seed": seed, "steps": steps, "dt": dt,
                  "alive": alive, "mean_height": mean, "first3": first3},
    )


# --------------------------------------------------------------------------- #
# K-8 — броуновское движение
# --------------------------------------------------------------------------- #

K8_CAMERA = (0.0, 0.0, 16.0)
K8_LOOK = (0.0, 0.0, 0.0)
K8_LIGHT = (4.0, 6.0, 10.0)
K8_BOX_X = 6.0
K8_BOX_Y = 4.0

K8_FIELDS = """{
  "steps": <число сделанных шагов>,
  "particles": [{"x": <x>, "y": <y>}, ... по одной записи на частицу ...],
  "camera": [<камера сцены, 3 числа>],
  "objects": ["backdrop", "particles", "..."]
}"""

K8_PROMPT = """Собери сцену броуновского движения: частицы случайно блуждают в
плоскости, камера смотрит на них сверху (сцена плоская, z = 0).

Сцена:
* фон — плоскость 40x40 в плоскости z = -2, материал тёмно-синий (0x22303f),
  имя объекта `backdrop`;
* частицы — ровно одна `THREE.InstancedMesh` из {m} сфер радиуса 0.3,
  материал оранжевый (0xffa64d), имя `particles`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {camera} и смотрит
  в {look}; свет — `AmbientLight(0xffffff, 0.5)` и
  `DirectionalLight(0xffffff, 1.8)` из точки {light}.

Начальные позиции (формула, без ЛЦГ): `x[i] = -3 + 2*(i % 4)`,
`y[i] = 2 - 1.5*floor(i / 4)`.

Симуляция (`window.__step(dt)` — один шаг; на движение dt не влияет):
* {lcg}
* для частицы i = 0..{m_minus} по порядку: сначала `x[i] += (2*u - 1)*h`,
  затем `y[i] += (2*u - 1)*h`, где h = {h} — размер шага;
* после этого координаты частицы ограничиваются прямоугольником
  [-{bx}, {bx}] x [-{by}, {by}] (clamp без отскока).

`window.__probe()` возвращает ровно:

{fields}

Проверяется: состав сцены и камера, координаты всех частиц после заданного
числа шагов (сверка с той же схемой и тем же потоком ЛЦГ, допуск 1e-3), и
кадр — частицы видны и различимы, картинка воспроизводима.
"""

K8_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>brownian</title>
<style>html,body{{margin:0;background:#101318;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const M = {m};
const H = {h};
const BX = {bx};
const BY = {by};
const SEED = {seed};

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

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
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(...{light});
scene.add(key);

const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({{color: 0x22303f}}));
backdrop.position.z = -2;
backdrop.name = 'backdrop';
scene.add(backdrop);

const px = new Array(M), py = new Array(M);
for (let i = 0; i < M; i++) {{ px[i] = -3 + 2 * (i % 4); py[i] = 2 - 1.5 * Math.floor(i / 4); }}

const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.3, 12, 8), new THREE.MeshStandardMaterial({{color: 0xffa64d}}), M);
mesh.name = 'particles';
mesh.frustumCulled = false;
scene.add(mesh);
const dummy = new THREE.Object3D();
const round = (v) => Math.round(v * 1e4) / 1e4;
const sync = () => {{
  for (let i = 0; i < M; i++) {{ dummy.position.set(px[i], py[i], 0); dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix); }}
  mesh.instanceMatrix.needsUpdate = true;
}};

let steps = 0;
window.__step = (dt) => {{
  for (let i = 0; i < M; i++) {{
    px[i] += (2 * next() - 1) * H;
    py[i] += (2 * next() - 1) * H;
    px[i] = Math.min(BX, Math.max(-BX, px[i]));
    py[i] = Math.min(BY, Math.max(-BY, py[i]));
  }}
  steps += 1;
  sync();
  renderer.render(scene, camera);
  return window.__probe();
}};
window.__probe = () => ({{
    camera: [camera.position.x, camera.position.y, camera.position.z],
  steps,
  particles: px.map((_, i) => ({{ x: round(px[i]), y: round(py[i]) }})),
  objects: scene.children.map((o) => o.name).filter(Boolean),
}});

sync();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _brownian(m: int, h: float, seed: int, steps: int):
    """The K-8 page program in Python; returns the particle positions."""
    gen = _lcg(seed)
    px = [-3.0 + 2.0 * (i % 4) for i in range(m)]
    py = [2.0 - 1.5 * (i // 4) for i in range(m)]
    for _ in range(steps):
        for i in range(m):
            px[i] += (2.0 * next(gen) - 1.0) * h
            py[i] += (2.0 * next(gen) - 1.0) * h
            px[i] = min(K8_BOX_X, max(-K8_BOX_X, px[i]))
            py[i] = min(K8_BOX_Y, max(-K8_BOX_Y, py[i]))
    return px, py


def _build_k8(rng: random.Random) -> core.TaskInstance:
    m = int(rng.choice([8, 12, 16]))
    h = float(rng.choice([0.2, 0.3, 0.4]))
    seed = int(rng.randrange(1, 2**32))
    steps = int(rng.choice([20, 30, 40]))
    dt = 1 / 60
    px, py = _brownian(m, h, seed, steps)

    prompt = K8_PROMPT.format(
        m=m, m_minus=m - 1, h=h, bx=K8_BOX_X, by=K8_BOX_Y, camera=list(K8_CAMERA),
        look=list(K8_LOOK), light=list(K8_LIGHT), fields=K8_FIELDS, lcg=LCG_TEXT.format(seed=seed),
    )
    assert len({(round(x, 6), round(y, 6)) for x, y in zip(px, py)}) == m, "частицы должны остаться различимыми"
    assert any(abs(px[i] - (-3.0 + 2.0 * (i % 4))) > 1e-6 for i in range(m)), "ЛЦГ обязана сдвинуть частицы"
    assert all(-K8_BOX_X - 1e-9 <= x <= K8_BOX_X + 1e-9 and -K8_BOX_Y - 1e-9 <= y <= K8_BOX_Y + 1e-9
               for x, y in zip(px, py)), "эталон обязан остаться в рамке блуждания"
    _assert_in_frame(list(zip(px, py)), K8_CAMERA, K8_LOOK, "K-8")
    _assert_in_prompt(prompt, [str(m), str(h), str(K8_BOX_X), str(K8_BOX_Y), f"S0 = {seed}"])

    golden = K8_GOLDEN.format(
        m=m, h=h, bx=K8_BOX_X, by=K8_BOX_Y, seed=seed,
        camera=list(K8_CAMERA), look=list(K8_LOOK), light=list(K8_LIGHT),
    )
    scene = _plain_checks(K8_CAMERA, ["backdrop", "particles"])
    simulation = [visual.near("steps", steps, 0)] + [
        visual.near(f"particles.{i}.{axis}", value, 1e-3)
        for i in range(m) for axis, value in zip(("x", "y"), (px[i], py[i]))
    ]
    image = [
        visual.image_min("changed_fraction", 0.0035),
        visual.image_min("distinct_colors", 8),
        visual.image_color("warm", 0.0035, "блуждающие частицы различимы"),
        visual.image_centroid("warm", 0, 0.2, 0.8, "частицы держатся центра кадра"),
        visual.image_region(0, visual.cool, "верх кадра — тёмный фон"),
    ]
    return visual.build_task(
        task_id="K-8", title="K-8 — броуновское движение", prompt=prompt,
        spec=visual.VisualSpec(scene=scene, simulation=simulation, image=image, steps=steps, dt=dt),
        golden=golden,
        rng_meta={"m": m, "h": h, "seed": seed, "steps": steps, "dt": dt, "positions": list(zip(px, py))},
    )


# --------------------------------------------------------------------------- #
# K-9 — стая (boids)
# --------------------------------------------------------------------------- #

K9_CAMERA = (0.0, 0.0, 18.0)
K9_LOOK = (0.0, 0.0, 0.0)
K9_LIGHT = (5.0, 5.0, 10.0)
K9_N = 12

K9_FIELDS = """{
  "steps": <число сделанных шагов>,
  "agents": [{"x": <x>, "y": <y>, "vx": <vx>, "vy": <vy>}, ... 12 записей ...],
  "camera": [<камера сцены, 3 числа>],
  "objects": ["backdrop", "agent-0", "...", "agent-11"]
}"""

K9_PROMPT = """Собери стаю из 12 агентов (boids) — плоская 2D-сцена в плоскости
z = 0, камера смотрит на неё сверху.

Сцена:
* фон — плоскость 40x40 в плоскости z = -2, материал тёмно-синий (0x22303f),
  имя объекта `backdrop`;
* агенты — 12 сфер радиуса 0.3, материал оранжевый (0xffb03a), имена
  `agent-0` ... `agent-11`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {camera} и смотрит
  в {look}; свет — `AmbientLight(0xffffff, 0.5)` и
  `DirectionalLight(0xffffff, 1.8)` из точки {light}.

Начальное состояние:
* {lcg}
* для агента i = 0..11 по порядку: `x = -4 + 8*u`, `y = -4 + 8*u`,
  `vx = -1.5 + 3*u`, `vy = -1.5 + 3*u` (четыре обращения к ЛЦГ подряд).

Симуляция (`window.__step(dt)` — один шаг). Сначала по состоянию НА НАЧАЛЕ
шага считаются ускорения всех агентов (обновление одновременное, обход строго
по индексу i = 0..11), затем состояние обновляется:
* центр масс остальных `com_i = (Σ_{{j≠i}} p_j) / 11` и средняя скорость
  остальных `avg_i = (Σ_{{j≠i}} v_j) / 11`;
* отталкивание `sep_i = Σ_{{j≠i, 0 < |p_i - p_j| < r_sep}} (p_i - p_j) / |p_i - p_j|`;
* ускорение `a_i = w_c*(com_i - p_i) + w_s*sep_i + w_a*(avg_i - v_i)`;
* `v_i += a_i*dt`; если `|v_i| > max_speed`, скорость нормируется до `max_speed`;
* `p_i += v_i*dt` (уже новой скоростью).
* константы: w_c = {wc}, w_s = {ws}, w_a = {wa}, r_sep = {rsep}, max_speed = {vmax}.

`window.__probe()` возвращает ровно:

{fields}

Проверяется: состав сцены и камера, позиции и скорости всех 12 агентов после
заданного числа шагов (сверка с той же схемой, допуск 1e-3), и кадр — агенты
видны и различимы, картинка воспроизводима.
"""

K9_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>boids</title>
<style>html,body{{margin:0;background:#101318;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = 12;
const WC = {wc}, WS = {ws}, WA = {wa}, R_SEP = {rsep}, MAX_SPEED = {vmax};
const SEED = {seed};

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

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
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(...{light});
scene.add(key);

const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({{color: 0x22303f}}));
backdrop.position.z = -2;
backdrop.name = 'backdrop';
scene.add(backdrop);

const P = [], V = [], agents = [];
for (let i = 0; i < N; i++) {{
  P.push([-4 + 8 * next(), -4 + 8 * next()]);
  V.push([-1.5 + 3 * next(), -1.5 + 3 * next()]);
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.3, 12, 8), new THREE.MeshStandardMaterial({{color: 0xffb03a}}));
  mesh.name = `agent-${{i}}`;
  mesh.position.set(P[i][0], P[i][1], 0);
  scene.add(mesh);
  agents.push(mesh);
}}

const round = (v) => Math.round(v * 1e4) / 1e4;
let steps = 0;
window.__step = (dt) => {{
  const acc = [];
  for (let i = 0; i < N; i++) {{
    let cx = 0, cy = 0, ax = 0, ay = 0, sx = 0, sy = 0;
    for (let j = 0; j < N; j++) {{
      if (j === i) continue;
      cx += P[j][0]; cy += P[j][1]; ax += V[j][0]; ay += V[j][1];
      const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], d = Math.hypot(dx, dy);
      if (d > 0 && d < R_SEP) {{ sx += dx / d; sy += dy / d; }}
    }}
    const k = N - 1;
    acc.push([WC * (cx / k - P[i][0]) + WS * sx + WA * (ax / k - V[i][0]),
              WC * (cy / k - P[i][1]) + WS * sy + WA * (ay / k - V[i][1])]);
  }}
  for (let i = 0; i < N; i++) {{
    let nvx = V[i][0] + acc[i][0] * dt, nvy = V[i][1] + acc[i][1] * dt;
    const sp = Math.hypot(nvx, nvy);
    if (sp > MAX_SPEED) {{ nvx = nvx / sp * MAX_SPEED; nvy = nvy / sp * MAX_SPEED; }}
    V[i][0] = nvx; V[i][1] = nvy;
    P[i][0] += nvx * dt; P[i][1] += nvy * dt;
    agents[i].position.set(P[i][0], P[i][1], 0);
  }}
  steps += 1;
  renderer.render(scene, camera);
  return window.__probe();
}};
window.__probe = () => ({{
    camera: [camera.position.x, camera.position.y, camera.position.z],
  steps,
  agents: P.map((_, i) => ({{ x: round(P[i][0]), y: round(P[i][1]), vx: round(V[i][0]), vy: round(V[i][1]) }})),
  objects: scene.children.map((o) => o.name).filter(Boolean),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _boids_init(seed: int, n: int):
    gen = _lcg(seed)
    pos: list[list[float]] = []
    vel: list[list[float]] = []
    for _ in range(n):
        pos.append([-4.0 + 8.0 * next(gen), -4.0 + 8.0 * next(gen)])
        vel.append([-1.5 + 3.0 * next(gen), -1.5 + 3.0 * next(gen)])
    return pos, vel


def _boids(pos, vel, r_sep: float, w_c: float, w_s: float, w_a: float,
           max_speed: float, steps: int, dt: float):
    """The K-9 page program in Python; returns the final positions/velocities."""
    n = len(pos)
    p = [list(pt) for pt in pos]
    v = [list(w) for w in vel]
    for _ in range(steps):
        acc: list[tuple[float, float]] = []
        for i in range(n):
            cx = cy = ax = ay = sx = sy = 0.0
            for j in range(n):
                if j == i:
                    continue
                cx += p[j][0]
                cy += p[j][1]
                ax += v[j][0]
                ay += v[j][1]
                dx = p[i][0] - p[j][0]
                dy = p[i][1] - p[j][1]
                d = math.hypot(dx, dy)
                if 0.0 < d < r_sep:
                    sx += dx / d
                    sy += dy / d
            k = n - 1
            acc.append((
                w_c * (cx / k - p[i][0]) + w_s * sx + w_a * (ax / k - v[i][0]),
                w_c * (cy / k - p[i][1]) + w_s * sy + w_a * (ay / k - v[i][1]),
            ))
        for i in range(n):
            nvx = v[i][0] + acc[i][0] * dt
            nvy = v[i][1] + acc[i][1] * dt
            sp = math.hypot(nvx, nvy)
            if sp > max_speed:
                nvx = nvx / sp * max_speed
                nvy = nvy / sp * max_speed
            v[i][0], v[i][1] = nvx, nvy
            p[i][0] += nvx * dt
            p[i][1] += nvy * dt
    return p, v


def _build_k9(rng: random.Random) -> core.TaskInstance:
    seed = int(rng.randrange(1, 2**32))
    steps = int(rng.choice([24, 36, 45]))
    dt = 1 / 60
    w_c, w_s, w_a = 0.4, 1.0, 0.3
    r_sep = float(rng.choice([2.0, 2.5, 3.0]))
    max_speed = float(rng.choice([2.0, 2.5, 3.0]))
    pos, vel = _boids_init(seed, K9_N)
    fp, fv = _boids(pos, vel, r_sep, w_c, w_s, w_a, max_speed, steps, dt)

    prompt = K9_PROMPT.format(
        camera=list(K9_CAMERA), look=list(K9_LOOK), light=list(K9_LIGHT), lcg=LCG_TEXT.format(seed=seed),
        wc=w_c, ws=w_s, wa=w_a, rsep=r_sep, vmax=max_speed, fields=K9_FIELDS,
    )
    assert len({(round(a[0], 6), round(a[1], 6)) for a in pos}) == K9_N, "стартовые позиции должны быть различимы"
    assert len({(round(x, 6), round(y, 6)) for x, y in fp}) == K9_N, "после шагов агенты должны остаться различимыми"
    assert any(abs(fp[i][0] - pos[i][0]) + abs(fp[i][1] - pos[i][1]) > 0.05 for i in range(K9_N)), \
        "правила стаи обязаны сдвинуть агентов"
    # The naive "agents are static" answer is nowhere near the simulated one.
    assert max(math.hypot(fp[i][0] - pos[i][0], fp[i][1] - pos[i][1]) for i in range(K9_N)) > 0.2
    _assert_in_frame(fp, K9_CAMERA, K9_LOOK, "K-9")
    _assert_in_prompt(prompt, [f"S0 = {seed}", str(w_c), str(w_s), str(w_a), str(r_sep), str(max_speed)])

    golden = K9_GOLDEN.format(
        wc=w_c, ws=w_s, wa=w_a, rsep=r_sep, vmax=max_speed, seed=seed,
        camera=list(K9_CAMERA), look=list(K9_LOOK), light=list(K9_LIGHT),
    )
    scene = _plain_checks(K9_CAMERA, ["backdrop", "agent-0", "agent-11"])
    simulation = [visual.near("steps", steps, 0), visual.size_between("agents", K9_N, K9_N)] + [
        visual.near(f"agents.{i}.{key}", value, 1e-3)
        for i in range(K9_N)
        for key, value in zip(("x", "y", "vx", "vy"), (fp[i][0], fp[i][1], fv[i][0], fv[i][1]))
    ]
    image = [
        visual.image_min("changed_fraction", 0.004),
        visual.image_min("distinct_colors", 9),
        visual.image_color("warm", 0.004, "агенты стаи различимы"),
        visual.image_centroid("warm", 0, 0.2, 0.8, "стая держится центра кадра"),
        visual.image_region(0, visual.cool, "верх кадра — тёмный фон"),
    ]
    return visual.build_task(
        task_id="K-9", title="K-9 — стая (boids)", prompt=prompt,
        spec=visual.VisualSpec(scene=scene, simulation=simulation, image=image, steps=steps, dt=dt),
        golden=golden,
        rng_meta={"seed": seed, "steps": steps, "dt": dt, "r_sep": r_sep, "max_speed": max_speed,
                  "positions": fp, "velocities": fv},
    )


# --------------------------------------------------------------------------- #
# K-10 — частицы в коробке с гравитацией
# --------------------------------------------------------------------------- #

K10_CAMERA = (0.0, 4.0, 16.0)
K10_LOOK = (0.0, 2.0, 0.0)
K10_LIGHT = (6.0, 12.0, 8.0)
K10_R = 0.4
K10_HALF = 5.0
K10_TOP = 4.0

K10_FIELDS = """{
  "steps": <число сделанных шагов>,
  "bounces": <сколько отскоков всего случилось>,
  "balls": [{"x": <x>, "y": <y>, "vx": <vx>, "vy": <vy>}, ... по записи на шарик ...],
  "camera": [<камера сцены, 3 числа>],
  "objects": ["floor", "wall", "..."]
}"""

K10_PROMPT = """Собери частицы в закрытой коробке под гравитацией: шарики
отскакивают от пола, потолка и стен без потери скорости.

Сцена:
* пол — плоскость 40x40 в плоскости y=0, материал тёмно-синий (0x2b3a4a),
  имя объекта `floor`;
* задняя стена — плоскость 16x12 в плоскости z = -2, материал 0x22303f,
  имя `wall`;
* шарики — {n} сфер радиуса {r}, материал оранжевый (0xff8a3d), имена
  `ball-0` ...;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {camera} и смотрит
  в {look}; свет — `AmbientLight(0xffffff, 0.45)` и
  `DirectionalLight(0xffffff, 2.0)` из точки {light}.

Коробка: x ∈ [-{half}, {half}], y ∈ [0, {top}]; радиус шарика r = {r};
коэффициент восстановления 1 — скорость по нормали просто меняет знак.

Начальное состояние — из ЛЦГ:
* {lcg}
* для шарика i = 0..{n_minus} по порядку: `x = -4 + 8*u`, `y = 0.8 + 1.2*u`,
  `vx = -3 + 6*u`, `vy = 3 + 7*u`.

Симуляция (`window.__step(dt)` — один шаг), для каждого шарика по индексу:
* `vy -= g*dt` (g = {g}); `x += vx*dt`; `y += vy*dt`;
* если `y - r < 0`: `y = r`, `vy = -vy`, отскок; иначе если `y + r > {top}`:
  `y = {top} - r`, `vy = -vy`, отскок;
* если `x - r < -{half}`: `x = -{half} + r`, `vx = -vx`, отскок; иначе если
  `x + r > {half}`: `x = {half} - r`, `vx = -vx`, отскок;
* каждый отскок увеличивает общий счётчик `bounces`.

`window.__probe()` возвращает ровно:

{fields}

Проверяется: состав сцены и камера, координаты и скорости всех шариков и число
отскоков после заданного числа шагов (сверка с той же схемой, допуск 1e-3), и
кадр — шарики видны и различимы, картинка воспроизводима.
"""

K10_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>box</title>
<style>html,body{{margin:0;background:#101318;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = {n}, G = {g}, R = {r}, HALF = {half}, TOP = {top}, SEED = {seed};

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

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

const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({{color: 0x2b3a4a}}));
floor.rotation.x = -Math.PI / 2;
floor.name = 'floor';
scene.add(floor);
const wall = new THREE.Mesh(new THREE.PlaneGeometry(16, 12), new THREE.MeshStandardMaterial({{color: 0x22303f}}));
wall.position.z = -2;
wall.name = 'wall';
scene.add(wall);

const balls = [];
for (let i = 0; i < N; i++) {{
  const b = {{ x: -4 + 8 * next(), y: 0.8 + 1.2 * next(), vx: -3 + 6 * next(), vy: 3 + 7 * next() }};
  b.mesh = new THREE.Mesh(new THREE.SphereGeometry(R, 16, 12), new THREE.MeshStandardMaterial({{color: 0xff8a3d}}));
  b.mesh.name = `ball-${{i}}`;
  b.mesh.position.set(b.x, b.y, 0);
  scene.add(b.mesh);
  balls.push(b);
}}

const round = (v) => Math.round(v * 1e4) / 1e4;
let steps = 0, bounces = 0;
window.__step = (dt) => {{
  for (const b of balls) {{
    b.vy -= G * dt;
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    if (b.y - R < 0) {{ b.y = R; b.vy = -b.vy; bounces += 1; }}
    else if (b.y + R > TOP) {{ b.y = TOP - R; b.vy = -b.vy; bounces += 1; }}
    if (b.x - R < -HALF) {{ b.x = -HALF + R; b.vx = -b.vx; bounces += 1; }}
    else if (b.x + R > HALF) {{ b.x = HALF - R; b.vx = -b.vx; bounces += 1; }}
    b.mesh.position.set(b.x, b.y, 0);
  }}
  steps += 1;
  renderer.render(scene, camera);
  return window.__probe();
}};
window.__probe = () => ({{
    camera: [camera.position.x, camera.position.y, camera.position.z],
  steps,
  bounces,
  balls: balls.map((b) => ({{ x: round(b.x), y: round(b.y), vx: round(b.vx), vy: round(b.vy) }})),
  objects: scene.children.map((o) => o.name).filter(Boolean),
}});

renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _box(n: int, g: float, seed: int, steps: int, dt: float, bounce: bool = True):
    """The K-10 page program in Python; returns the balls and the bounce count.

    ``bounce=False`` is the naive answer (no walls): it exists only so the
    fixture can assert that the correct answer is not the trivial one.
    """
    gen = _lcg(seed)
    balls = []
    for _ in range(n):
        x = -4.0 + 8.0 * next(gen)
        y = 0.8 + 1.2 * next(gen)
        vx = -3.0 + 6.0 * next(gen)
        vy = 3.0 + 7.0 * next(gen)
        balls.append({"x": x, "y": y, "vx": vx, "vy": vy})
    bounces = 0
    r = K10_R
    for _ in range(steps):
        for b in balls:
            b["vy"] -= g * dt
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            if not bounce:
                continue
            if b["y"] - r < 0:
                b["y"] = r
                b["vy"] = -b["vy"]
                bounces += 1
            elif b["y"] + r > K10_TOP:
                b["y"] = K10_TOP - r
                b["vy"] = -b["vy"]
                bounces += 1
            if b["x"] - r < -K10_HALF:
                b["x"] = -K10_HALF + r
                b["vx"] = -b["vx"]
                bounces += 1
            elif b["x"] + r > K10_HALF:
                b["x"] = K10_HALF - r
                b["vx"] = -b["vx"]
                bounces += 1
    return balls, bounces


def _build_k10(rng: random.Random) -> core.TaskInstance:
    n = int(rng.choice([3, 4]))
    g = float(rng.choice([9.0, 12.0]))
    seed = int(rng.randrange(1, 2**32))
    steps = int(rng.choice([90, 105, 120]))
    dt = 1 / 60
    balls, bounces = _box(n, g, seed, steps, dt)

    prompt = K10_PROMPT.format(
        n=n, n_minus=n - 1, r=K10_R, half=K10_HALF, top=K10_TOP, g=f"{g:.2f}",
        camera=list(K10_CAMERA), look=list(K10_LOOK), light=list(K10_LIGHT),
        fields=K10_FIELDS, lcg=LCG_TEXT.format(seed=seed),
    )
    assert bounces > 0, "к моменту проверки должен случиться хотя бы один отскок"
    assert len({(round(b["x"], 6), round(b["y"], 6)) for b in balls}) == n, "шарики должны быть различимы"
    assert all(-K10_HALF <= b["x"] <= K10_HALF and 0 <= b["y"] <= K10_TOP for b in balls), \
        "эталон обязан остаться внутри коробки и в кадре"
    # The naive "no walls" answer diverges from the bouncing one.
    naive, _ = _box(n, g, seed, steps, dt, bounce=False)
    assert max(max(abs(b["x"] - v["x"]), abs(b["y"] - v["y"])) for b, v in zip(balls, naive)) > 0.2, \
        "наивный ответ без отскоков обязан разойтись с правильным"
    _assert_in_frame([(b["x"], b["y"]) for b in balls], K10_CAMERA, K10_LOOK, "K-10")
    _assert_in_prompt(prompt, [str(n), f"{g:.2f}", str(K10_R), str(K10_HALF), str(K10_TOP),
                               f"S0 = {seed}"])

    golden = K10_GOLDEN.format(
        n=n, g=g, r=K10_R, half=K10_HALF, top=K10_TOP, seed=seed,
        camera=list(K10_CAMERA), look=list(K10_LOOK), light=list(K10_LIGHT),
    )
    scene = _plain_checks(K10_CAMERA, ["floor", "wall", "ball-0"])
    simulation = [
        visual.near("steps", steps, 0),
        visual.near("bounces", bounces, 0),
        visual.size_between("balls", n, n),
    ] + [
        visual.near(f"balls.{i}.{key}", value, 1e-3)
        for i, b in enumerate(balls)
        for key, value in zip(("x", "y", "vx", "vy"), (b["x"], b["y"], b["vx"], b["vy"]))
    ]
    image = [
        visual.image_min("changed_fraction", 0.3),
        visual.image_min("distinct_colors", 9),
        visual.image_color("warm", 0.0016, "шарики различимы"),
        visual.image_centroid("warm", 1, 0.05, 0.95, "шарики видны в кадре"),
        visual.image_region(7, visual.cool, "низ кадра — тёмный пол"),
    ]
    return visual.build_task(
        task_id="K-10", title="K-10 — частицы в коробке с гравитацией", prompt=prompt,
        spec=visual.VisualSpec(scene=scene, simulation=simulation, image=image, steps=steps, dt=dt),
        golden=golden,
        rng_meta={"n": n, "g": g, "seed": seed, "steps": steps, "dt": dt, "bounces": bounces,
                  "balls": balls},
    )


# --------------------------------------------------------------------------- #
# K-11 — частицы в заданном поле скоростей
# --------------------------------------------------------------------------- #

K11_CAMERA = (4.0, 3.0, 18.0)
K11_LOOK = (4.0, 3.0, 0.0)
K11_LIGHT = (6.0, 10.0, 10.0)
K11_W = 8.0
K11_H = 6.0

K11_FIELDS = """{
  "steps": <число сделанных шагов>,
  "field": {"A": <A>, "B": <B>},
  "particles": [{"x": <x>, "y": <y>}, ... по одной записи на частицу ...],
  "camera": [<камера сцены, 3 числа>],
  "objects": ["backdrop", "particles", "..."]
}"""

K11_PROMPT = """Собери плоскую сцену: частицы плывут по заданному
аналитическому полю скоростей и оставляют видимый поток.

Сцена:
* фон — плоскость 40x40 в плоскости z = -2, материал тёмно-синий (0x22303f),
  имя объекта `backdrop`;
* частицы — ровно одна `THREE.InstancedMesh` из {n} сфер радиуса 0.3,
  материал оранжевый (0xffa64d), имя `particles`;
* домен поля — прямоугольник x ∈ [0, {w}], y ∈ [0, {h}];
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке {camera} и смотрит
  в {look}; свет — `AmbientLight(0xffffff, 0.5)` и
  `DirectionalLight(0xffffff, 1.8)` из точки {light}.

Начальные позиции — из ЛЦГ:
* {lcg}
* для частицы i = 0..{n_minus} по порядку: `x = {w}*u`, `y = {h}*u`.

Поле скоростей: `vx(x, y) = A*sin(pi*y/{h})`, `vy(x, y) = B*cos(pi*x/{w})`,
где A = {a}, B = {b}.

Симуляция (`window.__step(dt)` — один шаг), для каждой частицы по индексу:
посчитать `vx`, `vy` в её текущей точке, затем `x += vx*dt`, `y += vy*dt`.
Границ нет — частицы просто плывут по полю.

`window.__probe()` возвращает ровно:

{fields}

Проверяется: состав сцены и камера, параметры поля A и B, координаты всех
частиц после заданного числа шагов (сверка с той же схемой, допуск 1e-3), и
кадр — частицы видны и различимы, картинка воспроизводима.
"""

K11_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>field</title>
<style>html,body{{margin:0;background:#101318;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = {n}, A = {a}, B = {b}, W = {w}, H = {h}, SEED = {seed};

let s = SEED;
const next = () => {{ s = (1664525 * s + 1013904223) % 4294967296; return s / 4294967296; }};

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
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(...{light});
scene.add(key);

const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({{color: 0x22303f}}));
backdrop.position.z = -2;
backdrop.name = 'backdrop';
scene.add(backdrop);

const px = new Array(N), py = new Array(N);
for (let i = 0; i < N; i++) {{ px[i] = W * next(); py[i] = H * next(); }}

const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.3, 12, 8), new THREE.MeshStandardMaterial({{color: 0xffa64d}}), N);
mesh.name = 'particles';
mesh.frustumCulled = false;
scene.add(mesh);
const dummy = new THREE.Object3D();
const round = (v) => Math.round(v * 1e4) / 1e4;
const sync = () => {{
  for (let i = 0; i < N; i++) {{ dummy.position.set(px[i], py[i], 0); dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix); }}
  mesh.instanceMatrix.needsUpdate = true;
}};

let steps = 0;
window.__step = (dt) => {{
  for (let i = 0; i < N; i++) {{
    const vx = A * Math.sin(Math.PI * py[i] / H);
    const vy = B * Math.cos(Math.PI * px[i] / W);
    px[i] += vx * dt;
    py[i] += vy * dt;
  }}
  steps += 1;
  sync();
  renderer.render(scene, camera);
  return window.__probe();
}};
window.__probe = () => ({{
    camera: [camera.position.x, camera.position.y, camera.position.z],
  steps,
  field: {{ A, B }},
  particles: px.map((_, i) => ({{ x: round(px[i]), y: round(py[i]) }})),
  objects: scene.children.map((o) => o.name).filter(Boolean),
}});

sync();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
"""


def _field(n: int, a: float, b: float, seed: int, steps: int, dt: float):
    """The K-11 page program in Python; returns the particle positions."""
    gen = _lcg(seed)
    px = []
    py = []
    for _ in range(n):
        px.append(K11_W * next(gen))
        py.append(K11_H * next(gen))
    for _ in range(steps):
        for i in range(n):
            vx = a * math.sin(math.pi * py[i] / K11_H)
            vy = b * math.cos(math.pi * px[i] / K11_W)
            px[i] += vx * dt
            py[i] += vy * dt
    return px, py


def _build_k11(rng: random.Random) -> core.TaskInstance:
    n = int(rng.choice([20, 28, 36]))
    a = float(rng.choice([1.0, 1.5, 2.0]))
    b = float(rng.choice([1.0, 1.5, 2.0]))
    seed = int(rng.randrange(1, 2**32))
    steps = int(rng.choice([30, 45, 60]))
    dt = 1 / 60
    px, py = _field(n, a, b, seed, steps, dt)

    prompt = K11_PROMPT.format(
        n=n, n_minus=n - 1, a=a, b=b, w=K11_W, h=K11_H, camera=list(K11_CAMERA),
        look=list(K11_LOOK), light=list(K11_LIGHT), fields=K11_FIELDS, lcg=LCG_TEXT.format(seed=seed),
    )
    assert len({(round(x, 6), round(y, 6)) for x, y in zip(px, py)}) == n, "частицы должны остаться различимыми"
    assert all(-2.0 <= x <= K11_W + 2.0 and -2.0 <= y <= K11_H + 2.0 for x, y in zip(px, py)), \
        "эталон обязан остаться в кадре"
    # The naive "particles are static" answer is nowhere near the flowing one.
    gen = _lcg(seed)
    start = [(K11_W * next(gen), K11_H * next(gen)) for _ in range(n)]
    assert max(math.hypot(px[i] - start[i][0], py[i] - start[i][1]) for i in range(n)) > 0.05
    _assert_in_frame(list(zip(px, py)), K11_CAMERA, K11_LOOK, "K-11")
    _assert_in_prompt(prompt, [str(n), str(a), str(b), str(K11_W), str(K11_H), f"S0 = {seed}"])

    golden = K11_GOLDEN.format(
        n=n, a=a, b=b, w=K11_W, h=K11_H, seed=seed,
        camera=list(K11_CAMERA), look=list(K11_LOOK), light=list(K11_LIGHT),
    )
    scene = _plain_checks(K11_CAMERA, ["backdrop", "particles"])
    simulation = [
        visual.near("steps", steps, 0),
        visual.near("field.A", a, 1e-6),
        visual.near("field.B", b, 1e-6),
    ] + [
        visual.near(f"particles.{i}.{axis}", value, 1e-3)
        for i in range(n) for axis, value in zip(("x", "y"), (px[i], py[i]))
    ]
    image = [
        visual.image_min("changed_fraction", 0.006),
        visual.image_min("distinct_colors", 8),
        visual.image_color("warm", 0.006, "частицы потока различимы"),
        visual.image_centroid("warm", 0, 0.2, 0.8, "поток виден в центре кадра"),
        visual.image_region(0, visual.cool, "верх кадра — тёмный фон"),
    ]
    return visual.build_task(
        task_id="K-11", title="K-11 — частицы в заданном поле скоростей", prompt=prompt,
        spec=visual.VisualSpec(scene=scene, simulation=simulation, image=image, steps=steps, dt=dt),
        golden=golden,
        rng_meta={"n": n, "a": a, "b": b, "seed": seed, "steps": steps, "dt": dt,
                  "positions": list(zip(px, py))},
    )


def build(rng: random.Random) -> list[core.TaskInstance]:
    """All five scenarios, each drawing its fixture from the shared stream."""
    return [_build_k7(rng), _build_k8(rng), _build_k9(rng), _build_k10(rng), _build_k11(rng)]
