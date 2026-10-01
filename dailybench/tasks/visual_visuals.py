"""Visual scenarios K-17..K-20 — visualisations: a bar chart, an orrery, an
inverse-kinematics arm and a DFT spectrum.

In all four the picture *is* the report of a computation, and the check
recomputes that computation independently in Python:

* K-17 — bar heights are the fixture numbers, read back from the page after the
  grow-in animation has finished;
* K-18 — orbital angles follow the stated ``theta += 360 / T * dt`` scheme,
  accumulated exactly as the page accumulates it;
* K-19 — the two-link closed form is solved inside ``__step`` and the effector
  must land on the target;
* K-20 — the columns are the discrete Fourier amplitudes of the stated signal.

None of the four is chaotic, so comparing the page with the same scheme in
Python is a definition rather than a guess. Each reference page is a real
answer: ``selfcheck`` renders it in the headless browser and requires 4/4.
"""

from __future__ import annotations

import json
import math
import random

from .. import core, visual

IDS = ("K-17", "K-18", "K-19", "K-20")

#: Fixed camera per scenario, so every answer is judged on the same picture.
K17_CAMERA = (0.0, 7.0, 18.0)
K17_LOOK = (0.0, 3.0, 0.0)
K17_LIGHT = (8.0, 14.0, 8.0)
K17_GROW = 0.5

K18_CAMERA = (0.0, 10.0, 13.0)
K18_LOOK = (0.0, 0.0, 0.0)
K18_LIGHT = (7.0, 13.0, 7.0)

K19_CAMERA = (1.6, 2.0, 9.5)
K19_LOOK = (1.6, 2.0, 0.0)
K19_LIGHT = (5.0, 9.0, 8.0)

K20_CAMERA = (0.0, 5.0, 15.0)
K20_LOOK = (0.0, 2.5, 0.0)
K20_LIGHT = (7.0, 13.0, 8.0)


def _fill(template: str, **values: str) -> str:
    out = template
    for key, value in values.items():
        out = out.replace(f"@@{key}@@", value if isinstance(value, str) else str(value))
    assert "@@" not in out, f"в шаблоне остались плейсхолдеры: {out[:160]!r}"
    return out


def mentions(text: str, *values: str) -> bool:
    """Every fixture number is printed in the prompt, so the model can know it."""
    return all(value in text for value in values)


def in_frame(
    point: tuple[float, float, float],
    camera: tuple[float, float, float],
    look: tuple[float, float, float],
    fov: float = 50.0,
    aspect: float = 800 / 600,
    margin: float = 0.7,
) -> bool:
    """Whether a world point projects inside the frame, with a safety margin.

    ``margin`` < 1 keeps the point away from the border: a scene that only just
    fits the viewport would make the image check depend on the fixture instead
    of on the answer. This uses the same camera geometry Three.js uses, so it is
    a construction-time proof rather than an eyeball estimate.
    """

    def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def norm(a: tuple[float, float, float]) -> tuple[float, float, float]:
        length = math.sqrt(dot(a, a)) or 1.0
        return (a[0] / length, a[1] / length, a[2] / length)

    forward = norm(sub(look, camera))
    right = norm(cross(forward, (0.0, 1.0, 0.0)))
    up = cross(right, forward)
    view = sub(point, camera)
    depth = dot(view, forward)
    if depth <= 0.1:
        return False
    half_height = depth * math.tan(math.radians(fov) / 2)
    half_width = half_height * aspect
    return abs(dot(view, right)) <= margin * half_width and abs(dot(view, up)) <= margin * half_height


# --------------------------------------------------------------------------- #
# Local predicates (the framework makes them callable from the module)
# --------------------------------------------------------------------------- #


def marks_in_order(path: str, key: str, wanted: list[str]) -> visual.Predicate:
    """The probe lists the objects in the order the task gave them."""

    def check(payload: dict) -> tuple[bool, str]:
        rows = visual.path_of(payload, path, None)
        if not isinstance(rows, list) or len(rows) != len(wanted):
            return False, f"{path}: {rows!r} (ожидался список из {len(wanted)})"
        got = [row.get(key) if isinstance(row, dict) else None for row in rows]
        return got == list(wanted), f"{path}: {key} = {got} (ожидались {list(wanted)})"

    return check


def all_heights(path: str, wanted: list[float], tol: float = 1e-3) -> visual.Predicate:
    """Every bar height equals its datum, one by one."""

    def check(payload: dict) -> tuple[bool, str]:
        rows = visual.path_of(payload, path, None)
        if not isinstance(rows, list) or len(rows) != len(wanted):
            return False, f"{path}: {rows!r} (ожидался список из {len(wanted)})"
        bad = [
            (index, row.get("height"))
            for index, row in enumerate(rows)
            if not isinstance(row, dict) or abs(float(row.get("height", 0.0)) - wanted[index]) > tol
        ]
        return not bad, f"{path}: высоты не совпали {bad}" if bad else f"{path}: все {len(wanted)} высот на месте"

    return check


def angles_match(wanted: list[float], tol: float = 1e-3) -> visual.Predicate:
    """Both joint angles equal the closed-form solution."""

    def check(payload: dict) -> tuple[bool, str]:
        rows = visual.path_of(payload, "angles", None)
        if not isinstance(rows, list) or len(rows) != len(wanted):
            return False, f"angles: {rows!r}"
        bad = [index for index, value in enumerate(wanted) if abs(float(rows[index]) - value) > tol]
        return not bad, f"angles = {rows} (ожидались {wanted} ± {tol})"

    return check


def point_matches(path: str, wanted: tuple[float, float], tol: float = 1e-3) -> visual.Predicate:
    """A two-element [x, y] pair equals the wanted point."""

    def check(payload: dict) -> tuple[bool, str]:
        row = visual.path_of(payload, path, None)
        if not isinstance(row, list) or len(row) < 2:
            return False, f"{path}: {row!r}"
        ok = abs(float(row[0]) - wanted[0]) <= tol and abs(float(row[1]) - wanted[1]) <= tol
        return ok, f"{path} = {row} (ожидалось {list(wanted)} ± {tol})"

    return check


# --------------------------------------------------------------------------- #
# K-17 — 3D bar chart
# --------------------------------------------------------------------------- #

K17_PROBE = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "bar-0", "bar-1", "...", "marker"],
  "bars": [{"label": "<подпись столбца>", "height": <текущая высота>}, ...],
  "max_height": <максимальная высота среди столбцов>,
  "active": <индекс активного столбца>
}"""

K17_PROMPT = """Собери сцену столбчатой диаграммы: каждый столбец — параллелепипед,
стоящий на полу; высота столбца равна значению из таблицы, а подписи идут
слева направо ровно в заданном порядке.

Данные:
@@table@@

Сцена:
* пол — плоскость 24x24 в плоскости y=0, материал тёмно-синий (0x2b3a4a),
  имя объекта `ground`;
* столбец i — `BoxGeometry(1, 1, 1)` с оранжевым материалом (0xff8a3d),
  имя `bar-i`, стоит в точке x = (i - (n - 1) / 2) * 2.0, z = 0; масштаб по y
  равен текущей высоте h, центр по y равен h / 2 (нижняя грань лежит на полу);
* высота растёт во времени: h_i = value_i * min(1, t / @@grow@@), где t — время
  в секундах; при загрузке t = 0, поэтому все высоты равны 0;
* маркер — сфера радиуса 0.25, жёлтая (0xffe066), имя `marker`; маркер висит
  над активным столбцом на высоте h + 0.5;
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке @@camera@@ и смотрит
  в @@look@@; свет — `AmbientLight(0xffffff, 0.35)` и
  `DirectionalLight(0xffffff, 2.2)` из точки @@light@@.

Симуляция:
* шаг делается только вызовом `window.__step(dt)`;
* на шаге: `t += dt`, `steps += 1`, каждая высота пересчитывается по формуле
  h_i = value_i * min(1, t / @@grow@@), активный столбец — `steps % n` (n — число
  столбцов из таблицы), маркер
  переезжает над активным столбцом; по x и z столбцы никогда не двигаются;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: состав сцены, порядок подписей, высоты всех столбцов после
заданного числа шагов (сверка с данными, допуск 1e-3), максимальная высота и
кадр — столбцы различимы и картинка воспроизводима при повторной загрузке.

`objects` — имена прямых детей сцены; проверяется наличие `ground`, всех
`bar-*` и `marker`, лишние имена не мешают.
"""

K17_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>bars</title>
<style>html,body{margin:0;background:#101318;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const DATA = @@data@@;
const GROW = @@grow@@;
const N = DATA.length;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 200);
camera.position.set(...@@camera@@);
camera.lookAt(...@@look@@);

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(...@@light@@);
scene.add(key);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(24, 24),
  new THREE.MeshStandardMaterial({color: 0x2b3a4a})
);
ground.rotation.x = -Math.PI / 2;
ground.name = 'ground';
scene.add(ground);

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshStandardMaterial({color: 0xff8a3d});

const bars = DATA.map((item, index) => {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'bar-' + index;
  mesh.position.set((index - (N - 1) / 2) * 2, 0, 0);
  mesh.scale.set(1.2, 0.001, 1.2);
  scene.add(mesh);
  return {label: item.label, value: item.value, mesh};
});

const marker = new THREE.Mesh(
  new THREE.SphereGeometry(0.25, 20, 14),
  new THREE.MeshStandardMaterial({color: 0xffe066})
);
marker.name = 'marker';
scene.add(marker);

let steps = 0;
let elapsed = 0;

const heightOf = (bar) => bar.value * Math.min(1, elapsed / GROW);

function apply() {
  for (const bar of bars) {
    const h = heightOf(bar);
    bar.mesh.scale.y = h;
    bar.mesh.position.y = h / 2;
  }
  const active = steps % N;
  marker.position.set(bars[active].mesh.position.x, heightOf(bars[active]) + 0.5, 0);
  return active;
}

function frame() {
  renderer.render(scene, camera);
}

window.__step = (dt) => {
  elapsed += dt;
  steps += 1;
  apply();
  frame();
  return window.__probe();
};

const round = (value) => Math.round(value * 1e4) / 1e4;

window.__probe = () => {
  const rows = bars.map((bar) => ({label: bar.label, height: round(bar.mesh.scale.y)}));
  return {
    steps,
    t: round(elapsed),
    objects: scene.children.map((o) => o.name).filter(Boolean),
    bars: rows,
    max_height: round(Math.max(...rows.map((row) => row.height))),
    active: steps % N,
  };
};

apply();
frame();
window.__ready = true;
</script>
</body></html>
"""


def build_k17(rng: random.Random) -> core.TaskInstance:
    count = rng.choice([5, 6, 7])
    labels = [chr(ord("A") + index) for index in range(count)]
    values: list[float] = []
    while len(values) < count:  # the data must be distinct, so redraw until it is
        value = round(rng.uniform(1.5, 7.5), 2)
        if value not in values:
            values.append(value)
    steps = 45
    dt = 1 / 60
    elapsed = steps * dt

    # The fixture proves itself: distinct data, distinct labels, and the grow-in
    # animation is over by the sampled step — after that the height is the datum.
    assert len(set(values)) == count, "значения столбцов должны различаться"
    assert len(set(labels)) == count
    assert elapsed >= K17_GROW + 0.1, "к шагу проверки анимация роста должна закончиться"
    assert elapsed - K17_GROW < 0.5
    assert sum(abs(value) for value in values) > 0
    # Every bar (its base and its top, plus the marker above the tallest) fits
    # the frame with margin, so the image check fails only for a wrong answer.
    for index, value in enumerate(values):
        x = (index - (count - 1) / 2) * 2.0
        for y in (0.0, value):
            assert in_frame((x, y, 0.0), K17_CAMERA, K17_LOOK), f"столбец {index} выходит за кадр"
    assert in_frame((0.0, max(values) + 0.5, 0.0), K17_CAMERA, K17_LOOK), "маркер выходит за кадр"

    table = "\n".join(
        f"* столбец {index} — подпись «{label}», значение {value:.2f}"
        for index, (label, value) in enumerate(zip(labels, values))
    )
    prompt = _fill(
        K17_PROMPT.format(fields=K17_PROBE),
        table=table,
        grow=f"{K17_GROW:g}",
        camera=list(K17_CAMERA),
        look=list(K17_LOOK),
        light=list(K17_LIGHT),
    )
    data = json.dumps(
        [{"label": label, "value": value} for label, value in zip(labels, values)],
        ensure_ascii=False,
    )
    # The prompt has to carry every fixture number verbatim — a check the model
    # cannot satisfy because a value was never printed is a broken task.
    assert mentions(prompt, *[f"значение {value:.2f}" for value in values])
    assert mentions(prompt, *[f"подпись «{label}»" for label in labels])
    golden = _fill(
        K17_GOLDEN,
        data=data,
        grow=f"{K17_GROW:g}",
        camera=list(K17_CAMERA),
        look=list(K17_LOOK),
        light=list(K17_LIGHT),
    )
    assert "min(1, elapsed / GROW)" in golden

    spec = visual.VisualSpec(
        scene=[
            visual.contains_all("objects", ["ground"] + [f"bar-{i}" for i in range(count)] + ["marker"]),
            visual.size_between("bars", count, count),
            marks_in_order("bars", "label", labels),
            all_heights("bars", [0.0] * count, tol=0.02),  # до шагов столбцы ещё не выросли
            visual.near("steps", 0, 0.5),
        ],
        simulation=[
            visual.near("steps", steps, 0.5),
            visual.near("t", elapsed, 1e-4),
            visual.near("active", steps % count, 0.5),
            all_heights("bars", values),
            visual.near("max_height", max(values), 1e-3),
        ],
        image=[
            # Замер эталона на 12 seed'ах: warm 0.060..0.084, changed 0.478..0.497.
            visual.image_color("warm", 0.02, "оранжевые столбцы и маркер"),
            visual.image_min("changed_fraction", 0.05),
        ],
        steps=steps,
        dt=dt,
    )
    return visual.build_task(
        task_id="K-17",
        title="K-17 — 3D-гистограмма: высоты столбцов равны данным",
        prompt=prompt,
        spec=spec,
        golden=golden,
        rng_meta={"values": values, "labels": labels, "steps": steps, "dt": dt},
    )


# --------------------------------------------------------------------------- #
# K-18 — orrery
# --------------------------------------------------------------------------- #

K18_PROBE = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "camera": [0, 10, 13],
  "objects": ["sun", "orbit-0", "...", "<имя планеты>"],
  "planets": [{"name": "<имя>", "theta": <угол в градусах>, "x": <x>, "z": <z>}, ...]
}"""

K18_PROMPT = """Собери сцену оррерии: три планеты обращаются вокруг звезды по
круговым орбитам заданных радиусов, периоды и начальные фазы даны.

Сцена:
* звезда — сфера радиуса 0.6, жёлтая (0xffcf5c), имя `sun`, центр в точке
  (0, 0.6, 0);
* планета i — сфера радиуса 0.3, имя и цвет из таблицы; орбита — плоское
  кольцо `RingGeometry(R - 0.03, R + 0.03, 96)`, повёрнутое в горизонтальную
  плоскость (x-z) и поднятое на y = 0.3, цвет 0x2f4a63, имя `orbit-i`;
* планеты:
@@table@@
  (имя, радиус орбиты R, период T в секундах, начальная фаза theta0 в градусах,
  цвет материала);
* при загрузке theta каждой планеты равна её theta0;
* положение планеты i: x = R * cos(theta * PI / 180),
  z = R * sin(theta * PI / 180), y = 0.3 (theta — в градусах);
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке @@camera@@ и смотрит
  в @@look@@; свет — `AmbientLight(0xffffff, 0.35)` и
  `DirectionalLight(0xffffff, 2.2)` из точки @@light@@.

Симуляция:
* шаг делается только вызовом `window.__step(dt)`;
* схема шага ровно такая: `theta += 360 / T * dt`, затем `theta = theta % 360`
  (theta в градусах и всегда лежит в [0, 360)); после этого планета
  переставляется по формулам выше;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: состав сцены и камера, углы и координаты всех трёх планет после
заданного числа шагов (сверка с той же схемой, допуск 1e-3) и кадр — звезда и
планеты различимы, картинка воспроизводима при повторной загрузке.

`objects` — имена прямых детей сцены; проверяется наличие `sun`, всех `orbit-*`
и всех планет, лишние имена не мешают.
"""

K18_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>orrery</title>
<style>html,body{margin:0;background:#080a12;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const PLANETS = @@planets@@;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080a12);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 200);
camera.position.set(...@@camera@@);
camera.lookAt(...@@look@@);

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(...@@light@@);
scene.add(key);

const sun = new THREE.Mesh(
  new THREE.SphereGeometry(0.6, 24, 16),
  new THREE.MeshStandardMaterial({color: 0xffcf5c, emissive: 0x553300})
);
sun.position.set(0, 0.6, 0);
sun.name = 'sun';
scene.add(sun);

const bodies = PLANETS.map((planet, index) => {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(planet.radius - 0.03, planet.radius + 0.03, 96),
    new THREE.MeshBasicMaterial({color: 0x2f4a63, side: THREE.DoubleSide})
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.3;
  ring.name = `orbit-${index}`;
  scene.add(ring);

  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.3, 20, 14),
    new THREE.MeshStandardMaterial({color: planet.color})
  );
  mesh.name = planet.name;
  scene.add(mesh);
  return {spec: planet, mesh, theta: planet.theta0};
});

let steps = 0;
let elapsed = 0;

function place() {
  for (const body of bodies) {
    const radians = body.theta * Math.PI / 180;
    body.mesh.position.set(
      body.spec.radius * Math.cos(radians),
      0.3,
      body.spec.radius * Math.sin(radians)
    );
  }
}

function frame() {
  renderer.render(scene, camera);
}

window.__step = (dt) => {
  for (const body of bodies) {
    body.theta = (body.theta + 360 / body.spec.period * dt) % 360;
  }
  steps += 1;
  elapsed += dt;
  place();
  frame();
  return window.__probe();
};

const round = (value) => Math.round(value * 1e4) / 1e4;

window.__probe = () => ({
  steps,
  t: round(elapsed),
  camera: [camera.position.x, camera.position.y, camera.position.z],
  objects: scene.children.map((o) => o.name).filter(Boolean),
  planets: bodies.map((body) => ({
    name: body.spec.name,
    theta: round(body.theta),
    x: round(body.mesh.position.x),
    z: round(body.mesh.position.z),
  })),
});

place();
frame();
window.__ready = true;
</script>
</body></html>
"""


def _orrery(planets: list[dict], steps: int, dt: float) -> list[tuple[float, float, float]]:
    """The stated scheme, accumulated exactly as the reference page does."""
    out: list[tuple[float, float, float]] = []
    for planet in planets:
        theta = planet["theta0"]
        for _ in range(steps):
            theta = (theta + 360.0 / planet["period"] * dt) % 360.0
        radians = theta * math.pi / 180.0
        out.append(
            (
                round(theta, 4),
                round(planet["radius"] * math.cos(radians), 4),
                round(planet["radius"] * math.sin(radians), 4),
            )
        )
    return out


def build_k18(rng: random.Random) -> core.TaskInstance:
    names = ["Меркурий", "Венера", "Марс"]
    colors = [0x3D6BFF, 0x22B3A0, 0xFF5566]
    steps = 90
    dt = 1 / 60
    # The fixture proves itself: орбиты и периоды различимы, углы после шагов не
    # совпадают с начальными фазами и между собой (иначе проверка ничего не
    # ловит). Негодный набор перерисовывается — rng всё равно один.
    while True:
        radii = [
            round(rng.uniform(1.8, 2.4), 2),
            round(rng.uniform(3.2, 3.8), 2),
            round(rng.uniform(4.6, 5.4), 2),
        ]
        periods: list[float] = []
        while len(periods) < 3:
            period = round(rng.uniform(6.0, 20.0), 2)
            if period not in periods:
                periods.append(period)
        phases = [round(rng.uniform(0.0, 360.0), 2) for _ in range(3)]
        planets = [
            {"name": name, "radius": radii[i], "period": periods[i], "theta0": phases[i], "color": colors[i]}
            for i, name in enumerate(names)
        ]
        expected = _orrery(planets, steps, dt)
        if len({theta for theta, _, _ in expected}) != 3:
            continue
        if not all(abs(theta - planet["theta0"]) > 1.0 for (theta, _, _), planet in zip(expected, planets)):
            continue
        break

    assert len({planet["radius"] for planet in planets}) == 3
    assert len({planet["period"] for planet in planets}) == 3
    # Орбиты целиком в кадре (кольцо радиуса R + планета на нём), плюс звезда.
    assert in_frame((0.0, 0.6, 0.0), K18_CAMERA, K18_LOOK), "звезда выходит за кадр"
    for planet in planets:
        edge = planet["radius"] + 0.03
        for point in (
            (edge, 0.3, 0.0),
            (-edge, 0.3, 0.0),
            (0.0, 0.3, edge),
            (0.0, 0.3, -edge),
        ):
            assert in_frame(point, K18_CAMERA, K18_LOOK), f"орбита {planet['name']} выходит за кадр"

    table = "\n".join(
        f"* {planet['name']}: R = {planet['radius']:.2f}, T = {planet['period']:.2f} с, "
        f"theta0 = {planet['theta0']:.2f}°, цвет 0x{planet['color']:06x}"
        for planet in planets
    )
    prompt = _fill(
        K18_PROMPT.format(fields=K18_PROBE),
        table=table,
        camera=list(K18_CAMERA),
        look=list(K18_LOOK),
        light=list(K18_LIGHT),
    )
    golden = _fill(
        K18_GOLDEN,
        planets=json.dumps(planets, ensure_ascii=False),
        camera=list(K18_CAMERA),
        look=list(K18_LOOK),
        light=list(K18_LIGHT),
    )
    assert mentions(
        prompt,
        *[
            value
            for planet in planets
            for value in (
                f"R = {planet['radius']:.2f}",
                f"T = {planet['period']:.2f}",
                f"theta0 = {planet['theta0']:.2f}°",
                f"0x{planet['color']:06x}",
            )
        ],
    )

    spec = visual.VisualSpec(
        scene=[
            visual.contains_all(
                "objects", ["sun"] + [f"orbit-{i}" for i in range(3)] + names
            ),
            visual.near("camera.0", K18_CAMERA[0], 1e-4),
            visual.near("camera.1", K18_CAMERA[1], 1e-4),
            visual.near("camera.2", K18_CAMERA[2], 1e-4),
            visual.near("planets.0.theta", planets[0]["theta0"], 1e-3),
        ],
        simulation=[
            visual.near("steps", steps, 0.5),
            visual.near("t", steps * dt, 1e-4),
            visual.size_between("planets", 3, 3),
            marks_in_order("planets", "name", names),
        ]
        + [
            predicate
            for index, (theta, x, z) in enumerate(expected)
            for predicate in (
                visual.near(f"planets.{index}.theta", theta, 1e-3),
                visual.near(f"planets.{index}.x", x, 1e-3),
                visual.near(f"planets.{index}.z", z, 1e-3),
            )
        ],
        image=[
            # Замер эталона на 12 seed'ах: warm 0.0044..0.0051, cool 0.0095..0.0108.
            visual.image_color("warm", 0.0018, "звезда и тёплая планета"),
            visual.image_color("cool", 0.0035, "орбиты и холодные планеты"),
            visual.image_min("changed_fraction", 0.004),
        ],
        steps=steps,
        dt=dt,
    )
    return visual.build_task(
        task_id="K-18",
        title="K-18 — оррерия: углы и координаты трёх планет",
        prompt=prompt,
        spec=spec,
        golden=golden,
        rng_meta={"planets": planets, "steps": steps, "dt": dt},
    )


# --------------------------------------------------------------------------- #
# K-19 — two-link inverse kinematics
# --------------------------------------------------------------------------- #

K19_PROBE = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["base", "arm1", "target"],
  "angles": [<theta1>, <theta2>],
  "effector": [<x схвата>, <y схвата>],
  "target": [<x цели>, <y цели>]
}"""

K19_PROMPT = """Собери сцену плоского двузвенного манипулятора: он должен навести
схват точно в цель, решив обратную кинематику.

Данные: L1 = @@l1@@, L2 = @@l2@@, цель в точке (@@tx@@, @@ty@@) плоскости x-y.

Сцена:
* основание — цилиндр радиуса 0.4 и высоты 0.3, серый (0x8899aa), имя `base`,
  ось цилиндра по y, центр в (0, 0.15, 0);
* звено 1 — параллелепипед (L1, 0.3, 0.3) светлого цвета (0xcfd8e3) внутри
  группы `arm1`: группа стоит в начале координат и вращается вокруг оси z на
  угол theta1, звено смещено по x на L1 / 2 внутри группы;
* звено 2 — параллелепипед (L2, 0.3, 0.3) того же цвета внутри группы `arm2`:
  `arm2` вложена в `arm1` и стоит в точке (L1, 0, 0), вращается вокруг z на
  угол theta2, звено смещено по x на L2 / 2 внутри `arm2`;
* схват — сфера радиуса 0.2, цвет 0x44ddaa, имя `effector`, закреплена в
  `arm2` в точке (L2, 0, 0) (перед целью, чуть ближе к камере);
* цель — сфера радиуса 0.3, оранжевая (0xff8a3d), имя `target`, в точке цели
  (@@tx@@, @@ty@@, -0.2);
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке @@camera@@ и смотрит
  в @@look@@; свет — `AmbientLight(0xffffff, 0.35)` и
  `DirectionalLight(0xffffff, 2.2)` из точки @@light@@.

Симуляция:
* при загрузке `arm1.rotation.z = 0` и `arm2.rotation.z = 0`: обратная
  кинематика заранее НЕ считается, манипулятор вытянут по оси x;
* шаг делается только вызовом `window.__step(dt)`; и только здесь (в каждом
  шаге) решается обратная кинематика по цели (x, y):
  `c2 = (x*x + y*y - L1*L1 - L2*L2) / (2 * L1 * L2)`,
  `theta2 = -acos(clamp(c2, -1, 1))`  (локоть вверх — берём знак минус),
  `theta1 = atan2(y, x) - atan2(L2 * sin(theta2), L1 + L2 * cos(theta2))`;
  затем `arm1.rotation.z = theta1`, `arm2.rotation.z = theta2`; после этого
  `steps += 1`, `t += dt`;
* углы — в радианах, atan2 — обычный `Math.atan2`;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: до шагов углы равны нулю (значит, кинематика не посчитана заранее),
после заданного числа шагов углы совпадают с аналитическим решением, а схват
стоит ровно в цели (допуск 1e-3); кадр — звенья и цель различимы.

`objects` — имена прямых детей сцены (то есть `base`, `arm1` и `target`);
`effector` и `arm2` вложены в группы и в этот список не попадают.
"""

K19_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>ik-arm</title>
<style>html,body{margin:0;background:#101318;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const L1 = @@l1@@;
const L2 = @@l2@@;
const TARGET = {x: @@tx@@, y: @@ty@@};

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 200);
camera.position.set(...@@camera@@);
camera.lookAt(...@@look@@);

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(...@@light@@);
scene.add(key);

const base = new THREE.Mesh(
  new THREE.CylinderGeometry(0.4, 0.4, 0.3, 24),
  new THREE.MeshStandardMaterial({color: 0x8899aa})
);
base.position.set(0, 0.15, 0);
base.name = 'base';
scene.add(base);

const linkMaterial = new THREE.MeshStandardMaterial({color: 0xcfd8e3});

const arm1 = new THREE.Group();
arm1.name = 'arm1';
scene.add(arm1);
const link1 = new THREE.Mesh(new THREE.BoxGeometry(L1, 0.3, 0.3), linkMaterial);
link1.position.set(L1 / 2, 0, 0);
arm1.add(link1);

const arm2 = new THREE.Group();
arm2.name = 'arm2';
arm2.position.set(L1, 0, 0);
arm1.add(arm2);
const link2 = new THREE.Mesh(new THREE.BoxGeometry(L2, 0.3, 0.3), linkMaterial);
link2.position.set(L2 / 2, 0, 0);
arm2.add(link2);

const effector = new THREE.Mesh(
  new THREE.SphereGeometry(0.2, 20, 14),
  new THREE.MeshStandardMaterial({color: 0x44ddaa})
);
effector.position.set(L2, 0, 0);
effector.name = 'effector';
arm2.add(effector);

const target = new THREE.Mesh(
  new THREE.SphereGeometry(0.3, 20, 14),
  new THREE.MeshStandardMaterial({color: 0xff8a3d})
);
target.position.set(TARGET.x, TARGET.y, -0.2);
target.name = 'target';
scene.add(target);

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

let steps = 0;
let elapsed = 0;

function solve() {
  const x = TARGET.x;
  const y = TARGET.y;
  const c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  const theta2 = -Math.acos(clamp(c2, -1, 1));
  const theta1 = Math.atan2(y, x) - Math.atan2(L2 * Math.sin(theta2), L1 + L2 * Math.cos(theta2));
  arm1.rotation.z = theta1;
  arm2.rotation.z = theta2;
}

function frame() {
  renderer.render(scene, camera);
}

window.__step = (dt) => {
  solve();
  steps += 1;
  elapsed += dt;
  frame();
  return window.__probe();
};

const round = (value) => Math.round(value * 1e4) / 1e4;

window.__probe = () => {
  const world = new THREE.Vector3();
  effector.getWorldPosition(world);
  return {
    steps,
    t: round(elapsed),
    objects: scene.children.map((o) => o.name).filter(Boolean),
    angles: [round(arm1.rotation.z), round(arm2.rotation.z)],
    effector: [round(world.x), round(world.y)],
    target: [TARGET.x, TARGET.y],
  };
};

frame();
window.__ready = true;
</script>
</body></html>
"""


def _ik(l1: float, l2: float, x: float, y: float) -> tuple[float, float, float, float]:
    """Closed-form elbow-up solution plus the forward kinematics it implies."""
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    c2 = max(-1.0, min(1.0, c2))
    theta2 = -math.acos(c2)
    theta1 = math.atan2(y, x) - math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    ex = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    ey = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    return theta1, theta2, ex, ey


def build_k19(rng: random.Random) -> core.TaskInstance:
    l1 = round(rng.uniform(2.2, 3.0), 2)
    l2 = round(rng.uniform(1.6, 2.2), 2)
    # The target is drawn through the cosine of theta2, which puts it inside the
    # working zone with a margin *by construction*: |c2| <= 0.8 keeps the elbow
    # angle away from 0 and 180 degrees, so the closed form is never degenerate.
    c2_target = round(rng.uniform(-0.8, 0.8), 4)
    phi = rng.uniform(25.0, 70.0)
    radius = math.sqrt(l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * c2_target)
    tx = round(radius * math.cos(math.radians(phi)), 2)
    ty = round(radius * math.sin(math.radians(phi)), 2)
    steps = 30
    dt = 1 / 60
    theta1, theta2, ex, ey = _ik(l1, l2, tx, ty)

    # The fixture proves itself: цель достижима с запасом от границ рабочей зоны,
    # аналитика действительно попадает в цель одним и тем же ответом, а «наивный»
    # ответ (вытянутая рука, кинематика не решена) в цель не попадает.
    c2 = (tx * tx + ty * ty - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    assert -0.95 < c2 < 0.95, "цель должна быть внутри рабочей зоны с запасом"
    assert abs(ex - tx) < 1e-9 and abs(ey - ty) < 1e-9, "аналитика обязана попадать в цель"
    assert theta2 < 0.05, "локоть вверх: theta2 отрицателен и заметно не ноль"
    assert abs(theta2) > 0.2
    naive = math.hypot(l1 + l2 - tx, ty)
    assert naive > 0.5, "наивный ответ (0, 0) должен заметно отличаться от цели"
    assert abs(theta1) > 0.05
    # Основание, цель и локоть целиком в кадре — иначе image-проверка падала бы
    # не из-за ответа модели, а из-за фикстуры.
    elbow = (l1 * math.cos(theta1), l1 * math.sin(theta1))
    for point in ((0.0, 0.0, 0.0), (tx, ty, 0.0), (elbow[0], elbow[1], 0.0)):
        assert in_frame(point, K19_CAMERA, K19_LOOK), f"точка {point} выходит за кадр"

    prompt = _fill(
        K19_PROMPT.format(fields=K19_PROBE),
        l1=f"{l1:.2f}",
        l2=f"{l2:.2f}",
        tx=f"{tx:.2f}",
        ty=f"{ty:.2f}",
        camera=list(K19_CAMERA),
        look=list(K19_LOOK),
        light=list(K19_LIGHT),
    )
    golden = _fill(
        K19_GOLDEN,
        l1=f"{l1:.2f}",
        l2=f"{l2:.2f}",
        tx=f"{tx:.2f}",
        ty=f"{ty:.2f}",
        camera=list(K19_CAMERA),
        look=list(K19_LOOK),
        light=list(K19_LIGHT),
    )
    assert "Math.acos" in golden and "arm1.rotation.z = theta1" in golden
    assert mentions(
        prompt, f"L1 = {l1:.2f}", f"L2 = {l2:.2f}", f"({tx:.2f}, {ty:.2f})"
    ), "в промпте должны быть все числа фикстуры"

    scene_checks = [
        visual.contains_all("objects", ["base", "arm1", "target"]),
        angles_match([0.0, 0.0]),
        point_matches("effector", (round(l1 + l2, 4), 0.0)),
    ]
    sim_checks = [
        visual.near("steps", steps, 0.5),
        visual.near("t", steps * dt, 1e-4),
        angles_match([round(theta1, 4), round(theta2, 4)]),
        point_matches("effector", (round(ex, 4), round(ey, 4))),
        point_matches("target", (tx, ty)),
    ]
    spec = visual.VisualSpec(
        scene=scene_checks,
        simulation=sim_checks,
        image=[
            # Замер эталона на 12 seed'ах: warm 0.0010..0.0011,
            # bright 0.0149..0.0167, changed 0.0171..0.0191.
            visual.image_color("warm", 0.0004, "цель"),
            visual.image_color("bright", 0.006, "светлые звенья"),
            visual.image_min("changed_fraction", 0.006),
        ],
        steps=steps,
        dt=dt,
    )
    return visual.build_task(
        task_id="K-19",
        title="K-19 — обратная кинематика: схват попадает в цель",
        prompt=prompt,
        spec=spec,
        golden=golden,
        rng_meta={"l1": l1, "l2": l2, "target": [tx, ty], "steps": steps, "dt": dt},
    )


# --------------------------------------------------------------------------- #
# K-20 — signal spectrum (DFT amplitudes)
# --------------------------------------------------------------------------- #

K20_SAMPLES = 64
K20_RATE = 64.0  # equals the sample count, so bin k sits exactly on k Hz

K20_PROBE = """{
  "steps": <число сделанных шагов>,
  "t": <время в секундах, 4 знака>,
  "objects": ["ground", "bin-0", "...", "wave", "playhead"],
  "bins": [{"freq": <частота бина, Гц>, "amplitude": <амплитуда ДПФ>}, ...],
  "dominant": <частота бина с наибольшей амплитудой>,
  "playhead": <индекс отсчёта под бегунком>
}"""

K20_PROMPT = """Собери сцену спектра сигнала: столбцы показывают амплитуды
дискретного преобразования Фурье (ДПФ) сигнала на заданных бинах.

Сигнал: N = @@n@@ отсчётов с частотой дискретизации fs = @@rate@@ Гц,
`s[n] = A1 * sin(2*PI*k1*n/N) + A2 * sin(2*PI*k2*n/N)`, n = 0..N-1,
где A1 = @@a1@@, k1 = @@k1@@; A2 = @@a2@@, k2 = @@k2@@.

ДПФ: амплитуда на бине k (частота бина f = k * fs / N):
`Re_k = сумма по n от 0 до N-1 от s[n] * cos(2*PI*k*n/N)`,
`Im_k = сумма по n от 0 до N-1 от s[n] * sin(2*PI*k*n/N)`,
`amp_k = (2 / N) * sqrt(Re_k*Re_k + Im_k*Im_k)`.
Считать надо ровно эти три бина: @@bins@@ (в этом порядке).

Сцена:
* пол — плоскость 20x20 в плоскости y=0, тёмно-синий (0x2b3a4a), имя `ground`;
* столбец i — `BoxGeometry(1, 1, 1)` с оранжевым материалом (0xff8a3d), имя
  `bin-i`, стоит в точке x = (i - (m - 1) / 2) * 3.2, z = 0 (m — число
  столбцов, то есть число бинов); его высота
  h_i = amp_k * 1.5, центр по y равен h_i / 2;
* график сигнала — ломаная `THREE.Line` по отсчётам n = 0..N-1 в точках
  x = -6 + 12 * n / (N - 1), y = 4.2 + s[n] * 0.5, z = -2, материал 0x66e0ff,
  имя `wave`;
* бегунок — сфера радиуса 0.15, жёлтая (0xffe066), имя `playhead`, стоит в
  точке графика с индексом p = (steps * 4) % N (то есть z = -2);
* камера `PerspectiveCamera(50, 800/600, 0.1, 200)` в точке @@camera@@ и смотрит
  в @@look@@; свет — `AmbientLight(0xffffff, 0.35)` и
  `DirectionalLight(0xffffff, 2.2)` из точки @@light@@.

Симуляция:
* амплитуды и столбцы считаются один раз при загрузке и дальше не меняются;
* шаг делается только вызовом `window.__step(dt)`: `steps += 1`, `t += dt`,
  бегунок переезжает в точку графика с индексом p = (steps * 4) % N;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: амплитуды всех трёх бинов (сверка с той же формулой ДПФ в Python,
допуск 1e-3), доминирующая частота и кадр — столбцы и график различимы.

`objects` — имена прямых детей сцены; проверяется наличие `ground`, всех
`bin-*`, `wave` и `playhead`, лишние имена не мешают.
"""

K20_GOLDEN = """<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>spectrum</title>
<style>html,body{margin:0;background:#101318;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const N = @@n@@;
const FS = @@rate@@;
const A1 = @@a1@@;
const K1 = @@k1@@;
const A2 = @@a2@@;
const K2 = @@k2@@;
const BINS = @@bins@@;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 200);
camera.position.set(...@@camera@@);
camera.lookAt(...@@look@@);

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(...@@light@@);
scene.add(key);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(20, 20),
  new THREE.MeshStandardMaterial({color: 0x2b3a4a})
);
ground.rotation.x = -Math.PI / 2;
ground.name = 'ground';
scene.add(ground);

const samples = Array.from({length: N}, (_, n) =>
  A1 * Math.sin(2 * Math.PI * K1 * n / N) + A2 * Math.sin(2 * Math.PI * K2 * n / N)
);

function amplitude(k) {
  let re = 0;
  let im = 0;
  for (let n = 0; n < N; n++) {
    const angle = 2 * Math.PI * k * n / N;
    re += samples[n] * Math.cos(angle);
    im += samples[n] * Math.sin(angle);
  }
  return 2 / N * Math.hypot(re, im);
}

const amps = BINS.map((k) => amplitude(k));
const COLUMN = 1.5;

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshStandardMaterial({color: 0xff8a3d});
const M = BINS.length;

const columns = BINS.map((k, index) => {
  const height = amps[index] * COLUMN;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = `bin-${index}`;
  mesh.position.set((index - (M - 1) / 2) * 3.2, height / 2, 0);
  mesh.scale.set(1.6, Math.max(height, 0.001), 1.6);
  scene.add(mesh);
  return mesh;
});

const points = samples.map((value, n) =>
  new THREE.Vector3(-6 + 12 * n / (N - 1), 4.2 + value * 0.5, -2)
);
const wave = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints(points),
  new THREE.LineBasicMaterial({color: 0x66e0ff})
);
wave.name = 'wave';
scene.add(wave);

const playhead = new THREE.Mesh(
  new THREE.SphereGeometry(0.15, 16, 12),
  new THREE.MeshStandardMaterial({color: 0xffe066})
);
playhead.name = 'playhead';
scene.add(playhead);

let steps = 0;
let elapsed = 0;

const indexAt = (step) => (step * 4) % N;

function place() {
  const at = points[indexAt(steps)];
  playhead.position.copy(at);
}

function frame() {
  renderer.render(scene, camera);
}

window.__step = (dt) => {
  steps += 1;
  elapsed += dt;
  place();
  frame();
  return window.__probe();
};

const round = (value) => Math.round(value * 1e4) / 1e4;

window.__probe = () => {
  let best = 0;
  for (let i = 1; i < BINS.length; i++) if (amps[i] > amps[best]) best = i;
  return {
    steps,
    t: round(elapsed),
    objects: scene.children.map((o) => o.name).filter(Boolean),
    bins: BINS.map((k, i) => ({freq: k * FS / N, amplitude: round(amps[i])})),
    dominant: BINS[best] * FS / N,
    playhead: indexAt(steps),
  };
};

place();
frame();
window.__ready = true;
</script>
</body></html>
"""


def _signal(n: int, a1: float, k1: int, a2: float, k2: int) -> list[float]:
    return [
        a1 * math.sin(2.0 * math.pi * k1 * index / n) + a2 * math.sin(2.0 * math.pi * k2 * index / n)
        for index in range(n)
    ]


def _dft_amplitude(samples: list[float], k: int) -> float:
    """The stated DFT formula, summed in the same order as the page."""
    re = 0.0
    im = 0.0
    n = len(samples)
    for index, value in enumerate(samples):
        angle = 2.0 * math.pi * k * index / n
        re += value * math.cos(angle)
        im += value * math.sin(angle)
    return 2.0 / n * math.hypot(re, im)


def build_k20(rng: random.Random) -> core.TaskInstance:
    n = K20_SAMPLES
    rate = K20_RATE
    k1 = rng.randint(2, 6)
    k2 = k1 + rng.randint(3, 8)
    a1 = round(rng.uniform(0.8, 3.0), 2)
    a2 = round(rng.uniform(0.8, 3.0), 2)
    if abs(a1 - a2) < 0.6:  # the dominant bin must be unambiguous
        higher = round(a1 + 0.7, 2)
        a2 = higher if higher <= 3.0 else round(a1 - 0.7, 2)
    k3 = rng.choice([k for k in range(1, 31) if abs(k - k1) > 2 and abs(k - k2) > 2])
    bins = sorted([k1, k2, k3])
    steps = 50
    dt = 1 / 60

    samples = _signal(n, a1, k1, a2, k2)
    amplitudes = [_dft_amplitude(samples, k) for k in bins]
    dominant = bins[amplitudes.index(max(amplitudes))]

    # The fixture proves itself: both tones sit exactly on their bins, so the
    # amplitudes are the amplitudes themselves — the right answer is unique and
    # a page that skips the DFT lands on something else. The extra bin carries
    # no tone at all, and the dominant frequency is decided by the data.
    assert len(set(bins)) == 3 and bins[0] >= 1 and bins[-1] <= 30
    assert abs(a1 - a2) > 0.5
    assert abs(_dft_amplitude(samples, k1) - a1) < 1e-9
    assert abs(_dft_amplitude(samples, k2) - a2) < 1e-9
    assert k3 not in (k1, k2)
    assert abs(amplitudes[bins.index(k3)]) < 1e-9, "на третьем бине тона нет"
    assert amplitudes.count(max(amplitudes)) == 1
    assert dominant in (k1, k2) and dominant == (k1 if a1 > a2 else k2)
    assert abs(amplitudes[bins.index(dominant)] - max(a1, a2)) < 1e-9
    # Столбцы и весь график сигнала целиком в кадре (бегунок ходит по графику).
    for index, amplitude in enumerate(amplitudes):
        x = (index - (len(bins) - 1) / 2) * 3.2
        height = amplitude * 1.5
        for y in (0.0, height):
            assert in_frame((x, y, 0.0), K20_CAMERA, K20_LOOK), f"столбец {index} выходит за кадр"
    for index, value in enumerate(samples):
        point = (-6 + 12 * index / (n - 1), 4.2 + value * 0.5, -2.0)
        assert in_frame(point, K20_CAMERA, K20_LOOK), f"точка графика {index} выходит за кадр"

    prompt = _fill(
        K20_PROMPT.format(fields=K20_PROBE),
        n=n,
        rate=f"{rate:g}",
        a1=f"{a1:.2f}",
        k1=k1,
        a2=f"{a2:.2f}",
        k2=k2,
        bins=", ".join(str(k) for k in bins),
        camera=list(K20_CAMERA),
        look=list(K20_LOOK),
        light=list(K20_LIGHT),
    )
    golden = _fill(
        K20_GOLDEN,
        n=n,
        rate=f"{rate:g}",
        a1=f"{a1:.2f}",
        k1=k1,
        a2=f"{a2:.2f}",
        k2=k2,
        bins=list(bins),
        camera=list(K20_CAMERA),
        look=list(K20_LOOK),
        light=list(K20_LIGHT),
    )
    assert "Math.hypot" in golden
    assert mentions(
        prompt,
        f"A1 = {a1:.2f}",
        f"k1 = {k1}",
        f"A2 = {a2:.2f}",
        f"k2 = {k2}",
        f"N = {n}",
        f"fs = {rate:g}",
        ", ".join(str(k) for k in bins),
    ), "в промпте должны быть все числа фикстуры"

    spec = visual.VisualSpec(
        scene=[
            visual.contains_all("objects", ["ground"] + [f"bin-{i}" for i in range(3)] + ["wave", "playhead"]),
            visual.size_between("bins", 3, 3),
            visual.near("playhead", 0, 0.5),
        ],
        simulation=[
            visual.near("steps", steps, 0.5),
            visual.near("t", steps * dt, 1e-4),
            visual.near("playhead", (steps * 4) % n, 0.5),
            visual.near("dominant", float(dominant), 1e-3),
        ]
        + [
            visual.near(f"bins.{index}.freq", float(k), 1e-3)
            for index, k in enumerate(bins)
        ]
        + [
            visual.near(f"bins.{index}.amplitude", round(amplitude, 4), 1e-3)
            for index, amplitude in enumerate(amplitudes)
        ],
        image=[
            # Замер эталона на 12 seed'ах: warm 0.0367..0.0496, changed 0.449..0.461.
            visual.image_color("warm", 0.012, "оранжевые столбцы и бегунок"),
            visual.image_min("changed_fraction", 0.05),
        ],
        steps=steps,
        dt=dt,
    )
    return visual.build_task(
        task_id="K-20",
        title="K-20 — спектр сигнала: амплитуды ДПФ на заданных бинах",
        prompt=prompt,
        spec=spec,
        golden=golden,
        rng_meta={"k1": k1, "k2": k2, "a1": a1, "a2": a2, "bins": bins, "steps": steps, "dt": dt},
    )


def build(rng: random.Random) -> list[core.TaskInstance]:
    return [build_k17(rng), build_k18(rng), build_k19(rng), build_k20(rng)]
