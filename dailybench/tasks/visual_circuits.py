"""Visual scenarios K-24, K-25 — RC step response and thermostat hysteresis.

Same shape as the other visual slices: the prompt states the scene, the exact
update rule, the fixture constants and the exact probe fields; ``build()``
draws the constants from ``rng`` and renders the prompt *and* the reference
page from them, while the expected numbers are recomputed here in Python from
the stated rule. A check therefore asks "does the circuit obey the scheme it
was told", never "does it look like the reference".

Both scenarios are explicit-time integrators, so the constants are chosen to
keep them stable and to keep every snapshot meaningful:

* K-24 integrates the RC step with forward Euler at ``dt = 0.01`` for 300
  steps, with snapshots at ``t = 0``, ``t = 1`` and ``t = 3``; the fixture
  asserts monotonicity and that the final value is close to the source.
* K-25 is the hysteresis thermostat: the switch is decided *before* the
  temperature step, and the outside temperature drops at step 60. The check
  reads the state, the counters and the exact transition step numbers.

K-25 is a pilot scenario; K-24 belongs to the stretch profile.
"""

from __future__ import annotations

import math
import random
from string import Template

from .. import core, visual

IDS = ("K-24", "K-25")
IDS_BY_PROFILE = {"pilot": ("K-25",), "stretch-v1": IDS}

#: Euler sample points, in steps, for K-24: 1 s and 3 s at dt = 0.01.
SNAP_ONE = 100
SNAP_THREE = 300

#: The outside-temperature drop of K-25, in steps.
EVENT_STEP = 60

#: Fixed camera and light per scenario.
CAMERA24 = (9.0, 7.0, 12.5)
CAMERA25 = (9.0, 7.0, 12.5)
LIGHT24 = (6.0, 12.0, 9.0)
LIGHT25 = (5.0, 12.0, 8.0)

FIELDS24 = """{
  "steps": <число сделанных шагов>,
  "vin": <напряжение источника, В>,
  "v0": <Vc на t=0>, "v1": <Vc на t=1 c>, "v3": <Vc на t=3 c>,
  "i0": <ток на t=0>, "i1": <ток на t=1 c>, "i3": <ток на t=3 c>,
  "camera": [<камера сцены, 3 числа>],
  "objects": ["board", "source", "resistor", "capacitor", "gauge", "wires"]
}"""

FIELDS25 = """{
  "steps": <номер шага>,
  "on": <true/false — нагреватель включён>,
  "temperature": <температура T, 4 знака>,
  "heating_steps": <сколько шагов нагреватель был включён>,
  "switches_on": <сколько раз выключатель перешёл выкл -> вкл>,
  "switches_off": <сколько раз выключатель перешёл вкл -> выкл>,
  "on_steps": [<номера шагов перехода выкл -> вкл>],
  "off_steps": [<номера шагов перехода вкл -> выкл>],
  "outside": <температура снаружи, 4 знака>,
  "camera": [<камера сцены, 3 числа>],
  "objects": ["floor", "thermometer", "column", "heater", "dial"]
}"""

PROMPT24 = """Собери сцену заряда RC-цепи: источник подаёт ступеньку напряжения,
конденсатор заряжается по явной схеме Эйлера, на доске виден рост заряда.

Сцена:
* доска — `BoxGeometry(9.4, 0.3, 5.4)` в точке `(0, -0.15, 0)`, цвет
  0x16202c, имя `board`;
* источник — куб `BoxGeometry(1, 1.2, 1)` в точке `(-3.6, 0.6, 0)`, цвет
  0x2f6fb3, имя `source`;
* резистор — брусок `BoxGeometry(1.6, 0.5, 0.5)` в точке `(-1.6, 0.5, 0)`,
  цвет 0xb35b2f, имя `resistor`;
* конденсатор — две пластины `BoxGeometry(0.12, 1.6, 1.2)` в точках
  `(1.2, 0.8, 0)` и `(1.7, 0.8, 0)`, цвет 0x9aa7b8, имя `capacitor`;
* провода — три бруска 2.2x0.08x0.08 на высоте 0.05 вдоль оси X, цвет
  0x55606b, имя `wires`;
* индикатор заряда — брусок `BoxGeometry(0.5, 1, 0.5)` в точке
  `(3.4, h/2, 0)`, высота `h = 0.2 + 3 * (Vc / Vin)`, цвет
  `lerp(0x1d3f7a, 0xff5a1e, Vc / Vin)`, имя `gauge`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 1, 0); свет — `AmbientLight(0xffffff, 0.6)` и
  `DirectionalLight(0xffffff, 1.5)` из точки {light}.

Симуляция:
* `Vin = {vin}` В, `R = {r}` Ом, `C = {c}` Ф, в начале `Vc = 0`, ток
  `I = (Vin - Vc) / R`;
* шаг делается только вызовом `window.__step(dt)`; `dt = 0.01` с, ровно один
  вызов — один шаг явной схемы Эйлера:
  `Vc = Vc + dt * (Vin - Vc) / (R * C)`; ток после шага —
  `I = (Vin - Vc) / R`;
* снимки `v0/i0` берутся в `t = 0`, `v1/i1` — ровно после {snap_one} шагов
  (`t = 1`), `v3/i3` — ровно после {snap_three} шагов (`t = 3`); до своего
  момента снимок хранит начальное значение;
* состояние рисуется в `window.__probe()`: сначала обнови индикатор, потом
  `renderer.render(scene, camera)`, и только затем возвращай поля;
  `window.__step(dt)` состояние только продвигает и ничего не рисует;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: состояние сразу после загрузки, снимки `t = 1` и `t = 3`
(сверка с этой же схемой, допуск 1e-3), монотонность `Vc`, спад тока и
приближение `Vc` к `Vin` после {snap_three} шагов, и кадр — цепь видна,
индикатор вырос и потеплел, картинка воспроизводима при повторной загрузке.
"""

PROMPT25 = """Собери сцену термостата с гистерезисом: нагреватель включается при
низкой температуре, выключается при высокой, между порогами состояние не
меняется.

Сцена:
* пол — `BoxGeometry(9, 0.3, 5.4)` в точке `(0, -0.15, 0)`, цвет 0x16202c,
  имя `floor`;
* термометр — трубка `BoxGeometry(0.35, 5, 0.35)` в точке `(-1.8, 2.5, 0)`,
  цвет 0x27313d, имя `thermometer`;
* столбик — брусок `BoxGeometry(0.22, 1, 0.22)` в точке `(-1.8, h/2, 0)`,
  высота `h = 0.3 + 4 * (T - ({t_low} - 0.5)) / {gap}`, цвет
  `lerp(0x2f6fb3, 0xff3b1e, (T - ({t_low} - 0.5)) / {gap})`, имя `column`;
* нагреватель — куб `BoxGeometry(1.6, 1.2, 1.2)` в точке `(1.6, 0.6, 0)`,
  цвет 0xff7a1e при включённом нагревателе и 0x334455 при выключенном,
  имя `heater`;
* регулятор — куб `BoxGeometry(0.7, 0.7, 0.7)` в точке `(0.2, 0.35, 0)`,
  цвет 0x8a6b3d, имя `dial`;
* камера `PerspectiveCamera(50, 800/600, 0.1, 500)` в точке {cam} и смотрит в
  (0, 1.2, 0); свет — `AmbientLight(0xffffff, 0.6)` и
  `DirectionalLight(0xffffff, 1.5)` из точки {light}.

Симуляция:
* `T` — температура комнаты, старт `T = {t_start}`; снаружи сначала
  `Tout = {tout0}`, ровно на шаге {event} становится `Tout = {tout_event}`;
* нагреватель в начале выключен; шаг начинается с гистерезиса по текущей
  `T`: если `T <= {t_low}` — включить, иначе если `T >= {t_high}` —
  выключить, иначе состояние сохранить; переход засчитывается в счётчик и
  записывается номером шага, только когда состояние действительно меняется;
* затем температура меняется: при включённом нагревателе
  `T = T + {heat} * dt`, при выключенном `T = T - {k} * (T - Tout) * dt`;
  шаг делается только вызовом `window.__step(dt)` при `dt = 0.1` с, всего
  {steps} шагов;
* состояние рисуется в `window.__probe()`: сначала обнови столбик и цвет
  нагревателя, потом `renderer.render(scene, camera)`, и только затем
  возвращай поля; `window.__step(dt)` состояние только продвигает и ничего
  не рисует;
* `window.__probe()` возвращает ровно такие поля (числа — с 4 знаками):

{fields}

Проверяется: начальное состояние, состояние, счётчики и номера переходов
после {steps} шагов (сверка с этой же схемой, температура — допуск 1e-3), и
кадр — столбик и нагреватель видны, картинка воспроизводима при повторной
загрузке.
"""


GOLDEN24 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>rc step</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const VIN = $vin;
const R = $r;
const C = $c;
const V0 = 0;
const I0 = VIN / R;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 500);
camera.position.set(...$camera);
camera.lookAt(0, 1, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(...$light);
scene.add(key);

function board(x, y, z, sx, sy, sz, color, name, parent) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(sx, sy, sz),
    new THREE.MeshStandardMaterial({color})
  );
  mesh.position.set(x, y, z);
  if (name) mesh.name = name;
  (parent || scene).add(mesh);
  return mesh;
}

board(0, -0.15, 0, 9.4, 0.3, 5.4, 0x16202c, 'board');
board(-3.6, 0.6, 0, 1, 1.2, 1, 0x2f6fb3, 'source');
board(-1.6, 0.5, 0, 1.6, 0.5, 0.5, 0xb35b2f, 'resistor');
board(1.2, 0.8, 0, 0.12, 1.6, 1.2, 0x9aa7b8, 'capacitor');
board(1.7, 0.8, 0, 0.12, 1.6, 1.2, 'capacitor');
board(-2.4, 0.05, 0, 2.2, 0.08, 0.08, 0x55606b, 'wires');
board(0.1, 0.05, 0, 1.0, 0.08, 0.08, 0x55606b, 'wires');
board(2.5, 0.05, 0, 2.0, 0.08, 0.08, 0x55606b, 'wires');

const gauge = board(3.4, 0.1, 0, 0.5, 1, 0.5, 0x1d3f7a, 'gauge');
const cold = new THREE.Color(0x1d3f7a);
const hot = new THREE.Color(0xff5a1e);
const tint = new THREE.Color();

let vc = 0;
let clock = 0;
let steps = 0;
let v1 = V0, i1 = I0, v3 = V0, i3 = I0;

function paint() {
  const level = Math.min(1, vc / VIN);
  const h = 0.2 + 3 * level;
  gauge.scale.set(1, h, 1);
  gauge.position.set(3.4, h / 2 + 0.15, 0);
  tint.copy(cold).lerp(hot, level);
  gauge.material.color.copy(tint);
}

window.__step = (dt) => {
  vc += dt * (VIN - vc) / (R * C);
  clock += dt;
  steps += 1;
  if (steps === 100) { v1 = vc; i1 = (VIN - vc) / R; }
  if (steps === 300) { v3 = vc; i3 = (VIN - vc) / R; }
};

const round = (v) => Math.round(v * 1e4) / 1e4;

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  return {
    steps,
    vin: VIN,
    v0: V0, v1: round(v1), v3: round(v3),
    i0: round(I0), i1: round(i1), i3: round(i3),
    camera: [camera.position.x, camera.position.y, camera.position.z],
    objects: scene.children.map((o) => o.name).filter(Boolean),
  };
};

paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")

GOLDEN25 = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>thermostat hysteresis</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const TLOW = $t_low;
const THIGH = $t_high;
const HEAT = $heat;
const K = $k;
const TOUT_EVENT = $tout_event;

let T = $t_start;
let outside = $tout0;
let on = false;
let step = 0;
let heatingSteps = 0;
let onCount = 0;
let offCount = 0;
const onSteps = [];
const offSteps = [];

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 800 / 600, 0.1, 500);
camera.position.set(...$camera);
camera.lookAt(0, 1.2, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(...$light);
scene.add(key);

function box(x, y, z, sx, sy, sz, color, name) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(sx, sy, sz),
    new THREE.MeshStandardMaterial({color})
  );
  mesh.position.set(x, y, z);
  if (name) mesh.name = name;
  scene.add(mesh);
  return mesh;
}

box(0, -0.15, 0, 9, 0.3, 5.4, 0x16202c, 'floor');
box(-1.8, 2.5, 0, 0.35, 5, 0.35, 0x27313d, 'thermometer');
const column = box(-1.8, 0.15, 0, 0.22, 1, 0.22, 0x2f6fb3, 'column');
const heater = box(1.6, 0.6, 0, 1.6, 1.2, 1.2, 0x7a4a2e, 'heater');
box(0.2, 0.35, 0, 0.7, 0.7, 0.7, 0x8a6b3d, 'dial');

const cold = new THREE.Color(0x2f6fb3);
const hot = new THREE.Color(0xff3b1e);
const ON_COLOR = new THREE.Color(0xff7a1e);
const OFF_COLOR = new THREE.Color(0x7a4a2e);
const tint = new THREE.Color();
const SPAN = THIGH - (TLOW - 0.5);

function paint() {
  const level = Math.max(0, Math.min(1, (T - (TLOW - 0.5)) / SPAN));
  const h = 0.3 + 4 * level;
  column.scale.set(1, h, 1);
  column.position.set(-1.8, h / 2 + 0.15, 0);
  tint.copy(cold).lerp(hot, level);
  column.material.color.copy(tint);
  heater.material.color.copy(on ? ON_COLOR : OFF_COLOR);
}

window.__step = (dt) => {
  step += 1;
  if (step === $event) outside = TOUT_EVENT;
  if (T <= TLOW) {
    if (!on) { onCount += 1; onSteps.push(step); }
    on = true;
  } else if (T >= THIGH) {
    if (on) { offCount += 1; offSteps.push(step); }
    on = false;
  }
  if (on) heatingSteps += 1;
  T += (on ? HEAT : -K * (T - outside)) * dt;
};

const round = (v) => Math.round(v * 1e4) / 1e4;

window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  return {
    steps: step,
    on,
    temperature: round(T),
    heating_steps: heatingSteps,
    switches_on: onCount,
    switches_off: offCount,
    on_steps: onSteps.slice(),
    off_steps: offSteps.slice(),
    outside: round(outside),
    camera: [camera.position.x, camera.position.y, camera.position.z],
    objects: scene.children.map((o) => o.name).filter(Boolean),
  };
};

paint();
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


def rc_step(vin: float, r: float, c: float, steps: int, dt: float):
    """Forward Euler for ``C dVc/dt = (Vin - Vc) / R``; snapshots at 1 s and 3 s."""
    vc = 0.0
    v1 = i1 = v3 = i3 = 0.0
    for k in range(1, steps + 1):
        vc += dt * (vin - vc) / (r * c)
        if k == SNAP_ONE:
            v1, i1 = vc, (vin - vc) / r
        if k == SNAP_THREE:
            v3, i3 = vc, (vin - vc) / r
    return vc, v1, i1, v3, i3


def rc_exact(vin: float, r: float, c: float, t: float) -> float:
    """The analytic answer a page might use instead of Euler."""
    return vin * (1 - math.exp(-t / (r * c)))


def thermostat(
    t_start: float,
    t_low: float,
    t_high: float,
    tout0: float,
    tout_event: float,
    heat: float,
    k: float,
    steps: int,
    dt: float,
    *,
    single: bool = False,
    after: bool = False,
):
    """The hysteresis thermostat; ``single``/``after`` are the mutant programs."""

    def switch() -> None:
        nonlocal on, on_count, off_count, heating
        if single:
            target = T <= t_low
            if target != on:
                if target:
                    on_count += 1
                    on_steps.append(step)
                else:
                    off_count += 1
                    off_steps.append(step)
            on = target
            return
        if T <= t_low:
            if not on:
                on_count += 1
                on_steps.append(step)
            on = True
        elif T >= t_high:
            if on:
                off_count += 1
                off_steps.append(step)
            on = False

    T = t_start
    outside = tout0
    on = False
    heating = on_count = off_count = 0
    t_min = t_max = t_start
    on_steps: list[int] = []
    off_steps: list[int] = []
    for step in range(1, steps + 1):
        if step == EVENT_STEP:
            outside = tout_event
        if after:
            if on:
                heating += 1
            T += (heat if on else -k * (T - outside)) * dt
            switch()
        else:
            switch()
            if on:
                heating += 1
            T += (heat if on else -k * (T - outside)) * dt
        t_min = min(t_min, T)
        t_max = max(t_max, T)
    return T, on, heating, on_count, off_count, on_steps, off_steps, outside, t_min, t_max


# --------------------------------------------------------------------------- #
# Local predicates
# --------------------------------------------------------------------------- #


def _flag(path: str, value: bool) -> visual.Predicate:
    def check(payload: dict):
        actual = visual.path_of(payload, path, None)
        return actual is value, f"{path} = {actual} (ожидалось {value})"

    return visual._label(check, f"{path} = {value}")


def _camera_is(camera) -> visual.Predicate:
    def check(payload: dict):
        actual = visual.path_of(payload, "camera", None)
        good = isinstance(actual, list) and len(actual) == 3 and all(
            abs(float(a) - b) <= 1e-6 for a, b in zip(actual, camera)
        )
        return good, f"camera = {actual} (ожидалось {list(camera)})"

    return visual._label(check, "камера на месте")


def _rising(fields) -> visual.Predicate:
    def check(payload: dict):
        values = [visual.path_of(payload, path, None) for path in fields]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return False, f"{fields}: не числа ({values})"
        good = all(a < b for a, b in zip(values, values[1:]))
        return good, f"Vc = {values} (обязано строго расти)"

    return visual._label(check, "Vc монотонно растёт")


def _falling(fields) -> visual.Predicate:
    def check(payload: dict):
        values = [visual.path_of(payload, path, None) for path in fields]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return False, f"{fields}: не числа ({values})"
        good = all(a > b for a, b in zip(values, values[1:]))
        return good, f"I = {values} (обязан строго падать)"

    return visual._label(check, "ток монотонно спадает")


def _below(path: str, value: float) -> visual.Predicate:
    def check(payload: dict):
        actual, problem = visual._number(payload, path)
        if problem:
            return False, problem
        return actual <= value, f"{path} = {actual} (обязано быть <= {value})"

    return visual._label(check, f"{path} <= {value}")


def _ohm(v_path: str, i_path: str, vin: float, r: float) -> visual.Predicate:
    """Snapshot consistency: ``I = (Vin - Vc) / R``."""

    def check(payload: dict):
        v, problem = visual._number(payload, v_path)
        if problem:
            return False, problem
        i, problem = visual._number(payload, i_path)
        if problem:
            return False, problem
        expected = (vin - v) / r
        good = abs(i - expected) <= 1e-3
        return good, f"{v_path}/{i_path}: I = {i}, (Vin - Vc) / R = {expected}"

    return visual._label(check, f"{i_path} = (Vin - {v_path}) / R")


def _counter_len(count_path: str, list_path: str) -> visual.Predicate:
    def check(payload: dict):
        count, problem = visual._number(payload, count_path)
        if problem:
            return False, problem
        items = visual.path_of(payload, list_path, None)
        if not isinstance(items, list):
            return False, f"{list_path}: не список ({items!r})"
        return int(count) == len(items), f"{count_path} = {count}, длина {list_path} = {len(items)}"

    return visual._label(check, f"{count_path} сходится с {list_path}")


def _in_range(path: str, low: float, high: float) -> visual.Predicate:
    def check(payload: dict):
        actual, problem = visual._number(payload, path)
        if problem:
            return False, problem
        return low <= actual <= high, f"{path} = {actual} (ожидалось {low}..{high})"

    return visual._label(check, f"{path} в {low}..{high}")


def _heating_consistent(payload: dict) -> tuple[bool, str]:
    """``heating_steps`` must equal the length of every "on" interval."""
    on_steps = visual.path_of(payload, "on_steps", None)
    off_steps = visual.path_of(payload, "off_steps", None)
    steps, problem = visual._number(payload, "steps")
    if problem:
        return False, problem
    heating, problem = visual._number(payload, "heating_steps")
    if problem:
        return False, problem
    if not isinstance(on_steps, list) or not isinstance(off_steps, list):
        return False, "on_steps/off_steps: не списки"
    total = 0
    for index, start in enumerate(on_steps):
        stop = off_steps[index] if index < len(off_steps) else int(steps) + 1
        total += max(0, int(stop) - int(start))
    return total == int(heating), f"нагрев {heating} шагов, интервалы дают {total}"


# --------------------------------------------------------------------------- #
# Frame geometry: fixture objects must sit inside the frustum.
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


def _in_frame(point, camera, look, margin: float = FRAME_MARGIN) -> bool:
    forward = _unit(_sub(look, camera))
    right = _unit(_cross(forward, (0.0, 1.0, 0.0)))
    up = _cross(right, forward)
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


def _task_rc(rng: random.Random) -> core.TaskInstance:
    vin = float(rng.choice([12.0, 15.0, 18.0]))
    r = float(rng.choice([120.0, 150.0, 180.0]))
    c = float(rng.choice([0.0047, 0.0068]))
    dt = 0.01
    steps = SNAP_THREE
    vc, v1, i1, v3, i3 = rc_step(vin, r, c, steps, dt)
    tau = r * c
    exact1 = rc_exact(vin, r, c, 1.0)
    exact3 = rc_exact(vin, r, c, 3.0)

    # Fixture proves itself: the snapshots must be distinguishable, monotone,
    # below the source, and far enough from the analytic answer that a page
    # using the closed form instead of Euler cannot pass the 1e-3 comparison.
    assert 0.0 < v1 < v3 < vin, "Vc обязан строго расти и оставаться ниже Vin"
    assert (vin / r) > i1 > i3 > 0.0, "ток обязан строго спадать"
    assert len({round4(v1), round4(v3)}) == 2
    assert abs(v1 - exact1) > 0.01 and abs(v3 - exact3) > 0.002, "Эйлер обязан отличаться от точной формулы"
    assert v3 >= 0.85 * vin, "за 3 с Vc обязан подойти близко к Vin"

    prompt = PROMPT24.format(
        cam=list(CAMERA24), light=list(LIGHT24), vin=vin, r=r, c=c,
        snap_one=SNAP_ONE, snap_three=SNAP_THREE, fields=FIELDS24,
    )
    golden = GOLDEN24.substitute(
        vin=vin, r=r, c=c, camera=list(CAMERA24), light=list(LIGHT24),
    )
    _assert_literals(
        prompt,
        f"Vin = {vin}", f"R = {r}", f"C = {c}", "dt = 0.01",
        f"{SNAP_ONE} шагов", f"{SNAP_THREE} шагов",
        str(list(CAMERA24)), str(list(LIGHT24)),
    )
    _assert_in_frame(
        CAMERA24,
        (0.0, 1.0, 0.0),
        _box_corners(-4.7, 4.7, 0.0, 3.2, -2.7, 2.7),
        "K-24: цепь",
    )
    scene_checks = [
        visual.near("steps", 0, 0),
        visual.near("vin", vin, 1e-6),
        visual.near("v0", 0.0, 0),
        visual.near("v3", 0.0, 0),
        visual.near("i0", round4(vin / r), 1e-6),
        _camera_is(CAMERA24),
        visual.contains_all("objects", ["board", "source", "resistor", "capacitor", "gauge", "wires"]),
    ]
    sim_checks = [
        visual.near("steps", steps, 0),
        visual.near("v1", round4(v1), 1e-3),
        visual.near("v3", round4(v3), 1e-3),
        visual.near("i1", round4(i1), 1e-3),
        visual.near("i3", round4(i3), 1e-3),
        _rising(("v0", "v1", "v3")),
        _falling(("i0", "i1", "i3")),
        _ohm("v1", "i1", vin, r),
        _ohm("v3", "i3", vin, r),
        _below("v3", vin),
        _in_range("v3", 0.85 * vin, vin),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.01),
        visual.image_min("distinct_colors", 8),
        visual.image_color("warm", 0.002, "индикатор заряда потеплел"),
        visual.image_min("mean", 10.0),
    ]
    return visual.build_task(
        task_id="K-24",
        title="K-24 — ступенчатый отклик RC-цепи",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=steps, dt=dt),
        golden=golden,
        rng_meta={
            "vin": vin, "r": r, "c": c, "tau": tau, "steps": steps, "dt": dt,
            "v1": round4(v1), "v3": round4(v3), "i1": round4(i1), "i3": round4(i3),
            "vc": round4(vc),
        },
    )


def _task_thermostat(rng: random.Random) -> core.TaskInstance:
    t_start = float(rng.choice([18.4, 18.7, 19.0]))
    t_low = float(rng.choice([20.0, 20.5, 21.0]))
    t_high = t_low + float(rng.choice([2.0, 2.2, 2.5]))
    tout0 = float(rng.choice([8.0, 10.0, 12.0]))
    tout_event = float(rng.choice([0.0, 2.0, 4.0]))
    heat = float(rng.choice([2.4, 3.0, 3.6]))
    k = float(rng.choice([0.07, 0.08, 0.1]))
    steps, dt = 180, 0.1

    T, on, heating, on_count, off_count, on_steps, off_steps, outside, t_min, t_max = thermostat(
        t_start, t_low, t_high, tout0, tout_event, heat, k, steps, dt,
    )
    single = thermostat(
        t_start, t_low, t_high, tout0, tout_event, heat, k, steps, dt, single=True,
    )
    after = thermostat(
        t_start, t_low, t_high, tout0, tout_event, heat, k, steps, dt, after=True,
    )

    # Fixture proves itself: real oscillations, bounded temperature, and both
    # mutants visibly different, or the hysteresis would be untested.
    assert on_count >= 3 and off_count >= 3, "гистерезис обязан переключаться много кратно"
    assert len(on_steps) == on_count and len(off_steps) == off_count
    assert on_steps and off_steps and on_steps[0] < off_steps[0], "старт выключен, первым идёт включение"
    assert t_min >= t_start - 1e-9 and t_max <= t_high + 0.5, "температура обязана оставаться у порогов"
    assert max(on_steps) <= steps and max(off_steps) <= steps
    assert heating > 0 and heating < steps
    assert (on_count, off_count) != (single[3], single[4]) or on_steps != single[5], \
        "мутант с одним порогом обязан отличаться"
    assert (on_count, off_count) != (after[3], after[4]) or on_steps != after[5], \
        "мутант с переключением после шага обязан отличаться"

    gap = t_high - t_low
    prompt = PROMPT25.format(
        cam=list(CAMERA25), light=list(LIGHT25), t_start=t_start, t_low=t_low,
        t_high=t_high, gap=gap, tout0=tout0, tout_event=tout_event, heat=heat,
        k=k, event=EVENT_STEP, steps=steps, fields=FIELDS25,
    )
    golden = GOLDEN25.substitute(
        t_low=t_low, t_high=t_high, heat=heat, k=k, tout_event=tout_event,
        t_start=t_start, tout0=tout0, event=EVENT_STEP,
        camera=list(CAMERA25), light=list(LIGHT25),
    )
    _assert_literals(
        prompt,
        f"T = {t_start}", f"Tout = {tout0}", f"Tout = {tout_event}",
        f"T <= {t_low}", f"T >= {t_high}", f"{heat} * dt", f"{k} * (T - Tout)",
        "dt = 0.1", f"шаге {EVENT_STEP}", f"{steps} шагов",
        str(list(CAMERA25)), str(list(LIGHT25)),
    )
    _assert_in_frame(
        CAMERA25,
        (0.0, 1.2, 0.0),
        _box_corners(-4.5, 4.5, 0.0, 5.0, -2.7, 2.7),
        "K-25: термостат",
    )
    scene_checks = [
        visual.near("steps", 0, 0),
        _flag("on", False),
        visual.near("temperature", t_start, 1e-6),
        visual.near("heating_steps", 0, 0),
        visual.near("switches_on", 0, 0),
        _camera_is(CAMERA25),
        visual.contains_all("objects", ["floor", "thermometer", "column", "heater", "dial"]),
    ]
    sim_checks = [
        visual.near("steps", steps, 0),
        _flag("on", on),
        visual.near("temperature", round4(T), 1e-3),
        visual.near("outside", round4(outside), 1e-3),
        visual.near("heating_steps", heating, 0),
        visual.near("switches_on", on_count, 0),
        visual.near("switches_off", off_count, 0),
        visual.size_between("on_steps", len(on_steps), len(on_steps)),
        visual.size_between("off_steps", len(off_steps), len(off_steps)),
        visual.contains_all("on_steps", on_steps),
        visual.contains_all("off_steps", off_steps),
        _counter_len("switches_on", "on_steps"),
        _counter_len("switches_off", "off_steps"),
        _heating_consistent,
        _in_range("temperature", t_low - 0.5, t_high + 0.5),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.01),
        visual.image_min("distinct_colors", 6),
        visual.image_color("warm", 0.004, "нагреватель и столбик различимы"),
        visual.image_min("mean", 10.0),
    ]
    return visual.build_task(
        task_id="K-25",
        title="K-25 — термостат с гистерезисом",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=sim_checks, image=image_checks, steps=steps, dt=dt),
        golden=golden,
        rng_meta={
            "t_start": t_start, "t_low": t_low, "t_high": t_high, "tout0": tout0,
            "tout_event": tout_event, "heat": heat, "k": k, "steps": steps, "dt": dt,
            "on": on, "temperature": round4(T), "heating_steps": heating,
            "switches_on": on_count, "switches_off": off_count,
            "on_steps": on_steps, "off_steps": off_steps, "outside": round4(outside),
        },
    )


BUILDERS = (_task_rc, _task_thermostat)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Every scenario of the slice, each drawing its fixture from the same stream."""
    return [builder(rng) for builder in BUILDERS]
