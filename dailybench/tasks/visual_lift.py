"""Visual scenario K-28 — лифт с очередью вызовов и настоящими кнопками этажей.

The page is a lift shaft with six floor buttons. The host steps the page one
tick at a time; at ticks 5, 12 and 30 the page performs a *real* DOM click on
the button of a fixture floor (``button.click()`` — a genuine click event, not a
direct queue mutation), and the click handler is the only writer of the request
log. Everything the checks compare — car floor, door phase, direction, queue,
served floors, journal, clicks — is produced by the state machine below, and the
Python replica of that machine is the oracle. Three defects are therefore
visible in the journal and not only in the picture: a lost request, a
nearest-first pick, and a door that closes after one tick instead of two.

The scene is a fixed orthographic frame: the shaft on the left, the button
column on the right at x = 3.7..4.7. The buttons are DOM elements laid over the
canvas, so their clickability is real and their presence in the viewport is
measured by the probe.
"""

from __future__ import annotations

import random
from string import Template

from .. import core, visual

IDS = ("K-28",)

#: New scenario lives in the pilot slice only; the baseline does not know it.
IDS_BY_PROFILE = {"pilot": IDS, "stretch-v1": IDS}

#: Fixed frame of the scenario, in world units of an orthographic camera.
CAMERA = (0.0, 6.0, 30.0)
LOOK = (0.0, 6.0, 0.0)
ORTHO = (-8.0, 8.0, 7.0, -7.0)
FLOORS = 6
FLOOR_HEIGHT = 2
DOOR_TICKS = 2
MOVE_TICKS = 3

#: Ticks at which the host presses a button; the floors are the fixture.
PRESS_TICKS = (5, 12, 30)

FIELDS = """{
  "floors": 6,
  "car": 0,
  "door": "closed" | "opening" | "open" | "closing",
  "door_ticks": 0,
  "move_ticks": 0,
  "direction": "idle" | "up" | "down",
  "queue": [0],
  "served": [0],
  "journal": ["5:press 2"],
  "clicks": [{"tick": 5, "floor": 2}],
  "tick": 0,
  "buttons": [0, 1]
}"""

PROMPT = """Собери сцену лифта на шесть этажей: кнопки этажей — настоящие DOM-кнопки,
нажатие ставит вызов в очередь, кабина едет по одному этажу за три тика и
открывает дверь на два тика.

Сцена:
* камера `THREE.OrthographicCamera({left}, {right}, {top}, {bottom}, 0.1, 100)`
  в точке {cam} и смотрит в {look};
* шахта — каркас из тонких `BoxGeometry` (цвет 0x39424e) по краям
  x = -2.6 и x = 2.6, этажные перегородки на y = 2 * f для f = 0..6;
* кабина — `BoxGeometry(2.4, 1.2, 1.2)` цвета 0xff8c1a с именем `car` в
  точке `(0, car * 2 + 0.9, 0)`;
* дверь кабины — `BoxGeometry(0.15, 1.0, 1.2)` цвета 0xf2f4f8 в точке
  `(1.3, car * 2 + 0.9, 0)`;
* колонна индикаторов этажей — по одному `BoxGeometry(1.1, 0.8, 0.2)` цвета
  0x2fd3ff на каждом этаже в точке `(4.6, f * 2 + 1.0, 0)`; индикатор этажа
  из очереди перекрашивается в 0xffd27f;
* слева маркировка этажей — по одному `BoxGeometry(0.6, 0.6, 0.1)` в точке
  `(-4.6, f * 2 + 1.0, 0)` цветами по порядку этажей
  0x6a5a3a, 0x7a4a3a, 0x5a5a4a, 0x8a5a6a, 0x9a7a4a, 0x6a4a4a;
* свет — `AmbientLight(0xffffff, 0.9)` и `DirectionalLight(0xffffff, 1.6)` из
  точки (4, 12, 16); фон 0x0d1117.

Кнопки:
* справа поверх канваса — колонка из {floors} настоящих `<button>` (сверху
  этаж {top_floor}, снизу 0), у каждой атрибут `data-floor` и номер на подписи;
* кнопка этажа f при клике пишет `"<tick>:press f"` в журнал и добавляет `f` в
  очередь; если такой этаж уже в очереди, пишется ещё `"<tick>:ignored f"`, а
  очередь не меняется; каждый клик также добавляется в `clicks` как
  `{{"tick": tick, "floor": f}}`;
* в тики {t5}, {t12} и {t30} страница обязана нажать настоящую кнопку
  (`buttons[f].click()`) ровно этих этажей: {f1}, {f2} и {f3}; порядок нажатий
  внутри тика — по возрастанию этажа.

Симуляция (конечный автомат, один тик — ровно один проход):
* состояние: `car` (этаж кабины, старт 0), `door` (старт `"closed"`),
  `door_ticks`, `move_ticks`, `direction` (старт `"idle"`), `queue` (старт
  пустая), `served` (старт пустой), `tick` (старт 0);
* сначала тик увеличивается на 1, затем нажимаются кнопки этого тика, затем
  выполняется ровно одна ветка автомата:
  - если `door == "opening"`: `door_ticks += 1`, при `door_ticks >= {door_ticks}`
    дверь становится `"open"`, `door_ticks = 0`, в журнал `"<tick>:door open"`;
  - иначе если `door == "open"`: `door_ticks += 1`, при
    `door_ticks >= {door_ticks}` дверь становится `"closing"`, `door_ticks = 0`,
    в журнал `"<tick>:door closing"`;
  - иначе если `door == "closing"`: `door_ticks += 1`, при
    `door_ticks >= {door_ticks}` дверь становится `"closed"`, `door_ticks = 0`,
    в журнал `"<tick>:door closed"`;
  - иначе если `door == "closed"` и очередь не пуста: цель — первый элемент
    очереди (FIFO); если цель равна этажу кабины, она снимается с очереди,
    дверь становится `"opening"`, `door_ticks = 0`, `direction = "idle"`,
    этаж добавляется в `served`, в журнал `"<tick>:arrive <этаж>"` и сразу
    `"<tick>:door opening"`; иначе `direction` — `"up"` или `"down"` к цели,
    `move_ticks += 1`, и при `move_ticks >= {move_ticks}`: `move_ticks = 0`,
    `car` сдвигается на один этаж к цели, в журнал `"<tick>:move <car>"`, а
    если кабина встала на цель — цель снимается, `direction = "idle"`, этаж
    добавляется в `served`, дверь становится `"opening"`, `door_ticks = 0`, в
    журнал `"<tick>:arrive <car>"` и `"<tick>:door opening"`;
  - иначе `direction = "idle"`;
* шаг делает только `window.__step(dt)`, один вызов — один тик, `dt` не влияет;
* состояние рисуется в `window.__probe()`: сначала обнови положение кабины,
  двери и цвета индикаторов (индикатор этажа из очереди — 0xffd27f, иначе
  0x2fd3ff), затем `renderer.render(scene, camera)`, и только потом верни поля;
  `window.__step(dt)` ничего не рисует;
* `window.__probe()` возвращает ровно такие поля:

{fields}

`buttons` — этажи, чьи DOM-кнопки целиком лежат в кадре 800x600 (по
`getBoundingClientRect`).

Проверяется: стартовое состояние и все {floors} кнопок в кадре, состояние и
журнал ровно через {steps} тиков (сверка с этим автоматом, без допуска),
нажатия именно в тики {t5}, {t12}, {t30} и кадр — кабина и колонна кнопок
различимы, справа, картинка воспроизводима при повторной загрузке."""

GOLDEN = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>lift</title>
<style>html,body{margin:0;background:#0d1117;overflow:hidden}canvas{display:block}
#bar{position:fixed;right:24px;top:40px;display:flex;flex-direction:column;gap:16px}
button{width:34px;height:26px;border-radius:6px;border:1px solid #2fd3ff;background:#12303f;color:#d8f6ff;font:14px monospace}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
const FLOORS = 6, MOVE_TICKS = 3;
const DOOR_TICKS = $door_ticks;
const NEAREST = $nearest;
const LOST = $lost;
const PRESSES = {5: [$f1], 12: [$f2], 30: [$f3]};
let car = 0, door = 'closed', doorTicks = 0, moveTicks = 0, direction = 'idle', tick = 0;
const queue = [], served = [], journal = [], clicks = [];
const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.OrthographicCamera($left, $right, $top, $bottom, 0.1, 100);
camera.position.set(...$camera);
camera.lookAt(...$look);
scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const wall = new THREE.Mesh(new THREE.PlaneGeometry(9, 12),
  new THREE.MeshStandardMaterial({color: 0x22262b}));
wall.position.set(0.6, 5.9, -2);
scene.add(wall);
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(4, 12, 16);
scene.add(key);
const frame = new THREE.MeshStandardMaterial({color: 0x777777});
for (const x of [-2.6, 2.6]) {
  const rail = new THREE.Mesh(new THREE.BoxGeometry(0.26, 12, 1.8), frame);
  rail.position.set(x, 6, 0);
  scene.add(rail);
}
for (let f = 0; f <= FLOORS; f++) {
  const slab = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.12, 1.8), frame);
  slab.position.set(0, f * 2, 0);
  scene.add(slab);
}
const MARKS = [0x6a5a3a, 0x7a4a3a, 0x5a5a4a, 0x8a5a6a, 0x9a7a4a, 0x6a4a4a];
for (let f = 0; f < FLOORS; f++) {
  const mark = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 0.1),
    new THREE.MeshBasicMaterial({color: MARKS[f]}));
  mark.position.set(-4.6, f * 2 + 1.0, 0);
  scene.add(mark);
}
const carMesh = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.2, 1.2),
  new THREE.MeshBasicMaterial({color: 0xff8c1a}));
carMesh.name = 'car';
scene.add(carMesh);
const doorMesh = new THREE.Mesh(new THREE.BoxGeometry(0.15, 1.0, 1.2),
  new THREE.MeshBasicMaterial({color: 0xcfc3ae}));
scene.add(doorMesh);
const lamps = [];
for (let f = 0; f < FLOORS; f++) {
  const lamp = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.8, 0.2),
    new THREE.MeshBasicMaterial({color: 0x2fd3ff}));
  lamp.position.set(4.6, f * 2 + 1.0, 0);
  lamps.push(lamp);
  scene.add(lamp);
}
const bar = document.createElement('div');
bar.id = 'bar';
const buttons = [];
for (let f = FLOORS - 1; f >= 0; f--) {
  const btn = document.createElement('button');
  btn.dataset.floor = String(f);
  btn.textContent = String(f);
  btn.addEventListener('click', () => press(f));
  buttons[f] = btn;
  bar.appendChild(btn);
}
document.body.appendChild(bar);
const ON = new THREE.Color(0xffd27f), OFF = new THREE.Color(0x2fd3ff);
function press(f) {
  journal.push(tick + ':press ' + f);
  clicks.push({tick: tick, floor: f});
  if (LOST || queue.includes(f)) { journal.push(tick + ':ignored ' + f); return; }
  queue.push(f);
}
function paint() {
  carMesh.position.set(0, car * 2 + 0.9, 0);
  doorMesh.position.set(1.3, car * 2 + 0.9, 0);
  for (let f = 0; f < FLOORS; f++) lamps[f].material.color.copy(queue.includes(f) ? ON : OFF);
}
function serve(target) {
  const at = queue.indexOf(target);
  if (at >= 0) queue.splice(at, 1);
  door = 'opening';
  doorTicks = 0;
  direction = 'idle';
  served.push(car);
  journal.push(tick + ':arrive ' + car, tick + ':door opening');
}
function advance() {
  if (door === 'opening') {
    doorTicks += 1;
    if (doorTicks >= DOOR_TICKS) { door = 'open'; doorTicks = 0; journal.push(tick + ':door open'); }
  } else if (door === 'open') {
    doorTicks += 1;
    if (doorTicks >= DOOR_TICKS) { door = 'closing'; doorTicks = 0; journal.push(tick + ':door closing'); }
  } else if (door === 'closing') {
    doorTicks += 1;
    if (doorTicks >= DOOR_TICKS) { door = 'closed'; doorTicks = 0; journal.push(tick + ':door closed'); }
  } else if (queue.length) {
    let target = queue[0];
    if (NEAREST) target = queue.reduce((best, f) => Math.abs(f - car) < Math.abs(best - car) ? f : best, queue[0]);
    if (target === car) { serve(target); return; }
    direction = target > car ? 'up' : 'down';
    moveTicks += 1;
    if (moveTicks >= MOVE_TICKS) {
      moveTicks = 0;
      car += target > car ? 1 : -1;
      journal.push(tick + ':move ' + car);
      if (car === target) serve(target);
    }
  } else direction = 'idle';
}
window.__step = (dt) => {
  tick += 1;
  for (const f of PRESSES[tick] || []) buttons[f].click();
  advance();
};
function visibleButtons() {
  const vw = window.innerWidth, vh = window.innerHeight, out = [];
  for (let f = 0; f < FLOORS; f++) {
    const r = buttons[f].getBoundingClientRect();
    if (r.width > 0 && r.left >= 0 && r.top >= 0 && r.right <= vw && r.bottom <= vh) out.push(f);
  }
  return out;
}
window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  return {floors: FLOORS, car, door, door_ticks: doorTicks, move_ticks: moveTicks, direction,
    queue: queue.slice(), served: served.slice(), journal: journal.slice(), clicks: clicks.slice(),
    tick, buttons: visibleButtons()};
};
paint();
renderer.render(scene, camera);
window.__ready = true;
</script>
</body></html>
""")


# --------------------------------------------------------------------------- #
# Python replica of the state machine
# --------------------------------------------------------------------------- #


def lift(
    presses: dict[int, tuple[int, ...]],
    ticks: int,
    *,
    nearest: bool = False,
    lost: bool = False,
    door_ticks: int = DOOR_TICKS,
) -> dict:
    """Run the stated machine for ``ticks`` ticks and return the probe payload.

    ``nearest`` picks the closest queued floor instead of the first one,
    ``lost`` drops every press, ``door_ticks`` is the door phase length — the
    three defects the test page turns on.
    """
    car, door, door_ticks_left, move_ticks = 0, "closed", 0, 0
    direction, tick = "idle", 0
    queue: list[int] = []
    served: list[int] = []
    journal: list[str] = []
    clicks: list[dict] = []

    for _ in range(ticks):
        tick += 1
        for floor in sorted(presses.get(tick, ())):
            journal.append(f"{tick}:press {floor}")
            clicks.append({"tick": tick, "floor": floor})
            if lost or floor in queue:
                journal.append(f"{tick}:ignored {floor}")
            else:
                queue.append(floor)

        if door == "opening":
            door_ticks_left += 1
            if door_ticks_left >= door_ticks:
                door, door_ticks_left = "open", 0
                journal.append(f"{tick}:door open")
        elif door == "open":
            door_ticks_left += 1
            if door_ticks_left >= door_ticks:
                door, door_ticks_left = "closing", 0
                journal.append(f"{tick}:door closing")
        elif door == "closing":
            door_ticks_left += 1
            if door_ticks_left >= door_ticks:
                door, door_ticks_left = "closed", 0
                journal.append(f"{tick}:door closed")
        elif door == "closed" and queue:
            target = queue[0]
            if nearest:
                target = min(queue, key=lambda floor: (abs(floor - car), queue.index(floor)))
            if target == car:
                queue.remove(target)
                door, door_ticks_left, direction = "opening", 0, "idle"
                served.append(car)
                journal += [f"{tick}:arrive {car}", f"{tick}:door opening"]
            else:
                direction = "up" if target > car else "down"
                move_ticks += 1
                if move_ticks >= MOVE_TICKS:
                    move_ticks = 0
                    car += 1 if target > car else -1
                    journal.append(f"{tick}:move {car}")
                    if car == target:
                        queue.remove(target)
                        direction = "idle"
                        served.append(car)
                        door, door_ticks_left = "opening", 0
                        journal += [f"{tick}:arrive {car}", f"{tick}:door opening"]
        else:
            direction = "idle"

    return {
        "floors": FLOORS,
        "car": car,
        "door": door,
        "door_ticks": door_ticks_left,
        "move_ticks": move_ticks,
        "direction": direction,
        "queue": queue,
        "served": served,
        "journal": journal,
        "clicks": clicks,
        "tick": tick,
        "buttons": list(range(FLOORS)),
    }


def _probe_steps(presses: dict[int, tuple[int, ...]]) -> int:
    """First tick whose state separates FIFO from every mutant.

    The window starts after the last press and requires a closed door with the
    third request still queued and the car away from it: then a lost request,
    a nearest-first pick and a one-tick door all change the state.
    """
    last = presses[max(presses)]
    for tick in range(max(PRESS_TICKS) + 5, 90):
        base = lift(presses, tick)
        if base["queue"] != list(last) or base["door"] != "closed" or base["car"] == last[0]:
            continue
        if len(base["served"]) != 2:
            continue
        variants = {
            repr(lift(presses, tick, nearest=True)),
            repr(lift(presses, tick, lost=True)),
            repr(lift(presses, tick, door_ticks=1)),
            repr(base),
        }
        if len(variants) == 4:
            return tick
    raise AssertionError(f"нет тика, различающего мутантов: {presses}")


# --------------------------------------------------------------------------- #
# Predicates and frame geometry
# --------------------------------------------------------------------------- #


def _equals(path: str, value) -> visual.Predicate:
    def check(payload: dict) -> tuple[bool, str]:
        try:
            actual = visual.path_of(payload, path)
        except Exception as exc:
            return False, f"{path}: {exc}"
        good = actual == value
        return good, f"{path} = {actual!r} (ожидалось {value!r})"

    return visual._label(check, f"{path} == {value!r}")


def _assert_in_frame(points, label: str) -> None:
    """Orthographic frame, with a margin against the edges."""
    left, right, top, bottom = -8.0, 8.0, 13.0, -1.0  # world frame
    mx, my = (right - left) * 0.08, (top - bottom) * 0.06
    outside = [
        (x, y)
        for x, y in points
        if not (left + mx <= x <= right - mx and bottom + my <= y <= top - my)
    ]
    assert not outside, f"{label}: {len(outside)} точек вне кадра, например {outside[:2]}"


def _box_points(x0: float, x1: float, y0: float, y1: float) -> list[tuple[float, float]]:
    return [(x, y) for x in (x0, x1) for y in (y0, y1)]


def _assert_literals(prompt: str, *values: str) -> None:
    missing = [value for value in values if value not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #


def _task_lift(rng: random.Random) -> core.TaskInstance:
    first = int(rng.choice([4, 5]))
    second = int(rng.choice([0, 1]))
    third = int(rng.choice([2, 3]))
    presses = {5: (first,), 12: (second,), 30: (third,)}
    steps = _probe_steps(presses)
    final = lift(presses, steps)

    assert final["queue"] == [third], "к проверке третий вызов обязан ждать в очереди"
    assert final["served"] == [first, second], "первые два вызова обязаны быть обслужены по FIFO"
    assert final["door"] == "closed" and final["car"] != third, "кабина должна быть в пути"
    assert [c["tick"] for c in final["clicks"]] == [5, 12, 30], "клики обязаны быть ровно в 5, 12, 30"
    assert final["journal"] == [f"{t}:{event}" for t, event in _events(final)]
    nearest = lift(presses, steps, nearest=True)
    assert (nearest["queue"], nearest["served"], nearest["journal"]) != (
        final["queue"], final["served"], final["journal"]
    ), "nearest-first обязан переставить вызовы"
    assert lift(presses, steps, lost=True)["served"] == [], "потеря запроса обязана оставить кабину без вызовов"
    assert lift(presses, steps, door_ticks=1)["journal"] != final["journal"], "раннее закрытие обязано сдвинуть дверь"

    prompt = PROMPT.format(
        left=ORTHO[0],
        right=ORTHO[1],
        top=ORTHO[2],
        bottom=ORTHO[3],
        cam=list(CAMERA),
        look=list(LOOK),
        floors=FLOORS,
        top_floor=FLOORS - 1,
        t5=PRESS_TICKS[0],
        t12=PRESS_TICKS[1],
        t30=PRESS_TICKS[2],
        f1=first,
        f2=second,
        f3=third,
        door_ticks=DOOR_TICKS,
        move_ticks=MOVE_TICKS,
        fields=FIELDS,
        steps=steps,
    )
    golden = GOLDEN.substitute(
        door_ticks=DOOR_TICKS,
        nearest="false",
        lost="false",
        f1=first,
        f2=second,
        f3=third,
        left=ORTHO[0],
        right=ORTHO[1],
        top=ORTHO[2],
        bottom=ORTHO[3],
        camera=list(CAMERA),
        look=list(LOOK),
    )
    _assert_literals(
        prompt,
        str(list(CAMERA)),
        f"({ORTHO[0]}, {ORTHO[1]}, {ORTHO[2]}, {ORTHO[3]}, 0.1, 100)",
        str(list(LOOK)),
        "этажные перегородки",
        f"в тики 5, 12 и 30",
        f"ровно этих этажей: {first}, {second} и {third}",
        str(steps),
    )
    _assert_in_frame(_box_points(-2.6, 2.6, 0.0, 12.0), "K-28: шахта")
    _assert_in_frame(_box_points(3.7, 4.7, 1.0, 11.0), "K-28: кнопки")
    _assert_in_frame(_box_points(-1.2, 1.2, 0.3, 11.7), "K-28: кабина")
    _assert_in_frame(_box_points(-3.9, 5.1, -0.1, 11.9), "K-28: стена")
    _assert_in_frame(_box_points(-4.9, 5.15, 0.6, 11.4), "K-28: маркировка")

    scene_checks = [
        _equals("floors", FLOORS),
        _equals("car", 0),
        _equals("door", "closed"),
        _equals("direction", "idle"),
        _equals("tick", 0),
        visual.contains_all("queue", []),
        visual.contains_all("buttons", list(range(FLOORS))),
        visual.size_between("journal", 0, 0),
    ]
    sim_checks = [
        _equals("tick", steps),
        _equals("car", final["car"]),
        _equals("door", final["door"]),
        _equals("door_ticks", final["door_ticks"]),
        _equals("move_ticks", final["move_ticks"]),
        _equals("direction", final["direction"]),
        _equals("queue", final["queue"]),
        _equals("served", final["served"]),
        _equals("clicks", final["clicks"]),
        _equals("journal", final["journal"]),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.10),
        visual.image_min("distinct_colors", 10),
        visual.image_color("warm", 0.006, "кабина тёплого цвета различима"),
        visual.image_centroid("cool", 0, 0.62, 0.98, "колонна индикаторов справа"),
        visual.image_color("bright", 0.02, "индикаторы и кабина освещены"),
    ]
    return visual.build_task(
        task_id="K-28",
        title="K-28 — лифт: очередь вызовов, точный журнал",
        prompt=prompt,
        spec=visual.VisualSpec(
            scene=scene_checks,
            simulation=sim_checks,
            image=image_checks,
            steps=steps,
            dt=1 / 60,
        ),
        golden=golden,
        rng_meta={
            "presses": {"5": first, "12": second, "30": third},
            "steps": steps,
            "car": final["car"],
            "door": final["door"],
            "queue": final["queue"],
            "served": final["served"],
            "journal": final["journal"],
            "clicks": final["clicks"],
        },
    )


def _events(final: dict) -> list[tuple[int, str]]:
    """The final journal split back into ``(tick, event)`` for the assertion."""
    out = []
    for entry in final["journal"]:
        tick, _, event = entry.partition(":")
        out.append((int(tick), event))
    return out


BUILDERS = (_task_lift,)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Every scenario of the slice, drawing its fixture from the shared stream."""
    return [builder(rng) for builder in BUILDERS]
