"""Visual scenario K-27 — сеть FIFO-каналов с задержками и пропускной способностью.

Same shape as ``visual_fields``: the prompt states the scene, the exact tick
order, the exact channel table and the exact probe fields; ``build()`` draws the
arrival table and the TTL from ``rng`` and renders *both* the prompt and the
reference page from them, and the expected counters are recomputed here in
Python from the stated rule.

The scheme is integer-only on purpose: packets are counted objects, every
counter is an exact integer, and only the marker coordinates are real numbers.
The three properties a wrong implementation would break are spelled out as
mutants of the Python replica (a branch that duplicates instead of alternating,
a full channel that loses instead of waiting, a node that serves its queue
back-to-front), so the published check has something to discriminate.

The fixture is checked in four frames — ticks 0, 40, 80 and 120 — and the page
records them by itself at exactly those ticks, because the harness samples the
probe at one point only.
"""

from __future__ import annotations

import random
from string import Template

from .. import core, visual

IDS = ("K-27",)

#: The pilot profile gets one scenario; the wide profile gets the whole slice.
IDS_BY_PROFILE = {"pilot": ("K-27",), "stretch-v1": IDS}

#: Ticks the page snapshots the queue lengths at, and the tick the run stops at.
FRAMES = (0, 40, 80, 120)
TICKS = 120

#: Nodes in routing order. The routing stage walks exactly this order.
NODES = ("S", "B", "L", "R", "M")
SOURCE, BRANCH, LEFT, RIGHT, MERGE = range(5)

#: Where the nodes stand in the orthographic frame; the probe reports them so a
#: shifted layout is caught to 1e-3.
POSITIONS = ((-8.0, 3.0), (-3.0, 3.0), (0.0, 6.2), (0.0, -0.2), (6.5, 3.0))

#: Node queue capacities, same order as ``NODES``.
CAPACITIES = (6, 4, 3, 3, 3)

#: Channels as ``(from, to, delay, capacity)``; the index is the channel id.
CHANNELS = (
    (SOURCE, BRANCH, 3, 3),
    (BRANCH, LEFT, 3, 3),
    (BRANCH, RIGHT, 4, 3),
    (LEFT, MERGE, 3, 3),
    (RIGHT, MERGE, 3, 3),
)

#: Packets already in the network at tick 0: queue lengths and channel loads.
#: The first frame is a populated network, not an empty one — the image check has
#: something to look at from the very first render.
PRELOAD_QUEUES = (3, 0, 1, 1, 1)
PRELOAD_CHANNELS = ((2,), (3, 2), (4,), (3, 2), (3,))

#: Orthographic frame: ``left, right, top, bottom``.
FRAME = (-10.0, 10.0, 7.5, -7.5)
CAMERA = (0.0, 0.0, 50.0)

FIELDS = """{
  "offered": <принято заявок на входе за всё время>,
  "created": <пакетов принято в сеть>,
  "delivered": <пакетов доставлено в сток M>,
  "in_flight": <пакетов сейчас в каналах>,
  "dropped": <пакетов потеряно на полном входе S>,
  "expired": <пакетов истекло по TTL>,
  "queues": [<длина очереди S>, <B>, <L>, <R>, <M>],
  "frames": [[...5 длин на тике 0...], [...на тике 40...], [...на тике 80...], [...на тике 120...]],
  "markers": [[x, y] каждого из пяти узлов],
  "balance": <true, если offered = delivered + in_flight + expired + dropped + сумма queues>
}"""

PROMPT = Template("""Собери сцену сети FIFO-каналов между пятью узлами.

Сцена:
* пять узлов `S`, `B`, `L`, `R`, `M` в точках $positions (порядок именно
  такой: S — вход, B — развилка, L и R — промежуточные, M — сток);
* узел — диск радиуса 0.55 в плоскости XY, цвет по роли: S 0x2f5fa8, B 0x8a6bd1,
  L и R 0x2f8f7a, M 0xc9762e; у каждого узла маркер — куб 0.55x0.55x0.2 цвета
  0xff3bd0 ровно в точке узла;
* каждый канал — отрезок между узлами (толщина 0.16, цвет 0x24425e), пакет —
  куб 0.28 цвета 0xffb347;
* камера ортографическая и фиксирована:
  `OrthographicCamera(-10, 10, 7.5, -7.5, 0.1, 100)` в точке $camera и смотрит
  в (0, 0, 0), так что мир накрывает кадр один в один: точка `(x, y)` попадает
  в пиксель `((x + 10) / 20 * 800, (7.5 - y) / 15 * 600)`.

Сеть:
* узлы `(S, B, L, R, M)` идут в этом порядке и при маршрутизации, и в `queues`;
* каналы перечислены по индексам: 0 — `S->B`, задержка 3, ёмкость 3;
  1 — `B->L`, задержка 3, ёмкость 3; 2 — `B->R`, задержка 4, ёмкость 3;
  3 — `L->M`, задержка 3, ёмкость 3; 4 — `R->M`, задержка 3, ёмкость 3;
* ёмкости очередей узлов: S 6, B 4, L 3, R 3, M 3; узел M очереди не имеет
  выхода наружу — голова его очереди доставляется сразу;
* маршрут узла фиксирован: S -> канал 0, L -> канал 3, R -> канал 4; B
  чередует каналы: первый ушедший из B пакет идёт в 1, следующий в 2, затем
  снова в 1 (счётчик чередования растёт только на фактически ушедшем пакете);
* если целевой канал полон, пакет остаётся в голове очереди и ждёт — потерь
  при маршрутизации нет; голова очереди уходит не более одного пакета за тик;
* начальное состояние на тике 0 уже непустое, `born = 0` у всех этих пакетов:
  в очередях S 3, L 1, R 1, M 1 пакет; в каналах 0 — один пакет с
  `remaining = 2`, 1 — два пакета с `remaining = 3` и `2`, 2 — один с
  `remaining = 4`, 3 — два с `remaining = 3` и `2`, 4 — один с `remaining = 3`;
  `created` и `offered` стартуют с 13;
* TTL пакета — $ttl тиков с момента приёма; на тике `t` истекает пакет с
  `t - born >= $ttl`, где бы он ни был — в очереди узла или в канале.

Симуляция — ровно $ticks тиков, ровно в таком порядке внутри тика:
1. истечения: из всех пяти очередей и из всех пяти каналов удаляются пакеты с
   `t - born >= $ttl`, порядок обхода — NODES, затем каналы 0..4;
2. маршрутизация: для каждого узла в порядке NODES, если очередь непуста,
   рассматривается её голова; голова M доставляется и убирается; иначе, если
   первый по правилу узла канал свободен, голова уходит в него с
   `remaining = delay`, иначе (и только тогда) пробуются остальные каналы узла
   по порядку; занятой она остаётся, если свободных каналов нет;
3. FIFO-шаг: для каждого канала 0..4, если он непуст, у головы
   `remaining -= 1`; при `remaining <= 0` пакет переходит в хвост очереди
   получателя, если она не заполнена, иначе остаётся головой с
   `remaining = 0` (заторы копятся в канале, не теряются);
4. прибытия: заявки таблицы этого тика встают в хвост S, пока там есть место,
   остальные теряются и считаются в `dropped`.

Таблица прибытий `(тик, сколько заявок)` — остальные тики пустые:

$arrivals

Проба: состояние пишется вызовом `window.__probe()` — сначала `paint()`,
потом `renderer.render(scene, camera)`, и только затем возвращаются поля;
`window.__step(dt)` состояние продвигает на один тик и ничего не рисует.
Длины очередей `S`, `B`, `L`, `R`, `M` дописываются в `frames` на тиках
$frame_list (в этот момент, после шага тика); `offered` — сумма прибытий до
текущего тика включительно. `window.__probe()` возвращает ровно такие поля:

$fields

Пакет рисуется в канале с прогрессом `(delay - remaining) / delay` по отрезку
от узла-источника к узлу-получателю, пакеты очереди — столбиком под узлом с
шагом 0.3, начиная с `y - 0.4`. Все счётчики — целые; координаты маркеров
сверяются с допуском 1e-3, счётчики — точно.
""")


GOLDEN = Template("""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>fifo channel network</title>
<style>html,body{margin:0;background:#0a0e14;overflow:hidden}canvas{display:block}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';

const NODES = ['S', 'B', 'L', 'R', 'M'];
const POS = $positions, CAPS = $caps, CH = $channels;
const TTL = $ttl, FRAMES = $frames, ARRIVALS = $arrivals;
const PRELOAD_Q = $preload_q, PRELOAD_C = $preload_c;

const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1); renderer.setSize(800, 600, false);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e14);
const camera = new THREE.OrthographicCamera(-10, 10, 7.5, -7.5, 0.1, 100);
camera.position.set(...$camera); camera.lookAt(0, 0, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const unit = new THREE.PlaneGeometry(1, 1);
const ROLE = [0x2f5fa8, 0x8a6bd1, 0x2f8f7a, 0x2f8f7a, 0xc9762e];
NODES.forEach((name, i) => {
  const disc = new THREE.Mesh(unit, new THREE.MeshBasicMaterial({color: ROLE[i]}));
  disc.position.set(POS[i][0], POS[i][1], 0);
  disc.scale.set(1.1, 1.1, 1);
  disc.name = 'node-' + name;
  scene.add(disc);
  const marker = new THREE.Mesh(unit, new THREE.MeshBasicMaterial({color: 0xff3bd0}));
  marker.position.set(POS[i][0], POS[i][1], 1);
  marker.scale.set(0.55, 0.55, 1);
  marker.name = 'marker-' + name;
  scene.add(marker);
  for (let k = 0; k < CAPS[i]; k++) {
    const slot = new THREE.Mesh(unit, new THREE.MeshBasicMaterial({color: 0x14202c}));
    slot.position.set(POS[i][0], POS[i][1] - 0.4 - 0.3 * k, 0);
    slot.scale.set(0.34, 0.2, 1);
    scene.add(slot);
  }
});
for (const [a, b] of CH) {
  const [ax, ay] = POS[a], [bx, by] = POS[b];
  const mesh = new THREE.Mesh(unit, new THREE.MeshBasicMaterial({color: 0x24425e}));
  mesh.position.set((ax + bx) / 2, (ay + by) / 2, -1);
  mesh.rotation.z = Math.atan2(by - ay, bx - ax);
  mesh.scale.set(Math.hypot(bx - ax, by - ay), 0.16, 1);
  scene.add(mesh);
}
const packets = new THREE.Group();
scene.add(packets);
const dot = new THREE.PlaneGeometry(0.28, 0.28);
const dotMat = new THREE.MeshBasicMaterial({color: 0xffb347});
const waitMat = new THREE.MeshBasicMaterial({color: 0xd98a2b});
const queues = NODES.map(() => []), channels = CH.map(() => []);
PRELOAD_Q.forEach((count, node) => { for (let k = 0; k < count; k++) queues[node].push({born: 0, wait: 0}); });
PRELOAD_C.forEach((remaining, id) => { for (const wait of remaining) channels[id].push({born: 0, wait}); });
let tick = 0, alt = 0, expired = 0, delivered = 0, dropped = 0;
let created = queues.reduce((s, q) => s + q.length, 0) + channels.reduce((s, l) => s + l.length, 0);
let offered = created;
const frames = [queues.map((q) => q.length)];

function expire(list) {
  let kept = 0;
  for (const packet of list) {
    if (tick - packet.born < TTL) list[kept++] = packet;
    else expired += 1;
  }
  list.length = kept;
}
function routeOrder(node) {
  if (node === 0) return [0];
  if (node === 1) return alt % 2 === 0 ? [1, 2] : [2, 1];
  if (node === 2) return [3];
  if (node === 3) return [4];
  return [];
}
window.__step = (dt) => {
  tick += 1;
  for (const q of queues) expire(q);
  for (const list of channels) expire(list);
  for (let node = 0; node < NODES.length; node++) {
    const q = queues[node];
    if (!q.length) continue;
    const head = q[0];
    if (node === 4) { q.shift(); delivered += 1; continue; }
    let target = -1;
    for (const id of routeOrder(node)) if (channels[id].length < CH[id][3]) { target = id; break; }
    if (target < 0) continue;
    q.shift();
    channels[target].push({born: head.born, wait: CH[target][2]});
    if (node === 1) alt += 1;
  }
  for (let id = 0; id < channels.length; id++) {
    const list = channels[id];
    if (!list.length) continue;
    const head = list[0];
    head.wait -= 1;
    if (head.wait > 0) continue;
    const dst = queues[CH[id][1]];
    if (dst.length < CAPS[CH[id][1]]) { list.shift(); dst.push(head); }
    else head.wait = 0;
  }
  const load = ARRIVALS[tick] || 0;
  offered += load;
  for (let k = 0; k < load; k++) {
    if (queues[0].length >= CAPS[0]) { dropped += 1; continue; }
    queues[0].push({born: tick, wait: 0});
    created += 1;
  }
  if (FRAMES.includes(tick)) frames.push(queues.map((q) => q.length));
};
const round = (v) => Math.round(v * 1e4) / 1e4;
function paint() {
  for (const child of [...packets.children]) packets.remove(child);
  const add = (x, y, mat) => {
    const mesh = new THREE.Mesh(dot, mat);
    mesh.position.set(x, y, 0);
    packets.add(mesh);
  };
  channels.forEach((list, id) => {
    const [a, b, delay] = CH[id];
    for (const packet of list) {
      const f = (delay - packet.wait) / delay;
      add(POS[a][0] + (POS[b][0] - POS[a][0]) * f, POS[a][1] + (POS[b][1] - POS[a][1]) * f, dotMat);
    }
  });
  queues.forEach((q, node) => {
    for (let k = 0; k < q.length; k++) add(POS[node][0], POS[node][1] - 0.4 - 0.3 * k, waitMat);
  });
}
window.__probe = () => {
  paint();
  renderer.render(scene, camera);
  const queued = queues.reduce((s, q) => s + q.length, 0);
  const flying = channels.reduce((s, l) => s + l.length, 0);
  return {
    offered, created, delivered, in_flight: flying, dropped, expired,
    queues: queues.map((q) => q.length),
    frames,
    markers: POS.map((p) => [round(p[0]), round(p[1])]),
    balance: offered === delivered + flying + expired + dropped + queued,
  };
};
paint();
renderer.render(scene, camera); window.__ready = true;
</script>
</body></html>""")


# --------------------------------------------------------------------------- #
# Python replica of the stated scheme
# --------------------------------------------------------------------------- #


def arrivals_for(seed: int, ticks: int = TICKS) -> list[int]:
    """The arrival table the fixture hands out: ``arrivals[t]`` заявок на тике t."""
    rng = random.Random(seed)
    table = [0] * (ticks + 1)
    t = 1
    while t <= ticks - 10:
        count = rng.choice([0, 1, 1, 2, 2])
        if count:
            table[t] = count
        t += rng.choice([1, 2, 2, 3])
    return table


def replica(
    arrivals: list[int],
    ttl: int,
    *,
    dup: bool = False,
    lose: bool = False,
    lifo: bool = False,
    frames: tuple[int, ...] = FRAMES,
    ticks: int = TICKS,
) -> dict:
    """The scheme, exactly as stated. ``dup``/``lose``/``lifo`` are its mutants.

    ``dup`` duplicates a branch packet into both free channels, ``lose`` drops a
    packet whose channels are all full instead of letting it wait, ``lifo``
    serves the tail of a node queue instead of its head.
    """
    queues: list[list] = [[] for _ in NODES]
    channels: list[list] = [[] for _ in CHANNELS]
    offered = created = delivered = dropped = expired = 0
    alt = 0
    for node, count in enumerate(PRELOAD_QUEUES):
        for _ in range(count):
            queues[node].append([0, 0, 0])
    for channel, remaining in enumerate(PRELOAD_CHANNELS):
        for wait in remaining:
            channels[channel].append([0, 0, wait])
    created = offered = sum(PRELOAD_QUEUES) + sum(len(r) for r in PRELOAD_CHANNELS)
    history = [tuple(len(q) for q in queues)]

    for tick in range(1, ticks + 1):
        for queue in queues:
            keep = [p for p in queue if tick - p[0] < ttl]
            expired += len(queue) - len(keep)
            queue[:] = keep
        for queue in channels:
            keep = [p for p in queue if tick - p[0] < ttl]
            expired += len(queue) - len(keep)
            queue[:] = keep

        for node, queue in enumerate(queues):
            if not queue:
                continue
            index = -1 if lifo else 0
            packet = queue[index]
            if node == MERGE:
                queue.pop(index)
                delivered += 1
                continue
            if node == SOURCE:
                order = [0]
            elif node == BRANCH:
                pair = [1, 2] if alt % 2 == 0 else [2, 1]
                order = pair if not dup else [1, 2]
            elif node == LEFT:
                order = [3]
            else:
                order = [4]
            free = [c for c in order if len(channels[c]) < CHANNELS[c][3]]
            if not free:
                if lose:
                    queue.pop(index)
                    dropped += 1
                continue
            if dup and node == BRANCH and len(free) == 2:
                queue.pop(index)
                for channel in free:
                    channels[channel].append([packet[0], tick, CHANNELS[channel][2]])
            else:
                channel = free[0]
                queue.pop(index)
                channels[channel].append([packet[0], tick, CHANNELS[channel][2]])
            if node == BRANCH:
                alt += 1

        for index, queue in enumerate(channels):
            if not queue:
                continue
            head = queue[0]
            head[2] -= 1
            if head[2] <= 0:
                target = queues[CHANNELS[index][1]]
                if len(target) < CAPACITIES[CHANNELS[index][1]]:
                    queue.pop(0)
                    target.append(head)
                else:
                    head[2] = 0

        load = arrivals[tick] if tick < len(arrivals) else 0
        offered += load
        take = min(load, CAPACITIES[SOURCE] - len(queues[SOURCE]))
        dropped += load - take
        for _ in range(take):
            queues[SOURCE].append([tick, tick, 0])
            created += 1

        if tick in frames:
            history.append(tuple(len(q) for q in queues))

    in_flight = sum(len(queue) for queue in channels)
    return {
        "offered": offered,
        "created": created,
        "delivered": delivered,
        "in_flight": in_flight,
        "dropped": dropped,
        "expired": expired,
        "queues": [len(q) for q in queues],
        "frames": history,
        "balance": offered == delivered + in_flight + expired + dropped + sum(len(q) for q in queues),
    }


def _counters(path: str, value: int):
    return visual.near(path, value, 0)


def _true(path: str) -> visual.Predicate:
    """The flag at ``path`` is the boolean ``true``."""

    def check(payload: dict) -> tuple[bool, str]:
        actual = visual.path_of(payload, path, None)
        return actual is True, f"{path} = {actual!r}"

    return visual._label(check, f"{path} истинно")


def initial_probe() -> dict:
    """The tick-0 state: the preload, nothing delivered and nothing expired."""
    return {
        "offered": sum(PRELOAD_QUEUES) + sum(len(r) for r in PRELOAD_CHANNELS),
        "created": sum(PRELOAD_QUEUES) + sum(len(r) for r in PRELOAD_CHANNELS),
        "delivered": 0,
        "in_flight": sum(len(r) for r in PRELOAD_CHANNELS),
        "dropped": 0,
        "expired": 0,
        "queues": list(PRELOAD_QUEUES),
        "frames": [list(PRELOAD_QUEUES)],
    }


def _markers() -> list[visual.Predicate]:
    return [
        visual.near(f"markers.{i}.{axis}", POSITIONS[i][axis], 1e-3)
        for i in range(len(NODES))
        for axis in (0, 1)
    ]


def _frame_checks(frames: list[tuple[int, ...]]) -> list[visual.Predicate]:
    return [
        visual.near(f"frames.{index}.{node}", frames[index][node], 0)
        for index in range(1, len(frames))
        for node in range(len(NODES))
    ]


# --------------------------------------------------------------------------- #
# Frame geometry
# --------------------------------------------------------------------------- #

FRAME_MARGIN = 0.9


def _inside(x: float, y: float, margin: float = FRAME_MARGIN) -> bool:
    left, right, top, bottom = FRAME
    width = right - left
    height = top - bottom
    return (
        left + (1 - margin) * width / 2 <= x <= right - (1 - margin) * width / 2
        and bottom + (1 - margin) * height / 2 <= y <= top - (1 - margin) * height / 2
    )


def _assert_in_frame() -> None:
    """Nodes, their markers and the deepest queue stack stay inside the frame."""
    for index, (x, y) in enumerate(POSITIONS):
        assert _inside(x, y), f"{NODES[index]}: узел вне кадра"
        deepest = y - 0.4 - 0.3 * (CAPACITIES[index] - 1)
        assert _inside(x, deepest, 0.98), f"{NODES[index]}: очередь выходит из кадра"


def _assert_literals(prompt: str, *values: str) -> None:
    missing = [v for v in values if v not in prompt]
    assert not missing, f"в промпте нет чисел фикстуры: {missing}"


# --------------------------------------------------------------------------- #
# The scenario
# --------------------------------------------------------------------------- #


def _task_network(rng: random.Random) -> core.TaskInstance:
    seed = int(rng.randrange(1, 1 << 31))
    ttl = int(rng.choice([22, 24, 26]))
    arrivals = arrivals_for(seed)
    state = replica(arrivals, ttl)
    start = initial_probe()
    offered = state["offered"]

    assert state["balance"], "баланс пакетов обязан сходиться"
    assert state["delivered"] > 3, "часть пакетов обязана дойти до стока"
    assert state["expired"] > 3, "часть пакетов обязана истечь по TTL"
    assert state["dropped"] > 3, "часть заявок обязана упереться в полный вход S"
    assert len(state["frames"]) == len(FRAMES), "кадры снимаются на всех тиках"
    assert state["frames"][0] != state["frames"][-1], "очереди обязаны жить"
    assert state["frames"][0] == PRELOAD_QUEUES, "нулевой кадр — это начальная загрузка"

    table = "\n".join(
        f"{tick}: {count}" for tick, count in enumerate(arrivals) if count
    )
    prompt = PROMPT.substitute(
        positions=", ".join(f"({x}, {y})" for x, y in POSITIONS),
        camera=list(CAMERA),
        ttl=ttl,
        ticks=TICKS,
        arrivals=table,
        frame_list=", ".join(str(frame) for frame in FRAMES),
        fields=FIELDS,
    )
    golden = GOLDEN.substitute(
        positions=[list(p) for p in POSITIONS],
        caps=list(CAPACITIES),
        channels=[list(channel) for channel in CHANNELS],
        ttl=ttl,
        ticks=TICKS,
        frames=list(FRAMES),
        arrivals=arrivals,
        preload_q=list(PRELOAD_QUEUES),
        preload_c=[list(r) for r in PRELOAD_CHANNELS],
        camera=list(CAMERA),
    )
    _assert_literals(
        prompt,
        f"({POSITIONS[0][0]}, {POSITIONS[0][1]})",
        f"(x, y)",
        "TTL пакета — %d тиков" % ttl,
        "S 6, B 4, L 3, R 3, M 3",
        "S->B`, задержка 3, ёмкость 3",
        "0/40/80/120".replace("/", ", "),
        "0xff3bd0",
        "0xffb347",
        "`created` и `offered` стартуют с 13",
    )
    _assert_in_frame()

    scene_checks = [
        _counters("offered", start["offered"]),
        _counters("created", start["created"]),
        _counters("delivered", start["delivered"]),
        _counters("in_flight", start["in_flight"]),
        _counters("dropped", start["dropped"]),
        _counters("expired", start["expired"]),
        visual.size_between("queues", len(NODES), len(NODES)),
        *[_counters(f"queues.{i}", value) for i, value in enumerate(start["queues"])],
        visual.size_between("frames", 1, 1),
        *[_counters(f"frames.0.{i}", value) for i, value in enumerate(start["frames"][0])],
        visual.size_between("markers", len(NODES), len(NODES)),
        *_markers(),
        _true("balance"),
    ]
    simulation = [
        _counters("offered", offered),
        _counters("created", state["created"]),
        _counters("delivered", state["delivered"]),
        _counters("in_flight", state["in_flight"]),
        _counters("dropped", state["dropped"]),
        _counters("expired", state["expired"]),
        *[_counters(f"queues.{i}", value) for i, value in enumerate(state["queues"])],
        visual.size_between("frames", len(FRAMES), len(FRAMES)),
        *_frame_checks(state["frames"]),
        _true("balance"),
    ]
    image_checks = [
        visual.image_min("changed_fraction", 0.02),
        visual.image_min("distinct_colors", 8),
        visual.image_color("cool", 0.012, "каналы видны как тёмная сетка"),
        visual.palette_colour("ff3bd0", 0.0006, "маркеры узлов различимы"),
        visual.palette_colour("ffb347", 0.0006, "пакеты различимы"),
        visual.image_centroid("warm", 0, 0.3, 0.7, "пакеты держатся вдоль сети"),
    ]
    return visual.build_task(
        task_id="K-27",
        title="K-27 — сеть FIFO-каналов с задержками и ёмкостями",
        prompt=prompt,
        spec=visual.VisualSpec(scene=scene_checks, simulation=simulation, image=image_checks, steps=TICKS, dt=1 / 60),
        golden=golden,
        rng_meta={
            "seed": seed,
            "ttl": ttl,
            "arrivals": [value for value in arrivals],
            "offered": offered,
            "created": state["created"],
            "delivered": state["delivered"],
            "in_flight": state["in_flight"],
            "dropped": state["dropped"],
            "expired": state["expired"],
            "queues": state["queues"],
            "balance": state["balance"],
            "preload_queues": list(PRELOAD_QUEUES),
            "frames": [list(frame) for frame in state["frames"]],
        },
    )


BUILDERS = (_task_network,)


def build(rng: random.Random) -> list[core.TaskInstance]:
    """Every scenario of the slice, each drawing its fixture from the same stream."""
    return [builder(rng) for builder in BUILDERS]
