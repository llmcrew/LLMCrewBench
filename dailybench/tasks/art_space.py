"""Art scenarios L-25..L-29 — орбита: станция, Земля, катастрофа.

Пять сцен одного замысла: камера в космосе рядом с обитаемой станцией, под ней —
живая Земля с атмосферой, ночными городами и облаками, а на поверхности и вокруг
происходят стилизованные катастрофы: ядерные взрывы, падение метеорита,
столкновение планет, цепная реакция обломков, исход станции.

Общий приём — не «плоская картинка», а тело и свет: планета — сфера с шейдером
поверхности (континенты, огни городов на ночной стороне, облачный слой), у неё
атмосферный ореол аддитивным фрагментом, станция собрана коробками и трубами,
вспышка — аддитивная сфера, купол — расширяющееся полушарие, ударная волна —
кольцо. Сцена объёмная, камера медленно ходит, поэтому кадр читается как кадр из
фильма, а не как схема.

Баллов у арт-блока нет, поэтому проверки описывают кадр, а не судят вкус:
палитра, композиция (планета в кадре, ореол, вспышка), детализация и движение.
"""

from __future__ import annotations

import math
import random

from .. import art, core, visual

IDS = ("L-25", "L-26", "L-27", "L-28", "L-29")

TITLES = {
    "L-25": "L-25 — ядерные взрывы на Земле из космоса",
    "L-26": "L-26 — метеорит входит в атмосферу",
    "L-27": "L-27 — столкновение двух планет",
    "L-28": "L-28 — цепная реакция обломков",
    "L-29": "L-29 — станция уходит, Земля позади",
}

STEPS = {
    "L-25": 96, "L-26": 96, "L-27": 108, "L-28": 102, "L-29": 120,
}

FOV = 45.0
ASPECT = 16.0 / 9.0
LCG_A, LCG_C, LCG_M = 1103515245, 12345, 2147483648


def _share_text(share: float) -> str:
    if share >= 0.10:
        return f"{share:.0%}"
    if share >= 0.01:
        return f"{share * 100:.1f}%".replace(".0%", "%")
    return f"{share * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _lcg_text(seed: int) -> str:
    return (
        f"Случайность — только явный LCG: `lcg = (lcg * {LCG_A} + {LCG_C}) % {LCG_M}`, "
        f"`rnd() = lcg / {LCG_M}`, начальное `seed = {seed}`; никаких `Math.random()`. "
        "Две загрузки одной страницы обязаны дать побитово одинаковый кадр."
    )


LCG_JS = """const SEED = {seed};
let lcg = SEED;
const rnd = () => {{ lcg = (lcg * 1103515245 + 12345) % 2147483648; return lcg / 2147483648; }};"""

#: Общий каркас космической страницы: рендерер, свет, помощники, шейдеры Земли и
#: вспышки, постобработка (зерно, виньетка, хроматическая аберрация, bloom).
SPACE_JS = """const renderer = new THREE.WebGLRenderer({{antialias: true, preserveDrawingBuffer: true}});
renderer.setPixelRatio(1);
renderer.setSize(1920, 1080, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1920 / 1080, 0.05, 4000);
camera.position.set(0, 0.4, 6);
camera.lookAt(0, 0, -6);

// Свет: солнце сбоку и слабая подсветка, чтобы тёмная сторона не была чёрной.
const sun = new THREE.DirectionalLight(0xfff4e0, 3.6);
sun.position.set(-1, 0.35, 0.5);
scene.add(sun);
scene.add(new THREE.AmbientLight(0x2a3350, 0.9));
const rim = new THREE.DirectionalLight(0x6f8fd0, 0.6);
rim.position.set(0.6, -0.4, -0.8);
scene.add(rim);

const round4 = (v) => Math.round(v * 1e4) / 1e4;
const names = () => scene.children.map((o) => o.name).filter(Boolean);

// Финальный проход: постобработка поверх готового кадра. Дёшево (один
// полноэкранный фрагмент) и без внешних аддонов — только свой шейдер.
const postScene = new THREE.Scene();
const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const postMat = new THREE.ShaderMaterial({{
  uniforms: {{ uTex: {{value: null}}, uTime: {{value: 0}}, uGrain: {{value: 0.055}}, uAberr: {{value: 0.0016}} }},
  vertexShader: `varying vec2 vUv; void main() {{ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }}`,
  fragmentShader: `
    uniform sampler2D uTex; uniform float uTime, uGrain, uAberr;
    varying vec2 vUv;
    float hash(vec2 p) {{ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }}
    void main() {{
      // хроматическая аберрация: каналы чуть расходятся от центра
      vec2 d = (vUv - 0.5) * uAberr;
      vec3 c;
      c.r = texture2D(uTex, vUv + d).r;
      c.g = texture2D(uTex, vUv).g;
      c.b = texture2D(uTex, vUv - d).b;
      // мягкий bloom: три соседних отсчёта с весом
      vec3 b = texture2D(uTex, vUv + vec2(0.004, 0.0)).rgb
             + texture2D(uTex, vUv - vec2(0.004, 0.0)).rgb
             + texture2D(uTex, vUv + vec2(0.0, 0.006)).rgb
             + texture2D(uTex, vUv - vec2(0.0, 0.006)).rgb;
      c += max(vec3(0.0), b * 0.25 - 0.06) * 1.15;
      // зерно и виньетка
      float g = hash(vUv * vec2(1920.0, 1080.0) + uTime * 37.0) - 0.5;
      c += g * uGrain;
      float v = smoothstep(1.25, 0.35, length(vUv - 0.5));
      c *= mix(0.72, 1.0, v);
      gl_FragColor = vec4(c, 1.0);
      #include <colorspace_fragment>
    }}`,
}});
postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), postMat));
const postRT = new THREE.WebGLRenderTarget(1920, 1080, {{minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter}});

// Шейдер поверхности планеты: континенты по шуму, ночные огни городов, полярные
// шапки. Ночь определяется скалярным произведением нормали на направление солнца.
const earthMat = new THREE.ShaderMaterial({{
  uniforms: {{
    uTime: {{value: 0}},
    uLight: {{value: new THREE.Vector3(-1, 0.35, 0.5).normalize()}},
    uLand: {{value: new THREE.Color(0x2f6b46)}},
    uSea: {{value: new THREE.Color(0x13315e)}},
    uCity: {{value: new THREE.Color(0xffc46a)}},
    uDamage: {{value: 0.0}},
    uHits: {{value: [new THREE.Vector4(0.4, 0.2, 0, 1), new THREE.Vector4(-0.5, 0.35, 0, 1)]}},
  }},
  vertexShader: `varying vec3 vN; varying vec3 vP;
    void main() {{ vN = normalize(normalMatrix * normal); vP = normalize(position);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  fragmentShader: `
    uniform float uTime, uDamage; uniform vec3 uLight, uLand, uSea, uCity; uniform vec4 uHits[2];
    varying vec3 vN; varying vec3 vP;
    float hash(vec3 p) {{ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123); }}
    float noise(vec3 p) {{
      vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
      float n = mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                    mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
      return n;
    }}
    float fbm(vec3 p) {{
      float s = 0.0, a = 0.5;
      for (int i = 0; i < 5; i++) {{ s += a * noise(p); p *= 2.03; a *= 0.5; }}
      return s;
    }}
    void main() {{
      float h = fbm(vP * 2.6);
      float land = smoothstep(0.48, 0.55, h);
      vec3 base = mix(uSea, uLand, land);
      float polar = smoothstep(0.72, 0.94, abs(vP.y));
      base = mix(base, vec3(0.86, 0.9, 0.95), polar * 0.85);
      float lambert = max(0.0, dot(vN, normalize(uLight)));
      float night = 1.0 - smoothstep(0.02, 0.35, lambert);
      // огни городов: только на суше, только на ночной стороне
      float cities = step(0.52, h) * step(0.35, fbm(vP * 9.0)) * night;
      base += uCity * cities * 0.55;
      // след попаданий: тёмное пятно с тёплой каймой
      for (int i = 0; i < 2; i++) {{
        float d = distance(normalize(vP), normalize(uHits[i].xyz));
        base = mix(vec3(1.0, 0.72, 0.35), base, smoothstep(0.02, 0.16, d));
        base = mix(base, vec3(0.16, 0.09, 0.07), smoothstep(0.10, 0.34, d) * uDamage);
      }}
      base *= mix(0.25, 1.25, lambert);
      gl_FragColor = vec4(base, 1.0);
      #include <colorspace_fragment>
    }}`,
}});

// Атмосферный ореол: чуть большая сфера, аддитивная, светится по краю.
const atmoMat = new THREE.ShaderMaterial({{
  uniforms: {{ uColor: {{value: new THREE.Color(0x6fb6ff)}}, uLight: {{value: new THREE.Vector3(-1, 0.35, 0.5).normalize()}} }},
  vertexShader: `varying vec3 vN;
    void main() {{ vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  fragmentShader: `
    uniform vec3 uColor, uLight; varying vec3 vN;
    void main() {{
      float edge = pow(1.0 - abs(dot(vN, vec3(0.0, 0.0, 1.0))), 2.2);
      gl_FragColor = vec4(uColor * edge * 0.9, 1.0);
    }}`,
  transparent: true, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false,
}});

const sphere = (r, mat, name, seg) => {{
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, seg || 64, (seg || 64) / 2), mat);
  m.name = name;
  scene.add(m);
  return m;
}};
const box = (w, h, d, mat, name) => {{
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.name = name;
  return m;
}};
const lit = (c, o) => new THREE.MeshStandardMaterial({{color: c, roughness: 0.55, metalness: 0.25, opacity: o ?? 1, transparent: (o ?? 1) < 1}});
const glowMat = (c, o) => new THREE.MeshBasicMaterial({{color: c, transparent: true, opacity: o ?? 1, blending: THREE.AdditiveBlending, depthWrite: false}});

// Звёзды: облако точек на далёкой сфере, размер и яркость из LCG.
const starMat = new THREE.PointsMaterial({{size: 2.2, sizeAttenuation: false, transparent: true, opacity: 0.95, vertexColors: true}});
const starGeo = new THREE.BufferGeometry();
{{
  const n = 2200;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {{
    const u = rnd() * 2 - 1, th = rnd() * 6.2832, s = Math.sqrt(1 - u * u), R = 900 + rnd() * 260;
    pos[i * 3] = Math.cos(th) * s * R; pos[i * 3 + 1] = u * R; pos[i * 3 + 2] = Math.sin(th) * s * R;
    const t = 0.72 + rnd() * 0.28;
    col[i * 3] = t; col[i * 3 + 1] = t; col[i * 3 + 2] = Math.min(1, t * 1.1);
  }}
  starGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  starGeo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
}}
const stars = new THREE.Points(starGeo, starMat);
stars.name = 'stars';
scene.add(stars);

// Фильм-кадр: рисуем сцену в render target, затем натягиваем на экран через
// постобработку. Вызывается из __step вместо renderer.render.
const render = () => {{
  renderer.setRenderTarget(postRT);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  postMat.uniforms.uTex.value = postRT.texture;
  renderer.render(postScene, postCam);
}};"""


def _share_checks(prefix: str, palette: dict[str, float], notes: dict[str, str]) -> list:
    return [
        visual.palette_colour(colour, share, f"{prefix}: {notes[colour]}")
        for colour, share in palette.items()
    ]


def _dark_share(label: str, low: float):
    """Глубокий космос: доля тёмных пикселей кадра.

    Точный тон здесь не годится: звёзды и зерно постобработки разбрасывают фон,
    поэтому «космос» проверяется по доле тёмного, а не по попаданию в #05070f.
    """

    def check(pixels: dict) -> tuple[bool, str]:
        dark = float((pixels.get("dark") or {}).get("fraction") or 0.0)
        good = dark >= low
        return good, f"кадр: {label} — тёмного {dark:.4f} (нужно >= {low})"

    return visual._label(check, f"кадр: {label}")


def _planet_lower(label: str, low: float):
    """Планета занимает нижнюю половину: светлые холодные пиксели внизу кадра."""

    def check(pixels: dict) -> tuple[bool, str]:
        cool = pixels.get("cool") or {}
        centroid = cool.get("centroid")
        fraction = float(cool.get("fraction") or 0.0)
        if not isinstance(centroid, list):
            return False, f"кадр: {label} — холодных пикселей нет"
        good = fraction >= low and float(centroid[1]) >= 0.35
        return good, f"кадр: {label} — доля {fraction:.4f}, центр y {float(centroid[1]):.3f} (нужно >= {low} и низ кадра)"

    return visual._label(check, f"кадр: {label}")


def _glow_present(label: str, colour: str, low: float, tol: float = 0.22):
    """Аддитивное свечение (вспышка, ореол, след) — по доле пикселей тона."""

    def check(pixels: dict) -> tuple[bool, str]:
        histogram = pixels.get("hue_histogram") or []
        if not histogram:
            return False, "кадр: нет гистограммы"
        share = 0.0
        for hue, sat, val, fraction in histogram:
            if visual.hsv_error(colour, hue, sat, val) <= tol:
                share += float(fraction)
        good = share >= low
        return good, f"кадр: {label} — {colour} занимает {share:.4f} (нужно >= {low})"

    return visual._label(check, f"кадр: {label}")


COMMON_CHECKS = {
    "detail": [
        visual.image_min("changed_fraction", 0.25),
        visual.image_min("distinct_colors", 24),
        visual.rich_edges(0.02, "космос и станция не сливаются в заливку"),
        visual.image_min("saturation", 0.015),
        # Объём и свет: планета и корпус дают плавные перепады, а не заливку.
        visual.image_min("shade", 0.03),
    ],
}


def _build(
    *,
    task_id: str,
    prompt: str,
    golden: str,
    palette: dict[str, float],
    notes: dict[str, str],
    composition: list,
    fixture: dict,
    steps: int,
) -> core.TaskInstance:
    detail = list(COMMON_CHECKS["detail"])
    for literal in ("#", "window.__step", "window.__probe"):
        assert literal in prompt, literal
    assert golden.count("Math.random") == 0, f"{task_id}: эталон использует Math.random"
    return art.build_task(
        task_id=task_id,
        title=TITLES[task_id],
        prompt=prompt,
        spec=art.ArtSpec(
            palette=_share_checks(task_id, palette, notes),
            composition=composition,
            detail=detail,
            motion=steps,
        ),
        golden=golden,
        fixture=fixture,
    )


#: Палитра измеряется по цветным акцентам: фон-космос разъезжается от звёзд и
#: зерна, поэтому он проверяется отдельно, долей тёмных пикселей.
PALETTE_L25 = {"#6fb6ff": 0.010, "#ffd9a0": 0.30, "#ffd0d0": 0.30}
NOTES_L25 = {"#6fb6ff": "атмосферный ореол Земли", "#ffd9a0": "вспышка и раскалённое облако",
             "#ffd0d0": "огни городов на ночной стороне"}
PALETTE_L26 = {"#6fb6ff": 0.040, "#ffd0a0": 0.28, "#ffd0d0": 0.27}
NOTES_L26 = {"#6fb6ff": "атмосфера Земли", "#ffd0a0": "огненный след метеорита",
             "#ffd0d0": "огни городов ночной стороны"}
PALETTE_L27 = {"#6fb6ff": 0.023, "#ff7a3c": 0.29, "#d8b18a": 0.29}
NOTES_L27 = {"#6fb6ff": "атмосфера Земли", "#ff7a3c": "трещины лавы и обломки", "#d8b18a": "облако пыли"}
PALETTE_L28 = {"#6fb6ff": 0.017, "#7bf0c8": 0.007, "#ffb070": 0.007, "#5a4a8f": 0.31}
NOTES_L28 = {"#6fb6ff": "атмосфера Земли", "#7bf0c8": "полярное сияние и орбиты",
             "#ffb070": "раскалённые обломки", "#5a4a8f": "туманность позади"}
PALETTE_L29 = {"#5e9fe0": 0.004, "#ff8a4c": 0.32, "#8fd8ff": 0.004}
NOTES_L29 = {"#5e9fe0": "атмосфера повреждённой Земли", "#ff8a4c": "разломы и следы ударов",
             "#8fd8ff": "сопло станции"}


SPACE_TAIL = """

Требования к странице (общие для всех арт-заданий):
1. Ответ — один самодостаточный документ `<html>`, без внешних ресурсов.
   Библиотека рядом: `import * as THREE from './vendor/three.module.js'`.
2. Рендерер: `new THREE.WebGLRenderer({preserveDrawingBuffer: true})`,
   `renderer.setPixelRatio(1)`, `renderer.setSize(1920, 1080, false)`, canvas в
   body. Кадр 1920x1080 при devicePixelRatio 1.
3. Детерминизм: две загрузки дают побитово одинаковый кадр для одного момента.
   Никаких `Math.random()` без явного seed, никакой зависимости от
   `performance.now()`.
4. Время двигает только `window.__step(dt)` — ровно на `dt` секунд, и
   возвращает `window.__probe()`. Кадр после нуля шагов — начало сцены; кадр
   после шагов обязан отличаться.
5. `window.__probe()` возвращает `{"t": <4 знака>, "objects": [...]}` плюс поля,
   перечисленные в задании.
6. `window.__ready = true` — после первой отрисовки, когда `__probe()` отвечает.

Как это оценивают. Баллов у арт-блока нет — сцены смотрит человек. Проверки
только описывают кадр: палитра, композиция, детализация, движение. Поэтому форма
свободна: рисуй телами (сферы, коробки, кольца), ставь свет и материалы, строй
настоящую перспективу — важно, чтобы кадр читался объёмным и со светом, а не как
плоская заливка. Цвета задания — ориентир по тону, точный процент не требуется.
"""

# --------------------------------------------------------------------------- #
# L-25 — ядерные взрывы на Земле из космоса
# --------------------------------------------------------------------------- #

L25_PROMPT = """Создай масштабную 3D-сцену на Three.js: обитаемая орбитальная станция
на высокой орбите над Землёй. Земля реалистичная — атмосферное свечение, ночные
города, облака. На поверхности периодически происходят стилизованные ядерные
взрывы: яркая вспышка, затем расширяющийся светящийся купол, ударная волна,
грибовидное облако. Взрывы кинематографичны, но не реалистичная симуляция оружия.
Станция медленно вращается, на ней мигают огни.

Сцена (обязательные объекты с этими именами):
* `earth` — сфера радиуса 4.0 в точке (-1.2, -1.6, -14), шейдер: континенты по
  шуму, тёмно-синее море {c_sea}, зелёная суша {c_land}, белые полярные шапки,
  ночная сторона — с огнями городов {c_city};
* `atmosphere` — сфера радиуса 4.35 там же, аддитивный ореол цвета {c_atmo},
  ярче всего по краю диска (внутри — прозрачно);
* `clouds` — сфера радиуса 4.10, полупрозрачный белый слой облаков (opacity 0.30),
  медленно поворачивается;
* `station` — группа на переднем плане справа (0.62 ширины, 0.40 высоты, z = 0):
  корпус-коробка 0.9x0.30x0.30, две солнечные панели-плоскости по 0.55x0.16 по
  бокам, антенна-труба, три окна; станция вращается вокруг своей оси;
* `flash` — аддитивная сфера у поверхности Земли, растёт с 0.05 до 0.34 за 0.5 с
  и гаснет: цвет {c_flash};
* `dome` — полушарие, расширяется до 0.9 за 1.6 с, полупрозрачный {c_flash}
  (opacity 0.25), затем гаснет;
* `ring` — кольцо (`RingGeometry`) на поверхности, расходится до 1.8 за 2.4 с,
  цвета {c_flash}, opacity 0.5;
* `mushroom` — восходящий столб и шляпка из двух сфер, всплывает на 0.5 за 2 с;
* `stars` — 2200 точек на сфере радиуса ~1000, белые с разбросом яркости;
* взрывов два: первый стартует сразу, второй через 1.5 с. Точки лежат на
  поверхности, обращённой к камере, и заданы углами: направление равно
  `(sin(a)·cos(e), sin(e), cos(a)·cos(e))` при a = {boom_au:.3f} и {boom_bu:.3f},
  e = {boom_av:.3f} и {boom_bv:.3f} (a — поворот вбок, e — высота; при a = 0
  направление смотрит на камеру).

Камера: `PerspectiveCamera(45, 1920/1080, 0.05, 4000)` в (0, 0.4, 6), смотрит в
(0, 0, -6). Медленное дрожание: x колеблется на {shake} и y на {shake} * 0.6.

Постобработка поверх кадра: bloom, хроматическая аберрация, зерно, виньетка.

Анимация: `window.__step(dt)` двигает вспышку, купол, кольцо и гриб по таймингу
взрыва, крутит станцию и облака, дрожит камерой и возвращает `window.__probe()`.
`window.__probe()` возвращает `t`, `objects`, `camera: [x, y]` и `boom` — возраст
текущего взрыва (4 знака).

{lcg}

Палитра-ориентир (доли — минимум кадра, по тону с допуском):
* {c_atmo} — атмосферный ореол, не меньше {s_atmo};
* {c_flash} — вспышка и раскалённое облако, не меньше {s_flash};
* {c_city} — огни городов на ночной стороне, не меньше {s_city}.

Проверяется: палитра, планета в кадре, ореол и вспышка, детализация, движение.
"""

L25_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>orbital strikes</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + SPACE_JS + """

const EARTH = new THREE.Vector3(-1.2, -1.6, -14);
const earth = sphere(4.0, earthMat, 'earth');
earth.position.copy(EARTH);
const atmo = sphere(4.35, atmoMat, 'atmosphere');
atmo.position.copy(EARTH);
const clouds = sphere(4.10, new THREE.MeshStandardMaterial({{
  color: 0xdfe8f2, roughness: 1.0, transparent: true, opacity: 0.30, depthWrite: false,
}}), 'clouds');
clouds.position.copy(EARTH);

// Станция: корпус, панели, антенна, окна. Вращается целиком.
const station = new THREE.Group();
station.name = 'station';
station.position.set(1.6, 0.1, 0);
scene.add(station);
const hull = box(0.9, 0.30, 0.30, lit(0xb9c2cc), 'hull');
station.add(hull);
for (const sx of [-1, 1]) {{
  const panel = box(0.55, 0.02, 0.16, lit(0x2b4a8f), 'panel_' + (sx < 0 ? 'l' : 'r'));
  panel.position.set(sx * 0.72, 0, 0);
  station.add(panel);
  const strut = box(0.20, 0.03, 0.03, lit(0x8a939c), 'strut_' + (sx < 0 ? 'l' : 'r'));
  strut.position.set(sx * 0.45, 0, 0);
  station.add(strut);
}}
const mast = box(0.03, 0.34, 0.03, lit(0x9aa3ad), 'antenna');
mast.position.set(0, 0.30, 0);
station.add(mast);
for (let i = 0; i < 3; i++) {{
  const win = new THREE.Mesh(new THREE.PlaneGeometry(0.10, 0.07), glowMat(0x9fe0ff, 0.9));
  win.name = 'window' + i;
  win.position.set(0.18 + i * 0.16, 0.04, 0.16);
  station.add(win);
}}
const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), glowMat(0xff6b6b, 0.95));
beacon.name = 'beacon';
station.add(beacon);

// Взрывы: два очага на поверхности (заданы фикстурой).
const dirOf = (a, e) => new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e));
const BOOMS = [
  {{ dir: dirOf({boom_au}, {boom_av}), at: 0.0 }},
  {{ dir: dirOf({boom_bu}, {boom_bv}), at: 1.5 }},
];
const flashMat = glowMat({c_flash}, 1.0);
const flashes = [], domes = [], rings = [], shrooms = [];
for (let i = 0; i < 2; i++) {{
  const dir = BOOMS[i].dir.clone();
  const onSurface = EARTH.clone().add(dir.clone().multiplyScalar(4.05));
  const fl = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), flashMat.clone());
  fl.name = 'flash' + i; fl.position.copy(onSurface); scene.add(fl); flashes.push(fl);
  const dm = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12, 0, 6.2832, 0, Math.PI / 2), glowMat({c_flash}, 0.25));
  dm.name = 'dome' + i; dm.position.copy(onSurface); dm.lookAt(EARTH.clone().add(dir.clone().multiplyScalar(30)));
  dm.rotateX(Math.PI / 2); scene.add(dm); domes.push(dm);
  const rg = new THREE.Mesh(new THREE.RingGeometry(0.8, 1.0, 48), glowMat({c_flash}, 0.5));
  rg.name = 'ring' + i; rg.position.copy(onSurface);
  rg.lookAt(EARTH.clone().add(dir.clone().multiplyScalar(30)));
  scene.add(rg); rings.push(rg);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.11, 1, 12), glowMat({c_flash}, 0.7));
  stem.name = 'stem' + i; stem.position.copy(onSurface); scene.add(stem); shrooms.push(stem);
  const cap = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), glowMat({c_flash}, 0.55));
  cap.name = 'cap' + i; cap.position.copy(onSurface); scene.add(cap); shrooms.push(cap);
}}

earthMat.uniforms.uHits.value = BOOMS.map(b => new THREE.Vector4(b.dir.x, b.dir.y, b.dir.z, 1));

const clamp01 = (v) => Math.max(0, Math.min(1, v));
let t = 0;
window.__step = (dt) => {{
  t += dt;
  earthMat.uniforms.uTime.value = t;
  postMat.uniforms.uTime.value = t;
  clouds.rotation.y = t * 0.012;
  earth.rotation.y = t * 0.008;
  station.rotation.x = t * 0.16;
  station.rotation.y = t * 0.09;
  beacon.material.opacity = 0.35 + 0.5 * Math.abs(Math.sin(t * 3.1));
  for (let i = 0; i < 2; i++) {{
    const age = Math.max(0, t - BOOMS[i].at);
    const live = t >= BOOMS[i].at ? 1 : 0;
    // вспышка: 0.05 -> 0.34 за 0.5 с и гаснет
    const fl = clamp01(age / 0.5);
    flashes[i].scale.setScalar(live * (0.04 + 0.20 * fl));
    flashes[i].material.opacity = live * Math.max(0, 1 - age / 1.1);
    // купол: до 0.9 за 1.6 с
    const dm = clamp01(age / 1.6);
    domes[i].scale.setScalar(live * (0.10 + 0.62 * dm));
    domes[i].material.opacity = live * Math.max(0, 0.30 * (1 - dm));
    // кольцо: до 1.8 за 2.4 с
    const rg = clamp01(age / 2.4);
    rings[i].scale.setScalar(live * (0.15 + 1.65 * rg));
    rings[i].material.opacity = live * Math.max(0, 0.55 * (1 - rg));
    // гриб: столб и шляпка всплывают на 0.5 за 2 с
    const mh = clamp01(age / 2.0);
    const up = mh * 0.5;
    shrooms[i * 2].scale.set(live * 0.24, live * (0.25 + up), live * 0.24);
    shrooms[i * 2].position.y = flashes[i].position.y + 0.12 + up * 0.5;
    shrooms[i * 2].material.opacity = live * Math.max(0, 0.75 * (1 - mh));
    shrooms[i * 2 + 1].scale.setScalar(live * (0.08 + 0.24 * mh));
    shrooms[i * 2 + 1].position.y = flashes[i].position.y + 0.24 + up;
    shrooms[i * 2 + 1].material.opacity = live * Math.max(0, 0.6 * (1 - mh));
  }}
  earthMat.uniforms.uDamage.value = clamp01(t / 6);
  camera.position.x = Math.sin(t * 0.42) * {shake};
  camera.position.y = 0.4 + Math.sin(t * 0.31) * {shake} * 0.6;
  camera.lookAt(0, 0, -6);
  render();
  return window.__probe();
}};

window.__probe = () => ({{
  t: round4(t),
  objects: names(),
  camera: [round4(camera.position.x), round4(camera.position.y)],
  boom: round4(Math.max(0, t - BOOMS[0].at)),
}});

render();
window.__ready = true;
</script>
</body></html>
"""


def _l25(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    boom_au = round(rng.uniform(-0.42, -0.12), 3)
    boom_av = round(rng.uniform(0.20, 0.50), 3)
    boom_bu = round(rng.uniform(0.16, 0.46), 3)
    boom_bv = round(rng.uniform(-0.38, -0.10), 3)
    shake = round(rng.uniform(0.012, 0.030), 3)
    prompt = L25_PROMPT.format(
        c_sea="#13315e", c_land="#2f6b46", c_city="#ffd0d0", c_atmo="#6fb6ff", c_flash="#ffd9a0",
        s_atmo=_share_text(PALETTE_L25["#6fb6ff"]), s_flash=_share_text(PALETTE_L25["#ffd9a0"]),
        s_city=_share_text(PALETTE_L25["#ffd0d0"]),
        boom_au=boom_au, boom_av=boom_av, boom_bu=boom_bu, boom_bv=boom_bv,
        shake=f"{shake:.3f}", seed=seed, lcg=_lcg_text(seed),
    ) + SPACE_TAIL
    golden = L25_GOLDEN.format(
        seed=seed, c_flash="0xffd9a0",
        boom_au=boom_au, boom_av=boom_av, boom_bu=boom_bu, boom_bv=boom_bv,
        shake=f"{shake:.3f}",
    )
    for literal in (f"{boom_au:.3f}", f"{boom_av:.3f}", f"{boom_bu:.3f}", f"{boom_bv:.3f}",
                    f"{shake:.3f}", str(seed), "#13315e", "#6fb6ff", "#ffd9a0", "#ffd0d0"):
        assert literal in prompt, literal
    return _build(
        task_id="L-25", prompt=prompt, golden=golden,
        palette=PALETTE_L25, notes=NOTES_L25,
        composition=[
            _dark_share("глубокий космос вокруг", 0.30),
            _planet_lower("планета и станция в кадре", 0.030),
            _glow_present("атмосферный ореол", "#6fb6ff", 0.008),
            _glow_present("вспышка взрыва", "#ffd9a0", 0.003, 0.12),
        ],
        fixture={"seed": seed, "boom_a": [boom_au, boom_av], "boom_b": [boom_bu, boom_bv], "shake": shake},
        steps=STEPS["L-25"],
    )


# --------------------------------------------------------------------------- #
# L-26 — метеорит входит в атмосферу
# --------------------------------------------------------------------------- #

L26_PROMPT = """Создай объёмную 3D-сцену: гигантский метеорит входит в атмосферу Земли,
наблюдение с орбитальной станции. Метеорит оставляет огненный след, вокруг —
ударная волна, облака разлетаются, атмосфера светится. В момент удара — вспышка,
кольцо пыли, выброс раскалённых частиц. Станция на переднем плане, её огни
отражаются в стекле.

Сцена (имена обязательны):
* `earth` — сфера радиуса 4.0 в (-0.6, -2.0, -13), тот же шейдер поверхности, что
  в L-25: море {c_sea}, суша {c_land}, огни городов {c_city};
* `atmosphere` — ореол радиуса 4.4 цвета {c_atmo};
* `clouds` — полупрозрачный облачный слой (opacity 0.28);
* `meteor` — сфера радиуса 0.18 цвета {c_fire} на пути к планете, старт
  (1.9, 1.5, -6.0), скорость (dx = {mvx}, dy = {mvy}, dz = {mvz}) единиц в
  секунду; за ней тянется `trail` — 90 аддитивных точек цвета {c_fire},
  отстающих по пути;
* `shock` — кольцо вокруг метеорита, растёт до 0.6 за 0.4 с и повторяется;
* `impact_flash`, `impact_ring` — вспышка и кольцо пыли на поверхности, включаются
  в момент удара (t = {impact});
* `debris` — 120 раскалённых точек, разлетающихся от точки удара;
* `station` — группа слева (0.28 ширины, 0.55 высоты, z = 1.2): корпус, панели,
  иллюминатор-плоскость цвета {c_glass} с отражением вспышки (opacity 0.35);
* `stars` — 2000 точек.

Камера: смотрит на планету сбоку, медленно ведёт метеорит: позиция
({camx}, {camy}, 5.5), взгляд смещается с метеорита на точку удара.

Анимация: `__step(dt)` двигает метеорит и след, гасит след, включает удар,
разлетает обломки, возвращает `__probe()` с полями `t`, `objects`,
`meteor: [x, y, z]` (4 знака), `impact_age`.

{lcg}

Палитра-ориентир (доли — минимум кадра, по тону с допуском):
* {c_atmo} — атмосфера Земли, не меньше {s_atmo};
* {c_fire} — огненный след метеорита, не меньше {s_fire};
* {c_city} — огни городов ночной стороны, не меньше {s_city}.

Проверяется: палитра, планета в кадре, огненный след, детализация, движение.
"""

L26_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>meteor</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + SPACE_JS + """

const EARTH = new THREE.Vector3(-0.6, -2.0, -13);
const earth = sphere(4.0, earthMat, 'earth');
earth.position.copy(EARTH);
earthMat.uniforms.uHits.value = [new THREE.Vector4(0.2, -0.35, 0.5, 1), new THREE.Vector4(0.9, -0.6, 0.2, 1)];
const atmo = sphere(4.4, atmoMat, 'atmosphere'); atmo.position.copy(EARTH);
const clouds = sphere(4.12, new THREE.MeshStandardMaterial({{color: 0xe6eef7, roughness: 1.0, transparent: true, opacity: 0.28, depthWrite: false}}), 'clouds');
clouds.position.copy(EARTH);

const METEOR_START = new THREE.Vector3(1.9, 1.5, -6.0);
const METEOR_V = new THREE.Vector3({mvx}, {mvy}, {mvz});
const meteor = new THREE.Mesh(new THREE.SphereGeometry(0.18, 20, 14), glowMat({c_fire}, 1.0));
meteor.name = 'meteor';
meteor.position.copy(METEOR_START);
scene.add(meteor);
const tailMat = new THREE.PointsMaterial({{size: 7, sizeAttenuation: false, color: {c_fire},
  transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false}});
const tailGeo = new THREE.BufferGeometry();
const TAIL_N = 90;
tailGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(TAIL_N * 3), 3));
const trail = new THREE.Points(tailGeo, tailMat);
trail.name = 'trail';
scene.add(trail);
const tailAge = new Float32Array(TAIL_N).fill(-1);

const shock = new THREE.Mesh(new THREE.RingGeometry(0.86, 1.0, 40), glowMat({c_fire}, 0.5));
shock.name = 'shock';
scene.add(shock);

const IMPACT_AT = {impact};
const HIT = EARTH.clone().add(new THREE.Vector3(0.05, 2.4, 3.1).normalize().multiplyScalar(4.05));
const impactFlash = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), glowMat({c_fire}, 1.0));
impactFlash.name = 'impact_flash'; impactFlash.position.copy(HIT); scene.add(impactFlash);
const impactRing = new THREE.Mesh(new THREE.RingGeometry(0.85, 1.0, 52), glowMat(0xffe6c0, 0.55));
impactRing.name = 'impact_ring'; impactRing.position.copy(HIT);
impactRing.lookAt(EARTH.clone().add(new THREE.Vector3(0.05, 2.4, 3.1).normalize().multiplyScalar(20)));
scene.add(impactRing);
const debrisMat = new THREE.PointsMaterial({{size: 8, sizeAttenuation: false, color: 0xffb070,
  transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false}});
const debrisGeo = new THREE.BufferGeometry();
const DEB_N = 120;
debrisGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(DEB_N * 3), 3));
const debris = new THREE.Points(debrisGeo, debrisMat);
debris.name = 'debris';
scene.add(debris);
const debV = [];
for (let i = 0; i < DEB_N; i++) {{
  const u = rnd() * 2 - 1, th = rnd() * 6.2832, s = Math.sqrt(1 - u * u);
  debV.push(new THREE.Vector3(Math.cos(th) * s, u, Math.sin(th) * s).multiplyScalar(0.4 + rnd() * 1.6));
}}

const station = new THREE.Group();
station.name = 'station';
station.position.set(-2.4, 0.9, 1.2);
scene.add(station);
station.add(box(1.2, 0.34, 0.34, lit(0xb3bcc6), 'hull'));
for (const sx of [-1, 1]) {{
  const p = box(0.5, 0.02, 0.18, lit(0x2b4a8f), 'panel_' + (sx < 0 ? 'l' : 'r'));
  p.position.set(sx * 0.85, 0, 0); station.add(p);
}}
const port = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 0.24), new THREE.MeshBasicMaterial({{color: {c_glass}, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false}}));
port.name = 'viewport';
port.position.set(0, 0.06, 0.18);
station.add(port);

const clamp01 = (v) => Math.max(0, Math.min(1, v));
let t = 0;
window.__step = (dt) => {{
  t += dt;
  earthMat.uniforms.uTime.value = t;
  postMat.uniforms.uTime.value = t;
  clouds.rotation.y = t * 0.010;
  station.rotation.z = Math.sin(t * 0.12) * 0.05;
  // метеорит летит до удара, затем гаснет
  const fly = Math.min(t, IMPACT_AT);
  meteor.position.copy(METEOR_START).addScaledVector(METEOR_V, fly);
  meteor.visible = t < IMPACT_AT + 0.4;
  meteor.material.opacity = clamp01(1 - Math.max(0, t - IMPACT_AT) / 0.4);
  // след: точки отстают по траектории
  const pos = tailGeo.attributes.position;
  for (let i = 0; i < TAIL_N; i++) {{
    if (tailAge[i] < 0 && t > 0.02) tailAge[i] = t;
    const behind = fly - (i + 1) * 0.012;
    const p = METEOR_START.clone().addScaledVector(METEOR_V, Math.max(0, behind));
    const jitter = (i % 7 - 3) * 0.012;
    pos.setXYZ(i, p.x + jitter, p.y + jitter * 0.6, p.z);
  }}
  pos.needsUpdate = true;
  tailMat.opacity = 0.75 * clamp01(1 - Math.max(0, t - IMPACT_AT) / 1.2);
  // ударная волна вокруг метеорита идёт циклами
  const cycle = (t % 0.4) / 0.4;
  shock.scale.setScalar(0.18 + 0.42 * cycle);
  shock.material.opacity = 0.5 * (1 - cycle);
  shock.position.copy(meteor.position);
  shock.lookAt(camera.position);
  // удар по поверхности
  const age = Math.max(0, t - IMPACT_AT);
  const live = t >= IMPACT_AT ? 1 : 0;
  const f = clamp01(age / 0.45);
  impactFlash.scale.setScalar(live * (0.06 + 0.26 * f));
  impactFlash.material.opacity = live * Math.max(0, 1 - age / 1.2);
  const rg = clamp01(age / 2.6);
  impactRing.scale.setScalar(live * (0.2 + 1.5 * rg));
  impactRing.material.opacity = live * Math.max(0, 0.85 * (1 - rg));
  const dp = debrisGeo.attributes.position;
  for (let i = 0; i < DEB_N; i++) {{
    const d = age * 1.1;
    dp.setXYZ(i, HIT.x + debV[i].x * d, HIT.y + debV[i].y * d, HIT.z + debV[i].z * d);
  }}
  dp.needsUpdate = true;
  debrisMat.opacity = live * clamp01(1 - age / 3.4);
  earthMat.uniforms.uDamage.value = clamp01(age / 4);
  camera.position.set({camx}, {camy}, 5.5);
  camera.lookAt(meteor.position.clone().lerp(HIT, clamp01(age / 1.5)));
  render();
  return window.__probe();
}};

window.__probe = () => ({{
  t: round4(t),
  objects: names(),
  meteor: [round4(meteor.position.x), round4(meteor.position.y), round4(meteor.position.z)],
  impact_age: round4(Math.max(0, t - IMPACT_AT)),
}});

render();
window.__ready = true;
</script>
</body></html>
"""


def _l26(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    mvx = round(rng.uniform(-0.55, -0.35), 3)
    mvy = round(rng.uniform(-0.60, -0.40), 3)
    mvz = round(rng.uniform(-0.75, -0.55), 3)
    impact = round(rng.uniform(1.5, 2.3), 2)
    camx = round(rng.uniform(-0.4, 0.4), 2)
    camy = round(rng.uniform(0.8, 1.4), 2)
    prompt = L26_PROMPT.format(
        c_sea="#13315e", c_land="#2f6b46", c_city="#ffd0d0", c_atmo="#6fb6ff",
        c_fire="#ffd0a0", c_glass="#ffd9a0",
        s_atmo=_share_text(PALETTE_L26["#6fb6ff"]), s_fire=_share_text(PALETTE_L26["#ffd0a0"]),
        s_city=_share_text(PALETTE_L26["#ffd0d0"]),
        mvx=f"{mvx:.3f}", mvy=f"{mvy:.3f}", mvz=f"{mvz:.3f}", impact=f"{impact:.2f}",
        camx=f"{camx:.2f}", camy=f"{camy:.2f}", seed=seed, lcg=_lcg_text(seed),
    ) + SPACE_TAIL
    golden = L26_GOLDEN.format(
        seed=seed, c_fire="0xffd0a0", c_glass="0xffd9a0",
        mvx=f"{mvx:.3f}", mvy=f"{mvy:.3f}", mvz=f"{mvz:.3f}", impact=f"{impact:.2f}",
        camx=f"{camx:.2f}", camy=f"{camy:.2f}",
    )
    for literal in (f"{mvx:.3f}", f"{mvy:.3f}", f"{mvz:.3f}", f"{impact:.2f}",
                    f"{camx:.2f}", f"{camy:.2f}", str(seed), "#ffd0a0"):
        assert literal in prompt, literal
    return _build(
        task_id="L-26", prompt=prompt, golden=golden,
        palette=PALETTE_L26, notes=NOTES_L26,
        composition=[
            _dark_share("глубокий космос вокруг", 0.25),
            _planet_lower("Земля в нижней половине кадра", 0.030),
            _glow_present("атмосфера светится", "#6fb6ff", 0.008),
            _glow_present("огненный след метеорита", "#ffd0a0", 0.003, 0.12),
        ],
        fixture={"seed": seed, "meteor_v": [mvx, mvy, mvz], "impact_at": impact},
        steps=STEPS["L-26"],
    )


# --------------------------------------------------------------------------- #
# L-27 — столкновение двух планет
# --------------------------------------------------------------------------- #

L27_PROMPT = """Создай эпичную 3D-сцену: рядом с Землёй сталкиваются две гигантские
планеты. Земля на среднем плане, две планеты сближаются, их кора деформируется,
появляются трещины лавы, выбросы обломков, облака пыли, светящиеся дуги. Земля
реагирует: штормы, искажение атмосферы. Орбитальная станция разворачивается к
катастрофе. Стиль — кинематографичный космический катастрофизм.

Сцена (имена обязательны):
* `earth` — сфера радиуса 3.2 в (2.6, -1.4, -11) с шейдером поверхности
  (море {c_sea}, суша {c_land}, города {c_city});
* `atmosphere` — ореол цвета {c_atmo} вокруг Земли;
* `planet_a` — сфера радиуса 2.0 цвета {c_rock} в ({ax}, {ay}, -7), растёт и
  деформируется: масштаб по x сжимается, появляются трещины (эмиссивные полосы
  {c_lava});
* `planet_b` — сфера радиуса 1.7 цвета {c_rock2} в ({bx}, {by}, -9), идёт
  навстречу, врезается в `planet_a` на t = {hit};
* `shards` — 160 обломков (аддитивные точки цвета {c_lava}), разлетающихся в
  момент удара со скоростями из LCG;
* `dust` — облако пыли: 90 медленных точек цвета {c_dust};
* `station` — группа справа внизу, разворачивается (поворот по z на {turn} за 4 с);
* `stars` — 2200 точек.

Камера: `(0, 0.6, 8)`, медленно отъезжает на {dolly} и смотрит между планетами.

Анимация: `__step(dt)` сближает планеты, в момент удара (`t = {hit}`) включает
трещины, разлёт обломков, пыль, деформацию; возвращает `__probe()` с `t`,
`objects`, `gap` — расстояние между планетами (4 знака).

{lcg}

Палитра-ориентир (доли — минимум кадра, по тону с допуском):
* {c_atmo} — атмосфера Земли, не меньше {s_atmo};
* {c_lava} — трещины лавы и раскалённые обломки, не меньше {s_lava};
* {c_dust} — облако пыли, не меньше {s_dust}.

Проверяется: палитра, планеты и Земля в кадре, лава, детализация, движение.
"""

L27_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>planet collision</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + SPACE_JS + """

const EARTH = new THREE.Vector3(2.6, -1.4, -11);
const earth = sphere(3.2, earthMat, 'earth');
earth.position.copy(EARTH);
earthMat.uniforms.uHits.value = [new THREE.Vector4(1, 0.2, 0.4, 1), new THREE.Vector4(-0.7, 0.8, 0.6, 1)];
const atmo = sphere(3.5, atmoMat, 'atmosphere'); atmo.position.copy(EARTH);
const clouds = sphere(3.3, new THREE.MeshStandardMaterial({{color: 0xe8f0f8, roughness: 1.0, transparent: true, opacity: 0.22, depthWrite: false}}), 'clouds');
clouds.position.copy(EARTH);

const rockMat = (c) => new THREE.ShaderMaterial({{
  uniforms: {{ uBase: {{value: new THREE.Color(c)}}, uLava: {{value: new THREE.Color({c_lava})}}, uHeat: {{value: 0.0}} }},
  vertexShader: `varying vec3 vP; varying vec3 vN;
    void main() {{ vP = normalize(position); vN = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
  fragmentShader: `
    uniform vec3 uBase, uLava; uniform float uHeat; varying vec3 vP; varying vec3 vN;
    float hash(vec3 p) {{ return fract(sin(dot(p, vec3(41.7, 289.1, 137.3))) * 43758.5453123); }}
    float noise(vec3 p) {{ vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                 mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z); }}
    void main() {{
      float n = noise(vP * 5.0) * 0.6 + noise(vP * 13.0) * 0.4;
      // трещины: узкие полосы по гребням шума, раскаляются от uHeat
      float crack = smoothstep(0.52, 0.62, n) * smoothstep(0.70, 0.60, n);
      vec3 col = mix(uBase, uLava, clamp(crack * (0.25 + uHeat * 2.4) + uHeat * crack * 1.5, 0.0, 1.0));
      float lambert = max(0.15, dot(vN, normalize(vec3(-1.0, 0.4, 0.6))));
      gl_FragColor = vec4(col * (0.35 + 0.75 * lambert), 1.0);
      #include <colorspace_fragment>
    }}`,
}});
const planetA = sphere(2.0, rockMat({c_rock}), 'planet_a');
planetA.position.set({ax}, {ay}, -7);
const planetB = sphere(1.7, rockMat({c_rock2}), 'planet_b');
planetB.position.set({bx}, {by}, -9);

const SHARDS = 160;
const shardGeo = new THREE.BufferGeometry();
shardGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(SHARDS * 3), 3));
const shardMat = new THREE.PointsMaterial({{size: 6, sizeAttenuation: false, color: {c_lava},
  transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false}});
const shards = new THREE.Points(shardGeo, shardMat);
shards.name = 'shards';
scene.add(shards);
const shardV = [];
for (let i = 0; i < SHARDS; i++) {{
  const u = rnd() * 2 - 1, th = rnd() * 6.2832, s = Math.sqrt(1 - u * u);
  shardV.push(new THREE.Vector3(Math.cos(th) * s, u, Math.sin(th) * s).multiplyScalar(0.6 + rnd() * 2.4));
}}
const DUST = 90;
const dustGeo = new THREE.BufferGeometry();
dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(DUST * 3), 3));
const dustMat = new THREE.PointsMaterial({{size: 14, sizeAttenuation: false, color: {c_dust},
  transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false}});
const dust = new THREE.Points(dustGeo, dustMat);
dust.name = 'dust';
scene.add(dust);
const dustV = [];
for (let i = 0; i < DUST; i++) {{
  const u = rnd() * 2 - 1, th = rnd() * 6.2832, s = Math.sqrt(1 - u * u);
  dustV.push(new THREE.Vector3(Math.cos(th) * s, u, Math.sin(th) * s).multiplyScalar(0.15 + rnd() * 0.6));
}}

const station = new THREE.Group();
station.name = 'station';
station.position.set(-3.4, -1.8, 1.5);
scene.add(station);
station.add(box(1.0, 0.30, 0.30, lit(0xb6bfc9), 'hull'));
for (const sx of [-1, 1]) {{
  const p = box(0.46, 0.02, 0.16, lit(0x2b4a8f), 'panel_' + (sx < 0 ? 'l' : 'r'));
  p.position.set(sx * 0.72, 0, 0); station.add(p);
}}

const HIT_AT = {hit};
const MID = new THREE.Vector3(({ax} + {bx}) / 2, ({ay} + {by}) / 2, -8);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
let t = 0;
window.__step = (dt) => {{
  t += dt;
  earthMat.uniforms.uTime.value = t;
  postMat.uniforms.uTime.value = t;
  clouds.rotation.y = t * 0.02;
  const k = clamp01(t / HIT_AT);
  // планеты сходятся, после удара расталкиваются и деформируются
  const after = clamp01((t - HIT_AT) / 3.0);
  planetA.position.lerpVectors(new THREE.Vector3({ax}, {ay}, -7), MID.clone().add(new THREE.Vector3(-0.5, 0, 0)), k);
  planetB.position.lerpVectors(new THREE.Vector3({bx}, {by}, -9), MID.clone().add(new THREE.Vector3(0.5, 0, 0)), k);
  planetA.position.addScaledVector(new THREE.Vector3(-1, 0.2, 0), after * 0.6);
  planetB.position.addScaledVector(new THREE.Vector3(1, -0.1, 0), after * 0.6);
  planetA.scale.set(1 - 0.22 * after, 1 + 0.10 * after, 1 + 0.06 * after);
  planetB.scale.set(1 + 0.16 * after, 1 - 0.14 * after, 1 + 0.05 * after);
  planetA.material.uniforms.uHeat.value = after;
  planetB.material.uniforms.uHeat.value = after * 0.85;
  const age = Math.max(0, t - HIT_AT);
  const live = t >= HIT_AT ? 1 : 0;
  const sp = shardGeo.attributes.position;
  for (let i = 0; i < SHARDS; i++) {{
    const d = age * 1.4;
    sp.setXYZ(i, MID.x + shardV[i].x * d, MID.y + shardV[i].y * d, MID.z + shardV[i].z * d);
  }}
  sp.needsUpdate = true;
  shardMat.opacity = live * clamp01(1 - age / 4.2) * 0.95;
  const dp = dustGeo.attributes.position;
  for (let i = 0; i < DUST; i++) {{
    const d = age * 0.5;
    dp.setXYZ(i, MID.x + dustV[i].x * d, MID.y + dustV[i].y * d, MID.z + dustV[i].z * d);
  }}
  dp.needsUpdate = true;
  dustMat.opacity = live * clamp01(1 - age / 5.5) * 0.4;
  station.rotation.z = -clamp01(t / 4) * {turn};
  camera.position.set(0, 0.6, 8 + {dolly} * t);
  camera.lookAt(MID.x * 0.4, MID.y * 0.4, -8);
  render();
  return window.__probe();
}};

window.__probe = () => ({{
  t: round4(t),
  objects: names(),
  gap: round4(planetA.position.distanceTo(planetB.position)),
}});

render();
window.__ready = true;
</script>
</body></html>
"""


def _l27(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    ax, ay = round(rng.uniform(-4.6, -3.6), 2), round(rng.uniform(0.6, 1.6), 2)
    bx, by = round(rng.uniform(-2.2, -1.2), 2), round(rng.uniform(-1.6, -0.6), 2)
    hit = round(rng.uniform(2.4, 3.4), 2)
    turn = round(rng.uniform(0.5, 1.1), 2)
    dolly = round(rng.uniform(0.045, 0.09), 3)
    prompt = L27_PROMPT.format(
        c_sea="#13315e", c_land="#2f6b46", c_city="#ffd0d0", c_atmo="#6fb6ff",
        c_rock="#8a6a52", c_rock2="#6d5a72", c_lava="#ff7a3c", c_dust="#d8b18a",
        s_atmo=_share_text(PALETTE_L27["#6fb6ff"]), s_lava=_share_text(PALETTE_L27["#ff7a3c"]),
        s_dust=_share_text(PALETTE_L27["#d8b18a"]),
        ax=f"{ax:.2f}", ay=f"{ay:.2f}", bx=f"{bx:.2f}", by=f"{by:.2f}",
        hit=f"{hit:.2f}", turn=f"{turn:.2f}", dolly=f"{dolly:.3f}", seed=seed, lcg=_lcg_text(seed),
    ) + SPACE_TAIL
    golden = L27_GOLDEN.format(
        seed=seed, c_lava="0xff7a3c", c_rock="0x8a6a52", c_rock2="0x6d5a72", c_dust="0xd8b18a",
        ax=f"{ax:.2f}", ay=f"{ay:.2f}", bx=f"{bx:.2f}", by=f"{by:.2f}",
        hit=f"{hit:.2f}", turn=f"{turn:.2f}", dolly=f"{dolly:.3f}",
    )
    for literal in (f"{ax:.2f}", f"{ay:.2f}", f"{bx:.2f}", f"{by:.2f}", f"{hit:.2f}",
                    f"{turn:.2f}", f"{dolly:.3f}", str(seed), "#ff7a3c"):
        assert literal in prompt, literal
    return _build(
        task_id="L-27", prompt=prompt, golden=golden,
        palette=PALETTE_L27, notes=NOTES_L27,
        composition=[
            _dark_share("глубокий космос", 0.25),
            _planet_lower("планеты в кадре", 0.05),
            _glow_present("лава после удара", "#ff7a3c", 0.003),
            _glow_present("атмосфера Земли", "#6fb6ff", 0.006),
        ],
        fixture={"seed": seed, "planet_a": [ax, ay], "planet_b": [bx, by], "hit_at": hit},
        steps=STEPS["L-27"],
    )


# --------------------------------------------------------------------------- #
# L-28 — цепная реакция обломков
# --------------------------------------------------------------------------- #

L28_PROMPT = """Создай масштабную 3D-сцену: планета разрушается, её обломки запускают
цепную реакцию в системе. Земля неподалёку, орбитальная станция наблюдает, как
облако раскалённых фрагментов меняет орбиты спутников; часть сгорает в атмосфере,
часть проходит мимо станции. На Земле видны полярные сияния, штормы и вспышки.
Слои: звёзды, туманности, пылевые облака, светящиеся обломки, линии траекторий.

Сцена (имена обязательны):
* `earth` — сфера радиуса 3.0 в (3.0, -2.2, -14), шейдер поверхности;
* `atmosphere` — ореол цвета {c_atmo};
* `aurora` — полярные сияния: две полупрозрачные полосы (`TorusGeometry`,
  сплющенные) цвета {c_aurora} над полюсами Земли;
* `core` — остаток разрушенной планеты: сфера радиуса 1.1 цвета {c_core} в
  (-3.2, 1.4, -12), раскалённая;
* `cloud_debris` — 200 обломков (аддитивные точки {c_lava}), летящих от планеты
  к Земле и мимо станции, скорости из LCG;
* `orbits` — 4 светящихся кольца-траектории вокруг Земли (тонкие, opacity 0.25);
* `satellites` — 5 мелких точек-спутников, едущих по кольцам;
* `station` — группа у левого края, обломки проходят рядом;
* `nebula` — большая полупрозрачная плоскость цвета {c_nebula} позади (z = -60);
* `stars` — 2400 точек.

Камера: плавно от станции к общей панораме: позиция идёт от (0.6, 0.4, 4) к
({endx}, {endy}, 12) за 6 с.

Анимация: `__step(dt)` летит обломками, крутит спутники, мерцает сиянием,
двигает камеру; `__probe()` возвращает `t`, `objects`, `ahead` — сколько
обломков ближе дистанции {near} к станции.

{lcg}

Палитра-ориентир (доли — минимум кадра, по тону с допуском):
* {c_atmo} — атмосфера Земли, не меньше {s_atmo};
* {c_aurora} — полярное сияние и орбиты, не меньше {s_aurora};
* {c_lava} — раскалённые обломки, не меньше {s_lava};
* {c_nebula} — туманность позади, не меньше {s_nebula}.

Проверяется: палитра, Земля и обломки в кадре, сияние, детализация, движение.
"""

L28_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>chain reaction</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + SPACE_JS + """

const EARTH = new THREE.Vector3(3.0, -2.2, -14);
const earth = sphere(3.0, earthMat, 'earth');
earth.position.copy(EARTH);
earthMat.uniforms.uHits.value = [new THREE.Vector4(0.6, -0.4, 0.2, 1), new THREE.Vector4(-0.2, -0.8, 0.5, 1)];
const atmo = sphere(3.3, atmoMat, 'atmosphere'); atmo.position.copy(EARTH);

// Сияние: две сплющенные полосы над полюсами.
const auroraMat = new THREE.MeshBasicMaterial({{color: {c_aurora}, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false}});
const auroras = [];
for (const sy of [1, -1]) {{
  const a = new THREE.Mesh(new THREE.TorusGeometry(1.9, 0.10, 8, 64), auroraMat.clone());
  a.name = 'aurora' + (sy > 0 ? '_n' : '_s');
  a.position.copy(EARTH).add(new THREE.Vector3(0, sy * 2.7, 0));
  a.rotation.x = Math.PI / 2;
  a.scale.set(1, 1, 0.22);
  scene.add(a); auroras.push(a);
}}

const core = new THREE.Mesh(new THREE.SphereGeometry(1.1, 32, 20), glowMat({c_core}, 1.0));
core.name = 'core';
core.position.set(-3.2, 1.4, -12);
scene.add(core);

const N = 200;
const debGeo = new THREE.BufferGeometry();
debGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(N * 3), 3));
const debMat = new THREE.PointsMaterial({{size: 5, sizeAttenuation: false, color: {c_lava},
  transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false}});
const cloudDebris = new THREE.Points(debGeo, debMat);
cloudDebris.name = 'cloud_debris';
scene.add(cloudDebris);
const debV = [];
for (let i = 0; i < N; i++) {{
  const u = rnd() * 2 - 1, th = rnd() * 6.2832, s = Math.sqrt(1 - u * u);
  debV.push(new THREE.Vector3(Math.cos(th) * s, u, Math.sin(th) * s).multiplyScalar(0.25 + rnd() * 1.5)
    .add(new THREE.Vector3(0.55, -0.25, -0.15)));
}}

const ORBITS = 4;
const orbits = [];
for (let i = 0; i < ORBITS; i++) {{
  const o = new THREE.Mesh(new THREE.RingGeometry(3.4 + i * 0.42, 3.45 + i * 0.42, 72),
    new THREE.MeshBasicMaterial({{color: {c_aurora}, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide}}));
  o.name = 'orbit' + i;
  o.position.copy(EARTH);
  o.rotation.x = Math.PI / 2 - 0.25 - i * 0.08;
  o.rotation.y = i * 0.4;
  scene.add(o); orbits.push(o);
}}
const sats = [];
for (let i = 0; i < 5; i++) {{
  const s = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), glowMat(0xbfe6ff, 1.0));
  s.name = 'satellite' + i;
  scene.add(s); sats.push(s);
}}

const nebula = new THREE.Mesh(new THREE.PlaneGeometry(220, 130),
  new THREE.MeshBasicMaterial({{color: {c_nebula}, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false}}));
nebula.name = 'nebula';
nebula.position.set(4, 2, -60);
scene.add(nebula);

const station = new THREE.Group();
station.name = 'station';
station.position.set(-1.2, 0.9, 3.0);
scene.add(station);
station.add(box(0.62, 0.18, 0.18, lit(0xb8c1cb), 'hull'));
for (const sx of [-1, 1]) {{
  const p = box(0.30, 0.015, 0.10, lit(0x2b4a8f), 'panel_' + (sx < 0 ? 'l' : 'r'));
  p.position.set(sx * 0.46, 0, 0); station.add(p);
}}

const CAM_FROM = new THREE.Vector3(0.0, 0.6, 4.6);
const CAM_TO = new THREE.Vector3({endx}, {endy}, 11.0);
const NEAR = {near};
const clamp01 = (v) => Math.max(0, Math.min(1, v));
let t = 0;
window.__step = (dt) => {{
  t += dt;
  earthMat.uniforms.uTime.value = t;
  postMat.uniforms.uTime.value = t;
  earth.rotation.y = t * 0.01;
  core.material.opacity = 0.86 + 0.14 * Math.sin(t * 2.3);
  core.scale.setScalar(1 - 0.06 * clamp01(t / 8));
  for (let i = 0; i < auroras.length; i++) {{
    auroras[i].material.opacity = 0.22 + 0.22 * Math.abs(Math.sin(t * (0.7 + i * 0.4)));
  }}
  const p = debGeo.attributes.position;
  for (let i = 0; i < N; i++) {{
    const d = t * 0.9;
    p.setXYZ(i, core.position.x + debV[i].x * d, core.position.y + debV[i].y * d, core.position.z + debV[i].z * d);
  }}
  p.needsUpdate = true;
  for (let i = 0; i < sats.length; i++) {{
    const w = 0.25 + i * 0.06;
    const r = 3.6 + i * 0.42;
    const a = t * w + i;
    sats[i].position.set(EARTH.x + Math.cos(a) * r, EARTH.y + Math.sin(a * 0.9) * 0.5, EARTH.z + Math.sin(a) * r * 0.3);
  }}
  camera.position.lerpVectors(CAM_FROM, CAM_TO, clamp01(t / 6));
  camera.lookAt(0.2, -0.4, -11);
  render();
  return window.__probe();
}};

window.__probe = () => {{
  let ahead = 0;
  const p = debGeo.attributes.position;
  for (let i = 0; i < N; i++) {{
    const dx = p.getX(i) - station.position.x, dy = p.getY(i) - station.position.y, dz = p.getZ(i) - station.position.z;
    if (Math.sqrt(dx * dx + dy * dy + dz * dz) < NEAR) ahead += 1;
  }}
  return {{ t: round4(t), objects: names(), ahead }};
}};

render();
window.__ready = true;
</script>
</body></html>
"""


def _l28(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    endx = round(rng.uniform(-0.4, 0.6), 2)
    endy = round(rng.uniform(0.6, 1.6), 2)
    near = round(rng.uniform(2.2, 3.4), 2)
    prompt = L28_PROMPT.format(
        c_atmo="#6fb6ff", c_aurora="#7bf0c8", c_core="#ff9a5a", c_lava="#ffb070",
        s_atmo=_share_text(PALETTE_L28["#6fb6ff"]), s_aurora=_share_text(PALETTE_L28["#7bf0c8"]),
        s_lava=_share_text(PALETTE_L28["#ffb070"]), s_nebula=_share_text(PALETTE_L28["#5a4a8f"]),
        c_nebula="#5a4a8f", endx=f"{endx:.2f}", endy=f"{endy:.2f}", near=f"{near:.2f}",
        seed=seed, lcg=_lcg_text(seed),
    ) + SPACE_TAIL
    golden = L28_GOLDEN.format(
        seed=seed, c_aurora="0x7bf0c8", c_core="0xff9a5a", c_lava="0xffb070", c_nebula="0x5a4a8f",
        endx=f"{endx:.2f}", endy=f"{endy:.2f}", near=f"{near:.2f}",
    )
    for literal in (f"{endx:.2f}", f"{endy:.2f}", f"{near:.2f}", str(seed), "#7bf0c8", "#5a4a8f"):
        assert literal in prompt, literal
    return _build(
        task_id="L-28", prompt=prompt, golden=golden,
        palette=PALETTE_L28, notes=NOTES_L28,
        composition=[
            _dark_share("космос вокруг", 0.25),
            _planet_lower("Земля и обломки в кадре", 0.05),
            _glow_present("сияние над полюсами", "#7bf0c8", 0.003),
            _glow_present("облако обломков", "#ffb070", 0.003),
        ],
        fixture={"seed": seed, "camera_end": [endx, endy], "near": near},
        steps=STEPS["L-28"],
    )


# --------------------------------------------------------------------------- #
# L-29 — станция уходит, Земля позади
# --------------------------------------------------------------------------- #

L29_PROMPT = """Создай финальную кинематографичную 3D-сцену: орбитальная станция
покидает орбиту Земли после глобальной катастрофы. Земля остаётся позади: штормы,
светящиеся разломы, следы ударов, облака пепла. Станция медленно удаляется, её
огни отражаются в космической пыли. Вдалеке виден источник катастрофы —
разрушенная Луна. Камера сначала показывает станцию крупно, затем отъезжает и
раскрывает сцену целиком: маленькая станция, огромная повреждённая Земля,
бесконечный космос.

Сцена (имена обязательны):
* `earth` — сфера радиуса 3.6 в (0.4, -2.6, -22), шейдер с повреждением:
  разломы цвета {c_lava}, пепел и штормы;
* `atmosphere` — ореол цвета {c_atmo}, тусклее обычного;
* `moon_broken` — разрушенная Луна: сфера радиуса 0.9 цвета {c_rock} с выбитым
  краем (вторая сфера-вырез) в (-2.6, 2.2, -34);
* `dust_ring` — кольцо космической пыли вокруг станции, 140 точек {c_dust};
* `station` — группа в центре кадра (0, 0.2, 3.6): корпус, панели, сопло, три
  `engine_glow` пятна цвета {c_engine} в корме, медленно удаляется от Земли;
* `debris_belt` — обломки спутников: 60 медленных точек между станцией и Землёй;
* `stars` — 2600 точек.

Камера: стартует крупно на станции и отъезжает: позиция идёт от
({startx}, {starty}, 2.2) к ({endx}, {endy}, 14) за {travel} с, глядя на станцию.

Анимация: `__step(dt)` гонит станцию от Земли, светит соплом, тянет пыль,
включает разломы на Земле; `__probe()` возвращает `t`, `objects`,
`camera: [x, y, z]`, `distance` — удаление станции от Земли (4 знака).

{lcg}

Палитра-ориентир (доли — минимум кадра, по тону с допуском):
* {c_atmo} — атмосфера повреждённой Земли, не меньше {s_atmo};
* {c_lava} — разломы и следы ударов, не меньше {s_lava};
* {c_engine} — сопло станции, не меньше {s_engine}.

Проверяется: палитра, Земля и станция в кадре, разломы, детализация, движение.
"""

L29_GOLDEN = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>departure</title>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body>
<script type="module">
import * as THREE from './vendor/three.module.js';
""" + LCG_JS + """

""" + SPACE_JS + """

const EARTH = new THREE.Vector3(0.4, -2.6, -22);
const earth = sphere(3.6, earthMat, 'earth');
earth.position.copy(EARTH);
earthMat.uniforms.uHits.value = [new THREE.Vector4(0.3, 0.1, 0.9, 1), new THREE.Vector4(-0.6, -0.5, 0.7, 1)];
const atmo = sphere(3.88, atmoMat, 'atmosphere');
atmo.position.copy(EARTH);
atmoMat.uniforms.uColor.value = new THREE.Color({c_atmo});

const moon = sphere(0.9, new THREE.MeshStandardMaterial({{color: {c_rock}, roughness: 0.95}}), 'moon_broken');
moon.position.set(-2.6, 2.2, -34);
const bite = new THREE.Mesh(new THREE.SphereGeometry(0.62, 24, 16), new THREE.MeshBasicMaterial({{color: 0x0b0d16}}));
bite.name = 'moon_bite';
bite.position.copy(moon.position).add(new THREE.Vector3(0.45, 0.35, 0.35));
scene.add(bite);

const station = new THREE.Group();
station.name = 'station';
station.position.set(0, 0.2, 3.6);
scene.add(station);
station.add(box(1.1, 0.34, 0.34, lit(0xc0c8d2), 'hull'));
for (const sx of [-1, 1]) {{
  const p = box(0.52, 0.02, 0.20, lit(0x2b4a8f), 'panel_' + (sx < 0 ? 'l' : 'r'));
  p.position.set(sx * 0.86, 0, 0); station.add(p);
}}
const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.20, 0.22, 16), lit(0x8d949c));
nozzle.name = 'nozzle';
nozzle.rotation.x = Math.PI / 2;
nozzle.position.set(0, 0, 0.28);
station.add(nozzle);
const engines = [];
for (let i = 0; i < 3; i++) {{
  const e = new THREE.Mesh(new THREE.SphereGeometry(0.07, 12, 8), glowMat({c_engine}, 0.9));
  e.name = 'engine_glow' + i;
  e.position.set(-0.14 + i * 0.14, 0, 0.42);
  station.add(e); engines.push(e);
}}

const DUST = 140;
const dustGeo = new THREE.BufferGeometry();
dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(DUST * 3), 3));
const dustMat = new THREE.PointsMaterial({{size: 4, sizeAttenuation: false, color: {c_dust},
  transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false}});
const dustRing = new THREE.Points(dustGeo, dustMat);
dustRing.name = 'dust_ring';
scene.add(dustRing);
const dustV = [];
for (let i = 0; i < DUST; i++) {{
  const u = rnd() * 2 - 1, th = rnd() * 6.2832, s = Math.sqrt(1 - u * u);
  dustV.push(new THREE.Vector3(Math.cos(th) * s, u, Math.sin(th) * s).multiplyScalar(0.3 + rnd() * 1.1));
}}

const BELT = 60;
const beltGeo = new THREE.BufferGeometry();
beltGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(BELT * 3), 3));
const belt = new THREE.Points(beltGeo, new THREE.PointsMaterial({{size: 3, sizeAttenuation: false, color: 0x9fb4cc,
  transparent: true, opacity: 0.7, depthWrite: false}}));
belt.name = 'debris_belt';
scene.add(belt);
const beltPos = [];
for (let i = 0; i < BELT; i++) {{
  beltPos.push(new THREE.Vector3((rnd() - 0.5) * 9, (rnd() - 0.5) * 4, -6 - rnd() * 12));
}}

const CAM_FROM = new THREE.Vector3({startx}, {starty}, 6.5);
const CAM_TO = new THREE.Vector3({endx}, {endy}, 22.0);
const TRAVEL = {travel};
const clamp01 = (v) => Math.max(0, Math.min(1, v));
let t = 0;
window.__step = (dt) => {{
  t += dt;
  earthMat.uniforms.uTime.value = t;
  postMat.uniforms.uTime.value = t;
  earthMat.uniforms.uDamage.value = clamp01(0.35 + t / 22);
  earth.rotation.y = t * 0.006;
  moon.rotation.y = t * 0.02;
  const k = clamp01((t - 1.0) / TRAVEL);
  station.position.z = -0.6 - k * 3.4;
  station.position.x = -k * 2.2;
  station.position.y = 0.2 + k * 0.5;
  station.rotation.z = Math.sin(t * 0.25) * 0.10;
  station.rotation.y = Math.sin(t * 0.14) * 0.14;
  for (let i = 0; i < engines.length; i++) {{
    engines[i].material.opacity = 0.55 + 0.4 * Math.abs(Math.sin(t * (2.4 + i * 0.5)));
    engines[i].scale.setScalar(0.8 + 0.5 * Math.abs(Math.sin(t * 3.0 + i)));
  }}
  const p = dustGeo.attributes.position;
  for (let i = 0; i < DUST; i++) {{
    const d = -1.4 + (i % 40) * 0.032;
    p.setXYZ(i, station.position.x + dustV[i].x * 0.9,
      station.position.y + dustV[i].y * 0.9,
      station.position.z + d + dustV[i].z * 0.3);
  }}
  p.needsUpdate = true;
  dustMat.opacity = 0.5 * clamp01(1 - clamp01(t / 18));
  const bp = beltGeo.attributes.position;
  for (let i = 0; i < BELT; i++) {{
    bp.setXYZ(i, beltPos[i].x, beltPos[i].y + Math.sin(t * 0.1 + i) * 0.3, beltPos[i].z);
  }}
  bp.needsUpdate = true;
  camera.position.lerpVectors(CAM_FROM, CAM_TO, k);
  camera.lookAt(EARTH.x * 0.35 + station.position.x * 0.65, EARTH.y * 0.35 + station.position.y * 0.65, EARTH.z * 0.5 + station.position.z * 0.5);
  render();
  return window.__probe();
}};

window.__probe = () => ({{
  t: round4(t),
  objects: names(),
  camera: [round4(camera.position.x), round4(camera.position.y), round4(camera.position.z)],
  distance: round4(station.position.distanceTo(EARTH)),
}});

render();
window.__ready = true;
</script>
</body></html>
"""


def _l29(rng: random.Random) -> core.TaskInstance:
    seed = rng.randrange(1, 10 ** 6)
    startx, starty = round(rng.uniform(-0.3, 0.3), 2), round(rng.uniform(-0.1, 0.3), 2)
    endx, endy = round(rng.uniform(-1.4, -0.6), 2), round(rng.uniform(0.4, 1.0), 2)
    travel = round(rng.uniform(5.0, 8.0), 2)
    prompt = L29_PROMPT.format(
        c_lava="#ff8a4c", c_atmo="#5e9fe0", c_rock="#9a97a8", c_dust="#c8d4e4", c_engine="#8fd8ff",
        s_atmo=_share_text(PALETTE_L29["#5e9fe0"]), s_lava=_share_text(PALETTE_L29["#ff8a4c"]),
        s_engine=_share_text(PALETTE_L29["#8fd8ff"]),
        startx=f"{startx:.2f}", starty=f"{starty:.2f}", endx=f"{endx:.2f}", endy=f"{endy:.2f}",
        travel=f"{travel:.2f}", seed=seed, lcg=_lcg_text(seed),
    ) + SPACE_TAIL
    golden = L29_GOLDEN.format(
        seed=seed, c_lava="0xff8a4c", c_atmo="0x5e9fe0", c_rock="0x9a97a8",
        c_dust="0xc8d4e4", c_engine="0x8fd8ff",
        startx=f"{startx:.2f}", starty=f"{starty:.2f}", endx=f"{endx:.2f}", endy=f"{endy:.2f}",
        travel=f"{travel:.2f}",
    )
    for literal in (f"{startx:.2f}", f"{starty:.2f}", f"{endx:.2f}", f"{endy:.2f}",
                    f"{travel:.2f}", str(seed), "#ff8a4c", "#8fd8ff"):
        assert literal in prompt, literal
    return _build(
        task_id="L-29", prompt=prompt, golden=golden,
        palette=PALETTE_L29, notes=NOTES_L29,
        composition=[
            _dark_share("космос вокруг", 0.25),
            _planet_lower("повреждённая Земля в кадре", 0.010),
            _glow_present("разломы на Земле", "#ff8a4c", 0.002, 0.15),
            _glow_present("сопло станции", "#8fd8ff", 0.002),
        ],
        fixture={"seed": seed, "camera_from": [startx, starty], "camera_to": [endx, endy], "travel": travel},
        steps=STEPS["L-29"],
    )


def build(rng: random.Random) -> list[core.TaskInstance]:
    return [_l25(rng), _l26(rng), _l27(rng), _l28(rng), _l29(rng)]
