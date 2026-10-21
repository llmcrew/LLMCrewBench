"""Галерея ответов витрины: кадр, живая сцена и сырой HTML рядом.

Кадр — доказательство того, что модель нарисовала; но кадр нельзя потрогать.
Поэтому отчёт умеет три вещи на одну задачу:

* показать снимок (быстро, работает всегда);
* открыть **живую страницу** ответа в iframe — ту самую, что сняла кадр, с той
  же библиотекой, так что видно, как сцена двигается;
* отдать сырой HTML в новую вкладку, чтобы сравнить двух моделей рядом.

Плюс режим съёмки: кадры расставляются с фиксированным размером и порядком,
чтобы запись видео шла по одному клику.
"""

from __future__ import annotations

import html
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import core
from .visual import cache_dir, extract_html

#: Vendored library is served to the gallery pages from this relative folder.
ASSET_DIR = "_vendor"

#: Task ids become file names, so they must not carry a path. The registry uses
#: ``K-1``/``IF-2``; anything else is refused instead of being written with a
#: silent ``pages/../`` escape (R17). ``schema`` validates against this same
#: expression, so a suite that validates always produces a page name.
_ID_SCHEMA = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*$")


def _cell(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


STEP_HOOK = """
<script>
// Сцена играет сама: драйвер измеряет окно движения (сколько шагов подряд кадр
// реально меняется), разворачивает его на несколько секунд и зацикливает
// перезагрузкой. Без этого сцена выгорает за 1-2 с и стоит, как скриншот.
// Отчёт и запись могут вмешаться сообщениями 'step', 'pause', 'play', 'reset'.
(() => {
  const DT = 1 / 60;
  const QUIET = 90;          // столько шагов без изменений — окно кончилось
  const MEASURE_CAP = 1200;  // предел замера в шагах
  const MIN_WINDOW = 4;      // короче — сцена статична, повторять нечего
  const SPAN_PER_STEP = 6;   // кадров на шаг показа (~0.1 с при 60 к/с)
  const SPAN_MIN = 240;      // показ короче 4 с сливается
  const SPAN_MAX = 600;      // ...а длиннее 10 с — затягивается
  const HOLD_MS = 900;       // пауза перед повтором
  const MEASURE_MS = 8;      // бюджет замера на кадр, чтобы не морозить вкладку

  // three.js держит render собственным свойством экземпляра (this.render =
  // render), поэтому обёртка прототипа его не видит: подменяем конструктор,
  // а render перепривязываем на самом экземпляре. Так кадр обновляется, даже
  // если страница рисует только в __probe — иначе сцена стоит, как скриншот.
  let lastRender = null;
  const wrap = () => {
    const W = window.THREE && window.THREE.WebGLRenderer;
    if (!W || W.__dbWrapped) return !!W;
    const Wrapped = function (...args) {
      const inst = new W(...args);
      const orig = inst.render.bind(inst);
      inst.render = function (scene, camera) { lastRender = [inst, scene, camera]; return orig(scene, camera); };
      return inst;
    };
    Wrapped.prototype = W.prototype;
    Object.setPrototypeOf(Wrapped, W);
    Wrapped.__dbWrapped = true;
    window.THREE.WebGLRenderer = Wrapped;
    return true;
  };
  if (!wrap()) { let n = 0; const t = setInterval(() => { if (wrap() || ++n > 1500) clearInterval(t); }, 4); }

  const step = () => { if (typeof window.__step === 'function') { try { window.__step(DT); } catch (e) {} } };
  const probe = () => { if (typeof window.__probe === 'function') { try { window.__probe(); } catch (e) {} } };
  const draw = () => { if (lastRender) { try { lastRender[0].render(lastRender[1], lastRender[2]); } catch (e) {} } };
  // Один шаг — и сразу кадр: probe зовём всегда, часть страниц рисует в нём.
  const beat = () => { step(); probe(); draw(); };

  let oc = null, ox = null;
  const sig = () => {  // дешёвая подпись кадра: 64x48 достаточно для «меняется/нет»
    try {
      const cs = document.querySelectorAll('canvas');
      if (!cs.length) return -1;
      if (!oc) { oc = document.createElement('canvas'); oc.width = 64; oc.height = 48; ox = oc.getContext('2d', {willReadFrequently: true}); }
      // Сумма по ВСЕМ canvas: страница может держать два canvas, первый — WebGL без
      // preserveDrawingBuffer, и подпись только по нему читалась как пустая:
      // движущаяся сцена получала freeze=0 и показывалась статичным кадром.
      let total = 0;
      for (const c of cs) {
        if (!c.width || !c.height) continue;
        ox.clearRect(0, 0, 64, 48);
        ox.drawImage(c, 0, 0, 64, 48);
        const d = ox.getImageData(0, 0, 64, 48).data;
        let s = 0;
        for (let i = 0; i < d.length; i += 4) s = (s + d[i] + d[i + 1] * 3 + d[i + 2] * 7) % 1000003;
        total = (total + s + cs.length * 7) % 1000003;
      }
      return total;
    } catch (e) { return -2; }
  };

  // Замер должен пережить перезагрузку (её использует повтор): sessionStorage,
  // а если он недоступен — window.name (оба живут в пределах вкладки).
  const KEY = 'dailybench.scene:' + location.pathname;
  const put = (v) => {
    try { sessionStorage.setItem(KEY, JSON.stringify(v)); } catch (e) {}
    try { window.name = 'dbscene:' + JSON.stringify(v); } catch (e) {}
  };
  const got = () => {
    try { const v = sessionStorage.getItem(KEY); if (v) return JSON.parse(v); } catch (e) {}
    try { if (window.name.indexOf('dbscene:') === 0) return JSON.parse(window.name.slice(8)); } catch (e) {}
    return null;
  };

  let freeze = 0, span = SPAN_MAX, frames = 0, shown = 0, phase = 'measuring', holdUntil = 0, auto = true;
  const scene = {mode: 'measuring', freeze: 0, step: 0};
  // Ответы сами выставляют `window.__scene` (свой THREE.Scene) — у deepseek это
  // уже занято, и модульный скрипт ответа исполняется ПОСЛЕ этого хука. Поэтому
  // драйвер живёт под своим именем, а `__scene` остаётся странице.
  window.__dbscene = scene;
  window.__beat = beat;
  const publish = () => {   // для записи и отладки: состояние драйвера наружу
    scene.mode = phase; scene.freeze = freeze; scene.step = shown;
    // seek(n) — ручной покадровый прогон до шага n (гарантирует первый кадр)
    scene.seek = (n) => {
      auto = false;
      const target = Math.max(0, n | 0);
      // Нулевой кадр — это состояние до первого шага: seek(0) только рисует,
      // но физическое время не двигает. Шагов ровно столько, сколько просят:
      // повторный seek(n) на том же шаге уже ничего не продвигает.
      while (shown < target) { beat(); shown++; }
      probe(); draw();
      scene.step = shown;
      return shown;
    };
  };
  publish();

  const play = () => {
    const cache = got();
    freeze = (cache && cache.freeze | 0) || 0;
    // Окно движения разворачиваем примерно на 0.1 с на шаг: сцена из 240 шагов
    // идёт 24 с — так движение видно, а не проскакивает за пару секунд.
    span = Math.min(SPAN_MAX, Math.max(SPAN_MIN, freeze * SPAN_PER_STEP));
    phase = freeze < MIN_WINDOW ? 'static' : 'play';
    frames = 0; shown = 0;
    publish();
    if (phase === 'play') requestAnimationFrame(tick);
  };

  const tick = (now) => {
    if (phase === 'play' && auto) {
      frames++;
      const target = Math.min(freeze, Math.round(frames * freeze / span));
      while (shown < target) { beat(); shown++; }
      scene.step = shown;
      if (frames >= span) { phase = 'hold'; holdUntil = performance.now() + HOLD_MS; scene.mode = phase; }
    } else if (phase === 'hold' && auto && performance.now() >= holdUntil) {
      location.reload();  // сброса у страницы нет: повтор — только перезагрузкой
    }
    requestAnimationFrame(tick);
  };

  if (got() === null) {
    // Первый заход: дождаться кадра страницы, измерить окно движения и
    // перезагрузиться — показ пойдёт с чистого состояния, замер уже в кэше.
    const t0 = performance.now();
    let prev = null, lastChange = 0, k = 0, started = false;
    const measureFrame = () => {
      const t1 = performance.now();
      while (performance.now() - t1 < MEASURE_MS && k < MEASURE_CAP) {
        beat(); k++;
        const s = sig();
        if (s !== prev) { lastChange = k; prev = s; }
        else if (k - lastChange >= QUIET) break;
      }
      if ((k >= MEASURE_CAP || k - lastChange >= QUIET) && k > 0) {
        put({freeze: lastChange});
        location.reload();
        return;
      }
      requestAnimationFrame(measureFrame);
    };
    const wait = () => {
      if (window.__ready !== true && performance.now() - t0 < 8000) { requestAnimationFrame(wait); return; }
      probe(); draw(); prev = sig();
      if (!started) { started = true; requestAnimationFrame(measureFrame); }
    };
    requestAnimationFrame(wait);
  } else {
    play();
  }

  const resume = () => { auto = true; if (phase !== 'play') { phase = 'play'; publish(); requestAnimationFrame(tick); } };
  window.addEventListener('message', (event) => {
    const cmd = event.data;
    if (cmd === 'step') { auto = false; beat(); shown++; scene.step = shown; }
    else if (cmd === 'pause') { auto = false; }
    else if (cmd === 'play') { resume(); }
    else if (cmd === 'reset') { window.location.reload(); }
    else return;
    window.dispatchEvent(new Event('dailybench-step'));
  });

  window.addEventListener('keydown', (event) => {
    if (event.target && /INPUT|TEXTAREA/.test(event.target.tagName || '')) return;
    if (event.key === ' ') { event.preventDefault(); auto = !auto; if (auto) resume(); }
    if (event.key === 'ArrowRight') { event.preventDefault(); auto = false; beat(); shown++; scene.step = shown; }
    if (event.key === 'r' || event.key === 'R' || event.key === 'к' || event.key === 'К') window.location.reload();
  });
})();
</script>
"""


#: Every way an answer may import the vendored library: the relative path the
#: contract gives, and the bare ``three`` specifier of an importmap.
_THREE_SOURCE = r"(?:\./vendor/three\.module\.js|three)"
_IMPORT_FROM = re.compile(
    rf"""import\s+(?P<what>[^'"]+?)\s+from\s+['"](?:{_THREE_SOURCE})['"];?"""
)
_IMPORT_BARE = re.compile(rf"""import\s+['"](?:{_THREE_SOURCE})['"];?""")
_JS_NAME = r"[A-Za-z_$][\w$]*"


def _three_binding(what: str) -> str:
    """``window.THREE`` equivalent of one supported import clause.

    Supported forms: ``* as N``, ``{A}``, ``{A as B}`` and empty braces. A
    default import or a default/namespace combination has no equivalent in the
    bundle and is refused explicitly instead of leaving a broken module (R27).
    """
    clause = what.strip()
    namespace = re.fullmatch(rf"\*\s+as\s+({_JS_NAME})", clause)
    if namespace:
        return f"const {namespace.group(1)} = window.THREE;"
    named = re.fullmatch(r"\{(.*)\}", clause, re.S)
    if not named:
        raise ValueError(f"standalone-экспорт не поддерживает import {clause!r}")
    pairs: list[str] = []
    for part in named.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        pieces = [piece.strip() for piece in part.split(" as ")]
        if len(pieces) == 1 and re.fullmatch(_JS_NAME, pieces[0]):
            pairs.append(pieces[0])
        elif len(pieces) == 2 and all(re.fullmatch(_JS_NAME, p) for p in pieces):
            # A property renamed by the module: destructuring keeps the alias.
            pairs.append(f"{pieces[0]}: {pieces[1]}")
        else:
            raise ValueError(f"standalone-экспорт не поддерживает import {clause!r}")
    return "const { " + ", ".join(pairs) + " } = window.THREE;"


def standalone_html(page: str, bundle: str = "./three.global.js") -> str:
    """Ответ, который запускается прямо с диска (``file://``), без сервера.

    Страница ответа импортирует ``./vendor/three.module.js`` — это ES-модуль, а
    под ``file://`` Chrome не грузит внешние модули (политика CORS для origin
    ``null``). Поэтому: классический бандл ``three.global.js`` кладётся рядом и
    подключается первым, а каждый импорт заменяется деструктуризацией
    ``window.THREE`` — с сохранением локальных имён и алиасов. Импорт, у
    которого нет эквивалента (default), — явная ошибка экспорта, а не молча
    сломанный HTML.
    """
    named = list(_IMPORT_FROM.finditer(page))
    bare = list(_IMPORT_BARE.finditer(page))
    if not named and not bare:
        return page

    edits: list[tuple[int, int, str]] = [(match.span(), "") for match in bare]
    for match in named:
        edits.append((match.span(), _three_binding(match.group("what"))))
    # С конца, чтобы более ранние смещения не сдвигались.
    for (start, end), replacement in sorted(edits, reverse=True):
        page = page[:start] + replacement + page[end:]

    inject = f"<script src='{bundle}'></script>\n"
    if "</head>" in page:
        return page.replace("</head>", inject + "</head>", 1)
    return inject + page


def three_bundle(vendor: Path) -> str:
    """Собирает ``vendor/three/*.js`` в один классический скрипт под ``file://``.

    Оба файла — ES-модули: ``three.core.js`` экспортирует имена, ``three.module.js``
    импортирует их и реэкспортирует ядро. Снимаем ``import``/``export``, оба файла
    оборачиваем в IIFE и кладём результат в ``window.THREE`` — получается тот же
    объект, что давал `import * as THREE`.
    """
    core = (vendor / "three.core.js").read_text(encoding="utf-8")
    module = (vendor / "three.module.js").read_text(encoding="utf-8")

    def parts(stmt: str) -> list[str]:
        body = stmt[stmt.index("{") + 1: stmt.rindex("}")]
        return [p.strip() for p in body.split(",") if p.strip()]

    def bindings(stmt: str) -> str:
        out = []
        for p in parts(stmt):
            out.append(f"{p.split(' as ')[0].strip()}: {p.split(' as ')[-1].strip()}" if " as " in p else p)
        return ", ".join(out)

    def exported(stmt: str) -> list[str]:
        return [p.split(" as ")[-1].strip() for p in parts(stmt)]

    core_names: list[str] = []

    def cut_core(match: "re.Match[str]") -> str:
        core_names.extend(exported(match.group(0)))
        return ""

    core_code = re.sub(r"^export \{[^}]*\};?$", cut_core, core, flags=re.M)
    # Реэкспорт и импорт ядра одним и тем же способом превращаем в деструктуризацию.
    module = re.sub(
        r"^(?:export|import) \{([^}]*)\} from '\./three\.core\.js';?$",
        lambda m: f"const {{ {bindings(m.group(0))} }} = __CORE__;",
        module,
        flags=re.M,
    )
    module_names: list[str] = []

    def cut_module(match: "re.Match[str]") -> str:
        module_names.extend(exported(match.group(0)))
        return ""

    module_code = re.sub(r"^export \{[^}]*\};?$", cut_module, module, flags=re.M)
    return (
        "/* three.js offline-бандл для file://: снят import/export, THREE в window. */\n"
        "(function () {\n'use strict';\n"
        f"const __CORE__ = (() => {{\n{core_code}\nreturn {{ {', '.join(core_names)} }};\n}})();\n"
        f"const __MOD__ = (() => {{\n{module_code}\nreturn {{ {', '.join(module_names)} }};\n}})();\n"
        "window.THREE = Object.assign({}, __CORE__, __MOD__);\n"
        "})();\n"
    )


def answer_html(entry: dict[str, Any]) -> str:
    """The page a model produced for this task: the artefact, else its reply.

    An agent run writes the page to disk and its chat reply may be empty, so the
    transcript's recorded artefact comes first; a chat run has no artefact and
    falls back to the fenced block in the reply.
    """
    if (entry.get("task") or {}).get("category", entry.get("category")) in core.ART_CATEGORIES:
        return _raw_page(entry)
    transcript = entry.get("transcript") or {}
    recorded = ((transcript.get("extra") or {}).get("artefact") or {}).get("text") or ""
    if recorded.strip():
        page = extract_html(recorded)
        if page and "</body>" in page:
            return page.replace("</body>", STEP_HOOK + "</body>", 1)
        return page + STEP_HOOK if page else ""
    turns = transcript.get("turns") or []
    if not turns:
        return ""
    text = (turns[-1].get("reply") or {}).get("text") or ""
    page = extract_html(text)
    if page and "</body>" in page:
        return page.replace("</body>", STEP_HOOK + "</body>", 1)
    return page + STEP_HOOK if page else ""


def scene_page(item: dict[str, Any], target: Path, task_id: str) -> str:
    """Страница живой сцены: standalone-версия, путь к бандлу — от каталога отчёта."""
    if not item.get("html"):
        return ""
    depth = "../../../"  # pages/ -> <model>/ -> _showcase/ -> каталог отчёта
    caption = f"{item.get('model') or target.name} · {task_id}"
    banner = (
        "<div style='position:fixed;z-index:9;left:8px;top:8px;background:#000000b3;"
        "color:#c9d1d9;font:12px/1.4 ui-monospace,monospace;padding:4px 8px;border-radius:6px;"
        "pointer-events:none'>"
        + html.escape(caption)
        + " · пробел — пауза, → — шаг, R — сброс</div>"
    )
    page = standalone_html(item["html"], f"{depth}{ASSET_DIR}/three.global.js")
    if item.get("category") in core.ART_CATEGORIES:
        return page
    # Кадр сцены — 1920x1080. Ответ рисует canvas по своим числам, поэтому
    # масштабируем его целиком под окно: без обрезки и без полей, на любом
    # экране. `zoom` двигает и хит-тесты, поэтому он, а не transform: scale.
    fit = """<script>
(() => {
  const fit = () => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    const k = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    document.documentElement.style.zoom = k === 1 ? '' : String(k);
    document.body.style.overflow = 'hidden';
    document.body.style.display = 'flex';
    document.body.style.alignItems = 'center';
    document.body.style.justifyContent = 'center';
    document.body.style.height = '100vh';
  };
  window.addEventListener('resize', fit);
  window.addEventListener('load', fit);
  fit();
})();
</script>"""
    if "</body>" in page:
        page = page.replace("</body>", banner + fit + "</body>", 1)
    else:
        page += banner + fit
    return page


def _allowed_roots(run_root: Path | None) -> list[Path]:
    """Where a recorded frame may come from: the run and the render cache."""
    roots = [cache_dir()]
    if run_root is not None:
        roots.append(run_root)
    return [path.resolve() for path in roots]


def _within(path: Path, roots: list[Path]) -> bool:
    """``True`` when the resolved file lies inside one of the allowed roots."""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _shot_candidate(raw: str, run_root: Path | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() and run_root is not None:
        path = run_root / path
    return path


def showcase_entries(run: dict[str, Any], run_root: Path | None = None) -> list[dict[str, Any]]:
    """Every showcase task of a run: id, block, title, page, frame, spend."""
    if run_root is None and run.get("dir"):
        run_root = Path(str(run["dir"]))
    out: list[dict[str, Any]] = []
    for entry in run.get("tasks") or []:
        task = entry.get("task") or {}
        category = task.get("category", "")
        if category not in core.VISUAL_CATEGORIES:
            continue
        transcript = entry.get("transcript") or {}
        visual = ((transcript.get("extra") or {}).get("visual") or {})
        # A render that died before ``_observe`` (screenshot timeout) leaves
        # ``extra.visual`` empty — but the frame is still reproducible: the
        # sample point is in the task meta and both blocks step at 1/60. Without
        # this fallback such a page kept its live scene and lost its picture.
        meta = task.get("meta") or {}
        sample = {
            "shot": visual.get("shot") or "",
            "steps": visual.get("steps") or meta.get("steps"),
            "dt": visual.get("dt") or 1 / 60,
        }
        out.append(
            {
                "id": task.get("id", "?"),
                "model": (run.get("config") or {}).get("model", ""),
                "category": category,
                "title": task.get("title", ""),
                "html": answer_html(entry),
                "raw": _raw_page(entry),
                "shot": _shot_for(entry, sample, run_root),
                "score": entry.get("score") or {},
                "spend": transcript.get("spend") or {},

                "steps": sample["steps"],
                "dt": sample["dt"],
            }
        )
    return out


def _raw_page(entry: dict[str, Any]) -> str:
    """The answer page exactly as it was scored: no STEP_HOOK, no wrapping."""
    transcript = entry.get("transcript") or {}
    recorded = ((transcript.get("extra") or {}).get("artefact") or {}).get("text") or ""
    if recorded.strip():
        return extract_html(recorded)
    raw = "".join(
        str((turn.get("reply") or {}).get("text") or "")
        for turn in transcript.get("turns") or []
    )
    return extract_html(raw)


def _shot_for(entry: dict[str, Any], visual: dict[str, Any], run_root: Path | None = None) -> str:
    """The frame of this answer, re-rendered when the cached file is gone.

    The cache lives outside the run (``~/.cache/dailybench/visual``) and is keyed
    by page content: a purged or rotated cache used to leave the showcase with the
    live page but no picture at all, silently. A K render is deterministic, so
    the recovered frame is the same one the checks scored; a recovered live
    page (L) is *a* frame of the same work, not necessarily the byte-identical
    one the check scored. A browser-less machine just keeps whatever the run
    recorded.

    A recorded frame is only trusted inside the run directory or the render
    cache: an imported ``visual.shot`` used to be copied wherever it pointed,
    including a local file outside the run (R17).
    """
    roots = _allowed_roots(run_root)
    existing_path = _shot_candidate(str(visual.get("shot") or ""), run_root)
    existing = ""
    fallback = ""
    if existing_path is not None:
        if not existing_path.is_file():
            # A path that no longer exists cannot be proven to escape; keep it
            # as a fallback while a fresh render is attempted.
            fallback = str(existing_path)
        elif run_root is None or _within(existing_path, roots):
            # A caller without a known root (a hand-built dict) cannot prove
            # containment; keep the historical behaviour there.
            existing = str(existing_path)
    if existing:
        return existing
    steps, dt = visual.get("steps"), visual.get("dt")
    # The frame must be rendered from the *raw* answer, not from the gallery copy:
    # the copy carries STEP_HOOK, which auto-steps the scene from requestAnimation
    # -Frame, so a render of it lands on a different moment (measured: 14.6 % of
    # the pixels differed). The checks scored the raw page, and so must the frame.
    page = str(entry.get("raw") or "") or _raw_page(entry)
    if not page:
        return fallback
    from . import art, visual

    try:
        category = (entry.get("task") or {}).get("category", entry.get("category"))
        if category in core.ART_CATEGORIES:
            path = art.art_shot_path(page)
            if not path.is_file():
                art.cached_art_render(page)
        else:
            if not steps or not dt:
                return fallback
            steps, dt = int(steps), float(dt)
            path = visual.shot_path(page, steps, dt)
            if not path.is_file():
                visual._cached_render(page, steps, dt)
                # Запись запечатана отпечатком probe, и имя меняется при рендере:
                # путь переспрашивается, иначе кадр остался бы «потерянным».
                path = visual.shot_path(page, steps, dt)
        return str(path) if path.is_file() and _within(path, roots) else fallback
    except Exception:  # pragma: no cover - a broken frame must not lose the entry
        return fallback


def _check_id(task_id: str) -> None:
    if not _ID_SCHEMA.fullmatch(task_id):
        raise ValueError(
            f"недопустимый id задания {task_id!r}: ожидается схема вида K-1 "
            "(без '/' и '..')"
        )


def _index_existing(target: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    """Index of an already published showcase when the raw ``tasks.json`` is gone.

    A published run carries pages and frames but not the raw transcript. Leaving
    the showcase empty and pruning the files was destroying the only evidence
    (R24); the published files themselves are a sufficient manifest for a
    read-only rebuild.
    """
    scores = {score.get("task_id"): score for score in data.get("scores") or []}
    pages = {path.stem: path for path in (target / "pages").glob("*.html")}
    frames = {path.stem: path for path in (target / "frames").glob("*.png")}
    ids = set(pages) | set(frames)

    def order(task_id: str) -> tuple[int, int, str]:
        category = (scores.get(task_id) or {}).get("category", task_id.split("-")[0])
        number = int(task_id.split("-")[-1]) if task_id.split("-")[-1].isdigit() else 0
        rank = (core.SCORED_CATEGORIES.index(category) if category in core.SCORED_CATEGORIES
                else len(core.SCORED_CATEGORIES) + (core.SHOWCASE_CATEGORIES.index(category)
                                                    if category in core.SHOWCASE_CATEGORIES
                                                    else len(core.SHOWCASE_CATEGORIES)))
        return rank, number, task_id

    index: list[dict[str, Any]] = []
    for task_id in sorted(ids, key=order):
        score = scores.get(task_id) or {}
        record: dict[str, Any] = {
            "id": task_id,
            "category": score.get("category") or task_id.split("-")[0],
            "title": "",
            "score": score,
            "spend": {},
        }
        if task_id in pages:
            record["page"] = f"pages/{pages[task_id].name}"
        if task_id in frames:
            record["frame"] = f"frames/{frames[task_id].name}"
        index.append(record)
    return index


def _publish_into(data: dict[str, Any], run_dir: Path, target: Path) -> dict[str, Any]:
    """Build pages and frames in ``target``; the caller swaps it in afterwards."""
    pages = target / "pages"
    frames = target / "frames"
    pages.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    kept_pages: set[str] = set()
    kept_frames: set[str] = set()
    for item in showcase_entries(data, run_dir):
        task_id = item["id"]
        _check_id(task_id)
        record = {"id": task_id, "category": item["category"], "title": item["title"],
                  "score": item["score"], "spend": item["spend"]}
        if item["html"]:
            page = pages / f"{task_id}.html"
            page.write_text(scene_page(item, target, task_id), encoding="utf-8")
            record["page"] = f"pages/{task_id}.html"
            kept_pages.add(page.name)
        shot = Path(item["shot"]) if item["shot"] else None
        if shot and shot.is_file():
            frame = frames / f"{task_id}.png"
            shutil.copyfile(shot, frame)
            record["frame"] = f"frames/{task_id}.png"
            kept_frames.add(frame.name)
        index.append(record)
    # Republish must be idempotent: a scene that lost its page or frame (an
    # artefact rotated away, a shot purged) otherwise kept the *previous*
    # publish's file, and the gallery showed a live page the run no longer has.
    _prune(pages, kept_pages)
    _prune(frames, kept_frames)
    return {"model": (data.get("config") or {}).get("model", run_dir.name), "dir": run_dir.name, "tasks": index}


def materialise(run_dir: Path, target: Path) -> dict[str, Any]:
    """Copy frames and pages next to the gallery; returns per-model index.

    The library is copied once per report: an answer page imports
    ``./vendor/three.module.js``, so the copied page must sit next to a folder
    with the same name — otherwise the live view would need the network.

    A complete run (``tasks.json`` present) is built in a staging folder and then
    swapped in, so a failure halfway never leaves a half-replaced showcase. A
    published run without ``tasks.json`` is read-only: its existing pages and
    frames are indexed in place instead of being pruned into nothing (R24).
    """
    from .report import load_run

    data = load_run(run_dir)
    run_dir = Path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_vendor(target.parent.parent)  # <report>/_showcase/<model> -> <report>
    if "tasks" not in data:
        return {
            "model": (data.get("config") or {}).get("model", run_dir.name),
            "dir": run_dir.name,
            "tasks": _index_existing(target, data) if target.is_dir() else [],
        }

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        result = _publish_into(data, run_dir, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    backup = target.with_name(f"{target.name}.old")
    shutil.rmtree(backup, ignore_errors=True)
    try:
        if target.exists():
            target.rename(backup)
        staging.rename(target)
    except Exception:
        if not target.exists() and backup.exists():
            backup.rename(target)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    return result


def _prune(folder: Path, kept: set[str]) -> None:
    """Drop files a republish no longer produces, leaving foreign ones alone."""
    for path in folder.iterdir():
        if path.is_file() and path.suffix in {".html", ".png"} and path.name not in kept:
            path.unlink()


def ensure_vendor(report_dir: Path) -> None:
    """Положить рядом с отчётом всё, что нужно его страницам.

    Страницы сцен собираются standalone-версией (см. :func:`scene_page`): они
    грузят классический бандл ``_vendor/three.global.js``, собранный из
    пришпиленной ``vendor/three/``. Поэтому копии ES-модулей рядом с каждой
    моделью не нужны — это экономит около двух мегабайт на модель.
    """
    _write_three_bundle(report_dir)


def _write_three_bundle(report_dir: Path) -> None:
    """Один бандл three.js на отчёт: страницы сцен ссылаются на него."""
    from .browser import VENDOR

    bundle = report_dir / ASSET_DIR / "three.global.js"
    if bundle.is_file():
        return
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(three_bundle(VENDOR), encoding="utf-8")
    # Один классический скрипт на весь отчёт: под file:// внешний ES-модуль не
    # грузится (CORS для origin null), и без него живая сцена не запускается.

