"""Headless-browser rendering for the visual block (category K).

The visual scenarios are the only ones whose result cannot be read from text:
the model writes a Three.js page, so the evaluator has to *run* it. This module
owns that step and nothing else — how a scenario is scored lives in
``tasks/visual_*.py``.

Shared behavior across modes:

* the page is served from ``/app.html`` on a loopback port, with the pinned
  library from ``/vendor/``;
* the default viewport is 800x600 at device pixel ratio 1;
* network requests outside the loopback server (and ``data:`` URLs) are refused
  and reported, so a page cannot silently depend on a backend resource.

Mode-specific behavior:

* category K's ``fixed``/``time`` modes step ``window.__step(dt)`` and compare
  repeatable frames;
* the separate ``live`` mode behaves like an ordinary browser page, settles
  briefly after DOMContentLoaded, and samples twice using wall-clock time
  without requiring page hooks or repeatable animation frames.

When Playwright (or its Chromium) is not installed the scenario is reported as
unverified — the same rule as a missing Go toolchain: not verified is not a
zero.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import hashlib
import http.server
import json
import multiprocessing
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from . import telemetry

REPO = Path(__file__).resolve().parent.parent


def _asset(*parts: str) -> Path:
    """Vendored resources from a checkout, else from a wheel's data files.

    ``pyproject.toml`` installs them under ``<prefix>/vendor``; a source checkout
    keeps them in ``<repo>/vendor``. Looking up the prefix as well is what makes
    an isolated wheel install usable (R18).
    """
    candidates = [REPO / "vendor" / Path(*parts), Path(sys.prefix) / "vendor" / Path(*parts)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


VENDOR = _asset("three")
PLOTLY = _asset("plotly")
APP_NAME = "app.html"
DEFAULT_VIEWPORT = (800, 600)
#: Арт-блок рисует крупнее: кадр 1920x1080, чтобы сцены было видно на большом
#: экране и в записи. Категория K остаётся на 800x600 — там важны координаты.
ART_VIEWPORT = (1920, 1080)
DEFAULT_TIMEOUT_MS = 20_000
#: A page hook (``__ready``/``__probe``/``__step``) runs JavaScript in the page
#: thread, so Playwright's own per-call timeout cannot pre-empt a busy loop.
#: The whole render therefore runs in a separate process that is killed at the
#: absolute deadline; this grace is the IPC round-trip after it.
DEADLINE_GRACE_S = 0.04
WORKER_START_TIMEOUT_S = 60.0
#: After the deadline the parent waits this long for the child to move to the
#: next stage: an awaited Playwright timeout fires with a little jitter, and
#: the frame that follows it must still be taken. A hook that never returns
#: never bumps the progress counter and is killed after the same window.
STALL_S = 0.05
#: A hook that starts after the render deadline (the previous hook consumed
#: the whole budget, e.g. a page without ``__ready``) still gets this much time
#: to answer, so a working probe is not cut off together with the timeout.
HOOK_CONTINUATION_S = 3.0
#: Absolute cap on top of the render deadline. Screenshot stages budget
#: ``SCREENSHOT_TIMEOUT_MS`` each, so a Chromium that stops answering could
#: otherwise extend one render far beyond the announced deadline; this cap is
#: the point where the worker is killed regardless of stage.
HARD_GRACE_S = 20.0
#: Stage ids shared with the parent so it knows what it is waiting on.
STAGE_IDLE, STAGE_SETUP, STAGE_NAV, STAGE_HOOK, STAGE_SHOT = 0, 1, 2, 3, 4
#: Browser launch and context creation are infrastructure, not page work: a
#: cold Chromium start must not eat the hook budget.
SETUP_TIMEOUT_S = 60.0
#: Скриншот не наследует таймаут запроса: под параллельной нагрузкой (догрузка
#: сьюта рисует кадры в соседнем процессе) сцена 1920x1080 не укладывалась в
#: дефолтные 30 с Playwright и весь рендер падал «TimeoutError: Page.screenshot».
SCREENSHOT_TIMEOUT_MS = 90_000

#: Deterministic frame clock shared with ``video/tools/record_clip.py``. Scoring
#: renders install it before page scripts so two loads of a stable page stay
#: byte-identical; live art mode leaves the wall clock alone.
FRAME_CLOCK_TEMPLATE = r"""
(() => {
  if (window.__recordClock) return;
  const FPS = __FPS__;
  const STEP_MS = 1000 / FPS;
  const EPOCH = 946684800000;
  let frame = -1;
  let now = 0;
  let nextId = 1;
  let renderCount = 0;
  let lastRender = null;
  const rafs = new Map();
  const timers = new Map();
  const animationOrigins = new WeakMap();
  const unsupported = new Set();

  const schedule = (callback, delay, interval, args) => {
    const id = nextId++;
    const wait = Math.max(interval ? 1 : 0, Number(delay) || 0);
    timers.set(id, {callback, args, due: now + wait, interval: interval ? wait : 0});
    return id;
  };
  const invokeTimer = (timer) => {
    if (typeof timer.callback === 'function') {
      timer.callback.apply(window, timer.args);
    } else {
      (0, eval)(String(timer.callback));
    }
  };
  const runTimers = (through) => {
    let count = 0;
    while (true) {
      let dueId = 0;
      let dueAt = Infinity;
      for (const [id, timer] of timers) {
        if (timer.due < dueAt) { dueId = id; dueAt = timer.due; }
      }
      if (!dueId || dueAt > through) break;
      if (++count > 10000) throw new Error('recording clock timer runaway');
      const timer = timers.get(dueId);
      now = dueAt;
      if (timer.interval) timer.due += timer.interval;
      else timers.delete(dueId);
      invokeTimer(timer);
    }
    now = through;
  };
  const syncAnimations = () => {
    if (typeof document.getAnimations !== 'function') return;
    for (const animation of document.getAnimations()) {
      try {
        if (!animationOrigins.has(animation)) {
          animationOrigins.set(animation, now);
          animation.pause();
          animation.currentTime = 0;
        } else {
          animation.pause();
          animation.currentTime = now - animationOrigins.get(animation);
        }
      } catch (_) {
        unsupported.add('a Web Animations animation could not be driven by the recording clock');
      }
    }
  };
  const advanceFrame = () => {
    frame += 1;
    now = frame * STEP_MS;
    runTimers(now);
    syncAnimations();
    const callbacks = Array.from(rafs.values());
    rafs.clear();
    for (const callback of callbacks) callback(now);
    runTimers(now);
    syncAnimations();
    if (document.querySelector('video') &&
        typeof HTMLVideoElement !== 'undefined' &&
        HTMLVideoElement.prototype.requestVideoFrameCallback) {
      unsupported.add('video-frame callbacks follow the browser media clock');
    }
    return rafs.size > 0;
  };
  const wrapRenderer = (three) => {
    const Original = three && three.WebGLRenderer;
    if (typeof Original !== 'function' || Original.__recordClipWrapped) return;
    const Wrapped = function (...args) {
      const instance = Reflect.construct(
        Original, args, new.target === Wrapped ? Original : new.target
      );
      const render = instance.render;
      if (typeof render === 'function') {
        instance.render = function (...renderArgs) {
          lastRender = {instance: this, args: renderArgs};
          renderCount += 1;
          return Reflect.apply(render, this, renderArgs);
        };
      }
      return instance;
    };
    Wrapped.prototype = Original.prototype;
    Object.setPrototypeOf(Wrapped, Original);
    Wrapped.__recordClipWrapped = true;
    try { three.WebGLRenderer = Wrapped; }
    catch (_) { unsupported.add('a renderer constructor could not be hooked for redraw'); }
  };
  try {
    let three = window.THREE;
    Object.defineProperty(window, 'THREE', {
      configurable: true,
      enumerable: true,
      get: () => three,
      set: (value) => { three = value; wrapRenderer(value); }
    });
    wrapRenderer(three);
  } catch (_) {
    unsupported.add('the global THREE renderer could not be hooked for redraw');
  }

  const RealDate = Date;
  function RecordingDate(...args) {
    if (!new.target) return new RealDate(EPOCH + now).toString();
    return Reflect.construct(RealDate, args.length ? args : [EPOCH + now], new.target);
  }
  RecordingDate.prototype = RealDate.prototype;
  Object.setPrototypeOf(RecordingDate, RealDate);
  RecordingDate.now = () => EPOCH + now;
  RecordingDate.parse = RealDate.parse;
  RecordingDate.UTC = RealDate.UTC;
  try { window.Date = RecordingDate; }
  catch (_) { unsupported.add('the page Date constructor could not be controlled'); }
  try {
    Object.defineProperty(performance, 'now', {configurable: true, value: () => now});
  } catch (_) {
    try {
      Object.defineProperty(Object.getPrototypeOf(performance), 'now', {
        configurable: true, value: () => now
      });
    } catch (_) {
      unsupported.add('performance.now could not be controlled');
    }
  }
  try { Object.defineProperty(performance, 'timeOrigin', {configurable: true, value: EPOCH}); }
  catch (_) {}

  window.requestAnimationFrame = (callback) => {
    if (typeof callback !== 'function') throw new TypeError('requestAnimationFrame callback must be callable');
    const id = nextId++;
    rafs.set(id, callback);
    return id;
  };
  window.cancelAnimationFrame = (id) => { rafs.delete(id); };
  window.setTimeout = (callback, delay, ...args) => schedule(callback, delay, 0, args);
  window.setInterval = (callback, delay, ...args) => schedule(callback, delay, 1, args);
  window.clearTimeout = (id) => { timers.delete(id); };
  window.clearInterval = window.clearTimeout;
  window.requestIdleCallback = (callback, options = {}) =>
    schedule(() => callback({didTimeout: false, timeRemaining: () => 0}), options.timeout || 0, 0, []);
  window.cancelIdleCallback = window.clearTimeout;

  window.__recordClock = {
    get time() { return now; },
    get frame() { return frame; },
    warmup: (count) => { for (let i = 0; i < count; i++) advanceFrame(); },
    capture: (dt) => {
      const beforeRender = renderCount;
      const selfRaf = advanceFrame();
      if (!selfRaf && typeof window.__step === 'function') window.__step(dt);
      if (typeof window.__probe === 'function') window.__probe();
      syncAnimations();
      if (renderCount === beforeRender && lastRender) {
        const {instance, args} = lastRender;
        instance.render(...args);
      }
      return {selfRaf, frame, time: now, mode: selfRaf ? 'native-raf' : typeof window.__step === 'function' ? 'explicit-step' : 'static-or-css'};
    },
    warnings: () => Array.from(unsupported)
  };
})();
"""


def frame_clock(fps: int = 60) -> str:
    """Injected clock at the requested rate: one frame per step, no wall clock."""
    return FRAME_CLOCK_TEMPLATE.replace("__FPS__", str(int(fps)))


#: Two animation frames plus font readiness, awaited before every screenshot.
SCREENSHOT_SETTLE_JS = """async () => {
  if (document.fonts && document.fonts.ready) await document.fonts.ready;
  if (window.__recordClock) {
    await window.__recordClock.warmup(2);
  } else {
    await new Promise((resolve) => requestAnimationFrame(resolve));
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
}"""


#: ``live`` samples a normally loaded page after a short settle, then once more
#: after a fixed interval. This is observational, not an animation score.
LIVE_SETTLE_MS = 250
LIVE_SAMPLE_INTERVAL_MS = 250

#: Chromium is driven headless with software GL: a benchmark may not require a
#: display, and the same flags must hold on a headless Linux box and on a
#: workstation with a GPU.
LAUNCH_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--disable-lcd-text",
    "--force-device-scale-factor=1",
    "--hide-scrollbars",
]


#: Classes of a refused request. HTTP routing does not govern the WebSocket
#: handshake, so both paths exist — but they are one fault and share one
#: classifier, otherwise the same escape would cost a different score.
BLOCKED_HTTP = "http"
BLOCKED_WS = "websocket"


def external_class(entry: str) -> str:
    """Class of one refused request; the only classifier both paths use."""
    return BLOCKED_WS if entry.startswith(f"{BLOCKED_WS}: ") else BLOCKED_HTTP


def external_kinds(entries: Any) -> dict[str, int]:
    """Refused requests per class: ``{"http": n, "websocket": m}``."""
    kinds = {BLOCKED_HTTP: 0, BLOCKED_WS: 0}
    for entry in entries or ():
        kinds[external_class(str(entry))] += 1
    return kinds


def describe_externals(entries: Any) -> str:
    """One line naming both classes, so a report cannot blame one silently."""
    kinds = external_kinds(entries)
    listed = ", ".join(
        f"{kind} x{count}" for kind, count in kinds.items() if count
    )
    urls = "; ".join(str(entry)[:120] for entry in list(entries or ())[:3])
    return f"внешние запросы запрещены: {listed} ({urls})"


@dataclass
class RenderResult:
    """Everything the evaluator may look at. ``ok`` is about rendering."""

    ok: bool
    reason: str = ""
    failure: str | None = None
    """Taxonomy of a failed render: the worker itself is unavailable, or the
    candidate page exhausted the declared deadline. ``None`` on success and on
    an ordinary page defect (markup that is not a document)."""
    note: str = ""
    errors: list[str] = field(default_factory=list)
    external: list[str] = field(default_factory=list)
    probe: dict[str, Any] | None = None
    stepped: dict[str, Any] | None = None
    pixels: dict[str, Any] | None = None
    pixels_again: dict[str, Any] | None = None
    shot: bytes | None = None
    shot_sha: str = ""
    shot_sha_again: str = ""
    animated: bool | None = None
    """``True``/``False`` when the render asked for an animated scene."""

    probe_ok: bool = True
    """``False`` when the page drew a frame but never answered ``__probe()``.

    The frame is still usable (and shown), but no state can be compared, so a
    category that scores coordinates must treat it as a protocol failure.
    """
    ready_ok: bool = True
    """``False`` when ``window.__ready`` never became ``true``."""
    step_ok: bool = True
    """``False`` when ``__step`` was missing or threw while stepping."""
    shot_later_sha: str = ""
    duration_s: float = 0.0

    @property
    def same_frame(self) -> bool:
        """Two loads produced byte-identical frames."""
        return bool(self.shot_sha) and self.shot_sha == self.shot_sha_again

    @property
    def protocol_ok(self) -> bool:
        """The whole page protocol (``__ready``, ``__probe``, ``__step``) held."""
        return self.ready_ok and self.probe_ok and self.step_ok

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "errors": self.errors[:20],
            "external": self.external[:20],
            "probe": self.probe,
            "stepped": self.stepped,
            "pixels": self.pixels,
            "probe_ok": self.probe_ok,
            "ready_ok": self.ready_ok,
            "step_ok": self.step_ok,
            "same_frame": self.same_frame,
            "animated": self.animated,
            "shot_sha_later": self.shot_later_sha,
            "duration_s": round(self.duration_s, 3),
        }


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def availability() -> tuple[bool, str]:
    """``(usable, reason)`` — checked once per process."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the machine
        return False, f"playwright не установлен ({exc.__class__.__name__})"
    default_cache = (
        Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "ms-playwright"
        if os.name == "nt" else Path.home() / ".cache" / "ms-playwright"
    )
    cache = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or default_cache)
    mac_cache = Path.home() / "Library" / "Caches" / "ms-playwright"
    candidates = [
        root / entry
        for root in (cache, mac_cache)
        for entry in (root.iterdir() if root.is_dir() else [])
        if entry.name.startswith("chromium")
    ]
    # A chromium directory without its executable is a broken install, not a
    # usable one: a bare ``is_dir`` check used to pass and then ``launch``
    # failed inside ``render``, scoring the model zero for stand breakage (V6).
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            executable = Path(pw.chromium.executable_path)
        found = executable.is_file()
        if not found:
            return False, f"chromium есть, но executable недоступен: {executable}"
    except Exception as exc:  # pragma: no cover - depends on the machine
        return False, f"chromium не запускается ({exc.__class__.__name__})"
    if not candidates:
        return False, "chromium для playwright не установлен (playwright install chromium)"
    if not (VENDOR / "three.module.js").is_file():
        return False, f"нет библиотеки {VENDOR / 'three.module.js'}"
    return True, ""


# --------------------------------------------------------------------------- #
# Static server
# --------------------------------------------------------------------------- #


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the candidate page plus the pinned library, and stays quiet."""

    vendor_root: Path | None = None

    def log_message(self, *args: Any) -> None:  # pragma: no cover - noise
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        raw = unquote(urlparse(path).path)
        parts = [part for part in raw.split("/") if part and part not in (".",)]
        if ".." in parts:
            return super().translate_path("/__missing__")
        root = type(self).vendor_root
        if parts and parts[0] == "vendor" and root is not None:
            return str(root.joinpath(*parts[1:]))
        return super().translate_path(path)


_VENDOR_COPY: Path | None = None


def _prepare_vendor_tree() -> Path:
    """One prepared three.js tree per process, reused by every render."""
    global _VENDOR_COPY
    if _VENDOR_COPY is not None and _VENDOR_COPY.is_dir():
        return _VENDOR_COPY
    prepared = Path(tempfile.mkdtemp(prefix="dailybench-vendor-")) / "vendor"
    try:
        prepared.symlink_to(VENDOR, target_is_directory=True)
    except OSError:
        shutil.copytree(VENDOR, prepared)
    _VENDOR_COPY = prepared
    return prepared


@contextlib.contextmanager
def _served(html: str):
    """Serve ``html`` as /app.html with ./vendor/ next to it."""
    root = Path(tempfile.mkdtemp(prefix="dailybench-visual-"))
    try:
        (root / APP_NAME).write_text(html, encoding="utf-8")
        prepared = _prepare_vendor_tree()
        vendor = root / "vendor"
        try:
            vendor.symlink_to(prepared, target_is_directory=True)
        except OSError:
            # The handler serves the prepared tree directly, so a machine
            # without symlinks does not recopy three.js into every render root.
            pass
        handler_cls = type("_BoundHandler", (_Handler,), {"vendor_root": prepared})
        handler = functools.partial(handler_cls, directory=str(root))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            yield f"http://127.0.0.1:{port}/{APP_NAME}", port
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Playwright session
# --------------------------------------------------------------------------- #

_SESSION: dict[str, Any] = {}


def _browser() -> Any:
    from playwright.sync_api import sync_playwright

    if "pw" not in _SESSION:
        pw = sync_playwright().start()
        _SESSION["pw"] = pw
        _SESSION["browser"] = pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
    return _SESSION["browser"]


def _close() -> None:  # pragma: no cover - interpreter shutdown
    browser = _SESSION.pop("browser", None)
    pw = _SESSION.pop("pw", None)
    if browser is not None:
        with contextlib.suppress(Exception):
            browser.close()
    if pw is not None:
        with contextlib.suppress(Exception):
            pw.stop()


atexit.register(_close)


# --------------------------------------------------------------------------- #
# Page probes (run inside the browser)
# --------------------------------------------------------------------------- #

PIXEL_PROBE = """
() => {
  // Measure what the viewer sees: the largest canvas actually rendered on
  // screen. A hidden (display:none / zero-area) canvas used to be measured
  // all the same, so a page could pass image checks with an empty frame.
  const visible = Array.from(document.querySelectorAll('canvas')).filter((el) => {
    if (!el.width || !el.height) return false;
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || +style.opacity === 0) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 1 && rect.height > 1;
  });
  const c = visible.sort((a, b) => {
    const area = (el) => el.getBoundingClientRect().width * el.getBoundingClientRect().height;
    return area(b) - area(a);
  })[0] || null;
  if (!c) return null;
  const off = document.createElement('canvas');
  off.width = c.width; off.height = c.height;
  const ctx = off.getContext('2d');
  ctx.drawImage(c, 0, 0);
  const w = off.width, h = off.height;
  const data = ctx.getImageData(0, 0, w, h).data;
  const bg = [data[0], data[1], data[2]];
  const colors = new Set();
  const regionSum = Array.from({length: 9}, () => [0, 0, 0, 0]);
  // Objects are usually small: a per-region mean would drown them, so the
  // classes count pixels instead and report where those pixels are.
  const classes = {
    warm: {n: 0, x: 0, y: 0},
    cool: {n: 0, x: 0, y: 0},
    bright: {n: 0, x: 0, y: 0},
    dark: {n: 0, x: 0, y: 0},
  };
  // Colour histogram over a coarse grid: the art checks compare declared
  // palettes against what actually covers the frame.
  const bins = new Map();
  // Declared-palette matching needs tighter buckets than the statistics do: a
  // 32-step bucket bleeds into the neighbouring hue and a dark sky would never
  // match its own colour.
  const fine = new Map();
  const luma = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
  // Horizontal scan lines: a sky/horizon scenario has to show a gradient, a
  // flat fill would not.
  const rows = new Float32Array(h);
  const satRows = new Float32Array(h);
  // Column profile: an object placed at a declared x must be findable in the
  // columns, which a 3x3 grid is too coarse for.
  const cols = new Float32Array(w);
  const colN = new Float32Array(w);
  let edges = 0, edgePairs = 0, symSum = 0;
  let shadePairs = 0, flatPairs = 0, smoothPairs = 0;
  let changed = 0, lum = 0, sat = 0, n = 0;
  // Шаг выборки держим примерно в 2 px независимо от разрешения: на 1920x1080
  // фиксированные 2 px съедают слишком много пар, и тонкие линии (лучи, кабели)
  // перестают попадать в метрику границ. Шаг округляется до чётного.
  const st = Math.max(2, Math.round(w / 800) * 2);
  // Half-stride row offset: a regular grid aliases with patterns that repeat at
  // the stride (thin stripes shift the sampled share from 100% to 0% with a
  // one-pixel move). Staggering alternate rows by st/2 halves that dependence.
  let row = 0;
  for (let y = 0; y < h; y += st, row += 1) {
    let rowLum = 0, rowSat = 0, rowN = 0;
    const x0 = (row % 2) * (st >> 1);
    for (let x = x0; x < w; x += st) {

      const i = (y * w + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const mean = (r + g + b) / 3;
      lum += mean;
      sat += mx === 0 ? 0 : (mx - mn) / mx;
      n += 1;
      rowLum += mean;
      cols[x] += mean;
      colN[x] += 1;
      rowSat += mx === 0 ? 0 : (mx - mn) / mx;
      rowN += 1;
      if (Math.abs(r - bg[0]) + Math.abs(g - bg[1]) + Math.abs(b - bg[2]) > 24) changed += 1;
      colors.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4));
      const key = ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5);
      bins.set(key, (bins.get(key) || 0) + 1);
      const fkey = (r >> 3) * 1024 + (g >> 3) * 32 + (b >> 3);
      fine.set(fkey, (fine.get(fkey) || 0) + 1);
      for (const [name, hit] of [
        ['warm', r > b + 25 && r > 60],
        ['cool', b > r + 15],
        ['bright', mean > 110],
        ['dark', mean < 60],
      ]) {
        if (hit) { const c2 = classes[name]; c2.n += 1; c2.x += x; c2.y += y; }
      }
      const rx = Math.min(2, Math.floor(x * 3 / w)), ry = Math.min(2, Math.floor(y * 3 / h));
      const acc = regionSum[ry * 3 + rx];
      acc[0] += r; acc[1] += g; acc[2] += b; acc[3] += 1;
      // Edge density on the sample grid, and a mirror test against the row's
      // own centre — both are cheap and catch flat fills and white noise.
      const jx = (x + st < w) ? i + st * 4 : i;
      const jy = (y + st < h) ? i + w * st * 4 : i;
      const l0 = luma(r, g, b);
      edgePairs += 1;
      const dx = Math.abs(l0 - luma(data[jx], data[jx + 1], data[jx + 2]));
      const dy = Math.abs(l0 - luma(data[jy], data[jy + 1], data[jy + 2]));
      // Порог перепада растёт вместе с шагом выборки: иначе на 1920x1080 те же
      // самые детали дают вдвое меньше «границ», и метрика зависит от разрешения.
      const edgeStep = 16 * st / 2;
      if (dx > edgeStep) edges += 1;
      if (dy > edgeStep) edges += 1;
      // Объём: плавный перепад яркости между соседями (1..порог) — это затенение
      // поверхности или градиент; ровно 0 — плоская заливка, больше — контур.
      for (const d of [dx, dy]) {
        if (d === 0) flatPairs += 1;
        else if (d <= edgeStep) smoothPairs += 1;
      }
      shadePairs += 2;
      // mirror: compare with the opposite sample in the same row
      const mx2 = (w - st - x);
      const k = (y * w + mx2) * 4;
      symSum += Math.abs(l0 - luma(data[k], data[k + 1], data[k + 2]));
    }
    rows[y] = rowLum / rowN;
    satRows[y] = rowSat / rowN;
    // Строки между шагами выборки повторяют предыдущую: иначе на 1920x1080
    // половина профиля остаётся нулевой и полосы идут через одну пустую.
    for (let k = y + 1; k < Math.min(h, y + st); k++) { rows[k] = rows[y]; satRows[k] = satRows[y]; }
  }
  const round4 = (v) => Math.round(v * 1e4) / 1e4;
  const pack = (cls) => ({
    fraction: n ? round4(cls.n / n) : 0,
    count: cls.n,
    centroid: cls.n ? [round4(cls.x / cls.n / w), round4(cls.y / cls.n / h)] : null,
  });
  const hsv = new Map();
  for (let y = 0; y < h; y += st) {
    for (let x = 0; x < w; x += st) {
      const i = (y * w + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const span = mx - mn;
      let hue = 0;
      if (span > 0) {
        if (mx === r) hue = 60 * (((g - b) / span) % 6);
        else if (mx === g) hue = 60 * ((b - r) / span + 2);
        else hue = 60 * ((r - g) / span + 4);
      }
      hue = (hue + 360) % 360;
      const sat = mx === 0 ? 0 : span / mx;
      const val = mx / 255;
      const hb = Math.min(11, Math.floor(hue / 30));
      const sb = Math.min(2, Math.floor(sat * 3));
      // 4 ступени яркости: по одной градиент неотличим от плоской заливки, а
      // именно градиент делает сцену объёмной.
      const vb = Math.min(3, Math.floor(val * 4));
      const key = (hb * 12) + (sb * 4) + vb;
      hsv.set(key, (hsv.get(key) || 0) + 1);
    }
  }
  const hueHistogram = [];
  for (const [key, count] of hsv) {
    const hb = Math.floor(key / 12), sb = Math.floor((key % 12) / 4), vb = key % 4;
    hueHistogram.push([hb * 30 + 15, (sb + 0.5) / 3, (vb + 0.5) / 4, round4(count / n)]);
  }
  hueHistogram.sort((a, b) => b[3] - a[3]);

  const fineTop = [];
  for (const [key, count] of fine) {
    // bucket centre: 5-bit channel + half a bucket
    fineTop.push([
      ((key >> 10) & 31) * 8 + 4,
      ((key >> 5) & 31) * 8 + 4,
      (key & 31) * 8 + 4,
      count,
    ]);
  }
  fineTop.sort((a, b) => b[3] - a[3]);
  const top = [];
  for (const [key, count] of bins) {
    top.push([((key >> 6) & 7) * 32 + 16, ((key >> 3) & 7) * 32 + 16, (key & 7) * 32 + 16, count]);
  }
  top.sort((a, b) => b[3] - a[3]);
  const columnProfile = [];
  for (let band = 0; band < 8; band++) {
    const from = Math.floor(band * w / 8), to = Math.floor((band + 1) * w / 8);
    let acc = 0, m = 0;
    for (let x = from; x < to; x += st) { acc += colN[x] ? cols[x] / colN[x] : 0; m += 1; }
    columnProfile.push(m ? Math.round(acc / m * 10) / 10 : 0);
  }

  const profile = [];
  for (let band = 0; band < 8; band++) {
    const from = Math.floor(band * h / 8), to = Math.floor((band + 1) * h / 8);
    let acc = 0, satAcc = 0, m = 0;
    for (let y = from; y < to; y += st) { acc += rows[y]; satAcc += satRows[y]; m += 1; }
    profile.push({ lum: m ? Math.round(acc / m * 10) / 10 : 0, sat: m ? Math.round(satAcc / m * 1000) / 1000 : 0 });
  }
  return {
    width: w, height: h,
    background: bg,
    mean: n ? Math.round(lum / n * 10) / 10 : 0,
    saturation: n ? Math.round(sat / n * 1000) / 1000 : 0,
    changed_fraction: n ? Math.round(changed / n * 10000) / 10000 : 0,
    distinct_colors: colors.size,
    warm: pack(classes.warm),
    cool: pack(classes.cool),
    bright: pack(classes.bright),
    dark: pack(classes.dark),
    histogram: top.slice(0, 16).map(([r, g, b, count]) => [r, g, b, round4(count / n)]),
    histogram_fine: fineTop.slice(0, 96).map(([r, g, b, count]) => [r, g, b, round4(count / n)]),
    hue_histogram: hueHistogram,
    // Объём: доля соседних пар пикселей с плавным перепадом яркости — так
    // выглядит затенение поверхности (сфера, откос, блик). Плоская заливка и
    // градиент-заливка дают ~0, сцена с формой — заметно больше.
    shade: shadePairs ? round4(smoothPairs / shadePairs) : 0,
    flat: shadePairs ? round4(flatPairs / shadePairs) : 0,
    volume: (() => {
      const tones = new Map();
      for (const [hue, sat, val, fraction] of hueHistogram) {
        // 0.1% кадра: в тёмных космических сценах значимая деталь (планета,
        // станция) занимает меньше процента, и порог 0.4% её терял.
        if (fraction < 0.001) continue;
        const key = Math.round(hue / 30);
        const slot = tones.get(key) || {min: 1, max: 0, share: 0};
        slot.min = Math.min(slot.min, val);
        slot.max = Math.max(slot.max, val);
        slot.share += fraction;
        tones.set(key, slot);
      }
      let shadedShare = 0, spread = 0;
      for (const slot of tones.values()) {
        if (slot.max > slot.min) { shadedShare += slot.share; spread += slot.share * (slot.max - slot.min); }
      }
      // Если значимых тонов мало (тёмный космос), объём оцениваем по разбросу
      // яркости всей гистограммы: у сцены с формой он заметно больше нуля.
      if (tones.size < 4) {
        let lo = 1, hi = 0;
        for (const [, , val, fraction] of hueHistogram) {
          if (fraction < 0.001) continue;
          lo = Math.min(lo, val); hi = Math.max(hi, val);
        }
        shadedShare = Math.max(shadedShare, hi - lo);
      }
      return {tones: tones.size, shaded_share: round4(shadedShare), spread: round4(spread)};
    })(),
    // Профили яркости по полосам кадра: по ним проверяются горизонт, градиент
    // неба и положение солнца/предмета в кадре.
    profile,
    columns: columnProfile,
    edges: edgePairs ? round4(edges / edgePairs) : 0,
    // Зеркало — средняя разница яркости с отражённой точкой строки, и
    // сетка 3x3 средних цветов областей: по ней проверяют симметрию и то, что
    // области кадра различаются между собой.
    symmetry: round4(symSum / Math.max(1, edgePairs)),
    mirror_difference: round4(symSum / Math.max(1, edgePairs)),
    regions: regionSum.map((acc) => [
      Math.round(acc[0] / Math.max(1, acc[3])), Math.round(acc[1] / Math.max(1, acc[3])),
      Math.round(acc[2] / Math.max(1, acc[3]))]),
  };
}
"""


FRAME_STATS_VERSION = 1
"""Schema of :func:`frame_stats`. Bumped when the measurement changes."""


def frame_stats(shot: bytes | None) -> dict[str, Any]:
    """Frame statistics measured from the PNG, outside the page (H17).

    ``PIXEL_PROBE`` reads the canvas from inside the page's own realm: a black
    overlay above the canvas, a CSS transform or a faked ``getContext`` can make
    that payload disagree with what the screenshot shows. The screenshot is the
    evidence, so the statistics are computed from its bytes in Python — the
    page's realm contributes nothing but the numbers of its ``__probe()``.

    The output keeps the key shape the image predicates read (``warm``,
    ``regions``, ``profile``, ``edges`` …), so a predicate cannot tell which
    side of the page computed it. An unreadable frame yields ``{}``.
    """
    if not shot:
        return {}
    import io

    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(shot)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.int64)
    except Exception:
        return {}
    h, w = rgb.shape[:2]
    if not w or not h:
        return {}
    # Тот же шаг выборки, что у in-page пробы: около 2 px на любом разрешении,
    # с чередованием строк, чтобы регулярная сетка не алиасила с рисунком.
    st = max(2, round(w / 800) * 2)
    ys: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    for index, y in enumerate(range(0, h, st)):
        offset = (st >> 1) * (index % 2)
        x = np.arange(offset, w, st)
        ys.append(np.full(x.size, y))
        xs.append(x)
    yy = np.concatenate(ys)
    xx = np.concatenate(xs)
    sample = rgb[yy, xx]
    r, g, b = sample[:, 0], sample[:, 1], sample[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    span = mx - mn
    mean = (r + g + b) / 3.0
    sat = np.where(mx == 0, 0.0, span / np.maximum(mx, 1))
    n = int(sample.shape[0])
    if not n:
        return {}
    bg = [int(rgb[0, 0, 0]), int(rgb[0, 0, 1]), int(rgb[0, 0, 2])]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    round4 = lambda value: round(float(value), 4)  # noqa: E731

    def pack(hit: np.ndarray) -> dict[str, Any]:
        count = int(hit.sum())
        return {
            "fraction": round4(count / n) if n else 0.0,
            "count": count,
            "centroid": (
                [round4(float(xx[hit].mean()) / w), round4(float(yy[hit].mean()) / h)]
                if count
                else None
            ),
        }

    warm = (r > b + 25) & (r > 60)
    cool = b > r + 15
    bright = mean > 110
    dark = mean < 60

    # Границы: перепад с соседом по шагу; порог растёт вместе с шагом, иначе
    # метрика зависит от разрешения.
    edge_step = 16 * st / 2
    jx = np.minimum(xx + st, w - 1)
    jy = np.minimum(yy + st, h - 1)
    dx = np.abs(luma - (0.2126 * rgb[yy, jx, 0] + 0.7152 * rgb[yy, jx, 1] + 0.0722 * rgb[yy, jx, 2]))
    dy = np.abs(luma - (0.2126 * rgb[jy, xx, 0] + 0.7152 * rgb[jy, xx, 1] + 0.0722 * rgb[jy, xx, 2]))
    edges = int((dx > edge_step).sum() + (dy > edge_step).sum())
    edge_pairs = 2 * n
    flat = int((dx == 0).sum() + (dy == 0).sum())
    smooth = int(((dx > 0) & (dx <= edge_step)).sum() + ((dy > 0) & (dy <= edge_step)).sum())

    mirrored = np.abs(
        luma
        - (
            0.2126 * rgb[yy, np.maximum(0, w - st - xx), 0]
            + 0.7152 * rgb[yy, np.maximum(0, w - st - xx), 1]
            + 0.0722 * rgb[yy, np.maximum(0, w - st - xx), 2]
        )
    )
    symmetry = float(mirrored.sum()) / max(1, edge_pairs)

    # Гистограммы: грубая для цвета, HSV для палитры, 5-битная для «тонких»
    # совпадений. Ключи считаются так же, как в in-page пробе.
    keys = ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5)
    coarse, coarse_counts = np.unique(keys, return_counts=True)
    fine = (r >> 3) * 1024 + (g >> 3) * 32 + (b >> 3)
    fine_keys, fine_counts = np.unique(fine, return_counts=True)
    distinct = int(np.unique(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)).size)

    hue = np.zeros(n)
    safe = span > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        hue = np.where(
            mx == r,
            60 * (((g - b) / np.maximum(span, 1)) % 6),
            np.where(
                mx == g,
                60 * ((b - r) / np.maximum(span, 1) + 2),
                60 * ((r - g) / np.maximum(span, 1) + 4),
            ),
        )
    hue = np.where(safe, (hue + 360) % 360, 0.0)
    val = mx / 255.0
    hb = np.minimum(11, (hue / 30).astype(int))
    sb = np.minimum(2, (sat * 3).astype(int))
    vb = np.minimum(3, (val * 4).astype(int))
    hsv_key = hb * 12 + sb * 4 + vb
    hsv_keys, hsv_counts = np.unique(hsv_key, return_counts=True)
    hue_histogram = sorted(
        [
            [
                float((key // 12) * 30 + 15),
                float(((key % 12) // 4 + 0.5) / 3),
                float((key % 4 + 0.5) / 4),
                round4(count / n),
            ]
            for key, count in zip(hsv_keys.tolist(), hsv_counts.tolist())
        ],
        key=lambda row: -row[3],
    )

    # Объём: доля плавных перепадов (затенение) против плоской заливки.
    tones: dict[int, dict[str, float]] = {}
    for hue_value, _sat, val_value, fraction in hue_histogram:
        if fraction < 0.001:
            continue
        slot = tones.setdefault(round(hue_value / 30), {"min": 1.0, "max": 0.0, "share": 0.0})
        slot["min"] = min(slot["min"], val_value)
        slot["max"] = max(slot["max"], val_value)
        slot["share"] += fraction
    shaded = sum(slot["share"] for slot in tones.values() if slot["max"] > slot["min"])
    spread = sum(
        slot["share"] * (slot["max"] - slot["min"])
        for slot in tones.values()
        if slot["max"] > slot["min"]
    )
    if len(tones) < 4:
        significant = [row[2] for row in hue_histogram if row[3] >= 0.001]
        shaded = max(shaded, (max(significant) - min(significant)) if significant else 0.0)

    # Профили по полосам и колонкам: горизонт, градиент и место объекта.
    profile = []
    columns = []
    for band in range(8):
        rows_in = (yy >= band * h // 8) & (yy < (band + 1) * h // 8)
        cols_in = (xx >= band * w // 8) & (xx < (band + 1) * w // 8)
        profile.append(
            {
                "lum": round(float(mean[rows_in].mean()) * 10) / 10 if rows_in.any() else 0,
                "sat": round(float(sat[rows_in].mean()) * 1000) / 1000 if rows_in.any() else 0,
            }
        )
        columns.append(round(float(mean[cols_in].mean()) * 10) / 10 if cols_in.any() else 0)

    region_sum = np.zeros((3, 3, 4), dtype=np.float64)
    rx = np.minimum(2, xx * 3 // w)
    ry = np.minimum(2, yy * 3 // h)
    flat_region = ry * 3 + rx
    for channel, values in enumerate((r, g, b)):
        np.add.at(region_sum.reshape(9, 4)[:, channel], flat_region, values)
    np.add.at(region_sum.reshape(9, 4)[:, 3], flat_region, 1)

    return {
        "width": int(w),
        "height": int(h),
        "background": bg,
        "mean": round(float(mean.mean()) * 10) / 10,
        "saturation": round(float(sat.mean()) * 1000) / 1000,
        "changed_fraction": round4(int((np.abs(r - bg[0]) + np.abs(g - bg[1]) + np.abs(b - bg[2]) > 24).sum()) / n),
        "distinct_colors": distinct,
        "warm": pack(warm),
        "cool": pack(cool),
        "bright": pack(bright),
        "dark": pack(dark),
        "histogram": [
            [
                int(((key >> 6) & 7) * 32 + 16),
                int(((key >> 3) & 7) * 32 + 16),
                int((key & 7) * 32 + 16),
                round4(count / n),
            ]
            for key, count in sorted(
                zip(coarse.tolist(), coarse_counts.tolist()), key=lambda pair: -pair[1]
            )[:16]
        ],
        "histogram_fine": [
            [
                int(((key >> 10) & 31) * 8 + 4),
                int(((key >> 5) & 31) * 8 + 4),
                int((key & 31) * 8 + 4),
                round4(count / n),
            ]
            for key, count in sorted(
                zip(fine_keys.tolist(), fine_counts.tolist()), key=lambda pair: -pair[1]
            )[:96]
        ],
        "hue_histogram": hue_histogram,
        "shade": round4(smooth / edge_pairs),
        "flat": round4(flat / edge_pairs),
        "volume": {"tones": len(tones), "shaded_share": round4(shaded), "spread": round4(spread)},
        "profile": profile,
        "columns": columns,
        "edges": round4(edges / edge_pairs),
        "symmetry": round4(symmetry),
        "mirror_difference": round4(symmetry),
        "regions": [
            [round(float(acc[0] / max(1, acc[3]))), round(float(acc[1] / max(1, acc[3]))), round(float(acc[2] / max(1, acc[3])))]
            for row in region_sum
            for acc in row
        ],
    }


def _budget_ms(deadline: float, timeout_ms: int) -> int:
    """Timeout for one awaited call under the absolute deadline.

    Only 90 % of the remaining budget is given to a single call, so its own
    timeout fires just *before* the deadline: the render still has time to
    move to the next stage instead of being killed between two calls. A call
    that starts after the deadline gets the continuation window.
    """
    left = (deadline - time.monotonic()) * 1000
    if left <= 0:
        return max(1, int(HOOK_CONTINUATION_S * 1000))
    return max(1, min(int(left * 0.9), timeout_ms))


def _request_allowed(url: str, allowed: str) -> bool:
    """Loopback and non-image data URLs stay; inline images and blobs do not."""
    if url.startswith(allowed):
        return True
    if url.startswith(("blob:", "data:image/")):
        return False
    return url.startswith("data:")


def _page_pass(
    url: str,
    port: int,
    steps: int,
    dt: float,
    viewport: tuple[int, int],
    timeout_ms: int,
    mode: str = "fixed",
    stage: Any = None,
    progress: Any = None,
    probe: bool = True,
) -> dict[str, Any]:
    """Load a page and capture its observable output.

    Every await is bounded by the same absolute ``deadline`` (monotonic
    seconds): a navigation timeout is only a part of the budget, and a hook
    that overruns is cut by the parent process rather than resetting it.

    ``fixed`` and ``time`` use the page's deterministic stepping protocol.
    ``live`` waits for DOMContentLoaded and a brief settle, then takes two
    screenshots of the same page at a fixed wall-clock interval. It never calls
    page-defined hooks and treats pixel statistics as optional.

    ``stage``/``progress`` are shared with the parent process: a stage change
    or a new call bumps the progress counter, so the parent can tell a hook
    that is merely slow to start from one that is hung.
    """

    def _mark(kind: int) -> None:
        if stage is not None:
            stage.value = kind
        if progress is not None:
            progress.value += 1

    _mark(STAGE_SETUP)
    context = _browser().new_context(
        viewport={"width": viewport[0], "height": viewport[1]},
        device_scale_factor=1,
        reduced_motion="no-preference" if mode == "live" else "reduce",
    )
    # Browser launch and context creation are infrastructure: the page budget
    # starts here, not when the job was dispatched.
    deadline = time.monotonic() + timeout_ms / 1000
    errors: list[str] = []
    external: list[str] = []
    allowed = f"http://127.0.0.1:{port}/"

    def _refuse(kind: str, url: str) -> None:
        """Record one refused request; both transports go through here."""
        external.append(f"{kind}: {url}")

    try:
        context.route(
            "**/*",
            lambda route: (
                route.continue_()
                if _request_allowed(route.request.url, allowed)
                else (_refuse(BLOCKED_HTTP, route.request.url), route.abort())
            ),
        )
        # HTTP routing does not govern the WebSocket handshake: a ws:// channel
        # used to bypass ``route`` entirely. A routed socket is never connected,
        # so no traffic leaves and the attempt is recorded in ``external``.
        context.route_web_socket("**/*", lambda ws: _refuse(BLOCKED_WS, ws.url))
        if mode != "live":
            context.add_init_script(frame_clock(60))
        page = context.new_page()
        page.on("console", lambda msg: errors.append(f"console: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        if mode == "live":
            navigated = True
            try:
                _mark(STAGE_NAV)
                page.goto(url, wait_until="domcontentloaded", timeout=_budget_ms(deadline, timeout_ms))
            except Exception as exc:
                errors.append(f"navigation: {exc.__class__.__name__}: {str(exc)[:200]}")
                navigated = False
            # A page that never reaches DOMContentLoaded is busy in its own
            # script: screenshots would each hang for their full timeout
            # (measured: +180 s on a `while (true)` page). The render ends
            # here — a hung page gets a timeout, not the whole suite.
            shot = None
            shot_later = None
            if navigated:
                try:
                    page.wait_for_timeout(_budget_ms(deadline, LIVE_SETTLE_MS))
                except Exception as exc:
                    errors.append(f"settle: {exc.__class__.__name__}: {str(exc)[:200]}")
                try:
                    page.evaluate(SCREENSHOT_SETTLE_JS)
                except Exception as exc:
                    errors.append(f"fonts: {exc.__class__.__name__}: {str(exc)[:200]}")
                try:
                    _mark(STAGE_SHOT)
                    shot = page.screenshot(type="png", timeout=SCREENSHOT_TIMEOUT_MS)
                except Exception as exc:
                    errors.append(f"screenshot: {exc.__class__.__name__}: {str(exc)[:200]}")
                try:
                    page.wait_for_timeout(_budget_ms(deadline, LIVE_SAMPLE_INTERVAL_MS))
                except Exception as exc:
                    errors.append(f"sample wait: {exc.__class__.__name__}: {str(exc)[:200]}")
                try:
                    page.evaluate(SCREENSHOT_SETTLE_JS)
                except Exception as exc:
                    errors.append(f"fonts: {exc.__class__.__name__}: {str(exc)[:200]}")
                try:
                    _mark(STAGE_SHOT)
                    shot_later = page.screenshot(type="png", timeout=SCREENSHOT_TIMEOUT_MS)
                except Exception as exc:
                    errors.append(f"later screenshot: {exc.__class__.__name__}: {str(exc)[:200]}")

            # No pixel statistics in ``live``: the canvas probe is a K-only
            # measurement, and an arbitrary page may have no canvas at all —
            # or an evaluate call that never returns.
            return {
                "errors": errors,
                "external": external,
                "probe": None,
                "stepped": None,
                "pixels": None,
                "shot": shot or shot_later,
                "shot_later": shot_later,
                "ready_failed": False,
            }

        _mark(STAGE_NAV)
        page.goto(url, wait_until="domcontentloaded", timeout=_budget_ms(deadline, timeout_ms))
        try:
            page.evaluate(SCREENSHOT_SETTLE_JS)
        except Exception:
            pass
        run_probe = probe
        ready_failed = False
        try:
            _mark(STAGE_HOOK)
            page.wait_for_function("() => window.__ready === true", timeout=_budget_ms(deadline, timeout_ms))
        except Exception:
            # Протокол мог не дойти: страница падает до `__ready` (L-20,
            # pageerror в конструкторе) или вовсе его не выставляет. Исключение
            # раньше рушило весь проход, и витрина оставалась без кадра, хотя
            # страница уже что-то нарисовала. Теперь кадр снимается, а провал
            # протокола остаётся вердиктом.
            ready_failed = True
        _mark(STAGE_HOOK)
        probe = page.evaluate(
            # try — потому что __probe страницы может кидать (K-15: TypeError
            # внутри probe после шага), и падение evaluate рушило весь проход
            # вместе с кадром.
            "() => { try { return typeof window.__probe === 'function' ? window.__probe() : null; }"
            " catch (e) { return null; } }"
        )
        stepped = None
        step_failed = False
        if steps:
            # Шаг оборачиваем в try: страница без `__step` (частый дефект модели)
            # роняла весь проход. Отдельный исход отличает «шага нет» от «шаг
            # есть и отчитался»: протокол — свой статус, а не приговор кадру.
            _mark(STAGE_HOOK)
            outcome = page.evaluate(
                "([dt, n]) => {"
                " if (typeof window.__step !== 'function') return {ok: false};"
                " try { for (let i = 0; i < n; i++) window.__step(dt); }"
                " catch (e) { return {ok: false}; }"
                " try { return {ok: true, payload: typeof window.__probe === 'function' ? window.__probe() : null}; }"
                " catch (e) { return {ok: true, payload: null}; } }",
                [dt, steps],
            )
            if isinstance(outcome, dict):
                step_failed = not outcome.get("ok")
                stepped = outcome.get("payload")
            else:
                step_failed = True
        harness_error = ""
        pixels = None
        try:
            page.evaluate(SCREENSHOT_SETTLE_JS)
        except Exception as exc:
            errors.append(f"settle: {exc.__class__.__name__}: {str(exc)[:200]}")
        _mark(STAGE_SHOT)
        try:
            shot = page.screenshot(type="png", timeout=SCREENSHOT_TIMEOUT_MS)
        except Exception as exc:
            shot = b""
            errors.append(f"screenshot: {exc.__class__.__name__}: {str(exc)[:200]}")
        if run_probe:
            try:
                _mark(STAGE_HOOK)
                pixels = page.evaluate(PIXEL_PROBE)
            except Exception as exc:
                harness_error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        result: dict[str, Any] = {
            "errors": errors,
            "external": external,
            "probe": probe,
            "stepped": stepped,
            "pixels": pixels,
            "shot": shot,
            "ready_failed": ready_failed,
            "step_failed": step_failed,
            "harness_error": harness_error,
        }
        if mode == "time":
            # A declared animation must move; the second sample tells them apart
            # without trusting the page's own clock.
            # Проба без __probe/__step возвращает null — страница всё равно
            # должна досчитаться до кадра, иначе витрина остаётся без картинки.
            _mark(STAGE_HOOK)
            moved = page.evaluate(
                "([dt, n]) => { if (typeof window.__step !== 'function') return false;"
                " const grab = () => { try { return typeof window.__probe === 'function' ? window.__probe() : null; } catch (e) { return null; } };"
                " const before = grab();"
                " try { for (let i = 0; i < n; i++) window.__step(dt); } catch (e) { return true; }"
                " const after = grab();"
                " if (before === null || after === null) return true;"
                " return JSON.stringify(before) !== JSON.stringify(after); }",
                [dt, max(1, steps)],
            )
            try:
                page.evaluate(SCREENSHOT_SETTLE_JS)
            except Exception:
                pass
            result["animated"] = bool(moved)
            _mark(STAGE_SHOT)
            result["shot_later"] = page.screenshot(type="png", timeout=SCREENSHOT_TIMEOUT_MS)
        return result
    finally:
        context.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


_WORKER: dict[str, Any] = {}


def _recv(conn: Any) -> tuple[str, Any]:
    try:
        return conn.recv()
    except (EOFError, OSError) as exc:
        raise RuntimeError(f"рендер-процесс завершился: {exc}") from exc


def _kill_worker() -> None:
    """Force the render process down; the next render starts a fresh one."""
    proc = _WORKER.pop("proc", None)
    conn = _WORKER.pop("conn", None)
    for key in ("ready", "stage", "progress"):
        _WORKER.pop(key, None)
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()
    if proc is None:
        return
    if proc.is_alive():
        # SIGKILL: a child stuck in a page hook cannot run signal handlers or
        # flush Playwright state, and there is nothing to preserve between runs.
        proc.kill()
        proc.join(1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)


def _shutdown_worker() -> None:  # pragma: no cover - interpreter shutdown
    proc = _WORKER.get("proc")
    conn = _WORKER.get("conn")
    if proc is not None and proc.is_alive() and conn is not None:
        with contextlib.suppress(Exception):
            conn.send(None)
        proc.join(2.0)
    _kill_worker()


atexit.register(_shutdown_worker)


_DEVNULL: Any = None


def _worker_main(conn: Any, stage: Any, progress: Any) -> None:
    """Render-process entry point: one Playwright session, jobs over a pipe."""
    global _DEVNULL
    # The Playwright driver inherits stderr; a killed driver answers its closed
    # pipe with an EPIPE stack trace that would land in every run log.
    _DEVNULL = open(os.devnull, "w")
    sys.stderr = _DEVNULL
    try:
        conn.send(("ready", None))
        while True:
            try:
                job = conn.recv()
            except EOFError:
                return
            if job is None:
                return
            try:
                result: RenderResult = _render_inline(**job, stage=stage, progress=progress)
            except BaseException as exc:  # noqa: BLE001 - never leave a caller waiting
                result = RenderResult(ok=False, reason=f"{type(exc).__name__}: {str(exc)[:200]}")
            conn.send(("done", result))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def _ensure_worker() -> Any:
    proc = _WORKER.get("proc")
    if proc is not None and proc.is_alive() and _WORKER.get("ready"):
        return _WORKER["conn"]
    _kill_worker()
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    stage = ctx.Value("i", STAGE_IDLE)
    progress = ctx.Value("i", 0)
    process = ctx.Process(target=_worker_main, args=(child_conn, stage, progress), daemon=True)
    process.start()
    child_conn.close()
    _WORKER.update(proc=process, conn=parent_conn, ready=False, stage=stage, progress=progress)
    if not parent_conn.poll(WORKER_START_TIMEOUT_S):
        _kill_worker()
        raise RuntimeError(f"рендер-процесс не ответил за {WORKER_START_TIMEOUT_S:.0f} с")
    kind, message = _recv(parent_conn)
    if kind != "ready":
        _kill_worker()
        raise RuntimeError(str(message))
    _WORKER["ready"] = True
    return parent_conn


def _call_worker(job: dict[str, Any], deadline: float) -> RenderResult | None:
    """Send the job and wait for the result; ``None`` means the deadline hit.

    The budget is per stage: navigation and hooks share the render deadline,
    while a screenshot gets its own ``SCREENSHOT_TIMEOUT_MS`` (a 1920x1080
    frame under parallel load is slow even when the page is not). A hook that
    starts after the deadline gets a short continuation, because the previous
    hook legitimately spent the budget; a hook that never returns does not.
    """
    conn = _ensure_worker()
    stage = _WORKER["stage"]
    progress = _WORKER["progress"]
    try:
        conn.send(job)
    except (BrokenPipeError, OSError) as exc:
        _kill_worker()
        raise RuntimeError(f"рендер-процесс недоступен: {exc}") from exc
    armed = False
    seen = (-1, -1)
    stage_deadline = deadline
    last_progress = time.monotonic()
    expired_at: float | None = None
    while True:
        now = time.monotonic()
        key = (stage.value, progress.value)
        if key != seen:
            seen = key
            last_progress = now
            expired_at = None
            kind = key[0]
            if kind in (STAGE_IDLE, STAGE_SETUP):
                stage_deadline = now + SETUP_TIMEOUT_S
            elif kind == STAGE_SHOT:
                stage_deadline = now + SCREENSHOT_TIMEOUT_MS / 1000
            elif kind == STAGE_NAV:
                # A new page pass: its budget starts now (a cold context is
                # excluded, a hung navigation is not).
                deadline = now + job["timeout_ms"] / 1000
                stage_deadline = deadline
                armed = True
            elif kind == STAGE_HOOK and not armed:
                # The poll may miss a short navigation stage and observe the
                # first hook instead; the budget still starts at page work.
                deadline = now + job["timeout_ms"] / 1000
                stage_deadline = deadline
                armed = True
            elif now >= deadline:
                stage_deadline = now + HOOK_CONTINUATION_S
            else:
                stage_deadline = deadline
        if stage.value == STAGE_SHOT:
            if now >= stage_deadline:
                return None
        elif now >= deadline + HARD_GRACE_S:
            return None
        limit = max(stage_deadline, last_progress + STALL_S)
        remaining = limit - now
        if remaining <= 0:
            # The awaited call may be firing its own timeout right now; give
            # the child a short window to bump the progress counter.
            if expired_at is None:
                expired_at = now
            if now - expired_at > DEADLINE_GRACE_S:
                return None
            slice_s = 0.01
        else:
            expired_at = None
            slice_s = min(0.01, remaining)
        if conn.poll(slice_s):
            kind, payload = _recv(conn)
            if kind == "done" and isinstance(payload, RenderResult):
                return payload
            raise RuntimeError(str(payload))


def _render_inline(
    html: str,
    steps: int,
    dt: float,
    viewport: tuple[int, int],
    timeout_ms: int,
    again: bool,
    mode: str,
    probe: bool = True,
    stage: Any = None,
    progress: Any = None,
) -> RenderResult:
    """Render body; runs inside the worker process under the parent deadline."""
    started = time.time()
    try:
        with _served(html) as (url, port):
            first = _page_pass(
                url, port, steps, dt, viewport, timeout_ms, mode, stage, progress, probe,
            )
            if first.get("harness_error"):
                shot = first.get("shot") or b""
                return RenderResult(
                    ok=False,
                    shot=shot,
                    shot_sha=hashlib.sha256(shot).hexdigest() if shot else "",
                    failure="evaluator_failure",
                    reason=str(first["harness_error"])[:200],
                    errors=list(first.get("errors") or []),
                    external=list(first.get("external") or []),
                    probe=first.get("probe"),
                    stepped=first.get("stepped"),
                    pixels=first.get("pixels"),
                    ready_ok=not first.get("ready_failed"),
                    step_ok=not first.get("step_failed"),
                )
            second = (
                _page_pass(
                    url, port, steps, dt, viewport, timeout_ms, mode, stage, progress, probe,
                )
                if again and mode != "live"
                else None
            )
    except Exception as exc:
        return RenderResult(ok=False, reason=f"{exc.__class__.__name__}: {str(exc)[:200]}")

    shot = first["shot"] or b""
    later = first.get("shot_later")
    result = RenderResult(
        ok=bool(shot),
        errors=list(first["errors"]),
        external=list(first["external"]),
        probe=first["probe"],
        stepped=first["stepped"],
        pixels=first["pixels"],
        pixels_again=(second or {}).get("pixels"),
        shot=shot,
        shot_sha=hashlib.sha256(shot).hexdigest() if shot else "",
        shot_sha_again=(
            hashlib.sha256(second["shot"]).hexdigest() if second and second["shot"] else ""
        ),
        animated=first.get("animated"),
        ready_ok=not first.get("ready_failed"),
        step_ok=not first.get("step_failed"),
        shot_later_sha=hashlib.sha256(later).hexdigest() if later else "",
        duration_s=time.time() - started,
    )
    # Протокол, ошибки консоли и кадр — разные вещи (M09). Раньше любой
    # `console.error` и потерянный `__probe` выставляли `ok=False` и обнуляли
    # ВСЕ проверки; теперь `ok` значит «кадр снят», а остальное живёт своими
    # флагами. Без кадра оценивать нечего — только тогда рендер провален.
    if not shot:
        result.ok = False
        result.reason = result.reason or "кадр не снят"
    if first.get("ready_failed"):
        result.reason = result.reason or "страница не выставила window.__ready"
    elif result.errors and not result.reason:
        result.reason = result.errors[0][:200]
    # ``ok`` — про то, годится ли страница как ответ вообще. Кадр снимается и
    # без протокола, поэтому потерянный __probe помечаем отдельным флагом
    # (``probe_ok``), а не обнуляем весь рендер: витрина должна показать сцену.
    # ``plain`` здесь, а не при записи: probe со ссылкой ``mesh.parent.children[0]
    # is mesh`` ронял ``json.dumps`` кадра, и кадр терялся вместе с кэшем. Обе
    # ветки — свежий рендер и чтение из кэша — должны видеть один и тот же
    # JSON-safe payload.

    if mode == "live":
        # ``live`` deliberately has no page protocol: a static document is a
        # complete result, and random/realtime animation is not judged for
        # frame-to-frame reproducibility.
        result.probe_ok = False
        result.ok = not result.errors
        if result.errors:
            result.reason = result.errors[0][:200]
        return result

    result.probe = plain(result.probe) if result.probe is not None else None
    result.stepped = plain(result.stepped) if result.stepped is not None else None
    if result.probe is None:
        result.probe_ok = False
        result.reason = "страница не отдала window.__probe()"
    elif not isinstance(result.probe, dict):
        result.probe_ok = False
        result.reason = "window.__probe() вернул не объект"
    return result


def render(
    html: str,
    *,
    steps: int = 0,
    dt: float = 1 / 60,
    viewport: tuple[int, int] = DEFAULT_VIEWPORT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    again: bool = True,
    mode: str = "fixed",
    probe: bool = True,
) -> RenderResult:
    """Render one candidate page in a fixed, time-driven, or ordinary live mode.

    The render runs in a separate process under one absolute deadline: a page
    hook can be an infinite loop, which no Playwright timeout can pre-empt.
    When the deadline passes the process is killed and the task is reported as
    a timeout — the evaluator is never held by the page. Starting the worker is
    excluded from the deadline (a cold Playwright launch is not a hung page).
    """
    usable, reason = availability()
    if not usable:
        return RenderResult(ok=False, reason=reason)
    if not (html or "").strip():
        return RenderResult(ok=False, reason="пустой ответ: рендерить нечего")

    journal = telemetry.active()
    stage = "navigation" if mode == "live" else "hook"
    started = time.time()
    stage_started = time.monotonic()
    if journal is not None:
        journal.browser_stage(stage, status="start")
    try:
        _ensure_worker()
    except Exception as exc:
        return RenderResult(
            ok=False,
            reason=f"рендер-процесс не запустился: {exc}",
            failure="evaluator_failure",
        )

    deadline = time.monotonic() + timeout_ms / 1000
    job = {
        "html": html,
        "steps": steps,
        "dt": dt,
        "viewport": viewport,
        "timeout_ms": timeout_ms,
        "again": again,
        "mode": mode,
        "probe": probe,
    }
    try:
        result = _call_worker(job, deadline)
    except Exception as exc:
        _kill_worker()
        return RenderResult(
            ok=False,
            reason=f"{type(exc).__name__}: {str(exc)[:200]}",
            failure="evaluator_failure",
        )
    if result is not None and result.failure == "evaluator_failure":
        _kill_worker()
    if result is None:
        _kill_worker()
        # The navigation timeout is only a part of the budget, so the reason
        # names the whole render deadline; the stage is kept separately.
        if journal is not None:
            journal.browser_stage(
                stage, status="timeout", duration_s=time.monotonic() - stage_started
            )
        return RenderResult(
            ok=False,
            errors=[f"{stage}: TimeoutError: дедлайн рендера {timeout_ms} мс исчерпан"],
            reason=f"дедлайн рендера {timeout_ms} мс исчерпан: страница не вернула управление",
            failure="resource_exhausted",
            duration_s=time.time() - started,
        )
    result.duration_s = time.time() - started
    if journal is not None:
        journal.browser_stage(
            "render",
            status="ok" if result.ok else "error",
            duration_s=time.monotonic() - stage_started,
        )
    return result


def probe_of(result: RenderResult, stepped: bool = False) -> dict[str, Any]:
    """Probe payload of a render, or an empty dict when there is none."""
    payload = result.stepped if stepped else result.probe
    return payload if isinstance(payload, dict) else {}


def plain(value: Any, *, _path: set[int] | None = None, _depth: int = 0) -> Any:
    """JSON-safe copy of a probe payload that keeps dict/list shape.

    A page's ``__probe()`` often returns live three.js objects: in the parent
    link ``mesh.parent.children[0]`` *is* ``mesh``, so a straight ``json.dumps``
    of that payload dies with ``ValueError: Circular reference detected`` — and
    the frame cache, written in the same ``try``, was lost with it. Only true
    reference cycles are cut (path-based, not global ``seen``): shared subtrees
    stay as they are, because predicates read them by path.
    """
    if _depth > 32:
        return "<deep>"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (dict, list, tuple)):
        path = _path if _path is not None else set()
        if id(value) in path:
            return "<cycle>"
        path.add(id(value))
        try:
            if isinstance(value, dict):
                return {str(k): plain(v, _path=path, _depth=_depth + 1) for k, v in value.items()}
            return [plain(v, _path=path, _depth=_depth + 1) for v in value]
        finally:
            path.discard(id(value))
    return str(value)


def save_shot(result: RenderResult, path: Path) -> str:
    """Write the frame next to the run; returns the relative name or ``""``."""
    if not result.shot:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.shot)
    return path.name


def free_port() -> int:  # pragma: no cover - helper for local experiments
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
