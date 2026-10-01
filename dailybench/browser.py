"""Headless-browser rendering for the visual block (category K).

The visual scenarios are the only ones whose result cannot be read from text:
the model writes a Three.js page, so the evaluator has to *run* it. This module
owns that step and nothing else — how a scenario is scored lives in
``tasks/visual_*.py``.

What is deliberately fixed here, identically for every model:

* the page is served from ``/app.html`` on a loopback port, and the pinned
  library from ``/vendor/`` — no CDN, and every request that is not loopback is
  refused and reported, so an answer that secretly depends on the network scores
  zero instead of silently working on a machine with internet;
* the viewport is 800x600 at device pixel ratio 1, so two models are compared on
  the same frame;
* the scene is stepped through ``window.__step(dt)``, never through the wall
  clock, so a simulation is scored at a stated time rather than "after a while";
* the frame is rendered twice, in two separate page loads, and the two images
  are hashed: a scene that depends on ``Math.random()`` or on elapsed time is
  not reproducible and loses the image point.

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
import os
import shutil
import socket
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "vendor" / "three"
APP_NAME = "app.html"
DEFAULT_VIEWPORT = (800, 600)
#: Арт-блок рисует крупнее: кадр 1920x1080, чтобы сцены было видно на большом
#: экране и в записи. Категория K остаётся на 800x600 — там важны координаты.
ART_VIEWPORT = (1920, 1080)
DEFAULT_TIMEOUT_MS = 20_000

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


@dataclass
class RenderResult:
    """Everything the evaluator may look at. ``ok`` is about rendering."""

    ok: bool
    reason: str = ""
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
    shot_later_sha: str = ""
    duration_s: float = 0.0

    @property
    def same_frame(self) -> bool:
        """Two loads produced byte-identical frames."""
        return bool(self.shot_sha) and self.shot_sha == self.shot_sha_again

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
    cache = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or (Path.home() / ".cache" / "ms-playwright"))
    mac_cache = Path.home() / "Library" / "Caches" / "ms-playwright"
    found = any(
        (root / entry).is_dir()
        for root in (cache, mac_cache)
        for entry in (root.iterdir() if root.is_dir() else [])
        if entry.name.startswith("chromium")
    )
    if not found:
        return False, "chromium для playwright не установлен (playwright install chromium)"
    if not (VENDOR / "three.module.js").is_file():
        return False, f"нет библиотеки {VENDOR / 'three.module.js'}"
    return True, ""


# --------------------------------------------------------------------------- #
# Static server
# --------------------------------------------------------------------------- #


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the candidate page plus the pinned library, and stays quiet."""

    def log_message(self, *args: Any) -> None:  # pragma: no cover - noise
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


@contextlib.contextmanager
def _served(html: str):
    """Serve ``html`` as /app.html with ./vendor/ next to it."""
    root = Path(tempfile.mkdtemp(prefix="dailybench-visual-"))
    try:
        (root / APP_NAME).write_text(html, encoding="utf-8")
        vendor = root / "vendor"
        try:
            vendor.symlink_to(VENDOR, target_is_directory=True)
        except OSError:  # pragma: no cover - filesystem without symlinks
            shutil.copytree(VENDOR, vendor)
        handler = functools.partial(_Handler, directory=str(root))
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
  const c = document.querySelector('canvas');
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
  for (let y = 0; y < h; y += st) {
    let rowLum = 0, rowSat = 0, rowN = 0;
    for (let x = 0; x < w; x += st) {
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


def _page_pass(
    url: str,
    port: int,
    steps: int,
    dt: float,
    viewport: tuple[int, int],
    timeout_ms: int,
    mode: str = "fixed",
) -> dict[str, Any]:
    """One full load: errors, probe, stepping, pixels, frames.

    ``mode`` decides how the page is advanced before the frame is taken:

    * ``fixed`` — ``steps`` calls of ``__step(dt)`` (the simulation block);
    * ``time``  — the same, but the frames are also compared at two moments, so
      an animated art scene proves it animates and a static one proves it does
      not drift.
    """
    context = _browser().new_context(
        viewport={"width": viewport[0], "height": viewport[1]},
        device_scale_factor=1,
        reduced_motion="reduce",
    )
    errors: list[str] = []
    external: list[str] = []
    allowed = f"http://127.0.0.1:{port}/"
    try:
        context.route(
            "**/*",
            lambda route: (
                route.continue_()
                if route.request.url.startswith(allowed) or route.request.url.startswith("data:")
                else (external.append(route.request.url), route.abort())
            ),
        )
        page = context.new_page()
        page.on("console", lambda msg: errors.append(f"console: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_function("() => window.__ready === true", timeout=timeout_ms)
        probe = page.evaluate("() => (typeof window.__probe === 'function' ? window.__probe() : null)")
        stepped = None
        if steps:
            # Шаг оборачиваем в try: страница без __probe (частый дефект модели)
            # роняла весь проход, и у витрины не оставалось даже кадра.
            stepped = page.evaluate(
                "([dt, n]) => { try { for (let i = 0; i < n; i++) window.__step(dt); }"
                " catch (e) { return null; }"
                " return typeof window.__probe === 'function' ? window.__probe() : null; }",
                [dt, steps],
            )
        pixels = page.evaluate(PIXEL_PROBE)
        shot = page.screenshot(type="png")
        result: dict[str, Any] = {
            "errors": errors,
            "external": external,
            "probe": probe,
            "stepped": stepped,
            "pixels": pixels,
            "shot": shot,
        }
        if mode == "time":
            # A declared animation must move; the second sample tells them apart
            # without trusting the page's own clock.
            # Проба без __probe/__step возвращает null — страница всё равно
            # должна досчитаться до кадра, иначе витрина остаётся без картинки.
            moved = page.evaluate(
                "([dt, n]) => { if (typeof window.__step !== 'function') return false;"
                " const before = typeof window.__probe === 'function' ? window.__probe() : null;"
                " for (let i = 0; i < n; i++) window.__step(dt);"
                " const after = typeof window.__probe === 'function' ? window.__probe() : null;"
                " if (before === null || after === null) return true;"
                " return JSON.stringify(before) !== JSON.stringify(after); }",
                [dt, max(1, steps)],
            )
            result["animated"] = bool(moved)
            result["shot_later"] = page.screenshot(type="png")
        return result
    finally:
        context.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def render(
    html: str,
    *,
    steps: int = 0,
    dt: float = 1 / 60,
    viewport: tuple[int, int] = DEFAULT_VIEWPORT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    again: bool = True,
    mode: str = "fixed",
) -> RenderResult:
    """Run one candidate page and return everything observable about it."""
    usable, reason = availability()
    if not usable:
        return RenderResult(ok=False, reason=reason)
    if not (html or "").strip():
        return RenderResult(ok=False, reason="пустой ответ: рендерить нечего")

    started = time.time()
    try:
        with _served(html) as (url, port):
            first = _page_pass(url, port, steps, dt, viewport, timeout_ms, mode)
            second = _page_pass(url, port, steps, dt, viewport, timeout_ms, mode) if again else None
    except Exception as exc:
        return RenderResult(ok=False, reason=f"{exc.__class__.__name__}: {str(exc)[:200]}")

    shot = first["shot"] or b""
    later = first.get("shot_later")
    result = RenderResult(
        ok=not first["errors"],
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
        shot_later_sha=hashlib.sha256(later).hexdigest() if later else "",
        duration_s=time.time() - started,
    )
    # ``ok`` — про то, годится ли страница как ответ вообще. Кадр снимается и
    # без протокола, поэтому потерянный __probe помечаем отдельным флагом
    # (``probe_ok``), а не обнуляем весь рендер: витрина должна показать сцену.
    if result.probe is None:
        result.probe_ok = False
        result.reason = "страница не отдала window.__probe()"
    elif not isinstance(result.probe, dict):
        result.probe_ok = False
        result.reason = "window.__probe() вернул не объект"
    elif result.errors:
        result.ok = False
        result.reason = "ошибки в консоли страницы"
    return result


def probe_of(result: RenderResult, stepped: bool = False) -> dict[str, Any]:
    """Probe payload of a render, or an empty dict when there is none."""
    payload = result.stepped if stepped else result.probe
    return payload if isinstance(payload, dict) else {}


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
