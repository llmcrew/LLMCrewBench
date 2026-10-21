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
    try:
        context.route(
            "**/*",
            lambda route: (
                route.continue_()
                if route.request.url.startswith(allowed) or route.request.url.startswith("data:")
                else (external.append(route.request.url), route.abort())
            ),
        )
        # HTTP routing does not govern the WebSocket handshake: a ws:// channel
        # used to bypass ``route`` entirely. A routed socket is never connected,
        # so no traffic leaves and the attempt is recorded in ``external``.
        context.route_web_socket("**/*", lambda ws: external.append(f"websocket: {ws.url}"))
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
                    _mark(STAGE_SHOT)
                    shot = page.screenshot(type="png", timeout=SCREENSHOT_TIMEOUT_MS)
                except Exception as exc:
                    errors.append(f"screenshot: {exc.__class__.__name__}: {str(exc)[:200]}")
                try:
                    page.wait_for_timeout(_budget_ms(deadline, LIVE_SAMPLE_INTERVAL_MS))
                except Exception as exc:
                    errors.append(f"sample wait: {exc.__class__.__name__}: {str(exc)[:200]}")
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
        if steps:
            # Шаг оборачиваем в try: страница без __probe (частый дефект модели)
            # роняла весь проход, и у витрины не оставалось даже кадра.
            _mark(STAGE_HOOK)
            stepped = page.evaluate(
                "([dt, n]) => { try { for (let i = 0; i < n; i++) window.__step(dt); }"
                " catch (e) { return null; }"
                " try { return typeof window.__probe === 'function' ? window.__probe() : null; }"
                " catch (e) { return null; } }",
                [dt, steps],
            )
        _mark(STAGE_HOOK)
        pixels = page.evaluate(PIXEL_PROBE)
        _mark(STAGE_SHOT)
        shot = page.screenshot(type="png", timeout=SCREENSHOT_TIMEOUT_MS)
        result: dict[str, Any] = {
            "errors": errors,
            "external": external,
            "probe": probe,
            "stepped": stepped,
            "pixels": pixels,
            "shot": shot,
            "ready_failed": ready_failed,
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
        if now >= deadline + HARD_GRACE_S:
            # No stage may extend a render past the absolute cap: a Chromium
            # that stopped answering is infrastructure, not a slow page.
            return None
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
    stage: Any = None,
    progress: Any = None,
) -> RenderResult:
    """Render body; runs inside the worker process under the parent deadline."""
    started = time.time()
    try:
        with _served(html) as (url, port):
            first = _page_pass(
                url, port, steps, dt, viewport, timeout_ms, mode, stage, progress
            )
            second = (
                _page_pass(
                    url, port, steps, dt, viewport, timeout_ms, mode, stage, progress
                )
                if again and mode != "live"
                else None
            )
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
    if first.get("ready_failed"):
        # Страница нарисовала кадр, но протокол не дошёл: по контракту это
        # провал (раньше таймаут рушил проход целиком). Кадр и pixels при этом
        # сохранены — витрине есть что показать.
        result.ok = False
        if not result.reason:
            result.reason = "страница не выставила window.__ready"
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
    elif result.errors:
        result.ok = False
        result.reason = "ошибки в консоли страницы"
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

    started = time.time()
    try:
        _ensure_worker()
    except Exception as exc:
        return RenderResult(ok=False, reason=f"рендер-процесс не запустился: {exc}")

    deadline = time.monotonic() + timeout_ms / 1000
    job = {
        "html": html,
        "steps": steps,
        "dt": dt,
        "viewport": viewport,
        "timeout_ms": timeout_ms,
        "again": again,
        "mode": mode,
    }
    try:
        result = _call_worker(job, deadline)
    except Exception as exc:
        _kill_worker()
        return RenderResult(ok=False, reason=f"{type(exc).__name__}: {str(exc)[:200]}")
    if result is None:
        _kill_worker()
        # The navigation timeout is only a part of the budget, so the reason
        # names the whole render deadline; the stage is kept separately.
        stage = "navigation" if mode == "live" else "hook"
        return RenderResult(
            ok=False,
            errors=[f"{stage}: TimeoutError: дедлайн рендера {timeout_ms} мс исчерпан"],
            reason=f"дедлайн рендера {timeout_ms} мс исчерпан: страница не вернула управление",
            duration_s=time.time() - started,
        )
    result.duration_s = time.time() - started
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
