#!/usr/bin/env python3
"""Record a model-generated scene with a controlled frame clock.

Usage:
  python3 record_clip.py <model> <scene> <out_dir> [seconds] [warmup_s]
      [--force] [--source-root PATH] [--startup-timeout SECONDS] [--fps 120]
      [--crf 16] [--preset medium] [--chrome EXECUTABLE]

Source pages default to ``reports/comparison-deepseek-1021/_showcase``. To
record a repaired copy, pass ``--source-root
reports/comparison-deepseek-1021/_repairs``. Source files are never modified.
Frames are lossless PNG at 1920x1080, encoded at 120 fps. Missing or
pre-frame-failing pages produce typed media-status placeholders; capture and
encoding failures return a non-zero infrastructure code.
"""
from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import http.server
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "video"
sys.path.insert(0, str(Path(__file__).parent))
import media_status  # noqa: E402

REPORTS = ROOT / "reports" / "comparison-deepseek-1021"
OUT = VIDEO / "public" / "clips"
MODELS = ("deepseek-v4.1-flash-unc-1021", "deepseek-v4.1-flash-unc-1017-1021")
SCENES = tuple(f"K-{i}" for i in range(1, 21)) + tuple(f"L-{i}" for i in range(1, 30))

#: Native rendering is disabled for the page. Every driven frame advances this
#: clock by exactly one FPS interval, regardless of the host's elapsed time.
FPS = 120
CRF = 16
PRESET = "medium"
DT = 1 / FPS
W, H = 1920, 1080
DEFAULT_SECS = 20.0
DEFAULT_WARMUP = 0.6
STARTUP_TIMEOUT = 180.0

#: 0 - clip recorded, 3 - model page failed before its first frame,
#: 4 - capture/encoding infrastructure failed.
EXIT = {"ok": 0, "model_failed": 3, "capture_failed": 4}



def _find(name: str, *preferred: str, verify: bool = False) -> Path | None:
    """Find a platform binary, rejecting bundled executables for another OS."""
    for candidate in (*preferred, shutil.which(name) or ""):
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_file():
            continue
        if verify:
            try:
                probe = subprocess.run([str(path), "-version"], cwd=str(path.parent),
                                       capture_output=True, text=True)
            except OSError:
                continue
            if probe.returncode != 0:
                continue
        return path
    return None


def ffmpeg() -> Path:
    path = _find("ffmpeg", str(VIDEO / "bin" / "ffmpeg"), verify=True)
    if path is None:
        raise RuntimeError("ffmpeg not found: install it in PATH or video/bin/")
    return path


def ffprobe() -> Path:
    path = _find("ffprobe", str(VIDEO / "bin" / "ffprobe"), verify=True)
    if path is None:
        raise SystemExit("ffprobe не найден: положите его в PATH или в video/bin/")
    return path


def chrome(binary: str | Path | None = None) -> Path:
    if binary is not None:
        name = str(binary)
        candidate = Path(name).expanduser()
        preferred = str(candidate.resolve()) if candidate.is_file() else name
        path = _find(name, preferred)
        if path is None:
            raise RuntimeError(f"Chrome executable {name!r} was not found")
        return path
    path = _find(
        "google-chrome",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        str(VIDEO / "bin" / "chrome"),
        str(VIDEO / "bin" / "chrome.exe"),
    ) or _find("chromium") or _find("chromium-browser") or _find(
        "google-chrome-stable"
    ) or _find("chrome")
    if path is None:
        raise RuntimeError("Chrome not found: install Google Chrome or Chromium in PATH")
    return path


def run_bin(exe: Path, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Запуск комплектного бинарника: dylib лежат рядом, cwd обязан быть его."""
    return subprocess.run([str(exe), *args], cwd=str(exe.parent), check=True, **kwargs)


def check_target(model: str, scene: str, out_dir: Path) -> Path:
    """Validate model/scene and contain output under video/out or video/public."""
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}: {', '.join(MODELS)}")
    if scene not in SCENES:
        raise SystemExit(f"unknown scene {scene!r}: K-1..K-20, L-1..L-29")
    target = out_dir.expanduser().resolve()
    allowed = (VIDEO / "out").resolve(), (VIDEO / "public").resolve()
    if not any(target == root or root in target.parents for root in allowed):
        choices = " or ".join(str(root) for root in allowed)
        raise SystemExit(f"OUT {target} must be under {choices}")
    return target


FRAME_CLOCK_JS = r"""
(() => {
  if (window.__recordClock) return;
  const FPS = 120;
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


class PageJavaScriptError(RuntimeError):
    """A page exception before its first captured frame."""


def _javascript_error(ws) -> str:
    lock = getattr(ws, "lock", None)
    if lock is None:
        events = list(getattr(ws, "events", ()))
    else:
        with lock:
            events = list(ws.events)
    events = [event for event in events if event.get("method") == "Runtime.exceptionThrown"]
    if not events:
        return ""
    details = (events[0].get("params") or {}).get("exceptionDetails") or {}
    exception = details.get("exception") or {}
    text = exception.get("description") or details.get("text") or "JavaScript exception"
    line = details.get("lineNumber")
    url = details.get("url")
    where = f"{url}:{line + 1}" if url and isinstance(line, int) else ""
    return f"{where}: {text}" if where else str(text)


def _evaluate(call, expression: str, *, timeout: float = 60) -> dict:
    response = call("Runtime.evaluate", {
        "expression": expression,
        "returnByValue": True,
        "awaitPromise": False,
    }, timeout=timeout)
    result = response.get("result") or {}
    details = result.get("exceptionDetails")
    if details:
        exception = details.get("exception") or {}
        message = exception.get("description") or details.get("text") or "JavaScript exception"
        raise PageJavaScriptError(str(message))
    return response


def page_ready(call, ws, scene: str, timeout: float = STARTUP_TIMEOUT) -> str:
    """Wait for a ready marker or stable, rendered page without advancing its clock."""
    probe = """
(() => {
  if (window.__ready === true) return 'ready';
  const s = window.__dbscene;
  if (s && s.mode && s.mode !== 'measuring') return 'driver';
  if (document.readyState !== 'complete') return '';
  const c = document.querySelector('canvas');
  const live = (c && c.width > 1 && c.height > 1) ||
               document.querySelector('svg, img, video') ||
               (document.body && (document.body.innerText || '').trim().length > 0);
  if (!live) return '';
  return 'settled:' + [
    document.body ? document.body.querySelectorAll('*').length : 0,
    c ? c.width + 'x' + c.height : '',
    (document.body ? document.body.innerText : '').length,
  ].join('|');
})()"""
    deterministic = scene.startswith("K-")
    deadline = time.time() + timeout
    settled_at: float | None = None
    last = ""
    while time.time() < deadline:
        if _javascript_error(ws):
            return ""
        try:
            response = _evaluate(call, probe, timeout=5)
        except (TimeoutError, PageJavaScriptError):
            if _javascript_error(ws):
                return ""
            continue
        value = ((response.get("result") or {}).get("result") or {}).get("value") or ""
        if value == "ready":
            return "ready"
        if value == "driver":
            return "driver"
        if value.startswith("settled:"):
            mark = value.split(":", 1)[1]
            if mark == last:
                if settled_at is None:
                    settled_at = time.time()
                elif not deterministic and time.time() - settled_at >= 0.9:
                    return "settled"
            else:
                last = mark
                settled_at = None
        else:
            last = ""
            settled_at = None
        time.sleep(0.05)
    return ""








class WS:
    """Минимальный CDP-клиент (websocket с обязательной маскировкой фреймов).

    JS-диалог (`alert`/`confirm`/`prompt`) останавливает главный поток страницы:
    незакрытый, он вешает `Runtime.evaluate` до таймаута, и клип получает
    плашку. Диалоги принимаются сразу, их текст остаётся в `dialogs`.
    """

    def __init__(self, url: str):
        u = urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=30)
        key = base64.b64encode(b"0123456789abcdef").decode()
        self.sock.sendall(
            f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        self.sock.settimeout(None)
        self._id = 0
        self.lock = threading.Lock()
        #: Тексты показанных страницей диалогов (для маркера).
        self.dialogs: list[str] = []
        self._send_lock = threading.Lock()
        self.events: list[dict] = []
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        buf = b""
        while True:
            try:
                chunk = self.sock.recv(1 << 20)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while len(buf) >= 2:
                ln = buf[1] & 0x7F
                off = 2
                if ln == 126:
                    ln = int.from_bytes(buf[2:4], "big"); off = 4
                elif ln == 127:
                    ln = int.from_bytes(buf[2:10], "big"); off = 10
                if len(buf) < off + ln:
                    break
                payload, buf = buf[off:off + ln], buf[off + ln:]
                try:
                    msg = json.loads(payload)
                except Exception:
                    continue
                if msg.get("method") == "Page.javascriptDialogOpening":
                    # Ответ шлём из отдельного потока: читатель обязан читать
                    # дальше (в том числе ответ на handleJavaScriptDialog).
                    threading.Thread(target=self._accept_dialog, args=(msg,), daemon=True).start()
                with self.lock:
                    self.events.append(msg)

    def _accept_dialog(self, msg: dict) -> None:
        self.dialogs.append(str((msg.get("params") or {}).get("message") or ""))
        try:
            self.call("Page.handleJavaScriptDialog", {"accept": True},
                      session_id=msg.get("sessionId"), timeout=10)
        except Exception:
            pass

    def dialog_note(self) -> str:
        """Последний показанный диалог; ``''`` - их не было."""
        with self.lock:
            return self.dialogs[-1] if self.dialogs else ""

    def _send(self, payload: bytes) -> None:
        n = len(payload)
        head = b"\x81"
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + n.to_bytes(2, "big")
        else:
            head += bytes([0x80 | 127]) + n.to_bytes(8, "big")
        mask = os.urandom(4)
        frame = head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        # write может прийти из потока диалога: кадры не должны переплетаться.
        with self._send_lock:
            self.sock.sendall(frame)

    def call(self, method: str, params: dict | None = None, session_id: str | None = None, timeout: float = 120) -> dict:
        with self.lock:
            self._id += 1
            mid = self._id
        msg = {"id": mid, "method": method, "params": params or {}}
        if session_id:
            msg["sessionId"] = session_id
        self._send(json.dumps(msg).encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                for m in self.events:
                    if m.get("id") == mid:
                        self.events.remove(m)
                        return m
            time.sleep(0.003)
        raise TimeoutError(method)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
def cdp_ws(port: int, timeout: float = STARTUP_TIMEOUT) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=2
            ) as response:
                data = json.loads(response.read())
            return data["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.15)
    raise TimeoutError(f"Chrome CDP did not start within {timeout:g}s")


def locate_source(
    model: str, scene: str, source_root: Path = REPORTS
) -> tuple[Path | None, Path]:
    """Resolve a page only within the explicitly selected source root."""
    root = Path(source_root).expanduser().resolve()
    relative = Path(model) / "pages" / f"{scene}.html"
    if root.name == "_showcase":
        page = root / relative
    elif root.name == "pages":
        page = root / f"{scene}.html"
    else:
        page = root / "_showcase" / relative
    server_root = root.parent if root.name in ("_repairs", "_showcase") else root
    if root == REPORTS.resolve():
        server_root = root
    if page.is_file():
        try:
            page.relative_to(REPORTS.resolve())
            server_root = REPORTS.resolve()
        except ValueError:
            pass
        return page, server_root
    return None, server_root


def _served_html(page: Path) -> str:
    """Remove only the known gallery driver from the HTTP-served copy."""
    source = page.read_text(encoding="utf-8")
    # Published reports can contain an older driver revision. Identify our
    # injected script by its stable header, not by today's full driver text.
    return re.sub(
        r"<script>\s*// Сцена играет сама: драйвер измеряет окно движения\b.*?</script>",
        "", source, flags=re.DOTALL,
    )


def html_server(
    directory: Path = REPORTS,
    overlay_page: Path | None = None,
    overlay_html: str | None = None,
) -> tuple[http.server.ThreadingHTTPServer, str]:
    root = Path(directory).resolve()
    overlay_route = None
    overlay_bytes = None
    if overlay_page is not None and overlay_html is not None:
        overlay_route = overlay_page.resolve().relative_to(root).as_posix()
        overlay_bytes = overlay_html.encode("utf-8")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path.lstrip("/")
            if overlay_route is not None and route == overlay_route:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(overlay_bytes)))
                self.end_headers()
                self.wfile.write(overlay_bytes)
                return
            super().do_GET()

        def log_message(self, *args):  # noqa: D102
            pass

    handler = functools.partial(Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _frame_count(seconds: float, *, allow_zero: bool = False) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration must be a finite, non-negative number")
    count = round(seconds * FPS)
    if (not math.isclose(count / FPS, seconds, rel_tol=0, abs_tol=1e-9)
            or (count == 0 and not allow_zero)):
        raise ValueError(f"duration must be a positive multiple of 1/{FPS} second")
    return count


def _atomic_output(out: Path, args: list[str]) -> None:
    target = out.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.part.mp4")
    partial.unlink(missing_ok=True)
    try:
        run_bin(ffmpeg(), [*args, "-y", str(partial)])
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no output")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def encode(frames: Path, out: Path, frame_count: int | None = None) -> None:
    """Encode a lossless PNG sequence as a fast-start 120-fps H.264 clip."""
    if frame_count is not None and frame_count <= 0:
        raise ValueError("frame_count must be positive")
    args = [
        "-hide_banner", "-loglevel", "error",
        "-framerate", str(FPS), "-start_number", "0",
        "-i", str((frames / "%06d.png").resolve()),
    ]
    if frame_count is not None:
        args += ["-frames:v", str(frame_count)]
    args += [
        "-an", "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    _atomic_output(out, args)


def placeholder(out: Path, secs: float) -> None:
    """Write a fixed-rate dark placeholder of exactly the requested duration."""
    count = _frame_count(secs)
    _atomic_output(out, [
        "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}",
        "-frames:v", str(count), "-an",
        "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ])


def _write_capture_metadata(out: Path, source: Path | None, seconds: float,
                            status: str, details: dict | None = None) -> None:
    source_hash = None
    if source is not None and source.is_file():
        with source.open("rb") as original:
            source_hash = hashlib.file_digest(original, "sha256").hexdigest()
    data = {
        "clip": out.name, "status": status, "source_sha256": source_hash,
        "seconds": seconds, "fps": FPS, "frames": _frame_count(seconds),
        "captured_frames": 0, "crf": CRF, "preset": PRESET,
        "input_format": "png" if status == "ok" else "generated-placeholder",
        **(details or {}),
    }
    target = out.with_suffix(".capture.json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _stop_chrome(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait(timeout=10)


def _failure_note(ws, fallback: str = "") -> str:
    error = _javascript_error(ws) if ws is not None else ""
    if error:
        return error[:200]
    dialog = ws.dialog_note() if ws is not None else ""
    if dialog:
        return f"JavaScript dialog: {dialog}"[:200]
    return fallback[:200]


def record(
    model: str,
    scene: str,
    out_dir: Path,
    secs: float,
    warmup: float,
    base_url: str,
    *,
    source_root: Path = REPORTS,
    startup_timeout: float = STARTUP_TIMEOUT,
    chrome_binary: str | Path | None = None,
) -> tuple[str, str]:
    """Record a page; return (status, reason), raising on capture infrastructure errors."""
    src, server_root = locate_source(model, scene, source_root)
    if src is None:
        return "model_failed", "source page missing"
    if startup_timeout <= 0 or not math.isfinite(startup_timeout):
        raise ValueError("startup timeout must be finite and positive")
    count = _frame_count(secs)
    warmup_frames = _frame_count(warmup, allow_zero=True)
    route = src.relative_to(server_root).as_posix()
    url = f"{base_url}/{route}"
    out = out_dir / f"{model}__{scene}.mp4"
    port = free_port()
    exe = chrome(chrome_binary)
    profile = Path(tempfile.mkdtemp(prefix="record-chrome-"))
    proc = None
    ws = None
    captured = 0
    sample_hashes: dict[str, str] = {}
    distinct_hashes: set[str] = set()
    clock_modes: set[str] = set()
    first_clock_ms = None
    last_clock_ms = None

    with tempfile.TemporaryDirectory(prefix=f"record-{scene}-") as temp:
        frames = Path(temp)
        try:
            chrome_args = [
                str(exe), "--headless=new", f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
                "--hide-scrollbars", f"--window-size={W},{H}",
                "--force-device-scale-factor=1", "--password-store=basic",
                "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                "--disable-lcd-text",
            ]
            if sys.platform == "darwin":
                chrome_args.append("--use-angle=metal")
            elif sys.platform.startswith("linux"):
                chrome_args.append("--disable-dev-shm-usage")
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    chrome_args.append("--no-sandbox")
            proc = subprocess.Popen(
                chrome_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            ws = WS(cdp_ws(port, startup_timeout))
            target = ws.call("Target.createTarget", {"url": "about:blank"})
            target_id = target["result"]["targetId"]
            attached = ws.call(
                "Target.attachToTarget", {"targetId": target_id, "flatten": True}
            )
            session_id = attached["result"]["sessionId"]
            call = lambda method, params=None, **kwargs: ws.call(  # noqa: E731
                method, params, session_id=session_id, **kwargs
            )
            call("Page.enable")
            call("Runtime.enable")
            call("Page.addScriptToEvaluateOnNewDocument", {"source": FRAME_CLOCK_JS})
            call("Emulation.setDeviceMetricsOverride", {
                "width": W, "height": H, "deviceScaleFactor": 1, "mobile": False,
            })
            call("Page.navigate", {"url": url})
            ready = page_ready(call, ws, scene, timeout=startup_timeout)
            if not ready:
                return "model_failed", _failure_note(ws, "page did not become ready")

            note = _javascript_error(ws)
            if note:
                return "model_failed", note[:200]
            print(f"[rec] {model}/{scene}: ready={ready}; warming {warmup_frames} frames", flush=True)
            if warmup_frames:
                try:
                    _evaluate(call, f"window.__recordClock.warmup({warmup_frames})", timeout=120)
                except PageJavaScriptError as exc:
                    return "model_failed", _failure_note(ws, str(exc))
            note = _javascript_error(ws)
            if note:
                return "model_failed", note[:200]

            for index in range(count):
                note = _javascript_error(ws)
                if note:
                    if captured == 0:
                        return "model_failed", note[:200]
                    raise RuntimeError(f"page JavaScript failed during capture: {note[:200]}")
                try:
                    state_response = _evaluate(call, f"window.__recordClock.capture({DT!r})", timeout=120)
                    clock_state = state_response["result"]["result"]["value"]
                    expected_frame = warmup_frames + index
                    if clock_state["frame"] != expected_frame or not math.isclose(
                        clock_state["time"], expected_frame * 1000 / FPS, abs_tol=1e-7,
                    ):
                        raise RuntimeError(f"recording clock discontinuity at frame {index}: {clock_state}")
                except PageJavaScriptError as exc:
                    if captured == 0:
                        return "model_failed", _failure_note(ws, str(exc))
                    raise RuntimeError(f"page JavaScript failed during capture: {exc}") from exc
                note = _javascript_error(ws)
                if note:
                    if captured == 0:
                        return "model_failed", note[:200]
                    raise RuntimeError(f"page JavaScript failed during capture: {note[:200]}")

                screenshot = call(
                    "Page.captureScreenshot",
                    {"format": "png", "fromSurface": True, "captureBeyondViewport": False, "optimizeForSpeed": True},
                    timeout=120,
                )
                if "error" in screenshot:
                    raise RuntimeError(f"captureScreenshot failed: {screenshot['error']}")
                encoded = (screenshot.get("result") or {}).get("data") or ""
                if not encoded:
                    raise RuntimeError(f"empty PNG frame {index}: {json.dumps(screenshot)[:200]}")
                png = base64.b64decode(encoded, validate=True)
                if (not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 24
                        or int.from_bytes(png[16:20], "big") != W
                        or int.from_bytes(png[20:24], "big") != H):
                    raise RuntimeError(f"frame {index} is not a {W}x{H} PNG")
                (frames / f"{index:06d}.png").write_bytes(png)
                captured += 1
                if captured == 1 or captured % FPS == 0:
                    print(f"[rec] {model}/{scene}: captured {captured}/{count}", flush=True)
                digest = hashlib.sha256(png).hexdigest()
                distinct_hashes.add(digest)
                if index in (0, count // 2, count - 1):
                    sample_hashes[str(index)] = digest
                clock_modes.add(clock_state["mode"])
                if first_clock_ms is None:
                    first_clock_ms = clock_state["time"]
                last_clock_ms = clock_state["time"]
            note = _javascript_error(ws)
            if note:
                if captured == 0:
                    return "model_failed", note[:200]
                raise RuntimeError(f"page JavaScript failed during capture: {note[:200]}")

            warnings = _evaluate(call, "window.__recordClock.warnings()")
            warning_list = ((warnings.get("result") or {}).get("result") or {}).get("value") or []
            if warning_list:
                raise RuntimeError("recording clock cannot control: " + "; ".join(warning_list))
        finally:
            try:
                if ws is not None:
                    ws.sock.close()
            except OSError:
                pass
            finally:
                try:
                    _stop_chrome(proc)
                finally:
                    shutil.rmtree(profile, ignore_errors=True)

        if captured != count:
            raise RuntimeError(f"captured {captured} of {count} requested frames")
        encode(frames, out, count)
        _write_capture_metadata(out, src, secs, "ok", {
            "captured_frames": captured, "clock_modes": sorted(clock_modes),
            "clock_start_ms": first_clock_ms, "clock_end_ms": last_clock_ms,
            "clock_discontinuities": 0, "distinct_png_frames": len(distinct_hashes),
            "sample_hashes": sample_hashes,
        })
    return "ok", ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("seconds", type=float, nargs="?", default=DEFAULT_SECS)
    parser.add_argument("warmup_s", type=float, nargs="?", default=DEFAULT_WARMUP)
    parser.add_argument("--force", action="store_true", help="replace an existing clip")
    parser.add_argument("--source-root", type=Path, default=REPORTS)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--crf", type=int, default=CRF)
    parser.add_argument("--preset", default=PRESET)
    parser.add_argument("--chrome")
    parser.add_argument(
        "--startup-timeout", type=float, default=STARTUP_TIMEOUT,
        help="maximum browser/page startup wait in seconds (default: 180)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = check_target(args.model, args.scene, args.out_dir)
    try:
        _frame_count(args.seconds)
        _frame_count(args.warmup_s, allow_zero=True)
    except ValueError as exc:
        print(f"[rec] invalid duration: {exc}", file=sys.stderr)
        return 2
    if (args.fps, args.crf, args.preset) != (FPS, CRF, PRESET):
        print(
            f"[rec] encoding is fixed at {FPS} fps, CRF {CRF}, preset {PRESET}",
            file=sys.stderr,
        )
        return 2
    if args.startup_timeout <= 0 or not math.isfinite(args.startup_timeout):
        print("[rec] startup timeout must be finite and positive", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    model, scene = args.model, args.scene
    out = out_dir / f"{model}__{scene}.mp4"
    if out.is_file() and not args.force:
        print(f"[rec] {model}/{scene}: already recorded")
        return 0

    src, server_root = locate_source(model, scene, args.source_root)
    if src is None:
        reason = media_status.task_status(model, scene, ROOT / "runs") or "model_failed"
        if reason == "ok":
            reason = "model_failed"
        try:
            placeholder(out, args.seconds)
        except Exception as exc:
            media_status.write_marker(out_dir, model, scene, "capture_failed", str(exc))
            print(f"[rec] {model}/{scene}: placeholder failed: {exc}", file=sys.stderr, flush=True)
            return EXIT["capture_failed"]
        media_status.write_marker(out_dir, model, scene, reason)
        _write_capture_metadata(out, None, args.seconds, reason, {"reason": "source page missing"})
        print(f"[rec] {model}/{scene}: source missing -> placeholder ({reason})")
        return EXIT["model_failed"]

    server = None
    started = time.monotonic()
    status = "ok"
    note = ""
    try:
        try:
            server, base_url = html_server(server_root, src, _served_html(src))
            status, note = record(
                model, scene, out_dir, args.seconds, args.warmup_s, base_url,
                source_root=args.source_root, startup_timeout=args.startup_timeout,
                chrome_binary=args.chrome,
            )
        except Exception as exc:  # noqa: BLE001 - capture/encoding, not model verdict
            status = "capture_failed"
            note = str(exc)
            print(f"[rec] {model}/{scene}: capture failed: {exc}", file=sys.stderr, flush=True)

        if status != "ok":
            if status == "capture_failed":
                media_status.write_marker(out_dir, model, scene, status, note)
                return EXIT[status]
            try:
                placeholder(out, args.seconds)
            except Exception as exc:
                status = "capture_failed"
                note = f"{note}; placeholder failed: {exc}" if note else str(exc)
                print(f"[rec] {model}/{scene}: placeholder failed: {exc}", file=sys.stderr, flush=True)
            media_status.write_marker(out_dir, model, scene, status, note)
            _write_capture_metadata(out, src, args.seconds, status, {"reason": note})
            print(f"[rec] {model}/{scene}: placeholder ({status}{': ' + note if note else ''})")
            return EXIT[status]

        media_status.clear_marker(out_dir, model, scene)
        elapsed = time.monotonic() - started
        print(f"[rec] {model}/{scene}: {out.stat().st_size / 1e6:.1f} MB in {elapsed:.0f}s")
        return EXIT["ok"]
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()














if __name__ == "__main__":
    sys.exit(main())
