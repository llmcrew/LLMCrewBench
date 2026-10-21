#!/usr/bin/env python3
"""Запись живого клипа сцены: страница решения реально исполняется в Chrome.

Драйвер страницы (STEP_HOOK галереи) сам замеряет окно движения сцены и
разворачивает его на показ; клип снимает то же окно покадрово через
``__dbscene.seek(t)`` — кадр клипа равен кадру показа, движение не проскакивает.
Окно длиннее клипа играется в несколько заходов, между заходами страница
перезагружается: сброса у сцены нет, перезагрузка возвращает её в начало.
Живые кадры, не стоп-кадры.

Использование:
  python3 record_clip.py <model> <scene> <out_dir> [seconds] [warmup_s]

Источник — артефакт `reports/comparison/_showcase/<model>/pages/<scene>.html`
(отдаётся локальным http-сервером, `_vendor` рядом). Нет страницы — пишем
плашку (тёмный клип), причину кладём в маркер: `not_run`, если задачи нет в
прогоне, иначе `model_failed`; текст рисует Remotion. Коды выхода: 0 — клип,
3 — страница не дошла до кадра (`model_failed`), 4 — сбой записи
(`capture_failed`; страница при этом может быть жива).

Все клипы: 1920x1080, 120 fps, h264, 23 с (окно движения сцены целиком).
Частота выше частоты показа (120 против 60): при раскрытии полосы клип
переигрывается с первого кадра ускоренно (23 с клипа за 8 с показа), и запас
кадров держит это движение плавным.
Страницы K (canvas 800x600) вписываются в кадр целиком, а пустые поля по
сторонам заливаются растянутыми краями сцены — кадр идёт без рамки и полос.
"""
from __future__ import annotations

import base64
import functools
import http.server
import json
import os
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

sys.path.insert(0, str(Path(__file__).parent))
import media_status  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "video"
#: Источник страниц — последнее сравнение (два ingress'а одной модели).
ART = ROOT / "reports" / "comparison-deepseek-1017"
OUT = VIDEO / "public" / "clips"
MODELS = ("deepseek-v4.1-flash-unc", "deepseek-v4.1-flash-unc-1017")
SCENES = tuple(f"K-{i}" for i in range(1, 21)) + tuple(f"L-{i}" for i in range(1, 30))

#: Частота клипа. Пара с `FPS` в `src/video/data.ts`: клип снят покадрово, и
#: расхождение играло бы его рывками (кадр клипа не ложился бы на кадр ролика).
#: Шагов сцены на клип ровно `secs * FPS` — с ростом частоты шаг мельче.
FPS = 120
#: Шаг страницы без драйвера (`window.__step`): один кадр клипа — один шаг,
#: как в её живом показе (там тот же шаг зовётся из rAF).
DT = 1 / FPS
W, H = 1920, 1080

#: Коды выхода: 0 — клип снят, 3 — страница не дошла до кадра, 4 — сбой записи.
EXIT = {"ok": 0, "model_failed": 3, "capture_failed": 4}


def _find(name: str, *preferred: str, verify: bool = False) -> Path | None:
    """Бинарник: сначала кандидаты платформы (macOS), затем PATH.

    ``verify`` отсеивает бинарник чужой платформы: комплектный macOS-arm ffmpeg
    на Linux иначе выбран бы и упал на exec format.
    """
    for candidate in (*preferred, shutil.which(name) or ""):
        if not candidate:
            continue
        path = Path(candidate)
        if not (path.is_file() and os.access(path, os.X_OK)):
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
        raise SystemExit("ffmpeg не найден: положите его в PATH или в video/bin/")
    return path


def ffprobe() -> Path:
    path = _find("ffprobe", str(VIDEO / "bin" / "ffprobe"), verify=True)
    if path is None:
        raise SystemExit("ffprobe не найден: положите его в PATH или в video/bin/")
    return path


def chrome() -> Path:
    path = _find(
        "google-chrome",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        str(VIDEO / "bin" / "chrome"),
        verify=True,
    ) or _find("chromium", verify=True) or _find("chromium-browser", verify=True) or _find(
        "google-chrome-stable", verify=True
    )
    if path is None:
        raise SystemExit("Chrome не найден: поставьте Google Chrome или chromium в PATH")
    return path


def run_bin(exe: Path, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Запуск комплектного бинарника: dylib лежат рядом, cwd обязан быть его."""
    return subprocess.run([str(exe), *args], cwd=str(exe.parent), check=True, **kwargs)


def check_target(model: str, scene: str, out_dir: Path) -> Path:
    """Containment аргументов CLI: модель/сцена по схеме, OUT только в public/clips."""
    if model not in MODELS:
        raise SystemExit(f"неизвестная модель {model!r}: {', '.join(MODELS)}")
    if scene not in SCENES:
        raise SystemExit(f"неизвестная сцена {scene!r}: K-1..K-20, L-1..L-29")
    target = out_dir.resolve()
    try:
        target.relative_to(OUT.resolve())
    except ValueError:
        raise SystemExit(f"OUT {target} вне {OUT}") from None
    return target


#: Плашки страницы (модель · сцена · клавиши) в видео не нужны: подписи свои,
#: а плашка перекрывает сцену. Перезагрузка посреди клипа сносит DOM, поэтому
#: блок применяется после каждой загрузки.
#: NB: обёртка — IIFE: Runtime.evaluate исполняет ВЫРАЖЕНИЕ, и без вызова
#: `() => {...}` остаётся функцией, которая никогда не выполнится.
STYLE_JS = """
(() => {
  for (const d of document.body.children) {
    if (d.tagName !== 'DIV' && d.tagName !== 'SPAN') continue;
    const st = getComputedStyle(d);
    if (st.position !== 'fixed' && st.position !== 'absolute') continue;
    if (!d.querySelector('canvas')) d.style.display = 'none';
  }

  // Поля кадра. canvas вписывается целиком (K рисует 800x600, арт-сцены —
  // какую выбрала модель), а пустые поля заливаются растянутыми краями сцены:
  // у 800x600-страниц поле было белым (body страницы), у тёмных — своим цветом
  // с резкой границей у канваса, и кадр шёл в рамке. Тянутый край продолжает
  // сцену за её границей. Замер на пилотах: перепад на границе 0-1/255, у
  // заливки полей одним средним цветом — 48/255 (край сцены не одноцветный).
  let bg = document.getElementById('__bgfill');
  if (!bg) {
    bg = document.createElement('canvas');
    bg.id = '__bgfill';
    bg.style.cssText = 'position:fixed;left:0;top:0;width:100vw;height:100vh;z-index:0;pointer-events:none';
    document.body.appendChild(bg);
  }
  const g = bg.getContext('2d');
  // Образец краёв — отдельным канвасом в половину разрешения: средний цвет
  // ставится фоном html/body, под подложку (запасной слой на случай, если
  // подложка не отрисуется). Читать пиксели — только с CPU-канваса.
  const probe = document.createElement('canvas');
  const pg = probe.getContext('2d', {willReadFrequently: true});
  document.documentElement.style.colorScheme = 'dark';
  document.documentElement.style.background = '#000';
  document.body.style.background = 'transparent';

  // Перерисовка подложки: зовётся тем же evaluate, что и шаг сцены, — кадр
  // клипа снимается сразу после, и подложка обязана быть уже свежей.
  window.__refill = () => {
    // Именно canvas сцены: `querySelector('canvas')` вернул бы подложку, если
    // страница добавит свой canvas позже (её appendChild идёт после нашего).
    // Среди нескольких canvas сцены берём последний: `renderer.domElement`
    // страница всегда добавляет после своей разметки, а пустой canvas-декой
    // разметки (qwen36/L-2: `<canvas id="c">` в теле, сцена — в новом canvas,
    // добавленном ниже) иначе попадает в кадр пустым и сжимает сцену в половину
    // кадра. Остальные canvas-декои прячем, чтобы они не делили место в body.
    const cs = [...document.querySelectorAll('canvas')].filter((el) => el !== bg);
    const c = cs[cs.length - 1];
    if (!c || !c.width || !c.height) return false;
    for (const el of cs) el.style.display = el === c ? 'block' : 'none';
    const k = Math.min(1920 / c.width, 1080 / c.height);
    c.style.width = `${Math.round(c.width * k)}px`;
    c.style.height = `${Math.round(c.height * k)}px`;
    c.style.position = 'relative';
    c.style.zIndex = '1';
    const r = c.getBoundingClientRect();
    const x = Math.round(r.x), y = Math.round(r.y), w = Math.round(r.width), h = Math.round(r.height);
    const vw = Math.round(window.innerWidth), vh = Math.round(window.innerHeight);
    if (bg.width !== vw || bg.height !== vh) { bg.width = vw; bg.height = vh; }
    g.clearRect(0, 0, vw, vh);
    // Края сцены (полоса в 1 пиксель) тянутся в поля и в углы: сцена
    // продолжается, стыка не видно ни по бокам, ни сверху и снизу.
    if (x > 0) g.drawImage(c, 0, 0, 1, c.height, 0, 0, x, vh);
    if (x + w < vw) g.drawImage(c, c.width - 1, 0, 1, c.height, x + w, 0, vw - x - w, vh);
    if (y > 0) g.drawImage(c, 0, 0, c.width, 1, x, 0, w, y);
    if (y + h < vh) g.drawImage(c, 0, c.height - 1, c.width, 1, x, y + h, w, vh - y - h);
    g.drawImage(c, x, y, w, h);

    const pw = Math.max(1, Math.round(c.width / 2)), ph = Math.max(1, Math.round(c.height / 2));
    if (probe.width !== pw || probe.height !== ph) { probe.width = pw; probe.height = ph; }
    pg.drawImage(c, 0, 0, pw, ph);
    let sr = 0, sg = 0, sb = 0, sn = 0;
    const bands = [pg.getImageData(0, 0, 1, ph).data, pg.getImageData(pw - 1, 0, 1, ph).data,
                   pg.getImageData(0, 0, pw, 1).data, pg.getImageData(0, ph - 1, pw, 1).data];
    for (const band of bands) {
      for (let i = 0; i < band.length; i += 28) { sr += band[i]; sg += band[i + 1]; sb += band[i + 2]; sn++; }
    }
    const col = `rgb(${Math.round(sr / sn)},${Math.round(sg / sn)},${Math.round(sb / sn)})`;
    if (col !== window.__bgsolid) {
      window.__bgsolid = col;
      document.documentElement.style.background = col;
    }
    return true;
  };

  document.body.style.margin = '0';
  document.body.style.overflow = 'hidden';
  document.body.style.display = 'flex';
  document.body.style.alignItems = 'center';
  document.body.style.justifyContent = 'center';
  // Страница может задать body жёсткую ширину (у minimax `body{width:800px}`):
  // без растягивания body увеличенный canvas обрезается её размером.
  document.body.style.width = '100vw';
  document.body.style.height = '100vh';
  document.body.style.maxWidth = 'none';
  document.body.style.maxHeight = 'none';
  document.documentElement.style.zoom = '';
  window.__refill();
  return true;
})()"""

#: Ждём, пока драйвер страницы закончит замер окна движения: замер кончается
#: перезагрузкой страницы, поэтому ждать приходится через неё.
SCENE_READY = """
new Promise((res) => {
  const t0 = performance.now();
  const t = setInterval(() => {
    const s = window.__dbscene;
    if (s && s.mode && s.mode !== 'measuring') { clearInterval(t); res(true); return; }
    if (!s && performance.now() - t0 > 8000) { clearInterval(t); res(false); return; }
    if (performance.now() - t0 > 90000) { clearInterval(t); res(false); }
  }, 25);
})
"""


def page_ready(call, ws, scene: str, timeout: float = 12.0) -> str:
    """Дошла ли страница до кадра; ``''`` — нет, иначе способ готовности.

    K детерминированы и обязаны выставить ``__ready``/``__dbscene``. L хуков не
    обещают: там ждём settled-состояние — документ догружен, разметка перестала
    меняться, а на странице есть непустой текст/canvas/svg. ``__ready`` и драйвер
    принимаются и на L: свободная страница может их иметь.

    Поллим короткими evaluate: драйвер перезагружает вкладку и сносит контекст,
    висящий промис при этом обрывается. Упавшая страница не даст ни признака.
    """
    probe = """
(() => {
  if (window.__ready === true) return 'ready';
  const s = window.__dbscene;
  if (s && s.mode) return s.mode === 'measuring' ? '' : 'driver';
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
    deadline = time.time() + (timeout * 2 if not deterministic else timeout)
    settled_at: float | None = None
    last = ""
    while time.time() < deadline:
        if any(e.get("method") == "Runtime.exceptionThrown" for e in ws.events):
            # Исключение до кадра — вердикт, ждать больше нечего.
            return ""
        try:
            r = call("Runtime.evaluate", {"expression": probe, "returnByValue": True}, timeout=5)
        except TimeoutError:
            continue
        value = ((r.get("result") or {}).get("result") or {}).get("value") or ""
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
                    # K обязан выставить протокол: settled без хука — не K-кадр.
                    return "settled"
            else:
                last = mark
                settled_at = None
        if not deterministic and not value:
            last = ""
            settled_at = None
        time.sleep(0.05)
    return ""


def wait_scene(call, deadline: float = 100.0) -> bool:
    """Дождаться драйвера страницы; True — сцена играет своим ходом.

    Перезагрузка между замером и показом сносит контекст вместе с висящим
    awaitPromise, поэтому вызов повторяем, а не падаем.
    """
    limit = time.time() + deadline
    while time.time() < limit:
        try:
            r = call("Runtime.evaluate", {"expression": SCENE_READY, "awaitPromise": True,
                                          "returnByValue": True}, timeout=max(5.0, limit - time.time()))
        except TimeoutError:
            return False
        if "error" in r:
            time.sleep(0.05)
            continue
        return bool(((r.get("result") or {}).get("result") or {}).get("value"))
    return False


def apply_styles(call) -> None:
    call("Runtime.evaluate", {"expression": STYLE_JS, "returnByValue": True})


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
        """Последний показанный диалог; ``''`` — их не было."""
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


def cdp_ws(port: int) -> str:
    for _ in range(200):
        try:
            data = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2).read())
            return data["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.15)
    raise SystemExit("CDP не поднялся")


def html_server() -> tuple[http.server.ThreadingHTTPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D102
            pass
    handler = functools.partial(Quiet, directory=str(ART))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def encode(frames: Path, out: Path) -> None:
    exe = ffmpeg()
    run_bin(exe, ["-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
                  "-i", str(frames / "%06d.jpg"),
                  "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,scale={W}:{H}",
                  "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                  "-pix_fmt", "yuv420p", "-y", str(out)])


def placeholder(out: Path, secs: float) -> None:
    exe = ffmpeg()
    run_bin(exe, ["-hide_banner", "-loglevel", "error",
                  # В комплектном ffmpeg нет фильтра `color`: берём nullsrc.
                  "-f", "lavfi", "-i", f"nullsrc=s={W}x{H}:r={FPS}:d={secs}",
                  "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                  "-pix_fmt", "yuv420p", "-y", str(out)])


def record(model: str, scene: str, out_dir: Path, secs: float, warmup: float, base_url: str) -> tuple[str, str]:
    """Снять клип: ``("статус", причина)``. Сбой инфраструктуры — исключение.

    Причина — текст JS-диалога страницы, если он был: диалоги принимаются
    автоматически, но их след остаётся для маркера.
    """
    src = ART / "_showcase" / model / "pages" / f"{scene}.html"
    frames = out_dir / "_frames" / f"{model}__{scene}"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    port = free_port()
    exe = chrome()
    profile = Path(tempfile.mkdtemp(prefix="rec-chrome-"))
    proc = subprocess.Popen(
        [str(exe), "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check",
         "--use-angle=metal", "--disable-gpu-sandbox",
         "--hide-scrollbars", f"--window-size={W},{H}",
         "--force-device-scale-factor=1", "--use-mock-keychain", "--password-store=basic",
         "--disable-background-timer-throttling", "--disable-lcd-text",
         f"file://{src.resolve()}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:

        ws = WS(cdp_ws(port))
        tgt = ws.call("Target.createTarget", {"url": "about:blank"})
        sid = ws.call("Target.attachToTarget", {"targetId": tgt["result"]["targetId"], "flatten": True})["result"]["sessionId"]
        call = lambda m, p=None, **kw: ws.call(m, p, session_id=sid, **kw)  # noqa: E731
        url = f"{base_url}/_showcase/{model}/pages/{scene}.html"
        call("Page.enable")
        call("Runtime.enable")
        call("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
        call("Page.navigate", {"url": url})
        # Готовность — по режиму сцены, без единого требования `__ready` к L.
        # Страница, упавшая до первого кадра, честно получает model_failed:
        # тот же вердикт, что у harness (0 баллов), а не чёрный клип.
        ready = page_ready(call, ws, scene)
        if not ready:
            note = ws.dialog_note()
            shutil.rmtree(frames, ignore_errors=True)
            return "model_failed", (f"диалог: {note}" if note else "")
        # Драйвер страницы замеряет окно движения и перезагружается — только
        # после перезагрузки сцена стоит в начале (шаг 0). Паузу шлём тем же
        # evaluate, что застал драйвер живым: без лишнего round-trip, за который
        # автоплей успел бы уйти вперёд (seek назад не ходит).
        has_driver = ready == "driver"
        if not has_driver:
            has_driver = bool(((call("Runtime.evaluate", {
                "expression": "!!(window.__dbscene && window.__dbscene.mode)",
                "returnByValue": True})).get("result") or {}).get("result", {}).get("value"))
        has_driver = has_driver and wait_scene(call)
        base = 0
        if has_driver:
            r = call("Runtime.evaluate", {"expression": """
(() => {
  const s = window.__dbscene;
  if (!s) return -1;
  window.postMessage('pause', '*');
  return s.step | 0;
})()""", "returnByValue": True}, timeout=90)
            base = max(0, int((r.get("result") or {}).get("result", {}).get("value") or 0))
        apply_styles(call)
        time.sleep(warmup)

        # Окно движения укладывается в клип целиком: последний кадр — конец
        # окна, движение идёт все `secs` секунд. Живая страница тратит на тот
        # же шаг ~0.1 с (SPAN_PER_STEP), так что темп клипа и показа совпадают
        # на окнах от ~230 шагов и слегка медленнее на коротких — видимая
        # разница только на сценах, которые на живом показе пролетают за 4 с.
        info = call("Runtime.evaluate", {
            "expression": """
(() => {
  const s = window.__dbscene;
  if (!s || !s.freeze) return JSON.stringify({steps: 0});
  return JSON.stringify({steps: s.freeze});
})()""", "returnByValue": True})
        info = json.loads((info.get("result") or {}).get("result", {}).get("value") or "{}")
        steps = int(info.get("steps") or 0)
        n = int(round(secs * FPS))
        for k in range(n):
            if has_driver:
                # steps=0 (статичная сцена): seek(0) гарантирует первый кадр.
                # max(base, ...) — floor по шагам, которые автоплей сделал до
                # паузы: seek назад не ходит.
                t = max(base, round(k * steps / max(1, n - 1))) if steps else base
                js = f"window.__dbscene.seek({t});"
            else:
                # Страница без драйвера: шагаем сами, один кадр клипа — один
                # шаг её живого показа (шаг зовётся из rAF через DT). Следом —
                # кадр: у части страниц рендер живёт только в __probe, так их
                # зовёт живая витрина; без вызова клип остаётся чёрным
                # (qwen36/L-4: __step лишь копит время, render — в __probe).
                js = (f"try{{window.__step({DT});}}catch(e){{}} "
                      "try{window.__probe&&window.__probe();}catch(e){}")
            # Подложка полей перерисовывается тем же evaluate: между шагом и
            # кадром нет round-trip, за который буфер отрисовки мог бы остаться
            # от прежнего шага.
            call("Runtime.evaluate", {"expression": js + " window.__refill && window.__refill();",
                                      "returnByValue": False}, timeout=60)
            shot = call("Page.captureScreenshot", {"format": "jpeg", "quality": 90, "optimizeForSpeed": True}, timeout=120)
            data = shot.get("result", {}).get("data", "")
            if not data:
                raise RuntimeError(f"пустой кадр {k}: {json.dumps(shot)[:200]}")
            (frames / f"{k:06d}.jpg").write_bytes(base64.b64decode(data))
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)
    out = out_dir / f"{model}__{scene}.mp4"
    encode(frames, out)
    shutil.rmtree(frames, ignore_errors=True)
    return "ok", ""


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    model, scene = sys.argv[1], sys.argv[2]
    out_dir = check_target(model, scene, Path(sys.argv[3]))
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 23.0
    warmup = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    force = "--force" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = ROOT / "runs"
    src = ART / "_showcase" / model / "pages" / f"{scene}.html"
    out = out_dir / f"{model}__{scene}.mp4"
    if out.is_file() and not force:
        print(f"[rec] {model}/{scene}: уже есть")
        return 0
    if not src.is_file():
        # Отсутствие страницы — не всегда вина модели: задачи могло не быть в
        # прогоне. Причину пишем в маркер, а не общее `1`.
        reason = media_status.task_status(model, scene, runs_root) or "model_failed"
        if reason == "ok":
            reason = "model_failed"
        placeholder(out, secs)
        media_status.write_marker(out_dir, model, scene, reason)
        print(f"[rec] {model}/{scene}: нет страницы -> плашка ({reason})")
        return EXIT["model_failed"]
    srv, base = html_server()
    try:
        t0 = time.time()
        note = ""
        try:
            status, note = record(model, scene, out_dir, secs, warmup, base)
        except Exception as exc:  # noqa: BLE001 — сбой записи, а не вердикт модели
            status = "capture_failed"
            print(f"[rec] {model}/{scene}: сбой записи: {exc}", file=sys.stderr, flush=True)
        if status != "ok":
            # Плашка вместо черноты; маркер объясняет причину и не даёт ретраю
            # переснимать сцену зря.
            placeholder(out, secs)
            media_status.write_marker(out_dir, model, scene, status, note)
            print(f"[rec] {model}/{scene}: плашка ({status}{', ' + note if note else ''})")
            return EXIT[status]
        media_status.clear_marker(out_dir, model, scene)
        print(f"[rec] {model}/{scene}: {out.stat().st_size/1e6:.1f} МБ за {time.time()-t0:.0f}с")
    finally:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
