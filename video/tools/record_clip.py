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
плашку (тёмный клип), текст «модель не справилась» рисует Remotion.

Все клипы: 1920x1080, 60 fps, h264, 23 с (окно движения сцены целиком).
Страницы K (canvas 800x600) вписываются в кадр целиком, а пустые поля по
сторонам заливаются растянутыми краями сцены — кадр идёт без рамки и полос.
"""
from __future__ import annotations

import base64
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "reports" / "comparison"
BIN = ROOT / "video" / "bin"
FF = BIN / "ffmpeg"
#: Частота клипа. Пара с `FPS` в `src/video/data.ts`: клип снят покадрово, и
#: расхождение играло бы его рывками (кадр клипа не ложился бы на кадр ролика).
#: Шагов сцены на клип ровно `secs * FPS` — с ростом частоты шаг мельче.
FPS = 60
#: Шаг страницы без драйвера (`window.__step`): один кадр клипа — один шаг,
#: как в её живом показе (там тот же шаг зовётся из rAF).
DT = 1 / FPS
W, H = 1920, 1080
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

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


def page_ready(call, timeout: float = 12.0) -> bool:
    """Дошла ли страница до первого кадра: ``__ready`` ставит её последняя строка.

    Поллим коротышами, а не одним awaitPromise: драйвер страницы перезагружает
    вкладку (сносит контекст), и висящий промис обрывается ошибкой. Страница,
    упавшая на исключении, не выставит ``__ready`` никогда.
    """
    limit = time.time() + timeout
    while time.time() < limit:
        r = call("Runtime.evaluate", {"expression": "window.__ready === true", "returnByValue": True}, timeout=5)
        if "error" not in r and ((r.get("result") or {}).get("result") or {}).get("value") is True:
            return True
        time.sleep(0.05)
    return False


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
    """Минимальный CDP-клиент (websocket с обязательной маскировкой фреймов)."""

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
                with self.lock:
                    self.events.append(msg)

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
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

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
    subprocess.run([str(FF), "-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(frames / "%06d.jpg"),
                    "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,scale={W}:{H}",
                    "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                    "-pix_fmt", "yuv420p", "-y", str(out)], cwd=str(FF.parent), check=True)


def placeholder(out: Path, secs: float) -> None:
    subprocess.run([str(FF), "-hide_banner", "-loglevel", "error",
                    # В комплектном ffmpeg нет фильтра `color`: берём nullsrc.
                    "-f", "lavfi", "-i", f"nullsrc=s={W}x{H}:r={FPS}:d={secs}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                    "-pix_fmt", "yuv420p", "-y", str(out)], cwd=str(FF.parent), check=True)


def record(model: str, scene: str, out_dir: Path, secs: float, warmup: float, base_url: str) -> bool:
    """Снять клип. ``False`` — страница падает до первого кадра, звать плашку."""
    src = ART / "_showcase" / model / "pages" / f"{scene}.html"
    frames = out_dir / "_frames" / f"{model}__{scene}"
    subprocess.run(["rm", "-rf", str(frames)], check=False)
    frames.mkdir(parents=True)
    port = free_port()
    profile = f"/tmp/rec-chrome-{os.getpid()}"
    proc = subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
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
        call("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
        # Один awaitPromise-eval вместо тысяч round-trip: готовность сцены
        # наступает за десятки миллисекунд, а поллинг из Python по 0.1 с
        # растягивал её до десятков секунд на каждую сцену.
        call("Page.navigate", {"url": url})
        call("Runtime.evaluate", {"expression": """
new Promise((res) => {
  const t0 = performance.now();
  const t = setInterval(() => {
    const done = window.__ready === true ||
                 (!!document.querySelector('canvas') && performance.now() - t0 > 2500) ||
                 performance.now() - t0 > 60000;
    if (done) { clearInterval(t); res(true); }
  }, 25);
})""", "awaitPromise": True, "returnByValue": True}, timeout=90)
        # Страница, упавшая на исключении до первого кадра, отдаёт чёрный клип.
        # Вердикт тот же, что у harness (0 баллов), так что честнее плашка:
        # маркер .missing -> витрина нарисует «модель не справилась».
        ready = page_ready(call)
        if not ready:
            subprocess.run(["rm", "-rf", str(frames)], check=False)
            return False
        # Драйвер страницы замеряет окно движения и перезагружается — только
        # после перезагрузки сцена стоит в начале (шаг 0). Паузу шлём тем же
        # evaluate, что застал драйвер живым: без лишнего round-trip, за который
        # автоплей успел бы уйти вперёд (seek назад не ходит).
        has_driver = wait_scene(call)
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
        subprocess.run(["rm", "-rf", profile], check=False)
    out = out_dir / f"{model}__{scene}.mp4"
    encode(frames, out)
    subprocess.run(["rm", "-rf", str(frames)], check=False)
    return True


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    # resolve: encode() зовёт ffmpeg с cwd=bin, относительный путь до кадров
    # иначе ищется от bin/ и сборка падает.
    model, scene, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 23.0
    warmup = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ART / "_showcase" / model / "pages" / f"{scene}.html"
    out = out_dir / f"{model}__{scene}.mp4"
    if out.is_file() and "--force" not in sys.argv:
        print(f"[rec] {model}/{scene}: уже есть")
        return 0
    if not src.is_file():
        placeholder(out, secs)
        (out_dir / f"{model}__{scene}.missing").write_text("1")
        print(f"[rec] {model}/{scene}: нет страницы -> плашка")
        return 0
    srv, base = html_server()
    try:
        t0 = time.time()
        alive = record(model, scene, out_dir, secs, warmup, base)
        if not alive:
            # Страница упала до кадра — плашка вместо черноты (и маркер, чтобы
            # витрина её нарисовала, а ретрай не переснимал зря).
            placeholder(out, secs)
            (out_dir / f"{model}__{scene}.missing").write_text("1")
            print(f"[rec] {model}/{scene}: страница упала -> плашка")
            return 0
        (out_dir / f"{model}__{scene}.missing").unlink(missing_ok=True)
        print(f"[rec] {model}/{scene}: {out.stat().st_size/1e6:.1f} МБ за {time.time()-t0:.0f}с")
    finally:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
