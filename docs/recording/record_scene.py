#!/usr/bin/env python3
"""record_scene: запись живой веб-сцены в клип фиксированной длины.

Фреймворк отделён от бенчмарка: ему нужны только Chrome, ffmpeg и страница,
которая выставляет ``window.__ready`` и умеет шагать. Всё остальное —
детали конкретного использования, которые передаются параметрами.

Два режима, одинаковый контракт кадра:

* ``record_visited`` — страница исполняется в headless-Chrome, кадр за
  кадром снимается через CDP. Если на странице есть драйвер показа
  (``window.__dbscene``, см. ``page-driver.js``), кадры двигает он: клип
  повторяет то, что видит зритель, и ничего не проскакивает.
* ``record_steps`` — страница без драйвера: фреймворк сам зовёт
  ``window.__step(dt)`` и, если есть, ``window.__probe()``.

Клип всегда ровно ``secs * fps`` кадров: кадр клипа — кадр показа.

    python3 record_scene.py URL OUT.mp4 [--secs 23] [--warmup 0.6]
        [--width 1920] [--height 1080] [--fps 60] [--crf 21]
        [--chrome PATH] [--ffmpeg PATH] [--mode auto|steps|visited]

Библиотечное использование::

    from record_scene import Options, record_visited

    result = record_visited("model/scene", url, Path("out"), Options())
    if result.status != "ok":
        ...  # "missing": страница умерла до первого кадра
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEFAULT_FFMPEG = "ffmpeg"


@dataclasses.dataclass(frozen=True)
class Options:
    """Параметры записи. Замораживаются, чтобы клип и рендер не разошлись."""

    secs: float = 23.0
    warmup: float = 0.6
    width: int = 1920
    height: int = 1080
    fps: int = 60
    crf: int = 21
    preset: str = "medium"
    jpeg_quality: int = 90
    ready_timeout: float = 60.0
    canvas_fallback: float = 2.5
    chrome: str = DEFAULT_CHROME
    ffmpeg: str = DEFAULT_FFMPEG
    mode: str = "auto"

    @property
    def frames(self) -> int:
        return int(round(self.secs * self.fps))

    @property
    def dt(self) -> float:
        return 1 / self.fps


@dataclasses.dataclass
class Result:
    status: str            # "ok" | "missing"
    frames: int = 0
    seconds: float = 0.0
    reason: str = ""


# --------------------------------------------------------------------------- #
# CDP
# --------------------------------------------------------------------------- #


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def cdp_ws(port: int, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as page:
                return json.loads(page.read())["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001 - Chrome ещё поднимается
            time.sleep(0.1)
    raise RuntimeError("CDP не поднялся")


class WS:
    """Минимальный CDP-клиент: текстовые кадры, id, wait_id."""

    def __init__(self, url: str) -> None:
        parts = urllib.parse.urlsplit(url)
        self._sock = socket.create_connection((parts.hostname, parts.port or 80), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        path = parts.path + ("?" + parts.query if parts.query else "")
        self._sock.sendall(
            (
                f"GET {path} HTTP/1.1\r\nHost: {parts.hostname}:{parts.port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        # Дочитываем заголовки ответа целиком: остаток буфера уже принадлежит
        # websocket-потоку, потерять его нельзя.
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise RuntimeError("CDP закрыл соединение на рукопожатии")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"CDP отказал в upgrade: {head[:120]!r}")
        self._pending = rest
        self._id = 0
        self._lock = threading.Lock()
        self._answers: dict[int, dict] = {}
        self.events: list[dict] = []
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    # -- websocket framing -------------------------------------------------- #

    def _read_exact(self, n: int) -> bytes:
        out = b""
        if self._pending:
            take = self._pending[:n]
            self._pending = self._pending[n:]
            out = take
        while len(out) < n:
            chunk = self._sock.recv(n - len(out))
            if not chunk:
                raise RuntimeError("CDP-сокет закрыт")
            out += chunk
        return out

    def _reader(self) -> None:
        while True:
            try:
                head = self._read_exact(2)
                opcode = head[0] & 0x0F
                length = head[1] & 0x7F
                if length == 126:
                    length = int.from_bytes(self._read_exact(2), "big")
                elif length == 127:
                    length = int.from_bytes(self._read_exact(8), "big")
                payload = self._read_exact(length)
            except RuntimeError:
                return  # штатное закрытие: процесс Chrome погашен
            if opcode == 1:  # text
                message = json.loads(payload)
                if "id" in message:
                    self._answers[message["id"]] = message
                else:
                    with self.lock:
                        self.events.append(message)

    def _send(self, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytes([0x81])
        length = len(payload)
        if length < 126:
            header += bytes([0x80 | length])
        elif length < 65536:
            header += bytes([0x80 | 126]) + length.to_bytes(2, "big")
        else:
            header += bytes([0x80 | 127]) + length.to_bytes(8, "big")
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(header + mask + masked)

    def call(self, method: str, params: dict | None = None, *, session_id: str | None = None,
             timeout: float = 120.0) -> dict:
        with self._lock:
            self._id += 1
            call_id = self._id
        message: dict[str, Any] = {"id": call_id, "method": method, "params": params or {}}
        if session_id:
            message["sessionId"] = session_id
        self._send(json.dumps(message).encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            if call_id in self._answers:
                answer = self._answers.pop(call_id)
                if "error" in answer:
                    raise RuntimeError(f"{method}: {answer['error']}")
                return answer
            time.sleep(0.005)
        raise TimeoutError(f"{method}: нет ответа за {timeout} с")


# --------------------------------------------------------------------------- #
# HTTP-сервер статики
# --------------------------------------------------------------------------- #


def html_server_for(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Публичный синоним ``serve`` — имя читается в вызывающем коде."""
    return serve(root)


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # noqa: D102
            pass

    handler = functools.partial(Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def launch(options: Options) -> tuple[subprocess.Popen, str]:
    port = free_port()
    profile = f"/tmp/record-scene-{os.getpid()}-{threading.get_ident()}-{port}"
    process = subprocess.Popen(
        [
            options.chrome, "--headless=new", f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
            "--use-angle=metal", "--disable-gpu-sandbox", "--hide-scrollbars",
            f"--window-size={options.width},{options.height}", "--force-device-scale-factor=1",
            "--use-mock-keychain", "--password-store=basic",
            "--disable-background-timer-throttling", "--disable-lcd-text", "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, profile


# --------------------------------------------------------------------------- #
# Ожидание готовности и покадровая съёмка
# --------------------------------------------------------------------------- #


def wait_ready(call, options: Options) -> bool:
    """Ждём ``__ready``; страница с canvas и без флага тоже считается живой."""
    result = call(
        "Runtime.evaluate",
        {
            "expression": f"""
new Promise((res) => {{
  const t0 = performance.now();
  const timer = setInterval(() => {{
    const done = window.__ready === true ||
                 (!!document.querySelector('canvas') && performance.now() - t0 > {options.canvas_fallback * 1000}) ||
                 performance.now() - t0 > {options.ready_timeout * 1000};
    if (done) {{ clearInterval(timer); res(window.__ready === true); }}
  }}, 25);
}})""",
            "awaitPromise": True,
            "returnByValue": True,
        },
        timeout=options.ready_timeout + 15,
    )
    return bool(((result.get("result") or {}).get("result") or {}).get("value"))


def has_driver(call) -> bool:
    result = call("Runtime.evaluate", {"expression": "!!window.__dbscene", "returnByValue": True})
    return bool(((result.get("result") or {}).get("result") or {}).get("value"))


def pause_driver(call) -> int:
    """Останавливаем автоплей и возвращаем уже показанный шаг (seek назад не ходит)."""
    result = call(
        "Runtime.evaluate",
        {
            "expression": """
(() => {
  const scene = window.__dbscene;
  if (!scene) return -1;
  window.postMessage('pause', '*');
  return scene.step | 0;
})()""",
            "returnByValue": True,
        },
    )
    return max(0, int(((result.get("result") or {}).get("result") or {}).get("value") or 0))


def driver_window(call) -> int:
    result = call(
        "Runtime.evaluate",
        {
            "expression": """
(() => {
  const scene = window.__dbscene;
  if (!scene || !scene.freeze) return 0;
  return scene.freeze | 0;
})()""",
            "returnByValue": True,
        },
    )
    return int(((result.get("result") or {}).get("result") or {}).get("value") or 0)


def capture(call, frame: Path) -> None:
    shot = call(
        "Page.captureScreenshot",
        {"format": "jpeg", "quality": 90, "optimizeForSpeed": True},
        timeout=120,
    )
    data = (shot.get("result") or {}).get("data")
    if not data:
        raise RuntimeError(f"пустой кадр: {json.dumps(shot)[:200]}")
    frame.write_bytes(base64.b64decode(data))


def record_frames(call, frames_dir: Path, options: Options) -> bool:
    """Снять ``options.frames`` кадров. ``False`` — кадров не получилось."""
    driver = options.mode != "steps" and has_driver(call)
    if options.mode == "steps":
        driver = False
    base = pause_driver(call) if driver else 0
    total = options.frames
    steps = driver_window(call) if driver else 0
    for index in range(total):
        if driver:
            # Всё окно движения укладывается в клип: последний кадр — конец окна.
            step = max(base, round(index * steps / max(1, total - 1))) if steps else base
            script = f"window.__dbscene.seek({step});"
        else:
            # Один кадр — один шаг страницы; probe зовём всегда, часть страниц
            # рисует только в нём.
            script = (
                f"try{{window.__step({options.dt});}}catch(e){{}} "
                "try{window.__probe && window.__probe();}catch(e){}"
            )
        call("Runtime.evaluate", {"expression": script, "returnByValue": False}, timeout=60)
        capture(call, frames_dir / f"{index:06d}.jpg")
    return True


def _ffmpeg_env(options: Options) -> dict[str, str]:
    """Бандл ffmpeg ищет свои dylib рядом с собой; из PATH — не трогаем."""
    env = dict(os.environ)
    binary = Path(options.ffmpeg)
    if binary.parent != Path("."):
        env["DYLD_LIBRARY_PATH"] = str(binary.resolve().parent)
    return env


def encode(frames_dir: Path, out: Path, options: Options) -> None:
    subprocess.run(
        [
            options.ffmpeg, "-hide_banner", "-loglevel", "error", "-framerate", str(options.fps),
            "-i", str(frames_dir / "%06d.jpg"),
            "-vf", f"scale={options.width}:{options.height}:force_original_aspect_ratio=decrease,"
                   f"scale={options.width}:{options.height}",
            "-r", str(options.fps), "-c:v", "libx264", "-preset", options.preset,
            "-crf", str(options.crf), "-pix_fmt", "yuv420p", "-y", str(out),
        ],
        check=True,
        env=_ffmpeg_env(options),
        cwd=str(Path(options.ffmpeg).resolve().parent),
    )


def placeholder(out: Path, options: Options) -> None:
    """Тёмный клип той же длины — подпись «страницы нет» рисует монтаж."""
    subprocess.run(
        [
            options.ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"nullsrc=s={options.width}x{options.height}:r={options.fps}:d={options.secs}",
            "-c:v", "libx264", "-preset", options.preset, "-crf", str(options.crf),
            "-pix_fmt", "yuv420p", "-y", str(out),
        ],
        check=True,
        env=_ffmpeg_env(options),
        cwd=str(Path(options.ffmpeg).resolve().parent),
    )


def record_visited(name: str, url: str, out_dir: Path, options: Options) -> Result:
    """Записать клип по URL. Имя ``name`` уходит только в имя файла."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "_frames" / name
    subprocess.run(["rm", "-rf", str(frames_dir)], check=False)
    frames_dir.mkdir(parents=True)
    created = time.time()
    started = created
    process, profile = launch(options)
    try:
        ws = WS(cdp_ws(free_port_of(process)))
        target = ws.call("Target.createTarget", {"url": "about:blank"})
        session = ws.call(
            "Target.attachToTarget", {"targetId": target["result"]["targetId"], "flatten": True}
        )["result"]["sessionId"]

        def call(method: str, params: dict | None = None, **kwargs: Any) -> dict:
            return ws.call(method, params, session_id=session, **kwargs)

        call("Page.enable")
        call(
            "Emulation.setDeviceMetricsOverride",
            {"width": options.width, "height": options.height, "deviceScaleFactor": 1, "mobile": False},
        )
        call("Page.navigate", {"url": url})
        if not wait_ready(call, options):
            subprocess.run(["rm", "-rf", str(frames_dir)], check=False)
            return Result("missing", reason="страница не дошла до __ready")
        time.sleep(options.warmup)
        record_frames(call, frames_dir, options)
    finally:
        process.terminate()
        subprocess.run(["rm", "-rf", profile], check=False)
    out = out_dir / f"{name}.mp4"
    encode(frames_dir, out, options)
    subprocess.run(["rm", "-rf", str(frames_dir)], check=False)
    try:
        (out_dir / "_frames").rmdir()  # пустой каталог рабочих кадров не нужен
    except OSError:
        pass
    return Result("ok", frames=options.frames, seconds=round(time.time() - started, 1))


def free_port_of(process: subprocess.Popen) -> int:
    """Порт, переданный Chrome в командной строке (он же в /json/version)."""
    for part in process.args:
        if isinstance(part, str) and part.startswith("--remote-debugging-port="):
            return int(part.split("=", 1)[1])
    raise RuntimeError("у Chrome нет --remote-debugging-port")


def record_steps(name: str, url: str, out_dir: Path, options: Options) -> Result:
    return record_visited(name, url, out_dir, dataclasses.replace(options, mode="steps"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Записать живой клип веб-сцены.")
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--secs", type=float, default=23.0)
    parser.add_argument("--warmup", type=float, default=0.6)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--crf", type=int, default=21)
    parser.add_argument("--chrome", default=os.environ.get("CHROME", DEFAULT_CHROME))
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", DEFAULT_FFMPEG))
    parser.add_argument("--mode", default="auto", choices=("auto", "steps", "visited"))
    parser.add_argument("--placeholder", action="store_true", help="только тёмный клип")
    args = parser.parse_args()

    options = Options(
        secs=args.secs, warmup=args.warmup, width=args.width, height=args.height,
        fps=args.fps, crf=args.crf, chrome=args.chrome, ffmpeg=args.ffmpeg, mode=args.mode,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.placeholder:
        placeholder(out, options)
        print(f"[ok] плашка {out}")
        return 0
    result = record_visited(out.stem, args.url, out.parent, options)
    if result.status != "ok":
        print(f"[missing] {result.reason}")
        return 1
    print(f"[ok] {out} ({result.frames} кадров, {result.seconds} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
