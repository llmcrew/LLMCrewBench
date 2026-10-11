#!/usr/bin/env python3
"""Проверить страницы витрины: кто доехал до `__ready`, кто падает до кадра.

Страница, упавшая до первого кадра, снимается чёрной: витрине честнее показать
плашку «модель не справилась» (тот же вердикт, что и у harness — 0 баллов), чем
черноту. Проверка ставит/снимает маркер `<модель>__<сцена>.missing` — тот же,
что у страниц, которых нет вовсе; манифест витрины читает его как плашку.

  python3 probe_pages.py [workers]
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from record_all import MODELS, OUT, SCENES  # noqa: E402
from record_clip import ART, CHROME, H, W, WS, cdp_ws, free_port, html_server  # noqa: E402

READY_MS = 12000


def probe(model: str, scene: str, base: str) -> tuple[str, str, bool, str]:
    if not (ART / "_showcase" / model / "pages" / f"{scene}.html").is_file():
        return model, scene, False, "нет страницы"
    port = free_port()
    profile = f"/tmp/probe-chrome-{os.getpid()}-{threading.get_ident()}"
    proc = subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check",
         "--use-angle=metal", "--disable-gpu-sandbox",
         "--hide-scrollbars", f"--window-size={W},{H}",
         "--force-device-scale-factor=1", "--use-mock-keychain", "--password-store=basic",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws = WS(cdp_ws(port))
        tgt = ws.call("Target.createTarget", {"url": "about:blank"})
        sid = ws.call("Target.attachToTarget",
                      {"targetId": tgt["result"]["targetId"], "flatten": True})["result"]["sessionId"]
        call = lambda m, p=None, **kw: ws.call(m, p, session_id=sid, **kw)  # noqa: E731
        call("Page.enable")
        call("Runtime.enable")
        call("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
        call("Page.navigate", {"url": f"{base}/_showcase/{model}/pages/{scene}.html"})
        r = call("Runtime.evaluate", {"expression": f"""
new Promise((res) => {{
  const t0 = performance.now();
  const t = setInterval(() => {{
    if (window.__ready === true || performance.now() - t0 > {READY_MS}) {{
      clearInterval(t); res(window.__ready === true);
    }}
  }}, 25);
}})""", "awaitPromise": True, "returnByValue": True}, timeout=READY_MS / 1000 + 15)
        ready = bool(((r.get("result") or {}).get("result") or {}).get("value"))
        with ws.lock:
            errs = [e for e in ws.events if e.get("method") == "Runtime.exceptionThrown"]
        desc = (errs[0]["params"]["exceptionDetails"].get("exception", {}).get("description", "")
                .split("\n")[0] if errs else "")
        return model, scene, ready, desc
    finally:
        proc.terminate()
        subprocess.run(["rm", "-rf", profile], check=False)


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    pairs = [(m, s) for s in SCENES for m in MODELS
             if (ART / "_showcase" / m / "pages" / f"{s}.html").is_file()]
    srv, base = html_server()
    broken: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for model, scene, ready, desc in pool.map(lambda p: probe(*p, base), pairs):
                marker = OUT / f"{model}__{scene}.missing"
                if ready:
                    marker.unlink(missing_ok=True)
                else:
                    marker.write_text("1")
                    broken.append(f"{model}/{scene}: {desc or 'нет __ready'}")
                    print(f"[probe] {model}/{scene}: упала -> плашка ({desc or 'нет __ready'})", flush=True)
    finally:
        srv.shutdown()
    print(f"[probe] готово: упавших {len(broken)} из {len(pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
